"""
MCP Server implementation for mem0-server.

This module provides a standalone MCP server that can run without Docker or
the full OpenMemory API. It uses the mem0 library directly for memory operations.
"""

# type: ignore

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import time
import psutil
import os
from copy import deepcopy
from datetime import datetime
from typing import Callable, Any

import pytz

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

from mem0_server.config import ConfigLoader, Mem0ServerConfig

from mem0.configs.prompts import get_update_memory_messages
from mem0.memory.main import _build_filters_and_metadata
from mem0.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("mem0_server.performance")

# Global config loader reference for API routes
_config_loader: ConfigLoader | None = None

# Context variables for user_id and client_name
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")


def _create_perf_logged_tool(func: Callable, tool_name: str, config: Mem0ServerConfig) -> Callable:
    """Wrap a tool function with performance logging if enabled."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not config.server.performance_logging:
            return await func(*args, **kwargs)

        start_time = time.perf_counter()
        status = "success"
        error_msg = None

        try:
            result = await func(*args, **kwargs)
            # Check if result indicates an error
            if isinstance(result, str) and result.startswith("Error"):
                status = "error"
                error_msg = result[:100]
            return result
        except Exception as e:
            status = "exception"
            error_msg = str(e)[:100]
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            uid = user_id_var.get(None) or config.server.user_id

            log_parts = [
                f"[Perf] {tool_name}",
                f"user={uid}",
                f"time={elapsed_ms:.1f}ms",
                f"status={status}",
            ]
            if error_msg:
                log_parts.append(f"error={error_msg}")

            perf_logger.info(" | ".join(log_parts))

    return wrapper


async def _add_memories_with_profiling(
    *,
    memory_client: Any,
    text: Any,
    uid: str,
    client_name: str,
    config: Mem0ServerConfig,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    def _log_step(step: str, step_start: float, **extras: Any) -> None:
        elapsed_ms = (time.perf_counter() - step_start) * 1000
        log_parts = [
            "[Perf] add_memories",
            f"user={uid}",
            f"step={step}",
            f"time={elapsed_ms:.1f}ms",
        ]
        for key, value in extras.items():
            log_parts.append(f"{key}={value}")
        perf_logger.info(" | ".join(log_parts))

    metadata = {
        "source_app": "mem0-server",
        "mcp_client": client_name,
    }

    processed_metadata, effective_filters = _build_filters_and_metadata(
        user_id=uid,
        input_metadata=metadata,
    )

    if isinstance(text, str):
        messages = [{"role": "user", "content": text}]
    elif isinstance(text, dict):
        messages = [text]
    else:
        messages = text

    if memory_client.config.llm.config.get("enable_vision"):
        messages = parse_vision_messages(
            messages, memory_client.llm, memory_client.config.llm.config.get("vision_details")
        )
    else:
        messages = parse_vision_messages(messages)

    fact_start = time.perf_counter()
    parsed_messages = parse_messages(messages)
    if memory_client.config.custom_fact_extraction_prompt:
        system_prompt = memory_client.config.custom_fact_extraction_prompt
        user_prompt = f"Input:\n{parsed_messages}"
    else:
        should_use_agent = getattr(memory_client, "_should_use_agent_memory_extraction", None)
        is_agent_memory = False
        if callable(should_use_agent):
            is_agent_memory = should_use_agent(messages, processed_metadata)
        system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

    response = await asyncio.to_thread(
        memory_client.llm.generate_response,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        response = remove_code_blocks(response)
        if not response.strip():
            new_retrieved_facts = []
        else:
            try:
                new_retrieved_facts = json.loads(response)["facts"]
            except json.JSONDecodeError:
                extracted_json = extract_json(response)
                new_retrieved_facts = json.loads(extracted_json)["facts"]
    except Exception as e:
        logger.error(f"Error in new_retrieved_facts (profiling): {e}")
        new_retrieved_facts = []

    _log_step("fact_extraction", fact_start, facts=len(new_retrieved_facts))

    embed_start = time.perf_counter()
    new_message_embeddings: dict[str, Any] = {}
    for fact in new_retrieved_facts:
        embeddings = await asyncio.to_thread(memory_client.embedding_model.embed, fact, "add")
        new_message_embeddings[fact] = embeddings
    _log_step("embedding", embed_start, count=len(new_message_embeddings))

    search_start = time.perf_counter()
    retrieved_old_memory = []
    search_filters = {}
    if effective_filters.get("user_id"):
        search_filters["user_id"] = effective_filters["user_id"]
    if effective_filters.get("agent_id"):
        search_filters["agent_id"] = effective_filters["agent_id"]
    if effective_filters.get("run_id"):
        search_filters["run_id"] = effective_filters["run_id"]

    total_found = 0
    for fact in new_retrieved_facts:
        embeddings = new_message_embeddings.get(fact)
        existing_mems = await asyncio.to_thread(
            memory_client.vector_store.search,
            query=fact,
            vectors=embeddings,
            limit=5,
            filters=search_filters,
        )
        total_found += len(existing_mems)
        retrieved_old_memory.extend(
            [{"id": mem.id, "text": mem.payload.get("data", "")} for mem in existing_mems]
        )

    unique_data = {item["id"]: item for item in retrieved_old_memory}
    retrieved_old_memory = list(unique_data.values())
    temp_uuid_mapping = {}
    for idx, item in enumerate(retrieved_old_memory):
        temp_uuid_mapping[str(idx)] = item["id"]
        retrieved_old_memory[idx]["id"] = str(idx)

    _log_step("vector_search", search_start, matches=total_found)

    update_start = time.perf_counter()
    if new_retrieved_facts:
        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory,
            new_retrieved_facts,
            memory_client.config.custom_update_memory_prompt,
        )
        try:
            update_response = await asyncio.to_thread(
                memory_client.llm.generate_response,
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"Error in new memory actions response (profiling): {e}")
            update_response = ""
        try:
            if not update_response or not update_response.strip():
                new_memories_with_actions = {}
            else:
                update_response = remove_code_blocks(update_response)
                new_memories_with_actions = json.loads(update_response)
        except Exception as e:
            logger.error(f"Invalid JSON response (profiling): {e}")
            new_memories_with_actions = {}
    else:
        new_memories_with_actions = {}

    _log_step("update_decision", update_start)

    insert_start = time.perf_counter()
    returned_memories = []
    memory_tasks = []

    for resp in new_memories_with_actions.get("memory", []):
        try:
            action_text = resp.get("text")
            if not action_text:
                continue
            event_type = resp.get("event")

            if event_type == "ADD":
                task = asyncio.create_task(
                    memory_client._create_memory(
                        data=action_text,
                        existing_embeddings=new_message_embeddings,
                        metadata=deepcopy(processed_metadata),
                    )
                )
                memory_tasks.append((task, resp, "ADD", None))
            elif event_type == "UPDATE":
                task = asyncio.create_task(
                    memory_client._update_memory(
                        memory_id=temp_uuid_mapping[resp["id"]],
                        data=action_text,
                        existing_embeddings=new_message_embeddings,
                        metadata=deepcopy(processed_metadata),
                    )
                )
                memory_tasks.append((task, resp, "UPDATE", temp_uuid_mapping[resp["id"]]))
            elif event_type == "DELETE":
                memory_id = temp_uuid_mapping.get(resp.get("id"))
                if memory_id:
                    task = asyncio.create_task(memory_client._delete_memory(memory_id=memory_id))
                    memory_tasks.append((task, resp, "DELETE", memory_id))
            elif event_type == "NONE":
                memory_id = temp_uuid_mapping.get(resp.get("id"))
                if memory_id and (
                    processed_metadata.get("agent_id") or processed_metadata.get("run_id")
                ):

                    async def update_session_ids(mem_id, meta):
                        existing_memory = await asyncio.to_thread(
                            memory_client.vector_store.get, vector_id=mem_id
                        )
                        updated_metadata = deepcopy(existing_memory.payload)
                        if meta.get("agent_id"):
                            updated_metadata["agent_id"] = meta["agent_id"]
                        if meta.get("run_id"):
                            updated_metadata["run_id"] = meta["run_id"]
                        updated_metadata["updated_at"] = datetime.now(
                            pytz.timezone("US/Pacific")
                        ).isoformat()

                        await asyncio.to_thread(
                            memory_client.vector_store.update,
                            vector_id=mem_id,
                            vector=None,
                            payload=updated_metadata,
                        )

                    task = asyncio.create_task(update_session_ids(memory_id, processed_metadata))
                    memory_tasks.append((task, resp, "NONE", memory_id))
        except Exception as e:
            logger.error(f"Error processing memory action (profiling): {resp}, Error: {e}")

    for task, resp, event_type, mem_id in memory_tasks:
        try:
            result_id = await task
            if event_type == "ADD":
                returned_memories.append(
                    {"id": result_id, "memory": resp.get("text"), "event": event_type}
                )
            elif event_type == "UPDATE":
                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": resp.get("text"),
                        "event": event_type,
                        "previous_memory": resp.get("old_memory"),
                    }
                )
            elif event_type == "DELETE":
                returned_memories.append(
                    {"id": mem_id, "memory": resp.get("text"), "event": event_type}
                )
        except Exception as e:
            logger.error(f"Error awaiting memory task (profiling): {e}")

    _log_step("vector_insert", insert_start, inserted=len(returned_memories))

    graph_start = time.perf_counter()
    graph_relations = []
    if getattr(memory_client, "enable_graph", False) and getattr(memory_client, "graph", None):
        try:
            graph_relations = await memory_client._add_to_graph(
                messages,
                deepcopy(effective_filters),
            )
        except Exception as e:
            logger.error(f"Error adding to graph (profiling): {e}")
            graph_relations = []
    _log_step("graph", graph_start, relations=len(graph_relations))

    total_ms = (time.perf_counter() - total_start) * 1000
    perf_logger.info(f"[Perf] add_memories | user={uid} | total={total_ms:.1f}ms | status=success")

    if memory_client.enable_graph:
        return {"results": returned_memories, "relations": graph_relations}
    return {"results": returned_memories}


class MCPServerManager:
    """Manages the MCP server and mem0 memory client."""

    def __init__(self, config: Mem0ServerConfig):
        self.config = config
        self._memory_client = None
        self._memory_client_initialized = False
        self._memory_client_lock = asyncio.Lock()
        self._mcp = FastMCP("mem0-mcp-server")
        self._router = APIRouter(prefix="/mcp")
        self._sse = SseServerTransport("/mcp/messages/")
        self._setup_tools()
        self._setup_routes()

    @property
    def mcp(self) -> FastMCP:
        """Get the FastMCP instance."""
        return self._mcp

    @property
    def router(self) -> APIRouter:
        """Get the API router with MCP endpoints."""
        return self._router

    async def get_memory_client(self):
        """Get or initialize the mem0 async memory client.

        Uses double-checked locking to prevent race conditions during
        concurrent initialization requests.
        """
        # Fast path: already initialized
        if self._memory_client_initialized:
            return self._memory_client

        # Slow path: acquire lock for initialization
        async with self._memory_client_lock:
            # Double-check after acquiring lock
            if self._memory_client_initialized:
                return self._memory_client

            try:
                from mem0 import AsyncMemory

                mem0_config = self.config.to_mem0_config()
                logger.info(
                    f"Initializing mem0 AsyncMemory with config: {json.dumps(mem0_config, indent=2)}"
                )

                self._memory_client = await AsyncMemory.from_config(mem0_config)
                self._memory_client_initialized = True
                logger.info("mem0 AsyncMemory client initialized successfully")
                return self._memory_client

            except Exception as e:
                logger.error(f"Failed to initialize mem0 async client: {e}")
                raise

    async def get_memory_client_safe(self):
        """Get memory client with error handling. Returns None if unavailable."""
        try:
            return await self.get_memory_client()
        except Exception as e:
            logger.warning(f"Memory client unavailable: {e}")
            return None

    def _setup_tools(self) -> None:
        """Register MCP tools for memory operations."""

        def _log_perf(
            tool_name: str, start_time: float, status: str, error_msg: str | None = None
        ) -> None:
            if not self.config.server.performance_logging:
                return
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            uid = user_id_var.get(None) or self.config.server.user_id
            log_parts = [
                f"[Perf] {tool_name}",
                f"user={uid}",
                f"time={elapsed_ms:.1f}ms",
                f"status={status}",
            ]
            if error_msg:
                log_parts.append(f"error={error_msg[:100]}")
            perf_logger.info(" | ".join(log_parts))

        @self._mcp.tool(
            description="Add a new memory. Called when user shares information about themselves, "
            "preferences, or anything relevant for future conversations. "
            "Also called when user explicitly asks to remember something."
        )
        async def add_memories(text: str) -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id
            client_name = client_name_var.get(None) or "default"

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("add_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                if self.config.server.performance_logging:
                    try:
                        response = await _add_memories_with_profiling(
                            memory_client=memory_client,
                            text=text,
                            uid=uid,
                            client_name=client_name,
                            config=self.config,
                        )
                        return json.dumps(response, indent=2)
                    except Exception as e:
                        logger.warning(
                            f"Performance logging failed, falling back to standard add: {e}"
                        )

                response = await memory_client.add(
                    text,
                    user_id=uid,
                    metadata={
                        "source_app": "mem0-server",
                        "mcp_client": client_name,
                    },
                )
                _log_perf("add_memories", start_time, "success")
                return json.dumps(response, indent=2)
            except Exception as e:
                logger.exception(f"Error adding memory: {e}")
                _log_perf("add_memories", start_time, "exception", str(e))
                return f"Error adding to memory: {e}"

        @self._mcp.tool(
            description="Search through stored memories. Called when user asks anything "
            "to find relevant context from past conversations."
        )
        async def search_memory(query: str, limit: int = 10) -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("search_memory", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                results = await memory_client.search(query=query, user_id=uid, limit=limit)

                if not results or not results.get("results"):
                    _log_perf("search_memory", start_time, "success")
                    return json.dumps({"results": [], "message": "No relevant memories found."})

                _log_perf("search_memory", start_time, "success")
                return json.dumps(results, indent=2)
            except Exception as e:
                logger.exception(f"Error searching memory: {e}")
                _log_perf("search_memory", start_time, "exception", str(e))
                return f"Error searching memory: {e}"

        @self._mcp.tool(description="List all memories for the user.")
        async def list_memories() -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("list_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                memories = await memory_client.get_all(user_id=uid)
                _log_perf("list_memories", start_time, "success")
                return json.dumps(memories, indent=2)
            except Exception as e:
                logger.exception(f"Error listing memories: {e}")
                _log_perf("list_memories", start_time, "exception", str(e))
                return f"Error getting memories: {e}"

        @self._mcp.tool(description="Get a specific memory by ID.")
        async def get_memory(memory_id: str) -> str:
            start_time = time.perf_counter()
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("get_memory", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                memory = await memory_client.get(memory_id)
                _log_perf("get_memory", start_time, "success")
                return json.dumps(memory, indent=2)
            except Exception as e:
                logger.exception(f"Error getting memory: {e}")
                _log_perf("get_memory", start_time, "exception", str(e))
                return f"Error getting memory: {e}"

        @self._mcp.tool(description="Delete specific memories by their IDs.")
        async def delete_memories(memory_ids: list[str]) -> str:
            start_time = time.perf_counter()
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("delete_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                deleted = []
                for memory_id in memory_ids:
                    try:
                        await memory_client.delete(memory_id)
                        deleted.append(memory_id)
                    except Exception as e:
                        logger.warning(f"Failed to delete memory {memory_id}: {e}")

                _log_perf("delete_memories", start_time, "success")
                return f"Successfully deleted {len(deleted)} memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                _log_perf("delete_memories", start_time, "exception", str(e))
                return f"Error deleting memories: {e}"

        @self._mcp.tool(description="Delete all memories for the user.")
        async def delete_all_memories() -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("delete_all_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                await memory_client.delete_all(user_id=uid)
                _log_perf("delete_all_memories", start_time, "success")
                return "Successfully deleted all memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                _log_perf("delete_all_memories", start_time, "exception", str(e))
                return f"Error deleting memories: {e}"

    def _setup_routes(self) -> None:
        """Setup FastAPI routes for MCP endpoints."""

        @self._router.get("/{client_name}/sse/{user_id}")
        async def handle_sse(request: Request) -> None:
            """Handle SSE connections for a specific user and client."""
            uid = request.path_params.get("user_id", self.config.server.user_id)
            user_token = user_id_var.set(uid)
            client_name = request.path_params.get("client_name", "default")
            client_token = client_name_var.set(client_name)

            try:
                async with self._sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,
                ) as (read_stream, write_stream):
                    await self._mcp._mcp_server.run(
                        read_stream,
                        write_stream,
                        self._mcp._mcp_server.create_initialization_options(),
                    )
            finally:
                user_id_var.reset(user_token)
                client_name_var.reset(client_token)

        @self._router.post("/messages/")
        async def handle_post_message(request: Request) -> dict[str, str]:
            """Handle POST messages for SSE."""
            try:
                body = await request.body()

                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}

                async def send(message):
                    return {}

                await self._sse.handle_post_message(request.scope, receive, send)
                return {"status": "ok"}
            except Exception as e:
                logger.exception(f"Error handling post message: {e}")
                return {"status": "error", "message": str(e)}

        @self._router.post("/{client_name}/sse/{user_id}/messages/")
        async def handle_client_post_message(request: Request) -> dict[str, str]:
            """Handle POST messages for specific client/user."""
            return await handle_post_message(request)


class MCPServerManagerStdio:
    """Manages the MCP server for stdio transport without FastAPI/HTTP infrastructure."""

    def __init__(self, config: Mem0ServerConfig):
        start_time = time.perf_counter()

        self.config = config
        self._memory_client = None
        self._memory_client_initialized = False
        self._memory_client_lock = asyncio.Lock()

        mcp_start = time.perf_counter()
        self._mcp = FastMCP("mem0-mcp-server")
        mcp_time = (time.perf_counter() - mcp_start) * 1000

        tools_start = time.perf_counter()
        self._setup_tools()
        tools_time = (time.perf_counter() - tools_start) * 1000

        total_time = (time.perf_counter() - start_time) * 1000
        memory_bytes = psutil.Process(os.getpid()).memory_info().rss
        memory_mb = memory_bytes / (1024 * 1024)

        logger.info(
            f"[Performance] MCPServerManagerStdio initialization: "
            f"FastMCP creation={mcp_time:.1f}ms, "
            f"Tool registration={tools_time:.1f}ms, "
            f"Total time={total_time:.1f}ms, "
            f"Memory usage={memory_mb:.1f}MB (RSS)"
        )

    @property
    def mcp(self) -> FastMCP:
        """Get the FastMCP instance."""
        return self._mcp

    async def get_memory_client(self):
        """Get or initialize the mem0 async memory client.

        Uses double-checked locking to prevent race conditions during
        concurrent initialization requests.
        """
        # Fast path: already initialized
        if self._memory_client_initialized:
            return self._memory_client

        # Slow path: acquire lock for initialization
        async with self._memory_client_lock:
            # Double-check after acquiring lock
            if self._memory_client_initialized:
                return self._memory_client

            try:
                from mem0 import AsyncMemory

                mem0_config = self.config.to_mem0_config()
                logger.info(
                    f"Initializing mem0 AsyncMemory with config: {json.dumps(mem0_config, indent=2)}"
                )

                self._memory_client = await AsyncMemory.from_config(mem0_config)
                self._memory_client_initialized = True
                logger.info("mem0 AsyncMemory client initialized successfully")
                return self._memory_client

            except Exception as e:
                logger.error(f"Failed to initialize mem0 async client: {e}")
                raise

    async def get_memory_client_safe(self):
        """Get memory client with error handling. Returns None if unavailable."""
        try:
            return await self.get_memory_client()
        except Exception as e:
            logger.warning(f"Memory client unavailable: {e}")
            return None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - clean up resources."""
        try:
            if self._memory_client_initialized and self._memory_client:
                logger.info("Cleaning up AsyncMemory client...")
                if hasattr(self._memory_client, "close"):
                    await self._memory_client.close()
                elif hasattr(self._memory_client, "__aexit__"):
                    await self._memory_client.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        return False

    def _setup_tools(self) -> None:
        """Register MCP tools for memory operations."""

        def _log_perf(
            tool_name: str, start_time: float, status: str, error_msg: str | None = None
        ) -> None:
            if not self.config.server.performance_logging:
                return
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            uid = user_id_var.get(None) or self.config.server.user_id
            log_parts = [
                f"[Perf] {tool_name}",
                f"user={uid}",
                f"time={elapsed_ms:.1f}ms",
                f"status={status}",
            ]
            if error_msg:
                log_parts.append(f"error={error_msg[:100]}")
            perf_logger.info(" | ".join(log_parts))

        @self._mcp.tool(
            description="Add a new memory. Called when user shares information about themselves, "
            "preferences, or anything relevant for future conversations. "
            "Also called when user explicitly asks to remember something."
        )
        async def add_memories(text: str) -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id
            client_name = client_name_var.get(None) or "default"

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("add_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                if self.config.server.performance_logging:
                    try:
                        response = await _add_memories_with_profiling(
                            memory_client=memory_client,
                            text=text,
                            uid=uid,
                            client_name=client_name,
                            config=self.config,
                        )
                        return json.dumps(response, indent=2)
                    except Exception as e:
                        logger.warning(
                            f"Performance logging failed, falling back to standard add: {e}"
                        )

                response = await memory_client.add(
                    text,
                    user_id=uid,
                    metadata={
                        "source_app": "mem0-server",
                        "mcp_client": client_name,
                    },
                )
                _log_perf("add_memories", start_time, "success")
                return json.dumps(response, indent=2)
            except Exception as e:
                logger.exception(f"Error adding memory: {e}")
                _log_perf("add_memories", start_time, "exception", str(e))
                return f"Error adding to memory: {e}"

        @self._mcp.tool(
            description="Search through stored memories. Called when user asks anything "
            "to find relevant context from past conversations."
        )
        async def search_memory(query: str, limit: int = 10) -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("search_memory", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                results = await memory_client.search(query=query, user_id=uid, limit=limit)

                if not results or not results.get("results"):
                    _log_perf("search_memory", start_time, "success")
                    return json.dumps({"results": [], "message": "No relevant memories found."})

                _log_perf("search_memory", start_time, "success")
                return json.dumps(results, indent=2)
            except Exception as e:
                logger.exception(f"Error searching memory: {e}")
                _log_perf("search_memory", start_time, "exception", str(e))
                return f"Error searching memory: {e}"

        @self._mcp.tool(description="List all memories for the user.")
        async def list_memories() -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("list_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                memories = await memory_client.get_all(user_id=uid)
                _log_perf("list_memories", start_time, "success")
                return json.dumps(memories, indent=2)
            except Exception as e:
                logger.exception(f"Error listing memories: {e}")
                _log_perf("list_memories", start_time, "exception", str(e))
                return f"Error getting memories: {e}"

        @self._mcp.tool(description="Get a specific memory by ID.")
        async def get_memory(memory_id: str) -> str:
            start_time = time.perf_counter()
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("get_memory", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                memory = await memory_client.get(memory_id)
                _log_perf("get_memory", start_time, "success")
                return json.dumps(memory, indent=2)
            except Exception as e:
                logger.exception(f"Error getting memory: {e}")
                _log_perf("get_memory", start_time, "exception", str(e))
                return f"Error getting memory: {e}"

        @self._mcp.tool(description="Delete specific memories by their IDs.")
        async def delete_memories(memory_ids: list[str]) -> str:
            start_time = time.perf_counter()
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("delete_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                deleted = []
                for memory_id in memory_ids:
                    try:
                        await memory_client.delete(memory_id)
                        deleted.append(memory_id)
                    except Exception as e:
                        logger.warning(f"Failed to delete memory {memory_id}: {e}")

                _log_perf("delete_memories", start_time, "success")
                return f"Successfully deleted {len(deleted)} memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                _log_perf("delete_memories", start_time, "exception", str(e))
                return f"Error deleting memories: {e}"

        @self._mcp.tool(description="Delete all memories for the user.")
        async def delete_all_memories() -> str:
            start_time = time.perf_counter()
            uid = user_id_var.get(None) or self.config.server.user_id

            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                _log_perf("delete_all_memories", start_time, "error", "Memory system unavailable")
                return "Error: Memory system is currently unavailable. Please try again later."

            try:
                await memory_client.delete_all(user_id=uid)
                _log_perf("delete_all_memories", start_time, "success")
                return "Successfully deleted all memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                _log_perf("delete_all_memories", start_time, "exception", str(e))
                return f"Error deleting memories: {e}"


def create_mcp_manager(config: Mem0ServerConfig) -> MCPServerManager:
    """Create an MCPServerManager instance without HTTP server.

    This function is used for stdio mode which only needs the MCP protocol layer
    without the FastAPI HTTP server infrastructure.

    Args:
        config: The server configuration.

    Returns:
        Configured MCPServerManager instance.
    """
    return MCPServerManager(config)


def create_mcp_manager_stdio(config: Mem0ServerConfig) -> MCPServerManagerStdio:
    """Create an MCPServerManagerStdio instance for stdio transport.

    This function creates an MCP manager optimized for stdio transport without
    any FastAPI HTTP infrastructure. It only manages the FastMCP instance and tools.

    Args:
        config: The server configuration.

    Returns:
        Configured MCPServerManagerStdio instance.
    """
    return MCPServerManagerStdio(config)


def create_app(config: Mem0ServerConfig, config_loader: ConfigLoader | None = None) -> FastAPI:
    """Create the FastAPI application with MCP server and REST API.

    Args:
        config: The server configuration.
        config_loader: Optional config loader for saving configuration changes.

    Returns:
        Configured FastAPI application.
    """
    global _config_loader
    _config_loader = config_loader

    app = FastAPI(
        title="mem0-server",
        description="Standalone MCP server for mem0 memory management",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize MCP server
    mcp_manager = MCPServerManager(config)

    # Include MCP router
    app.include_router(mcp_manager.router)

    # Include REST API router with full CRUD capabilities
    from mem0_server.api.routes import create_api_router

    api_router = create_api_router(
        config=config,
        config_loader=config_loader or ConfigLoader(),
        get_memory_client_func=mcp_manager.get_memory_client_safe,
    )
    app.include_router(api_router)

    # Root health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "mem0-server"}

    # Redirect root to docs
    @app.get("/")
    async def root():
        return {
            "service": "mem0-server",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
            "mcp": "/mcp/{client_name}/sse/{user_id}",
        }

    return app


def run_server(config: Mem0ServerConfig, config_loader: ConfigLoader | None = None) -> None:
    """Run the MCP server with uvicorn.

    Args:
        config: The server configuration.
        config_loader: Optional config loader for saving configuration changes.
    """
    import uvicorn

    app = create_app(config, config_loader)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level,
    )
