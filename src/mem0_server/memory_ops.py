"""Shared memory operations used by both CLI tests and MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from datetime import datetime
from typing import Any

import pytz

from mem0.configs.prompts import get_update_memory_messages
from mem0.memory.main import _build_filters_and_metadata
from mem0.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)

from mem0_server.config import Mem0ServerConfig

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("mem0_server.performance")


async def add_memory_with_profiling(
    *,
    memory_client: Any,
    text: Any,
    user_id: str,
    metadata: dict[str, Any] | None,
    config: Mem0ServerConfig,
) -> dict[str, Any]:
    """Add memory with detailed profiling steps.

    This is the shared implementation used by both MCP tools and CLI tests.
    """
    _ = config
    total_start = time.perf_counter()

    def _log_step(step: str, step_start: float, **extras: Any) -> None:
        elapsed_ms = (time.perf_counter() - step_start) * 1000
        log_parts = [
            "[Perf] add_memories",
            f"user={user_id}",
            f"step={step}",
            f"time={elapsed_ms:.1f}ms",
        ]
        for key, value in extras.items():
            log_parts.append(f"{key}={value}")
        perf_logger.info(" | ".join(log_parts))

    processed_metadata, effective_filters = _build_filters_and_metadata(
        user_id=user_id,
        input_metadata=metadata or {},
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
    perf_logger.info(
        f"[Perf] add_memories | user={user_id} | total={total_ms:.1f}ms | status=success"
    )

    if memory_client.enable_graph:
        return {"results": returned_memories, "relations": graph_relations}
    return {"results": returned_memories}


async def add_memory_op(
    *,
    memory_client: Any,
    text: str,
    user_id: str,
    metadata: dict[str, Any] | None,
    config: Mem0ServerConfig,
) -> dict[str, Any]:
    """Shared add memory operation with optional profiling."""
    if config.server.performance_logging:
        try:
            return await add_memory_with_profiling(
                memory_client=memory_client,
                text=text,
                user_id=user_id,
                metadata=metadata,
                config=config,
            )
        except Exception as e:
            logger.warning(f"Performance logging failed, falling back to standard add: {e}")

    if metadata:
        return await memory_client.add(text, user_id=user_id, metadata=metadata)
    return await memory_client.add(text, user_id=user_id)


async def search_memory_op(
    *,
    memory_client: Any,
    query: str,
    user_id: str,
    limit: int,
    config: Mem0ServerConfig,
) -> dict[str, Any]:
    """Search memories with optional performance logging."""
    _ = config
    results = await memory_client.search(query=query, user_id=user_id, limit=limit)
    if not results or not results.get("results"):
        return {"results": [], "message": "No relevant memories found."}
    return results


async def list_memories_op(
    *,
    memory_client: Any,
    user_id: str,
    config: Mem0ServerConfig,
) -> dict[str, Any]:
    """List all memories for user."""
    _ = config
    return await memory_client.get_all(user_id=user_id)


async def get_memory_op(
    *,
    memory_client: Any,
    memory_id: str,
    config: Mem0ServerConfig,
) -> dict[str, Any]:
    """Get a specific memory by ID."""
    _ = config
    return await memory_client.get(memory_id)


async def delete_memory_op(
    *,
    memory_client: Any,
    memory_id: str,
    config: Mem0ServerConfig,
) -> str:
    """Delete a single memory by ID."""
    _ = config
    await memory_client.delete(memory_id)
    return memory_id


async def delete_memories_op(
    *,
    memory_client: Any,
    memory_ids: list[str],
    config: Mem0ServerConfig,
) -> list[str]:
    """Delete multiple memories by IDs."""
    _ = config
    deleted = []
    for memory_id in memory_ids:
        try:
            await memory_client.delete(memory_id)
            deleted.append(memory_id)
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")
    return deleted


async def delete_all_memories_op(
    *,
    memory_client: Any,
    user_id: str,
    config: Mem0ServerConfig,
) -> None:
    """Delete all memories for the user."""
    _ = config
    await memory_client.delete_all(user_id=user_id)
