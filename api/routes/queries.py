"""知识库查询和 SSE 流式查询接口。"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import QueryRequest, QueryResponse
from api.dependencies import CurrentUser
from api.markdown import render_markdown
from api.serializers import serialize_sources
from api.sse import encode_sse
from api.user_scope import internal_session_id
from processor.query_processor.main_graph import KBQueryWorkflow
from processor.query_processor.state import create_default_query_state

router = APIRouter(prefix="/queries", tags=["queries"])
logger = logging.getLogger(__name__)


def _build_query_state(
    payload: QueryRequest,
    *,
    is_stream: bool,
    user_id: str,
) -> tuple[dict[str, Any], str]:
    session_id = payload.session_id or f"session-{uuid4().hex}"
    message_id = payload.message_id or f"message-{uuid4().hex}"
    task_id = payload.task_id or f"query-{uuid4().hex}"
    state = create_default_query_state(
        task_id=task_id,
        session_id=internal_session_id(user_id, session_id),
        message_id=message_id,
        original_query=payload.query,
        item_names=payload.item_names,
        search_mode=payload.mode,
        is_stream=is_stream,
    )
    return state, session_id


@router.post(
    "",
    response_model=QueryResponse,
    summary="执行一次非流式知识库查询",
)
async def query(payload: QueryRequest, current_user: CurrentUser) -> QueryResponse:
    state, public_session_id = _build_query_state(
        payload,
        is_stream=False,
        user_id=str(current_user["id"]),
    )
    try:
        result = await run_in_threadpool(KBQueryWorkflow().run, state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识库查询失败: {exc}",
        ) from exc

    return QueryResponse(
        task_id=str(result["task_id"]),
        session_id=public_session_id,
        message_id=str(result["message_id"]),
        item_names=list(result.get("item_names") or []),
        rewritten_query=str(result.get("rewritten_query") or payload.query),
        answer=str(result.get("answer") or ""),
        answer_html=render_markdown(str(result.get("answer") or "")),
        sources=serialize_sources(result.get("reranked_docs")),
    )


@router.post(
    "/stream",
    response_class=StreamingResponse,
    summary="执行一次 POST-SSE 流式知识库查询",
)
async def stream_query(
    payload: QueryRequest,
    current_user: CurrentUser,
) -> StreamingResponse:
    state, public_session_id = _build_query_state(
        payload,
        is_stream=True,
        user_id=str(current_user["id"]),
    )

    async def event_stream() -> AsyncIterator[str]:
        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        event_loop = asyncio.get_running_loop()
        active_stage = {"name": "item_confirm"}

        def stream_writer(event: dict[str, Any]) -> None:
            if event.get("event") == "final":
                event = dict(event)
                data = dict(event.get("data") or {})
                data["answer_html"] = render_markdown(
                    str(data.get("answer") or "")
                )
                event["data"] = data
            event_loop.call_soon_threadsafe(event_queue.put_nowait, event)

        def emit_progress(stage: str, stage_status: str, **extra: Any) -> None:
            if stage_status == "running":
                active_stage["name"] = stage
            stream_writer(
                {
                    "event": "progress",
                    "data": {
                        "task_id": state["task_id"],
                        "stage": stage,
                        "status": stage_status,
                        **extra,
                    },
                }
            )

        def run_workflow() -> None:
            workflow = KBQueryWorkflow(stream_writer=stream_writer)
            mode = str(state.get("search_mode") or "fast")
            emit_progress("item_confirm", "running")
            for update in workflow.stream_updates(state):
                for node_name, values in update.items():
                    value_map = values if isinstance(values, dict) else {}
                    if node_name == "node_item_name_confirm":
                        emit_progress(
                            "item_confirm",
                            "completed",
                            item_names=value_map.get("item_names") or [],
                        )
                        if not value_map.get("answer"):
                            emit_progress("embedding", "running")
                            emit_progress(
                                "hyde",
                                "running" if mode == "deep" else "skipped",
                            )
                            emit_progress(
                                "web",
                                "running" if mode == "deep" else "skipped",
                            )
                    elif node_name == "node_search_embedding":
                        emit_progress(
                            "embedding",
                            "completed",
                            count=len(value_map.get("embedding_chunks") or []),
                        )
                    elif node_name == "node_search_embedding_hyde":
                        if mode == "deep":
                            emit_progress(
                                "hyde",
                                "completed",
                                count=len(value_map.get("hyde_embedding_chunks") or []),
                            )
                    elif node_name == "node_web_search_mcp":
                        if mode == "deep":
                            emit_progress(
                                "web",
                                "completed",
                                count=len(value_map.get("web_search_docs") or []),
                            )
                    elif node_name == "node_rrf":
                        emit_progress(
                            "rrf",
                            "skipped" if mode == "fast" else "completed",
                            count=len(value_map.get("rrf_chunks") or []),
                        )
                        emit_progress("rerank", "running")
                    elif node_name == "node_rerank":
                        emit_progress(
                            "rerank",
                            "completed",
                            count=len(value_map.get("reranked_docs") or []),
                        )
                        emit_progress("answer", "running")
                    elif node_name == "node_answer_output":
                        emit_progress("answer", "completed")

        async def execute_workflow() -> None:
            try:
                await run_in_threadpool(run_workflow)
            except Exception as exc:
                logger.exception(
                    "流式查询失败 | task_id=%s",
                    state["task_id"],
                )
                await event_queue.put(
                    {
                        "event": "progress",
                        "data": {
                            "task_id": state["task_id"],
                            "stage": active_stage["name"],
                            "status": "failed",
                        },
                    }
                )
                await event_queue.put(
                    {
                        "event": "error",
                        "data": {
                            "task_id": state["task_id"],
                            "message": str(exc),
                        },
                    }
                )
            finally:
                await event_queue.put(None)

        worker = asyncio.create_task(execute_workflow())
        try:
            yield encode_sse(
                {
                    "event": "start",
                    "data": {
                        "task_id": state["task_id"],
                        "session_id": public_session_id,
                        "message_id": state["message_id"],
                    },
                }
            )
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield encode_sse(event)
        finally:
            if not worker.done():
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
