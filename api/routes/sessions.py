"""最近会话和历史消息接口。"""

from fastapi import APIRouter, Query, status
from starlette.concurrency import run_in_threadpool

from api.markdown import render_markdown
from api.dependencies import CurrentUser
from api.schemas import SessionMessagesResponse, SessionSummary
from api.user_scope import internal_session_id, session_prefix
from utils.mongodb_utils import get_mongodb_util

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary], summary="读取最近会话")
async def list_sessions(
    current_user: CurrentUser,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[SessionSummary]:
    mongodb = get_mongodb_util()
    sessions = await run_in_threadpool(
        mongodb.list_sessions,
        limit=limit,
        session_prefix=session_prefix(str(current_user["id"])),
    )
    return [SessionSummary.model_validate(session) for session in sessions]


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除一个会话的全部消息",
)
async def delete_session(
    session_id: str,
    current_user: CurrentUser,
) -> None:
    mongodb = get_mongodb_util()
    await run_in_threadpool(
        mongodb.clear_session,
        internal_session_id(str(current_user["id"]), session_id),
    )


@router.get(
    "/{session_id}/messages",
    response_model=SessionMessagesResponse,
    summary="读取一个会话的历史消息",
)
async def get_session_messages(
    session_id: str,
    current_user: CurrentUser,
    limit: int = Query(default=50, ge=1, le=100),
) -> SessionMessagesResponse:
    mongodb = get_mongodb_util()
    memories = await run_in_threadpool(
        mongodb.get_memories,
        internal_session_id(str(current_user["id"]), session_id),
        limit=limit,
    )
    messages = []
    for memory in memories:
        content = str(memory.get("content") or "")
        messages.append(
            {
                "message_id": str(memory.get("message_id") or ""),
                "role": str(memory.get("role") or ""),
                "content": content,
                "content_html": (
                    render_markdown(content)
                    if memory.get("role") == "assistant"
                    else ""
                ),
                "created_at": memory.get("created_at"),
                "metadata": dict(memory.get("metadata") or {}),
            }
        )
    return SessionMessagesResponse(session_id=session_id, messages=messages)
