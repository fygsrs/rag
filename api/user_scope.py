"""将公开会话 ID 映射到用户隔离的内部 ID。"""


def session_prefix(user_id: str) -> str:
    return f"{user_id}:"


def internal_session_id(user_id: str, session_id: str) -> str:
    return f"{session_prefix(user_id)}{session_id}"
