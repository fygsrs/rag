"""Server-Sent Events 编码工具。"""

import json
from typing import Any


def encode_sse(event: dict[str, Any]) -> str:
    """把内部事件编码为浏览器可消费的 SSE 文本。"""
    event_name = str(event.get("event") or "message")
    data = event.get("data", {})
    payload = json.dumps(data, ensure_ascii=False, default=str)
    event_id = event.get("id")
    id_line = f"id: {event_id}\n" if event_id is not None else ""
    return f"{id_line}event: {event_name}\ndata: {payload}\n\n"
