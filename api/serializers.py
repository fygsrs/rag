"""将工作流状态转换为稳定的 API 数据结构。"""

from typing import Any

from utils.text_utils import summarize_content


def serialize_sources(
    documents: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """只返回标题、摘要和定位信息，避免把完整切片发送给页面。"""
    sources = []
    for index, document in enumerate(documents or [], start=1):
        source = str(document.get("source") or "internal")
        score = document.get("score")
        summary = summarize_content(
            str(document.get("summary") or document.get("content") or "")
        )
        sources.append(
            {
                "index": index,
                "id": document.get("id"),
                "title": str(document.get("title") or ""),
                "url": str(document.get("url") or ""),
                "score": float(score) if isinstance(score, (int, float)) else None,
                "source": source,
                "source_type": (
                    "web" if source == "web_search" else "internal"
                ),
                "summary": summary,
                "metadata": dict(document.get("metadata") or {}),
            }
        )
    return sources
