"""知识库查询工作流的状态定义。"""

from copy import deepcopy
from typing import Any, TypedDict


class RetrievedDocument(TypedDict, total=False):
    """各检索节点之间传递的统一文档结构。"""

    id: str | int
    title: str
    content: str
    url: str
    score: float
    source: str
    metadata: dict[str, Any]


class QueryGraphState(TypedDict, total=False):
    """查询流程中由各节点逐步补充的共享状态。"""

    session_id: str
    message_id: str
    original_query: str

    embedding_chunks: list[RetrievedDocument]
    hyde_embedding_chunks: list[RetrievedDocument]
    web_search_docs: list[RetrievedDocument]

    rrf_chunks: list[RetrievedDocument]
    reranked_docs: list[RetrievedDocument]

    prompt: str
    answer: str

    item_names: list[str]
    rewritten_query: str
    hyde_doc: str
    history: list[dict[str, Any]]
    is_stream: bool


QUERY_DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "message_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "hyde_doc": "",
    "history": [],
    "is_stream": False,
}


def create_default_query_state(**overrides: Any) -> QueryGraphState:
    """创建相互隔离的默认查询状态，并允许覆盖指定字段。"""
    state = deepcopy(QUERY_DEFAULT_STATE)
    state.update(overrides)
    return state
