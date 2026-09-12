"""E：RRF 多路结果融合节点。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from processor.query_processor.base import NodeBase
from processor.query_processor.config import QueryConfig
from processor.query_processor.state import QueryGraphState


class NodeRrf(NodeBase):
    """使用加权倒数排名融合普通检索和 HyDE 检索结果。"""

    name = "node_rrf"

    def __init__(self, config: QueryConfig | None = None) -> None:
        super().__init__()
        self.config = config or QueryConfig()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行 RRF 融合排序", self.name)
        embedding_chunks = self._get_documents(state, "embedding_chunks")
        hyde_chunks = self._get_documents(state, "hyde_embedding_chunks")
        rrf_chunks = self._rrf_merge(
            [
                (embedding_chunks, float(self.config.rrf_embedding_weight)),
                (hyde_chunks, float(self.config.rrf_hyde_weight)),
            ]
        )
        return {"rrf_chunks": rrf_chunks}

    @staticmethod
    def _get_documents(
        state: QueryGraphState,
        field_name: str,
    ) -> list[dict[str, Any]]:
        documents = state.get(field_name) or []
        if not isinstance(documents, list):
            raise ValueError(f"{field_name} 必须是文档列表")
        return documents

    def _rrf_merge(
        self,
        inputs: list[tuple[list[dict[str, Any]], float]],
    ) -> list[dict[str, Any]]:
        rrf_k = int(self.config.rrf_k)
        max_results = int(self.config.rrf_max_results)
        if rrf_k < 0:
            raise RuntimeError("RRF_K 不能小于 0")
        if max_results <= 0:
            raise RuntimeError("RRF_MAX_RESULTS 必须大于 0")

        scores: dict[str, float] = {}
        documents_by_key: dict[str, dict[str, Any]] = {}
        for documents, weight in inputs:
            if weight < 0:
                raise RuntimeError("RRF 检索权重不能小于 0")
            for rank, document in enumerate(documents, start=1):
                if not isinstance(document, dict):
                    raise ValueError("RRF 输入中包含无效文档")
                key = self._document_key(document)
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
                if key not in documents_by_key:
                    saved_document = deepcopy(document)
                    saved_document.pop("score", None)
                    documents_by_key[key] = saved_document

        ordered_keys = sorted(scores, key=scores.get, reverse=True)
        return [
            documents_by_key[key]
            for key in ordered_keys[:max_results]
        ]

    @staticmethod
    def _document_key(document: dict[str, Any]) -> str:
        document_id = document.get("id")
        if document_id is not None:
            return f"id:{document_id}"
        url = document.get("url")
        if isinstance(url, str) and url.strip():
            return f"url:{url.strip()}"
        content = document.get("content")
        if isinstance(content, str) and content.strip():
            return f"content:{' '.join(content.casefold().split())}"
        raise ValueError("RRF 文档缺少 id、url 和 content")
