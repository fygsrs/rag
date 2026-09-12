"""F：精排节点。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx

from processor.query_processor.base import NodeBase
from processor.query_processor.config import QueryConfig
from processor.query_processor.state import QueryGraphState


class NodeRerank(NodeBase):
    """使用 Rerank 模型对 RRF 结果进行二次排序。"""

    name = "node_rerank"

    def __init__(
        self,
        config: QueryConfig | None = None,
        *,
        http_client=None,
    ) -> None:
        super().__init__()
        self.config = config or QueryConfig()
        self._http_client = http_client or httpx

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("rewritten_query 必须是非空字符串")

        documents = self._merge_documents(
            state.get("rrf_chunks") or [],
            state.get("web_search_docs") or [],
        )
        if not documents:
            self.logger.info("精排跳过 | reason=no_documents")
            return {"reranked_docs": []}

        scores = self._rerank_documents(
            query.strip(),
            [document["content"] for document in documents],
        )
        ranked_documents = []
        for original_index, score in scores:
            document = deepcopy(documents[original_index])
            document["score"] = score
            ranked_documents.append(document)

        cutoff = self._find_cliff_cutoff(ranked_documents)
        top_score = float(ranked_documents[0]["score"]) if ranked_documents else 0.0
        self.logger.info(
            "精排完成 | input=%d | output=%d | cutoff=%d | top_score=%.4f",
            len(documents),
            min(cutoff, len(ranked_documents)),
            cutoff,
            top_score,
        )
        return {"reranked_docs": ranked_documents[:cutoff]}

    def _merge_documents(
        self,
        rrf_chunks: list[dict[str, Any]],
        web_search_docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(rrf_chunks, list) or not isinstance(web_search_docs, list):
            raise ValueError("rrf_chunks 和 web_search_docs 必须是文档列表")

        merged = []
        seen = set()
        for document in [*rrf_chunks, *web_search_docs]:
            if not isinstance(document, dict):
                raise ValueError("重排输入中包含无效文档")
            content = document.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            key = self._document_key(document)
            if key in seen:
                continue
            seen.add(key)
            merged.append(deepcopy(document))
        return merged

    @staticmethod
    def _document_key(document: dict[str, Any]) -> str:
        document_id = document.get("id")
        if document_id is not None:
            return f"id:{document_id}"
        url = document.get("url")
        if isinstance(url, str) and url.strip():
            return f"url:{url.strip()}"
        content = str(document.get("content") or "")
        return f"content:{' '.join(content.casefold().split())}"

    def _rerank_documents(
        self,
        query: str,
        contents: list[str],
    ) -> list[tuple[int, float]]:
        rerank_url = str(self.config.rerank_url or "").strip()
        api_key = str(self.config.dashscope_api_key or "").strip()
        model = str(self.config.rerank_model or "").strip()
        if not rerank_url:
            raise RuntimeError("未配置 DASHSCOPE_RERANK_URL")
        if not api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")
        if not model:
            raise RuntimeError("未配置 DASHSCOPE_RERANK_MODEL")

        response = self._http_client.post(
            rerank_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": {
                    "query": query,
                    "documents": contents,
                },
                "parameters": {
                    "top_n": len(contents),
                    "instruct": (
                        "Given a web search query, retrieve relevant passages "
                        "that answer the query."
                    ),
                },
            },
            timeout=float(self.config.rerank_timeout),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code"):
            raise RuntimeError(
                f"DashScope Rerank 调用失败: "
                f"{payload.get('message') or payload['code']}"
            )
        results = payload.get("output", {}).get("results")
        if not isinstance(results, list) or len(results) != len(contents):
            raise RuntimeError("DashScope Rerank 返回结果数量不正确")

        ranked = []
        seen_indices = set()
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("DashScope Rerank 返回了无效结果")
            index = item.get("index")
            score = item.get("relevance_score")
            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(contents)
                or index in seen_indices
                or not isinstance(score, (int, float))
            ):
                raise RuntimeError("DashScope Rerank 返回的索引或分数无效")
            seen_indices.add(index)
            ranked.append((index, float(score)))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _find_cliff_cutoff(
        self,
        ranked_documents: list[dict[str, Any]],
    ) -> int:
        upper_bound = min(
            int(self.config.rerank_max_results),
            len(ranked_documents),
        )
        if upper_bound <= 0:
            return 0
        lower_bound = min(
            max(1, int(self.config.rerank_min_results)),
            upper_bound,
        )
        absolute_threshold = float(self.config.rerank_absolute_gap)
        relative_threshold = float(self.config.rerank_relative_gap)

        for index in range(lower_bound - 1, upper_bound - 1):
            current_score = float(ranked_documents[index]["score"])
            next_score = float(ranked_documents[index + 1]["score"])
            absolute_gap = current_score - next_score
            relative_gap = absolute_gap / max(abs(current_score), 1e-12)
            if (
                absolute_gap > absolute_threshold
                and relative_gap > relative_threshold
            ):
                return index + 1
        return upper_bound
