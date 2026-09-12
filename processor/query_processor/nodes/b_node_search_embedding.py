"""B：普通混合向量检索节点。"""

from __future__ import annotations

import json
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from processor.query_processor.base import NodeBase
from processor.query_processor.config import QueryConfig
from processor.query_processor.state import QueryGraphState


class NodeSearchEmbedding(NodeBase):
    """根据商品名和改写后的问题检索知识切片。"""

    name = "node_search_embedding"

    def __init__(
        self,
        config: QueryConfig | None = None,
        *,
        embedding_tool=None,
        milvus_client=None,
    ) -> None:
        super().__init__()
        self.config = config or QueryConfig()
        self._embedding_tool = embedding_tool
        self._milvus_client = milvus_client

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行稠密/稀疏混合检索", self.name)
        rewritten_query, item_names = self._get_query_context(state)
        chunks = self._search_chunks(
            rewritten_query,
            item_names,
            source="embedding",
        )
        return {"embedding_chunks": chunks}

    @staticmethod
    def _get_query_context(
        state: QueryGraphState,
    ) -> tuple[str, list[str]]:
        rewritten_query = state.get("rewritten_query")
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            raise ValueError("rewritten_query 必须是非空字符串")

        raw_item_names = state.get("item_names") or []
        if not isinstance(raw_item_names, list):
            raise ValueError("item_names 必须是字符串列表")

        item_names = []
        seen = set()
        for item_name in raw_item_names:
            if not isinstance(item_name, str) or not item_name.strip():
                raise ValueError("item_names 必须是非空字符串列表")
            normalized_name = item_name.strip()
            key = normalized_name.casefold()
            if key not in seen:
                seen.add(key)
                item_names.append(normalized_name)
        return rewritten_query.strip(), item_names

    def _search_chunks(
        self,
        query_text: str,
        item_names: list[str],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        collection_name = str(self.config.chunks_collection or "").strip()
        if not collection_name:
            raise RuntimeError("未配置 CHUNKS_COLLECTION")

        top_k = int(self.config.search_top_k)
        if top_k <= 0:
            raise RuntimeError("QUERY_SEARCH_TOP_K 必须大于 0")

        client = self._get_milvus_client()
        if not client.has_collection(collection_name):
            raise RuntimeError(f"Milvus Collection 不存在: {collection_name}")

        dense_vector, sparse_vector = self._generate_query_vectors(query_text)
        expression = self._build_item_filter(item_names)
        requests = [
            AnnSearchRequest(
                data=[dense_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=top_k,
                expr=expression,
            ),
            AnnSearchRequest(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}},
                limit=top_k,
                expr=expression,
            ),
        ]
        results = client.hybrid_search(
            collection_name=collection_name,
            reqs=requests,
            ranker=WeightedRanker(
                float(self.config.search_dense_weight),
                float(self.config.search_sparse_weight),
            ),
            limit=top_k,
            output_fields=[
                "chunk_id",
                "file_title",
                "title",
                "content",
                "chunk_order",
                "item_name",
                "metadata",
            ],
        )
        if not results:
            return []
        if len(results) != 1:
            raise RuntimeError("Milvus 返回的查询结果数量不正确")
        return self._normalize_hits(results[0], source=source)

    def _generate_query_vectors(
        self,
        query_text: str,
    ) -> tuple[list[float], dict[int, float]]:
        embedding_tool = self._embedding_tool
        if embedding_tool is None:
            from utils.embedding_utils import embedding_tool as default_embedding_tool

            embedding_tool = default_embedding_tool

        dense_vectors, sparse_vectors = embedding_tool.embed_dense_and_sparse(
            [query_text],
            text_type="query",
        )
        if len(dense_vectors) != 1 or not dense_vectors[0]:
            raise RuntimeError("查询嵌入模型未返回有效稠密向量")
        if len(sparse_vectors) != 1 or not sparse_vectors[0]:
            raise RuntimeError("查询嵌入模型未返回有效稀疏向量")
        return dense_vectors[0], sparse_vectors[0]

    @staticmethod
    def _build_item_filter(item_names: list[str]) -> str | None:
        if not item_names:
            return None
        encoded_names = json.dumps(item_names, ensure_ascii=False)
        return f"item_name in {encoded_names}"

    @staticmethod
    def _normalize_hits(
        hits: list[dict[str, Any]],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        chunks = []
        for hit in hits:
            if not isinstance(hit, dict):
                raise RuntimeError("Milvus 返回了无法识别的文档切片")
            entity = hit.get("entity")
            if not isinstance(entity, dict):
                raise RuntimeError("Milvus 文档切片缺少 entity")

            metadata_value = entity.get("metadata")
            metadata = (
                dict(metadata_value)
                if isinstance(metadata_value, dict)
                else {}
            )
            for field_name in ("file_title", "chunk_order", "item_name"):
                value = entity.get(field_name)
                if value is not None:
                    metadata[field_name] = value

            chunk_id = hit.get("id", entity.get("chunk_id"))
            content = entity.get("content")
            if chunk_id is None or not isinstance(content, str):
                raise RuntimeError("Milvus 文档切片缺少主键或正文")

            score = hit.get("distance", hit.get("score", 0.0))
            chunks.append(
                {
                    "id": chunk_id,
                    "title": str(
                        entity.get("title") or entity.get("file_title") or ""
                    ),
                    "content": content,
                    "score": float(score),
                    "source": source,
                    "metadata": metadata,
                }
            )
        return chunks

    def _get_milvus_client(self):
        if self._milvus_client is not None:
            return self._milvus_client

        milvus_url = str(self.config.milvus_url or "").strip()
        if not milvus_url:
            raise RuntimeError("未配置 MILVUS_URL")
        self._milvus_client = MilvusClient(uri=milvus_url)
        return self._milvus_client
