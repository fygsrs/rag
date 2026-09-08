from __future__ import annotations

from typing import Any

import httpx

from config.embedding_config import embedding_config


SparseVector = dict[int, float]


class EmbeddingTool:
    """默认使用阿里云，也可按配置切换到本地 BGE-M3。

    BGE-M3 使用 ``FlagEmbedding`` 同时生成 dense/sparse；阿里云使用
    DashScope 原生接口，因为 OpenAI 兼容接口只返回稠密向量。
    """

    def __init__(self, config=None):
        self.config = config or embedding_config
        self._bge_model: Any = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对文档列表生成稠密向量。"""
        dense_vectors, _ = self._embed(texts, output_type="dense")
        return dense_vectors

    def embed_query(self, text: str) -> list[float]:
        """对查询文本生成稠密向量。"""
        dense_vectors, _ = self._embed(
            [text],
            output_type="dense",
            text_type="query",
        )
        return dense_vectors[0]

    def embed_dense_and_sparse(
        self,
        texts: list[str],
        *,
        text_type: str = "document",
    ) -> tuple[list[list[float]], list[SparseVector]]:
        """同时生成稠密和稀疏向量。"""
        return self._embed(
            texts,
            output_type="dense&sparse",
            text_type=text_type,
        )

    def _embed(
        self,
        texts: list[str],
        *,
        output_type: str,
        text_type: str = "document",
    ) -> tuple[list[list[float]], list[SparseVector]]:
        if not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("向量化文本必须是非空字符串列表")

        provider = (self.config.provider or "aliyun").strip().lower()
        if provider == "bge":
            return self._embed_bge(texts, output_type=output_type)
        if provider == "aliyun":
            return self._embed_aliyun(
                texts,
                output_type=output_type,
                text_type=text_type,
            )
        raise ValueError(f"不支持的向量提供方: {self.config.provider}")

    def _get_bge_model(self):
        if self._bge_model is not None:
            return self._bge_model

        from FlagEmbedding import BGEM3FlagModel

        model_name = self.config.bge_m3_path or self.config.bge_m3
        if not model_name:
            raise ValueError("未配置 BGE-M3 模型，请设置 BGE_M3_PATH 或 BGE_M3")

        kwargs = {"use_fp16": self.config.bge_fp16}
        if self.config.bge_device:
            kwargs["device"] = self.config.bge_device
        self._bge_model = BGEM3FlagModel(model_name, **kwargs)
        return self._bge_model

    def _embed_bge(
        self,
        texts: list[str],
        *,
        output_type: str,
    ) -> tuple[list[list[float]], list[SparseVector]]:
        need_sparse = output_type != "dense"
        result = self._get_bge_model().encode(
            texts,
            batch_size=self.config.batch_size,
            return_dense=True,
            return_sparse=need_sparse,
            return_colbert_vecs=False,
        )
        dense_vectors = [vector.tolist() for vector in result["dense_vecs"]]
        if not need_sparse:
            return dense_vectors, []

        sparse_vectors = [
            {int(index): float(value) for index, value in weights.items()}
            for weights in result["lexical_weights"]
        ]
        return dense_vectors, sparse_vectors

    def _embed_aliyun(
        self,
        texts: list[str],
        *,
        output_type: str,
        text_type: str,
    ) -> tuple[list[list[float]], list[SparseVector]]:
        if not self.config.dashscope_api_key:
            raise ValueError("未配置阿里云向量 API Key，请设置 DASHSCOPE_API_KEY")

        response = httpx.post(
            self.config.dashscope_native_url,
            headers={
                "Authorization": f"Bearer {self.config.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.dashscope_model,
                "input": {"texts": texts},
                "parameters": {
                    "dimension": self.config.dimension,
                    "output_type": output_type,
                    "text_type": text_type,
                },
            },
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code"):
            raise RuntimeError(
                f"阿里云向量接口调用失败: {payload.get('message') or payload['code']}"
            )

        embeddings = payload.get("output", {}).get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("阿里云向量接口返回的向量数量不正确")

        embeddings = sorted(embeddings, key=lambda item: item.get("text_index", 0))
        dense_vectors: list[list[float]] = []
        sparse_vectors: list[SparseVector] = []
        for item in embeddings:
            dense = item.get("embedding")
            if not isinstance(dense, list) or not dense:
                raise RuntimeError("阿里云向量接口未返回稠密向量")
            dense_vectors.append([float(value) for value in dense])

            if output_type != "dense":
                sparse = item.get("sparse_embedding")
                if not isinstance(sparse, list) or not sparse:
                    raise RuntimeError(
                        f"模型 {self.config.dashscope_model} 未返回稀疏向量"
                    )
                sparse_vectors.append(
                    {
                        int(value["index"]): float(value["value"])
                        for value in sparse
                    }
                )

        return dense_vectors, sparse_vectors


embedding_tool = EmbeddingTool()


if __name__ == "__main__":
    print(f"当前向量 provider: {embedding_config.provider}")
    print(f"向量维度: {len(embedding_tool.embed_query('测试'))}")
