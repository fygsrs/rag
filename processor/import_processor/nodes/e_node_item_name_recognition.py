from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from pymilvus import DataType, MilvusClient

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.config import get_config
from processor.import_processor.exceptions import (
    EmbeddingError,
    LLMError,
    MilvusError,
    StateFieldError,
)
from processor.import_processor.state import (
    ChunkDict,
    ImportGraphState,
)


class NodeItemNameRecognition(BaseNode):
    """识别商品主体，并把商品级向量写入 Milvus。"""

    name = "node_item_name_recognition"
    _ITEM_NAME_MAX_BYTES = 512

    def __init__(
        self,
        config=None,
        *,
        llm_client=None,
        embedding_tool=None,
        milvus_client=None,
    ):
        super().__init__(config)
        self._llm_client = llm_client
        self._embedding_tool = embedding_tool
        self._milvus_client = milvus_client

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """执行商品主体识别、向量化及商品 Collection 入库流程。"""
        # 1. 提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 2. 构建大模型识别上下文
        context = self._step_2_build_context(chunks)

        # 3. 调用大模型识别商品名称
        item_name = self._step_3_call_llm(file_title, context)

        # 4. 回填商品名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)

        # 5. 生成商品名称的稠密/稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)

        # 6. 将商品级数据存入 Milvus
        self._step_6_save_to_milvus(
            file_title,
            item_name,
            dense_vector,
            sparse_vector,
        )

        self.logger.info("--- 识别完成: %s ---", item_name)
        return state

    def _step_1_get_inputs(
        self,
        state: ImportGraphState,
    ) -> tuple[str, list[ChunkDict]]:
        """提取并校验文件标题和切片。"""
        file_title = state.get("file_title")
        chunks = state.get("chunks")

        if not isinstance(file_title, str) or not file_title.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="file_title",
                expected_type=str,
            )
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                expected_type=list,
            )

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise StateFieldError(
                    node_name=self.name,
                    field_name=f"chunks[{index}]",
                    expected_type=dict,
                )
            content = chunk.get("content")
            metadata = chunk.get("metadata")
            if not isinstance(content, str) or not content.strip():
                raise StateFieldError(
                    node_name=self.name,
                    field_name=f"chunks[{index}].content",
                    expected_type=str,
                )
            if not isinstance(metadata, dict):
                raise StateFieldError(
                    node_name=self.name,
                    field_name=f"chunks[{index}].metadata",
                    expected_type=dict,
                )

        return file_title.strip(), chunks

    def _step_2_build_context(self, chunks: list[ChunkDict]) -> str:
        """从文档前几个切片构建商品主体识别上下文。"""
        chunk_count = max(1, self.config.item_name_chunk_k)
        chunk_size = max(1, self.config.item_name_chunk_size)
        context_parts = []

        for chunk in chunks[:chunk_count]:
            title = chunk.get("title", "").strip()
            content = chunk["content"].strip()[:chunk_size]
            context_parts.append(f"[章节：{title or '无标题'}]\n{content}")

        return "\n\n".join(context_parts)

    def _step_3_call_llm(self, file_title: str, context: str) -> str:
        """调用大模型，返回唯一的商品主体名称。"""
        client = self._llm_client
        if client is None:
            from utils.llm_utils import get_llm_client

            client = get_llm_client(
                model=self.config.item_model,
                json_mode=True,
            )

        prompt = (
            "你是商品知识库的主体识别器。请根据文件标题和文档片段，"
            "提取文档描述的唯一商品、产品或设备名称。\n"
            "要求：\n"
            "1. 保留能区分商品的品牌、系列和型号；\n"
            "2. 不要把章节名、文档类型、公司名称或宣传语当成商品名；\n"
            "3. 不要解释，只返回 JSON：{\"item_name\": \"商品名称\"}；\n"
            "4. 文档信息不足时，优先从文件标题中提取，不要编造。\n\n"
            f"文件标题：{file_title}\n\n"
            f"文档片段：\n{context}"
        )

        try:
            response = client.invoke(prompt)
            response_text = self._response_to_text(response)
            item_name = self._parse_item_name(response_text)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                message="商品名称识别失败",
                node_name=self.name,
                cause=exc,
            ) from exc

        if len(item_name.encode("utf-8")) > self._ITEM_NAME_MAX_BYTES:
            raise LLMError(
                message="大模型返回的商品名称过长",
                node_name=self.name,
            )
        return item_name

    @staticmethod
    def _response_to_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "\n".join(texts).strip()
        return str(content).strip()

    def _parse_item_name(self, response_text: str) -> str:
        text = response_text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                message="商品名称识别结果不是有效 JSON",
                node_name=self.name,
                cause=exc,
            ) from exc

        item_name = payload.get("item_name") if isinstance(payload, dict) else None
        if not isinstance(item_name, str) or not item_name.strip():
            raise LLMError(
                message="商品名称识别结果缺少 item_name",
                node_name=self.name,
            )
        return item_name.strip()

    @staticmethod
    def _step_4_update_chunks(
        state: ImportGraphState,
        chunks: list[ChunkDict],
        item_name: str,
    ) -> None:
        """回填状态及每个切片的 metadata，不改变 ChunkDict 结构。"""
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["metadata"]["item_name"] = item_name

    def _step_5_generate_vectors(
        self,
        item_name: str,
    ) -> tuple[list[float], dict[int, float]]:
        """使用当前配置的单一向量提供方生成商品名称向量。"""
        tool = self._embedding_tool
        if tool is None:
            from utils.embedding_utils import embedding_tool

            tool = embedding_tool

        try:
            dense_vectors, sparse_vectors = tool.embed_dense_and_sparse([item_name])
        except Exception as exc:
            raise EmbeddingError(
                message=f"商品名称向量生成失败: {item_name}",
                node_name=self.name,
                cause=exc,
            ) from exc

        if (
            len(dense_vectors) != 1
            or len(dense_vectors[0]) != self.config.embedding_dim
        ):
            actual_dim = len(dense_vectors[0]) if dense_vectors else 0
            raise EmbeddingError(
                message=(
                    f"稠密向量维度不匹配，期望 {self.config.embedding_dim}，"
                    f"实际 {actual_dim}"
                ),
                node_name=self.name,
            )
        if len(sparse_vectors) != 1 or not sparse_vectors[0]:
            raise EmbeddingError(
                message="向量模型未返回稀疏向量",
                node_name=self.name,
            )
        return dense_vectors[0], sparse_vectors[0]

    def _step_6_save_to_milvus(
        self,
        file_title: str,
        item_name: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
    ) -> None:
        """创建商品 Collection（如需要），并按商品 ID 执行 upsert。"""
        client = self._get_milvus_client()
        collection_name = self.config.item_name_collection
        if not collection_name:
            raise MilvusError(
                message="未配置 ITEM_NAME_COLLECTION",
                node_name=self.name,
            )

        try:
            if not client.has_collection(collection_name):
                self._create_item_collection(client, collection_name)

            provider, model = self._get_embedding_identity()
            self._validate_collection_embedding(client, collection_name, provider, model)
            client.upsert(
                collection_name=collection_name,
                data={
                    "id": self._make_item_id(item_name),
                    "file_title": file_title,
                    "item_name": item_name,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                    "metadata": {"source": "item_name_recognition"},
                    "embedding_provider": provider,
                    "embedding_model": model,
                    "created_at": int(time.time() * 1000),
                },
            )
        except MilvusError:
            raise
        except Exception as exc:
            raise MilvusError(
                message=f"商品名称写入 Milvus 失败: {item_name}",
                node_name=self.name,
                cause=exc,
            ) from exc

    def _get_milvus_client(self):
        if self._milvus_client is not None:
            return self._milvus_client
        if not self.config.milvus_url:
            raise MilvusError(
                message="未配置 MILVUS_URL",
                node_name=self.name,
            )

        self._milvus_client = MilvusClient(uri=self.config.milvus_url)
        return self._milvus_client

    def _create_item_collection(self, client, collection_name: str) -> None:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
        )
        schema.add_field("file_title", DataType.VARCHAR, max_length=1024)
        schema.add_field("item_name", DataType.VARCHAR, max_length=512)
        schema.add_field(
            "dense_vector",
            DataType.FLOAT_VECTOR,
            dim=self.config.embedding_dim,
        )
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding_provider", DataType.VARCHAR, max_length=32)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=128)
        schema.add_field("created_at", DataType.INT64)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="item_name",
            index_name="item_name_index",
            index_type="INVERTED",
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _validate_collection_embedding(
        self,
        client,
        collection_name: str,
        provider: str,
        model: str,
    ) -> None:
        """阻止不同向量模型的数据写进同一个 Collection。"""
        rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["embedding_provider", "embedding_model"],
            limit=1,
        )
        if not rows:
            return

        existing_provider = rows[0].get("embedding_provider")
        existing_model = rows[0].get("embedding_model")
        if existing_provider != provider or existing_model != model:
            raise MilvusError(
                message=(
                    f"Collection {collection_name} 已使用向量模型 "
                    f"{existing_provider}/{existing_model}，不能写入 {provider}/{model}"
                ),
                node_name=self.name,
            )

    @staticmethod
    def _make_item_id(item_name: str) -> str:
        normalized_name = " ".join(item_name.casefold().split())
        return hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()

    def _get_embedding_identity(self) -> tuple[str, str]:
        from config.embedding_config import embedding_config

        tool_config = getattr(self._embedding_tool, "config", embedding_config)
        provider = (tool_config.provider or "aliyun").strip().lower()
        if provider == "aliyun":
            return provider, tool_config.dashscope_model
        return provider, tool_config.bge_m3 or tool_config.bge_m3_path


def main() -> None:
    """使用最小示例数据手动测试商品主体识别完整流程。"""
    setup_logging()
    config = get_config()
    state: ImportGraphState = {
        "file_title": "HAK180 产品安全手册",
        "chunks": [
            {
                "file_title": "HAK180 产品安全手册",
                "title": "产品介绍",
                "content": (
                    "# HAK180 产品安全手册\n\n"
                    "HAK180 安全栅用于工业自动化现场，为控制系统与危险区域设备"
                    "之间提供信号隔离和安全保护。"
                ),
                "order": 1,
                "metadata": {},
            },
            {
                "file_title": "HAK180 产品安全手册",
                "title": "技术参数",
                "content": (
                    "## 技术参数\n\n"
                    "产品型号为 HAK180，额定工作电压为 24V DC，"
                    "安装前请确认接线端子和使用环境。"
                ),
                "order": 2,
                "metadata": {},
            },
        ],
    }

    print(
        "开始测试 NodeItemNameRecognition："
        f"Milvus={config.milvus_url}，Collection={config.item_name_collection}"
    )
    result = NodeItemNameRecognition(config)(state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
