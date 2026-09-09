from __future__ import annotations

import json
import time
from copy import deepcopy

from pymilvus import DataType, MilvusClient

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.config import get_config
from processor.import_processor.exceptions import MilvusError, StateFieldError
from processor.import_processor.state import EmbeddedChunkDict, ImportGraphState


class NodeImportMilvus(BaseNode):
    """数据入库节点：将向量化后的正文切片写入 Milvus。"""

    name = "node_import_milvus"
    _FILE_TITLE_MAX_BYTES = 1024
    _TITLE_MAX_BYTES = 2048
    _ITEM_NAME_MAX_BYTES = 512
    _CONTENT_MAX_BYTES = 65535

    def __init__(self, config=None, *, milvus_client=None):
        super().__init__(config)
        self._milvus_client = milvus_client

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """校验、准备 Collection、幂等清理、插入并更新状态。"""
        # 1. 输入数据校验并生成 Milvus 行数据
        chunks, rows, vector_dimension = self._step_1_check_input(state)

        # 2. 获取客户端并准备正文 Collection
        client = self._step_2_prepare_collection(vector_dimension)

        # 3. 按文件标题清理旧切片，保证重复导入幂等
        file_title = chunks[0]["file_title"].strip()
        self._step_3_clean_old_data(client, file_title)

        # 4. 批量写入并回填 Milvus 自动生成的 chunk_id
        updated_chunks = self._step_4_insert_data(client, chunks, rows)

        # 5. 更新工作流状态
        return self._step_5_update_state(state, updated_chunks)

    def _step_1_check_input(
        self,
        state: ImportGraphState,
    ) -> tuple[list[EmbeddedChunkDict], list[dict], int]:
        """校验入库关键条件，并转换成 Milvus 行数据。"""
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                expected_type=list,
            )

        first_chunk = chunks[0]
        file_title = (
            first_chunk.get("file_title") if isinstance(first_chunk, dict) else None
        )
        if not isinstance(file_title, str) or not file_title.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks[0].file_title",
                expected_type=str,
            )
        file_title = file_title.strip()

        state_item_name = str(state.get("item_name") or "").strip()
        rows: list[dict] = []
        provider, model = self._get_embedding_identity()
        created_at = int(time.time() * 1000)

        for index, chunk in enumerate(chunks):
            current_title = str(chunk.get("file_title") or "").strip()
            if current_title != file_title:
                raise MilvusError(
                    message="一次入库只能包含同一 file_title 的切片",
                    node_name=self.name,
                )

            dense_vector = chunk.get("dense_vector")
            sparse_vector = chunk.get("sparse_vector")
            if not isinstance(dense_vector, list) or len(
                dense_vector
            ) != self.config.embedding_dim:
                actual_dimension = len(dense_vector) if isinstance(dense_vector, list) else 0
                raise MilvusError(
                    message=(
                        f"第 {index + 1} 个切片向量维度为 {actual_dimension}，"
                        f"配置要求 {self.config.embedding_dim}"
                    ),
                    node_name=self.name,
                )
            if not isinstance(sparse_vector, dict) or not sparse_vector:
                raise MilvusError(
                    message=f"第 {index + 1} 个切片缺少 sparse_vector",
                    node_name=self.name,
                )

            metadata = chunk.get("metadata") or {}
            item_name = metadata.get("item_name") or state_item_name
            if not isinstance(item_name, str) or not item_name.strip():
                raise StateFieldError(
                    node_name=self.name,
                    field_name=f"chunks[{index}].metadata.item_name",
                    expected_type=str,
                )
            item_name = item_name.strip()

            rows.append(
                {
                    "file_title": file_title,
                    "title": chunk["title"].strip(),
                    "content": chunk["content"].strip(),
                    "chunk_order": chunk["order"],
                    "item_name": item_name,
                    "metadata": deepcopy(metadata),
                    "dense_vector": [float(value) for value in dense_vector],
                    "sparse_vector": {
                        int(key): float(value)
                        for key, value in sparse_vector.items()
                    },
                    "embedding_provider": provider,
                    "embedding_model": model,
                    "created_at": created_at,
                }
            )

        return chunks, rows, self.config.embedding_dim

    def _step_2_prepare_collection(self, vector_dimension: int):
        """取得 Milvus 客户端，不存在时创建正文 Collection。"""
        collection_name = self.config.chunks_collection
        if not collection_name:
            raise MilvusError(
                message="未配置 CHUNKS_COLLECTION",
                node_name=self.name,
            )

        client = self._get_milvus_client()
        try:
            if not client.has_collection(collection_name):
                self._create_chunks_collection(
                    client,
                    collection_name,
                    vector_dimension,
                )
            self._validate_collection_schema(
                client,
                collection_name,
                vector_dimension,
            )
            provider, model = self._get_embedding_identity()
            self._validate_collection_embedding(
                client,
                collection_name,
                provider,
                model,
            )
        except MilvusError:
            raise
        except Exception as exc:
            raise MilvusError(
                message=f"准备 Milvus Collection 失败: {collection_name}",
                node_name=self.name,
                cause=exc,
            ) from exc
        return client

    def _step_3_clean_old_data(self, client, file_title: str) -> None:
        """删除相同 file_title 的旧切片。"""
        try:
            result = client.delete(
                collection_name=self.config.chunks_collection,
                filter=f"file_title == {json.dumps(file_title, ensure_ascii=False)}",
            )
        except Exception as exc:
            raise MilvusError(
                message=f"清理旧切片失败: {file_title}",
                node_name=self.name,
                cause=exc,
            ) from exc

        delete_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        self.logger.info("已清理文档 %s 的 %d 条旧切片", file_title, delete_count)

    def _step_4_insert_data(
        self,
        client,
        chunks: list[EmbeddedChunkDict],
        rows: list[dict],
    ) -> list[EmbeddedChunkDict]:
        """批量写入 Milvus，并将自动主键回填到切片。"""
        try:
            result = client.insert(
                collection_name=self.config.chunks_collection,
                data=rows,
            )
        except Exception as exc:
            raise MilvusError(
                message="正文切片批量写入 Milvus 失败",
                node_name=self.name,
                cause=exc,
            ) from exc

        inserted_ids = result.get("ids", []) if isinstance(result, dict) else []
        if len(inserted_ids) != len(chunks):
            raise MilvusError(
                message=(
                    f"Milvus 返回的主键数量不匹配，期望 {len(chunks)}，"
                    f"实际 {len(inserted_ids)}"
                ),
                node_name=self.name,
            )

        updated_chunks: list[EmbeddedChunkDict] = []
        for chunk, chunk_id in zip(chunks, inserted_ids):
            updated: EmbeddedChunkDict = deepcopy(chunk)
            updated["chunk_id"] = int(chunk_id)
            updated_chunks.append(updated)

        self.logger.info("成功写入 %d 条正文切片", len(updated_chunks))
        return updated_chunks

    @staticmethod
    def _step_5_update_state(
        state: ImportGraphState,
        updated_chunks: list[EmbeddedChunkDict],
    ) -> ImportGraphState:
        state["chunks"] = updated_chunks
        return state

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

    def _create_chunks_collection(
        self,
        client,
        collection_name: str,
        vector_dimension: int,
    ) -> None:
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            "file_title",
            DataType.VARCHAR,
            max_length=self._FILE_TITLE_MAX_BYTES,
        )
        schema.add_field(
            "title",
            DataType.VARCHAR,
            max_length=self._TITLE_MAX_BYTES,
        )
        schema.add_field(
            "content",
            DataType.VARCHAR,
            max_length=self._CONTENT_MAX_BYTES,
        )
        schema.add_field("chunk_order", DataType.INT64)
        schema.add_field(
            "item_name",
            DataType.VARCHAR,
            max_length=self._ITEM_NAME_MAX_BYTES,
        )
        schema.add_field("metadata", DataType.JSON)
        schema.add_field(
            "dense_vector",
            DataType.FLOAT_VECTOR,
            dim=vector_dimension,
        )
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
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
            field_name="file_title",
            index_name="file_title_index",
            index_type="INVERTED",
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

    def _validate_collection_schema(
        self,
        client,
        collection_name: str,
        vector_dimension: int,
    ) -> None:
        description = client.describe_collection(collection_name)
        fields = {field["name"]: field for field in description.get("fields", [])}
        required_fields = {
            "chunk_id",
            "file_title",
            "title",
            "content",
            "chunk_order",
            "item_name",
            "metadata",
            "dense_vector",
            "sparse_vector",
            "embedding_provider",
            "embedding_model",
            "created_at",
        }
        missing_fields = sorted(required_fields - fields.keys())
        if missing_fields:
            raise MilvusError(
                message=(
                    f"Collection {collection_name} 缺少字段: "
                    f"{', '.join(missing_fields)}"
                ),
                node_name=self.name,
            )

        existing_dimension = int(fields["dense_vector"].get("params", {}).get("dim", 0))
        if existing_dimension != vector_dimension:
            raise MilvusError(
                message=(
                    f"Collection {collection_name} 的向量维度为 {existing_dimension}，"
                    f"当前数据为 {vector_dimension}"
                ),
                node_name=self.name,
            )

    def _validate_collection_embedding(
        self,
        client,
        collection_name: str,
        provider: str,
        model: str,
    ) -> None:
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
    def _get_embedding_identity() -> tuple[str, str]:
        from config.embedding_config import embedding_config

        provider = (embedding_config.provider or "aliyun").strip().lower()
        if provider == "aliyun":
            return provider, embedding_config.dashscope_model
        return provider, embedding_config.bge_m3 or embedding_config.bge_m3_path


def main() -> None:
    """向量化一条测试切片，写入 Milvus 后清理测试数据。"""
    from processor.import_processor.nodes.f_node_embedding import NodeEmbedding

    setup_logging()
    config = get_config()
    test_file_title = "__NodeImportMilvus_manual_test__"
    state: ImportGraphState = {
        "item_name": "HAK180 安全栅",
        "chunks": [
            {
                "file_title": test_file_title,
                "title": "测试章节",
                "content": "HAK180 安全栅用于工业现场的信号隔离与安全保护。",
                "order": 1,
                "metadata": {"item_name": "HAK180 安全栅"},
            }
        ],
    }

    import_node = NodeImportMilvus(config)
    import_started = False
    try:
        NodeEmbedding(config)(state)
        import_started = True
        result = import_node(state)
        summary = [
            {
                "chunk_id": chunk["chunk_id"],
                "file_title": chunk["file_title"],
                "dense_dimension": len(chunk["dense_vector"]),
                "sparse_dimension": len(chunk["sparse_vector"]),
            }
            for chunk in result["chunks"]
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if import_started:
            try:
                client = import_node._get_milvus_client()
                if client.has_collection(config.chunks_collection):
                    import_node._step_3_clean_old_data(client, test_file_title)
                    print("Milvus 测试数据已清理")
            except Exception as exc:
                print(f"Milvus 测试数据清理失败: {exc}")


if __name__ == "__main__":
    main()
