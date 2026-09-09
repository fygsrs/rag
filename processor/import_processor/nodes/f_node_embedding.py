from __future__ import annotations

import json
from copy import deepcopy

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import EmbeddingError, StateFieldError
from processor.import_processor.state import EmbeddedChunkDict, ImportGraphState


class NodeEmbedding(BaseNode):
    """混合向量化节点：支持阿里云和本地 BGE-M3。"""

    name = "node_embedding"

    def __init__(self, config=None, *, embedding_tool=None):
        super().__init__(config)
        self._embedding_tool = embedding_tool

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """校验切片、分批生成 dense/sparse，并写回状态。"""
        # 1. 输入校验
        chunks = self._step_1_validate_input(state)

        # 2. 分批生成稠密和稀疏向量
        embedded_chunks = self._step_2_generate_embeddings(state, chunks)

        # 3. 更新状态
        return self._step_3_update_state(state, embedded_chunks)

    def _step_1_validate_input(
        self,
        state: ImportGraphState,
    ) -> list[EmbeddedChunkDict]:
        """获取并校验待向量化的切片。"""
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                expected_type=list,
            )

        for index, chunk in enumerate(chunks):
            content = chunk.get("content") if isinstance(chunk, dict) else None
            if isinstance(content, str) and content.strip():
                continue
            raise EmbeddingError(
                message=f"第 {index + 1} 个切片内容为空",
                node_name=self.name,
            )
        return chunks

    def _step_2_generate_embeddings(
        self,
        state: ImportGraphState,
        chunks: list[EmbeddedChunkDict],
    ) -> list[EmbeddedChunkDict]:
        """分批向量化；每批只调用一次当前 provider。"""
        tool = self._get_embedding_tool()
        batch_size = self.config.embedding_batch_size
        if batch_size <= 0:
            raise EmbeddingError(
                message="embedding_batch_size 必须大于 0",
                node_name=self.name,
            )

        item_name = state.get("item_name")
        if not isinstance(item_name, str):
            item_name = ""
        item_name = item_name.strip()

        output: list[EmbeddedChunkDict] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [self._build_embedding_text(chunk, item_name) for chunk in batch]

            try:
                dense_vectors, sparse_vectors = tool.embed_dense_and_sparse(texts)
            except Exception as exc:
                raise EmbeddingError(
                    message=(
                        f"切片向量生成失败，批次范围 "
                        f"{start + 1}-{start + len(batch)}"
                    ),
                    node_name=self.name,
                    cause=exc,
                ) from exc

            self._validate_vectors(
                dense_vectors,
                sparse_vectors,
                expected_count=len(batch),
                batch_start=start,
            )
            for chunk, dense, sparse in zip(
                batch,
                dense_vectors,
                sparse_vectors,
            ):
                embedded: EmbeddedChunkDict = deepcopy(chunk)
                embedded["dense_vector"] = [float(value) for value in dense]
                embedded["sparse_vector"] = {
                    int(index): float(value) for index, value in sparse.items()
                }
                output.append(embedded)

            self.logger.info(
                "完成切片 %d-%d 的向量化",
                start + 1,
                start + len(batch),
            )
        return output

    @staticmethod
    def _step_3_update_state(
        state: ImportGraphState,
        embedded_chunks: list[EmbeddedChunkDict],
    ) -> ImportGraphState:
        """把向量化后的切片写回工作流状态。"""
        state["chunks"] = embedded_chunks
        return state

    @staticmethod
    def _build_embedding_text(
        chunk: EmbeddedChunkDict,
        state_item_name: str,
    ) -> str:
        """商品名参与向量化，增强同一商品下切片的语义归属。"""
        metadata = chunk.get("metadata") or {}
        metadata_item_name = metadata.get("item_name")
        item_name = (
            metadata_item_name.strip()
            if isinstance(metadata_item_name, str) and metadata_item_name.strip()
            else state_item_name
        )
        content = chunk["content"].strip()
        return f"{item_name}\n{content}" if item_name else content

    def _validate_vectors(
        self,
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
        *,
        expected_count: int,
        batch_start: int,
    ) -> None:
        if len(dense_vectors) != expected_count:
            raise EmbeddingError(
                message=(
                    f"稠密向量数量不匹配，批次起点 {batch_start + 1}，"
                    f"期望 {expected_count}，实际 {len(dense_vectors)}"
                ),
                node_name=self.name,
            )
        if len(sparse_vectors) != expected_count:
            raise EmbeddingError(
                message=(
                    f"稀疏向量数量不匹配，批次起点 {batch_start + 1}，"
                    f"期望 {expected_count}，实际 {len(sparse_vectors)}"
                ),
                node_name=self.name,
            )

        for offset, (dense, sparse) in enumerate(
            zip(dense_vectors, sparse_vectors)
        ):
            chunk_number = batch_start + offset + 1
            if len(dense) != self.config.embedding_dim:
                raise EmbeddingError(
                    message=(
                        f"第 {chunk_number} 个切片的稠密向量维度不匹配，"
                        f"期望 {self.config.embedding_dim}，实际 {len(dense)}"
                    ),
                    node_name=self.name,
                )
            if not isinstance(sparse, dict) or not sparse:
                raise EmbeddingError(
                    message=f"第 {chunk_number} 个切片缺少稀疏向量",
                    node_name=self.name,
                )

    def _get_embedding_tool(self):
        if self._embedding_tool is None:
            from utils.embedding_utils import embedding_tool

            self._embedding_tool = embedding_tool
        return self._embedding_tool


def main() -> None:
    """使用示例切片测试当前配置的阿里云或 BGE-M3 向量模型。"""
    setup_logging()
    state: ImportGraphState = {
        "item_name": "HAK180 安全栅",
        "chunks": [
            {
                "file_title": "NodeEmbedding 测试文档",
                "title": "产品介绍",
                "content": "HAK180 安全栅用于工业现场的信号隔离和安全保护。",
                "order": 1,
                "metadata": {"item_name": "HAK180 安全栅"},
            },
            {
                "file_title": "NodeEmbedding 测试文档",
                "title": "技术参数",
                "content": "额定工作电压为 24V DC。",
                "order": 2,
                "metadata": {"item_name": "HAK180 安全栅"},
            },
        ],
    }

    result = NodeEmbedding()(state)
    summary = [
        {
            "order": chunk["order"],
            "dense_dimension": len(chunk["dense_vector"]),
            "sparse_dimension": len(chunk["sparse_vector"]),
            "dense_preview": chunk["dense_vector"][:5],
        }
        for chunk in result["chunks"]
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
