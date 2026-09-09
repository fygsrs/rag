import json
from pathlib import Path
from uuid import uuid4

from pymilvus import MilvusClient

from processor.import_processor.config import ImportConfig
from processor.import_processor.main_graph import ImportWorkflow
from processor.import_processor.state import create_default_state


class TestImportProcessorWorkflow:
    """使用真实阿里云服务和 Milvus 测试完整 Markdown 导入流程。"""

    def test_markdown_import_full_process(self, tmp_path):
        test_id = uuid4().hex
        product_name = f"CodexFlowTest-{test_id}"
        file_title = f"{product_name}_产品说明"
        item_collection = f"test_item_names_{test_id}"
        chunks_collection = f"test_chunks_{test_id}"
        markdown_path = tmp_path / f"{file_title}.md"
        markdown_path.write_text(
            f"# {product_name}\n\n"
            f"本文档唯一描述的商品是 {product_name}，"
            "用于测试知识库完整导入流程。\n\n"
            "## 技术参数\n\n额定工作电压为 24V。",
            encoding="utf-8",
        )

        config = ImportConfig.from_env()
        config.item_name_collection = item_collection
        config.chunks_collection = chunks_collection
        config.max_content_length = 1000
        config.min_content_length = 0
        config.embedding_batch_size = 2

        project_root = Path(__file__).resolve().parents[1]
        backup_path = project_root / "output" / f"{file_title}_chunks.json"
        milvus_client = None

        try:
            state = create_default_state(import_file_path=str(markdown_path))
            result = ImportWorkflow(config).run(state)

            assert result["is_md_read_enabled"] is True
            assert result["is_pdf_read_enabled"] is False
            assert result["file_title"] == file_title
            assert result["item_name"]
            assert len(result["chunks"]) == 2
            assert all(
                chunk["metadata"]["item_name"] == result["item_name"]
                for chunk in result["chunks"]
            )
            assert all(
                len(chunk["dense_vector"]) == config.embedding_dim
                for chunk in result["chunks"]
            )
            assert all(chunk["sparse_vector"] for chunk in result["chunks"])
            assert all(
                isinstance(chunk["chunk_id"], int)
                for chunk in result["chunks"]
            )
            assert backup_path.is_file()

            milvus_client = MilvusClient(uri=config.milvus_url)
            milvus_client.flush(collection_name=chunks_collection)
            stored_chunks = milvus_client.query(
                collection_name=chunks_collection,
                filter=(
                    "file_title == "
                    f"{json.dumps(file_title, ensure_ascii=False)}"
                ),
                output_fields=["chunk_id", "item_name", "chunk_order"],
                limit=100,
            )
            assert len(stored_chunks) == len(result["chunks"])
            assert {row["chunk_id"] for row in stored_chunks} == {
                chunk["chunk_id"] for chunk in result["chunks"]
            }
            assert all(
                row["item_name"] == result["item_name"]
                for row in stored_chunks
            )

            milvus_client.flush(collection_name=item_collection)
            stored_items = milvus_client.query(
                collection_name=item_collection,
                filter=(
                    "item_name == "
                    f"{json.dumps(result['item_name'], ensure_ascii=False)}"
                ),
                output_fields=["item_name", "file_title"],
                limit=10,
            )
            assert len(stored_items) == 1
            assert stored_items[0]["file_title"] == file_title
        finally:
            backup_path.unlink(missing_ok=True)
            if milvus_client is None and config.milvus_url:
                try:
                    milvus_client = MilvusClient(uri=config.milvus_url)
                except Exception:
                    milvus_client = None
            if milvus_client is not None:
                for collection_name in (chunks_collection, item_collection):
                    if milvus_client.has_collection(collection_name):
                        milvus_client.drop_collection(collection_name)
