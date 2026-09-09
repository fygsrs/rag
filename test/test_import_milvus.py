from unittest.mock import Mock

import pytest

from processor.import_processor.config import ImportConfig
from processor.import_processor.exceptions import MilvusError
from processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus


def make_chunk(order: int):
    return {
        "file_title": "HAK180 产品手册",
        "title": f"章节 {order}",
        "content": f"正文 {order}",
        "order": order,
        "metadata": {"item_name": "HAK180 安全栅"},
        "dense_vector": [0.1, 0.2, 0.3],
        "sparse_vector": {10 + order: 0.8},
    }


def collection_description():
    names = [
        "chunk_id",
        "file_title",
        "title",
        "content",
        "chunk_order",
        "item_name",
        "metadata",
        "sparse_vector",
        "embedding_provider",
        "embedding_model",
        "created_at",
    ]
    fields = [{"name": name, "params": {}} for name in names]
    fields.append({"name": "dense_vector", "params": {"dim": 3}})
    return {"fields": fields}


def test_import_process_cleans_inserts_and_backfills_ids():
    client = Mock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = collection_description()
    client.query.return_value = []
    client.delete.return_value = {"delete_count": 2}
    client.insert.return_value = {"ids": [101, 102]}
    node = NodeImportMilvus(
        ImportConfig(
            embedding_dim=3,
            chunks_collection="kb_chunks_test",
            milvus_url="http://milvus.test:19530",
        ),
        milvus_client=client,
    )
    state = {
        "item_name": "HAK180 安全栅",
        "chunks": [make_chunk(1), make_chunk(2)],
    }

    result = node.process(state)

    assert result is state
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == [101, 102]
    assert client.delete.call_args.kwargs["filter"] == (
        'file_title == "HAK180 产品手册"'
    )
    rows = client.insert.call_args.kwargs["data"]
    assert rows[0]["chunk_order"] == 1
    assert rows[0]["item_name"] == "HAK180 安全栅"
    assert rows[0]["dense_vector"] == [0.1, 0.2, 0.3]
    assert rows[0]["sparse_vector"] == {11: 0.8}


def test_import_rejects_chunk_without_dense_vector():
    chunk = make_chunk(1)
    del chunk["dense_vector"]
    node = NodeImportMilvus(
        ImportConfig(embedding_dim=3, chunks_collection="kb_chunks_test")
    )

    with pytest.raises(MilvusError, match="向量维度"):
        node._step_1_check_input(
            {"item_name": "HAK180 安全栅", "chunks": [chunk]}
        )
