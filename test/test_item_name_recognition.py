from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from processor.import_processor.config import ImportConfig
from processor.import_processor.exceptions import LLMError, MilvusError
from processor.import_processor.nodes.e_node_item_name_recognition import (
    NodeItemNameRecognition,
)


class FakeEmbeddingTool:
    def embed_dense_and_sparse(self, texts):
        assert texts == ["HAK180 安全栅"]
        return [[0.1, 0.2, 0.3]], [{10: 0.8, 20: 0.4}]


def make_node(**kwargs):
    config = ImportConfig(
        embedding_dim=3,
        item_name_chunk_k=2,
        item_name_chunk_size=20,
        item_name_collection="kb_item_names",
    )
    return NodeItemNameRecognition(config, **kwargs)


def make_state():
    return {
        "file_title": "HAK180 产品安全手册",
        "chunks": [
            {
                "file_title": "HAK180 产品安全手册",
                "title": "产品介绍",
                "content": "HAK180 安全栅用于工业现场。",
                "order": 1,
                "metadata": {"source": "manual"},
            },
            {
                "file_title": "HAK180 产品安全手册",
                "title": "技术参数",
                "content": "额定工作电压为 24V。",
                "order": 2,
                "metadata": {},
            },
        ],
    }


def test_process_runs_all_steps_and_upserts_item():
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"item_name": "HAK180 安全栅"}'
    )
    milvus = Mock()
    milvus.has_collection.return_value = True
    milvus.query.return_value = []
    node = make_node(
        llm_client=llm,
        embedding_tool=FakeEmbeddingTool(),
        milvus_client=milvus,
    )
    state = make_state()

    result = node.process(state)

    assert result is state
    assert state["item_name"] == "HAK180 安全栅"
    assert all(
        chunk["metadata"]["item_name"] == "HAK180 安全栅"
        for chunk in state["chunks"]
    )
    assert state["chunks"][0]["metadata"]["source"] == "manual"
    row = milvus.upsert.call_args.kwargs["data"]
    assert row["item_name"] == "HAK180 安全栅"
    assert row["dense_vector"] == [0.1, 0.2, 0.3]
    assert row["sparse_vector"] == {10: 0.8, 20: 0.4}
    assert len(row["id"]) == 64


def test_build_context_uses_configured_chunk_count_and_size():
    node = make_node()
    context = node._step_2_build_context(make_state()["chunks"])

    assert "[章节：产品介绍]" in context
    assert "[章节：技术参数]" in context


def test_parse_item_name_accepts_json_code_fence():
    node = make_node()

    assert (
        node._parse_item_name('```json\n{"item_name": "HAK180 安全栅"}\n```')
        == "HAK180 安全栅"
    )


def test_parse_item_name_rejects_plain_text():
    node = make_node()

    with pytest.raises(LLMError, match="有效 JSON"):
        node._parse_item_name("HAK180 安全栅")


def test_collection_rejects_a_different_embedding_model():
    node = make_node()
    milvus = Mock()
    milvus.query.return_value = [
        {"embedding_provider": "aliyun", "embedding_model": "text-embedding-v4"}
    ]

    with pytest.raises(MilvusError, match="不能写入"):
        node._validate_collection_embedding(
            milvus,
            "kb_item_names",
            "bge",
            "BAAI/bge-m3",
        )
