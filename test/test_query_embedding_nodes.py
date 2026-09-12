from copy import deepcopy
from types import SimpleNamespace

import pytest

from processor.query_processor.nodes.b_node_search_embedding import (
    NodeSearchEmbedding,
)
from processor.query_processor.nodes.c_node_search_embedding_hyde import (
    NodeSearchEmbeddingHyde,
)


class FakeEmbeddingTool:
    def __init__(self):
        self.calls = []

    def embed_dense_and_sparse(self, texts, *, text_type="document"):
        self.calls.append((list(texts), text_type))
        return [[0.1, 0.2, 0.3]], [{1: 0.8}]


class FakeMilvusClient:
    def __init__(self, results=None, *, collection_exists=True):
        self.results = deepcopy(results or [[]])
        self.collection_exists = collection_exists
        self.search_kwargs = None

    def has_collection(self, collection_name):
        return self.collection_exists

    def hybrid_search(self, **kwargs):
        self.search_kwargs = kwargs
        return deepcopy(self.results)


class FakeLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def make_config():
    return SimpleNamespace(
        milvus_url="http://milvus.test:19530",
        chunks_collection="kb_chunks_test",
        search_top_k=5,
        search_dense_weight=0.8,
        search_sparse_weight=0.2,
        hyde_model="query-model",
    )


def make_hit():
    return {
        "id": 101,
        "distance": 0.91,
        "entity": {
            "chunk_id": 101,
            "file_title": "HAK180 产品手册",
            "title": "转印设置",
            "content": "进入打印设置并调整转印温度。",
            "chunk_order": 3,
            "item_name": "HAK180 安全栅",
            "metadata": {"page": 8},
        },
    }


def test_embedding_node_runs_filtered_dense_sparse_search():
    embedding = FakeEmbeddingTool()
    milvus = FakeMilvusClient([[make_hit()]])
    node = NodeSearchEmbedding(
        make_config(),
        embedding_tool=embedding,
        milvus_client=milvus,
    )

    result = node.process(
        {
            "rewritten_query": "HAK180 安全栅如何调整转印温度？",
            "item_names": ["HAK180 安全栅"],
        }
    )

    assert embedding.calls == [(["HAK180 安全栅如何调整转印温度？"], "query")]
    assert milvus.search_kwargs["collection_name"] == "kb_chunks_test"
    assert milvus.search_kwargs["limit"] == 5
    assert len(milvus.search_kwargs["reqs"]) == 2
    assert milvus.search_kwargs["reqs"][0]._expr == (
        'item_name in ["HAK180 安全栅"]'
    )
    assert result["embedding_chunks"] == [
        {
            "id": 101,
            "title": "转印设置",
            "content": "进入打印设置并调整转印温度。",
            "score": 0.91,
            "source": "embedding",
            "metadata": {
                "page": 8,
                "file_title": "HAK180 产品手册",
                "chunk_order": 3,
                "item_name": "HAK180 安全栅",
            },
        }
    ]


def test_embedding_node_omits_filter_without_item_names():
    milvus = FakeMilvusClient()
    node = NodeSearchEmbedding(
        make_config(),
        embedding_tool=FakeEmbeddingTool(),
        milvus_client=milvus,
    )

    result = node.process({"rewritten_query": "如何调整转印温度？"})

    assert result == {"embedding_chunks": []}
    assert milvus.search_kwargs["reqs"][0]._expr is None


def test_hyde_node_generates_document_and_searches_combined_text():
    embedding = FakeEmbeddingTool()
    llm = FakeLLM("先打开控制面板，再进入转印温度设置。")
    milvus = FakeMilvusClient([[make_hit()]])
    node = NodeSearchEmbeddingHyde(
        make_config(),
        llm_client=llm,
        embedding_tool=embedding,
        milvus_client=milvus,
    )

    result = node.process(
        {
            "rewritten_query": "HAK180 安全栅如何调整转印温度？",
            "item_names": ["HAK180 安全栅"],
        }
    )

    assert result["hyde_doc"] == "先打开控制面板，再进入转印温度设置。"
    assert result["hyde_embedding_chunks"][0]["source"] == "hyde_embedding"
    assert embedding.calls == [
        (
            [
                "HAK180 安全栅如何调整转印温度？\n"
                "先打开控制面板，再进入转印温度设置。"
            ],
            "query",
        )
    ]
    assert milvus.search_kwargs["limit"] == 5


def test_hyde_node_propagates_llm_error():
    expected_error = ConnectionError("LLM unavailable")
    node = NodeSearchEmbeddingHyde(
        make_config(),
        llm_client=FakeLLM(error=expected_error),
        embedding_tool=FakeEmbeddingTool(),
        milvus_client=FakeMilvusClient(),
    )

    with pytest.raises(ConnectionError) as raised:
        node.process({"rewritten_query": "问题", "item_names": []})

    assert raised.value is expected_error
