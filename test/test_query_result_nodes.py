from copy import deepcopy
from types import SimpleNamespace

from processor.query_processor.nodes.d_node_web_search_mcp import (
    NodeWebSearchMcp,
)
from processor.query_processor.nodes.e_node_rrf import NodeRrf
from processor.query_processor.nodes.f_node_rerank import NodeRerank


def make_config(**overrides):
    values = {
        "web_search_mcp_url": "https://example.test/mcp",
        "dashscope_api_key": "test-key",
        "web_search_tool": "bailian_web_search",
        "web_search_top_k": 5,
        "web_search_timeout": 10.0,
        "rrf_k": 60,
        "rrf_embedding_weight": 1.0,
        "rrf_hyde_weight": 0.7,
        "rrf_max_results": 10,
        "rerank_url": "https://workspace.test/text-rerank",
        "rerank_model": "qwen3.7-text-rerank",
        "rerank_timeout": 30.0,
        "rerank_min_results": 3,
        "rerank_max_results": 10,
        "rerank_absolute_gap": 0.5,
        "rerank_relative_gap": 0.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def successful_mcp_call(query, count):
    assert query == "HAK180 最近是否有固件更新？"
    assert count == 5
    return {
        "pages": [
            {
                "title": "HAK180 固件说明",
                "url": "https://example.com/hak180",
                "snippet": "最新固件修复了通信问题。",
            },
            {"title": "无摘要结果", "url": "https://example.com/empty"},
        ]
    }


def test_web_search_converts_mcp_pages_to_documents():
    node = NodeWebSearchMcp(make_config(), mcp_caller=successful_mcp_call)

    result = node.process({"rewritten_query": "HAK180 最近是否有固件更新？"})

    assert result["web_search_docs"] == [
        {
            "id": "https://example.com/hak180",
            "title": "HAK180 固件说明",
            "content": "最新固件修复了通信问题。",
            "url": "https://example.com/hak180",
            "source": "web_search",
            "metadata": {},
        }
    ]


def test_web_search_failure_degrades_to_empty_results():
    async def failing_call(query, count):
        raise TimeoutError("MCP timeout")

    node = NodeWebSearchMcp(make_config(), mcp_caller=failing_call)

    assert node.process({"rewritten_query": "问题"}) == {"web_search_docs": []}


def make_doc(doc_id, content, *, source="embedding"):
    return {
        "id": doc_id,
        "title": f"文档 {doc_id}",
        "content": content,
        "source": source,
        "metadata": {},
    }


def test_rrf_fuses_and_deduplicates_local_search_results():
    node = NodeRrf(make_config())
    first = make_doc(1, "共同命中文档")
    dense_only = make_doc(2, "普通检索文档")
    hyde_only = make_doc(3, "HyDE 检索文档", source="hyde_embedding")

    result = node.process(
        {
            "embedding_chunks": [first, dense_only],
            "hyde_embedding_chunks": [deepcopy(first), hyde_only],
        }
    )

    assert [doc["id"] for doc in result["rrf_chunks"]] == [1, 2, 3]
    assert all("rrf_score" not in doc for doc in result["rrf_chunks"])


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return deepcopy(self.payload)


class FakeHttpClient:
    def __init__(self, scores):
        self.scores = scores
        self.call = None

    def post(self, url, *, headers, json, timeout):
        self.call = {
            "url": url,
            "headers": headers,
            "json": deepcopy(json),
            "timeout": timeout,
        }
        results = [
            {"index": index, "relevance_score": score}
            for index, score in enumerate(self.scores)
        ]
        return FakeResponse({"output": {"results": results}})


def test_rerank_merges_web_docs_and_applies_cliff_cutoff():
    http_client = FakeHttpClient([0.95, 0.85, 0.76, 0.20, 0.10])
    node = NodeRerank(make_config(), http_client=http_client)
    local_docs = [make_doc(index, f"本地内容 {index}") for index in range(1, 5)]
    web_doc = make_doc(
        "https://example.com/latest",
        "网络内容",
        source="web_search",
    )

    result = node.process(
        {
            "rewritten_query": "HAK180 如何升级？",
            "rrf_chunks": local_docs,
            "web_search_docs": [web_doc],
        }
    )

    assert [doc["score"] for doc in result["reranked_docs"]] == [
        0.95,
        0.85,
        0.76,
    ]
    request = http_client.call
    assert request["json"]["model"] == "qwen3.7-text-rerank"
    assert request["json"]["input"]["query"] == "HAK180 如何升级？"
    assert len(request["json"]["input"]["documents"]) == 5


def test_rerank_keeps_at_most_ten_results_without_cliff():
    scores = [1 - index * 0.03 for index in range(12)]
    node = NodeRerank(make_config(), http_client=FakeHttpClient(scores))
    documents = [make_doc(index, f"内容 {index}") for index in range(12)]

    result = node.process(
        {
            "rewritten_query": "问题",
            "rrf_chunks": documents,
            "web_search_docs": [],
        }
    )

    assert len(result["reranked_docs"]) == 10
