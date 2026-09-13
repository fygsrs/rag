from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from utils.embedding_utils import EmbeddingTool


class FakeDenseVector(list):
    def tolist(self):
        return list(self)


def test_bge_dense_and_sparse_response_is_converted_for_milvus():
    config = SimpleNamespace(
        provider="bge",
        bge_m3_path="D:/models/bge-m3",
        bge_m3="BAAI/bge-m3",
        bge_device="cpu",
        bge_fp16=False,
        batch_size=2,
    )
    model = Mock()
    model.encode.return_value = {
        "dense_vecs": [FakeDenseVector([0.1, 0.2, 0.3])],
        "lexical_weights": [{"12": 0.9, "30": 0.4}],
    }
    tool = EmbeddingTool(config)
    tool._bge_model = model

    dense, sparse = tool.embed_dense_and_sparse(["测试商品"])

    assert dense == [[0.1, 0.2, 0.3]]
    assert sparse == [{12: 0.9, 30: 0.4}]
    assert model.encode.call_args.kwargs["return_sparse"] is True


def test_aliyun_dense_and_sparse_response_is_converted_for_milvus(monkeypatch):
    config = SimpleNamespace(
        provider="aliyun",
        dashscope_api_key="test-key",
        dashscope_native_url="https://example.test/embedding",
        dashscope_model="text-embedding-v4",
        dimension=3,
        request_timeout=10,
    )
    response = Mock()
    response.json.return_value = {
        "output": {
            "embeddings": [
                {
                    "text_index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                    "sparse_embedding": [
                        {"index": 12, "value": 0.9, "token": "商品"},
                        {"index": 30, "value": 0.4, "token": "型号"},
                    ],
                }
            ]
        }
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("utils.embedding_utils.httpx.post", post)

    dense, sparse = EmbeddingTool(config).embed_dense_and_sparse(["测试商品"])

    assert dense == [[0.1, 0.2, 0.3]]
    assert sparse == [{12: 0.9, 30: 0.4}]
    request = post.call_args.kwargs["json"]
    assert request["parameters"]["output_type"] == "dense&sparse"
    assert request["parameters"]["text_type"] == "document"


def make_aliyun_config(**overrides):
    values = {
        "provider": "aliyun",
        "dashscope_api_key": "test-key",
        "dashscope_native_url": "https://example.test/embedding",
        "dashscope_model": "text-embedding-v4",
        "dimension": 3,
        "request_timeout": 10,
        "max_retries": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_aliyun_payload():
    return {
        "output": {
            "embeddings": [
                {
                    "text_index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                    "sparse_embedding": [{"index": 12, "value": 0.9}],
                }
            ]
        }
    }


def test_aliyun_retries_transient_transport_error(monkeypatch):
    response = Mock()
    response.status_code = 200
    response.json.return_value = make_aliyun_payload()
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("temporary ssl eof")
        return response

    monkeypatch.setattr("utils.embedding_utils.httpx.post", fake_post)
    monkeypatch.setattr("utils.embedding_utils.time.sleep", lambda seconds: None)

    dense, sparse = EmbeddingTool(make_aliyun_config()).embed_dense_and_sparse(
        ["测试商品"]
    )

    assert calls["count"] == 2
    assert dense == [[0.1, 0.2, 0.3]]
    assert sparse == [{12: 0.9}]


def test_aliyun_gives_up_after_max_retries(monkeypatch):
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        raise httpx.ConnectError("temporary ssl eof")

    monkeypatch.setattr("utils.embedding_utils.httpx.post", fake_post)
    monkeypatch.setattr("utils.embedding_utils.time.sleep", lambda seconds: None)

    with pytest.raises(httpx.ConnectError):
        EmbeddingTool(make_aliyun_config(max_retries=1)).embed_dense_and_sparse(
            ["测试商品"]
        )

    assert calls["count"] == 2


def test_aliyun_retries_retryable_status_code(monkeypatch):
    unavailable = Mock()
    unavailable.status_code = 503
    ok_response = Mock()
    ok_response.status_code = 200
    ok_response.json.return_value = make_aliyun_payload()
    responses = [unavailable, ok_response]
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        response = responses[calls["count"]]
        calls["count"] += 1
        return response

    monkeypatch.setattr("utils.embedding_utils.httpx.post", fake_post)
    monkeypatch.setattr("utils.embedding_utils.time.sleep", lambda seconds: None)

    dense, _ = EmbeddingTool(make_aliyun_config()).embed_dense_and_sparse(
        ["测试商品"]
    )

    assert calls["count"] == 2
    assert dense == [[0.1, 0.2, 0.3]]
