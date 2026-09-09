from types import SimpleNamespace
from unittest.mock import Mock

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
