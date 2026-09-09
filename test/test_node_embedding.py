from processor.import_processor.config import ImportConfig
from processor.import_processor.nodes.f_node_embedding import NodeEmbedding


class FakeEmbeddingTool:
    def __init__(self):
        self.calls = []

    def embed_dense_and_sparse(self, texts):
        self.calls.append(texts)
        dense = [[float(index), 0.2, 0.3] for index, _ in enumerate(texts, 1)]
        sparse = [{index: 0.8} for index, _ in enumerate(texts, 1)]
        return dense, sparse


def make_chunk(order: int):
    return {
        "file_title": "HAK180 产品手册",
        "title": f"章节 {order}",
        "content": f"正文 {order}",
        "order": order,
        "metadata": {"item_name": "HAK180 安全栅"},
    }


def test_embedding_processes_chunks_in_batches_and_updates_state():
    tool = FakeEmbeddingTool()
    node = NodeEmbedding(
        ImportConfig(embedding_dim=3, embedding_batch_size=2),
        embedding_tool=tool,
    )
    source_chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    state = {"item_name": "HAK180 安全栅", "chunks": source_chunks}

    result = node.process(state)

    assert result is state
    assert tool.calls == [
        ["HAK180 安全栅\n正文 1", "HAK180 安全栅\n正文 2"],
        ["HAK180 安全栅\n正文 3"],
    ]
    assert result["chunks"][0]["dense_vector"] == [1.0, 0.2, 0.3]
    assert result["chunks"][0]["sparse_vector"] == {1: 0.8}
    assert "dense_vector" not in source_chunks[0]


def test_embedding_uses_state_item_name_when_metadata_has_none():
    chunk = make_chunk(1)
    chunk["metadata"] = {}

    text = NodeEmbedding._build_embedding_text(chunk, "HAK180 安全栅")

    assert text == "HAK180 安全栅\n正文 1"
