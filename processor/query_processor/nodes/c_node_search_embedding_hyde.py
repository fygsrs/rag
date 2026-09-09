"""C：HyDE 假设性文档检索节点。"""

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class NodeSearchEmbeddingHyde(NodeBase):
    """先生成假设性答案，再使用答案执行混合向量检索。"""

    name = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行 HyDE 检索", self.name)

        # TODO: 调用 LLM 生成假设性答案并对其向量化。
        # TODO: 限定 item_names 在 Milvus 中执行混合检索。
        return {"hyde_embedding_chunks": []}
