"""B：普通混合向量检索节点。"""

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class NodeSearchEmbedding(NodeBase):
    """根据商品名和改写后的问题检索知识切片。"""

    name = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行稠密/稀疏混合检索", self.name)

        # TODO: 向量化 rewritten_query，并限定 item_names 搜索 Milvus。
        return {"embedding_chunks": []}
