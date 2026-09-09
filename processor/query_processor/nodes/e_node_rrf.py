"""E：RRF 多路结果融合节点。"""

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class NodeRrf(NodeBase):
    """融合普通检索、HyDE 检索和联网搜索结果。"""

    name = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行 RRF 融合排序", self.name)

        # TODO: 按文档唯一标识去重，并计算 Reciprocal Rank Fusion 分数。
        return {"rrf_chunks": []}
