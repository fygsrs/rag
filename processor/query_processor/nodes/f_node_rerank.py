"""F：精排节点。"""

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class NodeRerank(NodeBase):
    """使用 Rerank 模型对 RRF 结果进行二次排序。"""

    name = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行精排", self.name)

        # TODO: 调用 Rerank 模型并截取最终 Top-K。
        return {"reranked_docs": list(state.get("rrf_chunks") or [])}
