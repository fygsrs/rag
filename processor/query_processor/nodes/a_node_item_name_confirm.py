"""A：商品主体确认节点。"""

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class NodeItemNameConfirm(NodeBase):
    """提取并确认商品名称，同时改写用户问题。"""

    name = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】确认商品主体并改写问题", self.name)

        # TODO: 结合 history 调用 LLM 提取商品名称并改写问题。
        # TODO: 在商品 Collection 中执行混合检索并确认标准商品名称。
        # TODO: 无法唯一确认时写入 answer，使工作流直接进入答案节点。
        original_query = str(state.get("original_query") or "").strip()
        return {
            "rewritten_query": state.get("rewritten_query") or original_query,
            "item_names": list(state.get("item_names") or []),
        }
