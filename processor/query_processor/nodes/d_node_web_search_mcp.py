"""D：联网搜索节点。"""

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class NodeWebSearchMcp(NodeBase):
    """通过百炼 MCP 联网搜索补充实时资料。"""

    name = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行联网搜索", self.name)

        # TODO: 调用百炼 MCP，并转换为 RetrievedDocument 结构。
        return {"web_search_docs": []}
