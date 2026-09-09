"""知识库查询工作流编排。"""

from langgraph.graph import END, StateGraph

from processor.query_processor.nodes.a_node_item_name_confirm import (
    NodeItemNameConfirm,
)
from processor.query_processor.nodes.b_node_search_embedding import (
    NodeSearchEmbedding,
)
from processor.query_processor.nodes.c_node_search_embedding_hyde import (
    NodeSearchEmbeddingHyde,
)
from processor.query_processor.nodes.d_node_web_search_mcp import (
    NodeWebSearchMcp,
)
from processor.query_processor.nodes.e_node_rrf import NodeRrf
from processor.query_processor.nodes.f_node_rerank import NodeRerank
from processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput
from processor.query_processor.state import QueryGraphState


class KBQueryWorkflow:
    """组织主体确认、多路召回、融合、精排和答案生成。"""

    def __init__(self) -> None:
        self._compiled_graph = None

    @staticmethod
    def _route_after_item_name_confirm(state: QueryGraphState) -> str:
        """已有反问或拒答内容时跳过检索，否则进入多路召回。"""
        if state.get("answer"):
            return "node_answer_output"
        return "node_multi_search"

    @staticmethod
    def _empty_update(_: QueryGraphState) -> QueryGraphState:
        """分叉和汇合节点不修改状态。"""
        return {}

    def build_graph(self):
        graph = StateGraph(QueryGraphState)
        graph.add_node("node_item_name_confirm", NodeItemNameConfirm())
        graph.add_node("node_multi_search", self._empty_update)
        graph.add_node("node_search_embedding", NodeSearchEmbedding())
        graph.add_node(
            "node_search_embedding_hyde",
            NodeSearchEmbeddingHyde(),
        )
        graph.add_node("node_web_search_mcp", NodeWebSearchMcp())
        graph.add_node("node_rrf", NodeRrf())
        graph.add_node("node_rerank", NodeRerank())
        graph.add_node("node_answer_output", NodeAnswerOutput())

        graph.set_entry_point("node_item_name_confirm")
        graph.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_multi_search": "node_multi_search",
            },
        )

        graph.add_edge("node_multi_search", "node_search_embedding")
        graph.add_edge("node_multi_search", "node_search_embedding_hyde")
        graph.add_edge("node_multi_search", "node_web_search_mcp")

        graph.add_edge(
            [
                "node_search_embedding",
                "node_search_embedding_hyde",
                "node_web_search_mcp",
            ],
            "node_rrf",
        )
        graph.add_edge("node_rrf", "node_rerank")
        graph.add_edge("node_rerank", "node_answer_output")
        graph.add_edge("node_answer_output", END)
        return graph.compile()

    @property
    def graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def run(self, state: QueryGraphState, stream: bool = False):
        """运行查询工作流；stream=True 时返回状态事件迭代器。"""
        if stream:
            return self.graph.stream(state, stream_mode="values")
        return self.graph.invoke(state)


if __name__ == "__main__":
    from processor.query_processor.logger import setup_logging
    from processor.query_processor.state import create_default_query_state

    setup_logging()
    initial_state = create_default_query_state(
        original_query="如何设置设备的转印温度？",
    )
    for event in KBQueryWorkflow().run(initial_state, stream=True):
        print(event)
