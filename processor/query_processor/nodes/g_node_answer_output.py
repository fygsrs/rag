"""G：答案生成节点。"""

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState


class NodeAnswerOutput(NodeBase):
    """组装提示词并生成最终答案。"""

    name = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        if state.get("answer"):
            self.logger.info("直接返回已有回答 | source=item_name_confirm")
            return {"answer": state["answer"]}

        # TODO: 根据 rewritten_query、history 和 reranked_docs 构建 Prompt。
        # TODO: 调用 LLM，按 is_stream 决定流式或非流式输出。
        self.logger.warning(
            "最终答案生成尚未实现 | reranked_docs=%d",
            len(state.get("reranked_docs") or []),
        )
        return {
            "prompt": state.get("prompt") or "",
            "answer": "",
        }
