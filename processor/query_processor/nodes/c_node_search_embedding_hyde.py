"""C：HyDE 假设性文档检索节点。"""

from __future__ import annotations

from typing import Any

from processor.query_processor.config import QueryConfig
from processor.query_processor.nodes.b_node_search_embedding import (
    NodeSearchEmbedding,
)
from processor.query_processor.state import QueryGraphState


class NodeSearchEmbeddingHyde(NodeSearchEmbedding):
    """先生成假设性答案，再使用答案执行混合向量检索。"""

    name = "node_search_embedding_hyde"

    def __init__(
        self,
        config: QueryConfig | None = None,
        *,
        llm_client=None,
        embedding_tool=None,
        milvus_client=None,
    ) -> None:
        super().__init__(
            config,
            embedding_tool=embedding_tool,
            milvus_client=milvus_client,
        )
        self._llm_client = llm_client

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】执行 HyDE 检索", self.name)
        rewritten_query, item_names = self._get_query_context(state)
        hyde_doc = self._generate_hypothetical_document(
            rewritten_query,
            item_names,
        )
        combined_text = f"{rewritten_query}\n{hyde_doc}"
        chunks = self._search_chunks(
            combined_text,
            item_names,
            source="hyde_embedding",
        )
        return {
            "hyde_embedding_chunks": chunks,
            "hyde_doc": hyde_doc,
        }

    def _generate_hypothetical_document(
        self,
        rewritten_query: str,
        item_names: list[str],
    ) -> str:
        client = self._llm_client
        if client is None:
            from utils.llm_utils import get_llm_client

            client = get_llm_client(model=self.config.hyde_model)

        item_context = "、".join(item_names) or "未限定商品"
        prompt = (
            "请根据用户问题生成一段可能出现在产品说明书、操作手册或知识库中的"
            "假设性答案文档，用于向量检索。请直接描述解决方法、操作步骤、"
            "参数和注意事项，不要回答你无法确定的信息，也不要说明这是"
            "假设性文档。\n\n"
            f"商品范围：{item_context}\n"
            f"用户问题：{rewritten_query}"
        )
        response = client.invoke(prompt)
        hyde_doc = self._response_to_text(response)
        if not hyde_doc:
            raise RuntimeError("HyDE 模型未返回假设性文档")
        return hyde_doc

    @staticmethod
    def _response_to_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_blocks = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "\n".join(text_blocks).strip()
        return str(content).strip()
