"""G：答案生成节点。"""

from __future__ import annotations

import json
from typing import Any, Callable

from processor.query_processor.base import NodeBase
from processor.query_processor.config import QueryConfig
from processor.query_processor.state import QueryGraphState
from utils.llm_utils import get_llm_client
from utils.mongodb_utils import get_mongodb_util
from utils.text_utils import normalize_markdown_images, summarize_content

StreamWriter = Callable[[dict[str, Any]], None]


class NodeAnswerOutput(NodeBase):
    """根据精排证据生成答案，并按需发送流式事件。"""

    name = "node_answer_output"
    no_evidence_answer = "未找到足够的相关资料"

    def __init__(
        self,
        config: QueryConfig | None = None,
        *,
        llm_client=None,
        mongodb_util=None,
        stream_writer: StreamWriter | None = None,
    ) -> None:
        super().__init__()
        self.config = config or QueryConfig()
        self._llm_client = llm_client
        self._mongodb_util = mongodb_util
        self._stream_writer = stream_writer

    def process(self, state: QueryGraphState) -> QueryGraphState:
        is_stream = self._validate_stream_mode(state.get("is_stream", False))
        existing_answer = state.get("answer")
        if isinstance(existing_answer, str) and existing_answer.strip():
            answer = existing_answer.strip()
            self.logger.info("直接返回已有回答 | persisted_by=previous_node")
            self._emit_complete_answer(answer, [], is_stream)
            return {"answer": answer}

        session_id = self._required_text(state, "session_id")
        message_id = self._required_text(state, "message_id")
        documents = self._get_documents(state)

        if not documents:
            answer = self.no_evidence_answer
            history = self._save_answer(
                state=state,
                session_id=session_id,
                message_id=message_id,
                answer=answer,
                sources=[],
                answer_type="no_evidence",
            )
            self.logger.info("无可用证据，返回固定回答")
            self._emit_complete_answer(answer, [], is_stream)
            return {"prompt": "", "answer": answer, "history": history}

        prompt = self._build_prompt(state, documents)
        sources = self._build_sources(documents)
        answer = normalize_markdown_images(
            self._generate_answer(prompt, is_stream)
        ).strip()
        if not answer:
            raise RuntimeError("大模型未返回答案")

        history = self._save_answer(
            state=state,
            session_id=session_id,
            message_id=message_id,
            answer=answer,
            sources=sources,
            answer_type="generated",
        )
        self._emit_final(answer, sources, is_stream)
        self.logger.info(
            "答案生成完成 | evidence=%d | answer_chars=%d | stream=%s",
            len(documents),
            len(answer),
            is_stream,
        )
        return {"prompt": prompt, "answer": answer, "history": history}

    @staticmethod
    def _validate_stream_mode(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("is_stream 必须是布尔值")
        return value

    @staticmethod
    def _required_text(state: QueryGraphState, field_name: str) -> str:
        value = state.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 必须是非空字符串")
        return value.strip()

    @staticmethod
    def _get_documents(state: QueryGraphState) -> list[dict[str, Any]]:
        documents = state.get("reranked_docs") or []
        if not isinstance(documents, list):
            raise ValueError("reranked_docs 必须是文档列表")

        valid_documents = []
        for document in documents:
            if not isinstance(document, dict):
                raise ValueError("reranked_docs 中包含无效文档")
            content = document.get("content")
            if isinstance(content, str) and content.strip():
                valid_documents.append(document)
        return valid_documents

    def _build_prompt(
        self,
        state: QueryGraphState,
        documents: list[dict[str, Any]],
    ) -> str:
        original_query = self._required_text(state, "original_query")
        rewritten_query = state.get("rewritten_query")
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            rewritten_query = original_query

        item_names = state.get("item_names") or []
        if not isinstance(item_names, list):
            raise ValueError("item_names 必须是字符串列表")
        normalized_names = [
            name.strip()
            for name in item_names
            if isinstance(name, str) and name.strip()
        ]

        history = state.get("history") or []
        if not isinstance(history, list):
            raise ValueError("history 必须是消息列表")
        history_messages = max(1, int(self.config.answer_history_messages))
        history_chars = max(1, int(self.config.answer_history_chars))
        history_lines = []
        for memory in history[-history_messages:]:
            if not isinstance(memory, dict):
                continue
            role = str(memory.get("role") or "unknown")
            content = str(memory.get("content") or "").strip()
            if content:
                history_lines.append(f"{role}: {content[:history_chars]}")

        evidence_blocks = []
        for index, document in enumerate(documents, start=1):
            title = str(document.get("title") or "未命名资料").strip()
            source = str(document.get("source") or "internal").strip()
            url = str(document.get("url") or "").strip()
            content = normalize_markdown_images(
                str(document["content"])
            ).strip()
            evidence_blocks.append(
                f"[{index}] 标题：{title}\n"
                f"来源：{source}\n"
                f"链接：{url or '无'}\n"
                f"内容：{content}"
            )

        return (
            "你是企业知识库问答助手。请严格依据给定资料回答，不得编造。\n"
            "要求：\n"
            "1. 使用清晰、直接的中文 Markdown。\n"
            "2. 每个关键事实在句末使用 [1]、[2] 形式标注资料编号。\n"
            "3. 如果资料冲突，明确指出冲突；如果资料不足，明确说明无法确定。\n"
            "4. 参考资料中的图片已经统一为 ![说明](URL) 格式。只保留与问题和具体操作直接相关的图片。\n"
            "5  保持图片与操作的对应关系。（没有图片不需要在答案后面解释）\n"
            "6. 不得修改、缩短、转义或重新包装图片 URL，不得把图片写成 HTML <img>，也不得把 URL 改成 [URL](URL)。\n"
            "\n\n"
            f"用户原问题：{original_query}\n"
            f"改写后问题：{rewritten_query.strip()}\n"
            f"已确认商品：{', '.join(normalized_names) or '无'}\n\n"
            "最近对话：\n"
            f"{chr(10).join(history_lines) or '无'}\n\n"
            "参考资料：\n"
            f"{chr(10).join(evidence_blocks)}\n\n"
            "请给出最终答案："
        )

    def _generate_answer(self, prompt: str, is_stream: bool) -> str:
        client = self._llm_client or get_llm_client(
            model=self.config.answer_model or None
        )
        if not is_stream:
            response = client.invoke(prompt)
            return self._response_to_text(response.content).strip()

        writer = self._require_stream_writer()
        chunks = []
        for response_chunk in client.stream(prompt):
            content = self._response_to_text(response_chunk.content)
            if not content:
                continue
            chunks.append(content)
            writer({"event": "answer", "data": {"content": content}})
        return "".join(chunks).strip()

    @staticmethod
    def _response_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)
        return str(content or "")

    @staticmethod
    def _build_sources(
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        sources = []
        for index, document in enumerate(documents, start=1):
            source = str(document.get("source") or "internal")
            summary = summarize_content(str(document.get("content") or ""))
            sources.append(
                {
                    "index": index,
                    "id": document.get("id"),
                    "title": str(document.get("title") or ""),
                    "url": str(document.get("url") or ""),
                    "score": document.get("score"),
                    "source": source,
                    "source_type": (
                        "web" if source == "web_search" else "internal"
                    ),
                    "summary": summary,
                    "metadata": document.get("metadata") or {},
                }
            )
        return sources

    def _save_answer(
        self,
        *,
        state: QueryGraphState,
        session_id: str,
        message_id: str,
        answer: str,
        sources: list[dict[str, Any]],
        answer_type: str,
    ) -> list[dict[str, Any]]:
        mongodb_util = self._mongodb_util or get_mongodb_util()
        mongodb_util.save_memory(
            session_id=session_id,
            message_id=f"{message_id}:answer",
            role="assistant",
            content=answer,
            metadata={
                "source_node": self.name,
                "answer_type": answer_type,
                "task_id": str(state.get("task_id") or ""),
                "item_names": list(state.get("item_names") or []),
                "sources": json.loads(json.dumps(sources, default=str)),
            },
        )
        return mongodb_util.get_memories(
            session_id,
            limit=int(self.config.history_limit),
        )

    def _require_stream_writer(self) -> StreamWriter:
        if self._stream_writer is None:
            raise RuntimeError("is_stream=True 时必须提供 stream_writer")
        return self._stream_writer

    def _emit_complete_answer(
        self,
        answer: str,
        sources: list[dict[str, Any]],
        is_stream: bool,
    ) -> None:
        if not is_stream:
            return
        writer = self._require_stream_writer()
        writer({"event": "answer", "data": {"content": answer}})
        writer({"event": "final", "data": {"answer": answer, "sources": sources}})

    def _emit_final(
        self,
        answer: str,
        sources: list[dict[str, Any]],
        is_stream: bool,
    ) -> None:
        if is_stream:
            self._require_stream_writer()(
                {"event": "final", "data": {"answer": answer, "sources": sources}}
            )
