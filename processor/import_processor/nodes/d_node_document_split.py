import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import DocumentSplitError, StateFieldError
from processor.import_processor.state import ChunkDict, ImportGraphState


class NodeDocumentSplit(BaseNode):
    """文档切分节点：优先按 Markdown 标题切分，再细化超长片段。"""

    name = "node_document_split"
    _TITLE_PATTERN = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)")
    _SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

    def process(self, state: ImportGraphState):
        self.logger.info("%s节点开始执行...", self.name)

        # 1. 参数处理
        content, file_title = self._step_1_get_inputs(state)

        # 2. 按标题初步切分
        sections, title_count, lines_count = self._step_2_split_by_title(
            content,
            file_title,
        )

        # 3. 无标题兜底
        sections = self._step_3_handle_no_title(
            content,
            sections,
            title_count,
            file_title,
        )

        # 4. 细化超长切片并合并过短切片
        sections = self._step_4_refine_chunks(sections)

        # 5. 打印统计信息
        self._step_5_print_stats(lines_count, sections)

        # 6. 写回状态
        self._step_6_backup(state, sections)
        return state

    def _step_1_get_inputs(
        self,
        state: ImportGraphState,
    ) -> tuple[str, str]:
        """获取 Markdown 内容和文件标题。"""
        content = state.get("md_content")
        file_title = state.get("file_title")

        if not isinstance(content, str) or not content.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="md_content",
                expected_type=str,
            )

        if not isinstance(file_title, str) or not file_title.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="file_title",
                expected_type=str,
            )

        content = content.replace("\r\n", "\n").replace("\r", "\n")
        return content.strip(), file_title.strip()

    def _step_2_split_by_title(
        self,
        content: str,
        file_title: str = "",
    ) -> tuple[list[ChunkDict], int, int]:
        """按 Markdown ATX 标题切分，忽略代码块中的 ``#``。"""
        lines = content.splitlines()
        sections: list[ChunkDict] = []
        current_lines: list[str] = []
        current_title = file_title
        title_count = 0
        fence_marker: str | None = None

        def flush() -> None:
            nonlocal current_lines
            text = "\n".join(current_lines).strip()
            if text:
                sections.append(self._make_chunk(file_title, current_title, text))
            current_lines = []

        for line in lines:
            stripped = line.lstrip()
            marker_match = re.match(r"(`{3,}|~{3,})", stripped)
            if marker_match:
                marker = marker_match.group(1)
                marker_char = marker[0]
                if fence_marker is None:
                    fence_marker = marker_char
                elif fence_marker == marker_char:
                    fence_marker = None

            is_title = fence_marker is None and bool(
                self._TITLE_PATTERN.match(line)
            )
            if is_title:
                title_count += 1
                flush()
                current_lines = [line.rstrip()]
                current_title = self._extract_title_text(line) or file_title
            else:
                current_lines.append(line.rstrip())

        flush()
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(
        self,
        content: str,
        sections: list[ChunkDict],
        title_count: int,
        file_title: str,
    ) -> list[ChunkDict]:
        """为无标题文档及首个标题前的正文补充文档标题。"""
        default_title = f"# {file_title}"
        if title_count == 0:
            return [
                self._make_chunk(
                    file_title,
                    file_title,
                    f"{default_title}\n\n{content.strip()}",
                )
            ]

        if sections and not self._TITLE_PATTERN.match(
            sections[0]["content"].splitlines()[0]
        ):
            first = sections[0]
            sections[0] = self._make_chunk(
                file_title,
                first["title"] or file_title,
                f"{default_title}\n\n{first['content']}",
                metadata=first.get("metadata"),
            )
        return sections

    def _step_4_refine_chunks(self, sections: list[ChunkDict]) -> list[ChunkDict]:
        """拆分超长片段，并在不超过上限的前提下合并短片段。"""
        max_length = self.config.max_content_length
        min_length = self.config.min_content_length
        if max_length <= 0:
            raise DocumentSplitError(
                message="max_content_length 必须大于 0",
                node_name=self.name,
            )
        if min_length < 0 or min_length > max_length:
            raise DocumentSplitError(
                message="min_content_length 必须在 0 到 max_content_length 之间",
                node_name=self.name,
            )

        refined: list[ChunkDict] = []
        for section in sections:
            if len(section["content"]) <= max_length:
                refined.append(section)
                continue
            for text in self._split_section(section["content"], max_length):
                refined.append(
                    self._make_chunk(
                        section["file_title"],
                        section["title"],
                        text,
                        metadata=section.get("metadata"),
                    )
                )

        merged = self._merge_short_chunks(refined, min_length, max_length)
        return [
            self._make_chunk(
                chunk["file_title"],
                chunk["title"],
                chunk["content"],
                order=index + 1,
                metadata=chunk.get("metadata"),
            )
            for index, chunk in enumerate(merged)
        ]

    def _step_5_print_stats(
        self,
        lines_count: int,
        sections: list[ChunkDict],
    ) -> None:
        """记录切分数量及长度统计。"""
        lengths = [len(section["content"]) for section in sections]
        if not lengths:
            self.logger.info("文档共 %d 行，未生成切片", lines_count)
            return
        self.logger.info(
            "文档共 %d 行，生成 %d 个切片，长度 min=%d, max=%d, avg=%.1f",
            lines_count,
            len(sections),
            min(lengths),
            max(lengths),
            sum(lengths) / len(lengths),
        )

    @staticmethod
    def _step_6_backup(
        state: ImportGraphState,
        sections: list[ChunkDict],
    ) -> None:
        """将切分结果写回流程状态，并导出 JSON 备份。"""
        state["chunks"] = sections

        file_title = state.get("file_title", "")
        file_dir = state.get("file_dir", "")
        if not file_dir:
            return
        output_path = Path(file_dir) / f"{file_title}_chunks.json"
        payload = {
            "file_title": file_title,
            "task_id": state.get("task_id", ""),
            "chunks": sections,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _make_chunk(
        file_title: str,
        title: str,
        content: str,
        order: int = 0,
        metadata: dict | None = None,
    ) -> ChunkDict:
        return {
            "file_title": file_title,
            "title": title,
            "content": content,
            "order": order,
            "metadata": dict(metadata) if metadata else {},
        }

    @staticmethod
    def _extract_title_text(line: str) -> str:
        """去掉标题行开头的 ``#`` 标记，返回纯标题文本。"""
        return re.sub(r"^#+\s*", "", line).strip()

    def _split_section(self, section: str, max_length: int) -> list[str]:
        section = section.strip()
        if not section:
            return []
        if len(section) <= max_length:
            return [section]

        first_line, separator, remainder = section.partition("\n")
        if separator and self._TITLE_PATTERN.match(first_line):
            heading = first_line.strip()
            body = remainder.strip()
        else:
            heading = ""
            body = section

        prefix = f"{heading}\n\n" if heading else ""
        body_limit = max_length - len(prefix)
        chunk_size = max_length if body_limit <= 0 else body_limit
        chunk_overlap = min(self.config.chunk_overlap, max(0, chunk_size - 1))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self._SEPARATORS,
            length_function=len,
        )
        if body_limit <= 0:
            # 极端的超长标题无法与正文同时容纳，对整段硬切。
            return [chunk for chunk in splitter.split_text(section) if chunk.strip()]

        body_chunks = splitter.split_text(body)
        return [f"{prefix}{chunk}".strip() for chunk in body_chunks if chunk.strip()]

    def _merge_short_chunks(
        self,
        chunks: list[ChunkDict],
        min_length: int,
        max_length: int,
    ) -> list[ChunkDict]:
        if min_length == 0:
            return chunks

        merged: list[ChunkDict] = []
        index = 0
        while index < len(chunks):
            current = chunks[index]
            if (
                len(current["content"]) < min_length
                and index + 1 < len(chunks)
                and current["title"] == chunks[index + 1]["title"]
                and len(current["content"])
                + 2
                + len(chunks[index + 1]["content"])
                <= max_length
            ):
                current = self._concat_chunks(current, chunks[index + 1])
                index += 1

            if (
                len(current["content"]) < min_length
                and merged
                and merged[-1]["title"] == current["title"]
                and len(merged[-1]["content"]) + 2 + len(current["content"])
                <= max_length
            ):
                merged[-1] = self._concat_chunks(merged[-1], current)
            elif current["content"]:
                merged.append(current)
            index += 1

        return merged

    @staticmethod
    def _concat_chunks(left: ChunkDict, right: ChunkDict) -> ChunkDict:
        """拼接两个同标题的切片，字段继承左侧，metadata 取并集。"""
        metadata = dict(left.get("metadata") or {})
        for key, value in (right.get("metadata") or {}).items():
            metadata.setdefault(key, value)
        return {
            "file_title": left["file_title"],
            "title": left["title"],
            "content": f"{left['content']}\n\n{right['content']}",
            "order": 0,
            "metadata": metadata,
        }
