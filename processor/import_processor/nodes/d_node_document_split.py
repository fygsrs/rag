import re

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import DocumentSplitError, StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """文档切分节点：优先按 Markdown 标题切分，再细化超长片段。"""

    name = "node_document_split"
    _TITLE_PATTERN = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)")
    _SENTENCE_BOUNDARY_PATTERN = re.compile(
        r"(?<=[。！？!?；;])|(?<=\.)(?=\s|$)"
    )

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
    ) -> tuple[list[str], int, int]:
        """按 Markdown ATX 标题切分，忽略代码块中的 ``#``。"""
        del file_title  # 保留参数以对应 process 中清晰的步骤接口。
        lines = content.splitlines()
        sections: list[str] = []
        current_lines: list[str] = []
        title_count = 0
        fence_marker: str | None = None

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
                self._append_section(sections, current_lines)
                current_lines = [line.rstrip()]
            else:
                current_lines.append(line.rstrip())

        self._append_section(sections, current_lines)
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(
        self,
        content: str,
        sections: list[str],
        title_count: int,
        file_title: str,
    ) -> list[str]:
        """为无标题文档及首个标题前的正文补充文档标题。"""
        default_title = f"# {file_title}"
        if title_count == 0:
            return [f"{default_title}\n\n{content.strip()}"]

        if sections and not self._TITLE_PATTERN.match(sections[0].splitlines()[0]):
            sections[0] = f"{default_title}\n\n{sections[0]}"
        return sections

    def _step_4_refine_chunks(self, sections: list[str]) -> list[str]:
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

        refined: list[str] = []
        for section in sections:
            refined.extend(self._split_section(section, max_length))

        return self._merge_short_chunks(refined, min_length, max_length)

    def _step_5_print_stats(
        self,
        lines_count: int,
        sections: list[str],
    ) -> None:
        """记录切分数量及长度统计。"""
        lengths = [len(section) for section in sections]
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
        sections: list[str],
    ) -> None:
        """将切分结果写回流程状态。"""
        state["chunks"] = sections

    @staticmethod
    def _append_section(sections: list[str], lines: list[str]) -> None:
        section = "\n".join(lines).strip()
        if section:
            sections.append(section)

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
        if body_limit <= 0:
            # 极端的超长标题无法与正文同时容纳，标题自身单独作为一个切片。
            return self._hard_split(section, max_length)

        body_chunks = self._split_body(body, body_limit)
        return [f"{prefix}{chunk}".strip() for chunk in body_chunks if chunk.strip()]

    def _split_body(self, body: str, limit: int) -> list[str]:
        """先按段落装箱，单个超长段落再按句子切分。"""
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body)]
        paragraphs = [part for part in paragraphs if part]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            parts = (
                [paragraph]
                if len(paragraph) <= limit
                else self._split_long_paragraph(paragraph, limit)
            )
            for part in parts:
                candidate = f"{current}\n\n{part}" if current else part
                if len(candidate) <= limit:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part

        if current:
            chunks.append(current)
        return chunks

    def _split_long_paragraph(self, paragraph: str, limit: int) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in self._SENTENCE_BOUNDARY_PATTERN.split(paragraph)
            if sentence.strip()
        ]
        if len(sentences) <= 1:
            return self._hard_split(paragraph, limit)

        chunks: list[str] = []
        current_sentences: list[str] = []
        overlap_count = max(0, self.config.overlap_sentences)

        for sentence in sentences:
            if len(sentence) > limit:
                if current_sentences:
                    chunks.append("".join(current_sentences))
                    current_sentences = []
                chunks.extend(self._hard_split(sentence, limit))
                continue

            candidate = "".join([*current_sentences, sentence])
            if current_sentences and len(candidate) > limit:
                chunks.append("".join(current_sentences))
                current_sentences = self._overlap_tail(
                    current_sentences,
                    overlap_count,
                    limit - len(sentence),
                )
            current_sentences.append(sentence)

        if current_sentences:
            chunks.append("".join(current_sentences))
        return chunks

    @staticmethod
    def _overlap_tail(
        sentences: list[str],
        count: int,
        available_length: int,
    ) -> list[str]:
        if count <= 0 or available_length <= 0:
            return []
        tail = sentences[-count:]
        while tail and len("".join(tail)) > available_length:
            tail.pop(0)
        return tail

    @staticmethod
    def _hard_split(text: str, limit: int) -> list[str]:
        return [text[index : index + limit] for index in range(0, len(text), limit)]

    @staticmethod
    def _merge_short_chunks(
        chunks: list[str],
        min_length: int,
        max_length: int,
    ) -> list[str]:
        if min_length == 0:
            return chunks

        merged: list[str] = []
        index = 0
        while index < len(chunks):
            current = chunks[index].strip()
            if (
                len(current) < min_length
                and index + 1 < len(chunks)
                and len(current) + 2 + len(chunks[index + 1].strip()) <= max_length
            ):
                current = f"{current}\n\n{chunks[index + 1].strip()}"
                index += 1

            if (
                len(current) < min_length
                and merged
                and len(merged[-1]) + 2 + len(current) <= max_length
            ):
                merged[-1] = f"{merged[-1]}\n\n{current}"
            elif current:
                merged.append(current)
            index += 1

        return merged
