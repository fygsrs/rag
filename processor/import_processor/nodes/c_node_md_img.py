import base64
import html
import json
import logging
import mimetypes
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from langchain_core.messages import HumanMessage

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import (
    FileProcessingError,
    ImageProcessingError,
    MinioError,
    StateFieldError,
)
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client


@dataclass(frozen=True)
class MarkdownImage:
    """Markdown 中的一条本地图片引用。"""

    markdown: str
    alt_text: str
    relative_path: str
    file_path: Path


class NodeMDImg(BaseNode):
    """Markdown 图片处理节点：生成图片摘要并上传至 MinIO。"""

    name = "node_md_img"
    _IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)]\((?P<target>[^)]+)\)")
    _HTML_IMAGE_PATTERN = re.compile(
        r"<img\b[^>]*\bsrc=[\"'](?P<target>[^\"']+)[\"'][^>]*>",
        re.IGNORECASE,
    )

    def process(self, state: ImportGraphState):
        # 1. 参数处理
        md_content, md_path_obj, images_dir = self._get_content(state)

        # 2. 图片扫描
        summary_images = self._scan_images(md_content, images_dir)
        html_images = self._scan_html_images(md_content, images_dir)
        target_images = []
        seen_paths: set[Path] = set()
        for image in [*summary_images, *html_images]:
            image_path = Path(image[1])
            if image_path in seen_paths:
                continue
            seen_paths.add(image_path)
            target_images.append(image)
        if not target_images:
            self.logger.info("Markdown 中未发现需要处理的本地图片")
            state["md_content"] = md_content
            return state

        # 3. 视觉模型摘要
        document_title = str(state.get("file_title") or md_path_obj.stem)
        summaries = self._generate_summaries(
            document_title,
            summary_images,
        )

        # 4. 上传 MinIO，并替换 Markdown 中的本地图片地址
        new_md_content = self._upload_and_replace(
            document_title,
            target_images,
            summaries,
            md_content,
        )
        self.logger.info(
            "图片上传及 Markdown 链接替换完成 | count=%d",
            len(target_images),
        )

        # 5. 保存处理后的 Markdown，不覆盖原文件
        new_md_file_name = self._backup_new_md_file(md_path_obj, new_md_content)
        state["md_path"] = new_md_file_name
        state["md_content"] = new_md_content
        self.logger.info(
            "图片处理后的 Markdown 已保存 | path=%s",
            new_md_file_name,
        )
        return state

    def _get_content(
        self,
        state: ImportGraphState,
    ) -> tuple[str, Path, Path]:
        """读取 Markdown 内容，并返回 Markdown 路径和图片目录。"""
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name="md_path", expected_type=str)

        md_path_obj = Path(md_path).expanduser().resolve()
        if not md_path_obj.is_file():
            raise FileProcessingError(message=f"Markdown 文件不存在: {md_path_obj}")
        if md_path_obj.suffix.lower() not in {".md", ".markdown"}:
            raise FileProcessingError(message=f"不是 Markdown 文件: {md_path_obj}")

        md_content = state.get("md_content")
        if not md_content:
            md_content = md_path_obj.read_text(encoding="utf-8")

        return md_content, md_path_obj, md_path_obj.parent / "images"

    def _scan_images(
        self,
        md_content: str,
        images_dir: Path,
    ):
        """扫描 Markdown 中存在的本地图片，忽略远程图片。"""
        target_images = []
        if not images_dir.is_dir():
            self.logger.warning("图片目录不存在: %s", images_dir)
            return target_images

        seen_files: set[str] = set()
        for match in self._IMAGE_PATTERN.finditer(md_content):
            relative_path = self._extract_image_path(match.group("target"))
            if self._is_remote_image(relative_path):
                continue
            image_file = Path(relative_path.replace("\\", "/")).name
            if not image_file or image_file in seen_files:
                continue
            lower = os.path.splitext(image_file)[1].lower()
            if lower not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持: {image_file}")
                continue
            img_path = images_dir / image_file
            if not img_path.is_file():
                self.logger.warning("Markdown 本地图片不存在: %s", img_path)
                continue
            seen_files.add(image_file)
            target_images.append(
                (image_file, img_path, self._extract_context(md_content, match))
            )

        self.logger.info("发现 %d 张需要视觉摘要的 Markdown 图片", len(target_images))
        return target_images

    def _scan_html_images(
        self,
        md_content: str,
        images_dir: Path,
    ) -> list[tuple[str, Path, tuple[str, str]]]:
        """扫描 HTML 表格里的本地图片；上传但不额外调用视觉模型。"""
        if not images_dir.is_dir():
            return []
        images = []
        seen_files: set[str] = set()
        for match in self._HTML_IMAGE_PATTERN.finditer(md_content):
            relative_path = unquote(match.group("target").strip())
            if self._is_remote_image(relative_path):
                continue
            image_file = Path(relative_path.replace("\\", "/")).name
            if not image_file or image_file in seen_files:
                continue
            image_path = images_dir / image_file
            if not image_path.is_file():
                self.logger.warning("HTML 本地图片不存在: %s", image_path)
                continue
            seen_files.add(image_file)
            images.append(
                (image_file, image_path, self._extract_context(md_content, match))
            )
        self.logger.info("发现 %d 张 HTML 表格图片", len(images))
        return images

    @staticmethod
    def _is_remote_image(target: str) -> bool:
        normalized = target.strip().casefold()
        return normalized.startswith(("http://", "https://", "data:", "//"))

    @staticmethod
    def _extract_context(
        md_content: str,
        match: re.Match,
        context_len: int = 100,
    ) -> tuple[str, str]:
        start, end = match.span()
        return (
            md_content[max(0, start - context_len) : start],
            md_content[end : min(len(md_content), end + context_len)],
        )

    @staticmethod
    def find_image_in_md(
        md_content: str,
        image_file: str,
        context_len: int = 100,
    ) -> tuple[str, str] | None:
        """查找图片在 Markdown 中的位置，并返回图片前后的文本。

        Args:
            md_content: Markdown 文档内容。
            image_file: 图片文件名，例如 ``example.jpg``。
            context_len: 图片前后分别截取的最大字符数，默认为 100。

        Returns:
            找到时返回 ``(上文, 下文)``，没有找到时返回 ``None``。
        """
        if not image_file or context_len < 0:
            return None

        pattern = re.compile(
            r"(?:!\[[^\]]*]\([^)]*?|<img\b[^>]*\bsrc=[\"'][^\"']*?)"
            + re.escape(image_file)
            + r"(?:[^)]*\)|[^\"']*[\"'][^>]*>)",
            re.IGNORECASE,
        )
        match = pattern.search(md_content)
        if not match:
            return None

        start, end = match.span()
        previous_text = md_content[max(0, start - context_len) : start]
        following_text = md_content[end : min(len(md_content), end + context_len)]
        return previous_text, following_text

    @staticmethod
    def _extract_image_path(raw_target: str) -> str:
        """从 Markdown 图片目标中提取路径，并移除可选标题。"""
        target = raw_target.strip()
        if target.startswith("<") and ">" in target:
            target = target[1 : target.index(">")]
        else:
            target = re.split(r"\s+(?=[\"'])", target, maxsplit=1)[0]
        return unquote(target.strip())

    def _wait_for_rate_limit_window(
        self,
        request_times: deque[float],
        max_requests: int,
        window_seconds: float = 60.0,
    ) -> None:
        """使用滑动时间窗口限制 API 请求次数。

        ``request_times`` 保存窗口内每次请求的开始时间。当最近一个窗口
        已达到最大请求数时，等待最早的请求离开窗口后再继续。
        """
        request_limit = max(max_requests, 1)
        if window_seconds <= 0:
            raise ValueError("window_seconds 必须大于 0")

        while True:
            current_time = time.monotonic()

            while (
                request_times
                and current_time - request_times[0] >= window_seconds
            ):
                request_times.popleft()

            if len(request_times) < request_limit:
                request_times.append(current_time)
                return

            wait_seconds = window_seconds - (current_time - request_times[0])
            if wait_seconds > 0:
                self.logger.info(
                    "大模型请求达到时间窗口上限，等待 %.2f 秒",
                    wait_seconds,
                )
                time.sleep(wait_seconds)

    def _generate_summaries(
        self,
        document_title: str,
        target_images: list,
    ) -> dict[Path, str]:
        """调用视觉模型，为每张图片生成适合知识检索的说明。"""
        client = get_llm_client(model=self.config.vl_model)
        summaries: dict[Path, str] = {}
        request_times: deque[float] = deque()

        for index, image in enumerate(target_images):
            image_file, image_path, context = image
            image_path = Path(image_path)
            previous_text, following_text = context or ("", "")

            mime_type = mimetypes.guess_type(image_file)[0] or "image/jpeg"
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"这张图片来自文档《{document_title}》。请准确描述图片内容，"
                            "重点提取标题、文字、数字、表格字段、部件名称及图中关系。"
                            "输出一段适合知识库检索的中文说明，不要使用 Markdown，不要臆测。\n"
                            f"图片上文：{previous_text or '无'}\n"
                            f"图片下文：{following_text or '无'}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ]
            )

            self._wait_for_rate_limit_window(
                request_times,
                self.config.requests_per_minute,
            )
            try:
                response = client.invoke([message])
                summary = self._response_to_text(response.content)
            except Exception as exc:
                raise ImageProcessingError(
                    message=f"图片摘要生成失败: {image_file}",
                    cause=exc,
                ) from exc

            if not summary:
                raise ImageProcessingError(
                    message=f"视觉模型未返回图片摘要: {image_file}"
                )
            summaries[image_path] = summary
            self.logger.info(
                "图片摘要生成完成 (%d/%d): %s",
                index + 1,
                len(target_images),
                image_file,
            )

        return summaries

    @staticmethod
    def _response_to_text(content: object) -> str:
        """兼容字符串和内容块形式的模型响应。"""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block["text"])
            return "\n".join(texts).strip()
        return str(content).strip()

    def _upload_and_replace(
        self,
        document_title: str,
        target_images: list,
        summaries: dict[Path, str],
        md_content: str,
    ) -> str:
        """上传图片到 MinIO，并替换 Markdown 图片链接和说明。"""
        client = get_minio_client()
        if client is None:
            raise MinioError(message="MinIO 客户端未初始化")

        new_md_content = md_content
        for image in target_images:
            image_file, image_path, _ = image
            image_path = Path(image_path)
            object_name = f"{document_title}/images/{image_file}"
            content_type = (
                mimetypes.guess_type(image_file)[0]
                or "application/octet-stream"
            )
            try:
                client.fput_object(
                    self.config.minio_bucket,
                    object_name,
                    str(image_path),
                    content_type=content_type,
                )
            except Exception as exc:
                raise MinioError(
                    message=f"图片上传失败: {image_file}",
                    cause=exc,
                ) from exc

            object_url = (
                f"{self.config.get_minio_base_url()}/"
                f"{quote(self.config.minio_bucket)}/{quote(object_name, safe='/')}"
            )
            summary = summaries.get(image_path, "").strip()
            if not summary:
                previous_text, following_text = image[2] or ("", "")
                context_text = re.sub(
                    r"<[^>]+>",
                    " ",
                    f"{previous_text} {following_text}",
                )
                summary = " ".join(context_text.split()) or "文档图片"
            alt_text = " ".join(summary.split())[:120]
            alt_text = alt_text.replace("[", "").replace("]", "")
            replacement = (
                f"![{alt_text}]({object_url})\n\n"
                f"> 图片说明：{summary}"
            )
            image_pattern = re.compile(
                r"!\[[^\]]*]\([^)]*?" + re.escape(image_file) + r"[^)]*\)"
            )
            new_md_content = image_pattern.sub(
                lambda _: replacement,
                new_md_content,
            )
            html_pattern = re.compile(
                r"<img\b[^>]*\bsrc=[\"'][^\"']*?"
                + re.escape(image_file)
                + r"[^\"']*[\"'][^>]*>",
                re.IGNORECASE,
            )
            html_replacement = (
                f'<img src="{object_url}" alt="{html.escape(alt_text, quote=True)}"/>'
            )
            new_md_content = html_pattern.sub(
                lambda _: html_replacement,
                new_md_content,
            )

        return new_md_content

    @staticmethod
    def _backup_new_md_file(md_path: Path | str, md_content: str) -> str:
        """将处理结果另存为 *_processed.md，并返回绝对路径。"""
        md_path_obj = Path(md_path).expanduser().resolve()
        stem = md_path_obj.stem
        if not stem.endswith("_processed"):
            stem = f"{stem}_processed"
        new_md_path = md_path_obj.with_name(f"{stem}{md_path_obj.suffix}")
        new_md_path.write_text(md_content, encoding="utf-8")
        return str(new_md_path)


if __name__ == "__main__":
    setup_logging()
    state = {
        "md_path":r"E:\code\py\rag\output\hak180产品安全手册\hybrid_auto\hak180产品安全手册.md"
    }
    node = NodeMDImg()
    result = node(state)
    dumps = json.dumps(result, indent=2)
    print(dumps)
