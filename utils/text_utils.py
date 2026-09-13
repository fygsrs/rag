"""Markdown 图片规范化和前端摘要所需的轻量文本处理。"""

import html
import re


_HTML_IMAGE_PATTERN = re.compile(
    r"\\?<img\b(?P<attributes>[^>]*)/?>",
    re.IGNORECASE,
)
_IMAGE_SOURCE_PATTERN = re.compile(
    r"\bsrc\s*=\s*([\"'])(?P<value>.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
_IMAGE_ALT_PATTERN = re.compile(
    r"\balt\s*=\s*([\"'])(?P<value>.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN_LINK_PATTERN = re.compile(
    r"\[(?P<label>https?://[^\]]+)]\((?P<url>https?://[^)\s]+)\)"
)
_HTTP_URL_PATTERN = re.compile(r"https?://[^\s\"'<>\]]+")


def normalize_markdown_images(content: str) -> str:
    """把 HTML 或被模型改坏的图片标签转换成安全 Markdown 图片。"""

    def replace_image(match: re.Match[str]) -> str:
        attributes = html.unescape(match.group("attributes"))
        source_match = _IMAGE_SOURCE_PATTERN.search(attributes)
        if source_match is None:
            return match.group(0)

        source = source_match.group("value").strip()
        markdown_link = _MARKDOWN_LINK_PATTERN.fullmatch(source)
        if markdown_link:
            source = markdown_link.group("url")
        else:
            url_match = _HTTP_URL_PATTERN.search(source)
            if url_match is None:
                return match.group(0)
            source = url_match.group(0)

        alt_match = _IMAGE_ALT_PATTERN.search(attributes)
        alt_text = html.unescape(alt_match.group("value")) if alt_match else "资料图片"
        alt_text = re.sub(r"<[^>]*>", " ", alt_text)
        alt_text = " ".join(alt_text.replace("[", "").replace("]", "").split())
        if not alt_text:
            alt_text = "资料图片"
        return f"![{alt_text[:120]}]({source})"

    return _HTML_IMAGE_PATTERN.sub(replace_image, content or "")


def summarize_content(content: str, *, max_length: int = 280) -> str:
    """将 Markdown 切片压缩为不包含长链接的单行摘要。"""
    text = re.sub(r"!\[([^]]*)]\([^)]*\)", r"\1", content or "")
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^[#>*+-]+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("`", "")
    text = " ".join(text.split())
    if len(text) > max_length:
        return f"{text[: max_length - 3]}..."
    return text
