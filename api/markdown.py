"""安全渲染大模型返回的 Markdown。"""

from markdown_it import MarkdownIt

from utils.text_utils import normalize_markdown_images

_renderer = MarkdownIt(
    "commonmark",
    {
        "html": False,
        "breaks": True,
        "linkify": False,
    },
)


def render_markdown(content: str) -> str:
    """禁用原始 HTML 后渲染 Markdown，供同源页面展示。"""
    return _renderer.render(normalize_markdown_images(content))
