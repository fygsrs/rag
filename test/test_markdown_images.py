from api.markdown import render_markdown
from utils.text_utils import normalize_markdown_images


def test_normalize_html_image_to_markdown():
    source = '<img src="http://minio/rag/touch.jpg" alt="双指滚动"/>'

    result = normalize_markdown_images(source)

    assert result == "![双指滚动](http://minio/rag/touch.jpg)"


def test_normalize_llm_wrapped_image_url():
    source = (
        r'\<img src="[http://minio/rag/touch.jpg]'
        r'(http://minio/rag/touch.jpg)" alt="双指滚动"/>'
    )

    result = normalize_markdown_images(source)

    assert result == "![双指滚动](http://minio/rag/touch.jpg)"


def test_render_normalized_image_without_enabling_raw_html():
    source = '<img src="http://minio/rag/touch.jpg" alt="双指滚动"/>'

    result = render_markdown(source)

    assert '<img src="http://minio/rag/touch.jpg" alt="双指滚动"' in result
