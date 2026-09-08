from processor.import_processor.config import ImportConfig
from processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit


def make_node(max_length=80, min_length=0, overlap=1):
    return NodeDocumentSplit(
        ImportConfig(
            max_content_length=max_length,
            min_content_length=min_length,
            chunk_overlap=overlap,
        )
    )


def contents(chunks):
    return [chunk["content"] for chunk in chunks]


def test_process_splits_markdown_titles_and_writes_chunks():
    state = {
        "file_title": "示例文档",
        "md_content": "前言\n\n## 第一章\n内容一\n\n## 第二章\n内容二",
    }

    result = make_node(max_length=100).process(state)

    assert result is state
    assert contents(result["chunks"]) == [
        "# 示例文档\n\n前言",
        "## 第一章\n内容一",
        "## 第二章\n内容二",
    ]
    assert all(chunk["file_title"] == "示例文档" for chunk in result["chunks"])
    assert all(chunk["order"] == i + 1 for i, chunk in enumerate(result["chunks"]))
    assert all(isinstance(chunk["metadata"], dict) for chunk in result["chunks"])
    assert result["chunks"][1]["title"] == "第一章"


def test_process_adds_default_title_when_document_has_no_title():
    state = {"file_title": "无标题文档", "md_content": "第一段。\n\n第二段。"}

    result = make_node(max_length=100).process(state)

    chunk = result["chunks"][0]
    assert contents(result["chunks"]) == ["# 无标题文档\n\n第一段。\n\n第二段。"]
    assert chunk["title"] == "无标题文档"
    assert chunk["file_title"] == "无标题文档"


def test_get_inputs_normalizes_line_endings():
    state = {
        "file_title": "混合换行",
        "md_content": "第一行\r\n第二行\r第三行\n第四行",
    }

    content, file_title = make_node()._step_1_get_inputs(state)

    assert content == "第一行\n第二行\n第三行\n第四行"
    assert file_title == "混合换行"


def test_long_section_respects_limit_and_repeats_heading():
    state = {
        "file_title": "长文档",
        "md_content": "## 章节\n" + "第一句很长。第二句也很长。" * 10,
    }

    chunks = make_node(max_length=40).process(state)["chunks"]

    assert len(chunks) > 1
    assert all(len(chunk["content"]) <= 40 for chunk in chunks)
    assert all(chunk["content"].startswith("## 章节") for chunk in chunks)
    assert all(chunk["title"] == "章节" for chunk in chunks)


def test_heading_inside_fenced_code_is_not_a_section_title():
    content = "## 正文\n```python\n# 这不是标题\nprint('ok')\n```\n结束"
    node = make_node(max_length=200)

    sections, title_count, _ = node._step_2_split_by_title(content)

    assert title_count == 1
    assert contents(sections) == [content]
    assert sections[0]["title"] == "正文"
