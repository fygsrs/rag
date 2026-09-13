from types import SimpleNamespace

from processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput


def make_config(**overrides):
    values = {
        "history_limit": 20,
        "answer_history_messages": 6,
        "answer_history_chars": 500,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_document(index, content):
    return {
        "id": index,
        "title": f"文档 {index}",
        "content": content,
        "url": f"https://example.com/{index}",
        "source": "embedding",
        "score": 0.9,
        "metadata": {},
    }


def make_state(history):
    return {
        "original_query": "B7-420 怎么设置触摸板？",
        "rewritten_query": "B7-420 怎么设置触摸板？",
        "item_names": ["B7-420"],
        "history": history,
    }


def test_answer_prompt_truncates_history_and_keeps_evidence_intact():
    long_answer = "历史回答内容。" * 200
    history = [
        (
            {"role": "user", "content": f"第 {index} 个问题", "metadata": {}}
            if index % 2 == 0
            else {"role": "assistant", "content": long_answer, "metadata": {}}
        )
        for index in range(10)
    ]
    evidence = "证据正文 " + "图片 ![说明](http://minio.test/rag/a.png) " * 80
    node = NodeAnswerOutput(make_config())

    prompt = node._build_prompt(
        make_state(history),
        [make_document(1, evidence)],
    )

    assert "第 0 个问题" not in prompt
    assert "第 2 个问题" not in prompt
    assert "第 4 个问题" in prompt
    assert long_answer not in prompt
    assert evidence.rstrip() in prompt


def test_answer_prompt_uses_config_history_window():
    history = [
        {"role": "user", "content": f"历史消息 {index}", "metadata": {}}
        for index in range(5)
    ]
    node = NodeAnswerOutput(make_config(answer_history_messages=2))

    prompt = node._build_prompt(make_state(history), [])

    assert "历史消息 3" in prompt
    assert "历史消息 4" in prompt
    assert "历史消息 2" not in prompt
