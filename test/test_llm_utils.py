from utils import llm_utils


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def setup_function():
    llm_utils._llm_client_cache.clear()


def teardown_function():
    llm_utils._llm_client_cache.clear()


def test_query_deepseek_client_explicitly_disables_thinking(monkeypatch):
    monkeypatch.setattr(llm_utils, "ChatOpenAI", FakeChatOpenAI)

    client = llm_utils._get_client(
        scope="query",
        model="deepseek-flash",
        json_mode=False,
        base_url="https://api.deepseek.com/v1",
        api_key="test-key",
    )

    assert client.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_non_deepseek_client_does_not_receive_deepseek_option(monkeypatch):
    monkeypatch.setattr(llm_utils, "ChatOpenAI", FakeChatOpenAI)

    client = llm_utils._get_client(
        scope="query",
        model="qwen3.7-flash",
        json_mode=False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test-key",
    )

    assert "extra_body" not in client.kwargs
