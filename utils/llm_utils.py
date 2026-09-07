from langchain_openai import ChatOpenAI

from config.llm_config import llm_config

_llm_client_cache: dict[tuple[str, bool], ChatOpenAI] = {}


def get_llm_client(
    model: str | None = None,
    json_mode: bool = False,
) -> ChatOpenAI:
    """获取并缓存 ChatOpenAI 客户端。

    Args:
        model: 模型名称；未传入时使用环境变量中的默认模型。
        json_mode: 是否要求模型返回 JSON 对象。
    """
    selected_model = model or llm_config.llm_model
    if not selected_model:
        raise ValueError("未配置 LLM 模型，请设置 LLM_DEFAULT_MODEL")

    cache_key = (selected_model, json_mode)
    if cache_key in _llm_client_cache:
        return _llm_client_cache[cache_key]

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model=selected_model,
        temperature=llm_config.llm_temperature,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model_kwargs=model_kwargs,
    )

    _llm_client_cache[cache_key] = client
    return client

if __name__ == "__main__":
    client = get_llm_client()
    print(client.invoke("你是谁"))
