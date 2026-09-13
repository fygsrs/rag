from langchain_openai import ChatOpenAI

from config.llm_config import llm_config

_llm_client_cache: dict[tuple[str, str, bool], ChatOpenAI] = {}


def _get_client(
    *,
    scope: str,
    model: str | None,
    json_mode: bool,
    base_url: str | None,
    api_key: str | None,
) -> ChatOpenAI:
    """按 provider 作用域缓存并返回 OpenAI 兼容客户端。"""
    if not model:
        if scope == "vl":
            raise ValueError("未配置视觉模型，请设置 VL_MODEL")
        raise ValueError("未配置 LLM 模型，请设置 LLM_DEFAULT_MODEL")

    cache_key = (scope, model, json_mode)
    cached = _llm_client_cache.get(cache_key)
    if cached is not None:
        return cached

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model=model,
        temperature=llm_config.llm_temperature,
        base_url=base_url,
        api_key=api_key,
        model_kwargs=model_kwargs,
    )

    _llm_client_cache[cache_key] = client
    return client


def get_llm_client(
    model: str | None = None,
    json_mode: bool = False,
) -> ChatOpenAI:
    """获取查询侧（商品提取 / HyDE / 答案生成）客户端。

    Args:
        model: 模型名称；未传入时使用环境变量中的默认模型。
        json_mode: 是否要求模型返回 JSON 对象。
    """
    return _get_client(
        scope="query",
        model=model or llm_config.llm_model,
        json_mode=json_mode,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )


def get_vl_client(model: str | None = None) -> ChatOpenAI:
    """获取视觉模型客户端（导入图片摘要），使用视觉专用端点。"""
    return _get_client(
        scope="vl",
        model=model or llm_config.vl_model,
        json_mode=False,
        base_url=llm_config.vl_base_url,
        api_key=llm_config.vl_api_key,
    )


if __name__ == "__main__":
    client = get_llm_client()
    print(client.invoke("你是谁"))
