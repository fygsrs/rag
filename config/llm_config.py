# config/llm_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    # 查询侧（商品提取 / HyDE / 答案生成）端点
    base_url: str
    api_key: str
    # 视觉模型（导入图片摘要）端点；未配置时回退查询侧
    vl_base_url: str
    vl_api_key: str
    vl_model: str
    llm_model: str
    item_model: str
    llm_temperature: float


llm_config = LLMConfig(
    base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    vl_base_url=(
        os.getenv("VL_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("LLM_BASE_URL")
    ),
    vl_api_key=(
        os.getenv("VL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
    ),
    vl_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    item_model=os.getenv("ITEM_MODEL"),
    llm_temperature=float(
        os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")
    )
)