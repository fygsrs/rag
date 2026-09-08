# config/embedding_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    provider: str  # "bge" 本地 / "aliyun" 阿里云
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool
    batch_size: int
    dashscope_api_key: str
    dashscope_native_url: str
    dashscope_model: str
    dimension: int
    request_timeout: float


embedding_config = EmbeddingConfig(
    provider=os.getenv("EMBEDDING_PROVIDER", "aliyun"),
    bge_m3_path=os.getenv("BGE_M3_PATH", ""),
    bge_m3=os.getenv("BGE_M3", "BAAI/bge-m3"),
    bge_device=os.getenv("BGE_DEVICE", "cpu"),
    # 特殊处理：将env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1),
    batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("OPENAI_API_KEY"),
    dashscope_native_url=os.getenv(
        "DASHSCOPE_NATIVE_URL",
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "text-embedding/text-embedding",
    ),
    dashscope_model=os.getenv(
        "DASHSCOPE_EMBEDDING_MODEL",
        "qwen3.7-text-embedding-flash",
    ),
    dimension=int(os.getenv("EMBEDDING_DIM", "1024")),
    request_timeout=float(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "60")),
)
