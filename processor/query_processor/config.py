"""知识库查询工作流配置。"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class QueryConfig:
    """知识库查询节点使用的模型、检索和判定配置。"""

    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    history_limit: int = field(
        default_factory=lambda: int(os.getenv("QUERY_HISTORY_LIMIT", "20"))
    )
    item_name_top_k: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_TOP_K", "5"))
    )
    item_name_confirm_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ITEM_NAME_CONFIRM_THRESHOLD", "0.85")
        )
    )
    item_name_candidate_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ITEM_NAME_CANDIDATE_THRESHOLD", "0.60")
        )
    )
    item_name_score_margin: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SCORE_MARGIN", "0.15"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.7"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.3"))
    )
    hyde_model: str = field(
        default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    search_top_k: int = field(
        default_factory=lambda: int(os.getenv("QUERY_SEARCH_TOP_K", "5"))
    )
    search_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("QUERY_SEARCH_DENSE_WEIGHT", "0.8"))
    )
    search_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("QUERY_SEARCH_SPARSE_WEIGHT", "0.2"))
    )
