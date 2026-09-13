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
    extract_history_messages: int = field(
        default_factory=lambda: int(
            os.getenv("QUERY_EXTRACT_HISTORY_MESSAGES", "8")
        )
    )
    extract_history_chars: int = field(
        default_factory=lambda: int(
            os.getenv("QUERY_EXTRACT_HISTORY_CHARS", "500")
        )
    )
    answer_history_messages: int = field(
        default_factory=lambda: int(
            os.getenv("ANSWER_HISTORY_MESSAGES", "6")
        )
    )
    answer_history_chars: int = field(
        default_factory=lambda: int(os.getenv("ANSWER_HISTORY_CHARS", "500"))
    )
    item_name_top_k: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_TOP_K", "5"))
    )
    item_name_confirm_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ITEM_NAME_CONFIRM_THRESHOLD", "0.80")
        )
    )
    item_name_candidate_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ITEM_NAME_CANDIDATE_THRESHOLD", "0.60")
        )
    )
    item_name_score_margin: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SCORE_MARGIN", "0.10"))
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
    answer_model: str = field(
        default_factory=lambda: os.getenv(
            "ANSWER_MODEL",
            os.getenv("LLM_DEFAULT_MODEL", ""),
        )
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
    dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )
    web_search_mcp_url: str = field(
        default_factory=lambda: os.getenv(
            "WEB_SEARCH_MCP_URL",
            "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
        )
    )
    web_search_tool: str = field(
        default_factory=lambda: os.getenv(
            "WEB_SEARCH_MCP_TOOL",
            "bailian_web_search",
        )
    )
    web_search_top_k: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_TOP_K", "5"))
    )
    web_search_timeout: float = field(
        default_factory=lambda: float(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_embedding_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_EMBEDDING_WEIGHT", "1.0"))
    )
    rrf_hyde_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_HYDE_WEIGHT", "0.7"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10"))
    )
    rerank_url: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_RERANK_URL", "")
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_RERANK_MODEL",
            "qwen3.7-text-rerank",
        )
    )
    rerank_timeout: float = field(
        default_factory=lambda: float(os.getenv("RERANK_TIMEOUT", "30"))
    )
    rerank_min_results: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_RESULTS", "3"))
    )
    rerank_max_results: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_RESULTS", "10"))
    )
    rerank_absolute_gap: float = field(
        default_factory=lambda: float(os.getenv("RERANK_ABSOLUTE_GAP", "0.5"))
    )
    rerank_relative_gap: float = field(
        default_factory=lambda: float(os.getenv("RERANK_RELATIVE_GAP", "0.25"))
    )
