"""查询流程节点基类。"""

from abc import ABC, abstractmethod
import time
from typing import TypeVar

from processor.query_processor.logger import get_query_logger

T = TypeVar("T")


class NodeBase(ABC):
    """为查询节点提供统一调用和异常日志。"""

    name = "base_node"

    def __init__(self) -> None:
        self.logger = get_query_logger(self.name)

    def __call__(self, state: T) -> T:
        started_at = time.perf_counter()
        context_parts = []
        if isinstance(state, dict):
            for field_name in ("session_id", "message_id"):
                value = str(state.get(field_name) or "").strip()
                if value:
                    context_parts.append(f"{field_name}={value}")
        context = f" | {' | '.join(context_parts)}" if context_parts else ""

        try:
            self.logger.info("--- %s 开始%s ---", self.name, context)
            result = self.process(state)
            self.logger.info(
                "--- %s 完成%s | elapsed=%.2fs ---",
                self.name,
                context,
                time.perf_counter() - started_at,
            )
            return result
        except Exception:
            self.logger.exception(
                "--- %s 失败%s | elapsed=%.2fs ---",
                self.name,
                context,
                time.perf_counter() - started_at,
            )
            raise

    @abstractmethod
    def process(self, state: T) -> T:
        """处理状态并返回本节点产生的状态更新。"""
        raise NotImplementedError
