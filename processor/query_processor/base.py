"""查询流程节点基类。"""

from abc import ABC, abstractmethod
from typing import TypeVar

from processor.query_processor.logger import get_query_logger

T = TypeVar("T")


class NodeBase(ABC):
    """为查询节点提供统一调用和异常日志。"""

    name = "base_node"

    def __init__(self) -> None:
        self.logger = get_query_logger(self.name)

    def __call__(self, state: T) -> T:
        try:
            self.logger.info("--- %s 开始 ---", self.name)
            result = self.process(state)
            self.logger.info("--- %s 完成 ---", self.name)
            return result
        except Exception:
            self.logger.exception("%s 执行失败", self.name)
            raise

    @abstractmethod
    def process(self, state: T) -> T:
        """处理状态并返回本节点产生的状态更新。"""
        raise NotImplementedError
