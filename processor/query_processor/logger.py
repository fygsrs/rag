"""查询流程日志配置。"""

import logging


def get_query_logger(node_name: str) -> logging.Logger:
    """获取带查询流程命名空间的日志对象。"""
    return logging.getLogger(f"query.{node_name}")


def setup_logging(level: int = logging.INFO) -> None:
    """配置查询流程控制台日志。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
