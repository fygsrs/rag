"""MongoDB 连接及记忆集合配置。"""

import os
from dataclasses import dataclass
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MongoDBConfig:
    host: str
    port: int
    username: str
    password: str
    auth_source: str
    database: str
    memory_collection: str
    server_selection_timeout_ms: int

    def get_uri(self) -> str:
        """生成经过 URL 转义的 MongoDB 连接地址。"""
        username = quote_plus(self.username)
        password = quote_plus(self.password)
        auth_source = quote_plus(self.auth_source)
        return (
            f"mongodb://{username}:{password}@{self.host}:{self.port}/"
            f"?authSource={auth_source}"
        )


mongodb_config = MongoDBConfig(
    host=os.getenv("MONGODB_HOST", "127.0.0.1"),
    port=int(os.getenv("MONGODB_PORT", "27017")),
    username=os.getenv("MONGODB_USERNAME", ""),
    password=os.getenv("MONGODB_PASSWORD", ""),
    auth_source=os.getenv("MONGODB_AUTH_SOURCE", "admin"),
    database=os.getenv("MONGODB_DATABASE", "rag_memory"),
    memory_collection=os.getenv(
        "MONGODB_MEMORY_COLLECTION",
        "conversation_memories",
    ),
    server_selection_timeout_ms=int(
        os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")
    ),
)
