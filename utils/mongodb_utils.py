"""基于 MongoDB 的会话记忆存储工具。"""

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient

from config.mongodb_config import MongoDBConfig, mongodb_config


class MongoDBUtil:
    """保存、读取和清理按会话组织的消息记忆。"""

    def __init__(self, config: MongoDBConfig | None = None):
        self.config = config or mongodb_config
        self._validate_config()
        self.client = MongoClient(
            self.config.get_uri(),
            serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
            connectTimeoutMS=self.config.server_selection_timeout_ms,
            tz_aware=True,
        )
        self.database = self.client[self.config.database]
        self.collection = self.database[self.config.memory_collection]
        self._ensure_indexes()

    def _validate_config(self) -> None:
        required_fields = {
            "host": self.config.host,
            "username": self.config.username,
            "password": self.config.password,
            "auth_source": self.config.auth_source,
            "database": self.config.database,
            "memory_collection": self.config.memory_collection,
        }
        missing = [name for name, value in required_fields.items() if not value]
        if missing:
            raise ValueError(f"MongoDB 配置缺失: {', '.join(missing)}")
        if self.config.port <= 0:
            raise ValueError("MONGODB_PORT 必须大于 0")

    def _ensure_indexes(self) -> None:
        """创建会话时间索引和消息幂等索引。"""
        self.collection.create_index(
            [("session_id", ASCENDING), ("created_at", DESCENDING)],
            name="session_created_at_idx",
        )
        self.collection.create_index(
            [("session_id", ASCENDING), ("message_id", ASCENDING)],
            name="session_message_id_unique_idx",
            unique=True,
            partialFilterExpression={"message_id": {"$type": "string"}},
        )

    def ping(self) -> bool:
        """验证 MongoDB 连接和认证是否正常。"""
        return bool(self.client.admin.command("ping").get("ok"))

    def save_memory(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        message_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """保存一条消息；传入 message_id 时按会话执行幂等更新。"""
        session_id = session_id.strip()
        role = role.strip()
        content = content.strip()
        message_id = message_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        if not role:
            raise ValueError("role 不能为空")
        if not content:
            raise ValueError("content 不能为空")

        now = datetime.now(timezone.utc)
        memory = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "updated_at": now,
        }
        metadata = dict(metadata or {})

        if not message_id:
            memory["metadata"] = metadata
            memory["created_at"] = now
            return str(self.collection.insert_one(memory).inserted_id)

        memory["message_id"] = message_id
        for key, value in metadata.items():
            memory[f"metadata.{key}"] = value
        query = {"session_id": session_id, "message_id": message_id}
        set_on_insert = {"created_at": now}
        if not metadata:
            set_on_insert["metadata"] = {}
        result = self.collection.update_one(
            query,
            {
                "$set": memory,
                "$setOnInsert": set_on_insert,
            },
            upsert=True,
        )
        if result.upserted_id is not None:
            return str(result.upserted_id)

        existing = self.collection.find_one(query, {"_id": 1})
        if existing is None:
            raise RuntimeError("MongoDB 已更新消息，但未能读取消息 ID")
        return str(existing["_id"])

    def get_memories(
        self,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """读取会话最近的记忆，并按创建时间从早到晚返回。"""
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须大于 0")

        cursor = (
            self.collection.find({"session_id": session_id})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        memories = list(cursor)
        memories.reverse()
        for memory in memories:
            memory["_id"] = str(memory["_id"])
        return memories

    def delete_memory(self, *, session_id: str, message_id: str) -> None:
        """按会话和消息 ID 删除一条幂等消息；消息不存在时不报错。"""
        session_id = session_id.strip()
        message_id = message_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        if not message_id:
            raise ValueError("message_id 不能为空")
        self.collection.delete_one(
            {"session_id": session_id, "message_id": message_id}
        )

    def update_message_item_names(
        self,
        *,
        session_id: str,
        message_id: str,
        item_names: list[str],
        rewritten_query: str,
        status: str,
        candidates: list[str] | None = None,
    ) -> None:
        """更新消息的商品识别结果，同时保留已有的其他元数据。"""
        session_id = session_id.strip()
        message_id = message_id.strip()
        rewritten_query = rewritten_query.strip()
        status = status.strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        if not message_id:
            raise ValueError("message_id 不能为空")
        if not rewritten_query:
            raise ValueError("rewritten_query 不能为空")
        if not status:
            raise ValueError("status 不能为空")

        now = datetime.now(timezone.utc)
        result = self.collection.update_one(
            {"session_id": session_id, "message_id": message_id},
            {
                "$set": {
                    "metadata.item_names": list(item_names),
                    "metadata.rewritten_query": rewritten_query,
                    "metadata.item_name_status": status,
                    "metadata.item_name_candidates": list(candidates or []),
                    "updated_at": now,
                }
            },
        )
        if result.matched_count == 0:
            raise RuntimeError(
                f"MongoDB 中不存在消息: {session_id}/{message_id}"
            )

    def clear_session(self, session_id: str) -> int:
        """删除一个会话的全部记忆，返回删除数量。"""
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        result = self.collection.delete_many({"session_id": session_id})
        return result.deleted_count

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "MongoDBUtil":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


_mongodb_util: MongoDBUtil | None = None


def get_mongodb_util() -> MongoDBUtil:
    """获取进程内复用的 MongoDB 记忆工具。"""
    global _mongodb_util
    if _mongodb_util is None:
        _mongodb_util = MongoDBUtil()
    return _mongodb_util
