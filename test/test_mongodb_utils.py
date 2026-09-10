import os
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from config.mongodb_config import mongodb_config
from utils.mongodb_utils import MongoDBUtil


class FakeCollection:
    def __init__(self, matched_count=1):
        self.matched_count = matched_count
        self.update_call = None

    def update_one(self, query, update, upsert=False):
        self.update_call = (query, update)
        return SimpleNamespace(
            matched_count=self.matched_count,
            upserted_id=None,
        )

    @staticmethod
    def find_one(query, projection):
        return {"_id": "stored-message"}


def test_update_message_item_names_preserves_other_metadata():
    util = MongoDBUtil.__new__(MongoDBUtil)
    util.collection = FakeCollection()

    util.update_message_item_names(
        session_id="session-1",
        message_id="message-1",
        item_names=["HAK180 安全栅"],
        rewritten_query="HAK180 安全栅怎么设置？",
        status="confirmed",
        candidates=[],
    )

    query, update = util.collection.update_call
    assert query == {"session_id": "session-1", "message_id": "message-1"}
    assert update["$set"]["metadata.item_names"] == ["HAK180 安全栅"]
    assert update["$set"]["metadata.rewritten_query"] == "HAK180 安全栅怎么设置？"
    assert update["$set"]["metadata.item_name_status"] == "confirmed"
    assert update["$set"]["metadata.item_name_candidates"] == []
    assert "metadata" not in update["$set"]


def test_update_message_item_names_rejects_missing_target_message():
    util = MongoDBUtil.__new__(MongoDBUtil)
    util.collection = FakeCollection(matched_count=0)

    with pytest.raises(RuntimeError, match="不存在"):
        util.update_message_item_names(
            session_id="session-1",
            message_id="missing-message",
            item_names=[],
            rewritten_query="问题",
            status="not_found",
        )


def test_idempotent_save_preserves_unrelated_metadata_fields():
    util = MongoDBUtil.__new__(MongoDBUtil)
    util.collection = FakeCollection()

    util.save_memory(
        session_id="session-1",
        message_id="message-1",
        role="user",
        content="HAK180 怎么设置？",
        metadata={"item_name_status": "processing"},
    )

    _, update = util.collection.update_call
    assert update["$set"]["metadata.item_name_status"] == "processing"
    assert "metadata" not in update["$set"]


@pytest.mark.skipif(
    os.getenv("RUN_MONGODB_INTEGRATION_TESTS") != "1",
    reason="需要显式启用真实 MongoDB 集成测试",
)
class TestMongoDBUtil:
    """使用真实 MongoDB 验证会话记忆的写入、读取和清理。"""

    def test_memory_lifecycle(self):
        test_id = uuid4().hex
        config = replace(
            mongodb_config,
            memory_collection=f"test_memories_{test_id}",
        )
        util = MongoDBUtil(config)
        session_id = f"test_session_{test_id}"

        try:
            assert util.ping() is True

            user_id = util.save_memory(
                session_id=session_id,
                message_id="message-1",
                role="user",
                content="HAK180 如何设置？",
                metadata={"source": "integration_test"},
            )
            assistant_id = util.save_memory(
                session_id=session_id,
                message_id="message-2",
                role="assistant",
                content="请先确认需要设置的具体参数。",
            )
            updated_user_id = util.save_memory(
                session_id=session_id,
                message_id="message-1",
                role="user",
                content="HAK180 如何设置转印温度？",
            )

            memories = util.get_memories(session_id)

            assert updated_user_id == user_id
            assert len(memories) == 2
            assert [memory["role"] for memory in memories] == [
                "user",
                "assistant",
            ]
            assert memories[0]["content"] == "HAK180 如何设置转印温度？"
            assert memories[0]["created_at"].tzinfo is not None
            assert memories[1]["_id"] == assistant_id
            assert util.clear_session(session_id) == 2
            assert util.get_memories(session_id) == []
        finally:
            util.database.drop_collection(config.memory_collection)
            util.close()
