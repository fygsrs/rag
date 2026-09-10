import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from processor.query_processor.nodes.a_node_item_name_confirm import (
    NodeItemNameConfirm,
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        content = self.payload
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return SimpleNamespace(content=content)


class FakeEmbeddingTool:
    def __init__(self):
        self.calls = []

    def embed_dense_and_sparse(self, texts, *, text_type="document"):
        self.calls.append((list(texts), text_type))
        dense = [[float(index + 1), 0.0, 0.0] for index in range(len(texts))]
        sparse = [{index + 1: 1.0} for index in range(len(texts))]
        return dense, sparse


class FakeMilvusClient:
    def __init__(self, results, *, collection_exists=True):
        self.results = results
        self.collection_exists = collection_exists
        self.search_kwargs = None

    def has_collection(self, collection_name):
        assert collection_name == "kb_item_names"
        return self.collection_exists

    def hybrid_search(self, **kwargs):
        self.search_kwargs = kwargs
        return deepcopy(self.results)


class FakeMongoDBUtil:
    def __init__(self, memories=None):
        self.memories = deepcopy(memories or [])

    def save_memory(
        self,
        *,
        session_id,
        role,
        content,
        message_id="",
        metadata=None,
    ):
        for memory in self.memories:
            if (
                memory["session_id"] == session_id
                and memory.get("message_id") == message_id
            ):
                memory.update(
                    role=role,
                    content=content,
                    metadata=dict(metadata or {}),
                )
                return message_id
        self.memories.append(
            {
                "_id": message_id,
                "session_id": session_id,
                "message_id": message_id,
                "role": role,
                "content": content,
                "metadata": dict(metadata or {}),
            }
        )
        return message_id

    def update_message_item_names(
        self,
        *,
        session_id,
        message_id,
        item_names,
        rewritten_query,
        status,
        candidates=None,
    ):
        for memory in self.memories:
            if (
                memory["session_id"] == session_id
                and memory.get("message_id") == message_id
            ):
                memory["metadata"].update(
                    item_names=list(item_names),
                    rewritten_query=rewritten_query,
                    item_name_status=status,
                    item_name_candidates=list(candidates or []),
                )
                return
        raise RuntimeError("目标消息不存在")

    def get_memories(self, session_id, *, limit=20):
        return deepcopy(
            [
                memory
                for memory in self.memories
                if memory["session_id"] == session_id
            ][-limit:]
        )

    def delete_memory(self, *, session_id, message_id):
        self.memories = [
            memory
            for memory in self.memories
            if not (
                memory["session_id"] == session_id
                and memory.get("message_id") == message_id
            )
        ]


def make_config(**overrides):
    values = {
        "item_model": "item-model",
        "milvus_url": "http://milvus.test:19530",
        "item_name_collection": "kb_item_names",
        "history_limit": 20,
        "item_name_top_k": 5,
        "item_name_confirm_threshold": 0.85,
        "item_name_candidate_threshold": 0.60,
        "item_name_score_margin": 0.15,
        "item_name_dense_weight": 0.7,
        "item_name_sparse_weight": 0.3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_state(**overrides):
    state = {
        "session_id": "session-1",
        "message_id": "message-1",
        "original_query": "HAK180 怎么设置转印温度？",
    }
    state.update(overrides)
    return state


def make_result(item_name, score):
    return {
        "id": f"id-{item_name}",
        "distance": score,
        "entity": {
            "item_name": item_name,
            "file_title": f"{item_name} 产品手册",
        },
    }


def make_node(llm_payload, results, *, memories=None, collection_exists=True):
    mongodb = FakeMongoDBUtil(memories)
    embedding = FakeEmbeddingTool()
    milvus = FakeMilvusClient(
        results,
        collection_exists=collection_exists,
    )
    node = NodeItemNameConfirm(
        make_config(),
        llm_client=FakeLLM(llm_payload),
        embedding_tool=embedding,
        milvus_client=milvus,
        mongodb_util=mongodb,
    )
    return node, mongodb, embedding, milvus


@pytest.mark.parametrize("field", ["session_id", "message_id", "original_query"])
def test_process_rejects_missing_required_fields(field):
    node, _, _, _ = make_node(
        {"item_names": ["HAK180"], "rewritten_query": "如何设置温度？"},
        [],
    )
    state = make_state()
    state.pop(field)

    with pytest.raises(ValueError, match=field):
        node.process(state)


@pytest.mark.parametrize(
    "field, expected_name",
    [
        ("item_model", "ITEM_MODEL"),
        ("milvus_url", "MILVUS_URL"),
        ("item_name_collection", "ITEM_NAME_COLLECTION"),
    ],
)
def test_process_rejects_missing_required_query_config(field, expected_name):
    mongodb = FakeMongoDBUtil()
    config = make_config(**{field: ""})
    node = NodeItemNameConfirm(
        config,
        llm_client=FakeLLM(
            {"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么设置？"}
        ),
        embedding_tool=FakeEmbeddingTool(),
        milvus_client=FakeMilvusClient([]),
        mongodb_util=mongodb,
    )

    with pytest.raises(RuntimeError, match=expected_name):
        node.process(make_state())

    assert mongodb.memories == []


@pytest.mark.parametrize(
    "overrides, expected_name",
    [
        ({"history_limit": 0}, "QUERY_HISTORY_LIMIT"),
        ({"item_name_top_k": 0}, "ITEM_NAME_TOP_K"),
        ({"item_name_candidate_threshold": 1.1}, "ITEM_NAME_CANDIDATE_THRESHOLD"),
        (
            {
                "item_name_candidate_threshold": 0.9,
                "item_name_confirm_threshold": 0.8,
            },
            "ITEM_NAME_CANDIDATE_THRESHOLD",
        ),
        ({"item_name_dense_weight": -0.1}, "ITEM_NAME_DENSE_WEIGHT"),
    ],
)
def test_process_rejects_invalid_numeric_query_config(overrides, expected_name):
    node, _, _, _ = make_node(
        {"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么设置？"},
        [],
    )
    node.config = make_config(**overrides)

    with pytest.raises(RuntimeError, match=expected_name):
        node.process(make_state())


def test_process_confirms_product_and_updates_user_memory():
    node, mongodb, embedding, milvus = make_node(
        "```json\n"
        '{"item_names": ["HAK180", "HAK180 ", ""], '
        '"rewritten_query": "怎么设置转印温度？"}\n'
        "```",
        [
            [
                make_result("HAK180 安全栅", 0.95),
                make_result("HAK180A 安全栅", 0.70),
            ]
        ],
        memories=[
            {
                "_id": "old-message",
                "session_id": "session-1",
                "message_id": "old-message",
                "role": "assistant",
                "content": "之前的回答",
                "metadata": {"item_names": ["旧商品"]},
            }
        ],
    )

    result = node.process(make_state())

    assert result["item_names"] == ["HAK180 安全栅"]
    assert "HAK180 安全栅" in result["rewritten_query"]
    assert result["answer"] == ""
    assert embedding.calls == [(["HAK180"], "query")]
    assert milvus.search_kwargs["collection_name"] == "kb_item_names"
    assert [memory["role"] for memory in result["history"]] == [
        "assistant",
        "user",
    ]
    user_memory = mongodb.memories[-1]
    assert user_memory["metadata"]["item_names"] == ["HAK180 安全栅"]
    assert user_memory["metadata"]["item_name_status"] == "confirmed"


def test_process_confirms_at_exact_score_and_margin_boundaries():
    node, _, _, _ = make_node(
        {"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么设置？"},
        [
            [
                make_result("HAK180 安全栅", 0.85),
                make_result("HAK180A 安全栅", 0.70),
            ]
        ],
    )

    result = node.process(make_state())

    assert result["item_names"] == ["HAK180 安全栅"]
    assert result["answer"] == ""


def test_process_returns_candidates_and_saves_idempotent_assistant_message():
    node, mongodb, _, _ = make_node(
        {"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么设置？"},
        [
            [
                make_result("HAK180 安全栅", 0.84),
                make_result("HAK180A 安全栅", 0.75),
            ]
        ],
    )

    first = node.process(make_state())
    second = node.process(make_state())

    assert first["item_names"] == []
    assert "1. HAK180 安全栅" in first["answer"]
    assert "2. HAK180A 安全栅" in first["answer"]
    assert second["answer"] == first["answer"]
    assert [m["message_id"] for m in mongodb.memories] == [
        "message-1",
        "message-1:item-name-confirm",
    ]
    assert mongodb.memories[0]["metadata"]["item_name_status"] == "needs_confirmation"


def test_confirmed_retry_removes_previous_confirmation_question():
    mongodb = FakeMongoDBUtil()
    ambiguous_node, _, _, _ = make_node(
        {"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么设置？"},
        [[make_result("HAK180 安全栅", 0.84)]],
    )
    ambiguous_node._mongodb_util = mongodb
    ambiguous_node.process(make_state())

    confirmed_node, _, _, _ = make_node(
        {"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么设置？"},
        [[make_result("HAK180 安全栅", 0.95)]],
    )
    confirmed_node._mongodb_util = mongodb

    result = confirmed_node.process(make_state())

    assert result["answer"] == ""
    assert [memory["message_id"] for memory in result["history"]] == ["message-1"]


def test_process_returns_not_found_when_no_candidate_reaches_threshold():
    node, mongodb, _, _ = make_node(
        {"item_names": ["未知型号"], "rewritten_query": "未知型号怎么设置？"},
        [[make_result("HAK180 安全栅", 0.59)]],
    )

    result = node.process(make_state(original_query="未知型号怎么设置？"))

    assert result["item_names"] == []
    assert "品牌、系列或型号" in result["answer"]
    assert mongodb.memories[0]["metadata"]["item_name_status"] == "not_found"
    assert result["history"][-1]["message_id"] == "message-1:item-name-confirm"


def test_process_requires_every_product_to_be_confirmed():
    node, _, embedding, _ = make_node(
        {
            "item_names": ["HAK180", "HAK190"],
            "rewritten_query": "HAK180 和 HAK190 有什么区别？",
        },
        [
            [make_result("HAK180 安全栅", 0.96)],
            [
                make_result("HAK190 安全栅", 0.82),
                make_result("HAK190A 安全栅", 0.72),
            ],
        ],
    )

    result = node.process(
        make_state(original_query="HAK180 和 HAK190 有什么区别？")
    )

    assert result["item_names"] == []
    assert "HAK190 安全栅" in result["answer"]
    assert embedding.calls == [(["HAK180", "HAK190"], "query")]


def test_process_rejects_invalid_llm_json():
    node, _, _, _ = make_node("HAK180", [])

    with pytest.raises(RuntimeError, match="JSON"):
        node.process(make_state())


def test_process_treats_missing_collection_as_infrastructure_error():
    node, _, _, _ = make_node(
        {"item_names": ["HAK180"], "rewritten_query": "怎么设置？"},
        [],
        collection_exists=False,
    )

    with pytest.raises(RuntimeError, match="Collection"):
        node.process(make_state())
