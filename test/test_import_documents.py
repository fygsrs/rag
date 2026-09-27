from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_current_user
from api.routes import imports as imports_route
from utils.import_task_utils import ImportTaskUtil


class FakeCursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, field, direction):
        reverse = direction < 0
        self.documents.sort(
            key=lambda item: item.get(field) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=reverse,
        )
        return self

    def __iter__(self):
        return iter(self.documents)


class FakeCollection:
    def __init__(self, documents):
        self.documents = list(documents)
        self.last_find = None
        self.update_many_calls = []

    def find(self, query):
        self.last_find = query
        matched = [
            item
            for item in self.documents
            if item.get("user_id") == query["user_id"]
            and item.get("status") == query["status"]
            and "document_deleted_at" not in item
        ]
        return FakeCursor(matched)

    def update_many(self, query, update):
        self.update_many_calls.append((query, update))
        matched = 0
        for item in self.documents:
            if (
                item.get("user_id") == query["user_id"]
                and item.get("status") == query["status"]
                and (item.get("result") or {}).get("file_title")
                == query["result.file_title"]
            ):
                item.update(update["$set"])
                item["revision"] = item.get("revision", 0) + update["$inc"]["revision"]
                matched += 1
        return SimpleNamespace(matched_count=matched)

    def update_one(self, query, update):
        return SimpleNamespace(matched_count=0)


def build_utility(documents):
    utility = ImportTaskUtil.__new__(ImportTaskUtil)
    utility.collection = FakeCollection(documents)
    return utility


def successful_task(task_id, user_id, title, updated_at, *, deleted=False):
    task = {
        "task_id": task_id,
        "user_id": user_id,
        "file_name": f"{title}.pdf",
        "status": "succeeded",
        "updated_at": updated_at,
        "revision": 1,
        "result": {
            "file_name": f"{title}.pdf",
            "file_title": title,
            "item_name": "测试商品",
            "chunk_count": 12,
        },
    }
    if deleted:
        task["document_deleted_at"] = updated_at
    return task


def test_list_documents_is_scoped_deduplicated_and_excludes_deleted():
    now = datetime.now(timezone.utc)
    utility = build_utility(
        [
            successful_task("latest", "user-1", "文档 A", now),
            successful_task("older", "user-1", "文档 A", now - timedelta(days=1)),
            successful_task("deleted", "user-1", "文档 B", now, deleted=True),
            successful_task("other-user", "user-2", "文档 C", now),
            {"task_id": "failed", "user_id": "user-1", "status": "failed"},
        ]
    )

    documents = utility.list_documents(user_id="user-1", limit=50)

    assert [document["task_id"] for document in documents] == ["latest"]
    assert utility.collection.last_find == {
        "user_id": "user-1",
        "status": "succeeded",
        "document_deleted_at": {"$exists": False},
    }


def test_mark_document_deleted_preserves_successful_history():
    now = datetime.now(timezone.utc)
    utility = build_utility(
        [
            successful_task("latest", "user-1", "文档 A", now),
            successful_task("older", "user-1", "文档 A", now - timedelta(days=1)),
        ]
    )

    marked = utility.mark_document_deleted(
        task_id="latest", user_id="user-1", file_title="文档 A"
    )

    assert marked is True
    assert all(task["status"] == "succeeded" for task in utility.collection.documents)
    assert all("document_deleted_at" in task for task in utility.collection.documents)


class FakeTaskUtil:
    def __init__(self, task):
        self.task = task
        self.marked = None

    def get(self, *, task_id, user_id):
        if task_id == self.task["task_id"] and user_id == self.task["user_id"]:
            return {key: value for key, value in self.task.items() if key != "user_id"}
        return None

    def list_documents(self, *, user_id, limit):
        if user_id != self.task["user_id"]:
            return []
        result = self.task["result"]
        return [
            {
                "task_id": self.task["task_id"],
                "file_name": result["file_name"],
                "file_title": result["file_title"],
                "item_name": result["item_name"],
                "chunk_count": result["chunk_count"],
                "imported_at": self.task["updated_at"],
            }
        ][:limit]

    def mark_document_deleted(self, *, task_id, user_id, file_title):
        self.marked = (task_id, user_id, file_title)
        return True


def test_delete_document_route_removes_chunks_and_keeps_history(monkeypatch):
    task = successful_task(
        "task-1", "user-1", "文档 A", datetime.now(timezone.utc)
    )
    tasks = FakeTaskUtil(task)
    deleted_titles = []

    class FakeImportNode:
        def delete_document_chunks(self, file_title):
            deleted_titles.append(file_title)
            return 12

    monkeypatch.setattr(imports_route, "get_import_task_util", lambda: tasks)
    monkeypatch.setattr(imports_route, "NodeImportMilvus", FakeImportNode)
    app = FastAPI()
    app.include_router(imports_route.router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: {"id": "user-1"}

    response = TestClient(app).delete("/api/v1/imports/documents/task-1")

    assert response.status_code == 204
    assert deleted_titles == ["文档 A"]
    assert tasks.marked == ("task-1", "user-1", "文档 A")


def test_list_documents_route_returns_typed_document(monkeypatch):
    task = successful_task(
        "task-1", "user-1", "文档 A", datetime.now(timezone.utc)
    )
    tasks = FakeTaskUtil(task)
    monkeypatch.setattr(imports_route, "get_import_task_util", lambda: tasks)
    app = FastAPI()
    app.include_router(imports_route.router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: {"id": "user-1"}

    response = TestClient(app).get("/api/v1/imports/documents?limit=1")

    assert response.status_code == 200
    assert response.json()[0]["file_title"] == "文档 A"
    assert response.json()[0]["chunk_count"] == 12
    assert "user_id" not in response.json()[0]
