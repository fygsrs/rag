"""MongoDB 持久化导入任务及其阶段进度。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING

from utils.mongodb_utils import get_mongodb_util


IMPORT_STAGES = [
    ("pdf_parse", "解析 PDF"),
    ("image_processing", "处理图片"),
    ("splitting", "文档切片"),
    ("embedding", "生成向量"),
    ("milvus_import", "写入知识库"),
]


class ImportTaskUtil:
    """提供可跨页面刷新读取的导入任务状态。"""

    def __init__(self) -> None:
        mongodb = get_mongodb_util()
        self.collection = mongodb.database[
            mongodb.config.import_task_collection
        ]
        self.collection.create_index(
            [("task_id", ASCENDING)], name="task_id_unique_idx", unique=True
        )
        self.collection.create_index(
            [("user_id", ASCENDING), ("created_at", DESCENDING)],
            name="user_created_at_idx",
        )

    def create(self, *, task_id: str, user_id: str, file_name: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        document = {
            "task_id": task_id,
            "user_id": user_id,
            "file_name": file_name,
            "status": "queued",
            "current_stage": "queued",
            "progress": 0,
            "stages": [
                {"key": key, "label": label, "status": "pending"}
                for key, label in IMPORT_STAGES
            ],
            "result": None,
            "error": "",
            "revision": 1,
            "created_at": now,
            "updated_at": now,
        }
        self.collection.insert_one(document)
        return self._public(document)

    def get(self, *, task_id: str, user_id: str) -> dict[str, Any] | None:
        document = self.collection.find_one(
            {"task_id": task_id, "user_id": user_id}
        )
        return self._public(document) if document else None

    def list(self, *, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        cursor = self.collection.find({"user_id": user_id}).sort(
            "created_at", DESCENDING
        ).limit(limit)
        return [self._public(document) for document in cursor]

    def list_documents(
        self, *, user_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """返回当前仍在知识库中的文档，每个标题只保留最近成功版本。"""
        if not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")

        cursor = self.collection.find(
            {
                "user_id": user_id,
                "status": "succeeded",
                "document_deleted_at": {"$exists": False},
            }
        ).sort("updated_at", DESCENDING)
        documents: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for task in cursor:
            result = task.get("result")
            if not isinstance(result, dict):
                continue
            file_name = str(result.get("file_name") or task.get("file_name") or "")
            file_title = str(result.get("file_title") or "").strip()
            if not file_title:
                file_title = file_name.rsplit(".", 1)[0].strip()
            normalized_title = file_title
            if not normalized_title or normalized_title in seen_titles:
                continue

            chunk_count = result.get("chunk_count", 0)
            if not isinstance(chunk_count, int) or isinstance(chunk_count, bool):
                chunk_count = 0
            documents.append(
                {
                    "task_id": str(task.get("task_id") or ""),
                    "file_name": file_name,
                    "file_title": file_title,
                    "item_name": str(result.get("item_name") or ""),
                    "chunk_count": max(0, chunk_count),
                    "imported_at": task.get("updated_at") or task.get("created_at"),
                }
            )
            seen_titles.add(normalized_title)
            if len(documents) >= limit:
                break
        return documents

    def mark_document_deleted(
        self,
        *,
        task_id: str,
        user_id: str,
        file_title: str,
    ) -> bool:
        """保留导入历史，并标记同一文档的历史成功版本已从知识库删除。"""
        now = datetime.now(timezone.utc)
        result = self.collection.update_many(
            {
                "user_id": user_id,
                "status": "succeeded",
                "result.file_title": file_title,
            },
            {
                "$set": {
                    "document_deleted_at": now,
                    "updated_at": now,
                },
                "$inc": {"revision": 1},
            },
        )
        if result.matched_count:
            return True

        fallback = self.collection.update_one(
            {
                "task_id": task_id,
                "user_id": user_id,
                "status": "succeeded",
            },
            {
                "$set": {
                    "document_deleted_at": now,
                    "updated_at": now,
                },
                "$inc": {"revision": 1},
            },
        )
        return bool(fallback.matched_count)

    def set_stage(
        self,
        *,
        task_id: str,
        user_id: str,
        stage: str,
        progress: int,
        status: str = "running",
    ) -> None:
        document = self.collection.find_one(
            {"task_id": task_id, "user_id": user_id}, {"stages": 1}
        )
        if document is None:
            raise RuntimeError(f"导入任务不存在: {task_id}")
        stages = list(document.get("stages") or [])
        reached = False
        for item in stages:
            if item.get("key") == stage:
                item["status"] = "running" if status == "running" else status
                reached = True
            elif not reached and item.get("status") != "skipped":
                item["status"] = "completed"
        self.collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {
                "$set": {
                    "status": "running",
                    "current_stage": stage,
                    "progress": max(0, min(progress, 99)),
                    "stages": stages,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$inc": {"revision": 1},
            },
        )

    def skip_stage(self, *, task_id: str, user_id: str, stage: str) -> None:
        document = self.collection.find_one(
            {"task_id": task_id, "user_id": user_id}, {"stages": 1}
        )
        if document is None:
            raise RuntimeError(f"导入任务不存在: {task_id}")
        stages = list(document.get("stages") or [])
        for item in stages:
            if item.get("key") == stage:
                item["status"] = "skipped"
        self.collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {
                "$set": {
                    "stages": stages,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$inc": {"revision": 1},
            },
        )

    def succeed(self, *, task_id: str, user_id: str, result: dict[str, Any]) -> None:
        document = self.collection.find_one(
            {"task_id": task_id, "user_id": user_id}, {"stages": 1}
        )
        if document is None:
            raise RuntimeError(f"导入任务不存在: {task_id}")
        stages = list(document.get("stages") or [])
        for item in stages:
            if item.get("status") != "skipped":
                item["status"] = "completed"
        self.collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {
                "$set": {
                    "status": "succeeded",
                    "current_stage": "completed",
                    "progress": 100,
                    "stages": stages,
                    "result": result,
                    "error": "",
                    "updated_at": datetime.now(timezone.utc),
                },
                "$inc": {"revision": 1},
            },
        )

    def fail(self, *, task_id: str, user_id: str, error: str) -> None:
        self.collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {
                "$set": {
                    "status": "failed",
                    "error": error,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$inc": {"revision": 1},
            },
        )

    @staticmethod
    def _public(document: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in document.items()
            if key not in {"_id", "user_id"}
        }


_import_task_util: ImportTaskUtil | None = None


def get_import_task_util() -> ImportTaskUtil:
    global _import_task_util
    if _import_task_util is None:
        _import_task_util = ImportTaskUtil()
    return _import_task_util
