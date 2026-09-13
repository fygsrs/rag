"""HTTP 请求与响应的数据模型。"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """拒绝未声明字段，及时发现前后端字段拼写错误。"""

    model_config = ConfigDict(extra="forbid")


class ImportResponse(StrictModel):
    task_id: str
    file_name: str
    file_title: str
    item_name: str
    chunk_count: int
    md_path: str


class ImportTaskResponse(StrictModel):
    task_id: str
    file_name: str
    status: Literal["queued", "running", "succeeded", "failed"]
    current_stage: str
    progress: int = Field(ge=0, le=100)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str = ""
    revision: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LoginRequest(StrictModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=5, max_length=128)


class UserResponse(StrictModel):
    id: str
    username: str
    created_at: datetime | None = None


class QueryRequest(StrictModel):
    query: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=200)
    message_id: str | None = Field(default=None, max_length=200)
    task_id: str | None = Field(default=None, max_length=200)
    item_names: list[str] = Field(default_factory=list, max_length=20)
    mode: Literal["fast", "deep"] = "fast"

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value

    @field_validator("session_id", "message_id", "task_id")
    @classmethod
    def normalize_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("item_names")
    @classmethod
    def normalize_item_names(cls, values: list[str]) -> list[str]:
        result = []
        seen = set()
        for value in values:
            name = value.strip()
            if name and name not in seen:
                seen.add(name)
                result.append(name)
        return result


class SourceResponse(StrictModel):
    index: int
    id: str | int | None = None
    title: str = ""
    url: str = ""
    score: float | None = None
    source: str = ""
    source_type: str = "internal"
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(StrictModel):
    task_id: str
    session_id: str
    message_id: str
    item_names: list[str]
    rewritten_query: str
    answer: str
    answer_html: str
    sources: list[SourceResponse]


class SessionSummary(StrictModel):
    session_id: str
    title: str
    last_message: str
    updated_at: datetime | None = None
    message_count: int


class SessionMessage(StrictModel):
    message_id: str = ""
    role: str
    content: str
    content_html: str
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMessagesResponse(StrictModel):
    session_id: str
    messages: list[SessionMessage]
