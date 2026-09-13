"""文档导入、持久化任务和 SSE 进度接口。"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.dependencies import CurrentUser
from api.schemas import ImportResponse, ImportTaskResponse
from api.sse import encode_sse
from processor.import_processor.main_graph import ImportWorkflow
from processor.import_processor.state import create_default_state
from utils.import_task_utils import get_import_task_util

router = APIRouter(prefix="/imports", tags=["imports"])
logger = logging.getLogger(__name__)
_ALLOWED_EXTENSIONS = {".pdf", ".md"}
_COPY_CHUNK_SIZE = 1024 * 1024


def _upload_root() -> Path:
    configured = os.getenv("API_UPLOAD_DIR", ".runtime/uploads")
    path = Path(configured)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _save_upload(upload: UploadFile) -> Path:
    filename = Path(upload.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="仅支持 PDF 和 Markdown 文件",
        )

    max_bytes = int(os.getenv("API_MAX_UPLOAD_MB", "100")) * 1024 * 1024
    target_dir = _upload_root() / uuid4().hex
    target_dir.mkdir(parents=True, exist_ok=False)
    target_path = target_dir / filename
    written = 0

    try:
        with target_path.open("wb") as output:
            while chunk := await upload.read(_COPY_CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"文件不能超过 {max_bytes // 1024 // 1024} MB",
                    )
                output.write(chunk)
    except Exception:
        target_path.unlink(missing_ok=True)
        try:
            target_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        await upload.close()
    return target_path


@router.post(
    "",
    response_model=ImportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传并导入一个知识库文档",
)
async def import_document(
    current_user: CurrentUser,
    file: UploadFile = File(...),
    task_id: str | None = Form(default=None),
) -> ImportResponse:
    saved_path = await _save_upload(file)
    normalized_task_id = (task_id or "").strip() or f"import-{uuid4().hex}"
    state = create_default_state(
        task_id=normalized_task_id,
        import_file_path=str(saved_path),
    )

    try:
        result = await run_in_threadpool(ImportWorkflow().run, state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档导入失败: {exc}",
        ) from exc

    return ImportResponse(
        task_id=str(result.get("task_id") or normalized_task_id),
        file_name=saved_path.name,
        file_title=str(result.get("file_title") or saved_path.stem),
        item_name=str(result.get("item_name") or ""),
        chunk_count=len(result.get("chunks") or []),
        md_path=str(result.get("md_path") or ""),
    )


def _run_import_task(
    *,
    task_id: str,
    user_id: str,
    saved_path: Path,
) -> None:
    """在线程池中运行真实导入图，并把节点进度写入 MongoDB。"""
    tasks = get_import_task_util()
    state = create_default_state(
        task_id=task_id,
        import_file_path=str(saved_path),
    )
    merged_state: dict[str, Any] = dict(state)
    is_pdf = saved_path.suffix.lower() == ".pdf"
    try:
        if not is_pdf:
            tasks.skip_stage(task_id=task_id, user_id=user_id, stage="pdf_parse")
            tasks.set_stage(
                task_id=task_id,
                user_id=user_id,
                stage="image_processing",
                progress=10,
            )
        else:
            tasks.set_stage(
                task_id=task_id,
                user_id=user_id,
                stage="pdf_parse",
                progress=5,
            )

        for update in ImportWorkflow().stream_updates(state):
            for node_name, values in update.items():
                if isinstance(values, dict):
                    merged_state.update(values)
                if node_name == "b_node_pdf_to_md":
                    tasks.set_stage(
                        task_id=task_id,
                        user_id=user_id,
                        stage="image_processing",
                        progress=35,
                    )
                elif node_name == "c_node_md_img":
                    tasks.set_stage(
                        task_id=task_id,
                        user_id=user_id,
                        stage="splitting",
                        progress=55,
                    )
                elif node_name == "e_node_item_name_recognition":
                    tasks.set_stage(
                        task_id=task_id,
                        user_id=user_id,
                        stage="embedding",
                        progress=72,
                    )
                elif node_name == "f_node_embedding":
                    tasks.set_stage(
                        task_id=task_id,
                        user_id=user_id,
                        stage="milvus_import",
                        progress=90,
                    )

        result = {
            "file_name": saved_path.name,
            "file_title": str(merged_state.get("file_title") or saved_path.stem),
            "item_name": str(merged_state.get("item_name") or ""),
            "chunk_count": len(merged_state.get("chunks") or []),
            "md_path": str(merged_state.get("md_path") or ""),
        }
        tasks.succeed(task_id=task_id, user_id=user_id, result=result)
    except Exception as exc:
        logger.exception("文档导入任务失败 | task_id=%s", task_id)
        tasks.fail(task_id=task_id, user_id=user_id, error=str(exc))


@router.post(
    "/tasks",
    response_model=ImportTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="创建后台文档导入任务",
)
async def create_import_task(
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    file: UploadFile = File(...),
) -> ImportTaskResponse:
    saved_path = await _save_upload(file)
    task_id = f"import-{uuid4().hex}"
    user_id = str(current_user["id"])
    task = await run_in_threadpool(
        get_import_task_util().create,
        task_id=task_id,
        user_id=user_id,
        file_name=saved_path.name,
    )
    background_tasks.add_task(
        _run_import_task,
        task_id=task_id,
        user_id=user_id,
        saved_path=saved_path,
    )
    return ImportTaskResponse.model_validate(task)


@router.get(
    "/tasks",
    response_model=list[ImportTaskResponse],
    summary="读取当前用户最近的导入任务",
)
async def list_import_tasks(
    current_user: CurrentUser,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ImportTaskResponse]:
    tasks = await run_in_threadpool(
        get_import_task_util().list,
        user_id=str(current_user["id"]),
        limit=limit,
    )
    return [ImportTaskResponse.model_validate(task) for task in tasks]


@router.get(
    "/tasks/{task_id}",
    response_model=ImportTaskResponse,
    summary="读取导入任务进度",
)
async def get_import_task(
    task_id: str,
    current_user: CurrentUser,
) -> ImportTaskResponse:
    task = await run_in_threadpool(
        get_import_task_util().get,
        task_id=task_id,
        user_id=str(current_user["id"]),
    )
    if task is None:
        raise HTTPException(status_code=404, detail="导入任务不存在")
    return ImportTaskResponse.model_validate(task)


@router.get(
    "/tasks/{task_id}/events",
    response_class=StreamingResponse,
    summary="订阅可恢复的导入任务 SSE 进度",
)
async def stream_import_task(
    task_id: str,
    current_user: CurrentUser,
    after_revision: int = Query(default=0, ge=0),
) -> StreamingResponse:
    user_id = str(current_user["id"])
    if await run_in_threadpool(
        get_import_task_util().get,
        task_id=task_id,
        user_id=user_id,
    ) is None:
        raise HTTPException(status_code=404, detail="导入任务不存在")

    async def events():
        revision = after_revision
        while True:
            task = await run_in_threadpool(
                get_import_task_util().get,
                task_id=task_id,
                user_id=user_id,
            )
            if task is None:
                yield encode_sse(
                    {"event": "error", "data": {"message": "导入任务不存在"}}
                )
                return
            current_revision = int(task.get("revision") or 0)
            if current_revision > revision:
                revision = current_revision
                yield encode_sse(
                    {"event": "progress", "id": revision, "data": task}
                )
            if task.get("status") in {"succeeded", "failed"}:
                yield encode_sse(
                    {"event": "done", "id": revision, "data": task}
                )
                return
            await asyncio.sleep(0.8)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
