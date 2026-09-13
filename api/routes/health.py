"""服务健康检查接口。"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="检查 API 进程是否存活")
def health() -> dict[str, str]:
    return {"status": "ok"}
