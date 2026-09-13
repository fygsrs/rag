"""RAG 服务的 FastAPI 应用入口。"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.auth import router as auth_router
from api.routes.imports import router as import_router
from api.routes.queries import router as query_router
from api.routes.sessions import router as session_router


def create_app() -> FastAPI:
    """创建 FastAPI 应用，便于 Uvicorn 启动和应用级测试。"""
    app = FastAPI(
        title="RAG Knowledge Base API",
        description="文档导入、知识库查询与流式回答接口",
        version="0.1.0",
    )

    origins = [
        origin.strip()
        for origin in os.getenv(
            "API_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(import_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(session_router, prefix="/api/v1")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
