# RAG 项目服务器部署手册

> 面向接手的 AI / 开发者：按本文档可以从零完成部署、更新和排障。
> 最后更新：2026-09-13（首次部署完成并验证通过）

## 1. 访问信息与部署位置（当前生产现状）

| 项目 | 值 |
|------|-----|
| 前端访问地址 | `http://203.195.206.242/` |
| API 前缀 | `http://203.195.206.242/api/v1/...`（nginx 反代到后端，后端 8000 不对外） |
| 图片外链 | `http://203.195.206.242:9000/rag/...`（MinIO，浏览器直接加载） |
| 登录账号 | `admin`（密码创建时指定，当前与前端预填一致） |
| 服务器 | 腾讯云 `203.195.206.242`，OpenCloudOS 9.6，x86_64，3.6G 内存 / 40G 磁盘 |
| 运行环境 | Docker 29.8.0 + Docker Compose v5.5.1 |
| 代码目录 | `/opt/rag` |
| 配置目录 | `/opt/rag/deploy/.env`（真实密钥，**不入库**） |
| Compose 工程 | 工程名 `rag`，入口 `/opt/rag/deploy/docker-compose.yml` |
| 数据卷 | `rag_rag-output`（MinerU 解析产物）、`rag_rag-runtime`（上传文件） |
| 应用容器 | `rag-backend-1`（uvicorn:8000，healthy）、`rag-frontend-1`（nginx:80） |

服务器上已有（不属于本仓库 compose，由运维维护）：

| 容器 | 镜像 | 端口/网络 |
|------|------|-----------|
| `mongodb` | mongo:8.0（有认证，`MONGO_INITDB_ROOT_*`） | 宿主机 27017 |
| `milvus-standalone` | milvusdb/milvus:v2.5.4 | 网络 `milvus`，19530 |
| `milvus-minio` | minio/minio（同时作为应用对象存储） | 网络 `milvus`，9000/9001 |
| `milvus-etcd` | etcd | 网络 `milvus` |
| `attu` | zilliz/attu（Milvus 控制台） | 宿主机 3000 |

## 2. 架构与网络关系

```text
浏览器 ──80──▶ frontend 容器(nignx)
                 ├─ 静态文件（Vue 构建产物）
                 └─ /api/ 反代 ──▶ backend 容器(uvicorn:8000)
                                      ├─ MongoDB    → host.docker.internal:27017
                                      ├─ MinIO(S3)  → milvus-minio:9000（docker 网络 milvus）
                                      ├─ Milvus     → milvus-standalone:19530（docker 网络 milvus）
                                      ├─ 图片 URL   → MINIO_PUBLIC_ENDPOINT（公网地址，浏览器用）
                                      └─ 公网 API   → DashScope / DeepSeek / MinerU
```

网络要点：

- backend 同时接入 compose 默认网络和外部网络 `milvus`（`docker-compose.yml` 中 `networks.milvus.external: true`），所以能用容器名访问 Milvus / MinIO
- Mongo 在默认 bridge 网络，用 `extra_hosts: host.docker.internal:host-gateway` 访问宿主机已发布端口
- 图片 URL 由后端拼装：`MINIO_ENDPOINT` 用于 S3 读写（内网），`MINIO_PUBLIC_ENDPOINT` 用于生成给浏览器的 URL（公网），代码见 `processor/import_processor/config.py::get_minio_base_url`

## 3. 仓库内相关文件

| 文件 | 作用 |
|------|------|
| `requirements-docker.txt` | 精简运行时依赖（**不要用 `requirements.txt`**，那是本地 conda 全量 freeze，含 torch 等 GB 级依赖，且缺少 httpx2/langchain-openai） |
| `Dockerfile` | 后端镜像（python:3.11-slim，非 root，健康检查） |
| `.dockerignore` | 构建上下文排除项 |
| `frontend/Dockerfile` | 前端多阶段构建（node:20 构建 → nginx:1.27 托管） |
| `frontend/nginx.conf` | 静态托管 + `/api` 反代 + SSE 不缓冲 + 100MB 上传限制 |
| `deploy/docker-compose.yml` | backend + frontend 编排 |
| `deploy/.env.production.example` | 服务器配置模板（全部 key 清单） |

## 4. 从零部署流程

### 4.1 前置条件

- 服务器已安装 Docker 与 Compose
- MongoDB / Milvus / MinIO 已就绪（见第 1 节），且 Mongo / MinIO 的账号密码可用
- 安全组放行：22、80；如浏览器要显示图片还需 9000（或后续用 nginx 反代 MinIO 后只放 443）
- 服务器 80 端口未被占用

### 4.2 本地打包

在项目根目录执行（Windows 用自带 `tar.exe` 同样可用）：

```bash
tar -czf rag-deploy.tar.gz \
  --exclude=node_modules --exclude=dist --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude=output --exclude=.runtime \
  --exclude=.playwright-cli --exclude='*.tsbuildinfo' \
  main.py api config processor utils scripts \
  requirements-docker.txt Dockerfile .dockerignore deploy frontend
```

注意：`.env` 不要打进包里（走单独通道，见 4.5）。

### 4.3 上传并解压

有 SSH 密钥时：

```bash
scp rag-deploy.tar.gz root@203.195.206.242:/tmp/
ssh root@203.195.206.242 'mkdir -p /opt/rag && tar -xzf /tmp/rag-deploy.tar.gz -C /opt/rag'
```

从 Windows 且只有密码时：`ssh/scp` 无法非交互输入密码，可用 Python `paramiko`（本机 base/rag conda 环境已装）做 SFTP 上传，参考临时脚本 `ssh_upload.py` 的模式（读环境变量中的账号密码，不落盘）。

### 4.4 生成 deploy/.env

方式 A（推荐，全新环境）：复制模板填写。

```bash
cd /opt/rag/deploy && cp .env.production.example .env && vi .env
```

必填 key：

| Key | 说明 |
|-----|------|
| `MONGODB_HOST/PORT/USERNAME/PASSWORD/AUTH_SOURCE` | 当前：`host.docker.internal` / `27017` / 容器内 `MONGO_INITDB_ROOT_*` / `admin` |
| `MILVUS_URL` | `http://milvus-standalone:19530` |
| `MINIO_ENDPOINT` | `milvus-minio:9000`（S3 读写） |
| `MINIO_PUBLIC_ENDPOINT` | `203.195.206.242:9000`（图片外链） |
| `MINIO_ACCESS_KEY/SECRET_KEY` | 取自 `docker inspect milvus-minio` 的 `MINIO_ACCESS_KEY/SECRET_KEY` |
| `MINIO_BUCKET_NAME` | `rag`（不存在时应用自动创建并设置匿名只读） |
| `LLM_BASE_URL/LLM_API_KEY` | 查询侧 LLM（当前 DeepSeek：`https://api.deepseek.com/v1`） |
| `LLM_DEFAULT_MODEL/ITEM_MODEL/ANSWER_MODEL` | `deepseek-flash` |
| `VL_BASE_URL/VL_API_KEY/VL_MODEL` | 视觉侧（DashScope + `qwen3.7-flash`），**不能换成 DeepSeek**（无视觉模型） |
| `DASHSCOPE_API_KEY` | Embedding / Rerank / WebSearch MCP |
| `DASHSCOPE_RERANK_URL` | 含业务空间 ID，从原环境迁移 |
| `MINERU_API_TOKEN` | PDF 解析 |
| `AUTH_SECRET_KEY` | 随机 ≥32 字符，`python3 -c "import secrets;print(secrets.token_hex(32))"` |
| `AUTH_COOKIE_SECURE` | HTTP 阶段 `False`，上 HTTPS 后 `True` |
| `API_CORS_ORIGINS` | `http://203.195.206.242` |

方式 B（从已有环境迁移，本次采用）：上传原 `.env` 到 `/tmp`，在服务器运行覆盖脚本（参考 `patch_env.py` 的思路）：

1. 解析原 `.env` 为键值对
2. 覆盖基础设施项：Mongo/Milvus/MinIO 地址与凭据（凭据用 `docker inspect` 从容器读取，**不打印**）、`API_CORS_ORIGINS`、`AUTH_SECRET_KEY`（新生成）等
3. 写入 `/opt/rag/deploy/.env`，随后立即删除 `/tmp` 中的原 `.env`

### 4.5 构建与启动

```bash
cd /opt/rag/deploy
docker compose config > /dev/null   # 校验编排和 env
docker compose build                # 首次约 3-8 分钟（已配 pypi 清华 / npm npmmirror 镜像）
docker compose up -d
docker compose ps                   # backend 应为 healthy
```

### 4.6 创建登录账号

密码长度至少 5 位：

```bash
cd /opt/rag/deploy
docker compose exec -T -e ADMIN_PASSWORD='<密码>' backend \
  python -m scripts.create_user admin --password-env ADMIN_PASSWORD
```

方案中前端登录表单预填了 `admin/admin`（`frontend/src/App.vue` 中 `DEFAULT_USERNAME/DEFAULT_PASSWORD`），如需一致则密码用 `admin`，但公网环境建议改密并去掉预填。

### 4.7 验证清单

```bash
# 1. 容器与健康
docker compose ps
docker compose logs backend | tail

# 2. API 通路（nginx 反代，未登录应 401 JSON）
curl -s -i http://127.0.0.1/api/v1/auth/me | head -5

# 3. 从外部浏览器：打开 http://203.195.206.242/ ，登录后：
#    - 导入一份短 PDF，确认进度 SSE 与成功状态
#    - fast / deep 各查询一次
#    - 答案中图片能正常显示（说明 MINIO_PUBLIC_ENDPOINT 与 9000 放行正确）
```

2026-09-13 实际验证结果供对照：导入 H3C 手册 53.2s 成功；fast 查询 5.3s；deep 查询 43.6s（HyDE 占 30s+）；图片外链 HTTP 200 `image/jpeg`。

## 5. 更新部署

```bash
# 上传新代码到 /opt/rag（重复 4.2-4.3，不改 .env）
cd /opt/rag/deploy
docker compose build
docker compose up -d
```

仅修改 `.env` 时：

```bash
docker compose up -d --force-recreate backend
```

## 6. 常用运维

```bash
docker compose logs -f backend        # 后端日志（含每个节点 elapsed 耗时）
docker compose logs -f frontend       # nginx 日志
docker compose restart backend
docker compose exec backend python -c "from utils.mongodb_utils import get_mongodb_util; print(get_mongodb_util().ping())"
docker volume ls | grep rag           # 应用数据卷
```

## 7. 排障对照表

| 症状 | 排查方向 |
|------|----------|
| 页面能开但接口 401/超时 | `docker compose logs backend`；确认 `.env` 中 Mongo 地址与密码；`host.docker.internal` 是否解析（compose 里有 extra_hosts） |
| 导入报 `Milvus Collection 不存在` 或连接失败 | backend 是否接入 `milvus` 网络（`docker network inspect milvus` 应含 `rag-backend-1`）；`MILVUS_URL` 是否为 `http://milvus-standalone:19530` |
| 图片不显示 | `MINIO_PUBLIC_ENDPOINT` 是否正确；安全组是否放行 9000；直接 curl 图片 URL 看状态码 |
| MinIO 报 AccessDenied | `.env` 的 MinIO key 是否取自 `docker inspect milvus-minio`；bucket `rag` 是否存在 |
| 构建卡住/失败 | 磁盘空间（`df -h`）、镜像源连通性；服务器仅 3.6G 内存，避免并行构建 |
| SSE 不流式、答案一次性出现 | 检查 `frontend/nginx.conf` 的 `proxy_buffering off` 是否被改动 |
| 查询报 LLM/Embedding 错误 | 对应 key 是否有效；DeepSeek 与 DashScope 的 key 分开配置，不要混用 |

## 8. 安全注意事项

- `.env`、服务器密码、API key 一律不入库、不打印；服务器 `/tmp` 中的临时环境文件用完立即删除
- 云安全组建议只放行 22、80、9000；**关闭 19530（Milvus）和 3000（attu）的公网暴露**（应用走 Docker 内网，不需要公网）
- 服务器 root 密码与 DeepSeek / DashScope / MinerU key 在首次部署过程中曾出现在对话里，上线后应轮换
- 生产化下一步：绑定域名 + HTTPS（Let's Encrypt），随后把 `AUTH_COOKIE_SECURE=True`；Mongo/MinIO 建议创建最小权限应用账号替代 root

## 9. 已知限制

- 导入任务基于 FastAPI `BackgroundTasks`，服务重启后运行中的任务不会自动恢复（Mongo 中保留进度可查）
- 查询 SSE 断线不可续传
- 前端为纯静态构建，修改前端后必须重新 `docker compose build frontend`
- `requirements.txt` 是本地 conda freeze，不要用于 Docker；维护依赖请更新 `requirements-docker.txt`
