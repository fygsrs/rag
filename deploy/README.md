# 服务器 Docker 部署说明

目标架构：

```text
浏览器 ──80──▶ frontend(nginx) ──/api──▶ backend(uvicorn:8000)
                                            ├─ Docker 内网: MongoDB / MinIO / Milvus
                                            └─ 公网: DashScope / DeepSeek / MinerU
```

## 目录约定

- 代码目录：`/opt/rag`
- 配置文件：`/opt/rag/deploy/.env`（从 `.env.production.example` 复制后填写，不提交到 git）

## 首次部署

```bash
# 1. 创建配置（填写 MONGODB_*、MINIO_*、各 API key、AUTH_SECRET_KEY）
cd /opt/rag/deploy
cp .env.production.example .env
vi .env

# 2. 构建并启动
docker compose build
docker compose up -d

# 3. 检查健康状态
docker compose ps
curl -s http://127.0.0.1/health

# 4. 创建登录账号（密码长度至少 5 位）
docker compose exec backend python -m scripts.create_user admin
```

浏览器访问 `http://203.195.206.242/`。

## 更新部署

```bash
cd /opt/rag
# 上传/更新代码后：
cd deploy && docker compose build && docker compose up -d
```

## 常用排障

```bash
docker compose logs -f backend          # 后端日志（含各节点耗时）
docker compose logs -f frontend         # nginx 日志
docker compose exec backend python -c "from utils.mongodb_utils import get_mongodb_util; print(get_mongodb_util().ping())"
```

## 关键配置说明

| 变量 | 说明 |
|------|------|
| `MONGODB_HOST` | 容器内访问宿主机端口用 `host.docker.internal` |
| `MILVUS_URL` | 使用 milvus 容器网络名，如 `http://milvus-standalone:19530` |
| `MINIO_ENDPOINT` | S3 上传/读取的内网地址，如 `milvus-minio:9000` |
| `MINIO_PUBLIC_ENDPOINT` | 浏览器加载图片的对外地址，如 `203.195.206.242:9000` |
| `API_CORS_ORIGINS` | 前端对外访问地址 |
| `AUTH_COOKIE_SECURE` | HTTP 环境为 False，HTTPS 环境改 True |

## 安全建议

- 云安全组只放行 22 / 80（MinIO 9000 如需浏览器访问再单独放行），不要暴露 27017/19530
- 服务器 root 密码与 DeepSeek/DashScope key 轮换后再上线
- 生产建议加域名和 HTTPS，再把 `AUTH_COOKIE_SECURE` 设为 True
