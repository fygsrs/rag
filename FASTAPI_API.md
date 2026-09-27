# FastAPI 接口说明

## 启动服务

```powershell
conda activate rag
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后可以打开：

- Swagger：<http://127.0.0.1:8000/docs>
- OpenAPI：<http://127.0.0.1:8000/openapi.json>
- 健康检查：<http://127.0.0.1:8000/health>

## 导入文档

除健康检查和登录外，接口均需要先登录。登录成功后服务通过 HttpOnly
Cookie 维持会话，账号由管理员使用命令创建：

```powershell
python -m scripts.create_user admin
```

前端使用的后台导入接口会立即返回任务 ID：

```text
POST /api/v1/imports/tasks
GET  /api/v1/imports/tasks
GET  /api/v1/imports/tasks/{task_id}
GET  /api/v1/imports/tasks/{task_id}/events
```

最后一个接口为可重新连接的 GET-SSE，任务及阶段进度保存在 MongoDB。

一次请求导入一个 PDF 或 Markdown 文件。接口会等待完整导入流程结束，再返回商品名、切片数量等摘要。

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/imports" `
  -F "file=@E:\docs\HAK180产品安全手册.pdf"
```

也可以由调用方传入 `task_id`：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/imports" `
  -F "file=@E:\docs\HAK180产品安全手册.pdf" `
  -F "task_id=import-hak180-001"
```

批量导入一个目录时建议顺序提交，避免图片摘要接口并发过高：

```powershell
$documentDir = "E:\code\py\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\doc"

Get-ChildItem -LiteralPath $documentDir -Filter "*.pdf" -File -Recurse | ForEach-Object {
    curl.exe -X POST "http://127.0.0.1:8000/api/v1/imports" `
      -F "file=@$($_.FullName)"
}
```

## 普通查询

`session_id`、`message_id` 和 `task_id` 都可以省略，服务会自动生成。多轮会话应持续使用相同的 `session_id`。

```powershell
$body = @{
    query = "H3C LA2608 室内无线网关如何配置 DCC 拨号？"
    session_id = "session-demo-001"
    message_id = "message-demo-001"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/queries" `
  -ContentType "application/json" `
  -Body $body
```

## POST-SSE 流式查询

该接口先返回 `start`，执行过程中返回 `progress` 和多个 `answer`，最后
返回 `final`。执行失败时返回 `progress/failed` 和 `error`。

```powershell
curl.exe -N -X POST "http://127.0.0.1:8000/api/v1/queries/stream" `
  -H "Content-Type: application/json" `
  -d '{"query":"H3C LA2608 室内无线网关如何配置 DCC 拨号？","session_id":"session-demo-001"}'
```

事件格式示例：

```text
event: answer
data: {"content":"配置"}

event: final
data: {"answer":"完整答案","sources":[...]}
```

## 环境变量

```dotenv
API_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
API_MAX_UPLOAD_MB=100
API_UPLOAD_DIR=.runtime/uploads
AUTH_SECRET_KEY=至少32个随机字符
AUTH_COOKIE_SECURE=False
```

上传的原始文件暂存在 `.runtime/uploads`，该目录已被 Git 忽略。
