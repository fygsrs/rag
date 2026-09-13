# 知识库助手前端

Vue 3 + TypeScript + Vite + Tailwind CSS。开发时由 Vite 将 `/api` 代理到
`http://127.0.0.1:8000`，因此登录 Cookie、POST-SSE 查询和导入进度 SSE
都保持同源访问。

```powershell
cd frontend
npm install
npm run dev
```

后端启动命令：

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

首次使用前创建账号：

```powershell
python -m scripts.create_user admin
```
