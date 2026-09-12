"""D：联网搜索节点。"""

from __future__ import annotations

import asyncio
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from processor.query_processor.base import NodeBase
from processor.query_processor.config import QueryConfig
from processor.query_processor.state import QueryGraphState


class NodeWebSearchMcp(NodeBase):
    """通过百炼 MCP 联网搜索补充实时资料。"""

    name = "node_web_search_mcp"

    def __init__(
        self,
        config: QueryConfig | None = None,
        *,
        mcp_caller=None,
    ) -> None:
        super().__init__()
        self.config = config or QueryConfig()
        self._mcp_caller = mcp_caller

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query")
        if not isinstance(query, str) or not query.strip():
            self.logger.info("联网搜索跳过 | reason=empty_query")
            return {"web_search_docs": []}

        top_k = int(self.config.web_search_top_k)
        if top_k <= 0:
            raise RuntimeError("WEB_SEARCH_TOP_K 必须大于 0")

        caller = self._mcp_caller or self._call_web_search_mcp
        try:
            result = caller(query.strip(), top_k)
            if inspect.isawaitable(result):
                result = self._run_awaitable(result)
            documents = self._parse_search_result(result, limit=top_k)
        except Exception:
            self.logger.exception("百炼 MCP 联网搜索失败，降级为空结果")
            documents = []

        self.logger.info(
            "联网搜索完成 | requested_top_k=%d | hits=%d",
            top_k,
            len(documents),
        )
        return {"web_search_docs": documents}

    async def _call_web_search_mcp(self, query: str, count: int):
        api_key = str(self.config.dashscope_api_key or "").strip()
        mcp_url = str(self.config.web_search_mcp_url or "").strip()
        tool_name = str(self.config.web_search_tool or "").strip()
        if not api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")
        if not mcp_url:
            raise RuntimeError("未配置 WEB_SEARCH_MCP_URL")
        if not tool_name:
            raise RuntimeError("未配置 WEB_SEARCH_MCP_TOOL")

        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        from httpx2 import AsyncClient

        timeout = float(self.config.web_search_timeout)
        async with AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        ) as http_client:
            transport = streamable_http_client(
                mcp_url,
                http_client=http_client,
            )
            async with Client(
                transport,
                read_timeout_seconds=timeout,
            ) as client:
                return await client.call_tool(
                    tool_name,
                    {"query": query, "count": count},
                    read_timeout_seconds=timeout,
                )

    @staticmethod
    def _run_awaitable(awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, awaitable).result()

    def _parse_search_result(
        self,
        result: Any,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        payload = self._result_to_payload(result)
        pages = payload.get("pages") if isinstance(payload, dict) else None
        if not isinstance(pages, list):
            return []

        documents = []
        seen_urls = set()
        for page in pages:
            if not isinstance(page, dict):
                continue
            title = str(page.get("title") or "").strip()
            url = str(page.get("url") or "").strip()
            snippet = str(
                page.get("snippet") or page.get("content") or ""
            ).strip()
            if not snippet or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            documents.append(
                {
                    "id": url,
                    "title": title or url,
                    "content": snippet,
                    "url": url,
                    "source": "web_search",
                    "metadata": {},
                }
            )
            if len(documents) >= limit:
                break
        return documents

    @staticmethod
    def _result_to_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        if getattr(result, "is_error", False):
            raise RuntimeError("MCP 工具返回错误")

        structured_content = getattr(result, "structured_content", None)
        if isinstance(structured_content, dict):
            return structured_content

        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if not isinstance(text, str) and isinstance(block, dict):
                text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        return {}
