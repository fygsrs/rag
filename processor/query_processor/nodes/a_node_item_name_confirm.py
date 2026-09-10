"""A：商品主体确认节点。"""

from __future__ import annotations

import json
import re
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from processor.query_processor.base import NodeBase
from processor.query_processor.config import QueryConfig
from processor.query_processor.state import QueryGraphState


class NodeItemNameConfirm(NodeBase):
    """结合会话历史提取、检索并确认标准商品名称。"""

    name = "node_item_name_confirm"
    _MAX_ITEM_NAMES = 10
    _MAX_ITEM_NAME_BYTES = 512

    def __init__(
        self,
        config: QueryConfig | None = None,
        *,
        llm_client=None,
        embedding_tool=None,
        milvus_client=None,
        mongodb_util=None,
    ) -> None:
        super().__init__()
        self.config = config or QueryConfig()
        self._llm_client = llm_client
        self._embedding_tool = embedding_tool
        self._milvus_client = milvus_client
        self._mongodb_util = mongodb_util

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self.logger.info("【%s】确认商品主体并改写问题", self.name)
        session_id, message_id, original_query = self._validate_state(state)
        self._validate_config()
        mongodb = self._get_mongodb_util()

        history = self._load_history(mongodb, session_id, message_id)
        self._save_user_message(
            mongodb,
            session_id=session_id,
            message_id=message_id,
            content=original_query,
        )

        extracted_names, rewritten_query = self._extract_item_names(
            history,
            original_query,
            list(state.get("item_names") or []),
        )
        if not extracted_names:
            answer = self._build_not_found_answer([])
            return self._persist_result(
                mongodb,
                session_id=session_id,
                message_id=message_id,
                item_names=[],
                rewritten_query=rewritten_query or original_query,
                status="not_found",
                candidates=[],
                answer=answer,
            )

        search_results = self._search_item_candidates(extracted_names)
        confirmed_names, candidates, unmatched_names = self._align_candidates(
            extracted_names,
            search_results,
        )

        if unmatched_names:
            item_names: list[str] = []
            answer = self._build_not_found_answer(unmatched_names)
            status = "not_found"
        elif len(confirmed_names) != len(extracted_names):
            item_names = []
            answer = self._build_candidate_answer(candidates)
            status = "needs_confirmation"
        else:
            rewritten_query = self._apply_standard_names(
                rewritten_query or original_query,
                extracted_names,
                confirmed_names,
            )
            item_names = self._deduplicate(confirmed_names)
            answer = ""
            status = "confirmed"

        return self._persist_result(
            mongodb,
            session_id=session_id,
            message_id=message_id,
            item_names=item_names,
            rewritten_query=rewritten_query or original_query,
            status=status,
            candidates=candidates,
            answer=answer,
        )

    @staticmethod
    def _validate_state(state: QueryGraphState) -> tuple[str, str, str]:
        values = []
        for field_name in ("session_id", "message_id", "original_query"):
            value = state.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            values.append(value.strip())
        return values[0], values[1], values[2]

    def _validate_config(self) -> None:
        required_values = {
            "ITEM_MODEL": self.config.item_model,
            "MILVUS_URL": self.config.milvus_url,
            "ITEM_NAME_COLLECTION": self.config.item_name_collection,
        }
        missing = [
            name
            for name, value in required_values.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            raise RuntimeError(f"查询配置缺失: {', '.join(missing)}")

        numeric_values = {
            "QUERY_HISTORY_LIMIT": self.config.history_limit,
            "ITEM_NAME_TOP_K": self.config.item_name_top_k,
            "ITEM_NAME_CONFIRM_THRESHOLD": self.config.item_name_confirm_threshold,
            "ITEM_NAME_CANDIDATE_THRESHOLD": (
                self.config.item_name_candidate_threshold
            ),
            "ITEM_NAME_SCORE_MARGIN": self.config.item_name_score_margin,
            "ITEM_NAME_DENSE_WEIGHT": self.config.item_name_dense_weight,
            "ITEM_NAME_SPARSE_WEIGHT": self.config.item_name_sparse_weight,
        }
        try:
            numbers = {
                name: float(value)
                for name, value in numeric_values.items()
                if not isinstance(value, bool)
            }
        except (TypeError, ValueError) as exc:
            raise RuntimeError("查询数值配置格式不正确") from exc
        if len(numbers) != len(numeric_values):
            raise RuntimeError("查询数值配置不能使用布尔值")

        for name in ("QUERY_HISTORY_LIMIT", "ITEM_NAME_TOP_K"):
            value = numbers[name]
            if value <= 0 or not value.is_integer():
                raise RuntimeError(f"{name} 必须是正整数")

        for name in (
            "ITEM_NAME_CONFIRM_THRESHOLD",
            "ITEM_NAME_CANDIDATE_THRESHOLD",
            "ITEM_NAME_SCORE_MARGIN",
            "ITEM_NAME_DENSE_WEIGHT",
            "ITEM_NAME_SPARSE_WEIGHT",
        ):
            if not 0 <= numbers[name] <= 1:
                raise RuntimeError(f"{name} 必须在 0 到 1 之间")

        if (
            numbers["ITEM_NAME_CANDIDATE_THRESHOLD"]
            > numbers["ITEM_NAME_CONFIRM_THRESHOLD"]
        ):
            raise RuntimeError(
                "ITEM_NAME_CANDIDATE_THRESHOLD 不能高于 "
                "ITEM_NAME_CONFIRM_THRESHOLD"
            )
        if (
            numbers["ITEM_NAME_DENSE_WEIGHT"]
            + numbers["ITEM_NAME_SPARSE_WEIGHT"]
            <= 0
        ):
            raise RuntimeError("商品名称检索权重之和必须大于 0")

    def _load_history(
        self,
        mongodb,
        session_id: str,
        message_id: str,
    ) -> list[dict[str, Any]]:
        history_limit = max(1, int(self.config.history_limit))
        memories = mongodb.get_memories(session_id, limit=history_limit)
        excluded_ids = {message_id, self._assistant_message_id(message_id)}
        return [
            memory
            for memory in memories
            if memory.get("message_id") not in excluded_ids
        ]

    @staticmethod
    def _save_user_message(
        mongodb,
        *,
        session_id: str,
        message_id: str,
        content: str,
    ) -> None:
        mongodb.save_memory(
            session_id=session_id,
            message_id=message_id,
            role="user",
            content=content,
            metadata={
                "source": "node_item_name_confirm",
                "item_name_status": "processing",
            },
        )

    def _extract_item_names(
        self,
        history: list[dict[str, Any]],
        original_query: str,
        state_item_names: list[str],
    ) -> tuple[list[str], str]:
        client = self._llm_client
        if client is None:
            from utils.llm_utils import get_llm_client

            client = get_llm_client(
                model=self.config.item_model,
                json_mode=True,
            )

        prompt = self._build_extraction_prompt(
            history,
            original_query,
            state_item_names,
        )
        try:
            response = client.invoke(prompt)
        except Exception as exc:
            raise RuntimeError("商品名称提取模型调用失败") from exc

        return self._parse_extraction(self._response_to_text(response))

    @staticmethod
    def _build_extraction_prompt(
        history: list[dict[str, Any]],
        original_query: str,
        state_item_names: list[str],
    ) -> str:
        history_lines = []
        for memory in history:
            role = str(memory.get("role") or "unknown")
            content = str(memory.get("content") or "").strip()
            metadata = memory.get("metadata")
            known_names = (
                metadata.get("item_names", [])
                if isinstance(metadata, dict)
                else []
            )
            suffix = f"（已确认商品：{'、'.join(known_names)}）" if known_names else ""
            if content:
                history_lines.append(f"{role}: {content}{suffix}")

        history_text = "\n".join(history_lines) or "（无历史会话）"
        state_names_text = "、".join(
            name.strip()
            for name in state_item_names
            if isinstance(name, str) and name.strip()
        ) or "（无）"
        return (
            "你是商品知识库的查询理解器。请结合历史会话和当前问题，"
            "提取用户本轮要查询的所有商品、产品或设备名称，并将问题改写成"
            "脱离上下文也能理解的完整问题。\n"
            "要求：\n"
            "1. 保留品牌、系列和型号，不要臆测不存在的商品；\n"
            "2. 用户用序号或简称回复时，根据历史候选还原对应商品名；\n"
            "3. 无法提取商品时 item_names 返回空数组；\n"
            "4. 只返回 JSON，不要解释。格式："
            '{"item_names": ["商品名称"], "rewritten_query": "完整问题"}\n\n'
            f"历史会话：\n{history_text}\n\n"
            f"状态中的商品提示：{state_names_text}\n"
            f"当前问题：{original_query}"
        )

    def _parse_extraction(self, response_text: str) -> tuple[list[str], str]:
        text = response_text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("商品名称提取结果不是有效 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("商品名称提取结果必须是 JSON 对象")

        raw_names = payload.get("item_names")
        rewritten_query = payload.get("rewritten_query")
        if not isinstance(raw_names, list):
            raise RuntimeError("商品名称提取结果中的 item_names 必须是数组")
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            raise RuntimeError("商品名称提取结果缺少 rewritten_query")

        names = self._deduplicate(
            [name.strip() for name in raw_names if isinstance(name, str) and name.strip()]
        )
        if len(names) > self._MAX_ITEM_NAMES:
            raise RuntimeError("商品名称提取数量超过限制")
        if any(
            len(name.encode("utf-8")) > self._MAX_ITEM_NAME_BYTES
            for name in names
        ):
            raise RuntimeError("商品名称提取结果过长")
        return names, rewritten_query.strip()

    @staticmethod
    def _response_to_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "\n".join(texts).strip()
        return str(content).strip()

    def _search_item_candidates(
        self,
        item_names: list[str],
    ) -> list[list[dict[str, Any]]]:
        collection_name = str(self.config.item_name_collection or "").strip()
        if not collection_name:
            raise RuntimeError("未配置 ITEM_NAME_COLLECTION")
        client = self._get_milvus_client()
        if not client.has_collection(collection_name):
            raise RuntimeError(f"Milvus Collection 不存在: {collection_name}")

        embedding_tool = self._embedding_tool
        if embedding_tool is None:
            from utils.embedding_utils import embedding_tool as default_embedding_tool

            embedding_tool = default_embedding_tool

        try:
            dense_vectors, sparse_vectors = embedding_tool.embed_dense_and_sparse(
                item_names,
                text_type="query",
            )
        except Exception as exc:
            raise RuntimeError("商品名称向量生成失败") from exc
        if len(dense_vectors) != len(item_names) or len(sparse_vectors) != len(
            item_names
        ):
            raise RuntimeError("商品名称向量数量与输入数量不一致")

        top_k = max(1, int(self.config.item_name_top_k))
        requests = [
            AnnSearchRequest(
                data=dense_vectors,
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=top_k,
            ),
            AnnSearchRequest(
                data=sparse_vectors,
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}},
                limit=top_k,
            ),
        ]
        try:
            results = client.hybrid_search(
                collection_name=collection_name,
                reqs=requests,
                ranker=WeightedRanker(
                    float(self.config.item_name_dense_weight),
                    float(self.config.item_name_sparse_weight),
                ),
                limit=top_k,
                output_fields=["item_name", "file_title"],
            )
        except Exception as exc:
            raise RuntimeError("商品名称 Milvus 混合检索失败") from exc

        if not results:
            return [[] for _ in item_names]
        if len(results) != len(item_names):
            raise RuntimeError("Milvus 返回的查询结果数量不正确")
        return results

    def _get_milvus_client(self):
        if self._milvus_client is not None:
            return self._milvus_client
        milvus_url = str(self.config.milvus_url or "").strip()
        if not milvus_url:
            raise RuntimeError("未配置 MILVUS_URL")
        self._milvus_client = MilvusClient(uri=milvus_url)
        return self._milvus_client

    def _align_candidates(
        self,
        item_names: list[str],
        search_results: list[list[dict[str, Any]]],
    ) -> tuple[list[str], list[str], list[str]]:
        confirmed_names = []
        all_candidates = []
        unmatched_names = []

        for requested_name, hits in zip(item_names, search_results):
            ranked = self._normalize_hits(hits)
            candidates = [
                name
                for name, score in ranked
                if score >= float(self.config.item_name_candidate_threshold)
            ]
            if not candidates:
                unmatched_names.append(requested_name)
                continue

            best_name, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
            score_gap = best_score - second_score
            if (
                best_score >= float(self.config.item_name_confirm_threshold)
                and score_gap + 1e-12
                >= float(self.config.item_name_score_margin)
            ):
                confirmed_names.append(best_name)
            else:
                all_candidates.extend(candidates)

        return (
            confirmed_names,
            self._deduplicate(all_candidates),
            unmatched_names,
        )

    @staticmethod
    def _normalize_hits(hits: list[dict[str, Any]]) -> list[tuple[str, float]]:
        scores_by_name: dict[str, tuple[str, float]] = {}
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            entity = hit.get("entity")
            entity = entity if isinstance(entity, dict) else {}
            item_name = entity.get("item_name") or hit.get("item_name")
            score = hit.get("distance", hit.get("score"))
            if not isinstance(item_name, str) or not item_name.strip():
                continue
            if not isinstance(score, (int, float)):
                continue

            normalized_name = item_name.strip()
            key = normalized_name.casefold()
            current = scores_by_name.get(key)
            if current is None or float(score) > current[1]:
                scores_by_name[key] = (normalized_name, float(score))
        return sorted(scores_by_name.values(), key=lambda value: value[1], reverse=True)

    @staticmethod
    def _apply_standard_names(
        rewritten_query: str,
        extracted_names: list[str],
        confirmed_names: list[str],
    ) -> str:
        result = rewritten_query.strip()
        for extracted_name, confirmed_name in zip(
            extracted_names,
            confirmed_names,
        ):
            result = re.sub(
                re.escape(extracted_name),
                lambda _: confirmed_name,
                result,
                count=1,
                flags=re.IGNORECASE,
            )

        missing_names = [name for name in confirmed_names if name not in result]
        if missing_names:
            result = f"{'、'.join(missing_names)}：{result}"
        return result

    @staticmethod
    def _build_candidate_answer(candidates: list[str]) -> str:
        options = "\n".join(
            f"{index}. {item_name}"
            for index, item_name in enumerate(candidates, start=1)
        )
        return (
            f"请确认您要查询的商品：\n{options}\n"
            "请回复商品名称或序号。"
        )

    @staticmethod
    def _build_not_found_answer(unmatched_names: list[str]) -> str:
        if unmatched_names:
            names = "、".join(unmatched_names)
            return (
                f"暂时无法确认商品“{names}”，"
                "请提供完整的品牌、系列或型号。"
            )
        return "暂时无法识别您要查询的商品，请提供完整的品牌、系列或型号。"

    def _persist_result(
        self,
        mongodb,
        *,
        session_id: str,
        message_id: str,
        item_names: list[str],
        rewritten_query: str,
        status: str,
        candidates: list[str],
        answer: str,
    ) -> QueryGraphState:
        mongodb.update_message_item_names(
            session_id=session_id,
            message_id=message_id,
            item_names=item_names,
            rewritten_query=rewritten_query,
            status=status,
            candidates=candidates,
        )
        if answer:
            mongodb.save_memory(
                session_id=session_id,
                message_id=self._assistant_message_id(message_id),
                role="assistant",
                content=answer,
                metadata={
                    "source": "node_item_name_confirm",
                    "item_name_status": status,
                    "item_name_candidates": list(candidates),
                },
            )
        else:
            mongodb.delete_memory(
                session_id=session_id,
                message_id=self._assistant_message_id(message_id),
            )

        history = mongodb.get_memories(
            session_id,
            limit=max(1, int(self.config.history_limit)),
        )
        return {
            "item_names": list(item_names),
            "rewritten_query": rewritten_query,
            "answer": answer,
            "history": history,
        }

    def _get_mongodb_util(self):
        if self._mongodb_util is None:
            from utils.mongodb_utils import get_mongodb_util

            self._mongodb_util = get_mongodb_util()
        return self._mongodb_util

    @staticmethod
    def _assistant_message_id(message_id: str) -> str:
        return f"{message_id}:item-name-confirm"

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        result = []
        seen = set()
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result
