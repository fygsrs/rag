from processor.query_processor.main_graph import KBQueryWorkflow
from processor.query_processor.state import create_default_query_state


class TestKBQueryWorkflow:
    def test_full_retrieval_route_can_run(self):
        state = create_default_query_state(original_query="如何调整转印温度？")

        result = KBQueryWorkflow().run(state)

        assert result["rewritten_query"] == "如何调整转印温度？"
        assert result["embedding_chunks"] == []
        assert result["hyde_embedding_chunks"] == []
        assert result["web_search_docs"] == []
        assert result["rrf_chunks"] == []
        assert result["reranked_docs"] == []
        assert result["answer"] == ""

    def test_existing_answer_skips_retrieval_route(self):
        state = {
            "original_query": "这个怎么设置？",
            "answer": "请先确认具体商品型号。",
        }

        result = KBQueryWorkflow().run(state)

        assert result["answer"] == "请先确认具体商品型号。"
        assert "embedding_chunks" not in result
        assert "hyde_embedding_chunks" not in result
        assert "web_search_docs" not in result
