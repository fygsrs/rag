from langchain_core.embeddings import Embeddings

from config.embedding_config import embedding_config


class EmbeddingTool:
    """向量化工具：按配置选择本地 BGE-M3 或阿里云向量模型。"""

    def __init__(self, config=None):
        self.config = config or embedding_config
        self.embeddings: Embeddings = self._build_embeddings()

    def _build_embeddings(self) -> Embeddings:
        if self.config.provider == "aliyun":
            return self._build_aliyun()
        return self._build_bge()

    def _build_aliyun(self) -> Embeddings:
        """阿里云 DashScope 向量模型（OpenAI 兼容接口）。"""
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=self.config.dashscope_model,
            api_key=self.config.dashscope_api_key,
            base_url=self.config.dashscope_base_url,
        )

    def _build_bge(self) -> Embeddings:
        """本地 BGE-M3，使用 langchain_huggingface 加载。"""
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=self.config.bge_m3_path,
            model_kwargs={"device": self.config.bge_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对文档列表向量化。"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """对单条查询向量化。"""
        return self.embeddings.embed_query(text)


embedding_tool = EmbeddingTool()


if __name__ == "__main__":
    print(f"当前向量 provider: {embedding_config.provider}")
    print(f"向量维度: {len(embedding_tool.embed_query('测试'))}")
