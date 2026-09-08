from pathlib import Path


class BGEModelDownloader:
    """下载 BGE-M3 模型到项目目录的工具类。"""

    DEFAULT_MODEL_ID = "BAAI/bge-m3"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        if cache_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            cache_dir = project_root / "models"
        self.cache_dir = Path(cache_dir)

    def download(self) -> str:
        """下载模型，返回模型本地目录。"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from modelscope.hub.snapshot_download import snapshot_download

        model_dir = snapshot_download(
            model_id=self.model_id,
            cache_dir=str(self.cache_dir),
        )
        return model_dir


if __name__ == "__main__":
    downloader = BGEModelDownloader()
    model_dir = downloader.download()
    print(f"模型已下载到: {model_dir}")
