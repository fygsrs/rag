from processor.import_processor.config import ImportConfig
from utils.text_utils import normalize_markdown_images


def test_minio_public_endpoint_accepts_full_https_url():
    config = ImportConfig(
        minio_endpoint="milvus-minio:9000",
        minio_public_endpoint="https://inaiyo.online/",
    )

    assert config.get_minio_base_url() == "https://inaiyo.online"


def test_rewrites_legacy_http_minio_image_to_https(monkeypatch):
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://inaiyo.online")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "rag")
    source = (
        "![设备图片](http://203.195.206.242:9000/rag/manual/images/example.jpg)"
    )

    result = normalize_markdown_images(source)

    assert result == "![设备图片](https://inaiyo.online/rag/manual/images/example.jpg)"
