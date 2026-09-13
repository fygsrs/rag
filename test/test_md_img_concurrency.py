import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from processor.import_processor.config import ImportConfig
from processor.import_processor.exceptions import ImageProcessingError
from processor.import_processor.nodes import c_node_md_img
from processor.import_processor.nodes.c_node_md_img import NodeMDImg


class FakeVisionClient:
    def __init__(self, *, barrier=None, fail_all=False):
        self.barrier = barrier
        self.fail_all = fail_all
        self.images_seen = []
        self.lock = threading.Lock()

    def invoke(self, messages):
        image_url = messages[0].content[1]["image_url"]["url"]
        with self.lock:
            self.images_seen.append(image_url)
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if self.fail_all:
            raise RuntimeError("vision failed")
        return SimpleNamespace(content="图片摘要")


class FakeMinioClient:
    def __init__(self):
        self.uploads = []

    def fput_object(self, bucket, object_name, file_path, content_type=None):
        self.uploads.append((bucket, object_name))


def make_doc(tmp_path, image_count):
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    blocks = []
    for index in range(image_count):
        name = f"img{index}.png"
        (images_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
        blocks.append(f"![图{index}](images/{name})")
    md_path = tmp_path / "doc.md"
    md_path.write_text("\n\n".join(blocks), encoding="utf-8")
    return md_path


def make_config(**overrides):
    values = {
        "vl_model": "fake-vl",
        "image_summary_concurrency": 4,
        "requests_per_minute": 600,
        "minio_bucket": "test-bucket",
        "minio_endpoint": "minio.test:9000",
    }
    values.update(overrides)
    return ImportConfig(**values)


def patch_clients(monkeypatch, client, minio_client):
    monkeypatch.setattr(c_node_md_img, "get_vl_client", lambda model=None: client)
    monkeypatch.setattr(c_node_md_img, "get_minio_client", lambda: minio_client)


def test_generates_summaries_concurrently(tmp_path, monkeypatch):
    barrier = threading.Barrier(4)
    client = FakeVisionClient(barrier=barrier)
    minio = FakeMinioClient()
    patch_clients(monkeypatch, client, minio)
    node = NodeMDImg(make_config(image_summary_concurrency=4))
    md_path = make_doc(tmp_path, 4)

    result = node.process({"md_path": str(md_path), "file_title": "测试文档"})

    assert len(client.images_seen) == 4
    assert len(minio.uploads) == 4
    assert (
        result["md_content"].count("http://minio.test:9000/test-bucket/") == 4
    )
    assert (tmp_path / "doc_processed.md").is_file()


def test_raises_image_processing_error_when_vision_fails(tmp_path, monkeypatch):
    client = FakeVisionClient(fail_all=True)
    minio = FakeMinioClient()
    patch_clients(monkeypatch, client, minio)
    node = NodeMDImg(make_config(image_summary_concurrency=2))
    md_path = make_doc(tmp_path, 3)

    with pytest.raises(
        ImageProcessingError,
        match=r"图片摘要生成失败: img[012]\.png",
    ):
        node.process({"md_path": str(md_path), "file_title": "测试文档"})

    assert minio.uploads == []


def test_rate_limit_window_waits_after_limit():
    node = NodeMDImg(make_config(requests_per_minute=1))

    request_times: deque[float] = deque()
    node._wait_for_rate_limit_window(request_times, 1, window_seconds=0.2)
    node._wait_for_rate_limit_window(request_times, 1, window_seconds=0.2)

    started_at = time.perf_counter()
    node._wait_for_rate_limit_window(request_times, 1, window_seconds=0.2)
    waited = time.perf_counter() - started_at

    assert waited >= 0.15
