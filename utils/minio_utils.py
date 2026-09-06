import json
import logging

from minio import Minio

from config.minio_config import minio_config


def _create_minio_client() -> Minio:
    """创建 MinIO 客户端，并确保目标 Bucket 已初始化。"""
    client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        # 当前 MinIO 服务使用 HTTP；启用 HTTPS 后再改为 True。
        secure=False,
    )

    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)

    # 允许匿名读取 Bucket 中的对象，但不允许匿名上传或删除。
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"],
            }
        ],
    }
    client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))
    return client


try:
    minio_client = _create_minio_client()
except Exception:
    logging.exception("MinIO 客户端初始化失败")
    minio_client = None


def get_minio_client() -> Minio | None:
    """返回初始化完成的 MinIO 客户端；初始化失败时返回 None。"""
    return minio_client
