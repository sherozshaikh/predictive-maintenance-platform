import io
from pathlib import Path

from loguru import logger
from minio import Minio

from configs.settings import get_infra_settings


def _get_client() -> Minio:
    settings = get_infra_settings()
    return Minio(
        settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )


def ensure_bucket() -> None:
    settings = get_infra_settings()
    client = _get_client()
    if not client.bucket_exists(settings.minio.bucket):
        client.make_bucket(settings.minio.bucket)
        logger.info("created_minio_bucket", bucket=settings.minio.bucket)


def upload_file(local_path: str, object_name: str) -> None:
    settings = get_infra_settings()
    client = _get_client()
    client.fput_object(settings.minio.bucket, object_name, local_path)
    logger.info("uploaded_to_minio", object_name=object_name)


def upload_bytes(data: bytes, object_name: str) -> None:
    settings = get_infra_settings()
    client = _get_client()
    client.put_object(
        settings.minio.bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
    )
    logger.info("uploaded_bytes_to_minio", object_name=object_name)


def download_file(object_name: str, local_path: str) -> None:
    settings = get_infra_settings()
    client = _get_client()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client.fget_object(settings.minio.bucket, object_name, local_path)
    logger.info("downloaded_from_minio", object_name=object_name)
