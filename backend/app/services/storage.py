"""File storage abstraction — local filesystem or GCS."""

import os
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger

from app.config import get_settings


class StorageBackend(ABC):
    @abstractmethod
    async def save_video(self, user_id: str, scan_id: str, filename: str, content: bytes) -> str:
        ...

    @abstractmethod
    async def get_video_path(self, storage_path: str) -> str:
        ...

    @abstractmethod
    async def delete(self, storage_path: str) -> None:
        ...


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_video(self, user_id: str, scan_id: str, filename: str, content: bytes) -> str:
        dir_path = self.base_dir / user_id / scan_id
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / filename
        # Avoid overwrite — append suffix
        counter = 1
        while file_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            file_path = dir_path / f"{stem}_{counter}{suffix}"
            counter += 1
        file_path.write_bytes(content)
        logger.info(f"Saved video: {file_path} ({len(content)} bytes)")
        return str(file_path.relative_to(self.base_dir))

    async def get_video_path(self, storage_path: str) -> str:
        full = self.base_dir / storage_path
        if not full.exists():
            raise FileNotFoundError(f"Video not found: {storage_path}")
        return str(full)

    async def delete(self, storage_path: str) -> None:
        full = self.base_dir / storage_path
        if full.exists():
            full.unlink()


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend — lazy-loaded to avoid import cost."""

    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self._client = None

    def _get_bucket(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client()
        return self._client.bucket(self.bucket_name)

    async def save_video(self, user_id: str, scan_id: str, filename: str, content: bytes) -> str:
        blob_path = f"videos/{user_id}/{scan_id}/{filename}"
        blob = self._get_bucket().blob(blob_path)
        blob.upload_from_string(content, content_type="video/mp4")
        logger.info(f"Uploaded to GCS: gs://{self.bucket_name}/{blob_path}")
        return blob_path

    async def get_video_path(self, storage_path: str) -> str:
        blob = self._get_bucket().blob(storage_path)
        return blob.generate_signed_url(expiration=3600)

    async def delete(self, storage_path: str) -> None:
        blob = self._get_bucket().blob(storage_path)
        blob.delete()


_storage_instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _storage_instance
    if _storage_instance is None:
        settings = get_settings()
        if settings.STORAGE_BACKEND == "gcs" and settings.GCS_BUCKET:
            _storage_instance = GCSStorage(settings.GCS_BUCKET)
        else:
            _storage_instance = LocalStorage(settings.UPLOAD_DIR)
    return _storage_instance
