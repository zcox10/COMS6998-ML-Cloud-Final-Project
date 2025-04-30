import os
from pathlib import Path
from google.cloud import storage
from typing import Any, Tuple, List
import logging

from src.utils.local_file_handler import LocalFileHandler


class GcsFileHandler:
    """
    A utility class for downloading, loading, and uploading files between
    Google Cloud Storage (GCS) and the local filesystem, with behavior
    dependent on file type (JSON, JSONL, PDF, Parquet, PyTorch, etc.).
    """

    def __init__(self, bucket_name: str) -> None:
        """
        Initialize the GCSFileHandler.

        Args:
            bucket_name (str): The name of the GCS bucket to operate on.
        """
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self._local_file_handler = LocalFileHandler()

    def download_file(self, gcs_path: str) -> str:
        """
        Download a single file from GCS into a local directory determined by file suffix.

        Args:
            gcs_path (str): Path to the object in GCS (e.g., "data/papers/<entry_id>/<entry_id>.metadata.json").

        Returns:
            str: Local path where the file is downloaded.
        """
        file_suffix = os.path.splitext(gcs_path)[-1]
        local_dir = self._local_file_handler._get_local_dir(file_suffix)
        os.makedirs(local_dir, exist_ok=True)

        filename = os.path.basename(gcs_path)
        local_path = os.path.join(local_dir, filename)

        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        return local_path

    def load_file(self, gcs_path: str) -> Any:
        """
        Download and load a file from GCS into memory based on its file type.

        Args:
            gcs_path (str): Path to the object in GCS.

        Returns:
            Any: Loaded content, depending on file type (e.g., dict for JSON, str for PDF text, Tensor for .pth).
        """
        local_path = self.download_file(gcs_path)
        file_suffix = os.path.splitext(local_path)[-1]

        if file_suffix == ".json":
            obj = self._local_file_handler._load_json(local_path)
        elif file_suffix == ".jsonl":
            obj = self._local_file_handler._load_jsonl(local_path)
        elif file_suffix == ".pdf":
            obj = self._local_file_handler._load_pdf(local_path)
        elif file_suffix == ".pth":
            obj = self._local_file_handler._load_pth(local_path)
        elif file_suffix == ".parquet":
            obj = self._local_file_handler._load_parquet(local_path)
        elif file_suffix == ".md":
            obj = self._local_file_handler._load_md(local_path)
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")

        # delete local file after loading into memory
        self._local_file_handler.delete_file(local_path)
        return obj

    def upload_local_file(self, local_path: str, gcs_path: str) -> str:
        """
        Uploads a local file to GCS.
        """

        # Upload to GCS
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logging.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}\n")
        return f"gs://{self.bucket_name}/{gcs_path}"

    def upload_asset(self, obj: Any, gcs_path: str) -> str:
        """
        Upload an asset (obj) to GCS.

        Args:
            obj (Any): Common object gets detected and uploaded for a particular file type
            gcs_path (str): Destination path inside the GCS bucket.
        """
        # Save file locally to downloads
        filename = os.path.basename(gcs_path)
        local_path = self._local_file_handler.save_temp_file(obj, filename)

        # Upload to GCS
        full_gcs_path = self.upload_local_file(local_path, gcs_path)

        # Delete local file after upload to GCS
        self._local_file_handler.delete_file(local_path)
        return full_gcs_path

    def upload_dir(self, local_root: str, gcs_prefix: str, *, skip_hidden: bool = True) -> None:
        """
        Recursively upload every file under `local_root` to
        `gs://<bucket>/<gcs_prefix>/...`, keeping the relative
        sub-directory structure.

        Parameters:
            local_root (str): The directory you want to sync (e.g. "downloads/json/2504-17655v1").
            gcs_prefix (str): Destination path prefix inside the bucket (e.g. "data/papers/2504-17655v1").
            skip_hidden (bool): Skip files and dirs that start with '.'
        """
        local_root = Path(local_root).expanduser().resolve()
        if not local_root.is_dir():
            raise ValueError(f"{local_root} is not a directory")

        for path in local_root.rglob("*"):
            # skip directories
            if path.is_dir():
                continue

            # skip hidden files
            if skip_hidden and any(part.startswith(".") for part in path.parts):
                continue

            # relative path inside the local_root
            rel_path = path.relative_to(local_root)
            gcs_path = f"{gcs_prefix.rstrip('/')}/{rel_path.as_posix()}"

            # upload file to GCS
            _ = self.upload_local_file(str(path), gcs_path)

    def does_file_exist(self, gcs_path: str) -> bool:
        """
        Determine if a paper exists in GCS (via ArXiv entry_id)
        """
        blob = self.bucket.blob(gcs_path)
        return blob.exists()

    def docling_assets_exist(self, entry_id_path: str) -> bool:
        """
        Check whether both the Docling JSON and at least one artifact file exist under `entry_id_path`.

        Args:
            paper_prefix (str): e.g. "data/papers/2504-17655v1"  (no trailing slash)

        Returns bool determining if assets exist
        """
        # Check to see if <entry_id>.json file exists
        json_blob = f"{entry_id_path}/{Path(entry_id_path).name}.json"
        if not self.bucket.blob(json_blob).exists():
            return False

        # At least one artifact (png, jpg, etc.) in *_artifacts/ subfolder
        artifacts_prefix = f"{entry_id_path}/{Path(entry_id_path).name}_artifacts/"

        # list_blobs returns an iterator; fetch just one item
        iter = self.client.list_blobs(self.bucket_name, prefix=artifacts_prefix, max_results=1)
        try:
            next(iter)
            return True
        except StopIteration:
            return False

    def load_files(
        self,
        prefix: str,
        include: Tuple[str, ...] | None = None,
        exclude: Tuple[str, ...] | None = None,
    ) -> list[Any]:
        """
        List → download → deserialize → return Python objects.

        Example:
            metadata_files = gcs.load_files(
                "data/papers/",
                include=(".metadata.json",)
            )
        """
        paths = self._list_blob_names(prefix, include, exclude)
        objs = []
        for p in paths:
            try:
                objs.append(self.load_file(p))
            except Exception as e:
                logging.error("Failed to load %s: %s", p, e)
        return objs

    def download_files(
        self,
        prefix: str,
        include: Tuple[str, ...] | None = None,
        exclude: Tuple[str, ...] | None = None,
    ):
        """
        Downloads files from GCS locally.
        """
        paths = self._list_blob_names(prefix, include, exclude)
        for p in paths:
            try:
                self.download_file(p)
            except Exception as e:
                logging.error("Failed to download %s: %s", p, e)

    # def list_by_suffix(self, prefix: str, suffix: str) -> list[str]:
    #     """Return all blob names whose filename ends with `suffix`."""
    #     return self._list_blob_names(prefix, include=(suffix,))

    # def list_docling_json(self, prefix: str) -> list[str]:
    #     """All *.json that are *not* metadata."""
    #     return self._list_blob_names(
    #         prefix,
    #         include=(".json",),
    #         exclude=(".metadata.json",),
    #     )

    def list_metadata_json(self, prefix: str) -> list[str]:
        """All *.metadata.json objects under prefix."""
        return self._list_blob_names(prefix, include=(".metadata.json",))

    def download_docling_json_files(self, prefix: str) -> List[str]:
        """
        Download Docling files (stored as .json) locally from GCS (exclude .metadata.json files)
        """
        # Download to local directory
        self.download_files(prefix=prefix, include=(".json",), exclude=(".metadata.json",))
        local_dir = self._local_file_handler._get_local_dir(".json")
        return self._local_file_handler.list_local_file_names(
            prefix=local_dir, include=(".json",), exclude=(".metadata.json",)
        )

    def download_docling_and_metadta_files(self, prefix: str) -> List[str]:
        """
        Download Docling files (stored as .json) and metadata files (stored as .metadata.json) locally from GCS
        """
        # Download to local directory
        self.download_files(prefix=prefix, include=(".json",))
        local_dir = self._local_file_handler._get_local_dir(".json")
        return self._local_file_handler.list_local_file_names(
            prefix=local_dir, include=(".json",), exclude=(".metadata.json",)
        )

    def _list_blob_names(
        self,
        prefix: str,
        include: Tuple[str, ...] | None = None,
        exclude: Tuple[str, ...] | None = None,
    ) -> list[str]:
        """
        Return object *names* under `prefix`, filtered by simple substrings.

        Example:
            self._list_blob_names(
                "data/papers/",
                include=(".metadata.json",)
            ) -> ['data/papers/2504-17655v1/2504-17655v1.metadata.json', ...]
        """

        return [
            blob.name
            for blob in self.client.list_blobs(
                self.bucket_name,
                prefix=prefix.rstrip("/"),  # safety strip
            )
            if self._local_file_handler._match(blob.name, include, exclude)
        ]
