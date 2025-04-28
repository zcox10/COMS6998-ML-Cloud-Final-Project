import os
import json
import torch
import pyarrow.parquet as pq
from google.cloud import storage
import PyPDF2
from typing import Any, Dict, List
import logging

from src.utils.local_file_handler import LocalFileHandler
from src.utils.generic_utils import GenericUtils


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

        # General utils to enable logging and other utility functions
        self._generic_utils = GenericUtils()
        self._generic_utils.configure_component_logging(log_level=logging.INFO)

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
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")

        # delete local file after loading into memory
        self._local_file_handler.delete_file(local_path)
        return obj

    def upload_file(self, obj: Any, gcs_path: str) -> None:
        """
        Upload a local file to GCS.

        Args:
            local_path (str): Path to the local file on disk.
            gcs_path (str): Destination path inside the GCS bucket.
        """
        # Save file locally to downloads
        filename = os.path.basename(gcs_path)
        local_path = self._local_file_handler.save_temp_file(obj, filename)

        # Upload to GCS
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logging.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}\n")

        # Delete local file after upload to GCS
        self._local_file_handler.delete_file(local_path)

    def does_file_exist(self, gcs_path: str) -> bool:
        """
        Determine if a paper exists in GCS (via ArXiv entry_id)
        """
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
