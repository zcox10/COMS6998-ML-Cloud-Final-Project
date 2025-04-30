import os
import json
import torch
import pyarrow.parquet as pq
import PyPDF2
import shutil
from typing import Any, Dict, List, Tuple

import logging


class LocalFileHandler:
    """
    Handles local file I/O operations based on file type: JSON, JSONL, PDF, Parquet, PyTorch, etc.
    """

    def load_file(self, local_path: str) -> Any:
        """
        Load a file into memory based on its suffix.
        """
        suffix = os.path.splitext(local_path)[-1]
        handler = self._dispatch_by_suffix(suffix, mode="load")
        return handler(local_path)

    def save_file(self, obj: Any, local_path: str) -> None:
        """
        Save an object to a file based on its suffix.
        """
        suffix = os.path.splitext(local_path)[-1]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        handler = self._dispatch_by_suffix(suffix, mode="save")
        handler(obj, local_path)

    def save_temp_file(self, obj: Any, filename: str) -> None:
        """
        Save an object to the appropriate downloads/ subfolder based on file suffix.

        Args:
            obj (Any): The object to save.
            filename (str): The filename to use (e.g., "sample.json").
        """
        suffix = os.path.splitext(filename)[-1]
        local_dir = self._get_local_dir(suffix)
        os.makedirs(local_dir, exist_ok=True)

        # Build full path: e.g., downloads/json/sample.json
        local_path = os.path.join(local_dir, filename)

        handler = self._dispatch_by_suffix(suffix, mode="save")
        handler(obj, local_path)
        return local_path

    def delete_file(self, local_path: str) -> None:
        """
        Delete a file from the local filesystem.

        Args:
            local_path (str): Path to the file to delete.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File {local_path} does not exist.")

        os.remove(local_path)

    def delete_directory_tree(self, root_dir: str) -> None:
        """
        Recursively delete all files and subdirectories under `root_dir`, including the root directory itself.

        Args:
            root_dir (str): Path to the root directory to delete.

        Raises:
            FileNotFoundError: If the root directory does not exist.
        """
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} does not exist.")

        if not os.path.isdir(root_dir):
            raise ValueError(f"Path {root_dir} is not a directory.")

        shutil.rmtree(root_dir)
        logging.info(f"Deleted directory tree: {root_dir}")

    def list_local_file_names(
        self,
        prefix: str,
        include: Tuple[str, ...] | None = None,
        exclude: Tuple[str, ...] | None = None,
    ) -> list[str]:
        """
        Return local file paths under `prefix`, filtered by substring rules.

        Args:
            prefix (str): Root directory to start walking from.
            include (Tuple[str, ...], optional): Only include files containing these substrings.
            exclude (Tuple[str, ...], optional): Exclude files containing these substrings.

        Returns:
            list[str]: List of full file paths matching the criteria.
        """
        matched_files = []
        for root, _, files in os.walk(prefix):
            for fname in files:
                full_path = os.path.join(root, fname)
                if self._match(full_path, include, exclude):
                    matched_files.append(full_path)
        return matched_files

    def _match(
        self,
        name: str,
        include: Tuple[str, ...] | None = None,
        exclude: Tuple[str, ...] | None = None,
    ) -> bool:
        """
        Determines match for filename based on `include` and `exclude` parameters
        """
        if include and not any(substr in name for substr in include):
            return False
        if exclude and any(substr in name for substr in exclude):
            return False
        return True

    def _dispatch_by_suffix(self, suffix: str, *, mode: str) -> callable:
        """
        Dispatch to the correct load/save function based on suffix and mode.

        Args:
            suffix (str): File suffix (e.g., ".json", ".parquet")
            mode (str): "load" or "save"

        Returns:
            callable: The appropriate method to handle the file.
        """
        load_handlers = {
            ".json": self._load_json,
            ".jsonl": self._load_jsonl,
            ".pdf": self._load_pdf,
            ".pth": self._load_pth,
            ".parquet": self._load_parquet,
            ".md": self._load_md,
        }
        save_handlers = {
            ".json": self._save_json,
            ".jsonl": self._save_jsonl,
            ".pth": self._save_pth,
            ".parquet": self._save_parquet,
            ".md": self._save_md,
        }

        if mode == "load":
            handler = load_handlers.get(suffix)
        elif mode == "save":
            handler = save_handlers.get(suffix)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if handler is None:
            raise ValueError(f"Unsupported file type for {mode}: {suffix}")

        return handler

    def _get_local_dir(self, file_suffix: str) -> str:
        """
        Get the local directory name based on file suffix.

        Args:
            file_suffix (str): File suffix including dot (e.g., '.json', '.pdf').

        Returns:
            str: Local directory path relative to current directory.
        """
        suffix_to_folder = {
            ".pdf": "downloads/pdf",
            ".json": "downloads/json",
            ".jsonl": "downloads/jsonl",
            ".pth": "downloads/pth",
            ".parquet": "downloads/parquet",
            ".md": "downloads/markdown",
        }
        return suffix_to_folder.get(file_suffix, "downloads/other")

    ############################## Logic for save/load files ##############################
    # JSON logic
    def _load_json(self, local_path: str) -> Dict:
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, obj: Dict, local_path: str) -> None:
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    # JSONL logic
    def _load_jsonl(self, local_path: str) -> List[Dict]:
        data = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _save_jsonl(self, obj: List[Dict], local_path: str) -> None:
        with open(local_path, "w", encoding="utf-8") as f:
            for line in obj:
                f.write(json.dumps(line) + "\n")

    # PDF logic
    def _load_pdf(self, local_path: str) -> str:
        with open(local_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

    # PTH logic
    def _load_pth(self, local_path: str) -> Any:
        return torch.load(local_path)

    def _save_pth(self, obj: Any, local_path: str) -> None:
        torch.save(obj, local_path)

    # Parquet logic
    def _load_parquet(self, local_path: str) -> Dict:
        table = pq.read_table(local_path).to_pydict()
        return table

    def _save_parquet(self, obj: Dict, local_path: str) -> None:
        table = pq.Table.from_pydict(obj)
        pq.write_table(table, local_path)

    # Markdown logic
    def _load_md(self, local_path: str) -> str:
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()

    def _save_md(self, content: str, local_path: str) -> None:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(content)
