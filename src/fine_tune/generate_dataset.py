import logging
from typing import List
import os
from docling.datamodel.document import DoclingDocument

from src.utils.yaml_parser import YamlParser
from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.local_file_handler import LocalFileHandler
from src.utils.text_processing_utils import TextProcessingUtils


class GenerateDataset:
    """
    Handles the end-to-end process of generating a fine-tuning dataset
    from Docling JSON documents:
        - Download documents from GCS
        - Process and clean text
        - Save dataset locally as a Markdown file
        - Upload the final dataset back to GCS
    """

    def __init__(self):
        """
        Initializes file handlers, configuration, and utility classes.
        Sets up local and GCS file paths for the dataset.
        """

        # Load configuration from YAML
        self._config = YamlParser("./config.yaml")

        # Setup GCS paths
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]

        # Handlers
        self._gcs_file_handler = GcsFileHandler(bucket_name=self._gcs_bucket_name)
        self._local_file_handler = LocalFileHandler()

        # Local dataset paths
        self._local_dataset_directory = "downloads/json"
        self._local_dataset_filename = "docling_fine_tune_dataset.md"
        self._local_dataset_file_path = os.path.join(
            self._local_dataset_directory, self._local_dataset_filename
        )

        # GCS dataset upload path
        self._gcs_dataset_file_path = os.path.join(
            self._config.get_field("gcp.gcs.buckets")[0]["paths"]["fine_tune"]["data"],
            self._local_dataset_filename,
        )

        # Utility classes
        self._text_processing = TextProcessingUtils()

    def generate_docling_dataset(self) -> str:
        """
        Executes the full dataset generation pipeline:
            1. Download Docling JSON files from GCS
            2. Process and clean the text
            3. Save as a Markdown file locally
            4. Upload the final Markdown file back to GCS
            5. Clean up local temporary files

        Returns:
            str: GCS path to the uploaded fine-tuning dataset
        """
        # Download Docling JSON files from GCS
        file_paths = self._gcs_file_handler.download_docling_json_files(self._gcs_data_directory)

        # Process Docling JSON files and save final output as Markdown
        self._write_docling_json_to_local_file(
            file_paths, output_path=self._local_dataset_file_path
        )

        # Upload processed Markdown file to GCS
        gcs_final_path = self._gcs_file_handler.upload_local_file(
            local_path=self._local_dataset_file_path,
            gcs_path=self._gcs_dataset_file_path,
        )

        # Delete local dataset files
        self._local_file_handler.delete_directory_tree(root_dir=self._local_dataset_directory)

        return gcs_final_path

    def _write_docling_json_to_local_file(self, file_paths: List[str], output_path: str) -> None:
        """
        Processes a list of Docling JSON files:
            - Converts them to Markdown
            - Cleans and wraps the text
            - Writes the output sequentially to a single Markdown file

        Args:
            file_paths (List[str]): List of paths to Docling JSON files.
            output_path (str): Path to save the processed Markdown file.
        """
        with open(output_path, "w", encoding="utf-8") as outfile:
            for file in file_paths:
                try:
                    doc = DoclingDocument.load_from_json(file)
                    doc_md = doc.export_to_markdown()
                    final_md = self._text_processing.clean_and_wrap_markdown(doc_md)
                    outfile.write(final_md + "\n\n")  # Separate each document with spacing
                except Exception as e:
                    logging.error(f"Skipping file due to error: {file}\nError: {e}")
