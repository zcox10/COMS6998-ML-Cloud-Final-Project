import re
import os
import logging
from docling.datamodel.document import DoclingDocument

from src.utils.yaml_parser import YamlParser
from src.utils.gcs_file_handler import GcsFileHandler


class TextProcessingUtils:
    def __init__(self):
        # File handling
        self._config = YamlParser("./config.yaml")
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]
        self._gcs_file_handler = GcsFileHandler(bucket_name=self._gcs_bucket_name)
        self._local_file_handler = self._gcs_file_handler._local_file_handler
        self._local_dataset_directory = "downloads/json"
        self._local_markdown_directory = "downloads/markdown"

    def process_docling_files_to_markdown(self):
        file_names = self._gcs_file_handler.download_docling_json_files(self._gcs_data_directory)
        logging.info("All docling JSON files downloaded locally")

        for file_name in file_names:
            entry_id = file_name.split("/")[-1].split(".")[0]
            local_path = os.path.join(self._local_markdown_directory, entry_id + ".md")
            self.convert_docling_to_markdown(file_name, local_path)
            logging.info(f"{entry_id} converted to markdown and stored locally")

    def convert_docling_to_markdown(self, file, local_path):
        doc = DoclingDocument.load_from_json(file)
        doc_md = doc.export_to_markdown()
        final_md = self.clean_and_wrap_markdown(doc_md)
        self._local_file_handler.save_file(final_md, local_path)

    def clean_and_wrap_markdown(self, doc_md: str) -> str:
        # Define undesired sections
        undesired_sections = [
            "ACKNOWLEDGEMENTS",
            "ACKNOWLEDGEMENT",
            "ACKNOWLEDGMENT",
            "REFERENCE",
            "REFERENCES",
            "APPENDIX",
            "APPENDICES",
            "BIBLIOGRAPHY",
            "BIBLIOGRAPHIES",
        ]

        # Regex components:
        # - ^#{1,6}\s+: match any markdown heading level
        # - (?:[\w\d.]+\s+)? optional enumerator (e.g., "1. ", "I. ", "A. ")
        # - (ACKNOWLEDGEMENTS|...): one of the undesired sections
        pattern = re.compile(
            r"^#{1,6}\s+(?:[\w\d.]+\s+)?(" + "|".join(undesired_sections) + r")\b.*",
            re.IGNORECASE | re.MULTILINE,
        )

        # Find the start of the first undesired section
        match = pattern.search(doc_md)
        if match:
            doc_md = doc_md[: match.start()].rstrip()

        # Wrap with delimiters
        return f"<|startofpaper|>\n{doc_md}\n<|endofpaper|>"
