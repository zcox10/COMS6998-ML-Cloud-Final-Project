import logging
import json
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling.datamodel.document import DoclingDocument


from src.utils.yaml_parser import YamlParser
from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.text_processing_utils import TextProcessingUtils
from src.utils.embedding_model_utils import EmbeddingModelUtils


class VectorEmbeddings:
    def __init__(self, vector_db_url):
        # Load configuration from YAML
        self._config = YamlParser("./config.yaml")

        # GCS handler
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]
        self._gcs_file_handler = GcsFileHandler(bucket_name=self._gcs_bucket_name)

        # Utility classes
        self._text_processing = TextProcessingUtils()
        self._embedding = EmbeddingModelUtils(vector_db_url)

    def upsert_vector_embeddings(self):
        """
        Main entry point:
        1. Download Docling JSON and metadata files from GCS
        2. For each document:
           a. Derive entry_id from filename
           b. Chunk the document into smaller passages
           c. Compute and upsert embeddings via EmbeddingModelUtils
        Logs errors and skips problematic files.
        """

        # Download both .json (Docling) and .metadata.json files
        file_paths = self._gcs_file_handler.download_docling_and_metadta_files(
            self._gcs_data_directory
        )

        for file_path in file_paths:
            entry_id = self._retrieve_entry_id_from_file_path(file_path)

            try:
                docs = self._chunk_document(file_path)
                self._embedding.upsert_document_embedding(docs, entry_id)
            except Exception as e:
                logging.error(f"Skipping {entry_id} due to error: {e}")

        self._embedding.get_collection_point_count()

    def _retrieve_entry_id_from_file_path(self, file_path: str):
        # retrieve entry_id: "downloads/json/2504-19990v1.json" -> "2504-19990v1"
        return file_path.split("/")[-1].split(".")[0]

    def _chunk_document(self, file_path: str) -> List[Document]:
        """
        Load a Docling JSON document, clean its markdown, and split into chunks.

        Args:
            file_path (str): Path to the Docling .json file

        Returns:
            List[Document]: LangChain Document objects with metadata attached
        """

        # Load metadata
        metadata = json.load(open(file_path.replace(".json", ".metadata.json")))

        # Load and clean markdown
        doc = DoclingDocument.load_from_json(file_path)
        full_md = self._text_processing.clean_and_wrap_markdown(doc.export_to_markdown())

        # Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.create_documents([full_md], metadatas=[metadata])
