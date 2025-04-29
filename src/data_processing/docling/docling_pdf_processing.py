import os
import re
import logging
import time
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling_core.types.doc import ImageRefMode
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.local_file_handler import LocalFileHandler
from src.utils.yaml_parser import YamlParser
from src.utils.arxiv_utils import ArxivUtils
from src.utils.generic_utils import GenericUtils


class DoclingPdfProcessing:
    def __init__(self, device: str):
        self.device = device

        # Configure file handlers and utils
        self._config = YamlParser("./config.yaml")
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]
        self._gcs_file_handler = GcsFileHandler(bucket_name=self._gcs_bucket_name)
        self._local_file_handler = LocalFileHandler()

        # Utils
        self._generic_utils = GenericUtils()
        self._generic_utils.configure_component_logging(log_level=logging.INFO)
        self._arxiv_utils = ArxivUtils()

    def store_documents_in_gcs(self):
        # Define Document converter options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self._set_pipeline_options())
            }
        )

        for url in self._iter_arxiv_pdf_urls(prefix=self._gcs_data_directory):
            # Extract formatted entry id and gcs_root to store assets to in GCS
            formatted_entry_id = self._arxiv_utils.extract_formatted_entry_id_from_url(url)
            gcs_root = f"{self._gcs_data_directory}/{formatted_entry_id}"
            logging.info(f"Processing entry_id: {formatted_entry_id}; url: {url}")

            # Skip if Docling assets already exist
            if self._gcs_file_handler.docling_assets_exist(gcs_root):
                logging.info(f"{formatted_entry_id} Docling assets already exist in GCS")
                continue

            # Convert to Docling document assets
            start = time.time()
            doc = converter.convert(url).document

            # Generate local_path to save Docling JSON locally and local_root (used for GCS upload later)
            local_root, local_path = self._retrieve_local_json_path(formatted_entry_id)
            doc.save_as_json(local_path, image_mode=ImageRefMode.REFERENCED)

            # Upload assets to GCS
            self._gcs_file_handler.upload_dir(local_root, gcs_root, skip_hidden=True)

            end = time.time()
            total_time = round(end - start, 2)
            logging.info(f"Uploaded {url} Docling assets to GCS in {total_time} sec")
        return None

    def _retrieve_local_json_path(self, formatted_entry_id: str):
        # Create file name from formatted_entry_id
        filename = f"{formatted_entry_id}.json"

        # Derive suffix
        suffix = os.path.splitext(filename)[-1]

        # Find local dir to store JSON files to and create the dir if it does not exist
        local_dir = self._local_file_handler._get_local_dir(suffix)

        # Define local root
        local_root = os.path.join(local_dir, formatted_entry_id)
        os.makedirs(local_root, exist_ok=True)

        # Define local path
        local_path = os.path.join(local_root, filename)

        return local_root, local_path

    def _iter_arxiv_pdf_urls(self, prefix: str):
        """
        Iterate through the `prefix` directory and extract all ArXiv PDF url's
        """
        # Define entry_id_regex matching pattern
        escaped_prefix: str = re.escape(prefix.rstrip("/"))
        pattern_str: str = rf"{escaped_prefix}/([^/]+)/"
        entry_id_regex: re.Pattern = re.compile(pattern_str)

        for blob_name in self._gcs_file_handler.list_metadata_json(prefix):
            m = entry_id_regex.search(blob_name)
            if not m:
                continue
            formatted_entry_id = m.group(1)
            entry_id = formatted_entry_id.replace("-", ".")
            yield f"https://arxiv.org/pdf/{entry_id}"

    def _set_pipeline_options(self):
        """
        Defines pipeline options for PDF processing. Currently includes OCR,
        extracts table structure, code enrichment, formula enrichment,
        and image classification/description.
        """

        # options for device: CPU, MPS, AUTO, CUDA
        if self.device == "cuda":
            device = AcceleratorDevice.CUDA
            # cuda_use_flash_attention2 = True
            cuda_use_flash_attention2 = False
        else:
            device = AcceleratorDevice.CPU
            cuda_use_flash_attention2 = False

        # Set accelerator options
        num_cpus = max(1, os.cpu_count() - 1)
        accelerator_options = AcceleratorOptions(
            num_threads=num_cpus,
            device=device,
            cuda_use_flash_attention2=cuda_use_flash_attention2,
        )
        logging.info(f"Device for Docling PDF processing: {accelerator_options.device}")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        # pipeline_options.do_code_enrichment = True
        # pipeline_options.do_formula_enrichment = True
        # pipeline_options.generate_picture_images = True
        # pipeline_options.do_picture_classification = True
        # pipeline_options.do_picture_description = True

        return pipeline_options
