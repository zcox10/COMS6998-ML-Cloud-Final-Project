import logging
from typing import List
import arxiv

from src.utils.generic_utils import GenericUtils
from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.yaml_parser import YamlParser
from src.utils.arxiv_utils import ArxivUtils
from src.data_processing.arxiv.arxiv_category_taxonomy import ArxivCategoryTaxonomy


class ArxivDataCollection:
    """
    Download papers from ArXiv and store them with metadata to GCS.
    """

    def __init__(self):
        # Retrieve Arxiv category taxonomy map
        self._category_taxonomy = ArxivCategoryTaxonomy().retrieve_taxonomy()

        # General utils to enable logging and other utility functions
        self._generic_utils = GenericUtils()
        self._generic_utils.configure_component_logging(log_level=logging.INFO)
        self._arxiv_utils = ArxivUtils()

        # File handling
        self._config = YamlParser("./config.yaml")
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]
        self._gcs_file_handler = GcsFileHandler(bucket_name=self._gcs_bucket_name)

    def fetch_papers(self, query: str, max_results: int = 10) -> List[str]:
        """
        Fetch papers from ArXiv, saving metadata to GCS,
        continuing pagination until `max_results` new papers are stored.

        Returns: `List[str]` of ArXiv entry_ids that were downloaded.
        """

        client = arxiv.Client()
        downloaded_entry_ids = []

        search = arxiv.Search(
            query=query,
            max_results=max_results * 10,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        results = client.results(search)

        for result in results:
            # Create formatted entry_id suitable for file paths and the gcs_path
            formatted_entry_id = self._arxiv_utils.extract_formatted_entry_id_from_url(
                result.entry_id
            )
            gcs_path = f"{self._gcs_data_directory}/{formatted_entry_id}/{formatted_entry_id}.metadata.json"

            if self._gcs_file_handler.does_file_exist(gcs_path):
                logging.info(f"Paper {result.entry_id} already exists in GCS. Skipping.\n")
                continue

            # Extract metadata, dump to JSON, and upload to GCS
            metadata = self._extract_paper_metadata(result)
            self._gcs_file_handler.upload_asset(metadata, gcs_path)

            # Append to downloaded_entry_ids
            downloaded_entry_ids.append(formatted_entry_id)

            if len(downloaded_entry_ids) >= max_results:
                break

        logging.info(f"Downloaded {len(downloaded_entry_ids)} new papers.\n")
        return downloaded_entry_ids

    def _extract_paper_metadata(self, result) -> dict:
        """
        Extract metadata on an individual ArXiv PDF paper
        """
        return {
            "title": result.title,
            "entry_id": result.entry_id,
            "published": result.published.isoformat() if result.published else None,
            "updated": result.updated.isoformat() if result.updated else None,
            "summary": result.summary,
            "primary_category": {
                "id": result.primary_category,
                "name": self._category_taxonomy.get(result.primary_category, "Unknown"),
            },
            "categories": [
                {"id": cat, "name": self._category_taxonomy.get(cat, "Unknown")}
                for cat in result.categories
            ],
            "comment": result.comment,
            "journal_ref": result.journal_ref,
            "doi": result.doi,
            "arxiv_url": result.entry_id,
            "pdf_url": result.pdf_url,
        }
