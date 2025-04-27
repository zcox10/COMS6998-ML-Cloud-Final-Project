import json
import logging
from typing import List
import arxiv
import time
from google.cloud import storage

from src.utils.generic_utils import GenericUtils
from src.utils.yaml_parser import YamlParser
from src.data_processing.arxiv.arxiv_category_taxonomy import ArxivCategoryTaxonomy


class ArxivDataCollection:
    """
    Download papers from ArXiv and store them with metadata to GCS.
    """

    def __init__(self):
        self._config = YamlParser("./config.yaml").load_config()
        self._gcs_bucket_name = self._config["gcs_bucket_name"]
        self._gcs_output_base = self._config["gcs_file_paths"]["data_path"]
        self._storage_client = storage.Client()
        self._bucket = self._storage_client.bucket(self._config["gcs_bucket_name"])
        self._category_taxonomy = ArxivCategoryTaxonomy().retrieve_taxonomy()
        self._utils = GenericUtils()
        self._utils.configure_component_logging(log_level=logging.INFO)

    def fetch_papers(self, query: str, max_results: int = 10) -> List[str]:
        """
        Fetch papers from ArXiv, saving metadata to GCS,
        continuing pagination until `max_results` new papers are stored.

        Returns: `List[str]` of ArXiv entry_ids that were downloaded.
        """
        client = arxiv.Client()
        downloaded_entry_ids = []
        total_checked = 0
        page_size = 100  # number of results to pull per request

        while len(downloaded_entry_ids) < max_results:
            search = arxiv.Search(
                query=query,
                max_results=page_size,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
                start=total_checked,
            )
            time.sleep(2)

            results = list(client.results(search))

            # stop if Arxiv has no more papers matching query
            if not results:
                logging.warning("No more results available from Arxiv.")
                break

            for result in results:
                formatted_entry_id = self._format_entry_id(result.entry_id)
                logging.info(f"Processing {result.entry_id}: {result.title}")

                # track how many results checked overall
                total_checked += 1

                if self._does_paper_exist_in_gcs(formatted_entry_id):
                    logging.info(f"Paper {result.entry_id} already exists in GCS. Skipping.\n")
                    continue

                # Build GCS path: data/papers/<entry_id>/
                gcs_paper_path = f"{self._gcs_output_base}/{formatted_entry_id}"

                # Save Metadata to GCS
                metadata = self._extract_metadata(result)
                metadata_json = json.dumps(metadata, indent=2).encode("utf-8")
                self._utils.save_asset_to_gcs(
                    asset=metadata_json,
                    gcs_bucket_name=self._gcs_bucket_name,
                    gcs_output_path=gcs_paper_path,
                    save_filename_prefix=f"{formatted_entry_id}.metadata.json",
                )

                downloaded_entry_ids.append(formatted_entry_id)

                # stop inner loop once we hit max_results
                if len(downloaded_entry_ids) >= max_results:
                    break

        logging.info(f"Downloaded {len(downloaded_entry_ids)} new papers.")

        return downloaded_entry_ids

    def _format_entry_id(self, entry_id: str):
        """
        Parses entry id (e.g., `http://arxiv.org/abs/2504.17782v1`) to output `2504-17782v1`
        """
        return entry_id.split("/")[-1].replace(".", "-")

    def _does_paper_exist_in_gcs(self, entry_id: str) -> bool:
        """
        Determine if a paper exists in GCS (via ArXiv entry_id)
        """
        prefix = f"{self._gcs_output_base}/{entry_id}/"
        blobs = list(self._bucket.list_blobs(prefix=prefix))
        return len(blobs) > 0

    def _extract_metadata(self, result) -> dict:
        """
        Extract metadata on an individual ArXiv PDF paper
        """
        return {
            "title": result.title,
            "entry_id": self._format_entry_id(result.entry_id),
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
