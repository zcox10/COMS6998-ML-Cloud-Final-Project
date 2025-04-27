from kfp import dsl
from datetime import datetime
import logging
from typing import Dict

from src.utils.yaml_parser import YamlParser
from src.utils.kubeflow_pipeline_utils import KubeflowPipelineUtils

config = YamlParser("./config.yaml")
gcr_image = config.get_field("gcp.image")


@dsl.component(base_image=gcr_image)
def arxiv_data_collection(query: str, max_results: int) -> None:
    """
    Stream PDFs from ArXiV for storage in GCS
    """
    # imports
    import logging

    from src.utils.generic_utils import GenericUtils
    from src.data_processing.arxiv.arxiv_data_collection import ArxivDataCollection

    # enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    # Empty return
    _ = ArxivDataCollection().fetch_papers(query=query, max_results=max_results)
    return None


@dsl.component(base_image=gcr_image)
def docling_text_extract() -> Dict[str, str]:
    """
    Extract text from PDFs stored in GCS and store output plus paper metadata
    """

    print("docling_extract")
    gcs_uri = "gs://docling-extract"
    return {"data_path": gcs_uri}


@dsl.component(base_image=gcr_image)
def embed_text_chunks() -> Dict[str, str]:
    """
    Generate text embeddings on chunked text. Store in GCS
    """

    print("embed_chunks")
    gcs_uri = "gs://embed-chunks"
    return {"data_path": gcs_uri}


@dsl.component(base_image=gcr_image)
def fine_tune_model() -> Dict[str, str]:
    """
    Fine-tune model
    """

    print("model fine tuning")
    gcs_uri = "gs://fine-tuning"
    return {"model_path": gcs_uri}


# define Kubeflow pipeline
@dsl.pipeline(name="ml-cloud-pipeline")
def pipeline():
    # Initialize arguments
    global_cache = False

    # Run data collection component
    arxiv_data_collection_task = arxiv_data_collection(
        query="artificial intelligence", max_results=10
    )
    arxiv_data_collection_task.set_caching_options(global_cache)

    # Run Docling extract component
    docling_text_extract_task = docling_text_extract().after(arxiv_data_collection_task)
    docling_text_extract_task.set_caching_options(global_cache)

    # Run embed chunks component
    embed_text_chunks_task = embed_text_chunks().after(docling_text_extract_task)
    embed_text_chunks_task.set_caching_options(global_cache)

    # Run fine-tuning component
    fine_tune_model_task = fine_tune_model().after(embed_text_chunks_task)
    fine_tune_model_task.set_caching_options(global_cache)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="\n%(levelname)s: %(message)s\n")

    # constants
    kubeflow_pipeline_package_path = config.get_field("gcp.kubeflow.pipeline_package_path")
    pipeline_name = config.get_field("gcp.kubeflow.pipeline_name")
    experiment_name = config.get_field("gcp.kubeflow.experiment_name")
    host = config.get_field("gcp.kubeflow.host")

    # Job name versioning
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"ml-cloud-data-run-{timestamp}"

    # Compile pipeline to YAML, upload to Kubeflow, and run pipeline
    k = KubeflowPipelineUtils(host=host)
    k.compile_to_yaml(pipeline, kubeflow_pipeline_package_path)
    pipeline_id, pipeline_version_id = k.upload_pipeline(
        pipeline_name, kubeflow_pipeline_package_path, incrementor="minor"
    )
    k.run_kubeflow_pipeline(job_name, experiment_name, pipeline_id, pipeline_version_id)
