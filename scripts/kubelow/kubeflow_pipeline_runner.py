import sys
import pathlib
from kfp import dsl
from datetime import datetime
import logging
from typing import Dict

# Add root directory to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))


@dsl.component
def arxiv_data_collection(query: str, max_results: int) -> Dict[str, str]:
    """
    Stream PDFs from ArXiV for storage in GCS
    """
    import logging
    from src.utils.generic_utils import GenericUtils
    from src.data_processing.arxiv.arxiv_data_collection import ArxivDataCollection

    # Enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    # Run data collection
    _ = ArxivDataCollection().fetch_papers(query=query, max_results=max_results)
    return {"status": "complete"}


@dsl.component
def docling_pdf_processing(device: str) -> Dict[str, str]:
    """
    Process PDFs from ArXiv with Docling and store output in GCS
    """
    import logging
    from src.utils.generic_utils import GenericUtils
    from src.data_processing.docling.docling_pdf_processing import DoclingPdfProcessing

    # Enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    # Run pdf processing job
    DoclingPdfProcessing(device=device).store_documents_in_gcs()
    return {"status": "complete"}


@dsl.component
def generate_fine_tune_dataset() -> Dict[str, str]:
    """
    Generate the dataset for fine-tuning and store in GCS
    """
    import logging
    from src.utils.generic_utils import GenericUtils
    from src.fine_tune.generate_dataset import GenerateDataset

    # Enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    gcs_uri = GenerateDataset().generate_docling_dataset()
    return {"gcs_uri": gcs_uri}


@dsl.component
def embed_text_chunks() -> Dict[str, str]:
    """
    Generate text embeddings on chunked text. Store in Qdrant vector db on GKE
    """
    import logging
    from src.rag.vector_embeddings import VectorEmbeddings
    from src.utils.generic_utils import GenericUtils
    from src.utils.yaml_parser import YamlParser

    # Enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    # Retrieve vector db url
    config = YamlParser("./config.yaml")
    vector_db_url = config.get_field("gcp.gke.services.vector_db.url")

    # Run chunking process and upsert vector embeddings
    VectorEmbeddings(vector_db_url).upsert_vector_embeddings()
    return {"status": "complete"}


@dsl.component
def fine_tune_model() -> Dict[str, str]:
    """
    Fine-tune model
    """

    print("model fine tuning")
    gcs_uri = "gs://fine-tuning"
    return {"status": "complete"}


# define Kubeflow pipeline
@dsl.pipeline(name="ml-cloud-pipeline")
def pipeline():
    # Initialize arguments for each component; empty dict implies no arguments
    use_global_cache = False
    use_gpu_device = True
    task_args = {
        "arxiv_data_collection_task": {"query": "artificial intelligence", "max_results": 10},
        "docling_pdf_processing_task": {},
        "generate_fine_tune_dataset": {},
        "embed_text_chunks_task": {},
        "fine_tune_model_task": {},
    }

    # Initialize config for setting GCR image for each component
    from src.utils.yaml_parser import YamlParser

    config = YamlParser("./config.yaml")
    kubeflow_image_cpu = config.get_field("gcp.gke.services.kubeflow.images.cpu")
    kubeflow_image_gpu = config.get_field("gcp.gke.services.kubeflow.images.gpu")

    # # arxiv_data_collection_task
    # arxiv_data_collection_task = arxiv_data_collection(
    #     query=task_args["arxiv_data_collection_task"]["query"],
    #     max_results=task_args["arxiv_data_collection_task"]["max_results"],
    # )
    # arxiv_data_collection_task.set_caching_options(use_global_cache)
    # arxiv_data_collection_task.container_spec.image = kubeflow_image_cpu

    # # docling_pdf_processing_task
    # docling_pdf_processing_task = docling_pdf_processing(device="cuda" if use_gpu_device else "cpu")
    # docling_pdf_processing_task.after(arxiv_data_collection_task)
    # docling_pdf_processing_task.set_caching_options(use_global_cache)
    # docling_pdf_processing_task.container_spec.image = kubeflow_image_cpu

    # if use_gpu_device:
    #     docling_pdf_processing_task.container_spec.image = kubeflow_image_gpu
    #     docling_pdf_processing_task.set_gpu_limit(1)
    #     docling_pdf_processing_task.set_accelerator_type(accelerator="nvidia.com/gpu")

    # # generate_fine_tune_dataset_task
    # generate_fine_tune_dataset_task = generate_fine_tune_dataset()
    # generate_fine_tune_dataset_task.after(docling_pdf_processing_task)
    # generate_fine_tune_dataset_task.set_caching_options(use_global_cache)
    # generate_fine_tune_dataset_task.container_spec.image = kubeflow_image_cpu

    # embed_text_chunks_task
    embed_text_chunks_task = embed_text_chunks()
    # embed_text_chunks_task.after(generate_fine_tune_dataset_task)
    embed_text_chunks_task.set_caching_options(use_global_cache)
    embed_text_chunks_task.container_spec.image = kubeflow_image_cpu

    # fine_tune_model_task
    fine_tune_model_task = fine_tune_model()
    fine_tune_model_task.after(embed_text_chunks_task)
    fine_tune_model_task.set_caching_options(use_global_cache)
    fine_tune_model_task.container_spec.image = kubeflow_image_cpu

    if use_gpu_device:
        fine_tune_model_task.container_spec.image = kubeflow_image_gpu
        fine_tune_model_task.set_gpu_limit(1)
        fine_tune_model_task.set_accelerator_type(accelerator="nvidia.com/gpu")


if __name__ == "__main__":
    from src.utils.kubeflow_pipeline_utils import KubeflowPipelineUtils
    from src.utils.yaml_parser import YamlParser

    logging.basicConfig(level=logging.DEBUG, format="\n%(levelname)s: %(message)s\n")

    # constants
    config = YamlParser("./config.yaml")
    kubeflow_pipeline_package_path = config.get_field("gcp.gke.services.kubeflow.pipeline_path")
    pipeline_name = config.get_field("gcp.gke.services.kubeflow.pipeline_name")
    experiment_name = config.get_field("gcp.gke.services.kubeflow.experiment_name")
    host = config.get_field("gcp.gke.services.kubeflow.host")

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
