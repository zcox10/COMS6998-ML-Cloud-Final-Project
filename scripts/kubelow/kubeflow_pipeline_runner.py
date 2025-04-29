import sys
import pathlib
from kfp import dsl
from datetime import datetime
import logging
from typing import Dict

# Add root directory to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.utils.yaml_parser import YamlParser
from src.utils.kubeflow_pipeline_utils import KubeflowPipelineUtils

config = YamlParser("./config.yaml")
kubeflow_image_cpu = config.get_field("gcp.gke.services.kubeflow.images.cpu")
kubeflow_image_gpu = config.get_field("gcp.gke.services.kubeflow.images.gpu")


@dsl.component(base_image=kubeflow_image_cpu)
def arxiv_data_collection(query: str, max_results: int) -> Dict[str, str]:
    """
    Stream PDFs from ArXiV for storage in GCS
    """
    # imports
    import logging

    from src.utils.generic_utils import GenericUtils
    from src.data_processing.arxiv.arxiv_data_collection import ArxivDataCollection

    # Enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    # Run data collection
    _ = ArxivDataCollection().fetch_papers(query=query, max_results=max_results)
    return {"status": "complete"}


@dsl.component(base_image=kubeflow_image_cpu)
def docling_pdf_processing(device: str) -> Dict[str, str]:
    """
    Process PDFs from ArXiv with Docling and store output in GCS
    """

    # imports
    import logging

    from src.utils.generic_utils import GenericUtils
    from src.data_processing.docling.docling_pdf_processing import DoclingPdfProcessing

    # Enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    # Run pdf processing job
    DoclingPdfProcessing(device=device).store_documents_in_gcs()
    return {"status": "complete"}


@dsl.component(base_image=kubeflow_image_cpu)
def generate_fine_tune_dataset() -> Dict[str, str]:
    """
    Generate the dataset for fine-tuning and store in GCS
    """
    # imports
    import logging

    from src.utils.generic_utils import GenericUtils
    from src.fine_tune.generate_dataset import GenerateDataset

    # Enable logging
    GenericUtils().configure_component_logging(log_level=logging.INFO)

    gcs_uri = GenerateDataset().generate_docling_dataset()

    return {"gcs_uri": gcs_uri}


@dsl.component(base_image=kubeflow_image_cpu)
def embed_text_chunks() -> Dict[str, str]:
    """
    Generate text embeddings on chunked text. Store in GCS
    """

    print("embed_chunks")
    gcs_uri = "gs://embed-chunks"
    return {"status": "complete"}


@dsl.component(base_image=kubeflow_image_cpu)
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
    # Initialize arguments
    global_cache = False
    global_device = "cpu"
    task_args = {
        "arxiv_data_collection_task": {
            "query": "artificial intelligence",
            "max_results": 2,
        },
        "docling_pdf_processing_task": {
            "device": global_device,
        },
        "generate_fine_tune_dataset": {
            "device": global_device,
        },
        "embed_text_chunks_task": {
            "device": global_device,
        },
        "fine_tune_model_task": {
            "device": global_device,
        },
    }

    # # Run data collection component
    # arxiv_data_collection_task = arxiv_data_collection(
    #     query=task_args["arxiv_data_collection_task"]["query"],
    #     max_results=task_args["arxiv_data_collection_task"]["max_results"],
    # )
    # arxiv_data_collection_task.set_caching_options(global_cache)

    # # Run Docling extract component
    # docling_pdf_processing_task = docling_pdf_processing(
    #     device=task_args["docling_pdf_processing_task"]["device"]
    # )
    # docling_pdf_processing_task.set_caching_options(global_cache)
    # docling_pdf_processing_task.after(arxiv_data_collection_task)

    # if task_args["docling_pdf_processing_task"]["device"] == "cuda":
    #     docling_pdf_processing_task.set_gpu_limit(1)
    #     docling_pdf_processing_task.set_accelerator_type(accelerator="nvidia.com/gpu")

    # Generate fine-tune dataset component
    generate_fine_tune_dataset_task = generate_fine_tune_dataset()
    # generate_fine_tune_dataset_task.after(docling_pdf_processing_task)
    generate_fine_tune_dataset_task.set_caching_options(global_cache)

    # Run embed chunks component
    embed_text_chunks_task = embed_text_chunks()
    embed_text_chunks_task.after(generate_fine_tune_dataset_task)
    embed_text_chunks_task.set_caching_options(global_cache)

    if task_args["embed_text_chunks_task"]["device"] == "cuda":
        embed_text_chunks_task.set_gpu_limit(1)
        embed_text_chunks_task.set_accelerator_type(accelerator="nvidia.com/gpu")

    # Run fine-tuning component
    fine_tune_model_task = fine_tune_model()
    fine_tune_model_task.set_caching_options(global_cache)
    fine_tune_model_task.after(embed_text_chunks_task)

    if task_args["fine_tune_model_task"]["device"] == "cuda":
        fine_tune_model_task.set_gpu_limit(1)
        fine_tune_model_task.set_accelerator_type(accelerator="nvidia.com/gpu")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="\n%(levelname)s: %(message)s\n")

    # constants
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
