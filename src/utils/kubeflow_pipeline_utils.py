import logging
import time
from typing import Callable, Tuple
from datetime import datetime, timezone
from kfp import Client, compiler


class KubeflowPipelineUtils:
    """
    A utility class to compile, upload, version, and run Kubeflow Pipelines.

    Attributes:
        client (Client): The KFP SDK Client used to interact with the Kubeflow Pipelines API.
    """

    def __init__(self, host: str):
        """
        Initialize the RunPipeline object.

        Args:
            host (str): URL of the Kubeflow Pipelines API server.
        """
        self.client = Client(host=host)

    def compile_to_yaml(self, pipeline_func: Callable, pipeline_package_path: str) -> None:
        """
        Compile a pipeline function into a YAML file for upload.

        Args:
            pipeline_func (Callable): The pipeline function decorated with `@dsl.pipeline`.
            pipeline_package_path (str): Path to save the compiled YAML file.
        """
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=pipeline_package_path)
        logging.info(f"Pipeline compiled to {pipeline_package_path}")

    def upload_pipeline(
        self, pipeline_name: str, pipeline_package_path: str, incrementor: str
    ) -> Tuple[str, str]:
        """
        Upload a pipeline to Kubeflow Pipelines, either as a new pipeline or a new version.

        Args:
            pipeline_name (str): Name of the pipeline.
            pipeline_package_path (str): Path to the compiled pipeline YAML.
            incrementor (str): Strategy to increment version ("major" or "minor").

        Returns: Tuple[str, str] of (pipeline_id, pipeline_version_id).
        """
        pipeline_id = self.client.get_pipeline_id(pipeline_name)

        if pipeline_id:
            # Upload as new version if pipeline already exists
            logging.debug("Found pipeline_id, uploading new version.")
            version_name = self._get_version_name(pipeline_id, incrementor)
            pipeline_version = self.client.upload_pipeline_version(
                pipeline_package_path=pipeline_package_path,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id,
            )
            logging.info(
                f"Uploaded pipeline: {pipeline_name} version: {pipeline_version.display_name}"
            )
        else:
            # Upload as new pipeline if not found
            logging.debug("Uploading as new pipeline.")
            pipeline = self.client.upload_pipeline(
                pipeline_package_path, pipeline_name=pipeline_name
            )
            time.sleep(1.1)  # Slight delay to avoid race conditions
            pipeline_version = self.client.upload_pipeline_version(
                pipeline_package_path=pipeline_package_path,
                pipeline_version_name="1.0",
                pipeline_id=pipeline.pipeline_id,
            )
            logging.info(
                f"Uploaded pipeline: {pipeline_name} version: {pipeline_version.display_name}"
            )

        return pipeline_version.pipeline_id, pipeline_version.pipeline_version_id

    def run_kubeflow_pipeline(
        self, job_name: str, experiment_name: str, pipeline_id: str, pipeline_version_id: str
    ) -> None:
        """
        Submit a pipeline run to Kubeflow Pipelines.

        Args:
            job_name (str): Name of the job run.
            experiment_name (str): Name of the experiment to run under.
            pipeline_id (str): ID of the pipeline.
            pipeline_version_id (str): ID of the specific pipeline version to run.
        """
        experiment_id = self._get_or_create_experiment(experiment_name)
        run = self.client.run_pipeline(
            job_name=job_name,
            experiment_id=experiment_id,
            pipeline_id=pipeline_id,
            version_id=pipeline_version_id,
        )
        logging.info(f"Pipeline submitted. Run ID: {run.run_id}")

    def _get_latest_version(self, pipeline_id: str) -> str:
        """
        Find the latest version name for a given pipeline.

        Args:
            pipeline_id (str): ID of the pipeline.

        Returns: (str) The display name of the most recently created pipeline version.
        """
        pipeline_versions = self.client.list_pipeline_versions(
            pipeline_id=pipeline_id, page_size=100
        ).pipeline_versions

        max_version_timestamp = datetime.min.replace(tzinfo=timezone.utc)
        max_version_name = None

        for version in pipeline_versions:
            if version.created_at > max_version_timestamp:
                max_version_timestamp = version.created_at
                max_version_name = version.display_name

        return max_version_name

    def _get_version_name(self, pipeline_id: str, incrementor: str) -> str:
        """
        Generate a new version name based on the latest existing version.

        Args:
            pipeline_id (str): ID of the pipeline.
            incrementor (str): Strategy to increment version ("major" or "minor").

        Returns: (str) New version name, e.g., "2.0" or "1.1".
        """
        max_version_name = self._get_latest_version(pipeline_id)
        version_split = max_version_name.split(".")
        major_version = int(version_split[0])
        minor_version = int(version_split[1])

        if incrementor == "major":
            major_version += 1
            minor_version = 0  # reset minor on major bump
        else:
            minor_version += 1

        return f"{major_version}.{minor_version}"

    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """
        Retrieve an experiment ID, or create a new experiment if it does not exist.

        Args:
            experiment_name (str): Name of the experiment.

        Returns: (str) Experiment ID.
        """
        experiment_id = self.client.get_pipeline_id(experiment_name)
        if not experiment_id:
            experiment_id = self.client.create_experiment(name=experiment_name).experiment_id
        return experiment_id
