"""
Places the variables from CONFIG_FILE defined in tf_whitelist into the TFVARS_FILE
"""

import sys
import pathlib

# Add root directory to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.utils.yaml_parser import YamlParser


def main():
    config = YamlParser("./config.yaml")

    # Define whitelist paths you want in tfvars
    tf_whitelist = [
        "gcp.project_id",
        "gcp.region",
        "gcp.gcs.buckets",
        "gcp.gke.cluster_name",
        "gcp.gke.machine_type",
        "gcp.gke.service_account_name",
        "gcp.gke.services.arxiv_summarization_api.ksa_name",
        "gcp.gke.services.arxiv_summarization_api.namespace",
    ]

    output_path = config.get_field("terraform.tfvars.prod.path")
    config.export_to_tfvars(tf_whitelist, output_path)


if __name__ == "__main__":
    main()
