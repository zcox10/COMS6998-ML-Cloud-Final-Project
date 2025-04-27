#!/usr/bin/env bash
set -eo pipefail

# Description: a script to auto-populate the TFVARS_FILE for execution with terraform via the CONFIG_FILE

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
CONFIG_FILE="$ROOT_DIR/config.yaml"
TFVARS_FILE="$ROOT_DIR/infra/kubernetes_setup.tfvars"

# keys that must appear in TFVARS_FILE
TF_WHITELIST=(
    gcp_project_id
    gcp_region
    gcs_bucket_name
    gcp_service_account_key_path
    gke_cluster_name
)

# map whitelist-key -> YAML path
map_key_to_yaml_path() {
    case "$1" in
    gcp_project_id) echo "gcp.project_id" ;;
    gcp_region) echo "gcp.region" ;;
    gcs_bucket_name) echo "gcp.storage.bucket" ;;
    gcp_service_account_key_path) echo "gcp.service_account_key" ;;
    gke_cluster_name) echo "gcp.cluster" ;;
    *) echo "" ;;
    esac
}

# preconditions
command -v yq >/dev/null || {
    echo "yq not found"
    exit 1
}
[[ -f "$CONFIG_FILE" ]] || {
    echo "$CONFIG_FILE missing"
    exit 1
}

# write to TFVARS_FILE
{
    echo "# Auto generated from $(basename "$CONFIG_FILE")"

    for key in "${TF_WHITELIST[@]}"; do
        yaml_path=$(map_key_to_yaml_path "$key")
        [[ -z $yaml_path ]] && {
            echo "No YAML mapping for $key" >&2
            continue
        }

        value=$(yq -r ".${yaml_path}" "$CONFIG_FILE" | envsubst)

        [[ -z $value ]] && {
            echo "Missing value for path $yaml_path" >&2
            continue
        }

        echo "${key} = \"${value}\""
    done
} >"$TFVARS_FILE"

echo -e "\n========== Terraform tfvars generated at $(basename "$TFVARS_FILE") ==========\n"
