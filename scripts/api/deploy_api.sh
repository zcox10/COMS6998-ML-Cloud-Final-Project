#!/bin/bash

# Description: a bash script to run a Kubeflow pipeline:
# - Deletes old image containers
# - Pushes the latest Docker image
# - Runs the Kubeflow pipeline

set -euo pipefail

# Retrieve variables from CONFIG_FILE
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$CURRENT_DIR/../.."
CONFIG_FILE="$ROOT_DIR/config.yaml"

ARXIV_SUMMARIZATION_API_YAML_FILE="$ROOT_DIR/infra/k8s/arxiv-summarization-api.yaml"
ARXIV_SUMMARIZATION_API_DEPLOYMENT="arxiv-summarization-api-deployment"
ARXIV_SUMMARIZATION_API_SERVICE="arxiv-summarization-api-service"
ARXIV_SUMMARIZATION_API_NAMESPACE="arxiv-summarization-api"

DOCKERFILE=$(yq -r '.dockerfile.api' "$CONFIG_FILE")
IMAGE_NAME=$(yq -r '.gcp.gke.services.arxiv_summarization_api.image' "$CONFIG_FILE")

# Retrieve variables from CONFIG_FILE
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$CURRENT_DIR/../.."
CONFIG_FILE="$ROOT_DIR/config.yaml"
BUILD_FILE="$ROOT_DIR/infra/cloud_builds/cloud-build.yaml"

delete_old_container_images() {
    local IMAGE_NAME="$1"
    local KEEP=1

    echo -e "\n========== Pruning old digests in $IMAGE_NAME ==========\n"

    gcloud artifacts docker images list "$IMAGE_NAME" \
        --format="get(DIGEST)" \
        --sort-by="~CREATE_TIME" \
        --limit=unlimited | tail -n +2 | while read -r digest; do
        echo "Deleting digest: $digest"
        gcloud artifacts docker images delete "${IMAGE_NAME}@${digest}" --quiet
    done

    echo -e "\n========== Final list of remaining images ==========\n"
    gcloud artifacts docker images list ${IMAGE_NAME} --sort-by="~CREATE_TIME"

}

push_new_docker_image() {
    echo -e "\n========== Building and pushing Docker image with caching ==========\n"

    local DOCKERFILE="$1"
    local IMAGE_NAME="$2"

    gcloud builds submit \
        --config="${BUILD_FILE}" \
        --substitutions=_IMAGE_NAME="${IMAGE_NAME}",_DOCKERFILE="${DOCKERFILE}"
}

deploy_api() {
    echo -e "\n========== Deploy API and launch service =========="
    # install Nvidia drivers
    # kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml

    # apply API manifest in infra/k8s/arxiv-summarization-api.yaml
    kubectl delete -f "${ARXIV_SUMMARIZATION_API_YAML_FILE}" -n "${ARXIV_SUMMARIZATION_API_NAMESPACE}" || true
    kubectl apply -f "${ARXIV_SUMMARIZATION_API_YAML_FILE}"

    # wait for the API service to deploy
    kubectl rollout status deployment/"${ARXIV_SUMMARIZATION_API_DEPLOYMENT}" \
        -n "${ARXIV_SUMMARIZATION_API_NAMESPACE}" --timeout=30m

    # retrieve the external IP
    kubectl get svc "${ARXIV_SUMMARIZATION_API_SERVICE}" \
        -n "${ARXIV_SUMMARIZATION_API_NAMESPACE}"
}

delete_old_container_images "$IMAGE_NAME"
push_new_docker_image "$DOCKERFILE" "$IMAGE_NAME"

deploy_api
