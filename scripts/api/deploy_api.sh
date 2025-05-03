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

PROJECT_ID=$(yq -r '.gcp.project' "$CONFIG_FILE")
PLATFORM="linux/amd64"
TAG="latest"

DOCKERFILE=$(yq -r '.dockerfile.api' "$CONFIG_FILE")
IMAGE_NAME=$(yq -r '.gcp.gke.services.arxiv_summarization_api.image' "$CONFIG_FILE")

delete_old_container_images() {

    local IMAGE_NAME="$1"

    echo -e "\n========== Pruning old digests in $IMAGE_NAME ==========\n"

    local deleted=true
    while $deleted; do
        deleted=false
        gcloud container images list-tags "$IMAGE_NAME" \
            --format='value(digest,tags)' |
            while read -r digest tags; do
                if [[ $tags == *latest* || $tags == *cache* ]]; then
                    continue
                fi
                if gcloud container images delete "${IMAGE_NAME}@sha256:${digest}" \
                    --quiet --force-delete-tags 2>/dev/null; then
                    echo "Deleted sha256:${digest}"
                    deleted=true
                fi
            done
    done

    echo -e "\n========== Final list of remaining images ==========\n"
    gcloud container images list-tags "$IMAGE_NAME"
}

push_new_docker_image() {
    echo -e "\n========== Building and pushing Docker image with caching ==========\n"

    local DOCKERFILE="$1"
    local IMAGE_NAME="$2"
    local CACHE_NAME="${IMAGE_NAME}:cache"

    docker buildx build \
        --file "${DOCKERFILE}" \
        --platform "${PLATFORM}" \
        --tag "${IMAGE_NAME}:${TAG}" \
        --cache-from=type=registry,ref="${CACHE_NAME}" \
        --cache-to=type=registry,ref="${CACHE_NAME}",mode=max \
        --push \
        --progress="auto" .

    echo -e "\n========== Build and push complete: ${IMAGE_NAME}:${TAG} =========="
}

deploy_api() {
    echo -e "\n========== Deploy API and launch service =========="
    # install Nvidia drivers
    # kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

    # kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml

    # apply API manifest in infra/k8s/arxiv-summarization-api.yaml
    kubectl apply -f "${ARXIV_SUMMARIZATION_API_YAML_FILE}"

    # Patch the deployment to increase the progress deadline
    kubectl patch deployment "${ARXIV_SUMMARIZATION_API_DEPLOYMENT}" \
        -n "${ARXIV_SUMMARIZATION_API_NAMESPACE}" \
        --patch '{"spec":{"progressDeadlineSeconds":3600}}'

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
