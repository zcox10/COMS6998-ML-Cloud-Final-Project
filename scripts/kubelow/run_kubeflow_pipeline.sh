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

PROJECT_ID=$(yq -r '.gcp.project' "$CONFIG_FILE")
PLATFORM="linux/amd64"
TAG="latest"

DOCKERFILE_CPU=$(yq -r '.dockerfile.kubeflow_cpu' "$CONFIG_FILE")
IMAGE_NAME_CPU=$(yq -r '.gcp.gke.services.kubeflow.images.cpu' "$CONFIG_FILE")

DOCKERFILE_GPU=$(yq -r '.dockerfile.kubeflow_gpu' "$CONFIG_FILE")
IMAGE_NAME_GPU=$(yq -r '.gcp.gke.services.kubeflow.images.gpu' "$CONFIG_FILE")

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

run_kubeflow_pipeline() {
    echo -e "\n========== Upload pipeline and run =========="
    python "$CURRENT_DIR/kubeflow_pipeline_runner.py"
}

delete_old_container_images "$IMAGE_NAME_CPU"
delete_old_container_images "$IMAGE_NAME_GPU"

push_new_docker_image "$DOCKERFILE_CPU" "$IMAGE_NAME_CPU"
push_new_docker_image "$DOCKERFILE_GPU" "$IMAGE_NAME_GPU"

run_kubeflow_pipeline
