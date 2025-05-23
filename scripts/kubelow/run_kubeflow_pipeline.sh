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
BUILD_FILE="$ROOT_DIR/infra/cloud_builds/cloud-build.yaml"

DOCKERFILE_CPU=$(yq -r '.dockerfile.kubeflow_cpu' "$CONFIG_FILE")
IMAGE_NAME_CPU=$(yq -r '.gcp.gke.services.kubeflow.images.cpu' "$CONFIG_FILE")

DOCKERFILE_GPU=$(yq -r '.dockerfile.kubeflow_gpu' "$CONFIG_FILE")
IMAGE_NAME_GPU=$(yq -r '.gcp.gke.services.kubeflow.images.gpu' "$CONFIG_FILE")

delete_old_container_images() {

    local IMAGE_NAME="$1"

    echo -e "\n========== Pruning old digests in $IMAGE_NAME ==========\n"

    local KEEP=1

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

run_kubeflow_pipeline() {
    echo -e "\n========== Upload pipeline and run =========="
    python "$CURRENT_DIR/kubeflow_pipeline_runner.py"
}

# Time operation
start_time=$(date +%s)

# Run functions
delete_old_container_images "$IMAGE_NAME_CPU"
delete_old_container_images "$IMAGE_NAME_GPU"

push_new_docker_image "$DOCKERFILE_CPU" "$IMAGE_NAME_CPU"
push_new_docker_image "$DOCKERFILE_GPU" "$IMAGE_NAME_GPU"

run_kubeflow_pipeline

# Output time to build
end_time=$(date +%s)
elapsed=$((end_time - start_time))
mins=$((elapsed / 60))
secs=$((elapsed % 60))
echo -e "\n========== Total time elapsed: ${mins} min ${secs} sec ==========\n"
