#!/bin/bash

set -e

# Description: a script to setup a GKE cluster and install Kubeflow pipelines on the GKE cluster
# - Loads vars from config YAML file
# - Creates and activates a local gcloud config with the GCP project service account
# - Creates a GKE cluster and GCS storage bucket via Terraform
# - Installs Kubeflow piplines
# - Port forwards the Kubeflow UI for local UI access

echo -e "\n=============== Loading vars from config YAML file ===============\n"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
INFRA_DIR="$ROOT_DIR/infra"
CONFIG_FILE="$ROOT_DIR/config.yaml"

GCP_REGION=$(yq -r '.gcp.region' "$CONFIG_FILE")
GCP_PROJECT_ID=$(yq -r '.gcp.project' "$CONFIG_FILE")
GCP_SERVICE_ACCOUNT_KEY_PATH=$(yq -r '.gcp.service_account_key' "$CONFIG_FILE")
GKE_CLUSTER_NAME=$(yq -r '.gcp.cluster' "$CONFIG_FILE")

echo -e "\n=============== Reset gcloud config ===============\n"

echo "Using PROJECT_ID=$GCP_PROJECT_ID"
echo "Using SERVICE_ACCOUNT_KEY_PATH=$GCP_SERVICE_ACCOUNT_KEY_PATH"

# Create a new config based on the service account
gcloud config configurations activate default || true
gcloud --quiet config configurations delete ml-cloud || true
gcloud config configurations create ml-cloud
gcloud config configurations activate ml-cloud
gcloud auth activate-service-account --key-file="$GCP_SERVICE_ACCOUNT_KEY_PATH"

# Enable services for GCP project
gcloud services enable \
    compute.googleapis.com \
    container.googleapis.com \
    storage.googleapis.com \
    cloudresourcemanager.googleapis.com --project="$GCP_PROJECT_ID"

# Set GCP project
gcloud config set project "$GCP_PROJECT_ID"

# List available service on GCP project
gcloud services list --enabled

# Remove terraform resources
rm -rf ./infra/.terraform || true
rm -rf ./infra/.terraform.lock.hcl || true
rm -rf ./infra/terraform.tfstate || true
rm -rf ./infra/terraform.tfstate.backup || true

echo -e "\n=============== Generate TFVARS_FILE ===============\n"
bash "$SCRIPT_DIR/generate_tfvars.sh"

echo -e "\n=============== Running terraform init/apply ===============\n"
# temporarily cd into infra/
pushd "$INFRA_DIR" >/dev/null

# Run terraform
terraform init
terraform apply -var-file="kubernetes_setup.tfvars" -auto-approve

# return to original dir
popd >/dev/null

echo -e "\n=============== Fetching credentials for GKE cluster ===============\n"
gcloud container clusters get-credentials "$GKE_CLUSTER_NAME" --region "$GCP_REGION" --project "$GCP_PROJECT_ID"
kubectl create namespace kubeflow || true

export PIPELINE_VERSION=2.4.1
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"

echo -e "\n=============== Patching MinIO deployment to set fsGroup=1000 ===============\n"
# Wait until the deployment object exists
until kubectl -n kubeflow get deployment minio &>/dev/null; do
    sleep 2
done

# Patch MinIO with proper permissions
kubectl -n kubeflow patch deployment minio \
    --type='json' \
    -p='[{"op":"add","path":"/spec/template/spec/securityContext/fsGroup","value":1000}]' ||
    echo "\n=============== MinIO already patched ===============\n"

echo -e "\n=============== Waiting for Kubeflow Pipelines pods to become ready ===============\n"
kubectl wait --for=condition=Available deployment --all -n kubeflow --timeout=600s

echo -e "\n=============== Retrieve Kubeflow labels ===============\n"
kubectl get deployment ml-pipeline-ui -n kubeflow -o yaml

echo -e "\n=============== Listing services in kubeflow namespace ===============\n"
kubectl get svc -n kubeflow

echo -e "\n=============== Port forwarding for local UI access ===============\n"
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
