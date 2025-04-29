#!/bin/bash

set -e

# Description: a script to setup a GKE cluster and install Kubeflow pipelines on the GKE cluster
# - Loads vars from config YAML file
# - Creates and activates a local gcloud config with the GCP project service account
# - Creates a GKE cluster and GCS storage bucket via Terraform
# - Installs Kubeflow piplines
# - Port forwards the Kubeflow UI for local UI access

echo -e "\n=============== Loading vars from config YAML file ===============\n"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$CURRENT_DIR/../.."
TERRAFORM_DIR="$ROOT_DIR/infra/terraform"
CONFIG_FILE="$ROOT_DIR/config.yaml"

# Google Cloud Project ID and Region
GCP_PROJECT_ID=$(yq -r '.gcp.project_id' "$CONFIG_FILE")
GCP_REGION=$(yq -r '.gcp.region' "$CONFIG_FILE")

# Google Kubernetes Engine cluster name
GKE_CLUSTER_NAME=$(yq -r '.gcp.gke.cluster_name' "$CONFIG_FILE")

# Main GKE service account to bind to KSA's
GKE_GSA_NAME=$(yq -r '.gcp.gke.service_account_name' "$CONFIG_FILE")

# Arxiv Summarization API Kubernetes service account (KSA) and namespace
ARXIV_SUMMARIZATION_API_KSA_NAME=$(yq -r '.gcp.gke.services.arxiv_summarization_api.ksa_name' "$CONFIG_FILE")
ARXIV_SUMMARIZATION_API_NAMESPACE=$(yq -r '.gcp.gke.services.arxiv_summarization_api.namespace' "$CONFIG_FILE")

reset_gcloud_config() {
    bash "$CURRENT_DIR/reset_local_gcloud_config.sh"
}

cleanup_terraform_resources() {
    echo -e "\n=============== Cleanup Terraform resources ===============\n"
    rm -rf "$TERRAFORM_DIR/.terraform" || true
    rm -rf "$TERRAFORM_DIR/.terraform.lock.hcl" || true
    rm -rf "$TERRAFORM_DIR/terraform.tfstate" || true
    rm -rf "$TERRAFORM_DIR/terraform.tfstate.backup" || true
    rm -rf "$TERRAFORM_DIR/prod.tfvars" || true
}

generate_tfvars_file() {
    echo -e "\n=============== Generate TFVARS_FILE ===============\n"
    python "$CURRENT_DIR/generate_tfvars.py"
}

full_terraform_setup() {
    echo -e "\n=============== Running terraform init, validate, and apply ===============\n"
    pushd "$TERRAFORM_DIR" >/dev/null
    terraform init
    terraform validate
    terraform apply -var-file="prod.tfvars" -auto-approve
    popd >/dev/null
}

terraform_init() {
    echo -e "\n=============== Running terraform init ===============\n"
    # temporarily cd into infra/, run `terraform init`, and return to original dir
    pushd "$TERRAFORM_DIR" >/dev/null
    terraform init
    popd >/dev/null
}

install_terraform_iam_module() {
    echo -e "\n=============== Running terraform apply for IAM module ===============\n"
    # temporarily cd into infra/, run `terraform apply`, and return to original dir
    pushd "$TERRAFORM_DIR" >/dev/null
    terraform apply -target=module.iam -var-file="prod.tfvars" -auto-approve
    popd >/dev/null
}

install_terraform_gke_module() {
    echo -e "\n=============== Running terraform apply for IAM module ===============\n"
    # temporarily cd into infra/, run `terraform apply`, and return to original dir
    pushd "$TERRAFORM_DIR" >/dev/null
    terraform apply -target=module.gke -var-file="prod.tfvars" -auto-approve
    popd >/dev/null
}

install_terraform_storage_module() {
    echo -e "\n=============== Running terraform apply for IAM module ===============\n"
    # temporarily cd into infra/, run `terraform apply`, and return to original dir
    pushd "$TERRAFORM_DIR" >/dev/null
    terraform apply -target=module.storage -var-file="prod.tfvars" -auto-approve
    popd >/dev/null
}

install_terraform_network_module() {
    echo -e "\n=============== Running terraform apply for IAM module ===============\n"
    # temporarily cd into infra/, run `terraform apply`, and return to original dir
    pushd "$TERRAFORM_DIR" >/dev/null
    terraform apply -target=module.network -var-file="prod.tfvars" -auto-approve
    popd >/dev/null
}

fetch_gke_credentials() {
    echo -e "\n=============== Fetching credentials for GKE cluster and create namespaces ===============\n"
    gcloud container clusters get-credentials "$GKE_CLUSTER_NAME" --region "$GCP_REGION" --project "$GCP_PROJECT_ID"
}

install_kubeflow_pipelines() {
    echo -e "\n=============== Install Kubeflow Pipelines ===============\n"
    export PIPELINE_VERSION=2.4.1
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION&timeout=120s"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION&timeout=120s"
}

annotate_kubernetes_service_accounts() {
    echo -e "\n=============== Annotating Kubernetes Service Accounts for Workload Identity ===============\n"
    # default KSA
    kubectl annotate serviceaccount default \
        --namespace=default \
        iam.gke.io/gcp-service-account=${GKE_GSA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
        --overwrite

    # ml-pipeline KSA
    kubectl annotate serviceaccount pipeline-runner \
        --namespace=kubeflow \
        iam.gke.io/gcp-service-account=${GKE_GSA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
        --overwrite

    # api-server KSA
    kubectl annotate serviceaccount $ARXIV_SUMMARIZATION_API_KSA_NAME \
        --namespace=$ARXIV_SUMMARIZATION_API_NAMESPACE \
        iam.gke.io/gcp-service-account=${GKE_GSA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
        --overwrite
}

patch_minio() {
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
}

kubeflow_wait_for_availability() {
    echo -e "\n=============== Waiting for Kubeflow Pipelines pods to become ready ===============\n"
    kubectl wait --for=condition=Available deployment --all -n kubeflow --timeout=600s
}

retrieve_kubeflow_labels() {
    echo -e "\n=============== Retrieve Kubeflow labels ===============\n"
    kubectl get deployment ml-pipeline-ui -n kubeflow -o yaml
}

list_kubeflow_services() {
    echo -e "\n=============== Listing services in kubeflow namespace ===============\n"
    kubectl get svc -n kubeflow
}

port_forward_for_kubeflow_ui_access() {
    echo -e "\n=============== Port forwarding for local UI access ===============\n"
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
}

full_setup_install() {
    reset_gcloud_config
    cleanup_terraform_resources
    generate_tfvars_file
    full_terraform_setup
    fetch_gke_credentials
    install_kubeflow_pipelines
    annotate_kubernetes_service_accounts
    patch_minio
    kubeflow_wait_for_availability
    retrieve_kubeflow_labels
    list_kubeflow_services
    port_forward_for_kubeflow_ui_access
}

# Runs a full setup install for Kubernetes setup with Kubeflow
full_setup_install
