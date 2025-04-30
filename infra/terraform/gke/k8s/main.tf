# Provider for Kubernetes
terraform {
  required_providers {
    kubernetes = {
      source = "hashicorp/kubernetes"
    }
  }
}

# Create vector_db namespace
resource "kubernetes_namespace" "vector_db" {
  metadata {
    name = var.gcp_gke_services_vector_db_namespace
  }
}

# Create arxiv_summarization_api namespace
resource "kubernetes_namespace" "arxiv_summarization_api" {
  metadata {
    name = var.gcp_gke_services_arxiv_summarization_api_namespace
  }
}

# Create Kubernetes Service Account for API server
resource "kubernetes_service_account" "arxiv_summarization_api_ksa" {
  metadata {
    name      = var.gcp_gke_services_arxiv_summarization_api_ksa_name
    namespace = var.gcp_gke_services_arxiv_summarization_api_namespace
  }
  depends_on = [kubernetes_namespace.arxiv_summarization_api]
}

# Assign workloadIdentityUser role to Arxiv Summarization API KSA
resource "google_project_iam_member" "workload_identity_binding_arxiv_api_server" {
  project = var.gcp_project_id
  role    = "roles/iam.workloadIdentityUser"
  member  = "serviceAccount:${var.gcp_project_id}.svc.id.goog[${kubernetes_service_account.arxiv_summarization_api_ksa.metadata[0].namespace}/${kubernetes_service_account.arxiv_summarization_api_ksa.metadata[0].name}]"
}

# Assign workloadIdentityUser role to pipeline-runner KSA
resource "google_project_iam_member" "workload_identity_binding_pipeline_runner" {
  project = var.gcp_project_id
  role    = "roles/iam.workloadIdentityUser"
  member  = "serviceAccount:${var.gcp_project_id}.svc.id.goog[kubeflow/pipeline-runner]"
}
