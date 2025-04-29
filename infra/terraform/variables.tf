variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP Region"
  type        = string
}

variable "gcp_gcs_buckets" {
  description = "List of GCS buckets with their paths"
  type = list(object({
    name  = string
    paths = any
  }))
}

variable "gcp_gke_cluster_name" {
  description = "GKE Cluster Name"
  type        = string
}

variable "gcp_gke_machine_type" {
  description = "The machine type for GKE nodes"
  type        = string
}

variable "gcp_gke_service_account_name" {
  description = "The name of the GKE node main service account that binds permissions to Kubernetes service accounts"
  type        = string
}

variable "gcp_gke_services_arxiv_summarization_api_ksa_name" {
  description = "The name of the Kubernetes service account belonging to the Arxiv Summarization API namespace"
  type        = string
}

variable "gcp_gke_services_arxiv_summarization_api_namespace" {
  description = "The Kubernetes namespace of the Arxiv Summarization API"
  type        = string
}
