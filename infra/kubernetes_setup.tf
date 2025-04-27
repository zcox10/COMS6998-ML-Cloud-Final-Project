# Terraform configuration for GKE + Workload Identity + GCS + IAM

variable "gcp_project_id" {
  description = "The ID of the GCP project"
  type        = string
}

variable "gcp_region" {
  description = "The region to deploy resources"
  type        = string
  default     = "us-east1"
}

variable "gcs_bucket_name" {
  description = "The GCS bucket name"
  type        = string
}

variable "gcp_service_account_key_path" {
  description = "The local path to the GCP service account"
  type        = string
  default     = ""
}

variable "gke_cluster_name" {
  description = "The name of the GKE cluster"
  type        = string
  default     = ""
}


provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

resource "google_container_cluster" "kubeflow_gke" {
  name     = var.gke_cluster_name
  location = var.gcp_region

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  ip_allocation_policy {}
}

resource "google_container_node_pool" "primary_nodes" {
  name     = "default-node-pool"
  cluster  = google_container_cluster.kubeflow_gke.name
  location = var.gcp_region

  node_config {
    machine_type = "e2-standard-2"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    service_account = google_service_account.pipeline_runner.email
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }
}

resource "google_service_account" "pipeline_runner" {
  account_id   = "pipeline-runner"
  display_name = "Pipeline Runner Service Account"
}

resource "google_project_iam_member" "workload_identity_binding" {
  project    = var.gcp_project_id
  role       = "roles/iam.workloadIdentityUser"
  member     = "serviceAccount:${var.gcp_project_id}.svc.id.goog[default/pipeline-runner]"
  depends_on = [google_container_cluster.kubeflow_gke, google_service_account.pipeline_runner]
}

resource "google_storage_bucket" "storage_bucket_data" {
  name          = var.gcs_bucket_name
  location      = var.gcp_region
  force_destroy = true
}

resource "google_project_iam_member" "storage_access" {
  project = var.gcp_project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

resource "google_project_iam_member" "logging_writer" {
  project = var.gcp_project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

resource "google_project_iam_member" "token_creator" {
  project = var.gcp_project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

resource "google_project_iam_member" "artifact_registry_reader" {
  project = var.gcp_project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

resource "google_compute_firewall" "allow_kubeflow_ui_and_api" {
  name    = "allow-kubeflow-ui-and-api"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]

  target_tags = ["gke-cluster"]
}

# resource "null_resource" "testing_print" {
#   provisioner "local-exec" {
#     command = "bash ${path.module}/../scripts/hello_world.sh ${var.gcp_region} ${var.gcp_project_id}"
#   }
# }
