

resource "google_container_cluster" "kubeflow_gke" {
  name     = var.gcp_gke_cluster_name
  location = var.gcp_region

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  ip_allocation_policy {}

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_container_node_pool" "primary_nodes" {
  name     = "default-node-pool"
  cluster  = google_container_cluster.kubeflow_gke.name
  location = var.gcp_region

  node_config {
    machine_type = var.gcp_gke_machine_type
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    service_account = var.gcp_gke_service_account_email
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }
}
