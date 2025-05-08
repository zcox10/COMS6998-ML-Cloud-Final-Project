resource "google_container_cluster" "kubeflow_gke" {
  name     = var.gcp_gke_cluster_name
  location = var.gcp_region

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  ip_allocation_policy {}

  deletion_protection = false
}

resource "google_container_node_pool" "cpu_pool" {
  name     = "cpu-node-pool"
  cluster  = google_container_cluster.kubeflow_gke.name
  location = var.gcp_region

  node_config {
    machine_type    = var.gcp_gke_machine_type
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    service_account = var.gcp_gke_service_account_email
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 3
  }
}

resource "google_container_node_pool" "gpu_pool" {
  name           = "gpu-node-pool"
  cluster        = google_container_cluster.kubeflow_gke.name
  location       = var.gcp_region
  node_locations = ["${var.gcp_region}-b"]

  # Create 1 L4 per node
  node_config {
    machine_type    = "g2-standard-4"
    image_type      = "COS_CONTAINERD"
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    service_account = var.gcp_gke_service_account_email

    gcfs_config {
      enabled = true
    }

    gvnic {
      enabled = true
    }

    # enable NVIDIA driver
    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    # label nodes for selection
    labels = {
      accelerator = "nvidia-l4"
      gpu_pool    = "kubeflow"
    }

    # taint so only GPU-requesting Pods land here
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 2
  }
}
