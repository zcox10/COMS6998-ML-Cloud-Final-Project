output "endpoint" {
  value = google_container_cluster.kubeflow_gke.endpoint
}

output "master_auth" {
  value = google_container_cluster.kubeflow_gke.master_auth[0]
}
