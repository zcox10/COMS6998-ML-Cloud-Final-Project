output "gke_node_gsa_email" {
  description = "Google Service Account (GSA) email used for GKE node pods"
  value       = google_service_account.kubernetes_gsa.email
}

