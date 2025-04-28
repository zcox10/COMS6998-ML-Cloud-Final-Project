# Generate a Google Service Account to bind permissions to Kubernetes Service Accounts
resource "google_service_account" "kubernetes_gsa" {
  account_id   = var.gcp_gke_service_account_name
  display_name = "GKE Node Main Service Account"
}

# Grant roles to GSA
locals {
  gke_node_roles = [
    "roles/storage.objectAdmin",
    "roles/logging.logWriter",
    "roles/iam.serviceAccountTokenCreator",
    "roles/artifactregistry.reader",
  ]
}
resource "google_project_iam_binding" "gke_node_roles_binding" {
  for_each = toset(local.gke_node_roles)

  project = var.gcp_project_id
  role    = each.key

  members = [
    "serviceAccount:${google_service_account.kubernetes_gsa.email}",
  ]
}
