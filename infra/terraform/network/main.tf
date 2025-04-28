# Assign firewall rules
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
