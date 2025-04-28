provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "kubernetes" {
  alias                  = "gke"
  host                   = "https://${module.gke_cluster.endpoint}"
  cluster_ca_certificate = base64decode(module.gke_cluster.master_auth.cluster_ca_certificate)
  token                  = data.google_client_config.default.access_token
}

data "google_client_config" "default" {}

module "iam" {
  source                       = "./iam"
  gcp_project_id               = var.gcp_project_id
  gcp_gke_service_account_name = var.gcp_gke_service_account_name
}

module "storage" {
  source          = "./storage"
  gcp_region      = var.gcp_region
  gcp_gcs_buckets = var.gcp_gcs_buckets
}

module "network" {
  source = "./network"
}

module "gke_cluster" {
  source                        = "./gke/cluster"
  gcp_project_id                = var.gcp_project_id
  gcp_region                    = var.gcp_region
  gcp_gke_cluster_name          = var.gcp_gke_cluster_name
  gcp_gke_machine_type          = var.gcp_gke_machine_type
  gcp_gke_service_account_email = module.iam.gke_node_gsa_email
}

# Debugging
locals {
  gke_endpoint_raw = module.gke_cluster.endpoint
  gke_endpoint_host = (
    startswith(module.gke_cluster.endpoint, "https://") ?
    module.gke_cluster.endpoint :
    "https://${module.gke_cluster.endpoint}"
  )
}

output "gke_kubernetes_host_raw" {
  value = local.gke_endpoint_raw
}

output "gke_kubernetes_host_url" {
  value = local.gke_endpoint_host
}


module "gke_k8s" {
  source = "./gke/k8s"
  providers = {
    kubernetes = kubernetes.gke
  }

  gcp_project_id                                     = var.gcp_project_id
  gcp_gke_service_account_name                       = var.gcp_gke_service_account_name
  gcp_gke_services_arxiv_summarization_api_namespace = var.gcp_gke_services_arxiv_summarization_api_namespace
  gcp_gke_services_arxiv_summarization_api_ksa_name  = var.gcp_gke_services_arxiv_summarization_api_ksa_name
  depends_on                                         = [module.gke_cluster]
}

