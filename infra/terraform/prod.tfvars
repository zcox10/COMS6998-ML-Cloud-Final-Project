# Auto-generated from config.yaml
gcp_project_id                                     = "zsc-personal"
gcp_region                                         = "us-east1"
gcp_gcs_buckets                                    = [{ name = "ml-cloud-project-gcs", paths = { data = "data/papers", embeddings = "data/embeddings", fine_tune = { models = "fine_tune/models", data = "fine_tune/data" } } }]
gcp_gke_cluster_name                               = "gke-cluster-ml-cloud"
gcp_gke_machine_type                               = "e2-standard-2"
gcp_gke_service_account_name                       = "gke-ml-cloud-service-account"
gcp_gke_services_arxiv_summarization_api_ksa_name  = "arxiv-summarization-api-service-account"
gcp_gke_services_arxiv_summarization_api_namespace = "arxiv-summarization-api"
gcp_gke_services_vector_db_namespace               = "vector-db"
