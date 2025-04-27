# ML Cloud Final Project

Summary goes here....

## Setup Steps

1. Create a `config.yaml` file with the following parameters. Store the config file in root directory.

   ```yaml
   gcp_project_id: "<GCP_PROJECT_ID>"
   gcs_bucket_name: "<GCS_BUCKET_NAME>"
   gke_cluster_name: "<GKE_CLUSTER_NAME>"
   gcr_image_name: "gcr.io/${gcp_project_id}/ml-cloud-pipeline" # do not change
   gcp_region: "<GCP_REGION>"
   gcp_service_account_key_path: "<PATH/TO/google_application_credentials.json>"
   kubeflow_pipeline_name: "<KUBEFLOW_PIPELINE_NAME>"
   kubeflow_experiment_name: "<KUBEFLOW_EXPERIMENT_NAME>"
   kubeflow_pipeline_package_path: "<PATH/TO/KUBEFLOW_PIPELINE.yaml>"
   host: "http://localhost:8080" # for local 
   dockerfile_train: "Dockerfile.train" # do not change
   dockerfile_api: "Dockerfile.api" # do not change
   ```

2. Create a new Google Cloud Project

3. Create a service account with the following permissions:
   - owner

4. Install terraform via brew (on Mac) with the following commands

  ```bash
  brew update && brew upgrade
  brew tap hashicorp/tap
  brew install hashicorp/tap/terraform
  ```

5. Run the Kubernetes setup script from root with the following commands

   ```bash
   chmod +x ./scripts/kubernetes_setup.sh
   chmod +x ./scripts/load_env.sh 
   chmod +x ./scripts/run_kubeflow_pipeline.sh 
   ./scripts/kubernetes_setup.sh
   ```

6. Run Kubeflow Pipeline with `./scripts/run_kubeflow_pipeline.sh`

### Appendix

**Helpful commands:**

```bash
# initialize Terraform
terraform init

# To review what Terraform will create
terraform plan

# To apply changes
terraform apply

# Deletes all resources created by your .tf files (good for cleanup)
terraform destroy

# Checks if your Terraform files are syntactically valid
terraform validate

# Auto-formats your .tf files to clean style
terraform fmt

# To confirm connection to the cluster
gcloud container clusters get-credentials kubeflow-gke --region us-east1

# To get the status of all Kubernetes pods
kubectl get pods -A
```
