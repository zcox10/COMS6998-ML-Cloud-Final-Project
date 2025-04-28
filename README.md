# ML Cloud Final Project

Summary goes here....

## Setup Steps

1. Create a new Google Cloud Project

2. Create a service account with the `owner` permission. This service account will run the Terraform setup by setting IAM permissions, creating service accounts for Kubernetes, setting up a GCS storage bucket, configuring network settings, and creating a GKE cluster. **Note:** setting the main service account with `owner` permission does not align with best practices, but since this is a learning exercise, it is best just to get everything up and running.

3. Install terraform via brew (on Mac) with the following commands

  ```bash
  brew update && brew upgrade # optional, but recommended
  brew tap hashicorp/tap
  brew install hashicorp/tap/terraform
  ```

4. Run the Kubernetes setup script from root with the following commands

   ```bash
   chmod +x ./scripts/setup/kubernetes_setup.sh
   ./scripts/kubernetes_setup.sh
   ```

5. Run Kubeflow Pipeline with:

   ```bash
   chmod +x ./scripts/kubeflow/run_kubeflow_pipeline.sh
   ./scripts/kubeflow/run_kubeflow_pipeline.sh
   ```

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
