variable "gcp_region" {
  type = string
}

variable "gcp_gcs_buckets" {
  description = "List of GCS buckets with their paths"
  type = list(object({
    name  = string
    paths = any
  }))
}
