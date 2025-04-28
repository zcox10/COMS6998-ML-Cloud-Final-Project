variable "gcp_region" {
  type = string
}

variable "gcp_gcs_buckets" {
  type = list(object({
    name  = string
    paths = map(string)
  }))
}
