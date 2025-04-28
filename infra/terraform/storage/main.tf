# Generate a GCS storage bucket
resource "google_storage_bucket" "storage_bucket_data" {
  for_each = { for bucket in var.gcp_gcs_buckets : bucket.name => bucket }

  name          = each.value.name
  location      = var.gcp_region
  force_destroy = true

  uniform_bucket_level_access = true
  storage_class               = "STANDARD"
}
