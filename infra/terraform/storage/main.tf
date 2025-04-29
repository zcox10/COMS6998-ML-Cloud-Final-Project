# Generate a GCS storage bucket
resource "google_storage_bucket" "storage_bucket_data" {
  for_each = { for b in var.gcp_gcs_buckets : b.name => b }

  name                        = each.key
  location                    = var.gcp_region
  force_destroy               = true
  uniform_bucket_level_access = true
  storage_class               = "STANDARD"
}
