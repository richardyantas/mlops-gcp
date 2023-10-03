variable "project" {
  description = "mlops gcp id"
  default     = "mlops-400610"
}

variable "credentials_file" {
  description = "Credential file with project owner permissions"
  default     = "./mlops-400610-f704a2b7bb3c.json"
}

variable "region" {
  description = "Region for Resources"
  # default = "us-central1"
  default = "US"
}

# variable "zone" {
#   description = "Zone for Resources"
#   default = "us-central1-c"
# }

variable "bigquery_dataset" {
  description = "BigQuery Dataset"
  default     = "iris"
}

variable "storage_class" {
  description = "GCS Bucket Storage Class"
  default     = "STANDARD"
}