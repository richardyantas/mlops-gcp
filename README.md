# mlops-gcp


## Setup Development
- Create an environment: `make conda-update`
- install dependencies: `make pip-tools`
- To use secret manager make sure to login to gcloud
`gcloud auth application-default login`
- `Terraform init && Terraform plan && Terraform apply`
- compile and run pipeline use: `python pipeline/pipeline.py`
  
## Setup Production

- Go to the folder pipeline `cd /pipeline`
- run: `docker build -t mlops .`
- run: `push and tag to GCR`
