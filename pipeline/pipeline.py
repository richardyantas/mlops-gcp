import kfp
from kfp import dsl
from google.cloud import aiplatform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Tuple, Dict, List
import json
import kfp
import os
from kfp import dsl
from kfp.v2 import compiler
from datetime import datetime
from google.cloud import secretmanager
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv() 

print("default credential path: ", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
# GCS_BUCKET = f"gs://{BUCKET_NAME}"
GCS_BUCKET = os.getenv("GCS_BUCKET")
LOCATION = os.getenv("LOCATION")
DISPLAY_RUN_NAME = os.getenv("DISPLAY_RUN_NAME")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
BQ_PATH = f"bq://{PROJECT_ID}",
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")
PIPELINE_NAME = os.getenv("PIPELINE_NAME")

PACKAGE_PATH = f"iris-customtraining.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

DOCKER_IMAGE_URI = f'gcr.io/{PROJECT_ID}/workshop-sklearn-iris:v1'
print(f'La uri de la imagen es: {DOCKER_IMAGE_URI}')

print("GCS bucket: ", GCS_BUCKET)

secret_name = "projects/550919589615/secrets/mlops-secret/versions/latest"
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(name=secret_name)
secret_value = response.payload.data.decode("UTF-8")
print("secret value: ",secret_value)
secret_info = json.loads(secret_value)
storage_client = storage.Client.from_service_account_info(secret_info)

def pipeline(
    gcs_source: str = GCS_BUCKET, # "bq://sara-vertex-demos.beans_demo.large_dataset",
    bucket: str = BUCKET_NAME,
    project: str = PROJECT_ID,
    gcp_region: str = LOCATION,
    bq_dest: str = BQ_PATH, # ""
    container_uri: str = DOCKER_IMAGE_URI, # ""
    batch_destination: str = ""
):
    #cleaned_data = read_and_preprocess_data_op()
    
    dataset_create_op = aiplatform.TabularDataset(
        display_name="tabular-iris-dataset",
        # bq_source=bq_source,
        gcs_source = GCS_BUCKET,
        project=PROJECT_ID,
        location=LOCATION
    )

    training_op = aiplatform.CustomContainerTrainingJob(
        # preprocess data_op
        # split_data_op
        # train_model_op
        display_name=DISPLAY_RUN_NAME,
        container_uri=container_uri,
        project=project,
        location=gcp_region,
        dataset=dataset_create_op.outputs["dataset"],
        staging_bucket=gcs_source, # customtraining
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        bigquery_destination=bq_dest, # destination
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
        model_display_name="scikit-iris-model-pipeline",
        machine_type="n1-standard-4",
    )

    batch_predict_op = aiplatform.BatchPredictionJob(
        project=PROJECT_ID,
        location=LOCATION,
        job_display_name=f"{PIPELINE_NAME}-predict",
        model=training_op.outputs["model"],
        gcs_source_uris=[f"{GCS_BUCKET}/batch_examples.csv"],
        instances_format="csv",
        gcs_destination_output_uri_prefix=batch_destination,
        machine_type="n1-standard-4"
    )

# Ejecuta el pipeline

if __name__ == '__main__':
    aiplatform.init(project=PROJECT_ID, staging_bucket=GCS_BUCKET)    
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=PACKAGE_PATH)
    pipeline_job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=PACKAGE_PATH,
        job_id=f'{PIPELINE_NAME}-{TIMESTAMP}',
        parameter_values={
            "project": PROJECT_ID,
            "bucket": GCS_BUCKET,
            "bq_dest": BQ_PATH, # f"bq://{PROJECT_ID}",
            "container_uri": f"gcr.io/{PROJECT_ID}/pipeline-sklearn-iris:v1",
            "batch_destination": "{GCS_BUCKET}/batchpredresults"
        },
        enable_caching=True,
    )
    pipeline_job.submit()    