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
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

print("GCS bucket: ", GCS_BUCKET)

secret_name = "projects/550919589615/secrets/mlops-secret/versions/latest"
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(name=secret_name)
secret_value = response.payload.data.decode("UTF-8")
print("secret value: ",secret_value)
secret_info = json.loads(secret_value)
storage_client = storage.Client.from_service_account_info(secret_info)


@dsl.component
def read_and_preprocess_data_op() -> str:
    data = aiplatform.TabularDataset.create(
        display_name="iris_pipeline",
        gcs_source=f"{GCS_BUCKET}/iris.csv",
    )    
    df = pd.read_csv(data)
    df = df.dropna()    
    cleaned_data_path = f"{GCS_BUCKET}/cleaned_data_pipeline.csv"
    df.to_csv(cleaned_data_path, index=False)
    return cleaned_data_path
    

@dsl.component
def split_data_op(cleaned_data: str) -> dsl.Artifact:
    df = pd.read_csv(cleaned_data)
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data_path = f"{GCS_BUCKET}/train_iris.csv"
    test_data_path = f"{GCS_BUCKET}/test_iris.csv"
    X_train.to_csv(train_data_path, index=False)
    X_test.to_csv(test_data_path, index=False)

    output_artifact = dsl.Artifact(
        name="my_output_artifact",
        description="Description of the output artifact",
        schema_title="my_artifact_schema",
        schema_version="v1.0"
    )
    output_artifact.train_path=train_data_path
    output_artifact.test_path=test_data_path
    return output_artifact
    # return train_data_path, test_data_path

@dsl.component
def train_model_op(path_artifact: dsl.Artifact) -> str:    
    train_data = path_artifact.train_path
    df = pd.read_csv(train_data)
    X = df.drop('species', axis=1)
    y = df['species']
    model = RandomForestClassifier()
    model.fit(X, y)
    model_path = f"{GCS_BUCKET}/trained_model_pipeline.pkl"
    joblib.dump(model, model_path)
    return model_path

@dsl.component
def predict_op(model_path: str, path_artifact: dsl.Artifact) -> str:
    model = joblib.load(model_path)
    test_data = path_artifact.test_path
    df = pd.read_csv(test_data)
    X_test = df.drop('species', axis=1)
    predictions = model.predict(X_test)    
    predictions_path = f"{GCS_BUCKET}/predictions_pipeline.csv"
    df['predictions'] = predictions
    df.to_csv(predictions_path, index=False)
    return predictions_path

@dsl.component
def store_predictions_in_bigquery_op(predictions_path: str):    
    bq_client = bigquery.Client.from_service_account_info(secret_info)
    bucket = storage_client.get_bucket(BUCKET_NAME) # mlops-prueba , -> mlops-400610_bucket
    
    # Consulta BigQuery las predicciones
    query = f"SELECT * FROM {BQ_DATASET}.{BQ_TABLE}"
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        print("Row:", row)

    # Carga los datos desde el archivo CSV a BigQuery
    dataset_ref = bq_client.dataset(BQ_DATASET)
    table_ref = dataset_ref.table(BQ_TABLE)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Detecta automáticamente el esquema

    with open(predictions_path, "rb") as source_file:
        job = bq_client.load_table_from_file(source_file, table_ref, job_config=job_config)
    job.result() 
    return f"Datos almacenados en BigQuery: {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"


@dsl.pipeline(
    name='Iris Pipeline',
    description='Pipeline to process the Iris dataset'
)
def iris_pipeline():        
    cleaned_data = read_and_preprocess_data_op()
    artifact_path = split_data_op(cleaned_data = cleaned_data.output)
    model_path = train_model_op(path_artifact = artifact_path.output)
    predictions = predict_op(model_path = model_path.output, path_artifact = artifact_path.output)
    result = store_predictions_in_bigquery_op(predictions_path = predictions.output)

# Ejecuta el pipeline
if __name__ == '__main__':

    aiplatform.init(project=PROJECT_ID, staging_bucket=GCS_BUCKET)

    kfp.compiler.Compiler().compile(iris_pipeline, 'iris-pipeline.yaml')


    job = aiplatform.PipelineJob(
        display_name=DISPLAY_RUN_NAME,
        template_path="iris-pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        # parameter_values=parameter_values
    )

    # endpoint = job.submit()
    endpoint = job.run()
    print(endpoint)
    # query = "SELECT * FROM prueba5.mytabla5"
    # query_job = bq_client.query(query)
    # results = query_job.result()
    # for row in results:
    #     print("Row:", row)

    # endpoint = model.deploy(machine_type="n1-standard-2")

