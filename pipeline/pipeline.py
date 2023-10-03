import kfp
from kfp import dsl
from google.cloud import aiplatform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.cloud import bigquery
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
GCS_BUCKET = f"gs://{BUCKET_NAME}" # "gs://tu-bucket"  # Reemplaza con la ubicación de tu bucket de GCS
LOCATION = os.getenv("LOCATION")
DISPLAY_RUN_NAME = os.getenv("DISPLAY_RUN_NAME")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
# secret_name = f"projects/{PROJECT_ID}/secrets/mlops-secret/versions/1" # or latest 
secret_name = "projects/550919589615/secrets/mlops-secret/versions/latest"
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(name=secret_name)
secret_value = response.payload.data.decode("UTF-8")
print("secret value: ",secret_value)
secret_info = json.loads(secret_value)
storage_client = storage.Client.from_service_account_info(secret_info)

@dsl.component
def read_data_op():
    data = aiplatform.TabularDataset.create(
        display_name="iris",
        gcs_source=f"{GCS_BUCKET}/iris.csv",
    )    
    df = pd.read_csv(data)
    df = df.dropna()    
    return df

@dsl.component
def preprocess_data_op(df: str) -> str:
    
    cleaned_data_path = f"{GCS_BUCKET}/cleaned_data.csv"
    df.to_csv(cleaned_data_path, index=False)
    return cleaned_data_path
    

@dsl.component
def split_data_op(cleaned_data: str) -> str:
    df = pd.read_csv(cleaned_data)
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data_path = f"{GCS_BUCKET}/train_iris.csv"
    test_data_path = f"{GCS_BUCKET}/test_iris.csv"
    X_train.to_csv(train_data_path, index=False)
    X_test.to_csv(test_data_path, index=False)
    return train_data_path, test_data_path

@dsl.component
def train_model_op(train_data: str) -> str:    
    df = pd.read_csv(train_data)
    X = df.drop('species', axis=1)
    y = df['species']
    model = RandomForestClassifier()
    model.fit(X, y)
    model_path = f"{GCS_BUCKET}/trained_model.pkl"
    joblib.dump(model, model_path)
    return model_path

@dsl.component
def predict_op(model_path: str, test_data: str) -> str:
    model = joblib.load(model_path)
    df = pd.read_csv(test_data)
    X_test = df.drop('species', axis=1)
    predictions = model.predict(X_test)    
    predictions_path = f"{GCS_BUCKET}/predictions.csv"
    df['predictions'] = predictions
    df.to_csv(predictions_path, index=False)
    return predictions_path

@dsl.component
def store_predictions_in_bigquery_op(predictions_path: str):
    project_id = PROJECT_ID
    dataset_id = "resultados"
    table_id = "predicciones"
    # client = bigquery.Client(project=project_id)
    bq_client = bigquery.Client.from_service_account_info(secret_info)
    bucket = storage_client.get_bucket("mlops-prueba") # mlops-prueba , mlops-400610_bucket
    # Consulta BigQuery las predicciones
    query = "SELECT * FROM resultados.predicciones"
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        print("Row:", row)
    # Carga los datos desde el archivo CSV a BigQuery
    dataset_ref = bq_client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Detecta automáticamente el esquema    
    with open(predictions_path, "rb") as source_file:
        job = bq_client.load_table_from_file(source_file, table_ref, job_config=job_config)
    job.result() 
    return f"Datos almacenados en BigQuery: {project_id}.{dataset_id}.{table_id}"

# Define el pipeline

@dsl.pipeline(
    name='Iris Pipeline',
    description='Pipeline to process the Iris dataset'
)
def iris_pipeline():    
    data = read_data_op().outputs
    cleaned_data = preprocess_data_op(df = data).output
    train_data, test_data = split_data_op(cleaned_data = cleaned_data).output
    model_path = train_model_op(train_data = train_data)
    predictions = predict_op(model_path, test_data)
    result = store_predictions_in_bigquery_op(predictions)

# Ejecuta el pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(iris_pipeline, 'iris-pipeline.yaml')
