from google.cloud import secretmanager
from google.cloud import storage
from google.cloud import bigquery
import os
import json
from dotenv import load_dotenv

load_dotenv() 

PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
GCS_BUCKET = f"gs://{BUCKET_NAME}" # "gs://tu-bucket"  # Reemplaza con la ubicación de tu bucket de GCS
LOCATION = os.getenv("LOCATION")
DISPLAY_RUN_NAME = os.getenv("DISPLAY_RUN_NAME")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")

print("default credential path: ", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

# secret_name = f"projects/{PROJECT_ID}/secrets/mlops-secret/versions/1" # or latest 
secret_name = "projects/550919589615/secrets/mlops-secret/versions/latest"

client = secretmanager.SecretManagerServiceClient()

response = client.access_secret_version(name=secret_name)

# Se recupera el valor del secreto (las credenciales JSON)
secret_value = response.payload.data.decode("UTF-8")

secret_info = json.loads(secret_value)

# Se inicializa los clientes de GCS, Vertex AI y BigQuery con las credenciales
storage_client = storage.Client.from_service_account_info(secret_info)
bq_client = bigquery.Client.from_service_account_info(secret_info)

# Ahora puedes utilizar los clientes para interactuar con los servicios
bucket = storage_client.get_bucket("mlops-400610_bucket")
blobs = bucket.list_blobs()
for blob in blobs:
    print("Blob:", blob.name)

# Consulta BigQuery
query = "SELECT * FROM prueba5.mytabla5"
query_job = bq_client.query(query)
results = query_job.result()
for row in results:
    print("Row:", row)

# Utiliza Vertex AI, por ejemplo, para entrenar un modelo (requiere configuración adicional)
