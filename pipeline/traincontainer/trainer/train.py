
import os 
import json
import pandas as pd
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import secretmanager
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from dotenv import load_dotenv

load_dotenv() 

model_directory = os.environ["AIP_MODEL_DIR"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
FILE_NAME = 'iris-2.csv'
SECRET_MAME = os.environ["SECRET_NAME"]
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(name=SECRET_MAME)
SECRET_vALUE = response.payload.data.decode("UTF-8")
print("secret value: ",SECRET_vALUE)
SECRET_INFO = json.loads(SECRET_vALUE)
storage_client = storage.Client.from_service_account_info(SECRET_INFO)
bigquery_client = bigquery.Client.from_service_account_info(SECRET_INFO)

def download_table(bq_table_uri: str):
    prefix = "bq://"
    if bq_table_uri.startswith(prefix):
        bq_table_uri = bq_table_uri[len(prefix):]
    table = bigquery.TableReference.from_string(bq_table_uri)
    rows = bigquery_client.list_rows(
        table,
    )
    return rows.to_dataframe(create_bqstorage_client=False)


def upload_dataset_to_gcs(bucket_name: str = BUCKET_NAME):
    bucket = storage_client.get_bucket(BUCKET_NAME)
    iris = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
    csv_data = iris.to_csv(index=False)
    blob = bucket.blob(FILE_NAME)
    blob.upload_from_string(csv_data, 'text/csv')
    print(f'Conjunto de datos Iris cargado en gs://{BUCKET_NAME}/{FILE_NAME}')



def training():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    dump(rf_model, "model.joblib")
    bucket = storage_client.get_bucket(BUCKET_NAME)
    model_directory = os.environ["AIP_MODEL_DIR"]
    storage_path = os.path.join(model_directory, "model.joblib")
    blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
    blob.upload_from_filename("model.joblib")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # print(f"Precisión (Accuracy): {accuracy}")
    # print("Matriz de Confusión:")
    # print(confusion)
    # print("Informe de Clasificación:")
    # print(report)




training()