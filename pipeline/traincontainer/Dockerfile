# FROM gcr.io/deeplearning-platform-release/sklearn-cpu.0-23
FROM python:3.11-slim
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer/train.py /trainer/train.py
COPY trainer/.env /trainer/.env
RUN pip install scikit-learn google-cloud-bigquery joblib pandas google-cloud-storage
# prod.txt ?
# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]
