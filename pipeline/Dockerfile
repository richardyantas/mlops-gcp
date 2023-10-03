FROM python:3.9.16-slim-bullseye

COPY pipeline.py /app/pipeline.py
COPY dev.txt /app/dev.txt

RUN pip install -r /app/prod.txt
COPY .env /app/.env

# Directorio de trabajo
WORKDIR /app
