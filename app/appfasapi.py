
from fastapi import FastAPI
import os
app = FastAPI()
GCS_BUCKET = os.getenv("GCS_BUCKET")

@app.get("/")
async def root():
    return {"message": "Hello World"}