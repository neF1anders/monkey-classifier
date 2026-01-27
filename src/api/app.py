from fastapi import FastAPI, UploadFile, File
import shutil
from pathlib import Path

from ..inference.search_similar import search

app = FastAPI()

UPLOAD_DIR = Path("tmp")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pred_class, similar_images = search(str(file_path))

    return {
        "predicted_class": pred_class,
        "similar_images": similar_images
    }
