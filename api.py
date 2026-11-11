import io
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from face_core import embed, cosine_similarity

THRESHOLD = 0.7

app = FastAPI()

mean = np.load("mean.npy")
components = np.load("components.npy")
registry = pd.read_pickle("registry.pkl")


def image_to_vector(data):
    img = Image.open(io.BytesIO(data))
    img = img.convert("L").resize((64, 64))
    return (np.array(img) / 255.0).flatten()


@app.post("/verify")
def verify(person_id: int = Form(...), image: UploadFile = File(...)):
    rows = registry[registry["person_id"] == person_id]
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail="person not registered")

    data = image.file.read()
    fp = embed(image_to_vector(data), mean, components)
    scores = [cosine_similarity(fp, np.array(r)) for r in rows["fingerprint"]]
    best = max(scores)
    return {
        "person_id": person_id,
        "score": round(float(best), 3),
        "match": bool(best >= THRESHOLD),
    }
