from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from tensorflow import keras
import os
import uvicorn

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API running"}

# Load model
model = keras.models.load_model("flower_model.h5")

class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.70

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    img = Image.open(file.file).resize((180, 180))
    img = img.convert("RGB")
    img = np.array(img)
    img = np.expand_dims(img, 0)

    # Make prediction
    pred = model.predict(img)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return {"class": "no flower detected", "confidence": confidence}

    return {"class": class_names[idx], "confidence": confidence}

# Run Uvicorn with Render-compatible port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
