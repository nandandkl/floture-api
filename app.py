from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from tensorflow import keras

app = FastAPI()

# Allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = keras.models.load_model("flower_model.h5")

class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.50


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).resize((180, 180))
    img = img.convert("RGB")
    img = np.array(img)
    img = np.expand_dims(img, 0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx])

    # 🔥 If confidence is low → return no flower detected
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "class": "no flower detected",
            "confidence": confidence
        }

    # Otherwise return real class
    return {
        "class": class_names[idx],
        "confidence": confidence
    }
