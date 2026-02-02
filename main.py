from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ---------------- CONFIG ----------------
IMAGE_SIZE = 224
import os

# Use the absolute path you just confirmed
MODEL_PATH = "/Users/subhasisjena/crop_dieases_prediction_app/model.keras"

# Check if it exists before loading to catch errors early
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# ---------------- APP ----------------
app = FastAPI(title="Crop Disease Prediction")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- STATIC & TEMPLATES ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- LOAD MODEL ON STARTUP ----------------
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

# ---------------- UTILS ----------------
def preprocess_image(image_bytes):
    """Convert uploaded image bytes to model-ready array"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Explicitly set dtype to float32 to avoid NumPy 2.x object errors
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)
# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render homepage"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict crop disease from uploaded image"""
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"error": "Invalid file type. Please upload PNG or JPEG."})
    
    try:
        # Preprocess image
        image = preprocess_image(await file.read())
        
        # Predict
        prediction = model(image, training=False).numpy()
        
        # Get predicted class and confidence
        idx = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]) * 100)
        
        # Optionally return confidence for all classes
        all_confidences = {CLASS_NAMES[i]: round(float(pred)*100, 2) for i, pred in enumerate(prediction[0])}

        return {
            "predicted_class": CLASS_NAMES[idx],
            "confidence": round(confidence, 2),
            "all_confidences": all_confidences
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Failed to process the image"})
