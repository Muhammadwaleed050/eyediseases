import base64
import io
import json
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .gradcam import GradCAM, overlay_heatmap_on_image



# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "best_model.pth")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "outputs", "classes.json")
IMG_SIZE = 224

device = torch.device("cpu")

# -----------------------------
# Load classes
# -----------------------------
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)
num_classes = len(classes)

# -----------------------------
# Model (EfficientNet-B0)
# -----------------------------
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# Target layer for Grad-CAM (EfficientNet last feature block)
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

# -----------------------------
# Preprocess
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def predict_and_gradcam(image: Image.Image):
    # keep original for overlay
    original = image.convert("RGB")

    # model input
    x = preprocess(original).unsqueeze(0)  # (1,3,224,224)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # Grad-CAM needs gradients (no torch.no_grad)
    # compute cam on predicted class
    cam_01 = gradcam.generate(x, pred_idx)
    overlay = overlay_heatmap_on_image(original, cam_01, alpha=0.45)

    return {
        "predicted_class": pred_label,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": {classes[i]: float(probs[i]) for i in range(num_classes)},
        "gradcam_overlay_base64": pil_to_base64(overlay),
    }

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="Eye Disease Detection API",
    description=(
        "AI-powered eye disease classification using **EfficientNet-B0** "
        "with **Grad-CAM** explainability.\n\n"
        "Detects: Cataract, Diabetic Retinopathy, Glaucoma, Normal."
    ),
    version="1.0.0",
)

# allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate MIME type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    contents = await file.read()

    # Validate file size (max 10 MB)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 10 MB.")

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict_and_gradcam(img)
    return result
