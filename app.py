from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn.functional as F

from PIL import Image

import torchvision.transforms as transforms

import numpy as np

import io

from anglenet_models.anglenet import AngleNet

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()

# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print(f"DEVICE: {device}")

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading AngleNet...")

model = AngleNet()

model.load_state_dict(

    torch.load(
        "anglenet_weights.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

print("✅ AngleNet Loaded")

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()
])

# ============================================================
# LABELS
# ============================================================

severity_labels = [

    "Normal",

    "Mild",

    "Moderate",

    "Severe"
]

# ============================================================
# ROOT
# ============================================================

@app.get("/")

def root():

    return {

        "message": "AngleNet API Running"
    }

# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")

async def predict(
    file: UploadFile = File(...)
):

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = await file.read()

        image = Image.open(

            io.BytesIO(image_bytes)

        ).convert("RGBA")

        # ====================================================
        # TRANSFORM
        # ====================================================

        image_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        # ====================================================
        # PREDICTION
        # ====================================================

        with torch.no_grad():

            angle_pred, severity_pred = model(
                image_tensor
            )

            angle = angle_pred.item()

            severity_idx = torch.argmax(

                F.softmax(
                    severity_pred,
                    dim=1
                )

            ).item()

        severity = severity_labels[
            severity_idx
        ]

        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "cobb_angle": round(angle, 2),

            "severity": severity
        }

    except Exception as e:

        return {

            "error": str(e)
        }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000
    )
