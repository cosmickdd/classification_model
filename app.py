from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import io

app = FastAPI()

# Load model and processor once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use a lighter model (patch16 is smaller than patch32)
model_name = "openai/clip-vit-base-patch16"
try:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device).eval()
except Exception as e:
    import sys
    print(f"Error loading model: {e}", file=sys.stderr)
    processor = None
    model = None

texts = [
    "a photo of a mangrove forest",
    "a photo of a coastal area without mangroves",
    "a photo of trees but not mangrove",
    "a close-up of mangrove trees",
]

def predict_image(image: Image.Image, texts=texts, temp=1.0):
    if processor is None or model is None:
        return {"error": "Model not loaded. Check server logs."}
    try:
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image / temp
            probs = logits.softmax(dim=1)[0].cpu().numpy()
        mangrove_score = float(probs[0] + probs[3])  # combine mangrove prompts
        best_idx = int(probs.argmax())
        return {
            "best_text": texts[best_idx],
            "best_prob": float(probs[best_idx]),
            "mangrove_score": mangrove_score,
            "all_probs": [float(x) for x in probs],
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return JSONResponse(status_code=400, content={"error": "Only PNG and JPEG images are supported."})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {e}"})
    result = predict_image(image)
    if "error" in result:
        return JSONResponse(status_code=500, content=result)
    return result

@app.get("/")
def root():
    return {"message": "Mangrove image classification API. POST an image to /predict."}
