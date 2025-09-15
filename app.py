
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import io

app = FastAPI()

# Use MobileNetV2 (pretrained on ImageNet)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v2(pretrained=True)
model = model.to(device).eval()

# ImageNet class labels (first 1000)
import json
import urllib.request
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    categories = [line.strip().decode("utf-8") for line in f.readlines()]

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image: Image.Image):
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_catid = torch.topk(probs, 5)
        return {
            "top5": [
                {"label": categories[catid], "prob": float(prob)}
                for prob, catid in zip(top5_prob, top5_catid)
            ]
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
    return {"message": "MobileNetV2 ImageNet demo API. POST an image to /predict for top-5 labels."}