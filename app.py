

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import io
import os

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("mangrove_mobilenetv2.pth", map_location=device))
model = model.to(device).eval()

# Class mapping: 0 = forest, 1 = mangrove (ImageFolder sorts alphabetically)
class_names = ['forest', 'mangrove']

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image):
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = int(probs.argmax())
            is_mangrove = (class_names[pred_idx] == 'mangrove')
            return {
                "is_mangrove": is_mangrove,
                "probability": float(probs[pred_idx])
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
    return {"message": "Mangrove classifier API. POST an image to /predict. Returns is_mangrove: true/false."}