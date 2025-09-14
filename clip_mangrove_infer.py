 
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch16"

# Lazy load globals
model = None
processor = None

def load_model():
    """Load CLIP model only when needed."""
    global model, processor
    if model is None or processor is None:
        try:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            if device == "cuda":
                model = model.half()  # use half precision on GPU
            model = model.to(device).eval()
        except Exception as e:
            print(f"Error loading model or processor: {e}")
            processor = None
            model = None

texts = [
    "a photo of a mangrove forest",
    "a photo of a coastal area without mangroves",
    "a photo of trees but not mangrove",
    "a close-up of mangrove trees",
]

def predict_image(path, texts=texts, temp=1.0):
    load_model()
    try:
        image = Image.open(path).convert("RGB")
        image = image.resize((224, 224))  # shrink for memory safety
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    if model is None or processor is None:
        print(f"Model or processor not loaded. Types: model={type(model)}, processor={type(processor)}")
        return None
    try:
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
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
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clip_mangroove_infer.py samples/image.png")
    else:
        result = predict_image(sys.argv[1])
        if result is not None:
            print(result)