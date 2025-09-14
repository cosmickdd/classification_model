clip_patch_heatmap.py
import torch, numpy as np, cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "apple/mobileclip-vit-b32"

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
                model = model.half()
            model = model.to(device).eval()
        except Exception as e:
            print(f"Error loading model or processor: {e}")
            exit(1)

def patch_heatmap(image_path, text="a photo of a mangrove forest", patch=224, stride=112):
    load_model()
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    W, H = img.size
    img_np = np.array(img)

    # Encode text once
    text_inputs = processor(text=[text], return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        tfeat = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    heat = np.zeros((H, W), np.float32)
    count = np.zeros_like(heat)

    # Sliding window over image
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            crop = img.crop((x, y, x + patch, y + patch)).resize((224, 224))
            inputs = processor(images=crop, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                ifeat = model.get_image_features(pixel_values=inputs["pixel_values"])
                ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
                sim = (ifeat @ tfeat.T).cpu().item()
                score = (sim + 1) / 2.0
            heat[y:y+patch, x:x+patch] += score
            count[y:y+patch, x:x+patch] += 1

    heat[count > 0] /= count[count > 0]
    heat = cv2.resize(heat, (W, H))

    plt.imshow(img_np)
    plt.imshow(heat, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.title("Mangrove similarity heatmap")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clip_patch_heatmap.py samples/image.png")
    else:
        patch_heatmap(sys.argv[1])