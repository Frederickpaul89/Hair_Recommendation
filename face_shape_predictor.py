import os
import requests
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------
# OPTIONAL MODEL DOWNLOADER
# -----------------------------
def download_model_if_not_exists(url, model_path):
    """Download model from Hugging Face if not present."""
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        r = requests.get(url)
        if r.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(r.content)
            print("Model downloaded successfully.")
        else:
            raise ValueError("Failed to download model. Check URL.")
    else:
        print("Model already exists.")

# -----------------------------
# LOAD MODEL (CALL ONLY ONCE)
# -----------------------------
def load_model(model_path):
    """Load model from file ONCE and reuse it."""
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    model.to(device)
    return model

# -----------------------------
# PREPROCESS IMAGE
# -----------------------------
def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# -----------------------------
# GET SOFTMAX PROBABILITIES
# -----------------------------
def get_probabilities(logits):
    probs = F.softmax(logits, dim=1)
    return probs * 100  # convert to percentage

# -----------------------------
# PREDICT FACE SHAPE
# -----------------------------
def predict(image_path, model, class_names):
    img_tensor = preprocess_image(image_path).to(device)
    model.eval()

    with torch.inference_mode():
        outputs = model(img_tensor)
        percentages = get_probabilities(outputs)
        _, pred_idx = torch.max(outputs, 1)

    return class_names[pred_idx.item()], percentages

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = ["Heart", "Oblong", "Oval", "Round", "Square"]

# -----------------------------
# MAIN FUNCTION (NO MODEL LOADING)
# -----------------------------
def miain(image_path, model):
    """
    Predict face shape using an already LOADED model.
    """
    pred_label, percentages = predict(image_path, model, class_names)

    result = {
        class_names[i]: percentages[0, i].item()
        for i in range(len(class_names))
    }

    sorted_result = dict(
        sorted(result.items(), key=lambda x: x[1], reverse=True)
    )

    return sorted_result
