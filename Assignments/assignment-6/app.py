
import os
import time
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
import gradio as gr
import json

MODEL_PATH = "outputs/best_model.pth"

# Load class names
def load_class_names():
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            return json.load(f)
    return {f"class_{i}": f"class_{i}" for i in range(200)}

CLASS_NAMES = load_class_names()

# Download model from W&B if not present
if not os.path.exists(MODEL_PATH):
    try:
        import wandb
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            print("Downloading model from W&B...")
            wandb.login(key=wandb_api_key)
            api = wandb.Api()
            artifact_path = os.environ.get("WANDB_ARTIFACT", 
            "ir2023/tiny-imagenet-assignment/resnet18-tiny-imagenet:latest")
            print(f"Downloading: {artifact_path}")
            artifact = api.artifact(artifact_path)
            artifact.download(root="outputs")
            print("Model downloaded from W&B")
        else:
            print("WANDB_API_KEY not set in Space Secrets")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Load model
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 200)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def predict_image(img):
    if img is None:
        return {"error": "No image provided"}

    start_time = time.time()

    try:
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top5_probs, top5_idx = torch.topk(probs, 5)

        results = {}
        for prob, idx in zip(top5_probs[0], top5_idx[0]):
            class_idx = int(idx.item())
            class_name = list(CLASS_NAMES.values())[class_idx]
            class_id = list(CLASS_NAMES.keys())[class_idx]
            results[f"{class_name} ({class_id})"] = float(prob.item())

        latency = (time.time() - start_time) * 1000.0
        print(f"Prediction time: {latency:.2f}ms")

        return results

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Create Gradio interface
title = "Tiny ImageNet Classifier"
description = '''
Upload an image to classify it into one of 200 Tiny ImageNet classes.

**Model Details:**
- Architecture: ResNet-18
- Dataset: Tiny ImageNet (200 classes)
- Input Size: 224x224 RGB
- Framework: PyTorch

Model is automatically downloaded from W&B at startup.
'''

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title=title,
    description=description,
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
