# annotators/scene_classifier.py

import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import requests
import io

from pathlib import Path

# Load Places365 categories
def load_categories():
    file_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
    response = requests.get(file_url)
    return [line.strip().split(" ")[0][3:] for line in response.text.splitlines()]

CATEGORIES = load_categories()

# Load model
def load_model():
    model = resnet18(num_classes=365)
    weight_url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
    checkpoint = torch.hub.load_state_dict_from_url(weight_url, progress=True, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_scene(image: Image.Image) -> str:
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_class = torch.argmax(probs).item()
    return CATEGORIES[top_class]

def estimate_scene(video_path: Path) -> str:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "unknown"
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return classify_scene(img)
