import torch
import json
from PIL import Image
from torchvision import transforms, models
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/resnet18.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image_path):
    with open("classes.json") as f:
        classes = json.load(f)

    model = load_model(len(classes))

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()

    return classes[pred]


if __name__ == "__main__":
    image_path = sys.argv[1]
    print(predict(image_path))
