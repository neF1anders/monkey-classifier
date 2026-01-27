import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_backbone(num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/resnet18.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # remove last layer
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    return backbone


def image_to_vector(image_path, backbone):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        vec = backbone(x).squeeze().cpu().numpy()

    return vec