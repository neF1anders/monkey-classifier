import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import yaml
import json

from data.dataset import get_dataloaders
from models.resnet18 import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

EPOCHS = params["epochs"]
LR = params["lr"]
BATCH_SIZE = params["batch_size"]

def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return loss_sum / len(loader), correct / total

def main(output_path):
    train_loader, val_loader, num_classes = get_dataloaders(BATCH_SIZE)

    model = get_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        best_acc = max(best_acc, val_acc)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    with open("metrics.json", "w") as f:
        json.dump({"best_val_acc": best_acc}, f)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outs", type=str, required=True)
    args = parser.parse_args()

    main(args.outs)
