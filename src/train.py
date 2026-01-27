import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from data.dataset import get_dataloaders
from models.resnet18 import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
LR = 1e-3
BATCH_SIZE = 32

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

def main():
    train_loader, val_loader, num_classes = get_dataloaders(BATCH_SIZE)
    model = get_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for x,y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}, val_accuracy = {val_acc:.4f}")
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/resnet18.pt")

if __name__ == "main":
    main()
