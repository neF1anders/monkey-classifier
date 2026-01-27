from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = "../../data/processed/ready"

def get_dataloaders(batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(dataset.classes)
