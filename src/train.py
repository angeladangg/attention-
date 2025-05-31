# scripts/train.py

import torch
import torch.nn as nn
import torch.optim as optim

# 1) Import the CIFAR‑10 loader from data/cifar10.py
from data.cifar10 import get_cifar10_dataloaders

# 2) Import your VisionTransformer model
from models.vit_cifar import VisionTransformer

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) CALL get_cifar10_dataloaders → triggers download if needed
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=128,
        data_dir="./data/cifar10"
    )

    # 4) Instantiate your ViT model (match arguments to vit_cifar.py)
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}  Loss: {train_loss:.4f}  Test Acc: {val_acc*100:.2f}%")

    torch.save(model.state_dict(), "vit_cifar10_final.pth")

if __name__ == "__main__":
    main()
