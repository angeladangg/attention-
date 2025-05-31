# data/cifar10.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=128, data_dir="./data/cifar10"):
    """
    Returns two PyTorch DataLoaders: (train_loader, test_loader).
    If the CIFAR-10 files aren’t already in `data_dir`, torchvision
    will automatically download and extract them the first time you run this.
    """
    # 1) Create the target folder if it doesn't exist yet
    os.makedirs(data_dir, exist_ok=True)

    # 2) Define transforms:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2471, 0.2435, 0.2616)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 3) Instantiate the CIFAR‑10 datasets (download if needed):
    train_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transforms,
        download=True
    )
    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transforms,
        download=True
    )

    # 4) Wrap them in DataLoaders:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader
