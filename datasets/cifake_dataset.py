from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build_transforms(image_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tfms, test_tfms


def build_cifake_loaders(
    root: str = "data/CIFAKE",
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
):
    root = Path(root)
    train_dir = root / "train"
    test_dir = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_tfms, test_tfms = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader, train_dataset.classes
