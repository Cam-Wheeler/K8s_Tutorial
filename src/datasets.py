import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Tuple


def get_cifar10_dataloaders(
    data_dir: str | None = None,
    train_transforms=None,
    test_transforms=None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR10 train and test dataloaders.

    Args:
        data_dir: Directory to download/load CIFAR10 data
        train_transforms: Transforms for training data
        test_transforms: Transforms for test data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Make sure we assign some data directory. 
    if data_dir is None:
        raise ValueError("Data directory is not assigned. Please assign.")

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transforms
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transforms
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    return train_dataloader, test_dataloader


def get_class_names() -> list:
    """Get CIFAR10 class names."""
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
