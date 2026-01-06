from torchvision import transforms


def get_train_transforms():
    """
    Get training transforms with data augmentation for CIFAR10.

    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])


def get_test_transforms():
    """
    Get test transforms without augmentation for CIFAR10.

    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
