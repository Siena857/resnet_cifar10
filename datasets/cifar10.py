import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.config import CONFIG

def get_cifar10_dataloader():
    """
    加载CIFAR-10数据集，带数据增强（提升准确率核心）
    返回: train_loader, test_loader
    """
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 测试集仅归一化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载数据集（自动下载到data/目录）
    train_dataset = datasets.CIFAR10(
        root=CONFIG["path"]["data_path"],
        train=True,
        download=False,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=CONFIG["path"]["data_path"],
        train=True,
        download=False,
        transform=test_transform
    )

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["train"]["batch_size"],
        shuffle=True,
        num_workers=0  # Windows设0，Linux/macOS可设4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["train"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader