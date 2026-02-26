import os
import torch
import torch.nn as nn
from torchvision import models
from utils.config import CONFIG
from utils.logger import LOGGER

def init_resnet():
    """
    初始化ResNet模型（torchvision官方实现）
    """
    # 1. 实例化模型 + 加载官方预训练权重
    if CONFIG["model"]["name"] == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if CONFIG["model"]["pretrained"] else None
        )
    elif CONFIG["model"]["name"] == "resnet34":
        model = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if CONFIG["model"]["pretrained"] else None
        )
    else:
        raise ValueError(f"不支持的模型: {CONFIG['model']['name']}")

    # 2. 适配CIFAR-10的10分类（替换最后一层）
    model.fc = nn.Linear(model.fc.in_features, CONFIG["model"]["num_classes"])

    # 3. 加载自定义权重
    if CONFIG["model"]["weight_path"] and os.path.exists(CONFIG["model"]["weight_path"]):
        model.load_state_dict(
            torch.load(CONFIG["model"]["weight_path"], map_location=CONFIG["train"]["device"])
        )
        LOGGER.info(f"成功加载自定义权重: {CONFIG['model']['weight_path']}")

    # 4. 移到指定设备
    model = model.to(CONFIG["train"]["device"])
    LOGGER.info(f"模型初始化完成: {CONFIG['model']['name']}, 设备: {CONFIG['train']['device']}")
    return model