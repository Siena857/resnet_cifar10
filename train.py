import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# ====================== 1. 工具函数：日志配置 ======================
def setup_logger(config):
    """配置日志输出（控制台 + 文件）"""
    log_dir = config['path']['log_path']
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名
    log_name = f"resnet_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_name)
    
    # 配置logger
    logger = logging.getLogger('resnet_train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清空原有handler
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # 文件handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# ====================== 2. 工具函数：加载配置 ======================
def load_config(config_path="train_config.yaml"):
    """加载yaml配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# ====================== 3. 数据加载 ======================
def build_dataloader(config):
    """构建CIFAR10训练/验证数据集加载器"""
    data_path = config['path']['data_path']
    batch_size = config['train']['batch_size']
    
    # 数据预处理（适配CIFAR10）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=val_transform
    )
    
    # 构建DataLoader（Windows下num_workers=0避免报错）
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    
    return train_loader, val_loader

# ====================== 4. 模型构建 ======================
def build_model(config):
    """构建ResNet模型（支持resnet18/resnet34）"""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    weight_path = config['model']['weight_path']
    
    # 加载基础模型
    if model_name == "resnet18":
        model = resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        model = resnet34(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型 {model_name}，仅支持 resnet18/resnet34")
    
    # 修改最后一层适配CIFAR10分类
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 加载自定义权重
    if weight_path and os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"已加载自定义权重：{weight_path}")
    
    return model

# ====================== 5. Backbone冻结/解冻逻辑 ======================
def set_model_gradient(model, config, current_epoch):
    """根据当前轮数设置模型梯度（冻结/解冻backbone）"""
    freeze_backbone = config['model']['freeze_backbone']
    freeze_epochs = config['train']['freeze_backbone_epochs']
    unfreeze_layers = config['model']['unfreeze_layers']
    freeze_bn = config['model']['freeze_bn']
    
    # 1. 前freeze_epochs轮冻结backbone，之后解冻
    if freeze_backbone and current_epoch < freeze_epochs:
        logger.info(f"Epoch {current_epoch+1}: 冻结backbone，仅训练 {unfreeze_layers} 层")
        for name, param in model.named_parameters():
            # 强制不冻结的层（如fc）始终开启梯度
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # 解冻所有层（仅在解冻第一轮提示）
        if current_epoch == freeze_epochs and freeze_backbone:
            logger.info(f"Epoch {current_epoch+1}: 解冻backbone，训练所有层")
        for param in model.parameters():
            param.requires_grad = True
    
    # 2. 处理BN层（可选冻结）
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # BN层设为eval模式，不更新均值/方差
    
    # 3. 返回可训练的参数列表（解决梯度报错核心）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return model, trainable_params

# ====================== 6. 单轮训练/验证 ======================
def train_one_epoch(model, train_loader, criterion, optimizer, device, logger):
    """训练单个epoch"""
    model.train()  # 强制训练模式
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 梯度清零（必须步骤）
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播（确保loss可导）
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 计算指标
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += batch_size
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{total_loss/total_samples:.4f}",
            'acc': f"{total_correct/total_samples:.4f}"
        })
    
    # 计算epoch级指标
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    logger.info(f"Train | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

@torch.no_grad()  # 验证阶段禁用梯度
def val_one_epoch(model, val_loader, criterion, device, logger):
    """验证单个epoch"""
    model.eval()  # 强制验证模式
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播（无梯度）
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 计算指标
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += batch_size
        
        pbar.set_postfix({
            'val_loss': f"{total_loss/total_samples:.4f}",
            'val_acc': f"{total_correct/total_samples:.4f}"
        })
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    logger.info(f"Valid | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

# ====================== 7. 主训练逻辑 ======================
def main():
    # 全局logger
    global logger
    
    # 1. 加载配置
    config = load_config(config_path="./configs/train_config.yaml")
    # 2. 配置日志
    logger = setup_logger(config)
    # 3. 设置设备（CPU）
    device = torch.device(config['train']['device'])
    logger.info(f"训练设备：{device}")
    
    # 4. 构建数据加载器
    train_loader, val_loader = build_dataloader(config)
    logger.info(f"数据集加载完成 | 训练集数量：{len(train_loader.dataset)} | 验证集数量：{len(val_loader.dataset)}")
    
    # 5. 构建模型
    model = build_model(config)
    model = model.to(device)
    logger.info(f"模型构建完成：{config['model']['name']}")
    
    # 6. 定义损失函数和优化器基础配置
    criterion = nn.CrossEntropyLoss()
    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    unfreeze_lr = config['train']['unfreeze_lr']
    
    # 7. 创建输出目录
    output_dir = config['path']['output_path']
    os.makedirs(output_dir, exist_ok=True)
    
    # 8. 早停相关参数
    early_stop_patience = config['train']['early_stop_patience']
    best_val_acc = 0.0
    patience_counter = 0
    
    # 9. 训练主循环
    for epoch in range(config['train']['epochs']):
        logger.info(f"\n========== Epoch {epoch+1}/{config['train']['epochs']} ==========")
        
        # 9.1 设置模型梯度（冻结/解冻backbone）
        model, trainable_params = set_model_gradient(model, config, epoch)
        
        # 9.2 动态构建优化器（仅训练可导参数）
        optimizer = optim.SGD(
            trainable_params,
            lr=lr if epoch < config['train']['freeze_backbone_epochs'] else unfreeze_lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # 9.3 单轮训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # 9.4 单轮验证
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device, logger)
        
        # 9.5 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = os.path.join(output_dir, f"best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"保存最佳模型 | 验证准确率：{best_val_acc:.4f} | 路径：{best_model_path}")
        else:
            patience_counter += 1
        
        # 9.6 按间隔保存模型
        if (epoch+1) % config['train']['save_interval'] == 0:
            checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}_acc_{val_acc:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"保存轮次模型 | 路径：{checkpoint_path}")
        
        # 9.7 早停判断
        if patience_counter >= early_stop_patience:
            logger.info(f"早停触发 | 连续{early_stop_patience}轮验证准确率未提升 | 最佳准确率：{best_val_acc:.4f}")
            break
    
    # 10. 训练结束
    logger.info(f"\n训练完成 | 最佳验证准确率：{best_val_acc:.4f} | 模型保存路径：{output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if 'logger' in globals():
            logger.error(f"训练出错：{str(e)}", exc_info=True)
        else:
            print(f"训练出错：{str(e)}")
            import traceback
            traceback.print_exc()
        raise