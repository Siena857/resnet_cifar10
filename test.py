import os
import torch
from utils import CONFIG, LOGGER
from models import init_resnet
from datasets import get_cifar10_dataloader

def test():
    """测试入口函数：加载最佳权重评估最终准确率"""
    # 1. 初始化模型（关闭预训练，加载自定义权重）
    CONFIG["model"]["pretrained"] = False
    model = init_resnet()

    # 2. 加载最佳权重
    best_weight_path = os.path.join(CONFIG["path"]["output_path"], "best_model.pth")
    if not os.path.exists(best_weight_path):
        raise FileNotFoundError(f"最佳权重文件不存在: {best_weight_path}")
    model.load_state_dict(torch.load(best_weight_path, map_location=CONFIG["train"]["device"]))
    model.eval()  # 切换到评估模式
    LOGGER.info(f"成功加载最佳权重: {best_weight_path}")

    # 3. 加载测试集
    _, test_loader = get_cifar10_dataloader()
    LOGGER.info(f"测试集加载完成: {len(test_loader.dataset)} 条数据")

    # 4. 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(CONFIG["train"]["device"]), targets.to(CONFIG["train"]["device"])
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    LOGGER.info(f"最终测试准确率: {test_acc:.2f}%")
    return test_acc

if __name__ == "__main__":
    test()