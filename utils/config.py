import os
import yaml
from pathlib import Path

def load_config(config_path="configs/train_config.yaml"):
    """加载yaml配置文件，自动创建输出目录"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 自动创建输出目录（避免运行报错）
    Path(config["path"]["output_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["path"]["log_path"]).mkdir(parents=True, exist_ok=True)
    return config

# 全局配置对象（其他模块可直接导入）
CONFIG = load_config()