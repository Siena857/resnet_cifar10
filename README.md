# ResNet18/34 图像分类（CIFAR-10）

基于 PyTorch 实现的 ResNet 系列模型（ResNet18/ResNet34），针对 CIFAR-10 数据集完成图像分类任务，采用 YAML 配置文件统一管理超参数，支持 CPU/GPU 双环境训练。

## 项目简介

- **模型**：支持 ResNet18/ResNet34 切换，适配 CIFAR-10 数据集 32×32 输入尺寸与 10 分类任务
- **超参数管理**：使用 YAML 配置文件（'configs/train_config.yaml'）统一管理，参数修改无需改动代码
- **训练策略**：支持 Backbone 冻结训练（前N轮冻结，后N轮解冻），兼顾训练效率与效果
- **数据集**：CIFAR-10（torchvision 自动下载，无需手动预处理）
- **训练环境**：兼容 CPU/GPU 训练，**推荐 GPU 环境以获得更快训练速度**
- **日志输出**：训练过程自动记录，完整CPU训练日志见 'docs/resnet18_cifar10_train_log.md'，完整GPU训练日志见'works/logs'

## 环境准备

### 1. 克隆项目

```bash
git clone https://github.com/Siena857/resnet18-cifar10-cpu.git 
cd resnet18-cifar10-cpu
2. 创建并激活虚拟环境
Windows 系统（PowerShell/CMD）
powershell
复制
# 创建虚拟环境（Python 3.11 为例）
conda create -n resnet_cifar10 python=3.11 -y

# 激活虚拟环境（PowerShell）
conda activate resnet_cifar10
# 若报错"执行策略禁止"，先执行：Set-ExecutionPolicy RemoteSigned -Scope CurrentUser（按提示输入 Y 确认）

# 激活虚拟环境（CMD）
conda activate resnet_cifar10
macOS/Linux 系统
bash

# 创建虚拟环境
conda create -n resnet_cifar10 python=3.11 -y
# 激活虚拟环境
conda activate resnet_cifar10
3. 安装依赖包
方案 A：GPU 环境（推荐，需 NVIDIA 显卡）
针对 RTX 50 系列显卡（如 RTX 5060）及新架构 GPU：
⚠️ 重要提示：RTX 50 系列采用 Blackwell 架构（计算能力 12.0），需要 PyTorch 2.6.0+ 配合 CUDA 12.8+ 才能支持。
bash

# 安装 CUDA 12.8 版本的 PyTorch（支持 RTX 50 系列）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
其他 NVIDIA 显卡（RTX 40 系列及更早）：
bash

# CUDA 12.1 版本（适用于大多数现代 NVIDIA 显卡）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8 版本（适用于旧版显卡）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
方案 B：CPU 环境（无 NVIDIA 显卡）
bash

# CPU 版本 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
安装其他依赖
bash

# 安装剩余依赖
pip install tqdm pyyaml numpy==1.24.4

# 若下载速度慢，添加清华镜像源：
pip install tqdm pyyaml numpy==1.24.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
4. 验证环境
安装完成后，验证 PyTorch 是否正确识别 GPU：
bash

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
成功输出示例：
plain

PyTorch: 2.10.0+cu128
CUDA可用: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
快速开始
1. 配置超参数
编辑 configs/train_config.yaml 文件调整核心参数，关键配置示例：
yaml

#### 模型配置（切换 ResNet18/34）
model:
  name: "resnet18"        # 可选：resnet18 / resnet34
  num_classes: 10         # CIFAR-10 固定为10类
  freeze_backbone: True   # 启用 Backbone 冻结训练

#### 训练配置（切换 CPU/GPU）
train:
  epochs: 100             # 训练轮数
  batch_size: 128         # 批次大小（GPU 可设 128/256，CPU 建议 16/32）
  device: "cuda"          # 训练设备：cpu / cuda
  freeze_backbone_epochs: 20  # 冻结 Backbone 训练轮数

#### 路径配置
path:
  data_path: "./data"                  # 数据集保存路径
  output_path: "./work_dirs/resnet18"  # 模型权重输出路径
性能提示：
GPU 训练（推荐）：batch_size 建议 128 或 256，100 轮训练约 10-20 分钟（RTX 5060）
CPU 训练：batch_size 建议 16 或 32，100 轮训练时间较长
2. 训练模型
直接运行训练脚本，自动加载配置文件完成训练：
bash

# 启动训练（根据配置文件自动选择 CPU/GPU）
python train.py

# 切换 ResNet34 训练：修改 configs/train_config.yaml 中 model.name 为 "resnet34" 后重新运行
# 切换 CPU 训练：修改 configs/train_config.yaml 中 train.device 为 "cpu" 后重新运行
训练过程：
终端实时打印每轮 Loss/Acc 指标，进度条可视化训练进度
每 10 轮保存一次轮次模型，自动保存验证准确率最高的「最佳模型」
完整训练日志自动输出至终端，已整理至 docs/resnet18_cifar10_train_log.md
3. 训练结果说明

CPU 训练示例（10 轮验证）
本项目提供 CPU 环境 10 轮训练日志作为代码可行性验证：
表格
训练轮数	训练损失	训练准确率	验证损失	验证准确率	备注
1	1.7446	36.82%	1.3527	51.30%	冻结 Backbone
2	1.4279	48.80%	1.1966	57.80%	冻结 Backbone
3	1.2630	55.09%	1.0464	63.09%	冻结 Backbone
4	1.1424	59.77%	0.9880	65.19%	冻结 Backbone
5	1.0553	62.84%	0.9312	67.48%	冻结 Backbone
6	0.9113	67.90%	0.7967	71.83%	解冻 Backbone
7	0.8760	69.06%	0.7890	72.26%	解冻 Backbone
8	0.8598	69.72%	0.7695	72.56%	解冻 Backbone
9	0.8475	70.22%	0.7628	72.73%	解冻 Backbone
10	0.8374	70.50%	0.7633	72.78%	解冻 Backbone
最终效果：10 轮训练后最佳验证准确率达 72.78%
模型保存：最佳模型路径 ./work_dirs/resnet18/best_model.pth

GPU 训练效果（RTX 5060）
训练速度：100 轮约 10-20 分钟（batch_size=128）
最终精度：100 轮训练后验证准确率可达 85%
显存占用：约 2-4GB（batch_size=128）

项目结构
plain

resnet18-cifar10-cpu/
├── .vscode/                  # VS Code 配置文件夹（自动生成）
├── configs/                  # 配置文件夹
│   └── train_config.yaml     # 超参数配置文件（核心）
├── data/                     # 数据集文件夹（自动下载，不纳入版本控制）
│   ├── cifar-10-batches-py/  # CIFAR-10 原始数据文件夹
│   └── cifar-10-python.tar.gz # CIFAR-10 压缩包
├── datasets/                 # 自定义数据集加载模块
│   ├── __pycache__/          # Python 缓存（自动生成，不纳入版本控制）
│   ├── __init__.py           # 模块初始化文件
│   └── cifar10.py            # CIFAR-10 数据加载逻辑
├── docs/                     # 文档/日志文件夹（纳入版本控制）
│   └── resnet18_cifar10_train_log.md # 训练日志（含核心指标与总结）
├── losses/                   # 自定义损失函数模块
│   ├── __pycache__/          # Python 缓存（自动生成，不纳入版本控制）
│   ├── __init__.py           # 模块初始化文件
│   └── cross_entropy.py      # 交叉熵损失函数实现
├── models/                   # 模型定义模块
│   ├── __pycache__/          # Python 缓存（自动生成，不纳入版本控制）
│   ├── __init__.py           # 模块初始化文件
│   └── resnet.py             # ResNet18/34 模型结构定义
├── utils/                    # 工具函数模块
│   ├── __pycache__/          # Python 缓存（自动生成，不纳入版本控制）
│   ├── __init__.py           # 模块初始化文件
│   ├── config.py             # 配置解析工具
│   └── logger.py             # 日志打印工具
├── work_dirs/                # 模型权重输出（不纳入版本控制）
├── ├──logs                   #训练日志
│   ├── resnet18/             # ResNet18 权重保存路径
│   └── resnet34/             # ResNet34 权重保存路径
├── .gitignore                # Git 忽略规则（排除数据集、权重、缓存等）
├── environment.yml           # Conda 环境配置文件（可选）
├── README.md                 # 项目说明文档
├── test.py                   # 测试入口（可选）
└── train.py                  # 训练入口（含数据加载、模型构建、训练逻辑）

常见问题
Q: 安装 PyTorch 后提示 "no kernel image is available"？
A: 这是 GPU 架构与 CUDA 版本不兼容导致的。RTX 50 系列需要 CUDA 12.8+，请执行：
bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
Q: 如何查看自己的 CUDA 版本？
A: 在命令行执行 nvidia-smi，查看右上角显示的 CUDA Version。

Q: 训练时显存不足怎么办？
A: 减小 train_config.yaml 中的 batch_size（如从 128 改为 64 或 32）。