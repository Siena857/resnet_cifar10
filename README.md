# ResNet18/34 图像分类（CIFAR-10）
基于 PyTorch 实现的 ResNet 系列模型（ResNet18/ResNet34），针对 CIFAR-10 数据集完成图像分类任务，采用 YAML 配置文件统一管理超参数，支持 CPU/GPU 双环境训练。

## 项目简介
- 模型：支持 ResNet18/ResNet34 切换，适配 CIFAR-10 数据集 32×32 输入尺寸与 10 分类任务
- 超参数管理：使用 YAML 配置文件（`configs/train_config.yaml`）统一管理，参数修改无需改动代码
- 训练策略：支持 Backbone 冻结训练（前5轮冻结，后5轮解冻），兼顾训练效率与效果
- 数据集：CIFAR-10（torchvision 自动下载，无需手动预处理）
- 训练环境：兼容 CPU/GPU 训练，本项目因时间/设备限制采用 CPU 训练 10 轮，仅作代码可行性验证
- 日志输出：训练过程自动记录，完整训练日志见 `docs/resnet18_cifar10_train_log.md`

## 环境准备
### 1. 克隆项目
```bash
git clone https://github.com/Siena857/resnet18-cifar10-cpu.git
cd resnet18-cifar10-cpu
### 2. 创建并激活虚拟环境
Windows 系统（PowerShell/CMD）
powershell
#### 创建虚拟环境（Python 3.11 为例）
conda create -n resnet_cifar10 python=3.11 -y

#### 激活虚拟环境（PowerShell）
conda activate resnet_cifar10
若报错“执行策略禁止”，先执行：Set-ExecutionPolicy RemoteSigned -Scope CurrentUser（按提示输入 Y 确认）

#### 激活虚拟环境（CMD）
conda activate resnet_cifar10
macOS/Linux 系统
bash
运行
# 创建虚拟环境
conda create -n resnet_cifar10 python=3.11 -y
# 激活虚拟环境
conda activate resnet_cifar10
### 3. 安装依赖包
bash
运行
#### 基础依赖安装（兼容 CPU/GPU）
pip install torch==2.3.0 torchvision==0.18.0 tqdm pyyaml numpy==1.24.4
#### 若下载速度慢，添加清华镜像源：
pip install torch==2.3.0 torchvision==0.18.0 tqdm pyyaml numpy==1.24.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
## 快速开始
### 1. 配置超参数
编辑 configs/train_config.yaml 文件调整核心参数，关键配置示例：
yaml
#### 模型配置（切换 ResNet18/34）
model:
  name: "resnet18"        # 可选：resnet18 / resnet34
  num_classes: 10         # CIFAR-10 固定为10类
  freeze_backbone: True   # 启用 Backbone 冻结训练

#### 训练配置（切换 CPU/GPU）
train:
  epochs: 10              # 训练轮数（本项目仅验证，实际可增至100+轮）
  batch_size: 16          # 批次大小（GPU可设64/128，CPU建议16/32）
  device: "cpu"           # 训练设备：cpu / cuda
  freeze_backbone_epochs: 5 # 冻结 Backbone 训练轮数

#### 路径配置
path:
  data_path: "./data"                  # 数据集保存路径
  output_path: "./work_dirs/resnet18"  # 模型权重输出路径
###2. 训练模型
直接运行训练脚本，自动加载配置文件完成训练：
bash
运行
#### 启动训练（默认 ResNet18 + CPU）
python train.py

#### 切换 ResNet34 训练：修改 configs/train_config.yaml 中 model.name 为 "resnet34" 后重新运行
#### GPU 训练：修改 configs/train_config.yaml 中 train.device 为 "cuda" 后重新运行
训练过程：终端实时打印每轮 Loss/Acc 指标，进度条可视化训练进度
模型保存：每 10 轮保存一次轮次模型，自动保存验证准确率最高的「最佳模型」
日志输出：完整训练日志自动输出至终端，已整理至 docs/resnet18_cifar10_train_log.md
3. 训练结果说明
本项目因时间 / 设备限制仅执行 10 轮 CPU 训练，核心指标如下（完整日志见终端输出）：
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
最终效果：10 轮训练后最佳验证准确率达 72.78%，符合代码可行性验证预期
训练稳定性：全程无梯度 / 设备报错，CPU 环境下稳定完成训练
模型保存：最佳模型路径 ./work_dirs/resnet18/best_model.pth，第 10 轮模型路径 ./work_dirs/resnet18/epoch_10_acc_0.7278.pth

###项目结构
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
│   └── resnet18/             # ResNet18 权重保存路径
├── .gitignore                # Git 忽略规则（排除数据集、权重、缓存等）
├── environment.yml           # Conda 环境配置文件（可选）
├── README.md                 # 项目说明文档
├── test.py                   # 测试入口（可选）
└── train.py                  # 训练入口（含数据加载、模型构建、训练逻辑）