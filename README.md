# VMamba: 基于Mamba架构的多源遥感图像分类模型

## 项目简介

VMamba是一个基于Mamba架构设计的多源遥感图像分类模型,专门用于处理高光谱图像(HSI)和激光雷达(LiDAR)数据的融合分类任务。该模型采用创新的选择性扫描机制和多尺度特征提取策略,在多个基准数据集上取得了优异的分类性能。

### 主要特点

- 基于Mamba架构的高效序列建模
- 多源数据融合的交叉注意力机制
- 自适应特征融合策略
- 支持多个公开数据集
- 提供完整的训练和评估流程

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (用于GPU加速)

### 依赖包
```bash
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
pandas>=1.3.0
pyyaml>=5.4.0
einops>=0.4.0
timm>=0.6.0
fvcore>=0.1.5
```

## 安装说明

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/vmamba.git
cd vmamba
```

2. 创建并激活虚拟环境:
```bash
conda create -n vmamba python=3.8
conda activate vmamba
```

3. 安装依赖:
```bash
pip install -r requirements.txt
```

## 数据集准备

支持的数据集:
- Houston
- MUUFL
- Trento
- Augsburg
- Houston2018

数据集应按以下结构组织:
```
data/
├── Houston/
│   ├── HSI.mat
│   └── LiDAR.mat
├── MUUFL/
│   ├── HSI.mat
│   └── LiDAR.mat
...
```

## 使用说明

### 训练模型

```bash
python main.py \
    --dataset Houston \
    --window_size 13 \
    --batch_size 64 \
    --epochs 60 \
    --lr 0.0005 \
    --device cuda:0
```

主要参数说明:
- `--dataset`: 选择数据集 ['Houston', 'MUUFL', 'Trento', 'Augsburg', 'Houston2018']
- `--window_size`: 空间窗口大小
- `--batch_size`: 训练批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--device`: 训练设备

### 生成分类图

```bash
python generate_classification_map.py \
    --dataset Houston \
    --window_size 13 \
    --device cuda:0
```

## 实验结果

在多个数据集上的分类精度:

| 数据集 | OA (%) | AA (%) | Kappa (%) |
|--------|---------|---------|-----------|
| Houston | 98.10 | 98.06 | 97.88 |
| Trento | 99.27 | 97.95 | 98.86 |
| MUUFL | 97.85 | 97.32 | 97.15 |

## 项目结构

```
vmamba/
├── main.py              # 主训练脚本
├── dataset.py           # 数据加载和预处理
├── vmamba.py            # 模型架构定义
├── module.py            # 辅助模块
├── visualization.py     # 可视化工具
├── model_stats.py       # 模型统计
├── mamba/              # Mamba核心实现
│   ├── __init__.py
│   └── mambablock.py
├── models/             # 保存的模型
└── results/            # 实验结果
```

## 引用

如果您在研究中使用了本项目,请引用以下论文:

```bibtex
@article{zhang2024vmamba,
  title={VMamba: Visual Mamba for Multi-modal Remote Sensing Image Classification},
  author={Zhang, Yiyan and Others},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题,请通过以下方式联系:

- 作者: 张亦严
- 邮箱: your.email@example.com
- GitHub Issues: [提交问题](https://github.com/yourusername/vmamba/issues)
