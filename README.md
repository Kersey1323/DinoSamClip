# DinoSamClip: DINOv2 + SAM + CLIP 集成管道

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DinoSamClip 是一个集成了三种最先进计算机视觉模型的强大管道：**DINOv2**（自监督视觉特征提取）、**SAM**（Segment Anything Model，通用分割）和 **CLIP**（对比语言-图像预训练，零样本分类）。该项目实现了端到端的物体检测、分割和分类，无需针对特定类别的训练数据。

## ✨ 主要功能

- **智能物体检测**：使用 DINOv2 的注意力机制自动发现图像中的潜在物体区域
- **精确分割**：利用 SAM 对检测到的物体进行像素级精确分割
- **零样本分类**：使用 CLIP 对分割后的物体进行零样本分类，无需训练
- **完整可视化**：提供中间结果和最终结果的详细可视化
- **批量处理**：支持批量图像处理
- **模型微调**：包含 DINOv2 在特定数据集上的微调功能

## 📦 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/DinoSamClip.git
cd DinoSamClip
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载预训练模型
项目需要以下预训练模型：

| 模型 | 下载链接 | 保存路径 |
|------|----------|----------|
| DINOv2-large | [HuggingFace](https://huggingface.co/facebook/dinov2-large) | `models/dinov2-large/` |
| SAM (vit_h) | [官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints) | `models/sam_vit_h_4b8939.pth` |
| CLIP (vit-base-patch32) | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) | `models/clip-vit-base-patch32/` |

**注意**：请将模型下载后放置在 `MODEL_SOURCES` 路径下（默认为 `/media/dell/新加卷1/LLM/models/CV-models/`），或在 [src/config.py](src/config.py) 中修改路径配置。

## 🚀 快速开始

### 1. 单张图像处理
```bash
python src/demo.py --image path/to/your/image.jpg --classes person car dog cat --device cuda
```

参数说明：
- `--image`: 输入图像路径
- `--classes`: 候选类别列表（英文）
- `--device`: 运行设备 (`cuda` 或 `cpu`)
- `--num_prompts`: DINOv2 生成的点提示数量（默认：10）
- `--conf_thresh`: CLIP 置信度阈值（默认：0.15）
- `--auto`: 使用纯 SAM 自动网格模式（跳过 DINOv2）

### 2. 批量处理
```bash
python src/batch_inference.py --input_dir path/to/images --output_dir path/to/results --classes person car dog cat
```

### 3. 使用 Python API
```python
from src.pipeline import DinoSAMClipPipeline
from PIL import Image

# 初始化管道
pipeline = DinoSAMClipPipeline(device="cuda", candidate_classes=["person", "car", "dog"])

# 加载图像
image = Image.open("path/to/image.jpg").convert("RGB")

# 运行检测和分类
results = pipeline.detect_and_classify(image)

# 查看结果
print(f"检测到 {results['num_objects']} 个物体")
for i, det in enumerate(results["detections"]):
    print(f"  物体 {i+1}: {det['class']} ({det['confidence']:.2%})")
```

## 📁 项目结构

```
DinoSamClip/
├── src/                          # 主源代码
│   ├── components/               # 核心组件
│   │   ├── dinov2_extractor.py   # DINOv2 特征提取器
│   │   ├── sam_segmenter.py      # SAM 分割器
│   │   └── clip_classifier.py    # CLIP 分类器
│   ├── pipeline.py               # 主集成管道
│   ├── demo.py                   # 演示脚本
│   ├── batch_inference.py        # 批量推理脚本
│   ├── config.py                 # 配置文件
│   └── requirements.txt          # 依赖列表
├── dinov2_finetuning/            # DINOv2 微调相关
│   ├── calssifer/                # 分类器训练
│   ├── data/                     # 数据处理脚本
│   └── voc/                      # VOC 数据集处理
├── test/                         # 测试代码
├── requirements.txt              # 项目依赖
├── README.md                     # 本文件
└── .gitignore                    # Git 忽略文件
```

## ⚙️ 配置说明

主要配置在 [src/config.py](src/config.py) 中：

```python
# 模型路径配置
MODEL_SOURCES = "/media/dell/新加卷1/LLM/models/CV-models"

# DINOv2 配置
DINOV2_MODEL_NAME = f"{MODEL_SOURCES}/dinov2-large"
DINOV2_DEVICE = "cuda"

# SAM 配置
SAM_MODEL_TYPE = "vit_h"  # vit_b, vit_l, vit_h
SAM_CHECKPOINT_PATH = f"{MODEL_SOURCES}/sam_1/sam_vit_h_4b8939.pth"
SAM_DEVICE = "cuda"

# CLIP 配置
CLIP_MODEL_NAME = f"{MODEL_SOURCES}/clip-vit-base-patch32"
CLIP_DEVICE = "cuda"

# 管道参数
ATTENTION_THRESHOLD = 0.15    # DINOv2 注意力阈值
MIN_MASK_AREA = 500           # 最小掩码面积（像素）
NUM_PROMPTS = 10              # 提示点数量
CONFIDENCE_THRESHOLD = 0.2    # CLIP 置信度阈值
```

## 🎯 使用示例

### 示例 1：检测和分类常见物体
```bash
python src/demo.py --image examples/street.jpg --classes person car bicycle traffic\ light bus --device cuda
```

### 示例 2：使用自动模式（跳过 DINOv2）
```bash
python src/demo.py --image examples/indoor.jpg --classes chair table sofa tv --device cuda --auto
```

### 示例 3：自定义参数
```bash
python src/demo.py --image examples/park.jpg --classes person dog cat bird --device cuda --num_prompts 20 --conf_thresh 0.1
```

## 🔧 高级功能

### 1. DINOv2 微调
项目包含 DINOv2 在自定义数据集上的微调功能：

```bash
# 数据准备
python dinov2_finetuning/data/split_dataset.py

# 训练分类器
python dinov2_finetuning/calssifer/train_classifier.py

# VOC 数据集处理
python dinov2_finetuning/voc/data_clean.py
python dinov2_finetuning/voc/train_voc.py
```

### 2. 可视化中间结果
管道会生成以下可视化结果：
- `debug_pipeline_sam.png`: SAM 原始掩码可视化
- `*_intermediate.png`: DINOv2 提示点和 SAM 原始掩码
- `*_final.png`: 最终分割叠加结果、热力图和文本摘要

### 3. 自定义候选类别
可以通过修改 `DEFAULT_CANDIDATE_CLASSES` 或通过命令行参数指定任何类别的英文名称。

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出新功能建议！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

本项目基于以下开源项目构建：

- [DINOv2](https://github.com/facebookresearch/dinov2) - Facebook Research
- [Segment Anything](https://github.com/facebookresearch/segment-anything) - Meta AI
- [CLIP](https://github.com/openai/CLIP) - OpenAI
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [GitHub Issue](https://github.com/yourusername/DinoSamClip/issues)
- 或发送邮件至 your.email@example.com

---

**提示**：确保有足够的 GPU 内存运行这三个大型模型（DINOv2-large、SAM-vit-h、CLIP）。如果内存不足，可以考虑使用较小的模型变体或在 CPU 上运行。