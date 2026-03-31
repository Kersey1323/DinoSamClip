# DinoSamClip: DINOv2 + SAM + CLIP 集成推理系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DinoSamClip 集成 **DINOv2**（自监督视觉特征提取）、**SAM**（Segment Anything，通用分割）和 **CLIP**（零样本分类）三大模型，实现端到端的物体检测、分割与分类，并提供 REST API 供后端调用。

---

## 📁 项目结构

```
DinoSamClip/
├── core/                        # 算法层 — 纯模型代码，不依赖 Web 框架
│   ├── pipeline.py              # DINOv2 + SAM + CLIP 主管道
│   └── components/
│       ├── dinov2_extractor.py  # DINOv2 特征提取 & 注意力图
│       ├── sam_segmenter.py     # SAM 实例分割
│       └── clip_classifier.py  # CLIP 零样本分类
│
├── server/                      # Web 层 — FastAPI REST 服务
│   ├── main.py                  # 应用入口 + lifespan 单例加载
│   ├── dependencies.py          # Pipeline 单例依赖注入
│   ├── schemas.py               # Pydantic 响应模型
│   ├── utils.py                 # base64 / 可视化工具
│   └── routes/
│       ├── health.py            # GET  /health
│       ├── infer.py             # POST /infer  /infer/auto  /infer/attention
│       └── batch.py             # POST /batch/infer
│
├── training/                    # 训练层 — DINOv2 微调
│   ├── classifier/              # 分类器训练脚本
│   └── voc/                     # VOC 数据集处理
│
├── scripts/                     # 命令行工具
│   ├── demo.py                  # 单图演示脚本
│   └── batch_inference.py       # 批量推理 CLI
│
├── tests/                       # 单元测试
│
├── config/                      # 环境配置（INI 文件）
│   ├── local.ini                # 本地开发（本机 GPU 路径）
│   └── dev.ini                  # 生产/服务器（服务器路径）
│
├── weights/                     # 微调后的模型权重
├── settings.py                  # 全局配置加载器（读 INI，暴露 ModelConfig 等）
├── requirements.txt             # 算法依赖
├── requirements-server.txt      # Web 服务追加依赖
└── start_api.sh                 # 一键启动脚本
```

---

## ⚙️ 配置说明

配置通过 `config/` 目录下的 INI 文件管理，通过 `APP_ENV` 环境变量切换：

| 文件 | 适用场景 |
|------|---------|
| `config/local.ini` | 本地开发机（默认） |
| `config/dev.ini` | 生产 / 远程服务器 |

修改对应 INI 文件中的 `model_sources` 指定模型路径，无需改代码：

```ini
[models]
model_sources = /your/path/to/models
dinov2_model_name = %(model_sources)s/dinov2-large
sam_checkpoint_path = %(model_sources)s/sam_1/sam_vit_h_4b8939.pth
clip_model_name = %(model_sources)s/clip-vit-base-patch32

[pipeline]
device = cuda
confidence_threshold = 0.2

[server]
host = 0.0.0.0
port = 8000
workers = 2
```

---

## 📦 安装

```bash
# 1. 克隆仓库
git clone https://github.com/Kersey1323/DinoSamClip.git
cd DinoSamClip

# 2. 创建 conda 环境
conda create -n dsc python=3.10
conda activate dsc

# 3. 安装算法依赖
pip install -r requirements.txt

# 4. 安装 Web 服务依赖
pip install -r requirements-server.txt
```

---

## 🚀 启动 API 服务

```bash
# 本地开发（读 config/local.ini，自动热重载）
bash start_api.sh

# 生产服务器（读 config/dev.ini）
APP_ENV=dev bash start_api.sh

# 或直接启动
APP_ENV=local uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 `http://localhost:8000/docs` 查看完整 Swagger UI。

---

## 🔌 API 接口

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 服务 & 模型健康状态 |
| POST | `/infer` | 单图：DINOv2 → SAM → CLIP 完整推理 |
| POST | `/infer/auto` | 单图：SAM 自动网格 → CLIP（跳过DINOv2） |
| POST | `/infer/attention` | 单图：仅返回 DINOv2 注意力热力图 |
| POST | `/batch/infer` | 批量目录推理 |

### 图片输入方式（单图接口）

所有单图接口支持两种互斥的输入方式：

**方式一：上传图片文件**
```bash
curl -X POST http://localhost:8000/infer \
  -F "image_file=@/path/to/photo.jpg" \
  -F "candidate_classes=person,car,dog" \
  -F "confidence_threshold=0.2" \
  -F "return_visualization=false"
```

**方式二：传服务器端路径**
```bash
curl -X POST http://localhost:8000/infer \
  -F "image_path=/data/images/photo.jpg" \
  -F "candidate_classes=person,car"
```

### 批量推理
```bash
curl -X POST http://localhost:8000/batch/infer \
  -H "Content-Type: application/json" \
  -d '{
    "image_dir": "/data/images/",
    "output_dir": "/data/results/",
    "candidate_classes": ["person", "car", "dog"],
    "auto_mode": false
  }'
```

### 响应示例
```json
{
  "num_objects": 2,
  "detections": [
    { "class": "person", "confidence": 0.87, "bbox": [10, 20, 150, 300] },
    { "class": "car",    "confidence": 0.73, "bbox": [200, 50, 400, 280] }
  ],
  "visualization_base64": null,
  "elapsed_ms": 1240.5
}
```

---

## 🖥️ 命令行脚本

### 单图演示
```bash
python scripts/demo.py \
  --image path/to/image.jpg \
  --classes person car dog \
  --device cuda
```

### 批量推理（CLI）
```bash
python scripts/batch_inference.py \
  --input_dir path/to/images \
  --output_dir path/to/results \
  --classes person car dog
```

---

## 🔧 Python API（直接调用）

```python
from core.pipeline import DinoSAMClipPipeline
from PIL import Image

pipeline = DinoSAMClipPipeline(device="cuda", candidate_classes=["person", "car"])
image = Image.open("photo.jpg").convert("RGB")
results = pipeline.detect_and_classify(image)

print(f"检测到 {results['num_objects']} 个物体")
for det in results["detections"]:
    print(f"  {det['class']}: {det['confidence']:.1%}  bbox={det['bbox']}")
```

---

## 🏋️ 模型微调

```bash
# 分类器训练
python training/classifier/train_classifier.py

# VOC 数据集微调
python training/voc/data_clean.py
python training/voc/train_voc.py
```

---

## 🙏 致谢

- [DINOv2](https://github.com/facebookresearch/dinov2) — Facebook Research
- [Segment Anything](https://github.com/facebookresearch/segment-anything) — Meta AI
- [CLIP](https://github.com/openai/CLIP) — OpenAI
- [Transformers](https://github.com/huggingface/transformers) — Hugging Face
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 📄 许可证

MIT License — 查看 [LICENSE](LICENSE) 文件了解详情。