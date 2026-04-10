# DinoSamClip: DINOv2 + SAM + CLIP 集成推理系统 + 向量检索

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DinoSamClip 集成 **DINOv2**（自监督视觉特征提取）、**SAM**（Segment Anything，通用分割）、**CLIP**（零样本分类）和 **SigLIP**（图文检索）四大模型，实现端到端的物体检测、分割、分类与向量检索，并提供 REST API 供后端调用。

## ✨ 新增功能

- 🔍 **向量检索系统**：基于 SigLIP + DINOv2 的混合检索，支持图搜图和文搜图
- 🎯 **SAM3 预处理**：入库前自动分割背景，提升检索精度
- 📊 **Milvus 向量数据库**：高性能向量存储和检索
- 🔄 **两阶段检索**：粗排（全局特征）+ 精排（Patch Token 匹配）

---

## 📁 项目结构

```
DinoSamClip/
├── core/                        # 算法层 — 纯模型代码，不依赖 Web 框架
│   ├── pipeline.py              # DINOv2 + SAM + CLIP 主管道
│   └── components/
│       ├── dinov2_extractor.py  # DINOv2 特征提取 & 注意力图
│       ├── sam_segmenter.py     # SAM 实例分割
│       └── clip_classifier.py   # CLIP 零样本分类
│
├── vector_db/                   # 向量数据库系统 — 图像检索
│   ├── models/                  # 特征提取模型
│   │   ├── siglip_extractor.py  # SigLIP 图文特征提取
│   │   ├── dinov2_extractor.py  # DINOv2 向量提取
│   │   └── model_config.py      # 模型配置
│   ├── preprocessing/           # 数据预处理
│   │   └── sam3_preprocessor.py # SAM3 背景分割预处理
│   ├── storage/                 # 向量存储
│   │   ├── milvus_manager.py    # Milvus 统一管理器
│   │   └── collection_manager.py # Collection 管理
│   ├── indexing/                # 入库模块
│   │   ├── indexer.py           # 向量入库器
│   │   └── batch_indexer.py     # 批量入库
│   ├── data/                    # 数据访问
│   │   ├── db_connector.py      # PostgreSQL 连接
│   │   └── image_loader.py      # MinIO/本地图像加载
│   ├── scripts/                 # 脚本工具
│   │   ├── build_index.py       # 批量建库脚本
│   │   └── init_collections.py  # 初始化 Collection
│   ├── tests/                   # 检索测试
│   │   ├── test_siglip_search.py # SigLIP 检索测试
│   │   └── test_dinov2_search.py # DINOv2 检索测试
│   ├── config/                  # 配置文件
│   │   ├── vector_db.ini        # 向量数据库配置
│   │   └── db_config.ini        # 数据源配置
│   └── README_SAM3.md           # SAM3 预处理文档
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

## 🔍 向量检索系统

### 快速开始

#### 1. 初始化 Milvus Collection

```bash
python vector_db/scripts/init_collections.py
```

#### 2. 批量入库（支持 SAM3 预处理）

```bash
python vector_db/scripts/build_index.py \
  --config vector_db/config/vector_db.ini \
  --db-config vector_db/config/db_config.ini \
  --limit 1000 \
  --resume  # 支持断点续传
```

#### 3. 图像检索

```bash
# 图搜图
python vector_db/tests/test_siglip_search.py \
  --mode image \
  --query path/to/query.jpg \
  --top-k 10

# 文搜图（基于 description）
python vector_db/tests/test_siglip_search.py \
  --mode text \
  --query "左支座" \
  --top-k 20

# 文搜图（基于图像内容语义）
python vector_db/tests/test_siglip_search.py \
  --mode text \
  --query "左支座" \
  --top-k 20 \
  --use-image-vector
```

### SAM3 预处理

SAM3 预处理可在入库前自动分割物体并去除背景，提升检索精度。

**配置** (`vector_db/config/vector_db.ini`):
```ini
[sam3]
model_path = /path/to/sam3-video-base
mask_dilate = 20
bg_color = white
device = cuda
save_debug_images = true
debug_output_dir = vector_db/logs/sam3_debug
```

**特性**:
- ✅ 文本提示词引导分割（支持中文自动翻译）
- ✅ Mask 膨胀、闭运算等形态学处理
- ✅ 自动裁剪到边界框
- ✅ 可配置背景颜色（white/black/gray）
- ✅ 调试模式：保存原图/mask/结果

**使用示例**:
```python
from vector_db.storage.milvus_manager import MilvusManager
from PIL import Image

# 启用 SAM3 预处理
manager = MilvusManager(enable_sam3=True)

# 入库（自动应用 SAM3 分割）
image = Image.open('image.jpg')
manager.index_image(
    image=image,
    item_id=12345,
    item_name="左支座",
    item_code="ABC123",
    image_id=67890,
    image_url="path/to/image.jpg"
)

# 检索（查询图像也会应用 SAM3 分割）
results = manager.search_by_image(image, top_k=10, mode="hybrid")
```

详细文档：[vector_db/README_SAM3.md](vector_db/README_SAM3.md)

### 检索模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `siglip` | 仅使用 SigLIP 图像向量 | 快速检索，适合大规模数据 |
| `dinov2` | DINOv2 两阶段检索（全局+Patch） | 高精度，适合细粒度匹配 |
| `hybrid` | SigLIP + DINOv2 混合检索（RRF融合） | 平衡速度和精度（推荐） |

### Python API

```python
from vector_db.storage.milvus_manager import MilvusManager
from PIL import Image

# 初始化管理器
manager = MilvusManager(
    config_path='vector_db/config/vector_db.ini',
    db_config_path='vector_db/config/db_config.ini',
    enable_sam3=True  # 启用 SAM3 预处理
)

# 文本检索
results = manager.search_by_text("左支座", top_k=10)

# 图像检索
query_image = Image.open("query.jpg")
results = manager.search_by_image(
    image=query_image,
    top_k=10,
    mode="hybrid",      # siglip/dinov2/hybrid
    coarse_top_k=30,    # 粗排候选数
    alpha=0.6           # 加权系数
)

# 处理结果
for result in results:
    print(f"Item: {result['entity']['item_name']}")
    print(f"Score: {result['distance']:.4f}")
    print(f"Image: {result['entity']['image_url']}")
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
- [SAM3](https://github.com/facebookresearch/sam3) — Meta AI
- [CLIP](https://github.com/openai/CLIP) — OpenAI
- [SigLIP](https://github.com/google-research/big_vision) — Google Research
- [Transformers](https://github.com/huggingface/transformers) — Hugging Face
- [Milvus](https://milvus.io/) — Vector Database
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 📚 相关文档

- [SAM3 预处理使用指南](vector_db/README_SAM3.md)
- [向量数据库设计文档](docs/superpowers/specs/2026-04-09-vector-database-indexing-design.md)
- [向量数据库实现计划](docs/superpowers/specs/2026-04-09-vector-database-indexing-plan.md)

---

## 📄 许可证

MIT License — 查看 [LICENSE](LICENSE) 文件了解详情。