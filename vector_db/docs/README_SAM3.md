# SAM3 预处理集成说明

## 概述

SAM3 (Segment Anything Model 3) 预处理器已集成到向量数据库入库和检索流程中，用于在入库前对图像进行背景分割和去除，提高检索精度。

## 功能特性

1. **自动背景分割**：使用 SAM3 模型自动分割物体并去除背景
2. **文本提示支持**：支持中文/英文文本提示词，自动翻译中文为英文
3. **形态学处理**：支持 mask 膨胀、闭运算等后处理
4. **调试图像保存**：可选保存原图、mask、分割结果用于调试
5. **入库和检索集成**：入库和检索时都可以使用 SAM3 预处理

## 配置

在 `vector_db/config/vector_db.ini` 中添加 SAM3 配置：

```ini
[sam3]
model_path = /path/to/sam3-video-base
mask_dilate = 20
bg_color = white
device = cuda
save_debug_images = true
debug_output_dir = vector_db/logs/sam3_debug
```

### 配置参数说明

- `model_path`: SAM3 模型路径
- `mask_dilate`: Mask 膨胀像素数（推荐 20-40）
- `bg_color`: 背景颜色（white/black/gray）
- `device`: 设备（cuda/cpu）
- `save_debug_images`: 是否保存调试图像
- `debug_output_dir`: 调试图像输出目录

## 使用方法

### 1. 批量入库（使用 SAM3）

```bash
python vector_db/scripts/build_index.py \
    --config vector_db/config/vector_db.ini \
    --db-config vector_db/config/db_config.ini \
    --limit 100
```

SAM3 预处理会自动应用（如果配置文件中启用）。

### 2. 使用 MilvusManager（编程方式）

```python
from vector_db.storage.milvus_manager import MilvusManager
from PIL import Image

# 初始化管理器（启用 SAM3）
manager = MilvusManager(
    config_path='vector_db/config/vector_db.ini',
    db_config_path='vector_db/config/db_config.ini',
    enable_sam3=True  # 启用 SAM3 预处理
)

# 入库单张图像（会自动应用 SAM3 预处理）
image = Image.open('path/to/image.jpg')
success = manager.index_image(
    image=image,
    item_id=12345,
    item_name="左支座",
    item_code="ABC123",
    image_id=67890,
    image_url="path/to/image.jpg",
    description="左支座零件"
)

# 图像检索（查询图像也会应用 SAM3 预处理）
query_image = Image.open('path/to/query.jpg')
results = manager.search_by_image(
    image=query_image,
    top_k=10,
    mode="hybrid"  # siglip/dinov2/hybrid
)
```

### 3. 文本搜索（两种模式）

```bash
# 模式1：基于 description 文本匹配（默认）
python vector_db/tests/test_siglip_search.py \
    --mode text \
    --query "左支座" \
    --top-k 20

# 模式2：基于图像内容的语义相似度（推荐用于 SAM3 分割后的数据）
python vector_db/tests/test_siglip_search.py \
    --mode text \
    --query "左支座" \
    --top-k 20 \
    --use-image-vector
```

### 4. 图像搜索

```bash
# 使用原始图像搜索
python vector_db/tests/test_siglip_search.py \
    --mode image \
    --query path/to/query.jpg \
    --top-k 10

# 使用分割后的图像搜索（更精确）
python vector_db/tests/test_siglip_search.py \
    --mode image \
    --query vector_db/logs/sam3_debug/IMAGE_ID_TIMESTAMP/3_result.jpg \
    --top-k 10
```

## 工作流程

### 入库流程

```
原始图像 
  ↓
SAM3 分割（提取文本提示词 → 分割 → 去背景 → 裁剪）
  ↓
分割后图像
  ↓
特征提取（SigLIP + DINOv2）
  ↓
存入 Milvus
```

### 检索流程

```
查询图像
  ↓
SAM3 分割（可选）
  ↓
特征提取
  ↓
向量检索
  ↓
返回结果
```

## 调试图像

当 `save_debug_images=true` 时，每张处理的图像会保存到：

```
vector_db/logs/sam3_debug/{image_id}_{timestamp}/
├── 1_original.jpg    # 原始图像
├── 2_mask.jpg        # 分割 mask
├── 3_result.jpg      # 分割结果（去背景+裁剪）
└── info.txt          # 处理信息
```

## 性能优化建议

1. **Mask 膨胀参数**：
   - 小物体：10-20 像素
   - 中等物体：20-40 像素
   - 大物体：40-60 像素

2. **背景颜色选择**：
   - 白色背景：适合深色物体
   - 黑色背景：适合浅色物体
   - 灰色背景：通用选择

3. **设备选择**：
   - GPU (cuda)：推荐，速度快
   - CPU：备选，速度较慢

4. **批量处理**：
   - 使用 `build_index.py` 进行批量入库
   - 支持断点续传（`--resume` 参数）

## 常见问题

### Q1: SAM3 预处理失败怎么办？

A: 系统会自动回退到使用原始图像，不会中断入库流程。检查日志查看失败原因。

### Q2: 如何禁用 SAM3 预处理？

A: 
- 方法1：从配置文件中删除 `[sam3]` 部分
- 方法2：使用 `MilvusManager(enable_sam3=False)`

### Q3: 文本搜索为什么找不到分割后的数据？

A: 默认文本搜索使用 `text_vector`（基于 description），与图像内容无关。使用 `--use-image-vector` 参数可以基于图像内容进行语义搜索。

### Q4: 如何验证 SAM3 是否生效？

A: 
1. 检查日志：`vector_db/logs/indexing_YYYYMMDD.log`
2. 查看调试图像：`vector_db/logs/sam3_debug/`
3. 使用分割后的图像进行搜索，对比结果

## 相关文件

- 预处理器：`vector_db/preprocessing/sam3_preprocessor.py`
- 入库器：`vector_db/indexing/indexer.py`
- 管理器：`vector_db/storage/milvus_manager.py`
- 搜索测试：`vector_db/tests/test_siglip_search.py`
- 配置文件：`vector_db/config/vector_db.ini`
