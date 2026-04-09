# SigLIP 检索测试

测试 SigLIP 的图搜图和文搜图功能。

## 使用方法

### 1. 文搜图测试

```bash
python vector_db/tests/test_siglip_search.py \
    --mode text \
    --query "管卡" \
    --top-k 5 \
    --output-dir vector_db/tests/test_images/text_search_results
```

**常用查询示例：**
- "管卡" - 搜索管卡类零件
- "螺栓" - 搜索螺栓类零件
- "垫片" - 搜索垫片类零件
- "半单管卡DN25" - 精确搜索特定型号

### 2. 图搜图测试

首先需要准备一张查询图像，然后：

```bash
python vector_db/tests/test_siglip_search.py \
    --mode image \
    --query /path/to/query_image.jpg \
    --top-k 5 \
    --output-dir vector_db/tests/test_images/image_search_results
```

**获取查询图像的方法：**
1. 从数据库中随机选一张图像作为查询
2. 使用 MinIO 或本地文件系统中的图像
3. 拍摄实物照片进行测试

### 3. 从数据库随机选择图像作为查询

```bash
# 先运行一次文搜图，下载一些结果图像
python vector_db/tests/test_siglip_search.py \
    --mode text \
    --query "管卡" \
    --top-k 3

# 然后使用下载的图像进行图搜图测试
python vector_db/tests/test_siglip_search.py \
    --mode image \
    --query vector_db/tests/test_images/rank1_score0.9876_半单管卡DN25.jpg \
    --top-k 5
```

## 输出说明

脚本会：
1. 打印检索结果到终端（包括相似度分数、零件信息）
2. 下载结果图像到指定目录
3. 图像文件名格式：`rank{排名}_score{相似度}_{零件名称}.jpg`

## 评估检索效果

检查下载的图像，验证：
- **文搜图**：返回的图像是否与查询文本语义相关
- **图搜图**：返回的图像是否与查询图像视觉相似
- **排序质量**：相似度分数高的结果是否确实更相关

## 示例输出

```
Search Results:
================================================================================

Rank 1:
  Score: 0.8234
  Item ID: 1555467253485408256
  Item Name: 半单管卡DN25
  Item Code: BDG-DN25
  Image ID: 1983442698417344512
  Image URL: bfr-ai-files/small-1983442698220081152.png

Rank 2:
  Score: 0.7891
  ...
```

## 注意事项

- 确保 Milvus 服务正在运行
- 确保已经运行过 `build_index.py` 建立索引
- MinIO 凭证需要正确配置在 `vector_db/config/db_config.ini`
- 相似度分数范围：COSINE 距离，越接近 1 表示越相似
