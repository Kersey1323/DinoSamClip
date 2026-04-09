---
name: 向量库入库模块设计
description: 基于 Grounding DINO + SigLIP 2 + DINOv2 的向量库入库系统设计方案
type: design
date: 2026-04-09
---

# 向量库入库模块设计方案

## 1. 项目背景

### 1.1 业务场景

动车组维修现场的图像采集环境高度复杂，存在以下痛点：
- **大背景小目标**：目标零件占全图比例极小
- **视角畸变严重**：拍摄空间受限导致畸变
- **物理干扰**：零件表面覆盖油污和高光反射

### 1.2 技术方案

采用"定位-分割-多模态特征表征-细粒度特征提取"多级流水线：

1. **目标定位 (Grounding DINO)**：文本引导的零样本目标检测（仅用于检索阶段）
2. **多模态特征表征 (SigLIP 2)**：图文混合检索，支持中文描述
3. **细粒度特征提取 (DINOv2)**：自监督对比学习，关注机械结构特征

### 1.3 本次实现范围

**第一阶段：入库模块**（本次实现）
- 从 recognition 项目的 PostgreSQL 读取零件信息
- 图像从 MinIO/本地文件系统加载
- 提取 SigLIP 2（1152维）和 DINOv2（1024维）特征
- 存储到 Milvus 向量库

**第二阶段：检索模块**（后续实现）
- Grounding DINO 目标定位
- 两阶段检索（RRF粗排 + Patch精排）

---

## 2. 整体架构

### 2.1 目录结构

```
DinoSamClip/
├── vector_db/                          # 向量库系统根目录
│   ├── __init__.py
│   ├── models/                         # 模型封装层
│   │   ├── __init__.py
│   │   ├── siglip_extractor.py        # SigLIP 2 特征提取器
│   │   ├── dinov2_extractor.py        # DINOv2 特征提取器
│   │   └── model_config.py            # 模型配置
│   ├── storage/                        # 存储层
│   │   ├── __init__.py
│   │   ├── milvus_client.py           # Milvus 连接管理
│   │   └── collection_manager.py      # Collection 初始化和管理
│   ├── indexing/                       # 入库业务逻辑层
│   │   ├── __init__.py
│   │   ├── indexer.py                 # 单条入库逻辑
│   │   └── batch_indexer.py           # 批量入库协调器
│   ├── data/                           # 数据访问层
│   │   ├── __init__.py
│   │   ├── db_connector.py            # PostgreSQL 连接
│   │   └── image_loader.py            # 图像加载（MinIO/本地）
│   ├── config/                         # 配置文件
│   │   ├── vector_db.ini              # 向量库配置
│   │   └── db_config.ini              # 数据库配置
│   ├── scripts/                        # 命令行工具
│   │   ├── init_collections.py        # 初始化 Milvus Collections
│   │   └── build_index.py             # 批量入库脚本
│   └── utils/                          # 工具函数
│       ├── __init__.py
│       └── logger.py                  # 日志配置
```

### 2.2 数据流设计

```
[PostgreSQL] → 读取零件信息 → [图像加载] → [特征提取] → [Milvus存储]
     ↓                            ↓              ↓              ↓
  Items表                    MinIO/本地      SigLIP2      Collection1
  SgoItemImages表                           DINOv2       Collection2
```

### 2.3 核心流程

1. **数据读取**：从 PostgreSQL 批量读取零件信息（item_id, item_name, item_code, description）
2. **图像加载**：根据 image_url 从 MinIO 或本地加载图像
3. **特征提取**：
   - SigLIP 2：提取图像特征（1152维）+ 文本特征（基于 description）
   - DINOv2：提取全局特征（1024维）+ Patch Tokens（1369×1024）
4. **向量存储**：
   - Collection 1：SigLIP 2 图像向量 + 文本向量（共享空间）
   - Collection 2：DINOv2 全局向量 + Patch Tokens

---

## 3. Milvus Collection Schema 设计

### 3.1 Collection 1: SigLIP 图文共享空间

**名称**: `PARTS_SIGLIP_SO400M_1152D_V1`

**用途**: 存储 SigLIP 2 提取的图像特征和文本特征，支持图文混合检索

**Schema**:
```python
{
    "id": Int64 (Primary Key, Auto ID),
    "item_id": Int64 (零件ID, 索引),
    "item_name": VarChar(200) (零件名称),
    "item_code": VarChar(100) (零件编码, 索引),
    "image_id": Int64 (图像ID),
    "image_url": VarChar(500) (图像URL),
    "description": VarChar(1000) (文本描述),
    
    # 向量字段
    "image_vector": FloatVector(1152) (图像特征向量),
    "text_vector": FloatVector(1152) (文本特征向量),
    
    # 元数据
    "created_at": VarChar(50) (入库时间),
}
```

**索引配置**:
- `image_vector`: IVF_FLAT, metric_type="COSINE", nlist=1024
- `text_vector`: IVF_FLAT, metric_type="COSINE", nlist=1024
- `item_id`, `item_code`: 标量索引

### 3.2 Collection 2: DINOv2 高精度视觉特征

**名称**: `PARTS_DINOV2_LARGE_1024D_V1`

**用途**: 存储 DINOv2 全局特征和 Patch Tokens，用于精细化检索

**Schema**:
```python
{
    "id": Int64 (Primary Key, Auto ID),
    "item_id": Int64 (零件ID, 索引),
    "item_name": VarChar(200),
    "item_code": VarChar(100) (索引),
    "image_id": Int64,
    "image_url": VarChar(500),
    
    # 向量字段
    "global_vector": FloatVector(1024) (全局特征),
    "patch_tokens": FloatVector(1024) (Patch Token 均值, 用于粗排),
    
    # Patch Tokens 存储策略
    "patch_tokens_path": VarChar(500) (Patch Tokens .npy 文件路径),
    
    # 元数据
    "created_at": VarChar(50),
}
```

**索引配置**:
- `global_vector`: IVF_FLAT, metric_type="L2", nlist=1024
- `patch_tokens`: IVF_FLAT, metric_type="L2", nlist=1024
- `item_id`, `item_code`: 标量索引

### 3.3 关键设计决策

**为什么分两个 Collection？**
- SigLIP 和 DINOv2 特征维度不同，检索策略不同
- 便于独立优化索引参数
- 支持灵活的多路召回策略

**Patch Tokens 存储策略**
- 1369×1024 太大，不适合存入 Milvus
- 粗排阶段只用全局向量
- 精排阶段从文件系统加载完整 Patch Tokens
- 存储路径：`vector_db/patch_tokens/item_{item_id}_img_{image_id}.npy`

**文本向量存储位置**
- 存在 SigLIP Collection 中，与图像向量共享空间
- 每个零件的文本描述生成一个文本向量
- 检索时文本查询直接在 image_vector 字段中搜索

**版本化 Collection 命名**
- 命名包含模型信息和维度：`PARTS_{MODEL}_{DIM}D_V{VERSION}`
- 支持后续模型切换和 A/B 测试
- 配置文件驱动，修改配置即可切换模型

---

## 4. 模型封装层设计

### 4.1 SigLIP 2 特征提取器

**文件**: `vector_db/models/siglip_extractor.py`

**核心功能**:
```python
class SigLIPExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        """初始化 SigLIP 2 模型"""
        
    def extract_image_features(self, image: PIL.Image) -> np.ndarray:
        """提取图像特征向量 (1152,)"""
        
    def extract_text_features(self, text: str) -> np.ndarray:
        """提取文本特征向量 (1152,)"""
        
    def extract_batch_images(self, images: List[PIL.Image]) -> np.ndarray:
        """批量提取图像特征 (N, 1152)"""
```

**技术要点**:
- 使用 `transformers` 库加载 SigLIP 模型
- 图像预处理：Resize → Normalize
- 特征归一化：L2 normalization
- 支持批量处理（batch_size=32）

### 4.2 DINOv2 特征提取器

**文件**: `vector_db/models/dinov2_extractor.py`

**策略**: 复用现有的 `core/components/dinov2_extractor.py`，扩展向量库专用功能

**新增功能**:
```python
class DINOv2VectorExtractor(DinoV2Extractor):
    def extract_global_vector(self, image: PIL.Image) -> np.ndarray:
        """提取全局特征向量（CLS token）(1024,)"""
        
    def extract_patch_tokens(self, image: PIL.Image) -> np.ndarray:
        """提取完整 Patch Tokens (1369, 1024)"""
        
    def extract_patch_mean(self, image: PIL.Image) -> np.ndarray:
        """提取 Patch Tokens 的均值向量 (1024,)"""
        
    def save_patch_tokens(self, patch_tokens: np.ndarray, save_path: str):
        """保存 Patch Tokens 到 .npy 文件"""
```

### 4.3 模型配置管理

**文件**: `vector_db/models/model_config.py`

```python
@dataclass
class SigLIPConfig:
    model_name: str = "google/siglip-so400m-patch14-384"
    model_path: str = "/data/models/CV-models/siglip-so400m-patch14-384"
    vector_dim: int = 1152
    collection_name: str = "PARTS_SIGLIP_SO400M_1152D_V1"
    batch_size: int = 32
    device: str = "cuda"

@dataclass
class DINOv2Config:
    model_name: str = "facebook/dinov2-large"
    model_path: str = "/data/models/CV-models/dinov2-large"
    vector_dim: int = 1024
    collection_name: str = "PARTS_DINOV2_LARGE_1024D_V1"
    patch_tokens_dir: str = "vector_db/patch_tokens"
    device: str = "cuda"
```

---

## 5. 存储层设计

### 5.1 Milvus 连接管理

**文件**: `vector_db/storage/milvus_client.py`

```python
class MilvusConnectionManager:
    """Milvus 连接单例管理器"""
    
    @classmethod
    def get_client(cls, host: str, port: int) -> MilvusClient:
        """获取 Milvus 客户端单例"""
        
    @classmethod
    def close(cls):
        """关闭连接"""
```

### 5.2 Collection 管理器

**文件**: `vector_db/storage/collection_manager.py`

**核心功能**:
```python
class CollectionManager:
    def create_siglip_collection(self, collection_name: str, vector_dim: int) -> bool:
        """创建 SigLIP Collection"""
        
    def create_dinov2_collection(self, collection_name: str, vector_dim: int) -> bool:
        """创建 DINOv2 Collection"""
        
    def collection_exists(self, collection_name: str) -> bool:
        """检查 Collection 是否存在"""
        
    def drop_collection(self, collection_name: str):
        """删除 Collection（谨慎使用）"""
        
    def insert_siglip_records(self, collection_name: str, records: List[Dict]) -> List[int]:
        """批量插入 SigLIP 记录"""
        
    def insert_dinov2_records(self, collection_name: str, records: List[Dict]) -> List[int]:
        """批量插入 DINOv2 记录"""
```

**索引类型选择**:
- `IVF_FLAT`: 平衡精度和速度，适合中等规模数据（10万~100万）
- 如果数据量更大，可以切换到 `HNSW`

**度量类型**:
- SigLIP: `COSINE`（余弦相似度，适合归一化向量）
- DINOv2: `L2`（欧氏距离，符合论文描述）

---

## 6. 数据访问层设计

### 6.1 数据库连接器

**文件**: `vector_db/data/db_connector.py`

```python
class DatabaseConnector:
    """PostgreSQL 数据库连接器（同步版本）"""
    
    def __init__(self, config_path: str):
        """初始化数据库连接"""
        
    def fetch_items_with_images(
        self, 
        item_ids: Optional[List[int]] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """读取零件信息及其关联图像"""
        
    def mark_images_indexed(self, image_ids: List[int]):
        """标记图像已索引"""
```

**返回数据格式**:
```python
[
    {
        'item_id': 12345,
        'item_name': '弹簧座',
        'item_code': 'TZJ-001',
        'description': '左侧转向架弹簧座',
        'images': [
            {'image_id': 67890, 'image_url': 'minio://...'},
            {'image_id': 67891, 'image_url': 'minio://...'}
        ]
    },
    ...
]
```

### 6.2 图像加载器

**文件**: `vector_db/data/image_loader.py`

```python
class ImageLoader:
    """图像加载器（支持 MinIO 和本地文件系统）"""
    
    def load_image(self, image_url: str) -> Optional[Image.Image]:
        """加载图像（自动识别来源）"""
        
    def _load_from_minio(self, minio_url: str) -> Image.Image:
        """从 MinIO 加载图像"""
        
    def _load_from_url(self, url: str) -> Image.Image:
        """从 HTTP URL 加载图像"""
        
    def _load_from_local(self, file_path: str) -> Image.Image:
        """从本地文件系统加载图像"""
```

**支持格式**:
- `minio://bucket/path/to/image.jpg`
- `/local/path/to/image.jpg`
- `http://example.com/image.jpg`

---

## 7. 入库业务逻辑层设计

### 7.1 单条入库逻辑

**文件**: `vector_db/indexing/indexer.py`

```python
class VectorIndexer:
    """向量入库核心逻辑"""
    
    def index_item(
        self,
        item_id: int,
        item_name: str,
        item_code: str,
        description: str,
        images: List[Dict]
    ) -> Dict:
        """
        为单个零件的所有图像建立索引
        
        返回:
        {
            'item_id': 12345,
            'siglip_ids': [1, 2, 3],
            'dinov2_ids': [4, 5, 6],
            'indexed_image_ids': [67890, 67891],
            'failed_image_ids': []
        }
        """
```

**处理流程**:
1. 提取 SigLIP 特征（图像 + 文本）
2. 提取 DINOv2 特征（全局 + Patch Tokens）
3. 保存 Patch Tokens 到文件系统
4. 批量插入 Milvus

### 7.2 批量入库协调器

**文件**: `vector_db/indexing/batch_indexer.py`

```python
class BatchIndexer:
    """批量入库协调器（同步版本）"""
    
    def build_index(
        self,
        item_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
        resume: bool = False
    ):
        """批量构建索引"""
```

**核心功能**:
- 断点续传：支持从中断点继续
- 进度监控：使用 tqdm 显示进度
- 错误处理：单条失败不影响整体流程
- 检查点保存：定期保存已处理的 item_id
- 生成报告：记录成功/失败统计

---

## 8. 命令行工具

### 8.1 Collection 初始化脚本

**文件**: `vector_db/scripts/init_collections.py`

**用法**:
```bash
# 创建所有 Collections
python vector_db/scripts/init_collections.py

# 删除并重建（危险操作）
python vector_db/scripts/init_collections.py --drop
```

### 8.2 批量入库脚本

**文件**: `vector_db/scripts/build_index.py`

**用法**:
```bash
# 全量构建
python vector_db/scripts/build_index.py

# 指定零件ID
python vector_db/scripts/build_index.py --item-ids 12345 67890

# 限制数量（测试用）
python vector_db/scripts/build_index.py --limit 10

# 断点续传
python vector_db/scripts/build_index.py --resume
```

---

## 9. 配置文件

### 9.1 向量库配置

**文件**: `vector_db/config/vector_db.ini`

```ini
[milvus]
host = localhost
port = 19530
user = root
password = Milvus

[siglip]
model_name = google/siglip-so400m-patch14-384
model_path = /data/models/CV-models/siglip-so400m-patch14-384
vector_dim = 1152
collection_name = PARTS_SIGLIP_SO400M_1152D_V1
batch_size = 32
device = cuda

[dinov2]
model_name = facebook/dinov2-large
model_path = /data/models/CV-models/dinov2-large
vector_dim = 1024
collection_name = PARTS_DINOV2_LARGE_1024D_V1
patch_tokens_dir = vector_db/patch_tokens
device = cuda

[indexing]
checkpoint_file = vector_db/checkpoint.json
log_dir = vector_db/logs
```

### 9.2 数据库配置

**文件**: `vector_db/config/db_config.ini`

```ini
[postgresql]
host = 192.168.100.2
port = 5432
database = zcsf
user = root
password = Y2iaciej@bfr

[minio]
endpoint = 192.168.100.2:9000
access_key = root
secret_key = Y2iaciej@bfr
bucket = bfr-ai-files
secure = false

[local]
image_root = /data/railway_parts_images
```

---

## 10. 日志系统

**文件**: `vector_db/utils/logger.py`

**功能**:
- 控制台输出：INFO 级别
- 文件记录：DEBUG 级别
- 日志文件：`vector_db/logs/indexing_YYYYMMDD.log`
- 自动按日期分割日志文件

---

## 11. 异步优化方案（后续迭代）

### 11.1 性能瓶颈分析

建库过程的耗时操作：
1. 数据库查询：读取大量零件信息
2. 图像加载：从 MinIO/文件系统下载
3. 模型推理：SigLIP + DINOv2 特征提取（最耗时）
4. 向量插入：批量写入 Milvus

### 11.2 异步架构设计

```python
# 三层异步策略
1. 数据库查询层：异步 I/O（asyncio + asyncpg）
2. 特征提取层：多进程并行（multiprocessing）
3. 向量插入层：异步批量写入（asyncio + Milvus async client）
```

### 11.3 性能估算

假设：
- 10万张图像
- 4 GPU 并行
- 每张图像处理时间 0.5秒

**同步处理**: 100,000 × 0.5s = 13.9 小时
**异步处理**: 100,000 × 0.5s / 4 = **3.5 小时**

### 11.4 优化要点

1. **多 GPU 并行**：每个 worker 绑定一个 GPU
2. **断点续传**：记录已处理的 item_id
3. **内存控制**：限制队列大小，避免内存溢出
4. **错误处理**：单条失败不影响整体流程

---

## 12. 依赖项

### 12.1 Python 包

```txt
# 核心依赖
torch>=2.0.0
transformers>=4.30.0
pillow>=9.0.0
numpy>=1.24.0

# 向量数据库
pymilvus>=2.3.0

# 数据库
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# MinIO
minio>=7.1.0

# 工具
tqdm>=4.65.0
requests>=2.28.0
```

### 12.2 外部服务

- **Milvus**: 向量数据库（版本 >= 2.3）
- **PostgreSQL**: 关系数据库（recognition 项目）
- **MinIO**: 对象存储（可选）

---

## 13. 测试策略

### 13.1 单元测试

- 模型特征提取测试
- 数据库连接测试
- 图像加载测试
- Milvus 操作测试

### 13.2 集成测试

- 端到端入库流程测试
- 小批量数据测试（10条）
- 断点续传测试

### 13.3 性能测试

- 单张图像处理时间
- 批量入库吞吐量
- 内存占用监控

---

## 14. 风险与缓解

### 14.1 风险识别

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 模型加载失败 | 无法启动 | 提供详细错误日志，检查模型路径 |
| 图像加载失败 | 部分数据丢失 | 记录失败图像，支持重试 |
| Milvus 连接中断 | 入库中断 | 断点续传机制 |
| 内存溢出 | 程序崩溃 | 批量处理，限制队列大小 |
| 磁盘空间不足 | Patch Tokens 无法保存 | 监控磁盘空间，提前告警 |

### 14.2 数据一致性

- 使用事务确保数据库标记和向量插入的一致性
- 失败记录单独保存，支持手动重试
- 定期验证向量库和数据库的数据一致性

---

## 15. 后续扩展

### 15.1 第二阶段：检索模块

- Grounding DINO 目标定位
- 两阶段检索（RRF粗排 + Patch精排）
- REST API 接口

### 15.2 功能增强

- 增量更新：支持新增零件自动入库
- 向量更新：支持已入库零件的特征更新
- 数据清理：支持删除无效向量
- 监控面板：实时监控入库进度和向量库状态

### 15.3 性能优化

- 异步入库实现
- 多 GPU 并行
- 模型量化加速
- 向量压缩（PQ/SQ）

---

## 16. 总结

本设计方案提供了一个完整的向量库入库系统，具有以下特点：

1. **模块化设计**：各层职责清晰，易于维护和扩展
2. **配置驱动**：支持灵活切换模型和参数
3. **容错机制**：断点续传、错误日志、失败重试
4. **可扩展性**：预留异步优化和检索模块接口
5. **工程实践**：日志系统、进度监控、性能测试

该方案优先实现同步版本，快速跑通技术路线，后续可根据实际需求进行异步优化和功能扩展。
