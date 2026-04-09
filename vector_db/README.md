# Vector Database Indexing Module

向量库入库模块，用于将零件图像特征存储到 Milvus 向量数据库。

## 功能特性

- **多模态特征提取**: SigLIP 2 (图像+文本) + DINOv2 (细粒度视觉特征)
- **灵活的数据源**: 支持 PostgreSQL + MinIO/本地文件系统
- **断点续传**: 支持从中断点恢复批量入库
- **配置驱动**: 通过 INI 配置文件管理所有参数
- **模块化设计**: 清晰的分层架构，易于维护和扩展

## 目录结构

```
vector_db/
├── models/              # 模型封装层
│   ├── model_config.py  # 配置管理
│   ├── siglip_extractor.py   # SigLIP 2 特征提取
│   └── dinov2_extractor.py   # DINOv2 特征提取
├── storage/             # 存储层
│   ├── milvus_client.py      # Milvus 连接管理
│   └── collection_manager.py # Collection 管理
├── data/                # 数据访问层
│   ├── db_connector.py       # PostgreSQL 连接
│   └── image_loader.py       # 图像加载
├── indexing/            # 业务逻辑层
│   ├── indexer.py            # 单条入库
│   └── batch_indexer.py      # 批量入库
├── utils/               # 工具函数
│   └── logger.py             # 日志配置
├── config/              # 配置文件
│   ├── vector_db.ini         # 向量库配置
│   └── db_config.ini         # 数据库配置
└── scripts/             # 命令行工具
    ├── init_collections.py   # 初始化 Collections
    └── build_index.py        # 批量入库
```

## 快速开始

### 1. 初始化 Milvus Collections

```bash
python vector_db/scripts/init_collections.py
```

### 2. 批量入库

```bash
# 全量构建
python vector_db/scripts/build_index.py

# 指定零件 ID
python vector_db/scripts/build_index.py --item-ids 12345 67890

# 限制数量（测试用）
python vector_db/scripts/build_index.py --limit 10

# 断点续传
python vector_db/scripts/build_index.py --resume
```

## 配置说明

### vector_db.ini

```ini
[milvus]
host = localhost
port = 19530

[siglip]
model_path = /data/models/CV-models/siglip-so400m-patch14-384
vector_dim = 1152
collection_name = PARTS_SIGLIP_SO400M_1152D_V1

[dinov2]
model_path = /data/models/CV-models/dinov2-large
vector_dim = 1024
collection_name = PARTS_DINOV2_LARGE_1024D_V1
patch_tokens_dir = vector_db/patch_tokens
```

### db_config.ini

```ini
[postgresql]
host = 192.168.100.2
database = zcsf

[minio]
endpoint = 192.168.100.2:9000
bucket = bfr-ai-files
```

## 依赖项

```
torch>=2.0.0
transformers>=4.30.0
pymilvus>=2.3.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
minio>=7.1.0
pillow>=9.0.0
tqdm>=4.65.0
```

## Collection Schema

### SigLIP Collection (图文共享空间)

- `image_vector`: FloatVector(1152) - 图像特征
- `text_vector`: FloatVector(1152) - 文本特征
- 索引: IVF_FLAT, COSINE

### DINOv2 Collection (细粒度视觉特征)

- `global_vector`: FloatVector(1024) - 全局特征
- `patch_tokens`: FloatVector(1024) - Patch 均值
- `patch_tokens_path`: VarChar - 完整 Patch Tokens 文件路径
- 索引: IVF_FLAT, L2

## 后续扩展

- 检索模块 (Grounding DINO + 两阶段检索)
- 异步入库优化
- REST API 接口
- 监控面板
