# 向量库入库模块实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于 SigLIP 2 + DINOv2 的向量库入库系统，支持从 PostgreSQL 读取零件信息并提取多模态特征存储到 Milvus

**Architecture:** 分层架构 - 模型层（特征提取）、存储层（Milvus 管理）、数据访问层（DB + 图像加载）、业务逻辑层（入库协调）

**Tech Stack:** Python 3.10+, transformers, torch, pymilvus, sqlalchemy, Pillow, minio

---

## 文件结构概览

**新建文件：**
vector_db/
├── init.py
├── models/
│   ├── init.py
│   ├── model_config.py
│   ├── siglip_extractor.py
│   └── dinov2_extractor.py
├── storage/
│   ├── init.py
│   ├── milvus_client.py
│   └── collection_manager.py
├── data/
│   ├── init.py
│   ├── db_connector.py
│   └── image_loader.py
├── indexing/
│   ├── init.py
│   ├── indexer.py
│   └── batch_indexer.py
├── utils/
│   ├── init.py
│   └── logger.py
├── config/
│   ├── vector_db.ini
│   └── db_config.ini
└── scripts/
├── init_collections.py
└── build_index.py



**修改文件：**
- `core/components/dinov2_extractor.py` - 扩展向量提取功能

---

## Task 1: 项目基础结构和配置

**Files:**
- Create: `vector_db/__init__.py`
- Create: `vector_db/config/vector_db.ini`
- Create: `vector_db/config/db_config.ini`
- Create: `vector_db/utils/__init__.py`
- Create: `vector_db/utils/logger.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p vector_db/{models,storage,data,indexing,utils,config,scripts}
touch vector_db/__init__.py
touch vector_db/{models,storage,data,indexing,utils,scripts}/__init__.py
 Step 2: 创建向量库配置文件
创建 vector_db/config/vector_db.ini:


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
 Step 3: 创建数据库配置文件
创建 vector_db/config/db_config.ini:


[postgresql]
host = localhost
port = 5432
database = recognition_db
user = postgres
password = your_password

[minio]
endpoint = localhost:9000
access_key = minioadmin
secret_key = minioadmin
bucket = railway-parts
secure = false

[local]
image_root = /data/railway_parts_images
 Step 4: 创建日志工具
创建 vector_db/utils/logger.py:


import logging
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = "vector_db/logs") -> logging.Logger:
    """
    获取配置好的 logger
    
    Args:
        name: logger 名称
        log_dir: 日志目录
        
    Returns:
        配置好的 logger 实例
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    log_file = os.path.join(
        log_dir,
        f"indexing_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
 Step 5: 创建 utils init.py
创建 vector_db/utils/__init__.py:


from .logger import get_logger

__all__ = ['get_logger']
 Step 6: 提交基础结构

git add vector_db/
git commit -m "feat(vector_db): 初始化项目结构和配置文件"

Task 2: 模型配置管理

Files:

Create: vector_db/models/__init__.py

Create: vector_db/models/model_config.py

 Step 1: 创建模型配置类

创建 vector_db/models/model_config.py:


from dataclasses import dataclass
import configparser
import os

@dataclass
class SigLIPConfig:
    """SigLIP 模型配置"""
    model_name: str = "google/siglip-so400m-patch14-384"
    model_path: str = "/data/models/CV-models/siglip-so400m-patch14-384"
    vector_dim: int = 1152
    collection_name: str = "PARTS_SIGLIP_SO400M_1152D_V1"
    batch_size: int = 32
    device: str = "cuda"

    @classmethod
    def from_config(cls, config_path: str = "vector_db/config/vector_db.ini"):
        """从配置文件加载"""
        if not os.path.exists(config_path):
            return cls()

        config = configparser.ConfigParser()
        config.read(config_path)

        if 'siglip' not in config:
            return cls()

        siglip = config['siglip']
        return cls(
            model_name=siglip.get('model_name', cls.model_name),
            model_path=siglip.get('model_path', cls.model_path),
            vector_dim=siglip.getint('vector_dim', cls.vector_dim),
            collection_name=siglip.get('collection_name', cls.collection_name),
            batch_size=siglip.getint('batch_size', cls.batch_size),
            device=siglip.get('device', cls.device)
        )


@dataclass
class DINOv2Config:
    """DINOv2 模型配置"""
    model_name: str = "facebook/dinov2-large"
    model_path: str = "/data/models/CV-models/dinov2-large"
    vector_dim: int = 1024
    collection_name: str = "PARTS_DINOV2_LARGE_1024D_V1"
    patch_tokens_dir: str = "vector_db/patch_tokens"
    device: str = "cuda"

    @classmethod
    def from_config(cls, config_path: str = "vector_db/config/vector_db.ini"):
        """从配置文件加载"""
        if not os.path.exists(config_path):
            return cls()

        config = configparser.ConfigParser()
        config.read(config_path)

        if 'dinov2' not in config:
            return cls()

        dinov2 = config['dinov2']
        return cls(
            model_name=dinov2.get('model_name', cls.model_name),
            model_path=dinov2.get('model_path', cls.model_path),
            vector_dim=dinov2.getint('vector_dim', cls.vector_dim),
            collection_name=dinov2.get('collection_name', cls.collection_name),
            patch_tokens_dir=dinov2.get('patch_tokens_dir', cls.patch_tokens_dir),
            device=dinov2.get('device', cls.device)
        )
 Step 2: 创建 models init.py
创建 vector_db/models/__init__.py:


from .model_config import SigLIPConfig, DINOv2Config

__all__ = ['SigLIPConfig', 'DINOv2Config']
 Step 3: 提交模型配置

git add vector_db/models/
git commit -m "feat(vector_db): 添加模型配置管理"
Task 3: SigLIP 2 特征提取器
Files:

Create: vector_db/models/siglip_extractor.py

Modify: vector_db/models/__init__.py

 Step 1: 创建 SigLIP 提取器

创建 vector_db/models/siglip_extractor.py:


import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import AutoProcessor, AutoModel
from vector_db.utils.logger import get_logger

logger = get_logger(__name__)


class SigLIPExtractor:
    """
    SigLIP 2 特征提取器
    支持图像和文本的特征提取，输出在共享的潜空间中
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda"
    ):
        """
        初始化 SigLIP 2 模型
        
        Args:
            model_name: HuggingFace 模型名称或本地路径
            device: 运行设备 ('cuda' / 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        logger.info(f"Loading SigLIP model: {model_name}")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"SigLIP loaded successfully on {self.device}")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """
        提取图像特征向量
        
        Args:
            image: PIL Image 对象
            
        Returns:
            归一化的图像特征向量 (1152,)
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.get_image_features(**inputs)
            features = outputs.cpu().numpy()[0]
            
            # L2 归一化
            features = features / np.linalg.norm(features)
            
        return features

    def extract_text_features(self, text: str) -> np.ndarray:
        """
        提取文本特征向量
        
        Args:
            text: 文本描述（如"左侧转向架弹簧座"）
            
        Returns:
            归一化的文本特征向量 (1152,)
        """
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.get_text_features(**inputs)
            features = outputs.cpu().numpy()[0]
            
            # L2 归一化
            features = features / np.linalg.norm(features)
            
        return features

    def extract_batch_images(
        self,
        images: List[Image.Image],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        批量提取图像特征
        
        Args:
            images: PIL Image 列表
            batch_size: 批处理大小
            
        Returns:
            特征矩阵 (N, 1152)
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model.get_image_features(**inputs)
                features = outputs.cpu().numpy()
                
                # L2 归一化
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
                
                all_features.append(features)
        
        return np.vstack(all_features)
 Step 2: 更新 models init.py
修改 vector_db/models/__init__.py:


from .model_config import SigLIPConfig, DINOv2Config
from .siglip_extractor import SigLIPExtractor

__all__ = ['SigLIPConfig', 'DINOv2Config', 'SigLIPExtractor']
 Step 3: 提交 SigLIP 提取器

git add vector_db/models/
git commit -m "feat(vector_db): 添加 SigLIP 2 特征提取器"
Task 4: DINOv2 特征提取器扩展
Files:

Create: vector_db/models/dinov2_extractor.py

Modify: vector_db/models/__init__.py

 Step 1: 创建 DINOv2 向量提取器

创建 vector_db/models/dinov2_extractor.py:


import torch
import numpy as np
from PIL import Image
import os
from typing import Optional
import sys

# 导入现有的 DINOv2 提取器
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.components.dinov2_extractor import DinoV2Extractor

from vector_db.utils.logger import get_logger

logger = get_logger(__name__)


class DINOv2VectorExtractor(DinoV2Extractor):
    """
    DINOv2 向量提取器
    继承现有的 DinoV2Extractor，扩展向量库专用功能
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        device: str = "cuda",
        finetuned_checkpoint: Optional[str] = None
    ):
        """
        初始化 DINOv2 向量提取器
        
        Args:
            model_name: HuggingFace 模型名称
            device: 运行设备
            finetuned_checkpoint: 微调权重路径（可选）
        """
        super().__init__(
            model_name=model_name,
            device=device,
            finetuned_checkpoint=finetuned_checkpoint
        )
        logger.info("DINOv2VectorExtractor initialized")

    def extract_global_vector(self, image: Image.Image) -> np.ndarray:
        """
        提取全局特征向量（CLS token）
        
        Args:
            image: PIL Image 对象
            
        Returns:
            全局特征向量 (1024,)
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # CLS token 是第一个 token
            cls_token = outputs.last_hidden_state[:, 0, :]
            features = cls_token.cpu().numpy()[0]
            
        return features

    def extract_patch_tokens(self, image: Image.Image) -> np.ndarray:
        """
        提取完整 Patch Tokens
        
        Args:
            image: PIL Image 对象
            
        Returns:
            Patch Tokens 矩阵 (1369, 1024)
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # 去掉 CLS token，只保留 patch tokens
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            features = patch_tokens.cpu().numpy()[0]
            
        return features

    def extract_patch_mean(self, image: Image.Image) -> np.ndarray:
        """
        提取 Patch Tokens 的均值向量（用于粗排）
        
        Args:
            image: PIL Image 对象
            
        Returns:
            Patch Tokens 均值向量 (1024,)
        """
        patch_tokens = self.extract_patch_tokens(image)
        return patch_tokens.mean(axis=0)

    def save_patch_tokens(self, patch_tokens: np.ndarray, save_path: str):
        """
        保存 Patch Tokens 到 .npy 文件
        
        Args:
            patch_tokens: Patch Tokens 矩阵 (1369, 1024)
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, patch_tokens)
        logger.debug(f"Saved patch tokens to {save_path}")

    @staticmethod
    def load_patch_tokens(load_path: str) -> np.ndarray:
        """
        从 .npy 文件加载 Patch Tokens
        
        Args:
            load_path: 文件路径
            
        Returns:
            Patch Tokens 矩阵 (1369, 1024)
        """
        return np.load(load_path)
 Step 2: 更新 models init.py
修改 vector_db/models/__init__.py:


from .model_config import SigLIPConfig, DINOv2Config
from .siglip_extractor import SigLIPExtractor
from .dinov2_extractor import DINOv2VectorExtractor

__all__ = [
    'SigLIPConfig',
    'DINOv2Config',
    'SigLIPExtractor',
    'DINOv2VectorExtractor'
]
 Step 3: 提交 DINOv2 扩展

git add vector_db/models/
git commit -m "feat(vector_db): 添加 DINOv2 向量提取器扩展"
Task 5: Milvus 连接管理
Files:

Create: vector_db/storage/__init__.py

Create: vector_db/storage/milvus_client.py

 Step 1: 创建 Milvus 连接管理器

创建 vector_db/storage/milvus_client.py:


from pymilvus import MilvusClient
from typing import Optional
from vector_db.utils.logger import get_logger

logger = get_logger(__name__)


class MilvusConnectionManager:
    """
    Milvus 连接单例管理器
    """
    _instance = None
    _client: Optional[MilvusClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_client(
        cls,
        host: str = "localhost",
        port: int = 19530,
        user: str = "root",
        password: str = "Milvus"
    ) -> MilvusClient:
        """
        获取 Milvus 客户端单例
        
        Args:
            host: Milvus 主机地址
            port: Milvus 端口
            user: 用户名
            password: 密码
            
        Returns:
            MilvusClient 实例
        """
        if cls._client is None:
            uri = f"http://{host}:{port}"
            logger.info(f"Connecting to Milvus at {uri}")
            
            cls._client = MilvusClient(
                uri=uri,
                token=f"{user}:{password}"
            )
            
            logger.info("Milvus client connected successfully")
        
        return cls._client

    @classmethod
    def close(cls):
        """关闭连接"""
        if cls._client:
            cls._client.close()
            cls._client = None
            logger.info("Milvus client closed")
 Step 2: 创建 storage init.py
创建 vector_db/storage/__init__.py:


from .milvus_client import MilvusConnectionManager

__all__ = ['MilvusConnectionManager']
 Step 3: 提交 Milvus 连接管理

git add vector_db/storage/
git commit -m "feat(vector_db): 添加 Milvus 连接管理器"
Task 6: Milvus Collection 管理器
Files:

Create: vector_db/storage/collection_manager.py

Modify: vector_db/storage/__init__.py

 Step 1: 创建 Collection 管理器（第1部分：基础功能）

创建 vector_db/storage/collection_manager.py:


from pymilvus import MilvusClient, DataType
from typing import List, Dict
from vector_db.utils.logger import get_logger

logger = get_logger(__name__)


class CollectionManager:
    """
    Collection 创建、初始化、管理
    """

    def __init__(self, client: MilvusClient):
        """
        初始化 Collection 管理器
        
        Args:
            client: MilvusClient 实例
        """
        self.client = client

    def collection_exists(self, collection_name: str) -> bool:
        """
        检查 Collection 是否存在
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            是否存在
        """
        return self.client.has_collection(collection_name)

    def drop_collection(self, collection_name: str):
        """
        删除 Collection（谨慎使用）
        
        Args:
            collection_name: Collection 名称
        """
        if self.collection_exists(collection_name):
            self.client.drop_collection(collection_name)
            logger.warning(f"Dropped collection: {collection_name}")
 Step 2: 添加 SigLIP Collection 创建功能
在 vector_db/storage/collection_manager.py 中添加方法：


    def create_siglip_collection(
        self,
        collection_name: str,
        vector_dim: int = 1152
    ) -> bool:
        """
        创建 SigLIP Collection
        
        Args:
            collection_name: Collection 名称
            vector_dim: 向量维度
            
        Returns:
            是否创建成功
        """
        logger.info(f"Creating SigLIP collection: {collection_name}")
        
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )

        # 添加字段
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("item_id", DataType.INT64)
        schema.add_field("item_name", DataType.VARCHAR, max_length=200)
        schema.add_field("item_code", DataType.VARCHAR, max_length=100)
        schema.add_field("image_id", DataType.INT64)
        schema.add_field("image_url", DataType.VARCHAR, max_length=500)
        schema.add_field("description", DataType.VARCHAR, max_length=1000)
        schema.add_field("image_vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("text_vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("created_at", DataType.VARCHAR, max_length=50)

        # 创建索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="image_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024}
        )
        index_params.add_index(
            field_name="text_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024}
        )

        # 创建 Collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"SigLIP collection created: {collection_name}")
        return True
 Step 3: 添加 DINOv2 Collection 创建功能
在 vector_db/storage/collection_manager.py 中继续添加：


    def create_dinov2_collection(
        self,
        collection_name: str,
        vector_dim: int = 1024
    ) -> bool:
        """
        创建 DINOv2 Collection
        
        Args:
            collection_name: Collection 名称
            vector_dim: 向量维度
            
        Returns:
            是否创建成功
        """
        logger.info(f"Creating DINOv2 collection: {collection_name}")
        
        schema = self.client.create_schema(auto_id=True)

        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("item_id", DataType.INT64)
        schema.add_field("item_name", DataType.VARCHAR, max_length=200)
        schema.add_field("item_code", DataType.VARCHAR, max_length=100)
        schema.add_field("image_id", DataType.INT64)
        schema.add_field("image_url", DataType.VARCHAR, max_length=500)
        schema.add_field("global_vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("patch_tokens", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("patch_tokens_path", DataType.VARCHAR, max_length=500)
        schema.add_field("created_at", DataType.VARCHAR, max_length=50)

        # 创建索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="global_vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024}
        )
        index_params.add_index(
            field_name="patch_tokens",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024}
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"DINOv2 collection created: {collection_name}")
        return True
 Step 4: 添加数据插入功能
在 vector_db/storage/collection_manager.py 中继续添加：


    def insert_siglip_records(
        self,
        collection_name: str,
        records: List[Dict]
    ) -> List[int]:
        """
        批量插入 SigLIP 记录
        
        Args:
            collection_name: Collection 名称
            records: 记录列表
            
        Returns:
            插入的记录 ID 列表
        """
        result = self.client.insert(
            collection_name=collection_name,
            data=records
        )
        logger.debug(f"Inserted {len(records)} records to {collection_name}")
        return result["ids"]

    def insert_dinov2_records(
        self,
        collection_name: str,
        records: List[Dict]
    ) -> List[int]:
        """
        批量插入 DINOv2 记录
        
        Args:
            collection_name: Collection 名称
            records: 记录列表
            
        Returns:
            插入的记录 ID 列表
        """
        result = self.client.insert(
            collection_name=collection_