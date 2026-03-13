import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoImageProcessor, AutoModel

# 1. 设备和本地模型路径配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOCAL_MODEL_PATH = '/media/dell/新加卷1/LLM/models/CV-models/dinov2-large'
EXAMPLE_PATH = '/media/dell/新加卷1/CV/DinoSamClip/src'
folder_path = f'{EXAMPLE_PATH}/imgs/'

print(f"Loading HF DinoV2 from: {LOCAL_MODEL_PATH}")

# 2. 加载 Hugging Face 处理器和模型
processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
model.eval()

# 设定处理的图像数量 (原代码是 4)
num_images = 4
total_features = []
list_img_path = os.listdir(folder_path)[:num_images]

# 用于保存每张图的 patch 大小，后续 reshape 画图时需要用到
grid_sizes = []

print("Extracting features...")
with torch.no_grad():
    for img_name in list_img_path:
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # 使用 processor 替代 transform1
        # 默认情况下，处理器会将图像 resize 到模型期望的输入大小 (通常是 518x518 或相近倍数)
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        # 计算当前图像经过预处理后的实际长宽，用于推导 grid 大小
        _, _, h, w = inputs["pixel_values"].shape
        patch_size = model.config.patch_size # 自动读取模型配置的 patch 大小 (通常是 14)
        patch_h = h // patch_size
        patch_w = w // patch_size
        grid_sizes.append((patch_h, patch_w))
        
        # 前向传播
        outputs = model(**inputs)
        
        # 提取 Patch tokens (抛弃第 0 个 CLS token)
        # outputs.last_hidden_state 形状: [batch(1), num_patches + 1, feat_dim]
        patch_tokens = outputs.last_hidden_state[0, 1:, :] # 形状: [patch_h * patch_w, feat_dim]
        
        total_features.append(patch_tokens.cpu())

# 3. 准备 PCA 数据
# 将所有图像的特征拼接起来
# total_features 拼接后形状: [num_images * patch_h * patch_w, feat_dim]
total_features = torch.cat(total_features, dim=0).numpy()

# 动态获取特征维度 (对于 large 模型，这里应该是 1024)
feat_dim = total_features.shape[-1]
print(f"Feature dimension: {feat_dim}")

# ==========================================
# 下面的 PCA 逻辑和原博客基本一致，只做微调以适应可能不同的 patch 尺寸
# ==========================================
print("Running PCA...")
pca = PCA(n_components=3)
pca.fit(total_features)
pca_features = pca.transform(total_features)

# 绘制 PCA 前三个主成分的直方图
plt.figure(figsize=(15, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.hist(pca_features[:, i], bins=50)
    plt.title(f"PCA Component {i+1}")
plt.tight_layout()
plt.savefig("dino_pca_1_histograms.png", dpi=150)
plt.close() # 保存后关闭画布释放内存
print("已保存直方图 -> dino_pca_1_histograms.png")

# 对第一个主成分 (通常代表前景/背景区分度最高的特征) 进行 Min-Max 归一化
pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                     (pca_features[:, 0].max() - pca_features[:, 0].min())

# 绘制第一主成分的特征图
plt.figure(figsize=(10, 10))
start_idx = 0
for i in range(num_images):
    ph, pw = grid_sizes[i]
    num_patches = ph * pw
    
    plt.subplot(2, 2, i+1)
    # 提取当前图像对应的特征并 reshape 成 2D 图像
    img_pca = pca_features[start_idx : start_idx + num_patches, 0].reshape(ph, pw)
    plt.imshow(img_pca)
    plt.title(f"PCA Component 1 - Img {i+1}")
    plt.axis('off')
    
    start_idx += num_patches
plt.tight_layout()
plt.savefig("dino_pca_2_component1.png", dpi=150)
plt.close()
print("已保存第一主成分图 -> dino_pca_2_component1.png")

# 4. 基于第一主成分分离前景和背景
# 这里设定阈值为 0.5。注意：有时候前景是 >0.5，有时候是 <0.5，这取决于 PCA 拟合的方向
# 如果你发现抠出来的是背景，就把 < 0.5 改成 > 0.5
pca_features_bg = pca_features[:, 0] > 0.5 
pca_features_fg = ~pca_features_bg

# 绘制分离出的背景/前景掩码
plt.figure(figsize=(10, 10))
start_idx = 0
for i in range(num_images):
    ph, pw = grid_sizes[i]
    num_patches = ph * pw
    
    plt.subplot(2, 2, i+1)
    img_mask = pca_features_bg[start_idx : start_idx + num_patches].reshape(ph, pw)
    plt.imshow(img_mask, cmap='gray')
    plt.title(f"Background Mask - Img {i+1}")
    plt.axis('off')
    
    start_idx += num_patches
plt.tight_layout()
plt.savefig("dino_pca_3_masks.png", dpi=150)
plt.close()
print("已保存背景掩码图 -> dino_pca_3_masks.png")

# 5. 只针对前景区域再次进行 PCA，以获取丰富的颜色特征
print("Running PCA on Foreground only...")
pca_fg = PCA(n_components=3)
# 找出所有是前景的特征并进行 PCA
fg_features_transformed = pca_fg.fit_transform(total_features[pca_features_fg])

# 对前景的三个主成分进行独立归一化，以便映射到 RGB 的 0-1 范围
for i in range(3):
    min_val = fg_features_transformed[:, i].min()
    max_val = fg_features_transformed[:, i].max()
    fg_features_transformed[:, i] = (fg_features_transformed[:, i] - min_val) / (max_val - min_val + 1e-8)

# 组装最终的 RGB 可视化特征图
pca_features_rgb = pca_features.copy()
# 背景设为纯黑 (0,0,0)
pca_features_rgb[pca_features_bg] = 0
# 前景填入刚刚计算出的 3 维 RGB 色彩
pca_features_rgb[pca_features_fg] = fg_features_transformed

# 绘制最终的彩色特征图
plt.figure(figsize=(10, 10))
start_idx = 0
for i in range(num_images):
    ph, pw = grid_sizes[i]
    num_patches = ph * pw
    
    plt.subplot(2, 2, i+1)
    # 取出当前图片的 RGB 特征并 reshape
    img_rgb = pca_features_rgb[start_idx : start_idx + num_patches].reshape(ph, pw, 3)
    plt.imshow(img_rgb)
    plt.title(f"DINOv2 PCA RGB - Img {i+1}")
    plt.axis('off')
    
    start_idx += num_patches
start_idx += num_patches
# --- 修改这里 ---
plt.tight_layout()
plt.savefig("dino_pca_4_final_rgb.png", dpi=300) # 这个图很重要，可以存清晰点
plt.close()
print("已保存最终RGB特征图 -> dino_pca_4_final_rgb.png")

print("所有可视化步骤已完成，请在当前目录下查看图片！")

print("Visualization completed!")