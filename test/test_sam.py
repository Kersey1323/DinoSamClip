import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image

# 确保能导入 src 下的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.sam_segmenter import SAMSegmenter
from src.config import ModelConfig

# ==========================================
# Meta 官方的画图辅助函数 (完美复刻原版风格)
# ==========================================
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # 官方的经典科技蓝
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def download_official_test_image():
    
    image = Image.open('/media/dell/新加卷1/CV/DinoSamClip/src/imgs/truck.jpg').convert("RGB")
    return np.array(image)

# ==========================================
# 主测试逻辑
# ==========================================
def main():
    print("=" * 60)
    print("🚀 SAM Model 官方用例可视化测试启动")
    print("=" * 60)

    # 1. 加载模型
    sam = SAMSegmenter(
        model_type=ModelConfig.SAM_MODEL_TYPE,
        checkpoint_path=ModelConfig.SAM_CHECKPOINT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. 获取并设置图像
    img_array = download_official_test_image()
    sam.set_image(img_array)

    # 创建画板 (1行2列)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # --- 测试 1：单点提示 (官方样例：点在皮卡车的车窗上) ---
    print("\n[测试 1] 单点提示分割 (Target: Truck Window)...")
    point_coords = [(500, 375)]
    point_labels = [1]

    # 调用你的封装方法
    point_results = sam.segment_from_points(points=point_coords, labels=point_labels, multimask_output=True)

    # SAM 默认返回 3 个不同颗粒度的 mask，我们选得分最高的一个
    best_point_result = max(point_results, key=lambda x: x.get("score", x.get("iou_score", 0)))
    print(f"✅ 点提示分割完成，最高得分: {best_point_result['score']:.4f}")

    axes[0].imshow(img_array)
    show_mask(best_point_result["mask"], axes[0])
    show_points(np.array(point_coords), np.array(point_labels), axes[0])
    axes[0].set_title(f"Point Prompt (Window)\nScore: {best_point_result['score']:.3f}", fontsize=18)
    axes[0].axis('off')

    # --- 测试 2：边界框提示 (官方样例：框住皮卡车的整个前车头) ---
    print("\n[测试 2] 边界框提示分割 (Target: Front of the Truck)...")
    box_coords = (425, 600, 700, 875)

    box_results = sam.segment_from_box(box=box_coords, multimask_output=False)
    best_box_result = box_results[0]
    print(f"✅ 框提示分割完成，最高得分: {best_box_result['score']:.4f}")

    axes[1].imshow(img_array)
    show_mask(best_box_result["mask"], axes[1])
    show_box(box_coords, axes[1])
    axes[1].set_title(f"Box Prompt (Front Bumper)\nScore: {best_box_result['score']:.3f}", fontsize=18)
    axes[1].axis('off')

    # 保存并展示结果
    save_path = "sam_official_test_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✨ 可视化结果已保存至: {save_path}")
    print("请在文件管理器中打开该图片，查看官方级别的唯美分割效果！")

if __name__ == "__main__":
    main()