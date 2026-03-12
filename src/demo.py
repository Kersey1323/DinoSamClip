"""
Demo script for DinoV2 + SAM + CLIP Pipeline
Visualization Style: Overlay segmentation mask and outline on the ORIGINAL image.
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import random

# 添加项目根目录到 sys.path，确保优先导入本地 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")

# 将路径插入到 sys.path 的最前面 (index 0)，强制使用本地代码而不是 site-packages
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 打印路径确认 (调试用)
print(f"Project root added to sys.path: {project_root}")
print(f"Src dir added to sys.path: {src_dir}")

from src.pipeline import DinoSAMClipPipeline
from src.config import DEFAULT_CANDIDATE_CLASSES, ModelConfig

# 定义一些鲜艳的颜色用于不同物体的分割显示 (RGB格式)
COLORS = [
    (255, 0, 0),   # 红
    (0, 255, 0),   # 绿
    (0, 100, 255), # 蓝
    (255, 255, 0), # 黄
    (255, 0, 255), # 品红
    (0, 255, 255), # 青
    (255, 128, 0)  # 橙
]

def create_overlay_on_original(image_array: np.ndarray, detections: list, alpha: float = 0.5) -> np.ndarray:
    """
    核心可视化函数：在原图上叠加半透明 Mask 和轮廓描边。
    如果有多条检测结果，会使用不同颜色区分。
    """
    # 复制一份原图用于绘制
    output_img = image_array.copy()
    
    # 创建一个用于叠加颜色的图层
    overlay_layer = image_array.copy()

    # 1. 绘制半透明色块填充
    shapes_drawn = False
    for i, det in enumerate(detections):
        mask = det["mask"]
        # 循环选择颜色
        color = COLORS[i % len(COLORS)]
        
        # 在 Mask 区域填充纯色
        # 注意：OpenCV 处理图像时通常是 BGR 顺序，但 matplotlib 显示需要 RGB。
        # 这里我们将输入视为 RGB (因为我们在 main 里 convert('RGB') 了)，所以直接用 RGB 颜色。
        overlay_layer[mask] = np.array(color, dtype=np.uint8)
        shapes_drawn = True

    # 将半透明颜色层与原图混合
    if shapes_drawn:
        cv2.addWeighted(overlay_layer, alpha, output_img, 1 - alpha, 0, output_img)

    # 2. 绘制实线轮廓描边 (为了清晰，画在混合图层之上)
    for i, det in enumerate(detections):
        mask = det["mask"]
        color = COLORS[i % len(COLORS)]
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        # 寻找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓，线宽为 2
        cv2.drawContours(output_img, contours, -1, color, thickness=2)

    return output_img

def visualize_intermediate_states(image: Image.Image, prompts: list, all_masks: list, save_path: str):
    """可视化中间状态：DinoV2 提示点 和 SAM 原始 Mask"""
    if not prompts and not all_masks:
        print("没有中间状态数据可显示。")
        return
        
    img_array = np.array(image).copy()

    # 1. 绘制 DinoV2 提示点 (绿点白圈)
    prompt_img = img_array.copy()
    for x, y in prompts:
        cv2.circle(prompt_img, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(prompt_img, (x, y), 6, (255, 255, 255), 2)

    # 2. 准备画布：展示提示点图 + 最多前 3 个原始 Mask
    num_masks = min(len(all_masks), 3)
    fig, axes = plt.subplots(1, 1 + num_masks, figsize=(5 * (1 + num_masks), 5))
    if not isinstance(axes, np.ndarray): axes = [axes]

    axes[0].imshow(prompt_img)
    axes[0].set_title(f"DinoV2 Prompts ({len(prompts)})")
    axes[0].axis("off")

    # 3. 绘制 SAM 原始 Mask (用单一红色叠加显示)
    for i in range(num_masks):
        mask_dict = all_masks[i]
        mask = mask_dict["mask"]
        score = mask_dict.get("iou_score", mask_dict.get("score", 0.0)) # 兼容不同的 key

        overlay = img_array.copy()
        red_color = np.array([255, 0, 0], dtype=np.uint8)
        # 简单的半透明叠加
        overlay[mask] = (overlay[mask] * 0.5 + red_color * 0.5).astype(np.uint8)
        
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"Raw SAM Mask {i+1}\nScore: {score:.2f}")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔍 中间状态分析图已保存至: {save_path}")

def visualize_final_results(image: Image.Image, detections: list, attention_map: np.ndarray, save_path: str):
    """最终结果可视化：原图叠加分割 + 热力图 + 文本摘要"""
    if not detections:
        print("没有检测到物体，无法生成最终结果图。")
        return

    img_array = np.array(image)

    # --- 生成核心叠加叠加图 ---
    # 这里我们传入所有检测结果，create_overlay_on_original 会处理多颜色叠加
    final_overlay_rgb = create_overlay_on_original(img_array, detections, alpha=0.4)

    # --- 创建画布 ---
    fig, axes = plt.subplots(1, 3 if attention_map is not None else 2, figsize=(16, 6))
    if not isinstance(axes, np.ndarray): axes = [axes]

    # 子图 1: 最终分割叠加结果
    axes[0].imshow(final_overlay_rgb)
    axes[0].set_title(f"Final Segmentation Overlay")
    axes[0].axis("off")

    # 子图 2: DinoV2 热力图
    if attention_map is not None:
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("DinoV2 Similarity Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 子图 3: 文本摘要 (列出所有检测到的物体)
    ax = axes[-1]
    ax.axis("off")
    summary_text = "Detection Summary:\n\n"
    for i, det in enumerate(detections):
        summary_text += f"{i+1}. {det['class']}: {det['confidence']:.1%}\n"
        
    ax.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✨ 最终结果图已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="DinoV2 + SAM + CLIP Pipeline Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="Candidate classes (English)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="result_final.png", help="Output path for final result")
    # 增加若干参数用于调节Pipeline行为
    parser.add_argument("--num_prompts", type=int, default=10, help="Max number of prompts from DinoV2")
    parser.add_argument("--conf_thresh", type=float, default=0.15, help="CLIP confidence threshold to keep detection")
    parser.add_argument("--sam_checkpoint", type=str, default=ModelConfig.SAM_CHECKPOINT_PATH, help="Path to SAM checkpoint")
    
    args = parser.parse_args()

    # 加载图像
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        return
    image = Image.open(args.image).convert("RGB")

    # 初始化管道
    pipeline = DinoSAMClipPipeline(
        device=args.device,
        candidate_classes=args.classes,
        sam_checkpoint=args.sam_checkpoint
    )

    print("\n[Running Pipeline...]")
    # 运行检测，传入命令行参数
    results = pipeline.detect_and_classify(
        image, 
        num_prompts=args.num_prompts,
        confidence_threshold=args.conf_thresh
    )

    print(f"检测完成，共找到 {results['num_objects']} 个目标。")

# 1. 保存中间状态图 (DinoV2点 和 SAM原始Mask)
    inter_path = args.output.replace(".png", "_intermediate.png")
    visualize_intermediate_states(
        image, 
        prompts=results.get("prompts", []),     # <--- 加上 , [] 防御 None
        all_masks=results.get("all_masks", []), # <--- 加上 , [] 防御 None
        save_path=inter_path
    )

    # 2. 保存最终结果图 (原图叠加风格)
    visualize_final_results(
        image, 
        detections=results["detections"], 
        attention_map=results.get("attention_map"), 
        save_path=args.output
    )

if __name__ == "__main__":
    main()