"""
Demo script for DinoV2 + SAM + CLIP Pipeline
This script demonstrates how to use the complete pipeline with glowing cutout visualization.
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from PIL import ImageFilter
# 确保可以导入 src 目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DinoSAMClipPipeline
from src.config import DEFAULT_CANDIDATE_CLASSES


def create_glowing_cutout(
    image: Image.Image,
    mask: np.ndarray,
    glow_color: tuple = (0, 150, 255), # 现在使用 RGB 颜色 (如: 浅蓝色)
    thickness: int = 20,
    blur_radius: int = 15
) -> np.ndarray:
    """
    使用 PIL 的 Alpha 通道混合技术，完美保护原图不被高光破坏
    """
    # 1. 转换蒙版为 0/255 的图像格式
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8).convert("L")
    
    # 2. 创建底板 (纯白色)
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    
    # 3. 创建发光层 (扩张蒙版 -> 高斯模糊)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    dilated_mask_img = Image.fromarray(dilated_mask).convert("L")
    
    # 生成带颜色的透明图层
    glow_layer = Image.new("RGBA", image.size, glow_color + (255,))
    blurred_alpha = dilated_mask_img.filter(ImageFilter.GaussianBlur(blur_radius))
    glow_layer.putalpha(blurred_alpha) # 把模糊后的蒙版作为透明度
    
    # 4. 创建原图前景层 (仅保留蒙版内，其余完全透明)
    fg_layer = image.convert("RGBA").copy()
    fg_layer.putalpha(mask_img)
    
    # 5. 按顺序像汉堡一样叠加上去 (底板 -> 发光 -> 原图)
    bg.paste(glow_layer, (0, 0), glow_layer)
    bg.paste(fg_layer, (0, 0), fg_layer)
    
    return np.array(bg.convert("RGB"))
def visualize_results_with_glow(
    image: Image.Image,
    detections: list,
    attention_map: np.ndarray = None,
    save_path: str = "result_glow.png"
):
    if not detections:
        print("没有检测到物体，跳过可视化。")
        return

    # 【关键修复】：按置信度从高到低排序，确保展示的是最确定的结果
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    top_detection = detections[0]
    mask = top_detection["mask"]
    class_name = top_detection["class"]
    confidence = top_detection["confidence"]

    # 生成特效图 (直接返回 RGB)
    glowing_result_rgb = create_glowing_cutout(
        image, mask,
        glow_color=(30, 144, 255), # 道奇蓝
        thickness=20, blur_radius=15
    )

    # 创建画布
    fig, axes = plt.subplots(1, 3 if attention_map is not None else 2, figsize=(16, 6))
    if not isinstance(axes, np.ndarray): axes = [axes]

    axes[0].imshow(glowing_result_rgb)
    axes[0].set_title(f"Glowing Cutout: {class_name}")
    axes[0].axis("off")

    if attention_map is not None:
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("DinoV2 Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    ax = axes[-1]
    ax.axis("off")
    summary_text = f"Top Object: {class_name}\nConfidence: {confidence:.2%}\n"
    if len(top_detection["all_predictions"]) > 1:
        second_place = top_detection["all_predictions"][1]
        summary_text += f"\nRunner-up: {second_place['class']} ({second_place['confidence']:.2%})"
    ax.text(0.1, 0.5, summary_text, fontsize=14, verticalalignment="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
def main():
    parser = argparse.ArgumentParser(description="DinoV2 + SAM + CLIP Pipeline Demo with Glowing Cutout")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (Required)")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="Candidate classes (e.g., dog cat person)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="result_glow.png", help="Output path")
    parser.add_argument("--auto", action="store_true", help="Use automatic mode (SAM + CLIP only)")

    args = parser.parse_args()

    # 加载图像
    if not os.path.exists(args.image):
        print(f"Error: Image path not found: {args.image}")
        return
    image = Image.open(args.image).convert("RGB")
    print(f"已加载图像: {args.image}")

    # 初始化管线
    print("\n初始化管线...")
    pipeline = DinoSAMClipPipeline(
        device=args.device,
        candidate_classes=args.classes
    )

    # 运行推理
    print("\n开始推理 (DinoV2 -> SAM -> CLIP)...")
    if args.auto:
        results = pipeline.detect_and_classify_automatic(image, confidence_threshold=0.1)
    else:
        results = pipeline.detect_and_classify(image, num_prompts=10, confidence_threshold=0.1)

    # 可视化结果 (使用新的发光特效函数)
    print("\n生成发光特效图...")
    visualize_results_with_glow(
        image,
        results["detections"],
        attention_map=results.get("attention_map"),
        save_path=args.output
    )

    # 打印简要信息
    print("\n" + "="*40)
    print(f"检测到 {results['num_objects']} 个物体。")
    if results['num_objects'] > 0:
        top = results['detections'][0]
        print(f"置信度最高: {top['class']} ({top['confidence']:.2%})")
        print("详细可视化结果请查看生成的图片。")
    print("="*40)

if __name__ == "__main__":
    main()