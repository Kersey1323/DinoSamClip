"""
Batch Inference Script for DinoV2 + SAM + CLIP Pipeline
Process all images in a directory and save final visualization results.
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import time
from tqdm import tqdm
import torch
# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline import DinoSAMClipPipeline
from src.config import DEFAULT_CANDIDATE_CLASSES, ModelConfig

# Define colors (same as demo.py)
COLORS = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 100, 255), # Blue
    (255, 255, 0), # Yellow
    (255, 0, 255), # Magenta
    (0, 255, 255), # Cyan
    (255, 128, 0)  # Orange
]

def create_overlay_on_original(image_array: np.ndarray, detections: list, alpha: float = 0.5) -> np.ndarray:
    """
    Core visualization function: Overlays semi-transparent masks, solid outlines, and text labels.
    """
    output_img = image_array.copy()
    overlay_layer = image_array.copy()

    shapes_drawn = False
    for i, det in enumerate(detections):
        mask = det["mask"]
        color = COLORS[i % len(COLORS)]
        overlay_layer[mask] = np.array(color, dtype=np.uint8)
        shapes_drawn = True

    if shapes_drawn:
        cv2.addWeighted(overlay_layer, alpha, output_img, 1 - alpha, 0, output_img)

    for i, det in enumerate(detections):
        mask = det["mask"]
        color = COLORS[i % len(COLORS)]
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw thick contour
        cv2.drawContours(output_img, contours, -1, color, thickness=3)
        # Draw thin white inner contour for contrast
        cv2.drawContours(output_img, contours, -1, (255, 255, 255), thickness=1)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            label_text = f"{i+1}. {det['class']} ({det['confidence']:.1%})"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Text background
            cv2.rectangle(output_img, (x, y - text_h - baseline - 5), (x + text_w, y), color, -1)
            
            # Text color
            text_color = (255, 255, 255)
            if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 150:
                text_color = (0, 0, 0)

            cv2.putText(output_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    return output_img

def save_final_result(image: Image.Image, detections: list, attention_map: np.ndarray, save_path: str):
    """
    Save the final result image with overlay, heatmap and summary.
    """
    img_array = np.array(image)
    final_overlay_rgb = create_overlay_on_original(img_array, detections, alpha=0.4)

    # Create figure
    fig, axes = plt.subplots(1, 3 if attention_map is not None else 2, figsize=(16, 6))
    if not isinstance(axes, np.ndarray): axes = [axes]

    # 1. Overlay
    axes[0].imshow(final_overlay_rgb)
    axes[0].set_title(f"Final Segmentation Overlay")
    axes[0].axis("off")

    # 2. Heatmap
    if attention_map is not None:
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("DinoV2 Similarity Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Summary
    ax = axes[-1]
    ax.axis("off")
    summary_text = "Detection Summary:\n\n"
    if not detections:
        summary_text += "No objects detected."
    else:
        for i, det in enumerate(detections):
            summary_text += f"{i+1}. {det['class']}: {det['confidence']:.1%}\n"
    
    ax.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Batch Inference for DinoSAMClip")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="Candidate classes")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_prompts", type=int, default=10, help="Max prompts from DinoV2")
    parser.add_argument("--conf_thresh", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--sam_checkpoint", type=str, default=ModelConfig.SAM_CHECKPOINT_PATH, help="Path to SAM checkpoint")
    parser.add_argument("--auto", action="store_true", help="Use Auto SAM mode (no DinoV2)")
    
    # 🌟 新增参数：接收你微调好的权重文件路径
    parser.add_argument("--dino_checkpoint", type=str, default=None, help="Path to your fine-tuned DinoV2 .pth file")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    print("Initializing Pipeline...")
    pipeline = DinoSAMClipPipeline(
        device=args.device,
        candidate_classes=args.classes,
        sam_checkpoint=args.sam_checkpoint
    )

    # 🌟 核心修改点：如果你传入了微调权重，就在这里“偷天换日”！
    if args.dino_checkpoint and not args.auto:
        print(f"\n[🔥 检测到微调权重] 正在将基础 DinoV2 替换为: {args.dino_checkpoint}")
        try:
            # 找到你 Pipeline 里的 dinov2 模型 (也就是 AutoModel 本体)
            # 因为你在 DinoV2Extractor 里把模型存为了 self.model
            dino_model = pipeline.dinov2.model 
            
            # 读取本地微调的权重
            finetuned_state_dict = torch.load(args.dino_checkpoint, map_location=args.device)
            
            # 强行灌入微调好的骨干网络特征！
            dino_model.load_state_dict(finetuned_state_dict)
            dino_model.eval()
            print("✅ 微调版 DinoV2 权重加载成功！热力图精度即将暴增！")
        except Exception as e:
            print(f"❌ 加载微调权重失败，将退回使用官方默认权重。错误信息: {e}")

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_exts)]
    image_files.sort()

    print(f"\nFound {len(image_files)} images in {args.input_dir}")

    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(args.input_dir, img_file)
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 处理你在代码里提到的关于 output_dir 的报错问题
            # 我注意到你给 detect_and_classify 传了 output_dir 参数，
            # 但是原本的 pipeline.py 里面很可能没有这个参数！
            # 安全起见，如果在报错，请把 output_dir=tmp_debug_dir 去掉，
            # 我们只需要拿到 results 即可。
            
            if args.auto:
                results = pipeline.detect_and_classify_automatic(
                    image, 
                    confidence_threshold=args.conf_thresh
                )
            else:
                results = pipeline.detect_and_classify(
                    image, 
                    num_prompts=args.num_prompts,
                    confidence_threshold=args.conf_thresh
                    # ⚠️ 删除了 output_dir=tmp_debug_dir，防止 Pipeline 报错找不到参数
                )
            
            output_filename = os.path.splitext(img_file)[0] + "_result.png"
            save_path = os.path.join(args.output_dir, output_filename)
            
            save_final_result(
                image,
                detections=results["detections"],
                attention_map=results.get("attention_map"),
                save_path=save_path
            )
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()
