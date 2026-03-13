"""
Demo script for DinoV2 + SAM + CLIP Pipeline
Visualization Style: Overlay segmentation mask, outline, and class labels with confidence on the ORIGINAL image.
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import random

# Add project root to sys.path to ensure local src modules are imported first
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
output_dir = os.path.join(src_dir, "results")
# Insert paths at the beginning of sys.path to prioritize local code
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Print paths for confirmation (debugging)
print(f"Project root added to sys.path: {project_root}")
print(f"Src dir added to sys.path: {src_dir}")

from src.pipeline import DinoSAMClipPipeline
from src.config import DEFAULT_CANDIDATE_CLASSES, ModelConfig

# Define distinct colors for segmentation visualization (RGB format)
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
    Core visualization function: Overlays semi-transparent masks, solid outlines, and text labels on the original image.
    Uses different colors for different detections.
    """
    # Copy the original image for drawing
    output_img = image_array.copy()
    
    # Create a layer for color overlay
    overlay_layer = image_array.copy()

    # 1. Draw semi-transparent filled masks
    shapes_drawn = False
    for i, det in enumerate(detections):
        mask = det["mask"]
        # Cycle through colors
        color = COLORS[i % len(COLORS)]
        
        # Fill the mask region with solid color
        # Note: matplotlib displays RGB, and input image_array is RGB, so we use RGB color directly.
        overlay_layer[mask] = np.array(color, dtype=np.uint8)
        shapes_drawn = True

    # Blend the overlay layer with the original image
    if shapes_drawn:
        cv2.addWeighted(overlay_layer, alpha, output_img, 1 - alpha, 0, output_img)

    # 2. Draw solid outlines and text labels (drawn on top of the blended image for clarity)
    for i, det in enumerate(detections):
        mask = det["mask"]
        color = COLORS[i % len(COLORS)]
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours with thickness 2
        cv2.drawContours(output_img, contours, -1, color, thickness=2)

        # --- Add Text Label with Background Box ---
        # Find the largest contour to place the label near its top-left corner
        if contours:
            # Find the bounding box of the largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Prepare label text
            label_text = f"{i+1}. {det['class']} ({det['confidence']:.1%})"
            
            # Calculate text size to draw background box
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw filled rectangle for text background (using the same color as the mask)
            cv2.rectangle(output_img, (x, y - text_h - baseline - 5), (x + text_w, y), color, -1)
            
            # Draw white text on top of the background box
            # Determine text color based on background brightness for readability
            text_color = (255, 255, 255) # Default white
            if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 150: # If background is bright
                text_color = (0, 0, 0) # Use black text

            cv2.putText(output_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    return output_img

def visualize_intermediate_states(image: Image.Image, prompts: list, all_masks: list, save_path: str):
    """Visualizes intermediate states: DinoV2 prompts and raw SAM masks."""
    if not prompts and not all_masks:
        print("No intermediate state data to display.")
        return
        
    img_array = np.array(image).copy()

    # 1. Draw DinoV2 prompts (green points with white outline)
    prompt_img = img_array.copy()
    for x, y in prompts:
        cv2.circle(prompt_img, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(prompt_img, (x, y), 6, (255, 255, 255), 2)

    # 2. Prepare canvas: Show prompt image + up to top 3 raw SAM masks
    num_masks = min(len(all_masks), 3)
    fig, axes = plt.subplots(1, 1 + num_masks, figsize=(5 * (1 + num_masks), 5))
    if not isinstance(axes, np.ndarray): axes = [axes]

    axes[0].imshow(prompt_img)
    axes[0].set_title(f"DinoV2 Prompts ({len(prompts)})")
    axes[0].axis("off")

    # 3. Draw SAM raw masks (overlayed in single red color)
    for i in range(num_masks):
        mask_dict = all_masks[i]
        mask = mask_dict["mask"]
        score = mask_dict.get("iou_score", mask_dict.get("score", 0.0)) # Compatibility for different keys

        overlay = img_array.copy()
        red_color = np.array([255, 0, 0], dtype=np.uint8)
        # Simple semi-transparent overlay
        overlay[mask] = (overlay[mask] * 0.5 + red_color * 0.5).astype(np.uint8)
        
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"Raw SAM Mask {i+1}\nScore: {score:.2f}")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔍 Intermediate state analysis image saved to: {save_path}")

def visualize_final_results(image: Image.Image, detections: list, attention_map: np.ndarray, save_path: str):
    """Final result visualization: Original image with segmentation overlay + heatmap + text summary."""
    if not detections:
        print("No objects detected, cannot generate final result image.")
        return

    img_array = np.array(image)

    # --- Generate core overlay image ---
    # Pass all detections, create_overlay_on_original handles multi-color overlay and labeling
    final_overlay_rgb = create_overlay_on_original(img_array, detections, alpha=0.4)

    # --- Create canvas ---
    fig, axes = plt.subplots(1, 3 if attention_map is not None else 2, figsize=(16, 6))
    if not isinstance(axes, np.ndarray): axes = [axes]

    # Subplot 1: Final segmentation overlay result
    axes[0].imshow(final_overlay_rgb)
    axes[0].set_title(f"Final Segmentation Overlay")
    axes[0].axis("off")

    # Subplot 2: DinoV2 heatmap
    if attention_map is not None:
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("DinoV2 Similarity Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Subplot 3: Text summary (list all detected objects)
    ax = axes[-1]
    ax.axis("off")
    summary_text = "Detection Summary:\n\n"
    for i, det in enumerate(detections):
        summary_text += f"{i+1}. {det['class']}: {det['confidence']:.1%}\n"
        
    ax.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✨ Final result image saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="DinoV2 + SAM + CLIP Pipeline Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="Candidate classes (English)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default=f"{output_dir}/result_final.png", help="Output path for final result")
    # Add parameters to adjust Pipeline behavior
    parser.add_argument("--num_prompts", type=int, default=10, help="Max number of prompts from DinoV2")
    parser.add_argument("--conf_thresh", type=float, default=0.15, help="CLIP confidence threshold to keep detection")
    parser.add_argument("--sam_checkpoint", type=str, default=ModelConfig.SAM_CHECKPOINT_PATH, help="Path to SAM checkpoint")
    parser.add_argument("--auto", action="store_true", help="使用纯 SAM 的自动网格模式，跳过 DinoV2")
    args = parser.parse_args()

    
  
        
    import time

    # Generate timestamp for the output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    print(f"Created output directory: {timestamp_dir}")

    # Update output path to use the timestamped directory
    base_filename = os.path.basename(args.output)
    final_output_path = os.path.join(timestamp_dir, base_filename)
    
    # Load image
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        return
    image = Image.open(args.image).convert("RGB")

    # Initialize pipeline
    pipeline = DinoSAMClipPipeline(
        device=args.device,
        candidate_classes=args.classes,
        sam_checkpoint=args.sam_checkpoint
    )

    print("\n[Running Pipeline...]")
    # Run detection, pass command line arguments
    if args.auto:
            print("\n[启用 Auto 模式：无视 DinoV2，全图撒网扫描...]")
            results = pipeline.detect_and_classify_automatic(
                image, 
                confidence_threshold=args.conf_thresh
            )
    else:
            results = pipeline.detect_and_classify(
                image, 
                num_prompts=args.num_prompts,
                confidence_threshold=args.conf_thresh,
                output_dir=timestamp_dir
            )
    print(f"Detection complete, found {results['num_objects']} targets.")

    # 1. Save intermediate state image (DinoV2 points and SAM raw masks)
    inter_path = final_output_path.replace(".png", "_intermediate.png")
    visualize_intermediate_states(
        image, 
        prompts=results.get("prompts", []),     # <--- Add , [] to defend against None
        all_masks=results.get("all_masks", []), # <--- Add , [] to defend against None
        save_path=inter_path
    )

    # 2. Save final result image (original image overlay style with labels)
    visualize_final_results(
        image, 
        detections=results["detections"], 
        attention_map=results.get("attention_map"), 
        save_path=final_output_path
    )

if __name__ == "__main__":
    main()