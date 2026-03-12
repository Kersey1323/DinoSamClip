"""
Demo script for DinoV2 + SAM + CLIP Pipeline
This script demonstrates how to use the complete pipeline
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

from pipeline import DinoSAMClipPipeline
from config import DEFAULT_CANDIDATE_CLASSES


def visualize_results(
    image: Image.Image,
    detections: list,
    attention_map: np.ndarray = None,
    save_path: str = "result.png"
):
    """
    Visualize detection results

    Args:
        image: Original PIL Image
        detections: List of detection dictionaries
        attention_map: Optional attention map from DinoV2
        save_path: Path to save the visualization
    """
    # Convert to numpy
    img_array = np.array(image).copy()

    # Define colors for different objects
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]

    # Draw each detection
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]

        # Draw bounding box
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det['class']}: {det['confidence']:.2%}"
        cv2.putText(
            img_array,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # Overlay mask with transparency
        mask = det["mask"]
        overlay = img_array.copy()
        overlay[mask] = color
        img_array = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)

    # Create figure
    fig, axes = plt.subplots(1, 3 if attention_map is not None else 2, figsize=(15, 5))

    # Original image with detections
    axes[0].imshow(img_array)
    axes[0].set_title(f"Detected {len(detections)} Objects")
    axes[0].axis("off")

    # Attention map
    if attention_map is not None:
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("DinoV2 Attention Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

    # Detection summary
    ax = axes[-1]
    ax.axis("off")
    summary_text = "Detection Results:\n\n"
    for i, det in enumerate(detections):
        summary_text += f"{i+1}. {det['class']}: {det['confidence']:.2%}\n"
        if len(det["all_predictions"]) > 1:
            summary_text += f"   Other: {det['all_predictions'][1]['class']} ({det['all_predictions'][1]['confidence']:.2%})\n"
    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Results saved to {save_path}")


def create_sample_image():
    """
    Create a sample test image with multiple objects
    """
    # Create a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240

    # Draw some shapes to simulate objects
    # Red circle (simulating apple)
    cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)

    # Green rectangle (simulating book)
    cv2.rectangle(img, (250, 100), (400, 200), (0, 200, 0), -1)

    # Blue triangle (simulating something else)
    pts = np.array([[450, 100], [550, 200], [350, 200]], np.int32)
    cv2.fillPoly(img, [pts], (255, 0, 0))

    # Yellow ellipse
    cv2.ellipse(img, (200, 350), (60, 40), 0, 0, 360, (0, 255, 255), -1)

    return Image.fromarray(img)


def main():
    """
    Main demo function
    """
    parser = argparse.ArgumentParser(description="DinoV2 + SAM + CLIP Pipeline Demo")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--classes", type=str, nargs="+", help="Candidate classes")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="SAM checkpoint path")
    parser.add_argument("--output", type=str, default="result.png", help="Output path")
    parser.add_argument("--auto", action="store_true", help="Use automatic mode (SAM + CLIP only)")

    args = parser.parse_args()

    # Load or create image
    if args.image and os.path.exists(args.image):
        print(f"Loading image from: {args.image}")
        image = Image.open(args.image).convert("RGB")
    else:
        print("Creating sample test image...")
        image = create_sample_image()
        if args.image:
            image.save("test_input.png")
            print("Saved test image to test_input.png")

    # Define candidate classes
    candidate_classes = args.classes or [
        "apple", "book", "triangle", "circle", "square",
        "red object", "green object", "blue object", "yellow object",
        "person", "car", "dog", "cat"
    ]

    print("\nCandidate classes:", candidate_classes)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = DinoSAMClipPipeline(
        device=args.device,
        candidate_classes=candidate_classes,
        sam_checkpoint=args.sam_checkpoint
    )

    # Run detection
    print("\nRunning detection and classification...")

    if args.auto:
        # Automatic mode (SAM + CLIP only, no DinoV2)
        results = pipeline.detect_and_classify_automatic(
            image,
            candidate_classes=candidate_classes,
            num_points=64,
            min_mask_area=1000,
            confidence_threshold=0.1
        )
    else:
        # Full pipeline (DinoV2 + SAM + CLIP)
        results = pipeline.detect_and_classify(
            image,
            candidate_classes=candidate_classes,
            use_dinov2_prompts=True,
            num_prompts=10,
            attention_threshold=0.4,
            min_mask_area=1000,
            confidence_threshold=0.1
        )

    # Visualize results
    print("\nVisualizing results...")
    visualize_results(
        image,
        results["detections"],
        attention_map=results.get("attention_map"),
        save_path=args.output
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total objects detected: {results['num_objects']}")
    for i, det in enumerate(results["detections"]):
        print(f"  {i+1}. {det['class']}: {det['confidence']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
