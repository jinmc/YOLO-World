"""Visualize YOLO-format labels on images.

Usage:
    python tools/visualize_labels.py \
        --image-dir data/images/experiments/ratio_real_only \
        --label-dir data/labels/experiments/ratio_real_only \
        --output-dir data/viz/experiments/ratio_real_only \
        --splits train \
        --num-samples 5
"""
import os
import os.path as osp
import argparse
import random

import cv2
import numpy as np


# Distinct colors for up to 20 classes (BGR)
COLORS = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 128, 255),    # orange
    (128, 0, 255),    # purple
    (0, 255, 128),    # spring green
    (255, 128, 0),    # sky blue
    (128, 255, 0),    # chartreuse
    (255, 0, 128),    # rose
    (0, 128, 128),    # teal
    (128, 128, 0),    # olive
    (128, 0, 128),    # purple
    (64, 224, 208),   # turquoise
    (0, 69, 255),     # orange red
    (180, 105, 255),  # hot pink
    (19, 69, 139),    # saddle brown
    (238, 130, 238),  # violet
]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize YOLO labels')
    parser.add_argument('--image-dir', required=True,
                        help='root dir of images (e.g. data/images/experiments/ratio_real_only)')
    parser.add_argument('--label-dir', required=True,
                        help='root dir of labels (e.g. data/labels/experiments/ratio_real_only)')
    parser.add_argument('--output-dir', default=None,
                        help='output dir for visualizations (default: data/viz/...)')
    parser.add_argument('--splits', nargs='+', default=['train'],
                        help='splits to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='number of random samples per class (0 = all)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_classes(label_dir):
    classes_file = osp.join(label_dir, 'classes.txt')
    if osp.exists(classes_file):
        with open(classes_file) as f:
            return [line.strip() for line in f if line.strip()]
    return []


def draw_yolo_labels(image, labels, classes, img_w, img_h):
    """Draw YOLO bounding boxes on image."""
    for line in labels:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert normalized YOLO -> pixel coords
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)

        color = COLORS[cls_id % len(COLORS)]
        cls_name = classes[cls_id] if cls_id < len(classes) else f"cls_{cls_id}"

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{cls_name}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def main():
    args = parse_args()
    random.seed(args.seed)

    classes = load_classes(args.label_dir)
    if not classes:
        print("No classes.txt found!")
        return

    print(f"Classes ({len(classes)}): {classes}")

    if args.output_dir is None:
        args.output_dir = args.label_dir.replace('data/labels/', 'data/viz/')

    total_saved = 0

    for split in args.splits:
        split_img_dir = osp.join(args.image_dir, split)
        split_lbl_dir = osp.join(args.label_dir, split)

        if not osp.exists(split_img_dir):
            print(f"Skip split '{split}': image dir not found")
            continue

        for cls_name in sorted(os.listdir(split_img_dir)):
            cls_img_dir = osp.join(split_img_dir, cls_name)
            cls_lbl_dir = osp.join(split_lbl_dir, cls_name)

            if not osp.isdir(cls_img_dir):
                continue

            # Collect images that have labels
            img_files = [
                f for f in os.listdir(cls_img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]

            # Filter to those with non-empty labels
            labeled = []
            for f in img_files:
                lbl_path = osp.join(cls_lbl_dir, osp.splitext(f)[0] + '.txt')
                if osp.exists(lbl_path) and osp.getsize(lbl_path) > 0:
                    labeled.append(f)

            if not labeled:
                print(f"  [{split}] {cls_name}: no labeled images")
                continue

            # Sample
            if args.num_samples > 0 and len(labeled) > args.num_samples:
                sampled = random.sample(labeled, args.num_samples)
            else:
                sampled = labeled

            out_dir = osp.join(args.output_dir, split, cls_name)
            os.makedirs(out_dir, exist_ok=True)

            for img_file in sampled:
                img_path = osp.join(cls_img_dir, img_file)
                lbl_path = osp.join(cls_lbl_dir, osp.splitext(img_file)[0] + '.txt')

                image = cv2.imread(img_path)
                if image is None:
                    continue
                img_h, img_w = image.shape[:2]

                with open(lbl_path) as f:
                    labels = f.readlines()

                image = draw_yolo_labels(image, labels, classes, img_w, img_h)

                out_path = osp.join(out_dir, img_file)
                cv2.imwrite(out_path, image)
                total_saved += 1

            print(f"  [{split}] {cls_name}: saved {len(sampled)} visualizations "
                  f"(from {len(labeled)} labeled / {len(img_files)} total)")

    print(f"\nDone! {total_saved} images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
