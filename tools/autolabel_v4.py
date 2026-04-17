# Auto-labeling v4: 100% coverage + correct "empty" handling
# Every food image MUST have at least one detection.
# "empty" class images get EMPTY label files (no bboxes).
#
# Strategy:
#   1. SKIP_CLASSES (e.g. "empty") → write empty .txt (background)
#   2. Multi-prompt inference for food classes
#   3. If no detection → retry with progressively lower thresholds
#   4. If still nothing → use full-image bbox as fallback

import os
import os.path as osp
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


# ====================================================================
# SKIP CLASSES: these get empty label files (no bboxes)
# e.g., "empty" = empty tray with no food → background image
# ====================================================================
SKIP_CLASSES = {"empty"}

# ====================================================================
# PROMPT MAP: class_name -> list of text prompts to try
# More prompts = higher recall. Put best prompts first.
# ====================================================================
PROMPT_MAP = {
    "bacon":           ["bacon strips", "bacon", "cooked bacon", "pork", "meat strips"],
    "broccoli":        ["broccoli", "green vegetable", "vegetable"],
    "brussels-sprout": ["brussels sprout", "sprouts", "small vegetables"],
    "cauliflower":     ["cauliflower", "white vegetable", "vegetable", "food"],
    "chicken-nuggets": ["chicken nuggets", "nuggets", "fried chicken pieces", "fried food"],
    "chicken-wings":   ["chicken wings", "chicken wing", "fried chicken", "chicken", "wings", "poultry", "meat"],
    "cookies":         ["cookies", "cookie", "biscuit", "baked goods"],
    "cut-potatoes":    ["diced potatoes", "chopped potatoes", "cut potatoes", "vegetables", "potato pieces"],
    "french-fries":    ["french fries", "chips", "fries", "fried potatoes", "potato sticks"],
    "pizza":           ["pizza", "pizza slice", "pizza pie", "flatbread", "cheese pizza", "food"],
    "salmon":          ["fish", "cooked fish", "salmon", "fish fillet", "seafood"],
    "steak":           ["food on tray", "beef", "meat", "steak", "cooked meat", "beef steak"],
    "tots":            ["tots", "tater tots", "potato tots", "fried food", "small fried food", "potato balls"],
    "whole-chicken":   ["whole chicken", "roast chicken", "chicken", "roasted bird", "poultry"],
    # 17cls_sub extras
    "black_dough_cookies":  ["cookies", "dark cookies", "cookie", "biscuit", "baked goods"],
    "brown_dough_cookies":  ["cookies", "brown cookies", "cookie", "biscuit", "baked goods"],
    "white_dough_cookies":  ["cookies", "white cookies", "cookie", "biscuit", "baked goods"],
}

# Per-class primary threshold (first attempt)
THRESHOLD_MAP = {
    "salmon":        0.01,
    "steak":         0.01,
    "bacon":         0.02,
    "french-fries":  0.02,
    "cut-potatoes":  0.02,
    "chicken-wings": 0.01,
    "cauliflower":   0.02,
    "tots":          0.02,
    "pizza":         0.02,
    "chicken-nuggets": 0.03,
}

# Fallback thresholds: tried in order if primary gives no detection
FALLBACK_THRESHOLDS = [0.005, 0.003, 0.001]


def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO-World Auto-Labeling v4: 100% coverage + empty class handling')
    parser.add_argument('--config',
                        default='configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
    parser.add_argument('--checkpoint',
                        default='weights/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth')
    parser.add_argument('--data-root',
                        default='data/images/experiments/ratio_real_only')
    parser.add_argument('--output-dir',
                        default='data/labels_v4/experiments/ratio_real_only')
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--threshold', default=0.05, type=float,
                        help='default confidence threshold')
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--nms-iou', default=0.5, type=float,
                        help='NMS IoU threshold for merging multi-prompt results')
    parser.add_argument('--no-fullimg-fallback', action='store_true',
                        help='Disable full-image bbox fallback')
    return parser.parse_args()


def xyxy_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / img_w))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / img_h))
    w = max(0.0, min(1.0, (x2 - x1) / img_w))
    h = max(0.0, min(1.0, (y2 - y1) / img_h))
    return cx, cy, w, h


def nms_numpy(bboxes, scores, iou_threshold=0.5):
    """Simple NMS on numpy arrays. bboxes: (N,4) xyxy, scores: (N,)"""
    if len(bboxes) == 0:
        return np.array([], dtype=int)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def inference_single(model, image_rgb, texts, test_pipeline,
                     score_thr=0.05, max_dets=100):
    """Run inference with a single text prompt."""
    data_info = dict(img=image_rgb, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(
        inputs=data_info['inputs'].unsqueeze(0),
        data_samples=[data_info['data_samples']]
    )
    with torch.no_grad():
        output = model.test_step(data_batch)[0]

    pred = output.pred_instances
    pred = pred[pred.scores.float() > score_thr]
    if len(pred.scores) > max_dets:
        indices = pred.scores.float().topk(max_dets)[1]
        pred = pred[indices]

    pred = pred.cpu().numpy()
    return pred['bboxes'], pred['scores']


def inference_multi_prompt(model, image_rgb, prompts, test_pipeline,
                           score_thr=0.05, max_dets=100, nms_iou=0.5):
    """Try multiple prompts and merge results with NMS."""
    all_bboxes = []
    all_scores = []

    for prompt in prompts:
        texts = [[prompt], [' ']]
        model.reparameterize(texts)

        bboxes, scores = inference_single(
            model, image_rgb, texts, test_pipeline,
            score_thr=score_thr, max_dets=max_dets
        )
        if bboxes is not None and len(bboxes) > 0:
            all_bboxes.append(bboxes)
            all_scores.append(scores)

    if not all_bboxes:
        return np.array([]), np.array([])

    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # NMS to remove duplicates from different prompts
    keep = nms_numpy(all_bboxes, all_scores, iou_threshold=nms_iou)
    return all_bboxes[keep], all_scores[keep]


def label_image_with_fallback(model, image_path, prompts, test_pipeline,
                              primary_thr, fallback_thresholds,
                              max_dets=100, nms_iou=0.5,
                              use_fullimg_fallback=True):
    """
    Guarantee at least 1 detection per image:
      1) Try multi-prompt at primary_thr
      2) If nothing → retry at each fallback threshold (progressively lower)
      3) If still nothing → full-image bbox
    Returns: bboxes (N,4), scores (N,), (img_w, img_h), fallback_level
      fallback_level: 0=primary, 1/2/3=fallback threshold, 99=full-image
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, -1
    img_h, img_w = image.shape[:2]
    image_rgb = image[:, :, [2, 1, 0]]

    # 1) Primary threshold
    bboxes, scores = inference_multi_prompt(
        model, image_rgb, prompts, test_pipeline,
        score_thr=primary_thr, max_dets=max_dets, nms_iou=nms_iou
    )
    if len(bboxes) > 0:
        return bboxes, scores, (img_w, img_h), 0

    # 2) Fallback thresholds
    for fb_level, fb_thr in enumerate(fallback_thresholds, start=1):
        if fb_thr >= primary_thr:
            continue  # skip if not actually lower
        bboxes, scores = inference_multi_prompt(
            model, image_rgb, prompts, test_pipeline,
            score_thr=fb_thr, max_dets=max_dets, nms_iou=nms_iou
        )
        if len(bboxes) > 0:
            return bboxes, scores, (img_w, img_h), fb_level

    # 3) Full-image fallback: place one bbox covering the whole image
    if use_fullimg_fallback:
        bboxes = np.array([[0, 0, img_w, img_h]], dtype=np.float32)
        scores = np.array([0.001], dtype=np.float32)
        return bboxes, scores, (img_w, img_h), 99

    return np.array([]), np.array([]), (img_w, img_h), -1


def main():
    args = parse_args()

    # Discover classes
    classes = sorted([
        d for d in os.listdir(osp.join(args.data_root, args.splits[0]))
        if osp.isdir(osp.join(args.data_root, args.splits[0], d))
    ])
    class_to_id = {cls: i for i, cls in enumerate(classes)}

    print(f"Found {len(classes)} classes:")
    for i, cls in enumerate(classes):
        if cls in SKIP_CLASSES:
            print(f"  {i}: {cls:<25} ** SKIP (background, empty labels) **")
        else:
            prompts = PROMPT_MAP.get(cls, [cls.replace('-', ' ').replace('_', ' ')])
            thr = THRESHOLD_MAP.get(cls, args.threshold)
            print(f"  {i}: {cls:<25} thr={thr:.3f}  prompts={prompts}")
    print(f"\nSkip classes: {SKIP_CLASSES}")
    print(f"Fallback thresholds: {FALLBACK_THRESHOLDS}")
    print(f"Full-image fallback: {'OFF' if args.no_fullimg_fallback else 'ON'}")

    # Load model
    print(f"\nLoading model...")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join('./work_dirs', 'autolabel_v4')
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint,
                          device=args.device, palette='coco')

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, 'classes.txt'), 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")

    with open(osp.join(args.output_dir, 'dataset.yaml'), 'w') as f:
        f.write(f"path: {osp.abspath(args.data_root)}\n")
        f.write(f"train: train\nval: val\ntest: test\n\n")
        f.write(f"nc: {len(classes)}\nnames:\n")
        for i, cls in enumerate(classes):
            f.write(f"  {i}: {cls}\n")

    # Process
    grand_total_images = 0
    grand_total_dets = 0
    grand_fallback_stats = {0: 0}  # level -> count
    for fb_i in range(len(FALLBACK_THRESHOLDS)):
        grand_fallback_stats[fb_i + 1] = 0
    grand_fallback_stats[99] = 0
    grand_fallback_stats[-1] = 0

    for split in args.splits:
        split_dir = osp.join(args.data_root, split)
        if not osp.exists(split_dir):
            print(f"\nSkipping split '{split}': not found")
            continue

        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        split_images = 0
        split_dets = 0

        for cls_name in classes:
            cls_dir = osp.join(split_dir, cls_name)
            if not osp.exists(cls_dir):
                continue

            cls_id = class_to_id[cls_name]
            prompts = PROMPT_MAP.get(
                cls_name,
                [cls_name.replace('-', ' ').replace('_', ' ')]
            )
            primary_thr = THRESHOLD_MAP.get(cls_name, args.threshold)

            img_files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            if not img_files:
                continue

            # --- SKIP CLASSES: write empty label files (background) ---
            if cls_name in SKIP_CLASSES:
                for img_file in img_files:
                    img_path = osp.join(cls_dir, img_file)
                    rel_path = osp.relpath(img_path, args.data_root)
                    label_path = osp.join(
                        args.output_dir,
                        osp.splitext(rel_path)[0] + '.txt'
                    )
                    os.makedirs(osp.dirname(label_path), exist_ok=True)
                    with open(label_path, 'w') as f:
                        pass  # empty file = no objects (background)
                print(f"  {cls_name:<20} {len(img_files)} imgs, "
                      f"SKIPPED (background, empty labels)")
                split_images += len(img_files)
                continue

            cls_dets = 0
            cls_labeled = 0
            cls_fallback_stats = {k: 0 for k in grand_fallback_stats}

            for img_file in tqdm(img_files,
                                 desc=f"  [{split}] {cls_name} (thr={primary_thr})",
                                 leave=False):
                img_path = osp.join(cls_dir, img_file)

                bboxes, scores, size, fb_level = label_image_with_fallback(
                    model, img_path, prompts, test_pipeline,
                    primary_thr=primary_thr,
                    fallback_thresholds=FALLBACK_THRESHOLDS,
                    max_dets=args.topk,
                    nms_iou=args.nms_iou,
                    use_fullimg_fallback=not args.no_fullimg_fallback,
                )

                if size is None:
                    continue

                img_w, img_h = size
                cls_fallback_stats[fb_level] = cls_fallback_stats.get(fb_level, 0) + 1

                # Save label
                rel_path = osp.relpath(img_path, args.data_root)
                label_path = osp.join(
                    args.output_dir,
                    osp.splitext(rel_path)[0] + '.txt'
                )
                os.makedirs(osp.dirname(label_path), exist_ok=True)

                with open(label_path, 'w') as f:
                    if bboxes is not None and len(bboxes) > 0:
                        for bbox, score in zip(bboxes, scores):
                            cx, cy, w, h = xyxy_to_yolo(bbox, img_w, img_h)
                            f.write(f"{cls_id} {cx:.6f} {cy:.6f} "
                                    f"{w:.6f} {h:.6f}\n")

                n = len(bboxes) if bboxes is not None else 0
                cls_dets += n
                if n > 0:
                    cls_labeled += 1

            pct = cls_labeled / len(img_files) * 100
            # Build fallback info string
            fb_parts = []
            if cls_fallback_stats.get(0, 0) > 0:
                fb_parts.append(f"primary={cls_fallback_stats[0]}")
            for fb_i in range(len(FALLBACK_THRESHOLDS)):
                cnt = cls_fallback_stats.get(fb_i + 1, 0)
                if cnt > 0:
                    fb_parts.append(f"fb@{FALLBACK_THRESHOLDS[fb_i]}={cnt}")
            if cls_fallback_stats.get(99, 0) > 0:
                fb_parts.append(f"full-img={cls_fallback_stats[99]}")
            if cls_fallback_stats.get(-1, 0) > 0:
                fb_parts.append(f"FAILED={cls_fallback_stats[-1]}")
            fb_info = " | " + ", ".join(fb_parts) if fb_parts else ""

            print(f"  {cls_name:<20} {len(img_files)} imgs, "
                  f"{cls_labeled} labeled ({pct:.0f}%), "
                  f"{cls_dets} dets{fb_info}")
            split_images += len(img_files)
            split_dets += cls_dets

            # Accumulate
            for k, v in cls_fallback_stats.items():
                grand_fallback_stats[k] = grand_fallback_stats.get(k, 0) + v

        print(f"  >> {split} total: {split_images} images, "
              f"{split_dets} detections")
        grand_total_images += split_images
        grand_total_dets += split_dets

    print(f"\n{'='*60}")
    print(f"Auto-labeling v4 Complete!")
    print(f"{'='*60}")
    print(f"Total images : {grand_total_images}")
    print(f"Total dets   : {grand_total_dets}")
    print(f"Skip classes : {SKIP_CLASSES}")
    print(f"Labels saved : {args.output_dir}")
    print(f"\nFallback statistics:")
    print(f"  Primary threshold    : {grand_fallback_stats.get(0, 0)}")
    for fb_i, fb_thr in enumerate(FALLBACK_THRESHOLDS):
        print(f"  Fallback @{fb_thr:<10} : {grand_fallback_stats.get(fb_i + 1, 0)}")
    print(f"  Full-image fallback  : {grand_fallback_stats.get(99, 0)}")
    if grand_fallback_stats.get(-1, 0) > 0:
        print(f"  FAILED (no label)    : {grand_fallback_stats.get(-1, 0)}")


if __name__ == '__main__':
    main()
