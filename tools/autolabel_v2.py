# Enhanced Auto-labeling with multi-prompt + lower threshold fallback
# Strategy:
#   1. For each class, try MULTIPLE text prompts (not just class name)
#   2. Use a two-pass approach: primary prompt → fallback prompts
#   3. Union of all detections across prompts
#   4. Lower threshold for difficult classes

import os
import os.path as osp
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


# ====================================================================
# PROMPT MAP: class_name -> list of text prompts to try
# Each prompt is tried independently, and results are merged (union).
# Put the best/most-general prompt first.
# ====================================================================
PROMPT_MAP = {
    "bacon":           ["bacon strips", "bacon", "cooked bacon", "pork"],
    "broccoli":        ["broccoli", "green vegetable"],
    "brussels-sprout": ["brussels sprout", "sprouts"],
    "cauliflower":     ["cauliflower", "white vegetable"],
    "chicken-nuggets": ["chicken nuggets", "nuggets", "fried chicken pieces"],
    "chicken-wings":   ["chicken wings", "chicken wing", "fried chicken"],
    "cookies":         ["cookies", "cookie", "biscuit"],
    "cut-potatoes":    ["diced potatoes", "chopped potatoes", "cut potatoes", "vegetables"],
    "empty":           ["empty tray", "empty plate", "empty pan"],
    "french-fries":    ["french fries", "chips", "fries", "fried potatoes"],
    "pizza":           ["pizza", "pizza slice"],
    "salmon":          ["fish", "cooked fish", "salmon"],
    "steak":           ["food on tray", "beef", "meat", "steak"],
    "tots":            ["tots", "tater tots", "potato tots", "fried food"],
    "whole-chicken":   ["whole chicken", "roast chicken", "chicken"],
    # 17cls_sub extras
    "black_dough_cookies":  ["cookies", "dark cookies", "cookie", "biscuit"],
    "brown_dough_cookies":  ["cookies", "brown cookies", "cookie", "biscuit"],
    "white_dough_cookies":  ["cookies", "white cookies", "cookie", "biscuit"],
}

# Per-class threshold overrides (lower for hard classes)
THRESHOLD_MAP = {
    "salmon":       0.01,
    "steak":        0.01,
    "bacon":        0.02,
    "french-fries": 0.02,
    "cut-potatoes": 0.02,
    "empty":        0.02,
    "chicken-wings":0.02,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Enhanced YOLO-World Auto-Labeling with multi-prompt')
    parser.add_argument('--config',
                        default='configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
    parser.add_argument('--checkpoint',
                        default='weights/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth')
    parser.add_argument('--data-root',
                        default='data/images/experiments/ratio_real_only')
    parser.add_argument('--output-dir',
                        default='data/labels_v2/experiments/ratio_real_only')
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--threshold', default=0.05, type=float,
                        help='default confidence threshold')
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--nms-iou', default=0.5, type=float,
                        help='NMS IoU threshold for merging multi-prompt results')
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


def inference_multi_prompt(model, image_path, prompts, test_pipeline,
                           score_thr=0.05, max_dets=100, nms_iou=0.5):
    """Try multiple prompts and merge results with NMS."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None
    img_h, img_w = image.shape[:2]
    image_rgb = image[:, :, [2, 1, 0]]

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
        return np.array([]), np.array([]), (img_w, img_h)

    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # NMS to remove duplicates from different prompts
    keep = nms_numpy(all_bboxes, all_scores, iou_threshold=nms_iou)
    return all_bboxes[keep], all_scores[keep], (img_w, img_h)


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
        prompts = PROMPT_MAP.get(cls, [cls.replace('-', ' ').replace('_', ' ')])
        thr = THRESHOLD_MAP.get(cls, args.threshold)
        print(f"  {i}: {cls:<25} thr={thr:.3f}  prompts={prompts}")

    # Load model
    print(f"\nLoading model...")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join('./work_dirs', 'autolabel_v2')
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
            thr = THRESHOLD_MAP.get(cls_name, args.threshold)

            img_files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            if not img_files:
                continue

            cls_dets = 0
            cls_labeled = 0
            for img_file in tqdm(img_files,
                                 desc=f"  [{split}] {cls_name} (thr={thr})",
                                 leave=False):
                img_path = osp.join(cls_dir, img_file)

                bboxes, scores, (img_w, img_h) = inference_multi_prompt(
                    model, img_path, prompts, test_pipeline,
                    score_thr=thr, max_dets=args.topk,
                    nms_iou=args.nms_iou
                )

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
            print(f"  {cls_name:<20} {len(img_files)} imgs, "
                  f"{cls_labeled} labeled ({pct:.0f}%), "
                  f"{cls_dets} dets")
            split_images += len(img_files)
            split_dets += cls_dets

        print(f"  >> {split} total: {split_images} images, "
              f"{split_dets} detections")
        grand_total_images += split_images
        grand_total_dets += split_dets

    print(f"\n{'='*60}")
    print(f"Auto-labeling v2 Complete!")
    print(f"{'='*60}")
    print(f"Total images : {grand_total_images}")
    print(f"Total dets   : {grand_total_dets}")
    print(f"Labels saved : {args.output_dir}")


if __name__ == '__main__':
    main()
