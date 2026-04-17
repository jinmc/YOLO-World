# Auto-labeling script using YOLO-World for 17cls_sub dataset
# Strategy: Each subdirectory name IS the class label.
#   -> For each class dir, use ONLY that class name as text prompt
#   -> All detections in that folder get that class_id
#   -> Much more accurate than using all 17 classes at once
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO-World Auto-Labeling (per-class-dir strategy)')
    parser.add_argument('--config',
                        default='configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py',
                        help='model config file path')
    parser.add_argument('--checkpoint',
                        default='weights/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth',
                        help='checkpoint file')
    parser.add_argument('--data-root',
                        default='data/images/17cls_sub',
                        help='root directory of the dataset')
    parser.add_argument('--output-dir',
                        default='data/labels/17cls_sub',
                        help='output directory for labels')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'val', 'test'],
                        help='dataset splits to process')
    parser.add_argument('--threshold',
                        default=0.05,
                        type=float,
                        help='confidence score threshold')
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='max detections per image')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device for inference')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision')
    return parser.parse_args()


def xyxy_to_yolo(bbox, img_w, img_h):
    """Convert xyxy bbox to YOLO format (cx, cy, w, h) normalized."""
    x1, y1, x2, y2 = bbox
    cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / img_w))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / img_h))
    w = max(0.0, min(1.0, (x2 - x1) / img_w))
    h = max(0.0, min(1.0, (y2 - y1) / img_h))
    return cx, cy, w, h


def inference_single(model, image_path, texts, test_pipeline,
                     score_thr=0.05, max_dets=100, use_amp=False):
    """Run inference using LoadImageFromNDArray pipeline."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None
    img_h, img_w = image.shape[:2]
    image_rgb = image[:, :, [2, 1, 0]]  # BGR -> RGB

    data_info = dict(img=image_rgb, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(
        inputs=data_info['inputs'].unsqueeze(0),
        data_samples=[data_info['data_samples']]
    )

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    return pred_instances['bboxes'], pred_instances['labels'], \
           pred_instances['scores'], (img_w, img_h)


def main():
    args = parse_args()

    # Discover classes from directory structure
    classes = sorted([
        d for d in os.listdir(osp.join(args.data_root, args.splits[0]))
        if osp.isdir(osp.join(args.data_root, args.splits[0], d))
    ])
    class_to_id = {cls: i for i, cls in enumerate(classes)}

    print(f"Found {len(classes)} classes:")
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")

    # ---- Load model (palette='coco' skips LVIS dataset build) ----
    print(f"\nLoading model...")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join('./work_dirs', 'autolabel')
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint,
                          device=args.device, palette='coco')

    # Pipeline: use LoadImageFromNDArray so we can pass cv2 images
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    # ---- Save metadata ----
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
    print(f"Saved classes.txt & dataset.yaml to {args.output_dir}")

    # ---- Process each split ----
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
            # Human-readable text prompt for this class
            text_prompt = cls_name.replace('-', ' ').replace('_', ' ')

            # KEY IDEA: only search for THIS class in THIS folder
            texts = [[text_prompt], [' ']]
            model.reparameterize(texts)

            # Collect images
            img_files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            if not img_files:
                continue

            cls_dets = 0
            for img_file in tqdm(img_files,
                                 desc=f"  [{split}] {cls_name} (id={cls_id})",
                                 leave=False):
                img_path = osp.join(cls_dir, img_file)

                bboxes, labels, scores, (img_w, img_h) = \
                    inference_single(model, img_path, texts, test_pipeline,
                                     score_thr=args.threshold,
                                     max_dets=args.topk,
                                     use_amp=args.amp)
                if bboxes is None:
                    continue

                # Save YOLO-format label
                # All detections get cls_id (folder = ground truth class)
                rel_path = osp.relpath(img_path, args.data_root)
                label_path = osp.join(
                    args.output_dir,
                    osp.splitext(rel_path)[0] + '.txt'
                )
                os.makedirs(osp.dirname(label_path), exist_ok=True)

                with open(label_path, 'w') as f:
                    for bbox, score in zip(bboxes, scores):
                        cx, cy, w, h = xyxy_to_yolo(bbox, img_w, img_h)
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} "
                                f"{w:.6f} {h:.6f}\n")

                cls_dets += len(bboxes)

            print(f"  {cls_name}: {len(img_files)} images, "
                  f"{cls_dets} detections")
            split_images += len(img_files)
            split_dets += cls_dets

        print(f"  >> {split} total: {split_images} images, "
              f"{split_dets} detections")
        grand_total_images += split_images
        grand_total_dets += split_dets

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Auto-labeling Complete!")
    print(f"{'='*60}")
    print(f"Total images : {grand_total_images}")
    print(f"Total dets   : {grand_total_dets}")
    print(f"Labels saved : {args.output_dir}")


if __name__ == '__main__':
    main()
