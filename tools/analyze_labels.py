"""Analyze label quality: how many images have detections per class."""
import os
import sys

def analyze(image_dir, label_dir):
    classes = sorted(os.listdir(image_dir))
    
    header = f"{'Class':<20} {'Images':>7} {'Labeled':>8} {'%':>6} {'Empty':>7} {'NoFile':>7} {'AvgDets':>8}"
    print(header)
    print("-" * len(header))
    
    total_img = 0
    total_labeled = 0
    total_empty = 0
    total_none = 0
    
    for cls in classes:
        cls_img = os.path.join(image_dir, cls)
        cls_lbl = os.path.join(label_dir, cls)
        if not os.path.isdir(cls_img):
            continue
        
        img_files = [f for f in os.listdir(cls_img)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        has_label = 0
        empty_label = 0
        no_label = 0
        total_dets = 0
        
        for f in img_files:
            lbl_path = os.path.join(cls_lbl, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(lbl_path):
                if os.path.getsize(lbl_path) > 0:
                    with open(lbl_path) as fh:
                        lines = [l for l in fh.readlines() if l.strip()]
                        total_dets += len(lines)
                        has_label += 1
                else:
                    empty_label += 1
            else:
                no_label += 1
        
        avg = total_dets / has_label if has_label > 0 else 0
        pct = has_label / len(img_files) * 100 if img_files else 0
        print(f"{cls:<20} {len(img_files):>7} {has_label:>8} {pct:>5.0f}% {empty_label:>7} {no_label:>7} {avg:>8.1f}")
        
        total_img += len(img_files)
        total_labeled += has_label
        total_empty += empty_label
        total_none += no_label
    
    print("-" * len(header))
    pct = total_labeled / total_img * 100 if total_img else 0
    print(f"{'TOTAL':<20} {total_img:>7} {total_labeled:>8} {pct:>5.0f}% {total_empty:>7} {total_none:>7}")


if __name__ == '__main__':
    img_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/images/experiments/ratio_real_only/train'
    lbl_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/labels/experiments/ratio_real_only/train'
    analyze(img_dir, lbl_dir)
