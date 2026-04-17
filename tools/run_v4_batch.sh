#!/bin/bash
# Batch run autolabel_v4.py on remaining experiments + 17cls_sub
set -e

cd /home/aiteam/Develop/YOLO-World

echo "=========================================="
echo "  Auto-labeling v4 Batch Run"
echo "=========================================="

# --- Experiments: ratio_3to7 ---
echo ""
echo "[1/4] ratio_3to7"
python tools/autolabel_v4.py \
  --data-root data/images/experiments/ratio_3to7 \
  --output-dir data/labels_v4/experiments/ratio_3to7 \
  --splits train \
  --threshold 0.05

# --- Experiments: ratio_5to5 ---
echo ""
echo "[2/4] ratio_5to5"
python tools/autolabel_v4.py \
  --data-root data/images/experiments/ratio_5to5 \
  --output-dir data/labels_v4/experiments/ratio_5to5 \
  --splits train \
  --threshold 0.05

# --- Experiments: ratio_7to3 ---
echo ""
echo "[3/4] ratio_7to3"
python tools/autolabel_v4.py \
  --data-root data/images/experiments/ratio_7to3 \
  --output-dir data/labels_v4/experiments/ratio_7to3 \
  --splits train \
  --threshold 0.05

# --- 17cls_sub (train, val, test) ---
echo ""
echo "[4/4] 17cls_sub (train + val + test)"
python tools/autolabel_v4.py \
  --data-root data/images/17cls_sub \
  --output-dir data/labels_v4/17cls_sub \
  --splits train val test \
  --threshold 0.05

echo ""
echo "=========================================="
echo "  All v4 batch runs complete!"
echo "=========================================="
