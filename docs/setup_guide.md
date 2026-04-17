# Environment Setup & Git Workflow Guide

## 1. Environment Setup

### Prerequisites

- **OS**: Ubuntu 20.04+ (tested on Ubuntu 20.04)
- **GPU**: NVIDIA GPU with ≥8GB VRAM (tested on RTX 2080 Ti 11GB)
- **CUDA Driver**: ≥470 (check with `nvidia-smi`)
- **Conda**: Anaconda or Miniconda installed

### Check Your CUDA Version

```bash
nvidia-smi
# Look for "CUDA Version" in the top right — this is your driver's max supported CUDA
# e.g., "CUDA Version: 11.8" or "CUDA Version: 12.1"
```

### Step-by-Step Installation

```bash
# 1. Create conda environment
conda create -n yolo-world python=3.10 -y
conda activate yolo-world

# 2. Install PyTorch (match your CUDA version)
#    For CUDA 11.8:
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

#    For CUDA 12.1 (if your system has CUDA 12.x):
#    pip install torch==2.0.0+cu121 torchvision==0.15.1+cu121 \
#        --index-url https://download.pytorch.org/whl/cu121

# 3. Install mmcv (MUST use prebuilt wheel — do NOT run `pip install mmcv`)
#    For CUDA 11.8 + torch 2.0.0:
pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv-2.0.0-cp310-cp310-manylinux1_x86_64.whl

#    For CUDA 12.1 + torch 2.0.0:
#    pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.0.0/mmcv-2.0.0-cp310-cp310-manylinux1_x86_64.whl
#
#    Browse all prebuilt wheels at: https://download.openmmlab.com/mmcv/dist/

# 4. Install mmdet and mmengine
pip install mmdet==3.0.0 mmengine==0.10.3

# 5. Install mmyolo from third_party
cd third_party/mmyolo
pip install -e .
cd ../..

# 6. Install YOLO-World
pip install -e .

# 7. Install additional dependencies
pip install transformers==4.36.2 supervision==0.19.0 "numpy<2"

# 8. Download model weights
mkdir -p weights
wget -O weights/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth \
    https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth
```

### Verify Installation

```bash
conda activate yolo-world
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0)}')

import mmcv
print(f'mmcv: {mmcv.__version__}')

import mmdet
print(f'mmdet: {mmdet.__version__}')

import numpy as np
print(f'numpy: {np.__version__}')
"
```

Expected output:
```
PyTorch: 2.0.0+cu118
CUDA available: True
CUDA device: NVIDIA GeForce RTX 2080 Ti
mmcv: 2.0.0
mmdet: 3.0.0
numpy: 1.26.4
```

---

## 2. Common Pitfalls

### ❌ `pip install mmcv` fails to build
**Solution**: Always use the prebuilt wheel URL (step 3 above). Never build from source.

### ❌ `numpy 2.x` breaks everything
**Solution**: Pin `numpy<2`. PyTorch 2.0.0 is incompatible with numpy 2.x.
```bash
pip install "numpy<2"
```

### ❌ CUDA version mismatch (e.g., cu121 torch with cu118 system)
**Solution**: Match all three — torch, mmcv wheel, and your CUDA driver.
```bash
# Check what CUDA your torch was built with:
python -c "import torch; print(torch.version.cuda)"
```

### ❌ `FileNotFoundError: lvis/...` when running init_detector
**Solution**: Pass `palette='coco'` to `init_detector()`:
```python
model = init_detector(config, weights, palette='coco', device='cuda:0')
```

### ❌ `RuntimeError: "nms_cuda" not implemented for 'Half'`
**Solution**: Do NOT use AMP (automatic mixed precision). Run inference in fp32.

### ❌ New terminal forgets conda env
**Solution**: Always run `conda activate yolo-world` in each new terminal.

---

## 3. Running Auto-labeling

### Single dataset

```bash
conda activate yolo-world

python tools/autolabel_v4.py \
    --data-root data/images/experiments/ratio_real_only \
    --output-dir data/labels_v4/experiments/ratio_real_only \
    --splits train \
    --threshold 0.05
```

### Batch run (all experiments + 17cls_sub)

```bash
conda activate yolo-world
bash tools/run_v4_batch.sh
```

### Verify results

```bash
# Check file counts
for cls in data/labels_v4/experiments/ratio_real_only/train/*/; do
    cn=$(basename "$cls")
    t=$(find "$cls" -name "*.txt" | wc -l)
    e=$(find "$cls" -name "*.txt" -empty | wc -l)
    echo "$cn: $t labels, $e empty"
done

# Visualize
python tools/visualize_labels.py \
    --image-dir data/images/experiments/ratio_real_only \
    --label-dir data/labels_v4/experiments/ratio_real_only \
    --output-dir data/viz_v4/experiments/ratio_real_only \
    --splits train \
    --num-samples 3
```

---

## 4. Git Workflow

### Initial clone (on a new PC)

```bash
git clone -b feature/autolabeling-v4 https://github.com/jinmc/YOLO-World.git
cd YOLO-World
```

### Making changes

```bash
# 1. Create a new branch from feature/autolabeling-v4
git checkout feature/autolabeling-v4
git checkout -b feature/my-new-feature

# 2. Make your changes...

# 3. Stage and commit
git add <files>
git commit -m "feat: description of changes"

# 4. Push to your fork
git push origin feature/my-new-feature

# 5. Create PR on GitHub
#    Go to https://github.com/jinmc/YOLO-World
#    Click "Compare & pull request"
```

### What NOT to commit

These are excluded by `.gitignore`:

| Path | Reason |
|---|---|
| `data/images/` | Large image files (~several GB) |
| `data/labels/`, `data/labels_v2/`, `data/labels_v3/` | Old label versions |
| `data/labels_v4/` | Generated labels (106MB of text files) |
| `data/labels_v4.zip` | Zip of labels |
| `data/viz/`, `data/viz_v4/` | Visualization outputs |
| `weights/` | Model weights (422MB) |
| `work_dirs/` | Training outputs |

### Syncing with upstream

```bash
# Add upstream remote (only once)
git remote add upstream https://github.com/AILab-CVC/YOLO-World.git

# Fetch and merge upstream changes
git fetch upstream
git merge upstream/master
```

---

## 5. Project Structure

```
YOLO-World/
├── tools/
│   ├── autolabel.py          # v1: single-prompt baseline
│   ├── autolabel_v2.py       # v2: multi-prompt + per-class thresholds
│   ├── autolabel_v3.py       # v3: + progressive fallback
│   ├── autolabel_v4.py       # v4: + correct empty class handling (FINAL)
│   ├── visualize_labels.py   # Draw bboxes on images for inspection
│   ├── analyze_labels.py     # Per-class label quality analysis
│   └── run_v4_batch.sh       # Batch runner for all datasets
├── docs/
│   ├── autolabeling_report.md  # Comprehensive v1-v4 report
│   └── setup_guide.md         # This file
├── data/
│   ├── images/               # (gitignored) source images
│   ├── labels_v4/            # (gitignored) generated YOLO labels
│   └── viz_v4/               # (gitignored) visualization output
├── weights/                  # (gitignored) model weights
└── yolo_world/               # YOLO-World source code
```
