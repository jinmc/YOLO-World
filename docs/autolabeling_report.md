# Auto-Labeling Pipeline Report

## Overview

This document describes the iterative development of an auto-labeling pipeline using **YOLO-World v2 (Large)** for food image datasets. The pipeline generates YOLO-format bounding box annotations automatically from class-organized image directories.

### Environment

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 2080 Ti (11GB VRAM) |
| CUDA Toolkit | 11.8 |
| Driver | 570.181 |
| OS | Ubuntu 20.04 |
| Python | 3.10 (conda env: `yolo-world`) |
| PyTorch | 2.0.0+cu118 |
| mmcv | 2.0.0 |
| mmdet | 3.0.0 |
| mmyolo | 0.6.0 |
| Model | YOLO-World v2 Large (obj365v1+goldg pretrained, 422MB) |

### Dataset Structure

Images are organized in per-class subdirectories:
```
data/images/experiments/{experiment_name}/train/{class_name}/*.jpg
```

Each experiment contains **15 classes × 140 images = 2,100 images**:
- **14 food classes**: bacon, broccoli, brussels-sprout, cauliflower, chicken-nuggets, chicken-wings, cookies, cut-potatoes, french-fries, pizza, salmon, steak, tots, whole-chicken
- **1 background class**: empty (empty tray with no food)

Five experiment variants exist:
| Experiment | Description |
|---|---|
| `ratio_real_only` | 100% real images |
| `ratio_synth_only` | 100% synthetic images |
| `ratio_3to7` | 30% real + 70% synthetic |
| `ratio_5to5` | 50% real + 50% synthetic |
| `ratio_7to3` | 70% real + 30% synthetic |

### Output Format

YOLO-format `.txt` label files:
```
<class_id> <center_x> <center_y> <width> <height>
```
All coordinates are normalized to `[0, 1]`.

---

## Version History

### v1 — Baseline Single-Prompt (`tools/autolabel.py`)

**Strategy:**
- Single text prompt per class (the class directory name, e.g., `"salmon"`, `"steak"`)
- Fixed confidence threshold: `0.05`
- One inference pass per image

**Results on `ratio_real_only` (2,100 images):**

| Class | Images | Labeled | Coverage | Detections |
|---|---|---|---|---|
| bacon | 140 | 44 | 31% | 90 |
| broccoli | 140 | 128 | 91% | 619 |
| brussels-sprout | 140 | 140 | 100% | 1,366 |
| cauliflower | 140 | 102 | 73% | 201 |
| chicken-nuggets | 140 | 99 | 71% | 133 |
| chicken-wings | 140 | 47 | 34% | 76 |
| cookies | 140 | 130 | 93% | 295 |
| cut-potatoes | 140 | 57 | 41% | 58 |
| empty | 140 | 96 | 69% | 104 |
| french-fries | 140 | 44 | 31% | 52 |
| pizza | 140 | 104 | 74% | 105 |
| salmon | 140 | 3 | **2%** | 3 |
| steak | 140 | 10 | **7%** | 12 |
| tots | 140 | 100 | 71% | 103 |
| whole-chicken | 140 | 126 | 90% | 126 |
| **Total** | **2,100** | **1,230** | **59%** | **3,343** |

**Analysis:**
- Catastrophic failure on `salmon` (2%) and `steak` (7%)
- Poor coverage on `bacon` (31%), `french-fries` (31%), `chicken-wings` (34%), `cut-potatoes` (41%)
- Root cause: YOLO-World is highly sensitive to text prompt wording. For example, the word `"salmon"` yields max confidence ~0.003, but `"fish"` yields ~0.06 on the same images.
- The `empty` class was incorrectly treated as a detectable object (96 images had false detections)

**Output:** `data/labels/experiments/`

---

### v2 — Multi-Prompt + Per-Class Thresholds (`tools/autolabel_v2.py`)

**Key Improvements over v1:**
1. **Multi-prompt inference**: Each class uses multiple text prompts. All results are merged with NMS (IoU=0.5).
2. **Per-class threshold overrides**: Difficult classes use lower confidence thresholds.
3. **Custom NMS**: Numpy-based NMS to deduplicate detections across different prompts.

**Prompt Configuration:**

| Class | Threshold | Prompts |
|---|---|---|
| salmon | 0.01 | `fish`, `cooked fish`, `salmon` |
| steak | 0.01 | `food on tray`, `beef`, `meat`, `steak` |
| bacon | 0.02 | `bacon strips`, `bacon`, `cooked bacon`, `pork` |
| french-fries | 0.02 | `french fries`, `chips`, `fries`, `fried potatoes` |
| cut-potatoes | 0.02 | `diced potatoes`, `chopped potatoes`, `cut potatoes`, `vegetables` |
| chicken-wings | 0.02 | `chicken wings`, `chicken wing`, `fried chicken` |
| empty | 0.02 | `empty tray`, `empty plate`, `empty pan` |
| *(others)* | 0.05 | class name + 1-2 alternates |

**Results on `ratio_real_only` (2,100 images):**

| Class | v1 Coverage | **v2 Coverage** | v2 Detections |
|---|---|---|---|
| bacon | 31% | **86%** | 344 |
| broccoli | 91% | **97%** | 629 |
| brussels-sprout | 100% | **100%** | 1,452 |
| cauliflower | 73% | **74%** | 213 |
| chicken-nuggets | 71% | **83%** | 234 |
| chicken-wings | 34% | **57%** | 295 |
| cookies | 93% | **94%** | 403 |
| cut-potatoes | 41% | **98%** | 339 |
| empty | 69% | 45% | 64 |
| french-fries | 31% | **86%** | 160 |
| pizza | 74% | **77%** | 109 |
| salmon | 2% | **89%** | 284 |
| steak | 7% | **84%** | 213 |
| tots | 71% | **73%** | 111 |
| whole-chicken | 90% | **91%** | 128 |
| **Total** | **59%** | **82%** | **4,978** |

**Analysis:**
- Massive improvement on previously-failed classes: salmon (2%→89%), steak (7%→84%), bacon (31%→86%)
- Multi-prompt approach successfully captures detections that single prompts miss
- Still below 100% on many classes — model simply cannot detect some images even with multiple prompts at current thresholds
- `empty` class still incorrectly labeled (45% of empty trays had false positive detections)

**Output:** `data/labels_v2/experiments/`

---

### v3 — 100% Coverage Guarantee (`tools/autolabel_v3.py`)

**Key Improvements over v2:**
1. **Progressive fallback thresholds**: If primary threshold yields no detection, retry at `0.005 → 0.003 → 0.001`
2. **Full-image bbox fallback**: If even the lowest threshold yields nothing, place one bbox covering the entire image (since we know the food class is present)
3. **Enhanced prompts**: More prompt variations for difficult classes (e.g., chicken-wings gets 7 prompts)

**Results on `ratio_real_only` (2,100 images):**

| Class | v2 Coverage | **v3 Coverage** | v3 Detections | Fallback Breakdown |
|---|---|---|---|---|
| bacon | 86% | **100%** ✅ | 386 | primary=121, fb@0.005=18, fb@0.003=1 |
| broccoli | 97% | **100%** ✅ | 649 | primary=136, fb@0.005=4 |
| brussels-sprout | 100% | **100%** ✅ | 1,524 | primary=140 |
| cauliflower | 74% | **100%** ✅ | 455 | primary=125, fb@0.005=14, fb@0.001=1 |
| chicken-nuggets | 83% | **100%** ✅ | 401 | primary=123, fb@0.005=15, fb@0.003=1, fb@0.001=1 |
| chicken-wings | 57% | **100%** ✅ | 967 | primary=130, fb@0.005=5, fb@0.003=3, fb@0.001=2 |
| cookies | 94% | **100%** ✅ | 640 | primary=132, fb@0.005=8 |
| cut-potatoes | 98% | **100%** ✅ | 342 | primary=137, fb@0.005=2, fb@0.003=1 |
| empty | 45% | ~~100%~~ | 189 | *(incorrectly forced detections)* |
| french-fries | 86% | **100%** ✅ | 185 | primary=125, fb@0.005=15 |
| pizza | 77% | **100%** ✅ | 265 | primary=137, fb@0.005=1, fb@0.003=1, full-img=1 |
| salmon | 89% | **100%** ✅ | 342 | primary=127, fb@0.005=7, fb@0.003=4, fb@0.001=2 |
| steak | 84% | **100%** ✅ | 244 | primary=118, fb@0.005=8, fb@0.003=4, fb@0.001=9, full-img=1 |
| tots | 73% | **100%** ✅ | 413 | primary=133, fb@0.005=7 |
| whole-chicken | 91% | **100%** ✅ | 150 | primary=127, fb@0.005=12, fb@0.003=1 |

**Fallback Statistics (2,100 images total):**

| Level | Count | Percentage |
|---|---|---|
| Primary threshold | 1,940 | 92.4% |
| Fallback @0.005 | 125 | 5.9% |
| Fallback @0.003 | 18 | 0.9% |
| Fallback @0.001 | 15 | 0.7% |
| Full-image bbox | 2 | 0.1% |

**Issue Identified:**
- The `empty` class (empty trays with no food) was treated the same as food classes, receiving forced detections. This is semantically incorrect — empty images should have **empty label files** (background images with no bounding boxes).

**Output:** `data/labels_v3/experiments/`

---

### v4 — Correct Empty Class Handling (Final) (`tools/autolabel_v4.py`)

**Key Improvements over v3:**
1. **`SKIP_CLASSES` set**: Classes like `"empty"` are explicitly skipped — their label files are written as empty (0 bytes), indicating background images with no objects.
2. **No detection for background classes**: No inference is run, saving compute time.
3. **All food classes**: Retain 100% coverage guarantee with the same progressive fallback strategy.

**Semantic Correctness:**
- `empty` class images = empty trays → **no bounding boxes** (empty `.txt` file)
- All other classes = food present → **guaranteed ≥1 bounding box** per image

**Configuration:**
```python
SKIP_CLASSES = {"empty"}
```

**Results on `ratio_real_only` (2,100 images):**

| Class | Coverage | Detections | Fallback Breakdown |
|---|---|---|---|
| bacon | **100%** ✅ | 386 | primary=121, fb@0.005=18, fb@0.003=1 |
| broccoli | **100%** ✅ | 649 | primary=136, fb@0.005=4 |
| brussels-sprout | **100%** ✅ | 1,524 | primary=140 |
| cauliflower | **100%** ✅ | 455 | primary=125, fb@0.005=14, fb@0.001=1 |
| chicken-nuggets | **100%** ✅ | 401 | primary=123, fb@0.005=15, fb@0.003=1, fb@0.001=1 |
| chicken-wings | **100%** ✅ | 967 | primary=130, fb@0.005=5, fb@0.003=3, fb@0.001=2 |
| cookies | **100%** ✅ | 640 | primary=132, fb@0.005=8 |
| cut-potatoes | **100%** ✅ | 342 | primary=137, fb@0.005=2, fb@0.003=1 |
| **empty** | **SKIP** ✅ | **0** | 140 empty label files (background) |
| french-fries | **100%** ✅ | 185 | primary=125, fb@0.005=15 |
| pizza | **100%** ✅ | 265 | primary=137, fb@0.005=1, fb@0.003=1, full-img=1 |
| salmon | **100%** ✅ | 342 | primary=127, fb@0.005=7, fb@0.003=4, fb@0.001=2 |
| steak | **100%** ✅ | 244 | primary=118, fb@0.005=8, fb@0.003=4, fb@0.001=9, full-img=1 |
| tots | **100%** ✅ | 413 | primary=133, fb@0.005=7 |
| whole-chicken | **100%** ✅ | 150 | primary=127, fb@0.005=12, fb@0.003=1 |
| **Total** | **100%** | **6,963** | |

**Fallback Statistics (1,960 food images, excluding 140 empty):**

| Level | Count | Percentage |
|---|---|---|
| Primary threshold | 1,811 | 92.4% |
| Fallback @0.005 | 116 | 5.9% |
| Fallback @0.003 | 16 | 0.8% |
| Fallback @0.001 | 15 | 0.8% |
| Full-image bbox | 2 | 0.1% |
| Skipped (empty) | 140 | — |

**Output:** `data/labels_v4/experiments/`

---

## Coverage Progression Summary

| Class | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| bacon | 31% | 86% | 100% | **100%** ✅ |
| broccoli | 91% | 97% | 100% | **100%** ✅ |
| brussels-sprout | 100% | 100% | 100% | **100%** ✅ |
| cauliflower | 73% | 74% | 100% | **100%** ✅ |
| chicken-nuggets | 71% | 83% | 100% | **100%** ✅ |
| chicken-wings | 34% | 57% | 100% | **100%** ✅ |
| cookies | 93% | 94% | 100% | **100%** ✅ |
| cut-potatoes | 41% | 98% | 100% | **100%** ✅ |
| empty | ~~69%~~ | ~~45%~~ | ~~100%~~ | **skip** ✅ |
| french-fries | 31% | 86% | 100% | **100%** ✅ |
| pizza | 74% | 77% | 100% | **100%** ✅ |
| salmon | **2%** | 89% | 100% | **100%** ✅ |
| steak | **7%** | 84% | 100% | **100%** ✅ |
| tots | 71% | 73% | 100% | **100%** ✅ |
| whole-chicken | 90% | 91% | 100% | **100%** ✅ |
| **Overall** | **59%** | **82%** | **100%** | **100%** ✅ |

---

## Key Lessons Learned

### 1. Text Prompt Sensitivity
YOLO-World's open-vocabulary detection is **extremely sensitive** to prompt wording. The word `"salmon"` yields near-zero confidence, while `"fish"` on the same image yields detectable scores. This necessitated the multi-prompt approach.

### 2. Multi-Prompt + NMS Strategy
Running inference with multiple text prompts per class and merging results via NMS proved highly effective. This alone improved coverage from 59% (v1) to 82% (v2).

### 3. Progressive Threshold Fallback
For images where even multi-prompt at the primary threshold fails, progressively lowering the threshold (0.005 → 0.003 → 0.001) catches most remaining cases. 92.4% of images are handled at the primary threshold, with only 7.6% needing fallback.

### 4. Background Class Handling
Background/empty classes must be explicitly handled. These images should produce **empty label files** (no bounding boxes), not forced detections. This was corrected in v4.

### 5. Full-Image Fallback
As a last resort, placing a full-image bounding box is acceptable when we have strong prior knowledge (the class directory guarantees the object is present). Only 2 out of 2,100 images required this (0.1%).

---

## File Reference

| File | Description |
|---|---|
| `tools/autolabel.py` | v1: Single-prompt baseline |
| `tools/autolabel_v2.py` | v2: Multi-prompt + per-class thresholds |
| `tools/autolabel_v3.py` | v3: + Progressive fallback + full-image fallback |
| `tools/autolabel_v4.py` | v4: + Correct empty class handling (final) |
| `tools/visualize_labels.py` | Visualization tool for label inspection |
| `tools/analyze_labels.py` | Per-class label quality analysis |
| `data/labels/` | v1 label outputs |
| `data/labels_v2/` | v2 label outputs |
| `data/labels_v3/` | v3 label outputs |
| `data/labels_v4/` | v4 label outputs (final, recommended) |

---

## Appendix: v4 Full Prompt & Threshold Configuration

```python
SKIP_CLASSES = {"empty"}

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
}

THRESHOLD_MAP = {
    "salmon": 0.01, "steak": 0.01, "bacon": 0.02, "french-fries": 0.02,
    "cut-potatoes": 0.02, "chicken-wings": 0.01, "cauliflower": 0.02,
    "tots": 0.02, "pizza": 0.02, "chicken-nuggets": 0.03,
}
# Default threshold: 0.05

FALLBACK_THRESHOLDS = [0.005, 0.003, 0.001]
```

### Usage

```bash
# Run v4 on a single experiment
python tools/autolabel_v4.py \
  --data-root data/images/experiments/ratio_real_only \
  --output-dir data/labels_v4/experiments/ratio_real_only \
  --splits train \
  --threshold 0.05
```
