#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="data/labels_v4/merged_dataset_adaptive"
LOG_DIR="experiments/merged_adaptive/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/autolabel_v4_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
rm -rf "$OUTPUT_DIR"

python tools/autolabel_v4.py \
  --data-root data \
  --output-dir "$OUTPUT_DIR" \
  --splits merged_dataset_adaptive \
  "$@" 2>&1 | tee "$LOG_PATH"
