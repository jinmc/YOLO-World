# autolabel_v4 foodclassification

## Goal
Run `tools/autolabel_v4.py` on the repo-local dataset at `data/foodclassification_merged_v1`.

## Dataset
- Input: `data/foodclassification_merged_v1`
- Splits: `train val`
- Output: `data/labels_v4/foodclassification_merged_v1`

## Notes
- `empty` should stay a background class and produce empty label files.
- Keep the shell entrypoint simple so the experiment is easy to rerun.
- Extra CLI flags can still be passed through `run.sh` if needed.

## Run
```bash
bash experiments/autolabel_v4_foodclassification/run.sh
```
