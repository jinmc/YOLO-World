#!/usr/bin/env python3
"""Build a merged train/val image dataset from external food classification data.

The output layout matches the class-organized structure used by the auto-labeling
workflow in this repo:

    data/images/experiments/<dataset_name>/{train,val}/{class_name}/*

This script can either symlink or copy the source files into the output tree.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


REAL_SOURCES = {
    "coop_dataset_v1": Path(
        "/home/aiteam/Develop/ai.foodclassification/data/real_data/coop_dataset_v1"
    ),
    "15cls_justin_data": Path(
        "/home/aiteam/Develop/ai.foodclassification/data/real_data/15cls_justin_data"
    ),
}

SYNTH_SOURCE = Path("/home/aiteam/Develop/ai.foodclassification/data/synthetic_data_split")
DEFAULT_OUTPUT = Path(
    "/home/aiteam/Develop/YOLO-World/data/images/experiments/foodclassification_merged_v1"
)

CLASS_ALIASES = {
    "brussels-sprouts": "brussels-sprout",
    "potato-tots": "tots",
    "pizza copy": "pizza",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--real-train-ratio",
        type=float,
        default=0.8,
        help="Per-class train ratio for real datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the real-data split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing output directory if it already exists.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Whether to symlink to source images or copy them into the repo.",
    )
    return parser.parse_args()


def normalize_class_name(name: str) -> str:
    return CLASS_ALIASES.get(name, name)


def iter_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_link_name(source_name: str, image_path: Path) -> str:
    parent_name = image_path.parent.name.replace(" ", "_")
    cleaned = image_path.name.replace(" ", "_")
    return f"{source_name}__{parent_name}__{cleaned}"


def recreate_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {path}. Re-run with --force to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_image(dst_dir: Path, source_name: str, image_path: Path, mode: str) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    output_path = dst_dir / safe_link_name(source_name, image_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    if mode == "symlink":
        output_path.symlink_to(image_path)
    else:
        shutil.copy2(image_path, output_path)

    return output_path.name


def collect_real_images() -> dict[str, list[tuple[str, Path]]]:
    by_class: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for source_name, source_root in REAL_SOURCES.items():
        for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            class_name = normalize_class_name(class_dir.name)
            for image_path in iter_image_files(class_dir):
                by_class[class_name].append((source_name, image_path))
    return by_class


def collect_synth_images(split: str) -> dict[str, list[tuple[str, Path]]]:
    split_root = SYNTH_SOURCE / split
    by_class: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        class_name = normalize_class_name(class_dir.name)
        for image_path in iter_image_files(class_dir):
            by_class[class_name].append((f"synthetic_{split}", image_path))
    return by_class


def split_real_images(
    real_images: dict[str, list[tuple[str, Path]]], train_ratio: float, seed: int
) -> dict[str, dict[str, list[tuple[str, Path]]]]:
    rng = random.Random(seed)
    split_map: dict[str, dict[str, list[tuple[str, Path]]]] = {"train": {}, "val": {}}

    for class_name, items in sorted(real_images.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            train_items = shuffled
            val_items: list[tuple[str, Path]] = []
        else:
            train_count = int(len(shuffled) * train_ratio)
            train_count = max(1, min(len(shuffled) - 1, train_count))
            train_items = shuffled[:train_count]
            val_items = shuffled[train_count:]

        split_map["train"][class_name] = train_items
        split_map["val"][class_name] = val_items

    return split_map


def build_dataset(output_dir: Path, real_train_ratio: float, seed: int, mode: str) -> dict:
    manifest: dict[str, dict] = {
        "output_dir": str(output_dir),
        "seed": seed,
        "real_train_ratio": real_train_ratio,
        "mode": mode,
        "splits": {},
    }

    real_images = collect_real_images()
    real_splits = split_real_images(real_images, real_train_ratio, seed)
    synth_train = collect_synth_images("train")
    synth_val = collect_synth_images("val")

    split_inputs = {
        "train": [real_splits["train"], synth_train],
        "val": [real_splits["val"], synth_val],
    }

    for split_name, datasets in split_inputs.items():
        split_summary: dict[str, dict[str, int]] = {}
        for dataset in datasets:
            for class_name, items in sorted(dataset.items()):
                class_dir = output_dir / split_name / class_name
                linked_names = []
                source_counts = Counter()
                for source_name, image_path in items:
                    linked_names.append(
                        materialize_image(class_dir, source_name, image_path, mode)
                    )
                    source_counts[source_name] += 1

                class_summary = split_summary.setdefault(
                    class_name,
                    {"total": 0},
                )
                class_summary["total"] += len(linked_names)
                for source_name, count in source_counts.items():
                    class_summary[source_name] = class_summary.get(source_name, 0) + count

        manifest["splits"][split_name] = dict(sorted(split_summary.items()))

    return manifest


def main() -> None:
    args = parse_args()
    recreate_dir(args.output, force=args.force)
    manifest = build_dataset(args.output, args.real_train_ratio, args.seed, args.mode)

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Created merged dataset at: {args.output}")
    print(f"Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
