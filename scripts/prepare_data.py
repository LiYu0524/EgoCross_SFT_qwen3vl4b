#!/usr/bin/env python3
"""
Prepare EgoCross dataset by converting relative image paths to absolute paths.

Usage:
    python scripts/prepare_data.py --data_dir ./data --output_dir ./data_prepared
"""

import argparse
import json
import os
import shutil


def convert_paths(data, data_root):
    """Convert relative image paths to absolute paths."""
    for sample in data:
        sample["images"] = [
            os.path.join(data_root, img) for img in sample["images"]
        ]
    return data


def main():
    parser = argparse.ArgumentParser(description="Prepare EgoCross dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to downloaded data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: modify in place)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else data_dir

    if output_dir != data_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Copy frames directory
        if os.path.exists(os.path.join(data_dir, "frames")):
            print(f"Copying frames to {output_dir}/frames ...")
            shutil.copytree(
                os.path.join(data_dir, "frames"),
                os.path.join(output_dir, "frames"),
                dirs_exist_ok=True
            )

    # Process JSON files
    json_files = [
        "train.json",
        "train_animal.json",
        "train_industry.json",
        "train_xsports.json",
        "train_surgery.json",
    ]

    for json_file in json_files:
        src_path = os.path.join(data_dir, json_file)
        if not os.path.exists(src_path):
            print(f"Skipping {json_file} (not found)")
            continue

        print(f"Processing {json_file} ...")
        with open(src_path, "r") as f:
            data = json.load(f)

        # Convert paths
        data = convert_paths(data, output_dir)

        # Save
        dst_path = os.path.join(output_dir, json_file)
        with open(dst_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(data)} samples to {dst_path}")

    # Copy dataset_info.json
    src_info = os.path.join(data_dir, "dataset_info.json")
    if os.path.exists(src_info) and output_dir != data_dir:
        shutil.copy(src_info, os.path.join(output_dir, "dataset_info.json"))

    print(f"\nDone! Dataset prepared at: {output_dir}")
    print(f"Update your config: dataset_dir: {output_dir}")


if __name__ == "__main__":
    main()
