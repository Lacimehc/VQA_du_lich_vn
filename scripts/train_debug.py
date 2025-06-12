import os
import sys
import yaml
import json
import subprocess

# Thêm project root vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import train  # Gọi lại hàm train chính

def create_debug_annotations_if_needed(cfg):
    ann_dir = os.path.join(os.path.dirname(__file__), "..", "data", "annotations")
    train_debug = os.path.join(ann_dir, "train_debug.json")
    val_debug = os.path.join(ann_dir, "val_debug.json")

    if not os.path.exists(train_debug) or not os.path.exists(val_debug):
        print("🔧 Creating debug annotations...")
        subprocess.run([
            "python", "scripts/create_debug_annotation.py",
            "--train", os.path.join(ann_dir, "train.json"),
            "--val", os.path.join(ann_dir, "val.json"),
            "--out_dir", ann_dir,
            "--train_samples", str(cfg['debug']['train_samples']),
            "--val_samples", str(cfg['debug']['val_samples'])
        ])

def main():
    # Load config_debug.yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config_debug.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Tạo file debug nếu cần
    create_debug_annotations_if_needed(config)

    # Gọi train
    train(config)

if __name__ == "__main__":
    main()
