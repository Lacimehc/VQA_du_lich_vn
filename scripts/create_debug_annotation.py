import os
import json
import argparse

def create_debug_annotation(input_path, output_path, max_samples=1000):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Lấy danh sách câu hỏi từ key "questions"
    new_data = raw_data.get("questions", [])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved debug file to {output_path} ({len(new_data['questions'])} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to original train.json")
    parser.add_argument("--val", type=str, required=True, help="Path to original val.json")
    parser.add_argument("--out_dir", type=str, default="data/annotations", help="Where to save debug files")
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=500)
    args = parser.parse_args()

    train_out = os.path.join(args.out_dir, "train_debug.json")
    val_out = os.path.join(args.out_dir, "val_debug.json")

    create_debug_annotation(args.train, train_out, args.train_samples)
    create_debug_annotation(args.val, val_out, args.val_samples)
