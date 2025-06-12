import json
from underthesea import word_tokenize
from tqdm import tqdm
import os
import warnings

def segment_annotations(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = raw_data.get("questions", [])

    for item in tqdm(data, desc=f"Processing {os.path.basename(input_path)}"):
        if not isinstance(item, dict):
            warnings.warn(f"Bỏ qua item không hợp lệ: {item}")
            continue
        question = item.get("question", "")
        segmented = word_tokenize(question, format="text")
        item["segmented_question"] = segmented

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "info": raw_data.get("info", {}),
            "questions": data
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    segment_annotations("data/annotations/train.json", "data/annotations/train_segmented.json")
    segment_annotations("data/annotations/val.json",   "data/annotations/val_segmented.json")
    segment_annotations("data/annotations/test.json",  "data/annotations/test_segmented.json")
