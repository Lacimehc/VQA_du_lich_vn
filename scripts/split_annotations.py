import os
import json

RAW_ANN = "data/annotations/annotation_all.json"  # file annotation mới
PROCESSED_ROOT = "data/images/processed"  # thư mục chứa category/split/*.jpg
OUT_DIR = "data/annotations"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Quét folder ảnh để tạo dict split_map: split -> set(image_id)
split_map = {"train": set(), "val": set(), "test": set()}

for category in os.listdir(PROCESSED_ROOT):
    cat_dir = os.path.join(PROCESSED_ROOT, category)
    if not os.path.isdir(cat_dir):
        continue
    for split in split_map.keys():
        split_dir = os.path.join(cat_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if fname.lower().endswith((".pt")):
                image_id = os.path.splitext(fname)[0]
                split_map[split].add(image_id)

print("Số ảnh theo split:")
for k, v in split_map.items():
    print(f"{k}: {len(v)}")

# 2. Đọc annotation_all.json
with open(RAW_ANN, 'r', encoding='utf-8') as f:
    ann_all = json.load(f)

questions = ann_all["questions"]

# 3. Phân loại câu hỏi theo split dựa vào image_id
split_questions = {"train": [], "val": [], "test": []}
unknown_questions = []

for q in questions:
    img_id = q["image_id"]
    found = False
    for split, ids in split_map.items():
        if img_id in ids:
            split_questions[split].append(q)
            found = True
            break
    if not found:
        unknown_questions.append(q)

print(f"Tổng câu hỏi: {len(questions)}")
print(f"Câu hỏi không xác định split: {len(unknown_questions)}")

def extract_number_from_image_id(img_id):
    # Ví dụ: "lang_bac_38" -> 38
    parts = img_id.split('_')
    # lấy phần cuối cùng, cố gắng chuyển thành int, nếu lỗi thì trả về 0
    try:
        return int(parts[-1])
    except:
        return 0

def extract_number_from_question_id(q_id):
    # Ví dụ: "lang_bac_1_10" -> 10
    parts = q_id.split('_')
    try:
        return int(parts[-1])
    except:
        return 0

# 4. Lưu file json cho từng split
# Trước khi lưu file json, sắp xếp câu hỏi theo image_id tăng dần rồi question_id tăng dần
for split in ["train", "val", "test"]:
    split_questions[split].sort(key=lambda q: (
        extract_number_from_image_id(q["image_id"]),
        extract_number_from_question_id(q["question_id"])
    ))

    out_path = os.path.join(OUT_DIR, f"{split}.json")
    out_data = {
        "info": {
            "split": split
        },
        "questions": split_questions[split]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Đã tạo {out_path} với {len(split_questions[split])} câu hỏi.")

