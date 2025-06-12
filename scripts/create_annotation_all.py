import json
from pathlib import Path
import random
import re
import datetime

random.seed(42)

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
DATA_DIR = Path("data")
QUESTION_DIR = DATA_DIR / "questions"
ANNOTATION_DIR = DATA_DIR / "annotations"
IMAGE_DIR = DATA_DIR / "images" / "raw"

# Tạo thư mục nếu chưa tồn tại
QUESTION_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# ==================== LOAD DỮ LIỆU TỪ FILE ====================
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

GENERAL_QUESTIONS = load_json(QUESTION_DIR / "general_questions.json")
SPECIFIC_QUESTIONS = load_json(QUESTION_DIR / "landmark_specific_questions.json")
LANDMARK_INFO = load_json(QUESTION_DIR / "landmark_info.json")
YES_NO_TEMPLATES = load_json(QUESTION_DIR / "yesno_questions.json")

# ==================== HÀM CHÍNH ====================
def main():
    annotation_path = ANNOTATION_DIR / "annotation_all.json"
    if annotation_path.exists():
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            existing_questions = data.get("questions", [])
    else:
        existing_questions = []

    # Lấy danh sách ảnh và sắp xếp
    image_files = sorted(IMAGE_DIR.rglob("*.jpg"))
    landmark_keys = list(LANDMARK_INFO.keys())

    new_questions = []

    for img_path in image_files:
        image_id = img_path.stem
        
        # Reset question_id counter cho mỗi ảnh
        question_id = 0

        # Xác định địa danh từ tên ảnh
        landmark_key = next((k for k in LANDMARK_INFO if image_id.startswith(k)), None)
        if landmark_key is None:
            continue

        correct_readable = LANDMARK_INFO[landmark_key]["landmark_readable"]
        correct_province = LANDMARK_INFO[landmark_key]["province"]

        # ========== SINH CÂU HỎI CHUNG ==========
        for group, qa_list in GENERAL_QUESTIONS.items():
            sampled = random.sample(qa_list, min(1, len(qa_list)))
            for q, a, label in sampled:
                question_id += 1
                answer = a.format(
                    landmark_readable=correct_readable,
                    province=correct_province
                )
                answer_label = label.format(
                    landmark_label=LANDMARK_INFO[landmark_key]["label"],
                    province_label=LANDMARK_INFO[landmark_key]["province_label"]
                )
                new_questions.append({
                    "question_id": f"{image_id}_{question_id}",
                    "image_id": image_id,
                    "question": q,
                    "answer": answer,
                    "answer_label": answer_label,
                    "group": group
                })

        # ========== SINH CÂU HỎI RIÊNG ==========
        if landmark_key in SPECIFIC_QUESTIONS:
            qa_list = SPECIFIC_QUESTIONS[landmark_key]
            sampled = random.sample(qa_list, min(3, len(qa_list)))
            for qa in sampled:
                question_id += 1
                new_questions.append({
                    "question_id": f"{image_id}_{question_id}",
                    "image_id": image_id,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "answer_label": qa.get("answer_label", ""),
                    "group": "specific"
                })

        # ========== SINH CÂU HỎI YES/NO ==========
        other_keys = [k for k in landmark_keys if k != landmark_key]
        if other_keys:
            wrong_key = random.choice(other_keys)
            wrong_readable = LANDMARK_INFO[wrong_key]["landmark_readable"]
            wrong_province = LANDMARK_INFO[wrong_key]["province"]

            sampled_templates = random.sample(YES_NO_TEMPLATES, 2)
            for template in sampled_templates:
                question_id += 1
                new_questions.append({
                    "question_id": f"{image_id}_{question_id}",
                    "image_id": image_id,
                    "question": template["question"].format(
                        correct_landmark=correct_readable,
                        correct_province=correct_province,
                        wrong_landmark=wrong_readable,
                        wrong_province=wrong_province
                    ),
                    "answer": template["answer"].format(
                        correct_landmark=correct_readable,
                        correct_province=correct_province,
                        wrong_landmark=wrong_readable,
                        wrong_province=wrong_province
                    ),
                    "answer_label": template["answer_label"],
                    "group": "yesno"
                })

    # Gộp với câu hỏi cũ và lưu file
    def natural_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s["image_id"])]

    all_questions = sorted(existing_questions + new_questions, key=natural_key)
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump({
            "info": {
                "split": "all",
                "version": "1.0",
                "generated_at": datetime.datetime.now().isoformat()
            },
            "questions": all_questions
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã sinh {len(new_questions)} câu hỏi mới cho {len(set(q['image_id'] for q in new_questions))} ảnh")

if __name__ == "__main__":
    main()
