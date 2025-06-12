import json

train_path = "data/annotations/train.json"
answer2idx_path = "data/annotations/answer2idx.json"

# Đọc dữ liệu train
with open(train_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Tập hợp tất cả nhãn câu trả lời (answer_label)
answer_set = set()
for q in data["questions"]:
    answer_set.add(q["answer_label"])

# Tạo dict ánh xạ answer_label -> index (theo thứ tự alphabet)
answer2idx = {ans: idx for idx, ans in enumerate(sorted(answer_set))}

# Lưu file JSON
with open(answer2idx_path, "w", encoding="utf-8") as f:
    json.dump(answer2idx, f, ensure_ascii=False, indent=2)

print(f"Tạo file {answer2idx_path} với {len(answer2idx)} nhãn câu trả lời.")
