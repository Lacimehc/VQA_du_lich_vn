import json
from collections import defaultdict, Counter

with open("data/annotations/annotation_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = data["questions"]  # ⭐ Lấy danh sách câu hỏi

label_to_answers = defaultdict(list)

for item in questions:  # ⭐ Lặp qua danh sách câu hỏi
    label = item["answer_label"]
    answer = item["answer"]
    label_to_answers[label].append(answer)

# Lấy câu trả lời xuất hiện nhiều nhất cho mỗi label
label2answer = {
    label: Counter(answers).most_common(1)[0][0]
    for label, answers in label_to_answers.items()
}

with open("data/annotations/label2answer.json", "w", encoding="utf-8") as f:
    json.dump(label2answer, f, ensure_ascii=False, indent=2)
