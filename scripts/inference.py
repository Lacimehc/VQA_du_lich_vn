import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from models.vqa_model_phobert_vit import VQAModelPhoBERTViT
from data.tokenizer.phobert_tokenizer import PhoBertTokenizer

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load answer2idx
with open(config["data"]["answer2idx_path"], "r", encoding="utf-8") as f:
    answer2idx = json.load(f)
idx2answer = {v: k for k, v in answer2idx.items()}
num_answers = len(answer2idx)

with open("data/annotations/label2answer.json", "r", encoding="utf-8") as f:
    label2answer = json.load(f)

# Device
device = torch.device(config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

# Load model
model = VQAModelPhoBERTViT(config["model"]["hidden_dim"], num_answers).to(device)
checkpoint_path = os.path.join(config["training"]["save_dir"], "best_model.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference image + question
image_path = "data/images/raw/buu_dien_thanh_pho/buu_dien_thanh_pho_2.jpg"
question = "Bạn biết nơi này không?"

# Image transform (same as training preprocessing for ViT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# Tokenize question
phobert_tokenizer = PhoBertTokenizer()
encoded = phobert_tokenizer(question, max_length=config["data"]["max_question_length"])
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

# Run inference
with torch.no_grad():
    logits = model(input_ids, attention_mask, img_tensor)
    probs = torch.softmax(logits, dim=-1)
    top_prob, top_idx = torch.max(probs, dim=-1)
    confidence = top_prob.item()
    pred_label = idx2answer[top_idx.item()]
    pred_answer = label2answer.get(pred_label, pred_label)

    print(question)
    if confidence < 0.3:
        print("Predicted Answer: [Không chắc chắn – mô hình không tự tin về câu trả lời]")
    else:
        print(f"Predicted Answer: {pred_answer} (Confidence: {confidence:.4f})")

    # Optional: show top-5
    top5 = torch.topk(probs, 5)
    print("\nTop-5 Predictions:")
    for i in range(5):
        idx = top5.indices[0, i].item()
        label = idx2answer[idx]
        answer = label2answer.get(label, label)
        print(f"Top {i+1}: {answer} ({top5.values[0, i].item():.4f})")