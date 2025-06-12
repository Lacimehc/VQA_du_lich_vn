import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from models.vqa_model_phobert_vit import VQAModelPhoBERTViT
from data.dataset import VQADataset, collate_fn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

with open("configs/vqa_config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open(config["data"]["answer2idx_path"], "r", encoding="utf-8") as f:
    answer2idx = json.load(f)
num_answers = len(answer2idx)
idx2answer = {v: k for k, v in answer2idx.items()}

# Load model
device = torch.device(config["training"].get("device", "cuda"))
model = VQAModelPhoBERTViT(config["model"]["hidden_dim"], num_answers).to(device)
ckpt_path = os.path.join(config["training"]["save_dir"], "model_epoch_20.pth")  # chỉnh lại epoch nếu cần
model.load_state_dict(torch.load(ckpt_path))
model.eval()

val_set = VQADataset("val", config)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, pixel_values)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

acc = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {acc * 100:.2f}%")