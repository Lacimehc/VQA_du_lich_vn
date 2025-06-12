import os
import sys
import yaml
import json
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from torchvision import transforms
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import VQADataset, collate_fn
from models.vqa_model_phobert_vit import VQAModelPhoBERTViT

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, save_every_step, global_step, save_dir, epoch, resume_step_epoch):
    model.train()
    scaler = amp.GradScaler(enabled=device.type == 'cuda')
    total_loss = 0.0

    for step_in_epoch, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}"), 1):
        if step_in_epoch <= resume_step_epoch:
            continue
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        global_step += 1
        if global_step % save_every_step == 0:
            save_path = os.path.join(save_dir, f"step_{global_step}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': global_step,
                'epoch': epoch,
                'step_in_epoch': step_in_epoch, 
                'loss': loss.item()
            }, save_path)
            print(f"💾 Saved checkpoint at step {global_step} to {save_path}")

    return total_loss / len(dataloader), global_step

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            with amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(input_ids, attention_mask, pixel_values)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    return correct / total if total > 0 else 0.0

def train(config):
    print(f"\n=== GPU Check ===")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    set_seed(config['training'].get('seed', 42))
    device = torch.device(config['training']['device'])
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False, legacy=False)

    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dcfg = config['data']
    train_json = os.path.join(project_root, dcfg['train_annotation'])
    val_json = os.path.join(project_root, dcfg['val_annotation'])
    img_dir = os.path.join(project_root, dcfg['image_dir'])
    processed_dir = os.path.join(project_root, "data", "images", "processed")
    idx_path = os.path.join(project_root, dcfg['answer2idx_path'])
    max_len = dcfg['max_question_length']

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = VQADataset(train_json, img_dir, tokenizer, idx_path, image_transform, max_len, processed_dir)
    val_set = VQADataset(val_json, img_dir, tokenizer, idx_path, image_transform, max_len, processed_dir)

    batch_sz = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    train_loader = DataLoader(train_set, batch_size=batch_sz, shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_sz, shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Model
    mcfg = config['model']
    num_answers = len(json.load(open(idx_path, 'r', encoding='utf-8')))
    model = VQAModelPhoBERTViT(mcfg['hidden_dim'], num_answers).to(device)

    # Optimizer, Scheduler, Criterion
    lr = config['training']['learning_rate']
    epochs = config['training']['num_epochs']
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    global_step = 0

    # Resume if needed
    ckpt_dir = os.path.join(project_root, config['training']['save_dir'])
    resume_file = config['training'].get("resume_from", "last_checkpoint.pt")
    resume_path = os.path.join(ckpt_dir, resume_file)
    
    steps_per_epoch = len(train_loader)
    start_epoch = 1
    start_step_in_epoch = 1
    
    if config['training'].get("resume", False) and os.path.exists(resume_path):
        print(f"🔄 Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
            # Tính step trong epoch
        start_step_in_epoch = (global_step % steps_per_epoch) + 1
        if start_step_in_epoch == 1:
            start_epoch += 1
    else:
        global_step = 0
        start_epoch = 1
        start_step_in_epoch = 1

    save_every_step = config['training'].get('save_every_step', 100)

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, save_every_step, global_step, ckpt_dir, epoch,
            start_step_in_epoch if epoch == start_epoch else 1
        )
        val_acc = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': global_step,
                'epoch': epoch,
                'best_acc': best_acc
            }, best_path)
            print(f"✅ Saved best model to {best_path}")

        # Save last checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': global_step,
            'epoch': epoch,
            'best_acc': best_acc
        }, resume_path)

    print("=== Training finished ===")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(project_root, "configs", "config.yaml")
    config = load_config(cfg_path)
    train(config)