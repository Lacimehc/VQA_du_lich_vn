import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel
from models.co_attention import CoAttention

class VQAModelPhoBERTViT(nn.Module):
    def __init__(self, hidden_dim, num_answers):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base")
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.q_dim = 768
        self.v_dim = 768

        self.co_attention = CoAttention(q_dim=self.q_dim, v_dim=self.v_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.q_dim + self.v_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_answers)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # PhoBERT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        question_feats = text_outputs.last_hidden_state  # [B, L, 768]

        # ViT
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_feats = image_outputs.last_hidden_state[:, 1:, :]  # [B, P, 768]

        # Co-Attention
        v_hat, q_hat = self.co_attention(image_feats, question_feats)  # [B, 768], [B, 768]

        # Fusion
        fused = torch.cat([v_hat, q_hat], dim=-1)  # [B, 1536]
        logits = self.classifier(fused)  # [B, num_answers]

        return logits
