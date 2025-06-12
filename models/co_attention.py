import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, q_dim=768, v_dim=512, hidden_dim=512):
        super(CoAttention, self).__init__()
        self.Wq = nn.Linear(q_dim, hidden_dim)  # question 
        self.Wv = nn.Linear(v_dim, hidden_dim)  # visual 

    def forward(self, V, Q):  # V: [B, P, v_dim], Q: [B, L, q_dim]
        Q_proj = self.Wq(Q)     # [B, L, hidden_dim]
        V_proj = self.Wv(V)     # [B, P, hidden_dim]

        H = torch.tanh(Q_proj.bmm(V_proj.transpose(1, 2)))  # [B, L, P]

        A_v = F.softmax(H, dim=2)           # [B, L, P]
        V_tilde = A_v.bmm(V)                # [B, L, v_dim]

        A_q = F.softmax(H.transpose(1, 2), dim=2)  # [B, P, L]
        Q_tilde = A_q.bmm(Q)                       # [B, P, q_dim]

        v_hat = V_tilde.sum(dim=1)  # [B, v_dim]
        q_hat = Q_tilde.sum(dim=1)  # [B, q_dim]

        return v_hat, q_hat
