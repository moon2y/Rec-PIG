"""
Deep Prompt Token (DPT) HyperNet gθ: u(256) -> K tokens(768 each)
We use a small MLP with LayerNorm; K tokens are produced per user and
prepended to prompt_embeds during training/inference.
"""
import torch
import torch.nn as nn

class DeepPromptHyperNet(nn.Module):
    def __init__(self, in_dim=256, out_dim=768, K=8, hidden=512, dropout=0.0):
        super().__init__()
        self.K = K
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, K * out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: [B, in_dim] or [in_dim]
        return: deep prompt tokens [B, K, out_dim]
        """
        singleton = False
        if u.dim() == 1:
            u = u.unsqueeze(0)
            singleton = True
        B = u.size(0)
        x = self.proj(u)  # [B, K*out_dim]
        x = x.view(B, self.K, -1)  # [B,K,D]
        # normalize per token
        x = self.norm(x)
        if singleton:
            x = x.squeeze(0)
        return x  # [K,D] or [B,K,D]
