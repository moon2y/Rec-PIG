# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, MASK = 0, 1
OFFSET = 2

class FusionEmbedding(nn.Module):
    """Text & Image CLIP tables + fusion → hidden."""
    def __init__(self, text_embs, img_embs, hidden_size, fusion="concat", freeze=True):
        super().__init__()
        text = torch.tensor(text_embs, dtype=torch.float32)
        img  = torch.tensor(img_embs,  dtype=torch.float32)

        self.text_table = nn.Embedding.from_pretrained(text, freeze=freeze, padding_idx=None)
        self.img_table  = nn.Embedding.from_pretrained(img,  freeze=freeze, padding_idx=None)

        d_t, d_v = text.shape[1], img.shape[1]
        self.proj_t = nn.Linear(d_t, hidden_size, bias=False)
        self.proj_v = nn.Linear(d_v, hidden_size, bias=False)

        self.fusion = fusion
        if fusion == "concat":
            self.proj = nn.Linear(2*hidden_size, hidden_size)
        elif fusion == "wsum":
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion == "gate":
            self.gate = nn.Linear(2*hidden_size, 1)

        nn.init.xavier_uniform_(self.proj_t.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        if hasattr(self, "proj"):
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, idx):  # idx: item indices [*,] in [0..M-1]
        t = F.normalize(self.text_table(idx), p=2, dim=-1)
        v = F.normalize(self.img_table(idx),  p=2, dim=-1)
        t = self.proj_t(t)
        v = self.proj_v(v)
        if self.fusion == "concat":
            return self.proj(torch.cat([t, v], dim=-1))
        elif self.fusion == "wsum":
            a = torch.sigmoid(self.alpha)
            return a*t + (1-a)*v
        else:
            g = torch.sigmoid(self.gate(torch.cat([t, v], dim=-1)))
            return g*t + (1-g)*v

class BERT4Rec(nn.Module):
    """BERT4Rec with CLIP-fused item embeddings (no RecBole)."""
    def __init__(self, text_embs, img_embs, n_items, hidden=256, n_heads=4, n_layers=2, max_len=50, fusion="concat"):
        super().__init__()
        self.n_items = n_items  # M
        self.hidden = hidden

        self.fusion_table = FusionEmbedding(text_embs, img_embs, hidden, fusion=fusion, freeze=True)
        self.mask_emb = nn.Parameter(torch.randn(hidden))
        self.pad_emb  = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.pos_emb  = nn.Embedding(max_len, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def embed_tokens(self, seq):
        """seq: [B,L] with PAD=0, MASK=1, item=2+idx."""
        B, L = seq.shape
        embs = torch.zeros(B, L, self.hidden, device=seq.device)
        mask_pos = (seq == MASK)
        pad_pos  = (seq == PAD)
        item_pos = (seq >= OFFSET)

        if item_pos.any():
            embs[item_pos] = self.fusion_table(seq[item_pos] - OFFSET)
        if mask_pos.any():
            embs[mask_pos] = self.mask_emb
        if pad_pos.any():
            embs[pad_pos] = self.pad_emb

        pos_ids = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        embs = embs + self.pos_emb(pos_ids)
        return embs, pad_pos

    def forward(self, seq):
        embs, pad_pos = self.embed_tokens(seq)  # [B,L,H], [B,L]
        pad_pos = pad_pos.to(torch.bool)
        out = self.encoder(embs, src_key_padding_mask=pad_pos)  # [B,L,H]
        return out

    def sample_logits(self, h, pos_ids, neg_ids):
        """Compute sampled logits for evaluation.
        h: [B,H], pos_ids: [B], neg_ids: [B,K] (values in [0..M-1])
        returns logits: [B, 1+K] with first column as positive.
        """
        pos_vec = self.fusion_table(pos_ids)                        # [B,H]
        neg_vec = self.fusion_table(neg_ids)                        # [B,K,H]
        pos_logit = (h * pos_vec).sum(-1, keepdim=True)             # [B,1]
        neg_logit = torch.bmm(neg_vec, h.unsqueeze(-1)).squeeze(-1) # [B,K]
        return torch.cat([pos_logit, neg_logit], dim=1)             # [B,K+1]
