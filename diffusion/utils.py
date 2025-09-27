# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from .paths import USER_EMB_NPY, USER_EMB_JSON, PROJ_U2TXT_CKPT, LOG_DIR
import logging, sys

def setup_logger(name="diffusion", filename="run.log"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_user_embeddings() -> Tuple[np.ndarray, Dict[int, int]]:
    """Return (embs [U, D], uid2idx {uid->row})"""
    embs = np.load(USER_EMB_NPY)  # [U, D]
    meta = json.loads(Path(USER_EMB_JSON).read_text())
    users = meta["users"]
    uid2idx = {int(u): i for i, u in enumerate(users)}
    return embs, uid2idx

def get_user_vector(user_id: int) -> torch.Tensor:
    embs, uid2idx = load_user_embeddings()
    if user_id not in uid2idx:
        raise KeyError(f"user_id {user_id} not found in {USER_EMB_JSON}")
    vec = embs[uid2idx[user_id]]
    v = torch.from_numpy(vec).float()
    # L2 normalize for stability
    v = v / (v.norm(p=2) + 1e-8)
    return v

class ProjU2Txt(nn.Module):
    """Simple projection u(256) -> text embedding(768) for SD1.5 CLIP text encoder.
    You can enlarge to MLP with residual & LayerNorm if needed.
    """
    def __init__(self, in_dim=256, out_dim=768, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = self.net(u)
        # match token embedding scale by LN
        return self.norm(x)

def save_state(obj: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj.state_dict(), path)

def load_or_init_proj(device="cuda", path: Path = PROJ_U2TXT_CKPT, in_dim=256, out_dim=768):
    model = ProjU2Txt(in_dim=in_dim, out_dim=out_dim).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
    return model
