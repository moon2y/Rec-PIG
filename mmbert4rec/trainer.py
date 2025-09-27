import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import json
import logging

logger = logging.getLogger("mmbert4rec")

PAD, MASK = 0, 1
OFFSET = 2

def _core(model):
    """Return the underlying module if DataParallel is used."""
    return model.module if hasattr(model, "module") else model

def _maybe_unpack(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None

def train_epoch(model, loader, optimizer, device):
    """Training with full softmax for stability — single backward per batch."""
    model.train()
    core = _core(model)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        seq, labels, _ = _maybe_unpack(batch)
        seq    = seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward once per batch
        out = model(seq)  # [B,L,H]
        B, L, H = out.shape

        # Masked positions (compute all at once)
        mask = labels != -100                      # [B,L]
        if not mask.any():
            continue

        # Precompute all item embeddings once per batch (no grad path needed)
        with torch.no_grad():
            all_items = torch.arange(core.n_items, device=device, dtype=torch.long)
            item_embs = core.fusion_table(all_items)  # [M,H]

        # Gather hidden states / targets at masked positions
        h = out[mask]                # [N_mask, H]
        y = labels[mask]             # [N_mask]
        # Full-softmax logits against all items
        logits = torch.matmul(h, item_embs.T)  # [N_mask, M]
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg = (total_loss / num_batches) if num_batches > 0 else 0.0
    logger.info(f"TrainEpoch Loss={avg:.6f}")
    return avg

def _sample_negatives(n_items, pos_ids, K):
    B = pos_ids.shape[0]
    negs = []
    max_negs = max(1, min(K, n_items - 1))
    for i in range(B):
        pool = set()
        while len(pool) < max_negs:
            cand = random.randint(0, n_items - 1)
            if cand != int(pos_ids[i]):
                pool.add(cand)
        negs.append(list(pool))
    return torch.tensor(negs, dtype=torch.long)

def evaluate(model, loader, device, K_eval=10, num_negs=999):
    """Vectorized evaluation with shared negatives per batch.
    - 마스킹된 위치들을 한 번에 모아 평가
    - 배치 공통 음수 샘플 num_negs개 사용
    - 순위 계산 시 항상 +1 (rank ∈ [1..])
    """
    model.eval()
    core = _core(model)

    hits = 0.0
    ndcgs = 0.0
    mrrs = 0.0
    total = 0.0

    with torch.no_grad():
        for seq, labels, _ in tqdm(loader, desc="Eval", leave=False):
            seq    = seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Encoder forward
            out = model(seq)  # [B, L, H]

            # 모든 마스킹 위치 수집
            mask = (labels != -100)            # [B, L] (bool)
            if not mask.any():
                continue
            h   = out[mask]                    # [N, H]
            pos = labels[mask]                 # [N], 내부 아이템 인덱스 0..M-1
            N   = h.size(0)

            # 배치 공유 네거티브 샘플
            K = min(int(num_negs), core.n_items - 1)
            neg_ids = torch.randint(0, core.n_items, (K,), device=device)  # [K]

            # 로짓 계산 (벡터화)
            pos_vec = core.fusion_table(pos)        # [N, H]
            neg_vec = core.fusion_table(neg_ids)    # [K, H]

            pos_log = (h * pos_vec).sum(dim=1, keepdim=True)  # [N, 1]
            neg_log = h @ neg_vec.T                           # [N, K]

            # 순위 계산: 정답보다 큰 후보 수 + 1
            better = (neg_log > pos_log).sum(dim=1).float()   # [N]
            ranks = better + 1.0                              # [N], 최소 1

            # 메트릭 집계
            total += float(N)
            hits  += float((ranks <= K_eval).sum().item())
            ndcgs += float((1.0 / torch.log2(ranks + 1)).sum().item())  # DCG with rank>=1
            mrrs  += float((1.0 / ranks).sum().item())

    if total == 0:
        return 0.0, 0.0, 0.0
    return hits / total, ndcgs / total, mrrs / total

def save_checkpoint(model, optimizer, epoch, ckpt_dir, tag):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"bert4rec_clip_epoch{epoch:03d}_{tag}.pt"

    core = _core(model)
    state_dict = core.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": state_dict,
            "optim_state": optimizer.state_dict(),
            "n_items": core.n_items,
            "hidden": core.hidden,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")
    return str(path)

def export_user_embeddings(model, loader, device, save_path, agg="mean"):
    """Export per-user preference embeddings to save_path (npy + json)."""
    core = _core(model)
    core.eval()
    user_vecs = {}
    import numpy as np

    from tqdm import tqdm
    for seq, labels, users in tqdm(loader, desc="Export user embeddings", leave=False):
        with torch.no_grad():
            seq = seq.to(device, non_blocking=True)
            out = core(seq)                 # [B,L,H]
            pad_mask = (seq == PAD)
            not_pad = (~pad_mask).float()
            if agg == "mean":
                sums = (out * not_pad.unsqueeze(-1)).sum(dim=1)
                cnts = not_pad.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
                embs = sums / cnts
            else:  # 'last'
                rev = torch.flip(not_pad, dims=[1])
                last = not_pad.size(1) - 1 - rev.argmax(dim=1)
                embs = out[torch.arange(out.size(0)), last, :]
            for u, e in zip(users.numpy(), embs.cpu().numpy()):
                user_vecs[int(u)] = e

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    npy = np.stack([user_vecs[k] for k in sorted(user_vecs.keys())], axis=0)
    np.save(save_path, npy)
    (save_path.parent / (save_path.stem + "_users.json")).write_text(
        json.dumps({"users": list(map(int, sorted(user_vecs.keys())))}, indent=2, ensure_ascii=False)
    )
    logger.info(f"User embeddings saved: {save_path}")
    return str(save_path)
