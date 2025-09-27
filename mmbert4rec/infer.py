# PIG/mmbert4rec/export_all_user_embs.py
# -*- coding: utf-8 -*-
"""
모든 유저 선호도 임베딩 추출 스크립트
- 주어진 학습 체크포인트를 로드
- RecBole atomic .item/.inter로 전체 유저 시퀀스 로드 (full sequence)
- Transformer를 통과한 히든을 PAD 제외 평균(mean)으로 집약 → 유저 임베딩
- 결과를 PIG/embedding/ 아래에 저장:
    - user_pref_embs_all.npy          # [num_users, H]
    - user_pref_embs_all_users.json   # {"users":[...]} (저장 순서와 매핑)
사용 예:
python export_all_user_embs.py \
  --ckpt ../mmbert4rec/checkpoints_mllatest/bert4rec_clip_epoch100_hit0.7269_ndcg0.6040_mrr0.5042.pt
"""

import argparse
from pathlib import Path
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 로컬 모듈
from data import load_movielens, SeqDataset, PAD
from model import BERT4Rec

# ---------- 기본 경로 ----------
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0] # PIG/
EMB_DIR = ROOT / "embedding" / "user"        # 요청: PIG/embedding에 저장
EMB_DIR.mkdir(parents=True, exist_ok=True)

# run.py 기본과 동일한 디폴트
DEFAULT_ITEM = ROOT / "data" / "preprocessed" / "ml-latest.item"
DEFAULT_INTER = ROOT / "data" / "preprocessed" / "ml-latest.inter"
DEFAULT_TEXT = ROOT / "embedding" / "clip" / "mllatest_plots.npy"
DEFAULT_IMG  = ROOT / "embedding" / "clip" / "mllatest_posters.npy"

def build_all_users_loader(item_path, inter_path, max_len=50, batch_size=256):
    """
    전체 유저의 'full sequence' 를 쓰는 DataLoader.
    기존 SeqDataset(phase='test')는 full seq를 사용하므로 그대로 재사용.
    """
    user2seq, n_items = load_movielens(item_path, inter_path, max_len)
    ds = SeqDataset(user2seq, n_items, max_len=max_len, phase="test")  # full seq
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    return dl, n_items

@torch.no_grad()
def export_user_embeddings(model, loader, device, save_path):
    """
    모든 유저 임베딩 추출: PAD 제외 평균(mean pooling).
    - save_path: .../user_pref_embs_all.npy
    - users json: .../user_pref_embs_all_users.json
    """
    core = model.module if hasattr(model, "module") else model
    core.eval()

    user_vecs = {}
    for seq, labels, users in tqdm(loader, desc="Export all user embeddings"):
        seq = seq.to(device, non_blocking=True)
        out = core(seq)  # [B, L, H]
        pad_mask = (seq == PAD)
        not_pad = (~pad_mask).float()
        sums = (out * not_pad.unsqueeze(-1)).sum(dim=1)
        cnts = not_pad.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        embs = sums / cnts  # [B, H]
        for u, e in zip(users.cpu().numpy().tolist(), embs.cpu().numpy()):
            user_vecs[int(u)] = e

    # 정렬 저장
    users_sorted = sorted(user_vecs.keys())
    npy = np.stack([user_vecs[u] for u in users_sorted], axis=0)
    np.save(save_path, npy)
    (save_path.parent / (save_path.stem + "_users.json")).write_text(
        json.dumps({"users": users_sorted}, indent=2, ensure_ascii=False)
    )
    return str(save_path), users_sorted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="학습된 체크포인트 경로 (*.pt)")
    ap.add_argument("--item_path", type=str, default=str(DEFAULT_ITEM))
    ap.add_argument("--inter_path", type=str, default=str(DEFAULT_INTER))
    ap.add_argument("--text_path", type=str, default=str(DEFAULT_TEXT))
    ap.add_argument("--img_path",  type=str, default=str(DEFAULT_IMG))
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--batch",   type=int, default=256)
    # 학습 당시 설정 추정/재현 (hidden은 ckpt에 저장되어 있음, 나머지는 기본값과 동일했을 가능성 큼)
    ap.add_argument("--heads",  type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--fusion", type=str, default="concat", choices=["concat","wsum","gate"])
    ap.add_argument("--out_name", type=str, default="user_pref_embs_all.npy",
                    help="PIG/embedding/ 아래 저장될 파일명")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # 데이터 & 로더
    loader, n_items = build_all_users_loader(args.item_path, args.inter_path,
                                             max_len=args.max_len, batch_size=args.batch)

    # CLIP 임베딩
    text_np = np.load(args.text_path)  # [M, D_t]
    img_np  = np.load(args.img_path)   # [M, D_v]

    # 장치
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 생성 (hidden은 ckpt에서 읽음, heads/layers/fusion/max_len은 인자로 받음)
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden_ckpt = int(ckpt.get("hidden", 256))
    n_items_ckpt = int(ckpt.get("n_items", n_items))
    if n_items_ckpt != n_items:
        # 아이템 개수가 다르면 상태 로드가 어긋날 수 있음 (보통은 동일해야 정상)
        print(f"[Warn] n_items mismatch: ckpt={n_items_ckpt}, data={n_items}. 데이터와 ckpt의 아이템 정렬/개수를 확인하세요.")

    model = BERT4Rec(
        text_np, img_np, n_items,
        hidden=hidden_ckpt, n_heads=args.heads, n_layers=args.layers,
        max_len=args.max_len, fusion=args.fusion
    ).to(device)

    # 가중치 로드
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[Info] state_dict load: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:   print("  missing:", missing)
        if unexpected:print("  unexpected:", unexpected)

    model.eval()

    # 추출 & 저장
    save_path = EMB_DIR / args.out_name
    out_path, users_sorted = export_user_embeddings(model, loader, device, save_path)
    print(f"[Done] Saved all-user embeddings: {out_path}")
    print(f"[Info] Num users: {len(users_sorted)}")
    print(f"[Info] Mapping json: {save_path.parent / (save_path.stem + '_users.json')}")

if __name__ == "__main__":
    main()
