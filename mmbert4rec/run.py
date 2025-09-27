import argparse
from pathlib import Path
import logging
import sys
import numpy as np
import torch

from data import get_dataloaders
from model import BERT4Rec
from trainer import train_epoch, evaluate, save_checkpoint, export_user_embeddings

def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    logger = logging.getLogger("mmbert4rec")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ffmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(ffmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    cfmt = logging.Formatter("%(message)s")
    ch.setFormatter(cfmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to: {log_path}")
    return logger

def _warmup_cublas(device_ids):
    # 각 디바이스에서 작은 matmul 1회 실행 → cuBLAS 핸들 초기화
    for d in device_ids:
        with torch.cuda.device(d):
            a = torch.randn(8, 8, device=f"cuda:{d}")
            b = torch.randn(8, 8, device=f"cuda:{d}")
            _ = a @ b
            torch.cuda.synchronize()

def main(args):
    this_dir = Path(__file__).resolve().parent

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (this_dir / "checkpoints_mllatest")
    log_dir  = this_dir.parent / "log" / "mmbert4rec"
    user_emb_dir = this_dir.parent / "embedding" / "user"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    user_emb_dir.mkdir(parents=True, exist_ok=True)
    user_emb_path = user_emb_dir / "user_pref_embs.npy"

    logger = setup_logger(log_dir)
    logger.info(f"Checkpoints dir: {ckpt_dir}")
    logger.info(f"User embedding dir: {user_emb_dir}")

    text_np = np.load(args.text_path)
    img_np  = np.load(args.img_path)
    logger.info(f"Loaded CLIP: text {text_np.shape}, image {img_np.shape}")

    train_loader, valid_loader, test_loader, n_items = get_dataloaders(
        args.item_path, args.inter_path, max_seq_len=args.max_len, batch_size=args.batch
    )
    logger.info(f"Dataset: n_items={n_items}, max_len={args.max_len}, batch={args.batch}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = BERT4Rec(
        text_np, img_np, n_items,
        hidden=args.hidden, n_heads=args.heads, n_layers=args.layers,
        max_len=args.max_len, fusion=args.fusion
    ).to(device)

    # 가용 GPU 자동 탐지 + 워밍업
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        logger.info(f"Using {len(device_ids)} GPUs with DataParallel {device_ids}")
        _warmup_cublas(device_ids)  # cuBLAS 초기화 워밍업
        model = torch.nn.DataParallel(base_model, device_ids=device_ids, output_device=device_ids[0])
    else:
        logger.info("Using single GPU or CPU")
        model = base_model

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ndcg = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        logger.info(f"=== Epoch {epoch}/{args.epochs} ===")
        # non_blocking=True 로 전송 최적화 (trainer 내부에서 seq.to 호출할 때 사용)
        loss = train_epoch(model, train_loader, opt, device)
        hit, ndcg, mrr = evaluate(model, valid_loader, device, K_eval=args.k, num_negs=args.eval_negs)
        logger.info(f"[Epoch {epoch:03d}] TrainLoss={loss:.6f}  Valid@{args.k}  Hit={hit:.6f}  NDCG={ndcg:.6f}  MRR={mrr:.6f}")

        if epoch % 10 == 0:
            tag = f"hit{hit:.4f}_ndcg{ndcg:.4f}_mrr{mrr:.4f}"
            ckpt_path = save_checkpoint(model, opt, epoch, ckpt_dir, tag)
            logger.info(f"Saved checkpoint (every 10 epochs): {ckpt_path}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch

    logger.info(f"[Best Valid] epoch={best_epoch}  NDCG={best_ndcg:.6f}")

    hit, ndcg, mrr = evaluate(model, test_loader, device, K_eval=args.k, num_negs=args.eval_negs)
    logger.info(f"[Test@{args.k}] Hit={hit:.6f}  NDCG={ndcg:.6f}  MRR={mrr:.6f}")

    if args.export_user_embs:
        saved = export_user_embeddings(model, test_loader, device, user_emb_path, agg=args.emb_agg)
        logger.info(f"[Export] User preference embeddings saved to: {saved}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--item_path", default="data/preprocessed/ml-latest.item")
    ap.add_argument("--inter_path", default="data/preprocessed/ml-latest.inter")
    ap.add_argument("--text_path", default="embedding/clip/mllatest_plots.npy")
    ap.add_argument("--img_path",  default="embedding/clip/mllatest_posters.npy")
    # model
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads",  type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--fusion",  type=str, default="concat", choices=["concat","wsum","gate"])
    # train
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr",    type=float, default=1e-3)
    ap.add_argument("--epochs",type=int, default=100)
    # eval (sampled softmax)
    ap.add_argument("--k",          type=int, default=10)
    ap.add_argument("--eval_negs",  type=int, default=999)
    # io
    ap.add_argument("--ckpt_dir",   type=str, default="")
    # export user embeddings
    ap.add_argument("--export_user_embs", action="store_true")
    ap.add_argument("--emb_agg", type=str, default="mean", choices=["mean","last"])
    args = ap.parse_args()
    main(args)
