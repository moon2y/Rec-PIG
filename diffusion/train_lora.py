"""
PEFT-LoRA fine-tuning with cached VAE latents. (DDP 버전)
- 입력: preprocessing.latents.py가 만든 index.jsonl (user_id, item_id, latent_path, prompt)
- 유저 임베딩: PIG/embedding/user/user_pref_embs_all.npy (+ _users.json)
- 텍스트 임베딩 사전 캐싱 없음: 배치마다 tokenizer + text_encoder 호출
- 학습 대상: (A) UNet의 LoRA 어댑터(PEFT), (옵션) (B) Proj_u2txt
- 동결: SD v1.5의 VAE/TextEncoder/UNet 본체 (LoRA 어댑터만 학습)
- 멀티GPU: torch.distributed + DistributedDataParallel (각 GPU 1프로세스)

실행 예:
torchrun --nproc_per_node=4 -m diffusion.train_lora \
  --index ./embedding/vae/index.jsonl \
  --user_emb_npy ./embedding/user/user_pref_embs_all.npy \
  --user_emb_json ./embedding/user/user_pref_embs_all_users.json \
  --batch 1 --grad_accum 8 --epochs 5 --lr 1e-4 --train_proj \
  --lora_r 16 --lora_alpha 16 --lora_dropout 0.0 \
  --xformers --grad_checkpoint --num_workers 8
"""

import argparse, json, os, random
from contextlib import nullcontext, ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from peft import LoraConfig, get_peft_model  # PEFT 백엔드

from .paths import ROOT, CHECKPOINT_DIR, LOG_DIR
from .utils import setup_logger, load_or_init_proj


# ---------------- DDP helpers ----------------
def setup_distributed():
    """torchrun을 통한 분산 환경 초기화"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int):
    return rank == 0


# ---------------- autocast 호환 헬퍼 ----------------
def autocast_ctx(enabled: bool):
    if hasattr(torch, "autocast"):
        return torch.autocast("cuda", enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled=enabled)


# ---------------- Dataset ----------------
class LatentDataset(Dataset):
    """
    index.jsonl의 각 줄:
    {"user_id": 123, "item_id": 456, "latent_path": "embedding/vae/user123/movie456.pt", "prompt": "a movie poster"}
    """
    def __init__(self, index_jsonl: str, allowed_uids=None, logger=None):
        text = Path(index_jsonl).read_text().strip()
        rows = [json.loads(l) for l in text.splitlines()] if text else []
        total = len(rows)
        if allowed_uids is not None:
            rows = [r for r in rows if int(r["user_id"]) in allowed_uids]
        self.rows = rows
        self.root = ROOT

        if logger is not None:
            logger.info(
                f"Dataset loaded: total={total}, kept={len(self.rows)}, filtered_out={total - len(self.rows)}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        uid = int(ex["user_id"])
        mid = int(ex["item_id"])
        prompt = ex.get("prompt", "a movie poster")
        lat_path = self.root / ex["latent_path"]
        obj = torch.load(lat_path, map_location="cpu")
        lat = obj["latent"]  # [4,64,64], float16/float32
        return {
            "user_id": uid,
            "item_id": mid,
            "latent": lat,
            "prompt": prompt
        }


def collate_fn(batch):
    uids = torch.tensor([b["user_id"] for b in batch], dtype=torch.long)
    mids = torch.tensor([b["item_id"] for b in batch], dtype=torch.long)
    lats = torch.stack([b["latent"] for b in batch], dim=0)  # [B,4,64,64]
    lats = lats.contiguous(memory_format=torch.channels_last)
    prompts = [b["prompt"] for b in batch]
    return {"user_id": uids, "item_id": mids, "latent": lats, "prompts": prompts}


# ---------------- utils ----------------
def load_all_user_embeddings(user_emb_npy: Path, user_emb_json: Path):
    embs = np.load(user_emb_npy)  # [U, D]
    meta = json.loads(user_emb_json.read_text())
    users = [int(u) for u in meta["users"]]
    if embs.shape[0] != len(users):
        raise ValueError(f"Row mismatch: npy={embs.shape[0]} vs users={len(users)}")
    uid2idx = {u: i for i, u in enumerate(users)}
    return embs, uid2idx


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- PEFT-LoRA helpers ----------------
def add_peft_lora_to_unet(unet, r, alpha, dropout, device, dtype, logger=None):
    # 1) 본체 freeze
    for p in unet.parameters():
        p.requires_grad = False

    # 2) LoRA 설정
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        task_type="FEATURE_EXTRACTION",
    )

    # 3) UNet 내부에 LoRA 어댑터 주입 (get_peft_model로 감싸지 않음)
    unet.add_adapter(lora_cfg)

    # 4) 디바이스/정밀도 정렬 (본체 FP16로 줄이기)
    unet.to(device, dtype=dtype)

    # 5) 학습 대상(LoRA)만 FP32로 되돌리기
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()     # FP32
            # p.grad는 step 시점에 생기므로 여기서 변환 필요 없음

    # 6) 로깅
    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in unet.parameters())
    if logger:
        dtypes = {p.dtype for p in trainable}
        logger.info(f"trainable params: {n_train:,} || all params: {n_all:,} || trainable%: {100*n_train/n_all:.4f}")
        logger.info(f"trainable param dtypes: {dtypes}")  # ← {torch.float32} 가 떠야 정상

    return unet

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default=str(ROOT / "embedding" / "vae" / "index.jsonl"))
    ap.add_argument("--user_emb_npy", type=str, default=str(ROOT / "embedding" / "user" / "user_pref_embs_all.npy"))
    ap.add_argument("--user_emb_json", type=str, default=str(ROOT / "embedding" / "user" / "user_pref_embs_all_users.json"))

    ap.add_argument("--batch", type=int, default=1, help="GPU당 배치 크기 (DDP 기준)")
    ap.add_argument("--grad_accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every_steps", type=int, default=2000)

    ap.add_argument("--train_proj", action="store_true", help="유저 임베딩→텍스트 은닉(1 token) 프로젝션 학습")
    ap.add_argument("--K", type=int, default=0, help="(사용 안함) Deep Prompt legacy placeholder")

    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true")

    # LoRA 하이퍼파라미터 (PEFT)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    args = ap.parse_args()

    # DDP setup
    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = is_main_process(rank)

    # paths
    if is_main:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # seed (rank offset)
    seed_everything(args.seed + rank)

    # device/dtype
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if distributed else 0)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32

    # logger
    if is_main:
        logger = setup_logger(f"train_lora_peft_r{rank}", f"train_lora_peft_rank{rank}.log")
    else:
        logger = logging.getLogger(f"train_lora_peft_r{rank}")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.WARNING)

    # 1) user embeddings
    embs_np, uid2idx = load_all_user_embeddings(Path(args.user_emb_npy), Path(args.user_emb_json))
    embs = torch.from_numpy(embs_np).float().to(device)  # [U, D]
    allowed_uids = set(uid2idx.keys())
    if is_main:
        logger.info(f"Loaded user embeddings: {embs.shape} | allowed_uids={len(allowed_uids)}")

    # 2) dataset/loader
    ds = LatentDataset(args.index, allowed_uids=allowed_uids, logger=logger if is_main else None)
    if len(ds) == 0:
        if is_main:
            logger.error("No training samples found in index.jsonl after filtering with allowed_uids")
        cleanup_distributed()
        return

    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True if distributed else False,
    )
    if is_main:
        logger.info(f"Dataset size={len(ds)} | per-GPU batch={args.batch} | workers={args.num_workers}")

    # 3) pipeline (각 프로세스의 개별 GPU에 로드)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        dtype=dtype,
        safety_checker=None,   # 선택: 안전필터 비활성화(불필요한 의존성/오버헤드 제거)
    )

    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            if is_main: logger.info("Enabled xFormers memory efficient attention.")
        except Exception as e:
            if is_main: logger.warning(f"xFormers not available: {e}")
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass

    pipe = pipe.to(device)
    vae = pipe.vae
    text_encoder: CLIPTextModel = pipe.text_encoder
    tokenizer: CLIPTokenizer = pipe.tokenizer
    unet = pipe.unet
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # gradient checkpointing (LoRA에도 유효)
    if args.grad_checkpoint:
        try:
            unet.enable_gradient_checkpointing()
            if is_main: logger.info("Enabled UNet gradient checkpointing.")
        except Exception as e:
            if is_main: logger.warning(f"Grad checkpoint not enabled: {e}")

    # 고정(본체)
    for m in [vae, text_encoder, unet]:
        for p in m.parameters():
            p.requires_grad = False
    vae.eval(); text_encoder.eval(); unet.eval()

    # 3.5) UNet에 PEFT-LoRA 주입
    unet = add_peft_lora_to_unet(
        unet,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        device=device,
        dtype=dtype,
        logger=logger if is_main else None
    )

    # 4) 학습 모듈 준비
    in_dim = embs.shape[1]
    out_dim = text_encoder.config.hidden_size  # 768

    proj = load_or_init_proj(device=device, in_dim=in_dim, out_dim=out_dim).to(device)
    if not args.train_proj:
        for p in proj.parameters():
            p.requires_grad = False
        proj.eval()

    # DDP 래핑
    if distributed:
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank,
                   broadcast_buffers=False, find_unused_parameters=False)
        if args.train_proj:
            proj = DDP(proj, device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False, find_unused_parameters=False)

    # 옵티마이저 파라미터 (LoRA + (옵션) proj)
    trainable_params = []
    if isinstance(unet, DDP):
        trainable_params += [p for p in unet.module.parameters() if p.requires_grad]
    else:
        trainable_params += [p for p in unet.parameters() if p.requires_grad]
    if args.train_proj:
        if isinstance(proj, DDP):
            trainable_params += [p for p in proj.module.parameters() if p.requires_grad]
        else:
            trainable_params += [p for p in proj.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=use_fp16)

    # ddp 모델 리스트 (no_sync용)
    ddp_models = []
    if distributed:
        ddp_models.append(unet)
        if args.train_proj:
            ddp_models.append(proj)

    # 5) train loop
    step = 0
    for ep in range(1, args.epochs + 1):
        if distributed:
            sampler.set_epoch(ep)

        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        data_iter = tqdm(dl, desc=f"Epoch {ep} [r{rank}]", disable=not is_main)
        for it, batch in enumerate(data_iter, start=1):
            # 유저 임베딩 수집
            uids_list = batch["user_id"].tolist()
            u_batch = torch.stack([embs[uid2idx[int(u)]] for u in uids_list], dim=0).to(device)  # [B,in_dim]
            u_batch = u_batch / (u_batch.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            # noise 준비
            latents = batch["latent"].to(device).to(dtype)
            B = latents.size(0)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, t)

            # 텍스트 인코딩
            text_inputs = tokenizer(
                batch["prompts"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                with autocast_ctx(use_fp16):
                    base_embeds = text_encoder(input_ids=text_inputs.input_ids)[0]  # [B, T, 768]

            # grad accumulation 동안 통신 억제 (no_sync)
            use_no_sync = distributed and ((it % args.grad_accum) != 0)
            sync_ctx = ExitStack()
            if use_no_sync and len(ddp_models) > 0:
                for m in ddp_models:
                    sync_ctx.enter_context(m.no_sync())
            else:
                sync_ctx.enter_context(nullcontext())

            with sync_ctx:
                with autocast_ctx(use_fp16):
                    # (옵션) 유저 임베딩을 1개의 추가 토큰으로 붙임
                    if args.train_proj:
                        e_user = (proj(u_batch).unsqueeze(1)).to(base_embeds.dtype)  # [B,1,768]
                        cond = torch.cat([base_embeds, e_user], dim=1)               # [B, T+1, 768]
                    else:
                        cond = base_embeds

                    # UNet(PEFT-LoRA 주입됨)
                    pred = (unet(noisy, t, encoder_hidden_states=cond).sample
                            if isinstance(unet, DDP)
                            else unet(noisy, t, encoder_hidden_states=cond).sample)
                    loss = ((pred - noise) ** 2).mean()

                loss_to_bp = loss / max(1, args.grad_accum)
                scaler.scale(loss_to_bp).backward()

            running += float(loss.item())
            step += 1

            # 스텝/업데이트
            if (it % args.grad_accum == 0):
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # 로그 (rank0만)
            if is_main and (step % args.log_every == 0):
                avg = running / args.log_every
                logger.info(f"[ep {ep} step {step}] loss={loss.item():.4f} avg={avg:.4f} B={B}")
                running = 0.0

            # 중간 저장 (rank0만, PEFT 저장)
            if is_main and args.save_every_steps > 0 and (step % args.save_every_steps == 0):
                _unet = unet.module if (distributed and isinstance(unet, DDP)) else unet
                lora_dir = CHECKPOINT_DIR / f"lora_unet_step{step}"
                lora_dir.mkdir(parents=True, exist_ok=True)
                _unet.save_pretrained(lora_dir)  # PEFT 어댑터 저장
                if args.train_proj:
                    _proj = proj.module if (distributed and isinstance(proj, DDP)) else proj
                    torch.save(_proj.state_dict(), CHECKPOINT_DIR / f"proj_u2txt_step{step}.pt")
                logger.info(f"Saved LoRA (PEFT) checkpoint at step {step}")

        # 에폭 저장 (rank0만, PEFT 저장)
        if is_main:
            _unet = unet.module if (distributed and isinstance(unet, DDP)) else unet
            lora_dir = CHECKPOINT_DIR / f"lora_unet_ep{ep}"
            lora_dir.mkdir(parents=True, exist_ok=True)
            _unet.save_pretrained(lora_dir)  # PEFT 어댑터 저장
            if args.train_proj:
                _proj = proj.module if (distributed and isinstance(proj, DDP)) else proj
                torch.save(_proj.state_dict(), CHECKPOINT_DIR / f"proj_u2txt_lora_ep{ep}.pt")
            logger.info(f"[Epoch {ep}] saved LoRA (PEFT) + proj checkpoints.")

        if distributed:
            dist.barrier()

    if is_main:
        logger.info("Training done.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
