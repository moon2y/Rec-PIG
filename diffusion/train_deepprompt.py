"""

  * 손실 옵션: mse / clip / contrastive / mse+clip / mse+contrastive
  * 전역 DPT 사용 여부: --dpt_mode [hyper|global] (기본 hyper = 기존 동작)
  * Mid/Up 위주 삽입(마스킹): --enable_mid, --enable_up_from, (선택) --enable_down
  * DDP fix: user_ctr_head를 분산 시 DDP로 래핑, .no_sync() 호출 대상 필터링

"""

import argparse, json, os, random, logging
from contextlib import nullcontext, ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import GradScaler  # <- 경고 제거 버전
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel

from .paths import ROOT, CHECKPOINT_DIR, LOG_DIR
from .utils import setup_logger, load_or_init_proj
from .hypernet import DeepPromptHyperNet

# ---------------- DDP helpers ----------------
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]) ; world_size = int(os.environ["WORLD_SIZE"]) ; local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank:int):
    return rank == 0

# ---------------- autocast 호환 ----------------
def autocast_ctx(enabled: bool):
    if hasattr(torch, "autocast"):
        return torch.autocast("cuda", enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled=enabled)

# ---------------- Dataset ----------------
class LatentDataset(Dataset):
    def __init__(self, index_jsonl: str, allowed_uids=None, logger=None):
        text = Path(index_jsonl).read_text().strip()
        rows = [json.loads(l) for l in text.splitlines()] if text else []
        total = len(rows)
        if allowed_uids is not None:
            rows = [r for r in rows if int(r["user_id"]) in allowed_uids]
        self.rows = rows
        self.root = ROOT
        if logger is not None:
            logger.info(f"Dataset loaded: total={total}, kept={len(self.rows)}, filtered_out={total - len(self.rows)}")
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        ex = self.rows[idx]
        uid = int(ex["user_id"]) ; mid = int(ex["item_id"]) ; prompt = ex.get("prompt", "a movie poster")
        lat_path = self.root / ex["latent_path"]
        obj = torch.load(lat_path, map_location="cpu")
        lat = obj["latent"]  # [4,64,64]
        return {"user_id": uid, "item_id": mid, "latent": lat, "prompt": prompt}

def collate_fn(batch):
    uids = torch.tensor([b["user_id"] for b in batch], dtype=torch.long)
    mids = torch.tensor([b["item_id"] for b in batch], dtype=torch.long)
    lats = torch.stack([b["latent"] for b in batch], dim=0).contiguous(memory_format=torch.channels_last)
    prompts = [b["prompt"] for b in batch]
    return {"user_id": uids, "item_id": mids, "latent": lats, "prompts": prompts}

# ---------------- Utils ----------------
def load_all_user_embeddings(user_emb_npy: Path, user_emb_json: Path):
    embs = np.load(user_emb_npy)  # [U, D]
    meta = json.loads(user_emb_json.read_text())
    users = [int(u) for u in meta["users"]]
    if embs.shape[0] != len(users):
        raise ValueError(f"Row mismatch: npy={embs.shape[0]} vs users={len(users)}")
    uid2idx = {u:i for i,u in enumerate(users)}
    return embs, uid2idx

def seed_everything(seed: int):
    random.seed(seed) ; np.random.seed(seed) ; torch.manual_seed(seed) ; torch.cuda.manual_seed_all(seed)

def normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# ----- 전역 DPT/손실용 보조 -----
class GlobalDPT(nn.Module):
    def __init__(self, K: int, hidden: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(K, hidden) * 0.02)
    def forward(self, B: int):
        return self.tokens.unsqueeze(0).expand(B, -1, -1)  # [B,K,H]

class ContrastiveHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    def forward(self, u):
        return self.net(u)

def contrastive_loss_sym(z_u, z_i, tau=0.07):
    sim = (z_u @ z_i.t()) / tau
    labels = torch.arange(z_u.size(0), device=z_u.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

def clip_similarity_loss(z_u, z_i):
    return (1 - (z_u * z_i).sum(dim=-1)).mean()

def reconstruct_x0_from_pred(noisy, pred_noise, t, alphas_cumprod):
    a = alphas_cumprod[t].view(-1, 1, 1, 1)
    return (noisy - torch.sqrt(1 - a) * pred_noise) / torch.sqrt(a)

def clip_image_preprocess(img):
    x = (img.clamp(-1,1) + 1) / 2.0
    x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
    return (x - mean) / std

# ----- 레이어 선택적 주입: 마스킹 프로세서 (옵션) -----
try:
    from diffusers.models.attention_processor import AttnProcessor2_0, Attention
    class MaskingAttnProcessor(nn.Module):
        def __init__(self, inner: AttnProcessor2_0, K: int, T: int, allow: bool):
            super().__init__()
            self.inner = inner ; self.K = K ; self.T = T ; self.allow = allow
        def forward(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            if (not self.allow) and (encoder_hidden_states is not None):
                B,L,H = encoder_hidden_states.shape
                if L == self.K + self.T + 1:
                    # DPT(K)와 e_user(마지막 1) 마스킹(0)
                    x = encoder_hidden_states
                    dpt_zero = torch.zeros_like(x[:, :self.K, :])
                    eu_zero  = torch.zeros_like(x[:, -1:, :])
                    encoder_hidden_states = torch.cat([dpt_zero, x[:, self.K:self.K+self.T, :], eu_zero], dim=1)
            return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
    def attach_masking_processors(unet, K: int, T: int, enable_down: bool, enable_mid: bool, enable_up_from: int):
        def wrap(attn2, allow):
            if isinstance(attn2.processor, AttnProcessor2_0):
                attn2.processor = MaskingAttnProcessor(attn2.processor, K=K, T=T, allow=allow)
        # down
        for db in getattr(unet, 'down_blocks', []):
            for blk in getattr(db, 'attentions', []):
                for tb in getattr(blk, 'transformer_blocks', []):
                    if hasattr(tb, 'attn2'): wrap(tb.attn2, allow=enable_down)
        # mid
        mb = getattr(unet, 'mid_block', None)
        if mb is not None:
            for blk in getattr(mb, 'attentions', []):
                for tb in getattr(blk, 'transformer_blocks', []):
                    if hasattr(tb, 'attn2'): wrap(tb.attn2, allow=enable_mid)
        # up
        for i, ub in enumerate(getattr(unet, 'up_blocks', [])):
            allow = (i >= enable_up_from)
            for blk in getattr(ub, 'attentions', []):
                for tb in getattr(ub, 'transformer_blocks', []):
                    if hasattr(tb, 'attn2'): wrap(tb.attn2, allow=allow)
except Exception:
    AttnProcessor2_0 = None
    def attach_masking_processors(*args, **kwargs):
        pass

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default=str(ROOT / "embedding" / "vae" / "index.jsonl"))
    ap.add_argument("--user_emb_npy", type=str, default=str(ROOT / "embedding" / "user" / "user_pref_embs_all.npy"))
    ap.add_argument("--user_emb_json", type=str, default=str(ROOT / "embedding" / "user" / "user_pref_embs_all_users.json"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--train_proj", action="store_true")  # 원본 유지 (e_user)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every_steps", type=int, default=2000)
    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--dpt_mode", type=str, default="hyper", choices=["hyper","global"], help="전역 DPT 사용 여부")
    ap.add_argument("--loss_mode", type=str, default="mse", choices=["mse","clip","contrastive","mse+clip","mse+contrastive"])
    ap.add_argument("--lambda_personal", type=float, default=0.5)
    ap.add_argument("--enable_down", action="store_true")
    ap.add_argument("--enable_mid", action="store_true")
    ap.add_argument("--enable_up_from", type=int, default=1)

    args = ap.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = is_main_process(rank)

    if is_main:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if distributed else 0)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32

    if is_main:
        logger = setup_logger(f"train_batch_r{rank}", f"train_batch_rank{rank}.log")
    else:
        logger = logging.getLogger(f"train_batch_r{rank}")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.propagate = False ; logger.setLevel(logging.WARNING)

    # user embeddings
    embs_np, uid2idx = load_all_user_embeddings(Path(args.user_emb_npy), Path(args.user_emb_json))
    embs = torch.from_numpy(embs_np).float().to(device)
    allowed_uids = set(uid2idx.keys())
    if is_main:
        logger.info(f"Loaded user embeddings: {embs.shape} | allowed_uids={len(allowed_uids)}")

    # dataset/loader
    ds = LatentDataset(args.index, allowed_uids=allowed_uids, logger=logger if is_main else None)
    if len(ds) == 0:
        if is_main: logger.error("No training samples after filtering with allowed_uids")
        cleanup_distributed() ; return
    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle = False
    else:
        sampler = None ; shuffle = True
    dl = DataLoader(ds, batch_size=args.batch, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=True if args.num_workers>0 else False, prefetch_factor=2 if args.num_workers>0 else None,
                    collate_fn=collate_fn, drop_last=True if distributed else False)
    if is_main:
        logger.info(f"Dataset size={len(ds)} | per-GPU batch={args.batch} | workers={args.num_workers}")

    # pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)
    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            if is_main: logger.info("Enabled xFormers.")
        except Exception as e:
            if is_main: logger.warning(f"xFormers not available: {e}")
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass
    pipe = pipe.to(device)
    vae = pipe.vae ; text_encoder: CLIPTextModel = pipe.text_encoder ; tokenizer: CLIPTokenizer = pipe.tokenizer ; unet = pipe.unet
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    if args.grad_checkpoint:
        try:
            unet.enable_gradient_checkpointing()
            if is_main: logger.info("Enabled UNet gradient checkpointing.")
        except Exception as e:
            if is_main: logger.warning(f"Grad checkpoint not enabled: {e}")

    for m in [vae, text_encoder, unet]:
        for p in m.parameters(): p.requires_grad = False
    vae.eval(); text_encoder.eval(); unet.eval()

    # 학습 모듈 준비
    in_dim = embs.shape[1]
    out_dim = text_encoder.config.hidden_size

    proj = load_or_init_proj(device=device, in_dim=in_dim, out_dim=out_dim).to(device)  # 원본 e_user projector
    if args.dpt_mode == "hyper":
        hyper = DeepPromptHyperNet(in_dim=in_dim, out_dim=out_dim, K=args.K, hidden=512, dropout=0.0).to(device)
        dpt_global = None
    else:
        hyper = None
        dpt_global = GlobalDPT(K=args.K, hidden=out_dim).to(device)

    # personalization loss 구성 요소 (옵션)
    use_personal = (args.loss_mode != "mse")
    if use_personal:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        clip_model.eval()
        for p in clip_model.parameters(): p.requires_grad = False
        clip_dim = clip_model.config.projection_dim
        user_ctr_head = ContrastiveHead(in_dim=in_dim, out_dim=clip_dim).to(device)
    else:
        user_ctr_head = None

    # Optimizer & DDP 래핑
    params = []
    if hyper is not None: params += list(hyper.parameters())
    if dpt_global is not None: params += list(dpt_global.parameters())
    if args.train_proj: params += list(proj.parameters())
    if use_personal and (user_ctr_head is not None): params += list(user_ctr_head.parameters())

    if distributed:
        if hyper is not None:
            hyper = DDP(hyper, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if args.train_proj:
            proj = DDP(proj, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if dpt_global is not None:
            dpt_global = DDP(dpt_global, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if use_personal and (user_ctr_head is not None):
            user_ctr_head = DDP(user_ctr_head, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler("cuda", enabled=use_fp16)

    # attn 주입 레이어 제어
    if (args.enable_down or args.enable_mid or args.enable_up_from is not None) and ('tokenizer' in locals()) and (tokenizer is not None) and (AttnProcessor2_0 is not None):
        T = tokenizer.model_max_length
        enable_mid = True if args.enable_mid else True  # Mid는 기본 허용 권장
        enable_up_from = args.enable_up_from if args.enable_up_from is not None else 0
        attach_masking_processors(unet, K=args.K, T=T, enable_down=args.enable_down, enable_mid=enable_mid, enable_up_from=enable_up_from)

    # ddp_models: DDP 객체만(no_sync 보유) 포함
    ddp_models = []
    if distributed:
        for m in [hyper, proj if args.train_proj else None, dpt_global, user_ctr_head if use_personal else None]:
            if m is not None and hasattr(m, "no_sync"):
                ddp_models.append(m)

    # 학습 루프
    step = 0
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    for ep in range(1, args.epochs + 1):
        if distributed and isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(ep)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        data_iter = tqdm(dl, desc=f"Epoch {ep} [r{rank}]", disable=not is_main)
        for it, batch in enumerate(data_iter, start=1):
            uids_list = batch["user_id"].tolist()
            u_batch = torch.stack([embs[uid2idx[int(u)]] for u in uids_list], dim=0).to(device)  # [B,in_dim]
            u_batch = normalize(u_batch)

            latents = batch["latent"].to(device).to(dtype)
            B = latents.size(0)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, t)

            text_inputs = tokenizer(batch["prompts"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                with autocast_ctx(use_fp16):
                    base_embeds = text_encoder(input_ids=text_inputs.input_ids)[0]  # [B,T,H]

            use_no_sync = distributed and ((it % args.grad_accum) != 0) and (len(ddp_models) > 0)
            sync_ctx = ExitStack()
            if use_no_sync:
                for m in ddp_models:
                    sync_ctx.enter_context(m.no_sync())
            else:
                sync_ctx.enter_context(nullcontext())

            with sync_ctx:
                with autocast_ctx(use_fp16):
                    # e_user 토큰
                    if args.train_proj:
                        e_user = (proj.module if isinstance(proj, DDP) else proj)(u_batch).unsqueeze(1).to(base_embeds.dtype)
                    else:
                        e_user = u_batch.new_zeros((u_batch.size(0),1,out_dim), dtype=base_embeds.dtype)

                    # DPT 토큰
                    if hyper is not None:
                        hyper_call = hyper.module if isinstance(hyper, DDP) else hyper
                        dp_tokens = hyper_call(u_batch).to(base_embeds.dtype)
                    else:
                        dpt_call = dpt_global.module if isinstance(dpt_global, DDP) else dpt_global
                        dp_tokens = dpt_call(B).to(base_embeds.dtype)

                    cond = torch.cat([dp_tokens, base_embeds, e_user], dim=1)  # [B, K+T+1, H]

                    pred = unet(noisy, t, encoder_hidden_states=cond).sample
                    loss_mse = ((pred - noise) ** 2).mean()

                    # personalization loss (옵션)
                    loss_personal = latents.new_zeros(())
                    if use_personal:
                        x0_hat = reconstruct_x0_from_pred(noisy, pred, t, alphas_cumprod).float()
                        with torch.no_grad():
                            imgs = vae.decode(x0_hat).sample  # [-1,1]
                        clip_in = clip_image_preprocess(imgs)
                        with torch.no_grad():
                            z_img = normalize(CLIPModel.get_image_features(clip_model, pixel_values=clip_in).float())
                        ctr_call = user_ctr_head.module if (distributed and isinstance(user_ctr_head, DDP)) else user_ctr_head
                        if "contrastive" in args.loss_mode:
                            z_u = normalize(ctr_call(u_batch).float())
                            loss_personal = contrastive_loss_sym(z_u, z_img, tau=0.07)
                        elif "clip" in args.loss_mode:
                            z_u = normalize(ctr_call(u_batch).float())
                            loss_personal = clip_similarity_loss(z_u, z_img)

                    # 총 손실
                    if args.loss_mode == "mse":
                        loss = loss_mse
                    elif args.loss_mode in ("clip","contrastive"):
                        loss = args.lambda_personal * loss_personal
                    elif args.loss_mode == "mse+clip":
                        loss = loss_mse + args.lambda_personal * loss_personal
                    elif args.loss_mode == "mse+contrastive":
                        loss = loss_mse + args.lambda_personal * loss_personal
                    else:
                        loss = loss_mse

                    loss_to_bp = loss / max(1, args.grad_accum)
                    scaler.scale(loss_to_bp).backward()

            if (it % args.grad_accum == 0):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

            if is_main and (it % args.log_every == 0):
                logger.info(f"[ep {ep} step {it}] loss={loss.item():.4f} mse={loss_mse.item():.4f} pers={(float(loss_personal.item()) if use_personal else 0.0):.4f} B={B}")

        # (에포크 저장)
        if is_main:
            _proj = proj.module if (distributed and isinstance(proj, DDP)) else proj
            if hyper is not None:
                _hyper = hyper.module if (distributed and isinstance(hyper, DDP)) else hyper
                torch.save(_hyper.state_dict(), CHECKPOINT_DIR / f"dpt_hypernet_lp{args.lambda_personal}_ep{ep}.pt")
            if dpt_global is not None:
                _dpt = dpt_global.module if (distributed and isinstance(dpt_global, DDP)) else dpt_global
                torch.save(_dpt.state_dict(), CHECKPOINT_DIR / f"dpt_global_lp{args.lambda_personal}_ep{ep}.pt")
            if args.train_proj:
                torch.save(_proj.state_dict(), CHECKPOINT_DIR / f"proj_u2txt_dpt_lp{args.lambda_personal}_ep{ep}.pt")
            if use_personal and (user_ctr_head is not None):
                _ctr = user_ctr_head.module if (distributed and isinstance(user_ctr_head, DDP)) else user_ctr_head
                torch.save(_ctr.state_dict(), CHECKPOINT_DIR / f"user_ctr_head_lp{args.lambda_personal}_ep{ep}.pt")
            logger.info(f"[Epoch {ep}] checkpoints saved.")

        if distributed:
            dist.barrier()

    if is_main:
        logger.info("Training done.")
    cleanup_distributed()

if __name__ == "__main__":
    main()
