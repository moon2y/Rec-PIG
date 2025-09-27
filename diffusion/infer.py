import argparse
from pathlib import Path
import json
import csv
import time
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm

from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel

# safetensors / peft (LoRA 강제 로딩용)
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
try:
    from peft.utils.other import set_peft_model_state_dict
except Exception:
    try:
        from peft.utils.save_and_load import set_peft_model_state_dict
    except Exception:
        set_peft_model_state_dict = None

# ---------------- Project paths ----------------
from .paths import (
    RESULTS_DIR, SD15_MODEL_ID, PROJ_U2TXT_CKPT, ROOT, CHECKPOINT_DIR
)
from .utils import setup_logger, get_user_vector, load_or_init_proj

# ---------------- Data locations ----------------
PREP_DIR = ROOT / "data" / "preprocessed"
ITEM_ATOMIC = PREP_DIR / "ml-latest.item"       # (참고용)
INTER_ATOMIC = PREP_DIR / "ml-latest.inter"     # 최근 시청 아이템 조회용

RAW_ML = ROOT / "data" / "raw" / "ml-latest"    # MovieLens links.csv
RAW_KG = ROOT / "data" / "raw" / "ml-latest"    # Kaggle movies_metadata.csv

# TMDb poster URL (no API)
IMG_BASE = "https://image.tmdb.org/t/p/"
IMG_SIZE = "w500"
SLEEP_IMG = 0.02

# ---------------- Helpers ----------------
def load_links():
    """MovieLens links.csv → {movieId: tmdbId}."""
    path = RAW_ML / "links.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Please place MovieLens links.csv there.")
    m = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                mid = int(row["movieId"])
            except Exception:
                continue
            tid = row.get("tmdbId")
            try:
                tmdb = int(float(tid)) if tid and str(tid).strip() != "" else None
            except Exception:
                tmdb = None
            if tmdb is not None:
                m[mid] = tmdb
    return m

def load_kaggle_poster_map():
    """Kaggle movies_metadata.csv → {tmdbId: poster_path}."""
    path = RAW_KG / "movies_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Please place Kaggle movies_metadata.csv there.")
    df = pd.read_csv(path, low_memory=False)
    df = df[["id", "poster_path"]].dropna()

    def to_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    df["id"] = df["id"].map(to_int)
    df = df.dropna(subset=["id"]).astype({"id": int})
    return dict(zip(df["id"].tolist(), df["poster_path"].astype(str).tolist()))

def fetch_image(url: str):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def _resize_to_multiple_of_8(img: Image.Image, target_h: int = 0, target_w: int = 0) -> Image.Image:
    """Resize PIL image so that H, W are multiples of 8."""
    w, h = img.size
    if target_h > 0 and target_w > 0:
        H = int(target_h // 8) * 8
        W = int(target_w // 8) * 8
        H = max(H, 8); W = max(W, 8)
        return img.resize((W, H), Image.BICUBIC)
    H = max((h // 8) * 8, 8)
    W = max((w // 8) * 8, 8)
    if (H, W) != (h, w):
        img = img.resize((W, H), Image.BICUBIC)
    return img

# ---------------- Core: get recent movies for the user ----------------
def get_user_recent_movies(user_id: int, topk: int = 10):
    """
    RecBole atomic .inter에서 해당 유저의 최신 상호작용 아이템을 중복 없이 최근 순서로 topk 추출.
    """
    if not INTER_ATOMIC.exists():
        raise FileNotFoundError(f"Missing {INTER_ATOMIC}. Run your recbole script to build .inter first.")

    inter = pd.read_csv(INTER_ATOMIC, sep="\t")
    need_cols = {"user_id:token", "item_id:token", "timestamp:float"}
    if not need_cols.issubset(inter.columns):
        raise ValueError(f"{INTER_ATOMIC} must contain {need_cols} columns.")

    u_df = inter[inter["user_id:token"] == user_id].copy()
    if u_df.empty:
        return []

    u_df = u_df.sort_values("timestamp:float")
    seen = set()
    recent = []
    for mid in reversed(u_df["item_id:token"].astype(int).tolist()):
        if mid not in seen:
            seen.add(mid)
            recent.append(mid)
        if len(recent) >= topk:
            break
    recent.reverse()
    return recent

# === DPT utils & optional masking processors (match training) ===
class GlobalDPT(nn.Module):
    def __init__(self, K: int, hidden: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(K, hidden) * 0.02)
    def forward(self, B: int):
        return self.tokens.unsqueeze(0).expand(B, -1, -1)

def _normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# (선택) attn2 마스킹 프로세서: 특정 블록에서 DPT/e_user를 0으로 만들어 효과를 제한
try:
    from diffusers.models.attention_processor import AttnProcessor2_0, Attention
    class _MaskingAttnProcessor(nn.Module):
        def __init__(self, inner: AttnProcessor2_0, K: int, T: int, allow: bool):
            super().__init__()
            self.inner = inner ; self.K = K ; self.T = T ; self.allow = allow
        def forward(self, attn: "Attention", hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            if (not self.allow) and (encoder_hidden_states is not None):
                B,L,H = encoder_hidden_states.shape
                if L == self.K + self.T + 1:
                    x = encoder_hidden_states
                    dpt_zero = torch.zeros_like(x[:, :self.K, :])
                    eu_zero  = torch.zeros_like(x[:, -1:, :])
                    encoder_hidden_states = torch.cat([dpt_zero, x[:, self.K:self.K+self.T, :], eu_zero], dim=1)
            return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

    def _attach_masking_processors(unet, K: int, T: int, enable_down: bool, enable_mid: bool, enable_up_from: int):
        def wrap(attn2, allow):
            if isinstance(attn2.processor, AttnProcessor2_0):
                attn2.processor = _MaskingAttnProcessor(attn2.processor, K=K, T=T, allow=allow)
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
                for tb in getattr(blk, 'transformer_blocks', []):
                    if hasattr(tb, 'attn2'): wrap(tb.attn2, allow=allow)
except Exception:
    AttnProcessor2_0 = None
    def _attach_masking_processors(*args, **kwargs):
        pass

# ---------------- DPT Helpers ----------------
@torch.no_grad()
def build_dpt_pos_neg_embeds(
    pipe: StableDiffusionImg2ImgPipeline,
    prompt: str,
    neg_prompt: str,
    user_vec: torch.Tensor,
    proj_module: torch.nn.Module,
    dpt_ckpt_path: str,
    K: int,
    device: str,
    dpt_mode: str = "hyper",   # "hyper" or "global"
):
    """
    Return:
      pos_cond: [1, K+77+1, H] = [dp_tokens, base_pos, e_user]
      neg_cond: [1, K+77+1, H] = [zeros(K), base_neg, zeros(1)]
    """
    tokenizer: CLIPTokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    H = text_encoder.config.hidden_size

    # user -> e_user [1,1,H]
    e_user = proj_module(user_vec).to(text_encoder.dtype).unsqueeze(0).unsqueeze(1)

    # DPT 토큰 [1,K,H] (모드별)
    ckpt_path = Path(dpt_ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DPT checkpoint not found: {ckpt_path}")

    if dpt_mode == "hyper":
        # 학습 코드와 동일한 하이퍼넷 경로
        from .hypernet import DeepPromptHyperNet
        hyper = DeepPromptHyperNet(
            in_dim=user_vec.numel(),
            out_dim=H,
            K=K, hidden=512, dropout=0.0
        ).to(device)
        state = torch.load(str(ckpt_path), map_location=device)
        hyper.load_state_dict(state, strict=True)
        hyper.eval()
        dp_tokens = hyper(user_vec.unsqueeze(0)).to(text_encoder.dtype)   # [1,K,H]
    elif dpt_mode == "global":
        g = GlobalDPT(K=K, hidden=H).to(device)
        state = torch.load(str(ckpt_path), map_location=device)
        g.load_state_dict(state, strict=True)
        g.eval()
        dp_tokens = g(1).to(text_encoder.dtype)                           # [1,K,H]
    else:
        raise ValueError(f"Unknown dpt_mode: {dpt_mode}")

    # base text embeds (pos/neg)  [1,77,H]
    pos_inputs = tokenizer(prompt, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").to(device)
    base_pos = text_encoder(input_ids=pos_inputs.input_ids)[0]

    if neg_prompt is None:
        neg_prompt = ""
    neg_inputs = tokenizer(neg_prompt, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").to(device)
    base_neg = text_encoder(input_ids=neg_inputs.input_ids)[0]

    # concat
    pos_cond = torch.cat([dp_tokens, base_pos, e_user], dim=1)  # [1,K+77+1,H]
    zeros_dp = torch.zeros_like(dp_tokens)
    zeros_user = torch.zeros_like(e_user)
    neg_cond = torch.cat([zeros_dp, base_neg, zeros_user], dim=1)
    return pos_cond, neg_cond

# ---------------- LoRA loader (robust) ----------------
def load_lora_into_unet(
    pipe,
    lora_dir,
    *,
    device="cuda",
    dtype=torch.float16,
    logger=None,
    # ---- optional overrides from caller (모두 선택)
    r=None,
    lora_alpha=None,
    alpha=None,                 # alias for lora_alpha
    target_modules=None,
    lora_scale=1.0,
    lora_dropout=None,
    dropout=None,               # alias for lora_dropout
    weight_name=None,           # 표준 로라 파일명 지정 시
    **unused_kwargs,            # 예기치 않은 추가 키워드가 들어와도 에러없이 무시
):
    """
    Load LoRA trained for UNet.

    우선 순위:
      1) lora_dir에 표준 파일(adapter_model.safetensors / pytorch_lora_weights.{safetensors,bin})이 있으면
         -> diffusers의 pipe.load_lora_weights() 사용, 가능하면 pipe.fuse_lora(scale) 수행
      2) 위 파일이 없고 diffusion_pytorch_model.safetensors + config.json 조합이면
         -> PEFT로 동일 target_modules에 어댑터 장착 후 safetensors 로드
         -> merge_and_unload()로 base UNet에 병합 → pipe.unet 갱신 (PEFT 래퍼 제거)
    """
    if logger and unused_kwargs:
        logger.warning(f"[LoRA] Ignored extra kwargs: {sorted(unused_kwargs.keys())}")

    lora_path = Path(lora_dir).expanduser().resolve()
    if logger:
        logger.info(f"[LoRA] loading from: {lora_path}")

    # 1) 먼저 표준 파일명 확인 (diffusers native)
    std_files = ["adapter_model.safetensors", "pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"]
    if weight_name:
        std_candidates = [weight_name]
    else:
        std_candidates = std_files

    for fname in std_candidates:
        f = lora_path / fname
        if f.exists():
            if logger:
                logger.info(f"[LoRA] Found standard file: {fname} -> using pipe.load_lora_weights + fuse")
            pipe.load_lora_weights(str(lora_path), weight_name=fname)
            try:
                pipe.fuse_lora(lora_scale=lora_scale)
                if logger:
                    logger.info(f"[LoRA] Fused LoRA into UNet. (scale={lora_scale})")
            except Exception as e:
                if logger:
                    logger.warning(f"[LoRA] fuse_lora not available or failed: {e}")
            return pipe

    # 2) 커스텀 케이스: diffusion_pytorch_model.safetensors (+ config.json)
    diffusers_style = lora_path / "diffusion_pytorch_model.safetensors"
    cfg_json = lora_path / "config.json"

    if not diffusers_style.exists() or not cfg_json.exists():
        raise RuntimeError(
            "LoRA/UNet checkpoint not recognized.\n"
            f"Tried files: {std_files + ['diffusion_pytorch_model.safetensors']}\n"
            f"Directory contents: {sorted(p.name for p in lora_path.iterdir())}"
        )

    # 2-1) config.json 파싱
    with open(cfg_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if lora_alpha is None and alpha is not None:
        lora_alpha = alpha
    if lora_dropout is None and dropout is not None:
        lora_dropout = dropout

    r_cfg = int(cfg.get("r", 16))
    alpha_cfg = int(cfg.get("lora_alpha", r_cfg))
    tmods_cfg = cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])
    drop_cfg = float(cfg.get("lora_dropout", 0.0))

    r = int(r) if r is not None else r_cfg
    lora_alpha = int(lora_alpha) if lora_alpha is not None else alpha_cfg
    target_modules = target_modules if target_modules is not None else tmods_cfg
    lora_dropout = float(lora_dropout) if lora_dropout is not None else drop_cfg

    if logger:
        logger.info(f"[LoRA] Using PEFT attach flow (merge). "
                    f"r={r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={target_modules}")

    lcfg = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        inference_mode=True,
    )
    unet = pipe.unet.to(device=device, dtype=dtype)
    peft_unet = get_peft_model(unet, lcfg)
    peft_unet.to(device)

    sd = load_file(str(diffusers_style), device=device)
    normalized = {}
    peft_keys = set(peft_unet.state_dict().keys())
    for k, v in sd.items():
        nk = k
        if nk.startswith("base_model.model."):
            nk = nk[len("base_model.model."):]
        elif nk.startswith("model."):
            nk = nk[len("model."):]
        if nk in peft_keys:
            normalized[nk] = v

    if logger:
        logger.info(f"[LoRA] Loaded {len(sd)} tensors; matched {len(normalized)} to PEFT UNet keys.")

    if len(normalized) == 0:
        peft_unet.load_state_dict(sd, strict=False)
    else:
        peft_unet.load_state_dict(normalized, strict=False)

    try:
        peft_unet.merge_and_unload()
        merged_unet = peft_unet.model
    except Exception:
        merged_unet = getattr(peft_unet, "model", None) or getattr(peft_unet, "base_model", None) or peft_unet
        if logger:
            logger.warning("[LoRA] merge_and_unload not fully supported; using underlying model reference.")

    merged_unet.to(device=device, dtype=torch.float32)
    pipe.unet = merged_unet
    pipe = pipe.to(device=device, dtype=torch.float32)

    if logger:
        logger.info("[LoRA] Merged & set entire pipeline dtype to float32 to avoid dtype mismatches.")
    return pipe

# ---------------- util: proj ckpt 선택 (최신 파일 자동 탐색 포함) ----------------
def _find_latest(glob_pattern: str):
    cands = sorted(CHECKPOINT_DIR.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def resolve_proj_ckpt_path(use_dpt: bool, use_lora: bool,
                           proj_ckpt: str, proj_ckpt_dpt: str, proj_ckpt_lora: str,
                           logger) -> Path:
    """
    모드별 proj_u2txt 체크포인트 경로를 결정.
    우선순위:
      1) --proj_ckpt (명시 시 무조건 우선)
      2) --use_dpt → --proj_ckpt_dpt, 없으면 CHECKPOINT_DIR 내 최신 proj_u2txt_dpt*.pt
      3) --use_lora → --proj_ckpt_lora, 없으면 CHECKPOINT_DIR 내 최신 proj_u2txt_lora*.pt
      4) 마지막으로 PROJ_U2TXT_CKPT 상수
    """
    # 1) 명시적 경로
    if proj_ckpt:
        p = Path(proj_ckpt).expanduser().resolve()
        if p.exists():
            logger.info(f"[proj] Using explicit --proj_ckpt: {p}")
            return p
        logger.warning(f"[proj] --proj_ckpt not found: {p} → fallback")

    # 2) 모드별 전용
    if use_dpt:
        if proj_ckpt_dpt:
            p = Path(proj_ckpt_dpt).expanduser().resolve()
            if p.exists():
                logger.info(f"[proj] Using DPT-specific proj: {p}")
                return p
            logger.warning(f"[proj] --proj_ckpt_dpt not found: {p} → fallback")
        p = _find_latest("proj_u2txt_dpt*.pt")
        if p:
            logger.info(f"[proj] Using inferred latest DPT proj: {p}")
            return p
        logger.warning("[proj] No inferred DPT proj found. → fallback")

    if use_lora:
        if proj_ckpt_lora:
            p = Path(proj_ckpt_lora).expanduser().resolve()
            if p.exists():
                logger.info(f"[proj] Using LoRA-specific proj: {p}")
                return p
            logger.warning(f"[proj] --proj_ckpt_lora not found: {p} → fallback")
        p = _find_latest("proj_u2txt_lora*.pt")
        if p:
            logger.info(f"[proj] Using inferred latest LoRA proj: {p}")
            return p
        logger.warning("[proj] No inferred LoRA proj found. → fallback")

    # 3) 폴백
    p = Path(PROJ_U2TXT_CKPT).expanduser().resolve()
    if p.exists():
        logger.info(f"[proj] Using default PROJ_U2TXT_CKPT: {p}")
        return p
    raise FileNotFoundError(
        f"[proj] No valid proj_u2txt checkpoint found. "
        f"Tried: --proj_ckpt / mode-specific / inferred / PROJ_U2TXT_CKPT ({PROJ_U2TXT_CKPT})"
    )

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # personalization
    ap.add_argument("--user_id", type=int, required=True, help="User ID present in ml-latest.inter (user_id:token)")
    # prompts
    ap.add_argument("--prompt", type=str, default="a cinematic poster, <usr> style, high quality, highly detailed")
    ap.add_argument("--neg_prompt", type=str, default="")
    # generation knobs
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--strength", type=float, default=0.45, help="img2img denoising strength (0~1)")
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=0, help="Optional override height; rounded to /8*8")
    ap.add_argument("--width", type=int, default=0,  help="Optional override width;  rounded to /8*8")
    # fetch limit
    ap.add_argument("--topk", type=int, default=10)
    # DPT options (확장)
    ap.add_argument("--use_dpt", action="store_true", help="Use DeepPrompt tokens for personalization.")
    ap.add_argument("--dpt_mode", type=str, default="hyper", choices=["hyper","global"],
                    help="hyper: DeepPromptHyperNet per-user tokens, global: GlobalDPT shared tokens")
    ap.add_argument("--dpt_ckpt", type=str, default="", help="Path to DPT checkpoint (dpt_hypernet_*.pt or dpt_global_*.pt)")
    ap.add_argument("--K", type=int, default=8, help="Number of DPT tokens; MUST match training.")
    # (optional) masking processors to mirror training injection pattern
    ap.add_argument("--enable_down", action="store_true", help="Allow DPT/e_user in down blocks (default off).")
    ap.add_argument("--enable_mid", action="store_true", help="Allow in mid blocks (default on).")
    ap.add_argument("--enable_up_from", type=int, default=1, help="Allow in up blocks from this index (default=1).")
    # LoRA options
    ap.add_argument("--use_lora", action="store_true", help="Use LoRA-trained UNet adapters for personalization.")
    ap.add_argument("--lora_dir", type=str, default="", help="Directory containing LoRA or UNet checkpoint.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    # Proj ckpt override(s)
    ap.add_argument("--proj_ckpt", type=str, default="", help="Explicit proj_u2txt checkpoint path (overrides mode-specific).")
    ap.add_argument("--proj_ckpt_dpt", type=str, default="", help="DPT-specific proj_u2txt checkpoint path.")
    ap.add_argument("--proj_ckpt_lora", type=str, default="", help="LoRA-specific proj_u2txt checkpoint path.")
    args = ap.parse_args()

    logger = setup_logger("img2img_batch", "img2img_batch.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1) Load pipeline (버전 호환: dtype은 이후 .to에서 통일)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD15_MODEL_ID)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # safety checker는 기본 유지

    # 2) Load user embedding + projection head (모드별 전용 proj ckpt 우선 사용)
    u = get_user_vector(args.user_id).to(device)  # [D], L2-normalized
    proj_path = resolve_proj_ckpt_path(
        use_dpt=args.use_dpt,
        use_lora=args.use_lora,
        proj_ckpt=args.proj_ckpt,
        proj_ckpt_dpt=args.proj_ckpt_dpt,
        proj_ckpt_lora=args.proj_ckpt_lora,
        logger=logger
    )
    proj = load_or_init_proj(
        device=device,
        path=proj_path,  # Path 객체로 전달
        in_dim=u.numel(),
        out_dim=pipe.text_encoder.config.hidden_size
    )
    proj.eval()

    # 2-1) LoRA UNet 어댑터 로드 (선택)
    if args.use_lora:
        if not args.lora_dir:
            raise ValueError("--use_lora requires --lora_dir")
        pipe = load_lora_into_unet(
            pipe, args.lora_dir, device=device, dtype=dtype, logger=logger,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
        )

    # (선택) DPT용 마스킹 프로세서 부착: 학습과 동일한 레이어 허용/차단을 재현
    if args.use_dpt and AttnProcessor2_0 is not None:
        enable_mid = True if args.enable_mid else True
        enable_up_from = args.enable_up_from if args.enable_up_from is not None else 0
        _attach_masking_processors(
            pipe.unet,
            K=args.K,
            T=pipe.tokenizer.model_max_length,
            enable_down=args.enable_down,
            enable_mid=enable_mid,
            enable_up_from=enable_up_from
        )

    # 3) Resolve user's recent movies
    movie_ids = get_user_recent_movies(args.user_id, topk=args.topk)
    if not movie_ids:
        logger.info(f"No recent items found for user {args.user_id}. Nothing to do.")
        return
    logger.info(f"User {args.user_id} recent unique movies: {movie_ids}")

    # 4) Build poster URLs
    links_map = load_links()
    tmdb2path = load_kaggle_poster_map()

    # 5) Output root dir
    if args.use_dpt:
        mode_dir = "img2img_dpt"
    elif args.use_lora:
        mode_dir = "img2img_lora"
    else:
        mode_dir = "img2img"
    user_root = RESULTS_DIR / mode_dir / f"user{args.user_id}"
    user_root.mkdir(parents=True, exist_ok=True)

    # 6) Precompute DPT embeds if needed (hyper/global 지원)
    pos_embeds = None
    neg_embeds = None
    if args.use_dpt:
        if not args.dpt_ckpt:
            raise ValueError("--use_dpt requires --dpt_ckpt path (dpt_hypernet_*.pt or dpt_global_*.pt).")
        pos_embeds, neg_embeds = build_dpt_pos_neg_embeds(
            pipe=pipe,
            prompt=args.prompt,
            neg_prompt=args.neg_prompt,
            user_vec=u,
            proj_module=proj,
            dpt_ckpt_path=args.dpt_ckpt,
            K=args.K,
            device=device,
            dpt_mode=args.dpt_mode,  # << hyper/global 선택
        )

    # 7) Loop over recent movies
    for mid in tqdm(movie_ids, desc="Personalize posters"):
        url = None
        try:
            url = movieid_to_poster_url(mid, links_map, tmdb2path)
        except Exception:
            url = None

        if not url:
            logger.info(f"[Skip] movieId {mid}: no poster URL.")
            continue

        img = fetch_image(url)
        time.sleep(SLEEP_IMG)
        if img is None:
            logger.info(f"[Skip] movieId {mid}: failed to fetch poster.")
            continue

        img = _resize_to_multiple_of_8(img, args.height, args.width)
        movie_dir = user_root / f"movie{mid}"
        movie_dir.mkdir(parents=True, exist_ok=True)

        # Save BEFORE
        before_path = movie_dir / "before.png"
        img.save(before_path)

        # Generate AFTER(s)
        for i in range(args.num_images):
            seed_i = args.seed + i
            gen = torch.Generator(device=device).manual_seed(seed_i)

            if args.use_dpt:
                # 임베딩 직접 전달(encode_prompt 호출 방지): 길이 동일한 pos/neg 사용
                result = pipe(
                    prompt_embeds=pos_embeds,
                    negative_prompt_embeds=neg_embeds,
                    image=img,
                    strength=args.strength,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                )
                suffix = f"dpt-{args.dpt_mode}_seed{seed_i}_str{args.strength:.2f}_{i:02d}.png"
            else:
                # 기존 문자열 프롬프트 경로 (<usr> 토큰 주입)
                with torch.no_grad():
                    e_user = proj(u).to(pipe.text_encoder.dtype)

                special_token = "<usr>"
                if special_token not in pipe.tokenizer.get_vocab():
                    pipe.tokenizer.add_tokens([special_token])
                    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
                token_id = pipe.tokenizer.convert_tokens_to_ids(special_token)
                with torch.no_grad():
                    emb = pipe.text_encoder.get_input_embeddings().weight
                    emb[token_id] = e_user

                result = pipe(
                    prompt=args.prompt,
                    negative_prompt=(args.neg_prompt if args.neg_prompt else None),
                    image=img,
                    strength=args.strength,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                )
                mode_tag = "lora" if args.use_lora else "tok"
                suffix = f"{mode_tag}_seed{seed_i}_str{args.strength:.2f}_{i:02d}.png"

            out_img = result.images[0]
            after_path = movie_dir / f"after_{suffix}"
            out_img.save(after_path)
            logger.info(f"[Saved] {after_path}")

    logger.info("All done.")

def movieid_to_poster_url(movie_id: int, links_map: dict, tmdb2path: dict) -> str:
    tid = links_map.get(movie_id)
    if tid is None:
        return ""
    poster_path = tmdb2path.get(tid, "")
    if not poster_path:
        return ""
    return f"{IMG_BASE}{IMG_SIZE}{poster_path}"

if __name__ == "__main__":
    main()
