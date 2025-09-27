# PIG/preprocessing/latents.py
# -*- coding: utf-8 -*-
"""
Streaming batch VAE encoding (no JPEG saved) - MOST-RECENT-WITH-POSTER per user ONLY
- 각 유저당 '포스터가 있는' 가장 최근(timestamp 최대)의 아이템 1개만 처리
- 네트워크에서 포스터를 메모리로 받아 배치 VAE 인코딩 → latent(.pt)만 저장
- resume-safe: 이미 있는 (uid, mid)는 스킵, 인덱스만 보충
- 4GPU DDP 유저 샤딩 (torchrun 사용)
- OOM 완화: TF32 허용, channels_last, allocator split-size 설정

실행 예:
torchrun --nproc_per_node=4 -m preprocessing.latents \
  --inter ./data/preprocessed/ml-latest.inter \
  --links ./data/raw/ml-latest/links.csv \
  --meta  ./data/raw/ml-latest/movies_metadata.csv \
  --out   ./embedding/vae \
  --prompt "a movie poster" \
  --batch 32 --prefetch 128 --concurrency 64
"""
# --- 메모리 단편화 완화(가능시) ---
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import argparse, csv, json, sys, time, asyncio, traceback
from io import BytesIO
from typing import List, Tuple, Set
from pathlib import Path

import numpy as np
import pandas as pd
import aiohttp
from PIL import Image

import torch
from diffusers.models import AutoencoderKL

# ---------------- DDP helpers ----------------
def get_rank_world():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world, local_rank

def shard_list(xs: List[int], rank: int, world: int) -> List[int]:
    n = len(xs); base = n // world; rem = n % world
    s = rank * base + min(rank, rem)
    e = s + base + (1 if rank < rem else 0)
    return xs[s:e]

# ---------------- paths & small utils ----------------
def resolve_root():
    return Path(__file__).resolve().parents[1]
ROOT = resolve_root()

IMG_BASE = "https://image.tmdb.org/t/p/"
IMG_SIZE = "w500"

def load_links_map(links_csv: Path) -> dict:
    if not links_csv.exists():
        raise FileNotFoundError(f"links.csv not found: {links_csv}")
    m = {}
    with links_csv.open(newline="", encoding="utf-8") as f:
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
            if tmdb is not None: m[mid] = tmdb
    return m

def load_poster_map(meta_csv: Path) -> dict:
    if not meta_csv.exists():
        raise FileNotFoundError(f"movies_metadata.csv not found: {meta_csv}")
    df = pd.read_csv(meta_csv, low_memory=False)
    if "id" not in df.columns or "poster_path" not in df.columns:
        raise ValueError("movies_metadata.csv must contain columns: id, poster_path")
    df = df[["id", "poster_path"]].dropna()
    def to_int(x):
        try: return int(float(x))
        except: return None
    df["id"] = df["id"].map(to_int)
    df = df.dropna(subset=["id"]).astype({"id": int})
    return dict(zip(df["id"].tolist(), df["poster_path"].astype(str).tolist()))

def make_url(movie_id: int, links_map: dict, tmdb2poster: dict) -> str:
    tid = links_map.get(movie_id)
    if tid is None: return ""
    p = tmdb2poster.get(tid, "")
    if not p: return ""
    return f"{IMG_BASE}{IMG_SIZE}{p}"

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # RGB -> [1,3,512,512] in [-1,1]
    img = img.resize((512, 512), Image.BICUBIC).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)  # H,W,3
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    t = t / 127.5 - 1.0
    return t  # float32 (caller will cast to vae.dtype)

# ---------------- resume helpers ----------------
def scan_done_pairs_fs(out_root: Path) -> Set[tuple]:
    """이미 저장된 latent 파일들 (uid,mid) 수집"""
    done = set()
    if not out_root.exists():
        return done
    for user_dir in out_root.glob("user*"):
        if not user_dir.is_dir(): continue
        try:
            uid = int(user_dir.name.replace("user", ""))
        except Exception:
            continue
        for pt in user_dir.glob("movie*.pt"):
            try:
                mid = int(pt.stem.replace("movie", ""))
                done.add((uid, mid))
            except Exception:
                continue
    return done

def read_index_pairs(p: Path) -> Set[tuple]:
    idx = set()
    if not p.exists(): return idx
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        try:
            ex = json.loads(line)
            idx.add((int(ex["user_id"]), int(ex["item_id"])))
        except Exception:
            continue
    return idx

def collect_indexed_pairs(out_root: Path, world: int) -> Set[tuple]:
    idx = set()
    idx |= read_index_pairs(out_root / "index.jsonl")
    for r in range(world):
        idx |= read_index_pairs(out_root / f"index.rank{r}.jsonl")
    return idx

# ---------------- async fetch ----------------
async def fetch_many(session: aiohttp.ClientSession,
                     tasks: List[Tuple[int,int,str]],
                     timeout: int = 30) -> List[Tuple[int,int,bytes]]:
    async def _one(uid, mid, url):
        if not url:
            return uid, mid, b""
        try:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    return uid, mid, b""
                data = await resp.read()
                return uid, mid, data
        except Exception:
            return uid, mid, b""
    return await asyncio.gather(*[_one(u,m,uurl) for (u,m,uurl) in tasks])

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", type=str, default=str(ROOT / "data" / "preprocessed" / "ml-latest.inter"))
    ap.add_argument("--links", type=str, default=str(ROOT / "data" / "raw" / "ml-latest" / "links.csv"))
    ap.add_argument("--meta",  type=str, default=str(ROOT / "data" / "raw" / "ml-latest" / "movies_metadata.csv"))
    ap.add_argument("--out",   type=str, default=str(ROOT / "embedding" / "vae"))
    ap.add_argument("--prompt", type=str, default="a movie poster")
    # 성능 파라미터
    ap.add_argument("--batch", type=int, default=32)       # 보수적 기본값
    ap.add_argument("--prefetch", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--local_only", action="store_true")
    args = ap.parse_args()

    rank, world, local_rank = get_rank_world()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    use_fp16 = torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    inter_path = Path(args.inter)
    links_csv  = Path(args.links)
    meta_csv   = Path(args.meta)
    out_root   = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"[DDP] world={world}", flush=True)
        print(f"[Paths] inter={inter_path.exists()} links={links_csv.exists()} meta={meta_csv.exists()}", flush=True)
        print(f"[Out] {out_root}", flush=True)

    # -------- 데이터 로드 & 유저 샤딩 --------
    inter = pd.read_csv(inter_path, sep="\t").sort_values(["user_id:token", "timestamp:float"])
    users_all = inter["user_id:token"].drop_duplicates().astype(int).tolist()
    users = shard_list(users_all, rank, world)
    if rank == 0:
        sizes = [len(shard_list(users_all, r, world)) for r in range(world)]
        print(f"[Shard] per-rank users: {sizes}", flush=True)

    links_map = load_links_map(links_csv)
    tmdb2poster = load_poster_map(meta_csv)

    # -------- VAE만 로드 --------
    load_kwargs = dict(torch_dtype=dtype, subfolder="vae")
    if args.local_only:
        load_kwargs["local_files_only"] = True
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", **load_kwargs).to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    # -------- RESUME: 완료/인덱스 쌍 수집 --------
    done_pairs_fs = scan_done_pairs_fs(out_root)
    indexed_pairs = collect_indexed_pairs(out_root, world)
    if rank == 0:
        print(f"[Resume] done_fs={len(done_pairs_fs)} indexed={len(indexed_pairs)}", flush=True)

    # -------- 각 유저의 '포스터가 있는' 가장 최근 1개만 선택 (벡터화) --------
    inter_shard = inter[inter["user_id:token"].isin(users)][["user_id:token","item_id:token","timestamp:float"]]

    links_df = pd.DataFrame({"item_id:token": list(links_map.keys()),
                             "tmdb_id": list(links_map.values())})
    poster_df = pd.DataFrame({"tmdb_id": list(tmdb2poster.keys()),
                              "poster_path": list(tmdb2poster.values())})

    df_all = (inter_shard
              .merge(links_df, on="item_id:token", how="left")
              .merge(poster_df, on="tmdb_id", how="left"))

    # 포스터가 있는 행만 유지
    df_all["has_poster"] = df_all["poster_path"].notna() & (df_all["poster_path"].astype(str) != "")
    df_all = df_all[df_all["has_poster"]].copy()

    # 각 유저별 가장 최근(timestamp 최대) 1개
    df_recent = (df_all.sort_values(["user_id:token","timestamp:float"])
                       .groupby("user_id:token", as_index=False)
                       .tail(1))

    # URL
    df_recent["url"] = (IMG_BASE + IMG_SIZE + df_recent["poster_path"].astype(str))

    # 키
    df_recent["key"] = df_recent.apply(lambda r: f"{int(r['user_id:token'])}:{int(r['item_id:token'])}", axis=1)
    done_keys    = set(f"{u}:{m}" for (u,m) in done_pairs_fs)
    indexed_keys = set(f"{u}:{m}" for (u,m) in indexed_pairs)

    # latent는 있으나 인덱스에 없는 쌍 → 인덱스만 보충
    need_index_only = df_recent[(df_recent["key"].isin(done_keys)) & (~df_recent["key"].isin(indexed_keys))]

    # latent가 아예 없고 URL이 있는 쌍만 실제 처리
    need_encode = df_recent[(~df_recent["key"].isin(done_keys)) & (df_recent["url"] != "")]
    triplets = list(zip(need_encode["user_id:token"].astype(int).tolist(),
                        need_encode["item_id:token"].astype(int).tolist(),
                        need_encode["url"].astype(str).tolist()))
    index_only_pairs = list(zip(need_index_only["user_id:token"].astype(int).tolist(),
                                need_index_only["item_id:token"].astype(int).tolist()))

    print(f"[Rank {rank}] todo (encode) users_with_poster={len(df_recent)}  triplets={len(triplets)}  index-only={len(index_only_pairs)}",
          flush=True)

    # -------- rank별 인덱스 파일 (append 모드) --------
    index_part = out_root / f"index.rank{rank}.jsonl"
    fw = index_part.open("a", encoding="utf-8")  # append

    # 인덱스만 보충
    for (uid, mid) in index_only_pairs:
        save_dir = out_root / f"user{uid}"
        save_path = save_dir / f"movie{mid}.pt"
        if save_path.exists():
            fw.write(json.dumps({
                "user_id": uid,
                "item_id": mid,
                "latent_path": str(save_path.relative_to(ROOT)),
                "prompt": args.prompt
            }, ensure_ascii=False) + "\n")

    # -------- 메인 루프: prefetch 단위로 비동기 다운로드 → 고정 배치 VAE 인코딩 --------
    written = 0
    skipped = 0
    total = len(triplets)

    async def runner():
        nonlocal written, skipped
        conn = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)
        async with aiohttp.ClientSession(connector=conn) as session:
            for s in range(0, total, args.prefetch):
                chunk = triplets[s: s + args.prefetch]
                # 1) 병렬 다운로드
                fetched = await fetch_many(session, chunk, timeout=args.timeout)
                # 2) bytes -> PIL -> tensor
                batch_imgs = []
                batch_meta = []
                for uid, mid, data in fetched:
                    if not data:
                        skipped += 1
                        continue
                    try:
                        img = Image.open(BytesIO(data)).convert("RGB")
                        t = pil_to_tensor(img)  # [1,3,512,512]
                        batch_imgs.append(t)
                        batch_meta.append((uid, mid))
                    except Exception:
                        skipped += 1
                        continue
                # 3) VAE 인코딩 (고정 배치; OOM 나면 --batch 낮춰 재실행 권장)
                i = 0
                while i < len(batch_imgs):
                    sub = batch_imgs[i: i + args.batch]
                    meta = batch_meta[i: i + args.batch]
                    i += args.batch
                    if not sub:
                        continue
                    x = torch.cat(sub, dim=0)  # [B,3,512,512]
                    x = x.to(device=device, dtype=vae.dtype, memory_format=torch.channels_last, non_blocking=True)
                    with torch.no_grad():
                        z = vae.encode(x).latent_dist.sample() * 0.18215  # [B,4,64,64]
                    # 저장 + 인덱스
                    for k in range(z.size(0)):
                        uid, mid = meta[k]
                        save_dir = out_root / f"user{uid}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = save_dir / f"movie{mid}.pt"
                        torch.save({"latent": z[k].cpu(), "user_id": uid, "item_id": mid}, save_path)
                        fw.write(json.dumps({
                            "user_id": uid,
                            "item_id": mid,
                            "latent_path": str(save_path.relative_to(ROOT)),
                            "prompt": args.prompt
                        }, ensure_ascii=False) + "\n")
                        written += 1

                if (s // max(1, args.prefetch)) % 10 == 0:
                    print(f"[Rank {rank}] progress: {min(s+args.prefetch,total)}/{total} "
                          f"written={written} skipped={skipped}", flush=True)

    try:
        if total > 0:
            asyncio.run(runner())
    except Exception as e:
        print(f"[Rank {rank} ERROR] {e}", flush=True)
        traceback.print_exc()
    finally:
        fw.close()

    print(f"[Rank {rank}] Done: newly_written={written} newly_indexed_only={len(index_only_pairs)} skipped_fetch={skipped}", flush=True)

    # -------- rank0: 부분 인덱스 병합 --------
    if rank == 0:
        time.sleep(2.0)
        parts = [out_root / f"index.rank{r}.jsonl" for r in range(world)]
        merged = out_root / "index.jsonl"
        with merged.open("w", encoding="utf-8") as fout:
            for p in parts:
                if p.exists():
                    fout.write(p.read_text())
        print(f"[Rank0] merged index -> {merged}", flush=True)

if __name__ == "__main__":
    main()
