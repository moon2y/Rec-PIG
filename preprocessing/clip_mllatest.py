# -*- coding: utf-8 -*-
# RecBole atomic (ml-latest.item / ml-latest.inter) 기반으로 CLIP 임베딩 생성
# 기존 item.csv가 있으면 자동 폴백 (백워드 호환)
from pathlib import Path
import os, csv
import pandas as pd, numpy as np, torch, open_clip, requests, json, time
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# 프로젝트 루트
BASE = Path(__file__).resolve().parents[1]

# 원본 ML + Kaggle 메타(포스터 경로)
RAW_ML = BASE / "data" / "raw" / "ml-latest"   # links.csv
RAW_KG = BASE / "data" / "raw" / "ml-latest"   # movies_metadata.csv (Kaggle 내려받아 여기에 둠)

# RecBole atomic 출력물 경로 (recbole_mllatest.py가 생성)
PREP_DIR = BASE / "data" / "preprocessed"
ITEM_ATOMIC = PREP_DIR / "ml-latest.item"      # TSV
INTER_ATOMIC = PREP_DIR / "ml-latest.inter"    # TSV

# 구(舊) 파이프라인 호환용
ITEM_CSV_FALLBACK = PREP_DIR / "ml-latest" / "item.csv"

FEAT_DIR = BASE / "embedding" / "clip"; FEAT_DIR.mkdir(parents=True, exist_ok=True)

BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_BACKBONE, CLIP_CKPT = "ViT-H-14", "laion2b_s32b_b79k"

# TMDb 포스터 URL (API 없이)
IMG_BASE = "https://image.tmdb.org/t/p/"
IMG_SIZE = "w500"
SLEEP_IMG = 0.02

def load_links():
    """MovieLens links.csv → {movieId: tmdbId} 매핑"""
    m = {}
    with open(RAW_ML / "links.csv", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                mid = int(row["movieId"])
            except:
                continue
            tid = row.get("tmdbId")
            # tmdbId가 비어있거나 실수/문자일 수 있어 보정
            try:
                tmdb = int(float(tid)) if tid and str(tid).strip() != "" else None
            except:
                tmdb = None
            if tmdb is not None:
                m[mid] = tmdb
    return m

def load_kaggle_poster_map():
    """Kaggle movies_metadata.csv → {tmdbId: poster_path}"""
    df = pd.read_csv(RAW_KG / "movies_metadata.csv", low_memory=False)
    df = df[["id", "poster_path"]].dropna()
    def to_int(x):
        try:
            return int(float(x))
        except:
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
    except:
        return None

@torch.no_grad()
def encode_text(model, tokenizer, texts):
    outs = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Encode text"):
        toks = tokenizer(texts[i:i+BATCH]).to(DEVICE)
        x = model.encode_text(toks)
        x = x / x.norm(dim=-1, keepdim=True)
        outs.append(x.cpu())
    return torch.cat(outs, dim=0).numpy()

@torch.no_grad()
def encode_images(model, preprocess, urls):
    N = len(urls)
    out = None
    for i in tqdm(range(0, N, BATCH), desc="Encode images"):
        batch, idxs = [], []
        for j, url in enumerate(urls[i:i+BATCH]):
            img = fetch_image(url) if url else None
            if img is None:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            batch.append(preprocess(img)); idxs.append(i + j)
            time.sleep(SLEEP_IMG)
        if not batch:
            continue
        bt = torch.stack(batch).to(DEVICE)
        f = model.encode_image(bt); f = f / f.norm(dim=-1, keepdim=True)
        f = f.cpu().numpy()
        if out is None:
            out = np.zeros((N, f.shape[1]), dtype=np.float32)
        out[idxs, :] = f
    if out is None:
        # 텍스트 projection 차원을 따르도록 안전장치 (구조 유지)
        out = np.zeros((N, model.text_projection.shape[1]), dtype=np.float32)
    return out

def load_items_and_filter_with_inter():
    """
    1) RecBole atomic(.item/.inter) 우선 사용
       - .item: TSV, 컬럼: item_id:token, title:token_seq, genres:token_seq, plot_text:token_seq
       - .inter: TSV, 컬럼: user_id:token, item_id:token, rating:float, timestamp:float
       → inter 존재 시 등장한 item_id만 서브셋
    2) 없으면 과거 CSV(item.csv)로 폴백
    """
    if ITEM_ATOMIC.exists():
        # Atomic 형식 읽기
        item_df = pd.read_csv(ITEM_ATOMIC, sep="\t")
        if INTER_ATOMIC.exists():
            inter_df = pd.read_csv(INTER_ATOMIC, sep="\t", usecols=["item_id:token"])
            used_items = set(inter_df["item_id:token"].astype(int).tolist())
            item_df = item_df[item_df["item_id:token"].astype(int).isin(used_items)].reset_index(drop=True)
        # 필요한 컬럼 보정
        item_df["plot_text:token_seq"] = item_df.get("plot_text:token_seq", "").astype(str)
        item_df["item_id:token"] = item_df["item_id:token"].astype(int)
        source_kind = "recbole_atomic"
        return item_df, source_kind
    elif ITEM_CSV_FALLBACK.exists():
        # 구(舊) CSV 경로 호환
        item_df = pd.read_csv(ITEM_CSV_FALLBACK)
        # 기존 스키마: plot_text:token_seq, item_id:token
        item_df["plot_text:token_seq"] = item_df.get("plot_text:token_seq", "").astype(str)
        item_df["item_id:token"] = item_df["item_id:token"].astype(int)
        source_kind = "legacy_item_csv"
        return item_df, source_kind
    else:
        raise FileNotFoundError(
            f"Neither '{ITEM_ATOMIC}' nor '{ITEM_CSV_FALLBACK}' exists. "
            f"먼저 recbole_mllatest.py를 실행해 .item/.inter를 생성하거나, 기존 item.csv를 준비하세요."
        )

def main():
    # 1) 아이템/텍스트 로드 (+ inter 기반 필터링)
    item_df, source_kind = load_items_and_filter_with_inter()
    texts     = item_df["plot_text:token_seq"].fillna("").astype(str).tolist()
    movie_ids = item_df["item_id:token"].astype(int).tolist()

    # 2) MovieLens links & Kaggle movies_metadata로 포스터 URL 구성
    ml_tmdb = load_links()               # movieId -> tmdbId
    tmdb2path = load_kaggle_poster_map() # tmdbId  -> poster_path

    urls = []
    for mid in movie_ids:
        tid = ml_tmdb.get(mid)
        p = tmdb2path.get(tid, "") if tid is not None else ""
        urls.append(f"{IMG_BASE}{IMG_SIZE}{p}" if p else "")

    # 3) OpenCLIP 로딩
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_BACKBONE, pretrained=CLIP_CKPT, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)
    model.eval()

    # 4) 임베딩 생성
    text_embs = encode_text(model, tokenizer, texts)
    img_embs  = encode_images(model, preprocess, urls)

    # 5) 저장
    np.save(FEAT_DIR / "mllatest_plots.npy",   text_embs)
    np.save(FEAT_DIR / "mllatest_posters.npy", img_embs)
    (FEAT_DIR / "mllatest_clip_meta.json").write_text(json.dumps({
        "clip_backbone": CLIP_BACKBONE,
        "checkpoint": CLIP_CKPT,
        "dim": int(text_embs.shape[1]),
        "source": f"{source_kind}+kaggle_poster_path",
        "size": IMG_SIZE
    }, indent=2))
    print(f"[Done] Saved features to {FEAT_DIR} (source={source_kind}, N={len(movie_ids)})")

if __name__ == "__main__":
    main()
