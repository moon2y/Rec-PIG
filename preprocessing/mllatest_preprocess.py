# -*- coding: utf-8 -*-
import os, re
from pathlib import Path
import pandas as pd
from collections import defaultdict

# 프로젝트 루트(PIG)
BASE = Path(__file__).resolve().parents[1]

RAW_DIR = BASE / "data" / "raw" / "ml-latest"          # 원본 MovieLens
OUT_BASE = BASE / "data" / "preprocessed"              # atomic 파일은 여기 최상위에 생성
OUT_BASE.mkdir(parents=True, exist_ok=True)

INTER_PATH = OUT_BASE / "ml-latest.inter"              # <-- RecBole atomic 규칙
ITEM_PATH  = OUT_BASE / "ml-latest.item"               # <-- RecBole atomic 규칙

def norm_seq(s: str) -> str:
    s = (s or "").strip().replace("|", " ")
    s = re.sub(r"\s+", " ", s)           # 공백 정규화
    s = s.replace("\t", " ")             # 탭 제거(TSV 안전)
    return s.lower()

def main():
    movies  = pd.read_csv(RAW_DIR / "movies.csv")
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    tags_fp = RAW_DIR / "tags.csv"
    tags    = pd.read_csv(tags_fp) if tags_fp.exists() else None

    # movieId별 대표 태그 최대 10개
    tag_map = defaultdict(list)
    if tags is not None and len(tags) > 0:
        grp = tags.dropna(subset=["tag"]).groupby("movieId")["tag"].apply(list)
        for mid, lst in grp.items():
            cleaned = []
            for t in lst:
                t = re.sub(r"[^\w\s\-&'+/]", " ", str(t))
                t = re.sub(r"\s+", " ", t).strip().lower()
                t = t.replace("\t", " ")
                if t:
                    cleaned.append(t)
            tag_map[int(mid)] = cleaned[:10]

    # ----- item(.item) 생성: [item_id:token, title:token_seq, genres:token_seq, plot_text:token_seq]
    item_rows = []
    for _, r in movies.iterrows():
        mid = int(r["movieId"])
        title_raw  = str(r["title"])  if pd.notnull(r["title"])  else ""
        genres_raw = str(r["genres"]) if pd.notnull(r["genres"]) else ""

        title_no_year = re.sub(r"\(\d{4}\)\s*$", "", title_raw).strip()
        title_seq  = norm_seq(title_no_year)
        genres_seq = norm_seq(genres_raw) if genres_raw != "(no genres listed)" else ""
        tags_seq   = " ".join(tag_map.get(mid, []))

        parts = []
        if title_seq:  parts.append(title_seq + ".")
        if genres_seq: parts.append("genres: " + genres_seq + ".")
        if tags_seq:   parts.append("tags: " + tags_seq + ".")
        plot_text = " ".join(parts).strip()

        item_rows.append({
            "item_id:token": mid,
            "title:token_seq": title_seq,
            "genres:token_seq": genres_seq,
            "plot_text:token_seq": plot_text
        })

    item_df = pd.DataFrame(item_rows, columns=[
        "item_id:token", "title:token_seq", "genres:token_seq", "plot_text:token_seq"
    ])
    # TSV(탭 구분)으로 저장 (헤더 포함, 인덱스 제외)
    item_df.to_csv(ITEM_PATH, sep="\t", index=False)

    # ----- inter(.inter) 생성: [user_id:token, item_id:token, rating:float, timestamp:float]
    inter_df = ratings.rename(columns={
        "userId": "user_id:token",
        "movieId": "item_id:token",
        "rating": "rating:float",
        "timestamp": "timestamp:float"
    })[["user_id:token", "item_id:token", "rating:float", "timestamp:float"]]

    inter_df.to_csv(INTER_PATH, sep="\t", index=False)

    print(f"[Done] {ITEM_PATH}")
    print(f"[Done] {INTER_PATH}")

if __name__ == "__main__":
    main()
