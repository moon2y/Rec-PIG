from pathlib import Path

# Project-relative fixed paths
ROOT = Path(__file__).resolve().parents[1]  # PIG/
DIFFUSION_DIR = ROOT / "diffusion"
CHECKPOINT_DIR = DIFFUSION_DIR / "checkpoints"
RESULTS_DIR = DIFFUSION_DIR / "results"
LOG_DIR = ROOT / "log" / "diffusion"

# mmbert4rec user embeddings (already produced)
USER_EMB_DIR = ROOT / "embedding" / "user"
USER_EMB_NPY = USER_EMB_DIR / "user_pref_embs_all.npy"
USER_EMB_JSON = USER_EMB_DIR / "user_pref_embs_all_users.json"

# default model id (Stable Diffusion v1.5)
SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# checkpoints for our small heads
PROJ_U2TXT_CKPT = CHECKPOINT_DIR / "proj_u2txt.pt"
DPT_HYPERNET_CKPT = CHECKPOINT_DIR / "dpt_hypernet.pt"

# make sure dirs exist when imported
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
