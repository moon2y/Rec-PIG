import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

PAD, MASK = 0, 1
OFFSET = 2  # item token offset (internal item index 0 -> token 2)

def load_movielens(item_path, inter_path, max_seq_len=50, min_len=5):
    """
    Read .item/.inter TSV and build user->sequence dict with *internal indices*.

    Key point:
      - Build pos_map from .item row order:
            ordered_ids = items["item_id:token"]
            pos_map[raw_item_id] = row_index (0..M-1)
      - Convert inter["item_id:token"] (raw ids) -> internal index using pos_map.
      - Token = OFFSET + internal_index
    """
    # 1) .item: make raw_id -> internal_index map from row order
    items = pd.read_csv(item_path, sep="\t")
    ordered_ids = items["item_id:token"].astype(int).tolist()
    pos_map = {rid: idx for idx, rid in enumerate(ordered_ids)}
    n_items = len(ordered_ids)  # M

    # 2) .inter: sort and convert raw ids to internal indices
    inter = pd.read_csv(inter_path, sep="\t")
    inter = inter.sort_values(["user_id:token", "timestamp:float"])

    user2seq = {}
    dropped_interactions = 0

    for uid, rows in inter.groupby("user_id:token"):
        # convert each raw item id to internal index if exists
        internal = []
        for raw in rows["item_id:token"].astype(int).tolist():
            idx = pos_map.get(raw, None)
            if idx is None:
                dropped_interactions += 1
                continue
            internal.append(idx)

        # too short after mapping -> skip
        if len(internal) < min_len:
            continue

        # keep only last max_seq_len and convert to tokens (OFFSET + index)
        seq = [idx + OFFSET for idx in internal[-max_seq_len:]]
        user2seq[int(uid)] = seq

    if dropped_interactions > 0:
        # 참고: 일부 inter의 item이 .item에 없을 경우가 있어 필터링됨
        print(f"[data] dropped interactions not found in .item: {dropped_interactions}")

    return user2seq, n_items

def mask_sequence(seq, mask_prob=0.15, n_items=None):
    """BERT4Rec masking rule: 80% [MASK], 10% random item, 10% keep.
       seq contains tokens in [OFFSET .. OFFSET + n_items - 1]
    """
    out_seq, labels = [], []
    for tok in seq:
        if random.random() < mask_prob:
            prob = random.random()
            if prob < 0.8:
                out_seq.append(MASK)
            elif prob < 0.9 and n_items is not None and n_items > 0:
                # sample random *internal* index then shift by OFFSET
                ridx = random.randint(0, n_items - 1)
                out_seq.append(OFFSET + ridx)
            else:
                out_seq.append(tok)
            labels.append(tok - OFFSET)  # target in [0..n_items-1]
        else:
            out_seq.append(tok)
            labels.append(-100)  # ignore
    return out_seq, labels

class SeqDataset(Dataset):
    """Returns (seq_tensor, labels_tensor, user_id)."""
    def __init__(self, user2seq, n_items, max_len=50, phase="train"):
        self.samples = []
        self.n_items = n_items
        self.max_len = max_len
        self.phase = phase
        for u, seq in user2seq.items():
            if len(seq) < 2:
                continue
            if phase == "train":
                self.samples.append((u, seq[:-2]))  # leave last 2 for val/test
            elif phase == "valid":
                self.samples.append((u, seq[:-1]))  # leave last for test
            else:
                self.samples.append((u, seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, seq = self.samples[idx]
        if self.phase == "train":
            seq, labels = mask_sequence(seq, n_items=self.n_items)
        elif self.phase == "valid":
            seq, labels = seq[:], [-100]*len(seq)
            if len(seq) > 1:
                # mask the second-to-last token (convert label back to internal index)
                labels[-2] = seq[-2] - OFFSET
                seq[-2] = MASK
        else:  # test
            seq, labels = seq[:], [-100]*len(seq)
            labels[-1] = seq[-1] - OFFSET
            seq[-1] = MASK

        # pad/trim to max_len
        if len(seq) < self.max_len:
            pad_len = self.max_len - len(seq)
            seq = [PAD]*pad_len + seq
            labels = [-100]*pad_len + labels
        else:
            seq = seq[-self.max_len:]
            labels = labels[-self.max_len:]

        return torch.tensor(seq), torch.tensor(labels), torch.tensor(u, dtype=torch.long)

def get_dataloaders(item_path, inter_path, max_seq_len=50, batch_size=128):
    user2seq, n_items = load_movielens(item_path, inter_path, max_seq_len)
    train_set = SeqDataset(user2seq, n_items, max_seq_len, "train")
    valid_set = SeqDataset(user2seq, n_items, max_seq_len, "valid")
    test_set  = SeqDataset(user2seq, n_items, max_seq_len, "test")
    # keep pin_memory=True for faster host->device transfer
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True),
        DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True),
        n_items
    )
