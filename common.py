"""Shared dataset loading and checkpoint discovery."""
import glob
import json
import os
import pickle
import re
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class Dataset:
    vocab: list             # list[str]
    articles_idx: list      # list[np.ndarray] of int indices
    split_indices: list     # list[int], train=[0,split), test=[split,L)


def load_dataset(path, num_articles, holdout_frac, vocab=None):
    with open(path) as f:
        texts = [d["text"] for d in json.load(f)[:num_articles]]
    if vocab is None:
        vocab = sorted(set("".join(texts)))
    char_to_index = {c: i for i, c in enumerate(vocab)}
    articles_idx = [
        np.array([char_to_index[c] for c in t if c in char_to_index], dtype=np.int64)
        for t in texts
    ]
    split_indices = [len(a) - int(len(a) * holdout_frac) for a in articles_idx]
    return Dataset(vocab=vocab, articles_idx=articles_idx, split_indices=split_indices)


def count_train_pairs(dataset, washout):
    return sum(max(0, s - 1 - washout) for s in dataset.split_indices)


def count_test_pairs(dataset):
    return sum(
        max(0, len(a) - 1 - s)
        for a, s in zip(dataset.articles_idx, dataset.split_indices)
    )


def drive_articles(model, dataset, washout, drive_full, on_train_pair, on_test_pair, desc, rng):
    """Drive each article (model state reset per article).
    drive_full=False stops at split (training accumulation).
    drive_full=True drives [0, L) and emits both train and test pairs (eval).
    """
    total = (
        sum(len(a) for a in dataset.articles_idx)
        if drive_full
        else sum(dataset.split_indices)
    )
    pbar = tqdm(total=total, desc=desc, mininterval=1.0, smoothing=0.0, ascii=True)
    for article, split in zip(dataset.articles_idx, dataset.split_indices):
        L = len(article)
        end = L if drive_full else split
        # model.randomize(rng)
        model.reset()
        for t in range(end):
            tok = int(article[t])
            model.step(tok)
            if t < L - 1:
                target = int(article[t + 1])
                if washout <= t < split - 1 and on_train_pair is not None:
                    on_train_pair(tok, target)
                elif drive_full and t >= split and on_test_pair is not None:
                    on_test_pair(tok, target)
            pbar.update(1)
    pbar.close()


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def latest_checkpoint():
    pattern = re.compile(r"^checkpoint-(\d+)-\d+\.pkl$")
    best = None
    for f in glob.glob("checkpoint-*.pkl"):
        m = pattern.match(os.path.basename(f))
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), f)
    return best[1] if best else None
