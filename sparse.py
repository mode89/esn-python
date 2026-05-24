"""Sparse Echo State Network with GPT-2 BPE tokenization (numpy + scipy.sparse).

Single reservoir with sparse recurrent W and sparse input W_in (column lookup
per token id). Readout features are [1, state]; the one-hot input is omitted
because the GPT-2 vocab (~50k) would otherwise dominate the feature dim, and
state already reflects the just-stepped token.

Readout is factored W_out = U_out @ V_out.T with U_out fixed and random with
orthonormal columns (V, r) and V_out learned (fdim, r). Targets are projected
to r-dim by row-lookup into U_out, so we never materialize an (n, V) one-hot
block or an (fdim, V) gram_targets. Inference: logits = U_out @ (V_out.T @ f).
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tiktoken
from tqdm import tqdm


# ---- Hyperparameters ----
NUM_NEURONS = 20000
LEAK_MIN = 0.0
LEAK_MAX = 1.0
SPECTRAL_RADIUS = 0.1
INPUT_SCALE = 0.01
RESERVOIR_DENSITY = 0.0003       # fraction of nonzero entries in W
INPUT_DENSITY = 0.03           # fraction of neurons each token writes into
RIDGE = 0.02
READOUT_RANK = 256              # rank r of the factored readout W_out = U_out @ V_out.T
WASHOUT = 0
HOLDOUT = 0.1                    # fraction of each article held out for test eval
CHUNK_SIZE = 2048
NUM_ARTICLES = 7000
EVAL_ARTICLES = 10
DATA_PATH = "data.json"
TOKENIZER_NAME = "gpt2"
RNG_SEED = 42
DTYPE = np.float32


# ---- Sparse matrix construction ----

def build_W(num_neurons, density, spectral_radius, rng):
    nnz = max(1, int(round(num_neurons * num_neurons * density)))
    rows = rng.integers(0, num_neurons, size=nnz)
    cols = rng.integers(0, num_neurons, size=nnz)
    vals = rng.uniform(-1.0, 1.0, size=nnz).astype(np.float64)
    W = sp.coo_matrix((vals, (rows, cols)), shape=(num_neurons, num_neurons)).tocsr()
    # Largest-magnitude eigenvalue via ARPACK; needs float64 for stability.
    eig = spla.eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=1000)
    cur_radius = float(abs(eig[0]))
    W.data *= spectral_radius / cur_radius
    return W.astype(DTYPE)


def build_W_in(num_neurons, vocab_size, density, input_scale, rng):
    k = max(1, int(round(num_neurons * density)))
    cols = np.repeat(np.arange(vocab_size, dtype=np.int64), k)
    rows = rng.integers(0, num_neurons, size=vocab_size * k)
    vals = rng.uniform(-input_scale, input_scale, size=vocab_size * k).astype(DTYPE)
    return sp.csc_matrix((vals, (rows, cols)), shape=(num_neurons, vocab_size))


# ---- Model ----

class SparseESN:
    def __init__(self, vocab_size, num_neurons, leak, W_in, W):
        self.vocab_size = vocab_size
        self.num_neurons = num_neurons
        self.leak = leak.astype(DTYPE)  # (N,) per-neuron leak in [LEAK_MIN, LEAK_MAX]
        self.W_in = W_in   # csc, (N, V)
        self.W = W         # csr, (N, N)
        self.x = np.zeros(num_neurons, dtype=DTYPE)
        self.U_out = None  # (V, r) fixed random output code, orthonormal columns
        self.V_out = None  # (fdim, r) learned, set after training

    @classmethod
    def build(cls, vocab_size, rng):
        W = build_W(NUM_NEURONS, RESERVOIR_DENSITY, SPECTRAL_RADIUS, rng)
        W_in = build_W_in(NUM_NEURONS, vocab_size, INPUT_DENSITY, INPUT_SCALE, rng)
        leak = rng.uniform(LEAK_MIN, LEAK_MAX, size=NUM_NEURONS)
        model = cls(vocab_size, NUM_NEURONS, leak, W_in, W)
        # Haar-random orthonormal columns: QR of an (V, r) Gaussian.
        G = rng.standard_normal((vocab_size, READOUT_RANK)).astype(np.float64)
        Q, _ = np.linalg.qr(G)
        model.U_out = Q.astype(DTYPE)
        return model

    def reset(self):
        self.x[:] = 0.0

    def step(self, tok):
        # Sparse column slice -> dense N-vector (W_in @ onehot(tok)).
        col = self.W_in.data[self.W_in.indptr[tok]:self.W_in.indptr[tok + 1]]
        idx = self.W_in.indices[self.W_in.indptr[tok]:self.W_in.indptr[tok + 1]]
        drive = np.zeros(self.num_neurons, dtype=DTYPE)
        np.add.at(drive, idx, col)
        self.x = (DTYPE(1.0) - self.leak) * self.x + self.leak * np.tanh(drive + self.W @ self.x)

    @property
    def feature_dim(self):
        return 1 + self.num_neurons

    def feature(self):
        f = np.empty(self.feature_dim, dtype=DTYPE)
        f[0] = 1.0
        f[1:] = self.x
        return f

    def logits(self):
        return self.U_out @ (self.V_out.T @ self.feature())


# ---- Data ----

def load_articles(path, num_articles, tokenizer):
    with open(path) as f:
        texts = [d["text"] for d in json.load(f)[1:num_articles]]
    return [np.array(tokenizer.encode(t), dtype=np.int64) for t in texts]


# ---- Training ----

def train(model, articles, rng):
    fdim = model.feature_dim
    r = READOUT_RANK
    U_out_f64 = model.U_out.astype(np.float64)  # (V, r) for target row lookup
    # Accumulate in float64 for numerical stability of the ridge solve.
    gram = np.zeros((fdim, fdim), dtype=np.float64)
    gram_targets = np.zeros((fdim, r), dtype=np.float64)

    F_buf = np.zeros((CHUNK_SIZE, fdim), dtype=DTYPE)
    target_buf = np.zeros(CHUNK_SIZE, dtype=np.int64)
    count = 0

    def flush(n):
        if n == 0:
            return
        F = F_buf[:n].astype(np.float64)
        gram[:] += F.T @ F
        # Y_proj[k] = U_out[target_k] is the rank-r projection of the one-hot target.
        Y_proj = U_out_f64[target_buf[:n]]            # (n, r)
        gram_targets[:] += F.T @ Y_proj

    splits = [len(a) - int(len(a) * HOLDOUT) for a in articles]
    total_train = sum(max(0, s - 1 - WASHOUT) for s in splits)
    pbar = tqdm(total=total_train, desc="train", mininterval=1.0, ascii=True)

    for article, split in zip(articles, splits):
        model.reset()
        for t in range(split):
            tok = int(article[t])
            model.step(tok)
            if WASHOUT <= t < split - 1:
                F_buf[count, 0] = 1.0
                F_buf[count, 1:] = model.x
                target_buf[count] = int(article[t + 1])
                count += 1
                if count == CHUNK_SIZE:
                    flush(count)
                    pbar.update(count)
                    count = 0
    flush(count)
    pbar.update(count)
    pbar.close()

    A = gram + RIDGE * np.eye(fdim, dtype=np.float64)
    model.V_out = np.linalg.solve(A, gram_targets).astype(DTYPE)  # (fdim, r)


# ---- Evaluation (next-token accuracy, split into train/test by HOLDOUT) ----

def evaluate(model, articles):
    splits = [len(a) - int(len(a) * HOLDOUT) for a in articles]
    train_correct = train_total = 0
    test_correct = test_total = 0
    pbar = tqdm(total=sum(len(a) for a in articles), desc="eval", mininterval=1.0, ascii=True)
    for article, split in zip(articles, splits):
        model.reset()
        L = len(article)
        for t in range(L):
            tok = int(article[t])
            model.step(tok)
            if WASHOUT <= t < L - 1:
                pred = int(model.logits().argmax())
                target = int(article[t + 1])
                hit = int(pred == target)
                if t < split - 1:
                    train_correct += hit
                    train_total += 1
                else:
                    test_correct += hit
                    test_total += 1
            pbar.update(1)
    pbar.close()
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    return train_acc, test_acc


# ---- Inference ----

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def generate(model, prompt_tokens, num_tokens, temperature, rng):
    model.reset()
    for tok in prompt_tokens:
        model.step(int(tok))
    output = list(prompt_tokens)
    for _ in range(num_tokens):
        ls = model.logits()
        if temperature <= 0:
            next_tok = int(np.argmax(ls))
        else:
            p = softmax(ls.astype(np.float64) / temperature)
            next_tok = int(rng.choice(p.shape[0], p=p))
        output.append(next_tok)
        model.step(next_tok)
    return output


def latest_checkpoint():
    pat = re.compile(r"^sparse-(\d+)\.pkl$")
    best = None
    for f in glob.glob("sparse-*.pkl"):
        m = pat.match(os.path.basename(f))
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), f)
    return best[1] if best else None


def run_infer(prompt, num_tokens, temperature, rng):
    path = latest_checkpoint()
    if path is None:
        print("no checkpoint found (sparse-*.pkl)", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"# loaded {path}  train_acc={ckpt.get('train_acc', float('nan')):.4f} "
          f"test_acc={ckpt.get('test_acc', float('nan')):.4f}", file=sys.stderr)
    model = ckpt["model"]
    tokenizer = tiktoken.get_encoding(ckpt["tokenizer_name"])
    prompt_tokens = tokenizer.encode(prompt) if prompt else []
    out_tokens = generate(model, prompt_tokens, num_tokens, temperature, rng)
    print(tokenizer.decode(out_tokens))


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, nargs="?", const="", default=None,
                        metavar="TEXT",
                        help="load latest sparse-*.pkl and generate from TEXT (default: empty prompt)")
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = argmax (greedy); >0 enables sampling")
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    if args.infer is not None:
        run_infer(args.infer, args.num_tokens, args.temperature, rng)
        return

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    vocab_size = tokenizer.n_vocab

    print("--- config ---")
    for k, v in [
        ("NUM_NEURONS", NUM_NEURONS),
        ("LEAK_MIN", LEAK_MIN), ("LEAK_MAX", LEAK_MAX),
        ("SPECTRAL_RADIUS", SPECTRAL_RADIUS), ("INPUT_SCALE", INPUT_SCALE),
        ("RESERVOIR_DENSITY", RESERVOIR_DENSITY), ("INPUT_DENSITY", INPUT_DENSITY),
        ("RIDGE", RIDGE), ("READOUT_RANK", READOUT_RANK),
        ("WASHOUT", WASHOUT), ("HOLDOUT", HOLDOUT),
        ("CHUNK_SIZE", CHUNK_SIZE), ("NUM_ARTICLES", NUM_ARTICLES),
        ("TOKENIZER_NAME", TOKENIZER_NAME), ("vocab_size", vocab_size),
    ]:
        print(f"  {k}: {v}")
    print("--------------")

    articles = load_articles(DATA_PATH, NUM_ARTICLES, tokenizer)
    total_tokens = sum(len(a) for a in articles)
    print(f"loaded {len(articles)} articles, {total_tokens} tokens "
          f"(min={min(len(a) for a in articles)}, max={max(len(a) for a in articles)})")

    model = SparseESN.build(vocab_size, rng)
    print(f"W nnz={model.W.nnz}  W_in nnz={model.W_in.nnz}  feature_dim={model.feature_dim}")

    train(model, articles, rng)
    train_acc, test_acc = evaluate(model, articles[:EVAL_ARTICLES])
    print(f"train accuracy: {train_acc:.4f}  test accuracy: {test_acc:.4f}")

    ckpt = {
        "model": model,
        "tokenizer_name": TOKENIZER_NAME,
        "hparams": {
            "NUM_NEURONS": NUM_NEURONS,
            "LEAK_MIN": LEAK_MIN, "LEAK_MAX": LEAK_MAX,
            "SPECTRAL_RADIUS": SPECTRAL_RADIUS, "INPUT_SCALE": INPUT_SCALE,
            "RESERVOIR_DENSITY": RESERVOIR_DENSITY, "INPUT_DENSITY": INPUT_DENSITY,
            "RIDGE": RIDGE, "READOUT_RANK": READOUT_RANK,
            "WASHOUT": WASHOUT, "HOLDOUT": HOLDOUT,
            "NUM_ARTICLES": NUM_ARTICLES,
        },
        "train_acc": train_acc,
        "test_acc": test_acc,
    }
    path = f"sparse-{int(time.time())}.pkl"
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"saved checkpoint to {path}")


if __name__ == "__main__":
    main()
