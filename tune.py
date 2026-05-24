"""Random search over sparse.py hyperparameters.

Overrides sparse module-level constants per trial, then runs the existing
build/train/evaluate pipeline. Each trial appends a JSON line to the log so
runs are resumable by concatenating logs. Tuning the full 25k-neuron / 1k
article configuration is slow; use --num-neurons / --num-articles to run on
a cheaper proxy and rank configurations.
"""
import argparse
import json
import math
import os
import time
import traceback

import numpy as np
import tiktoken

import sparse


def loguniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def logint(rng, lo, hi):
    return int(round(loguniform(rng, lo, hi)))


SEARCH_SPACE = (
    ("SPECTRAL_RADIUS",   loguniform, 0.1,    1.5),
    ("INPUT_SCALE",       loguniform, 0.001,  1.0),
    ("RESERVOIR_DENSITY", loguniform, 0.0002, 0.01),
    ("INPUT_DENSITY",     loguniform, 0.0001, 0.5),
    ("READOUT_RANK",      logint,     64,     2048),
    ("RIDGE",             loguniform, 1e-6,   1.0),
)

TUNED_KEYS = tuple(name for name, *_ in SEARCH_SPACE)


def sample_hparams(rng):
    return {name: fn(rng, lo, hi) for name, fn, lo, hi in SEARCH_SPACE}


def apply_overrides(overrides):
    for k, v in overrides.items():
        setattr(sparse, k, v)


def run_trial(hparams, articles, vocab_size, trial_seed):
    apply_overrides(hparams)
    rng = np.random.default_rng(trial_seed)
    t0 = time.time()
    model = sparse.SparseESN.build(vocab_size, rng)
    sparse.train(model, articles, rng)
    train_acc, test_acc = sparse.evaluate(model, articles[:EVAL_ARTICLES])
    return train_acc, test_acc, time.time() - t0


EVAL_ARTICLES = 10  # set by main from --eval-articles


def main():
    global EVAL_ARTICLES
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the hparam sampler (each trial uses seed+i for model rng)")
    parser.add_argument("--num-articles", type=int, default=10,
                        help="override sparse.NUM_ARTICLES for cheaper tuning runs")
    parser.add_argument("--num-neurons", type=int, default=20000,
                        help="override sparse.NUM_NEURONS for cheaper tuning runs")
    parser.add_argument("--eval-articles", type=int, default=5,
                        help="how many articles to evaluate on per trial")
    parser.add_argument("--log", type=str, default=f"tune-{int(time.time())}.jsonl")
    args = parser.parse_args()

    EVAL_ARTICLES = args.eval_articles
    sparse.NUM_NEURONS = args.num_neurons
    sparse.NUM_ARTICLES = args.num_articles

    tokenizer = tiktoken.get_encoding(sparse.TOKENIZER_NAME)
    vocab_size = tokenizer.n_vocab
    articles = sparse.load_articles(sparse.DATA_PATH, sparse.NUM_ARTICLES, tokenizer)
    print(f"loaded {len(articles)} articles, "
          f"{sum(len(a) for a in articles)} tokens; "
          f"NUM_NEURONS={sparse.NUM_NEURONS}", flush=True)

    sampler = np.random.default_rng(args.seed)
    results = []
    with open(args.log, "a") as f:
        for i in range(args.trials):
            hparams = sample_hparams(sampler)
            print(f"\n[trial {i+1}/{args.trials}] {hparams}", flush=True)
            record = {
                "trial": i,
                "seed": args.seed + i,
                "hparams": hparams,
                "num_neurons": sparse.NUM_NEURONS,
                "num_articles": sparse.NUM_ARTICLES,
                "eval_articles": EVAL_ARTICLES,
            }
            try:
                train_acc, test_acc, dt = run_trial(
                    hparams, articles, vocab_size, args.seed + i)
                record.update(train_acc=train_acc, test_acc=test_acc, seconds=dt)
                print(f"  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
                      f"({dt:.1f}s)", flush=True)
                results.append(record)
                if test_acc >= max(r["test_acc"] for r in results):
                    print(f"  ** new best test_acc={test_acc:.4f}", flush=True)
            except Exception as e:
                record["error"] = f"{type(e).__name__}: {e}"
                record["traceback"] = traceback.format_exc()
                print(f"  ERROR: {record['error']}", flush=True)
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    if results:
        top = sorted(results, key=lambda r: r["test_acc"], reverse=True)[:10]
        with open(args.log, "a") as f:
            f.write(json.dumps({"top10": top}) + "\n")
        print("\n=== top 10 ===")
        header = f"{'#':>2}  {'test_acc':>8}  {'train_acc':>9}  " + "  ".join(
            f"{k:>17}" for k in TUNED_KEYS)
        print(header)
        for rank, r in enumerate(top, 1):
            vals = "  ".join(f"{r['hparams'][k]:>17.6g}" for k in TUNED_KEYS)
            print(f"{rank:>2}  {r['test_acc']:>8.4f}  {r['train_acc']:>9.4f}  {vals}")
    print(f"\nlog: {args.log}")


if __name__ == "__main__":
    main()
