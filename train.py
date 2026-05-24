"""Train an ESN readout on character sequences from data.json and save a checkpoint."""
import argparse
import glob
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass

import numpy as np

from common import count_test_pairs, count_train_pairs, drive_articles, load_dataset
from esn import Model, StackModel, geometric_leaks, random_leaks, uniform_leaks


@dataclass(frozen=True)
class TrainConfig:
    washout: int
    holdout_frac: float
    ridge: float
    chunk_size: int  # 0 means a single chunk holding all train pairs
    epochs: int


# ---- training ----

def train(model, dataset, config, rng):
    n_train = count_train_pairs(dataset, config.washout)
    chunk_size = config.chunk_size if config.chunk_size > 0 else max(1, n_train)

    solver = model.ridge_solver(chunk_size)
    for epoch in range(config.epochs):
        drive_articles(
            model, dataset, config.washout,
            drive_full=False, on_train_pair=solver.collect, on_test_pair=None,
            desc=f"train {epoch + 1}/{config.epochs}", rng=rng,
        )
    model.W_out = solver.finalize(config.ridge)


# ---- checkpointing ----

def next_checkpoint_index():
    pattern = re.compile(r"^checkpoint-(\d+)-\d+\.pkl$")
    indices = []
    for f in glob.glob("checkpoint-*.pkl"):
        m = pattern.match(os.path.basename(f))
        if m:
            indices.append(int(m.group(1)))
    return max(indices, default=0) + 1


def save_checkpoint(path, model, config, run_args):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "config": asdict(config),
                "args": run_args,
            },
            f,
        )


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--min-leak", type=float, default=0.001)
    parser.add_argument("--max-leak", type=float, default=0.9)
    leak_group = parser.add_mutually_exclusive_group()
    leak_group.add_argument("--rand-log-leaks", action="store_true",
                            help="sample leaks log-uniformly in [min-leak, max-leak] instead of geometric spacing")
    leak_group.add_argument("--rand-leaks", action="store_true",
                            help="sample leaks uniformly in [min-leak, max-leak] instead of geometric spacing")
    parser.add_argument("--num-reservoirs", type=int, default=5)
    parser.add_argument("--input-scale", type=float, default=0.06)
    parser.add_argument("--neurons", type=int, default=1000,
                        help="target neuron count per reservoir; rounded to nearest perfect square")
    parser.add_argument("--ridge", type=float, default=3.0)
    parser.add_argument("--chunk", type=int, default=1000,
                        help="chunk size for feature accumulation; 0 = one big chunk")
    parser.add_argument("--num-articles", type=int, default=10)
    parser.add_argument("--data", type=str, default="data.json")
    parser.add_argument("--washout", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of passes over the training data")
    parser.add_argument("--stack", action="store_true",
                        help="stack reservoirs in sequence (each layer reads the previous layer's state)")
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.rng_seed)

    grid_side = int(round(args.neurons ** 0.5))
    num_neurons = grid_side * grid_side

    dataset = load_dataset(args.data, args.num_articles, args.holdout_frac)
    vocab_size = len(dataset.vocab)

    if args.rand_log_leaks:
        leaks = random_leaks(args.num_reservoirs, args.min_leak, args.max_leak, rng)
    elif args.rand_leaks:
        leaks = uniform_leaks(args.num_reservoirs, args.min_leak, args.max_leak, rng)
    else:
        leaks = geometric_leaks(args.num_reservoirs, args.min_leak, args.max_leak)
    model_cls = StackModel if args.stack else Model
    model = model_cls.build(
        vocab=dataset.vocab,
        num_neurons=num_neurons,
        leaks=leaks,
        input_scale=args.input_scale,
        spectral_radius=args.spectral_radius,
        rng=rng,
    )

    config = TrainConfig(
        washout=args.washout,
        holdout_frac=args.holdout_frac,
        ridge=args.ridge,
        chunk_size=args.chunk,
        epochs=args.epochs,
    )

    n_train = count_train_pairs(dataset, args.washout)
    n_test = count_test_pairs(dataset)
    print("--- config ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  num_neurons (rounded): {num_neurons}")
    print(f"  leaks: {[f'{l:g}' for l in leaks]}")
    print(f"  vocab_size: {vocab_size}  feature_dim: {model.feature_dim}")
    print(f"  articles: {len(dataset.articles_idx)}  train_pairs: {n_train}  test_pairs: {n_test}")
    print("--------------")

    train(model, dataset, config, rng)

    ckpt_path = f"checkpoint-{next_checkpoint_index():03d}-{int(time.time())}.pkl"
    save_checkpoint(ckpt_path, model, config, vars(args))
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
