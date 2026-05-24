"""Drive a trained ESN over the dataset and score next-token predictions."""
import argparse
import sys
from dataclasses import dataclass

import numpy as np

from common import (
    count_test_pairs,
    count_train_pairs,
    drive_articles,
    latest_checkpoint,
    load_checkpoint,
    load_dataset,
)


@dataclass(frozen=True)
class TrainResult:
    train_acc: float
    test_acc: float
    test_baseline: float


def evaluate(model, dataset, washout, rng):
    vocab_size = len(model.vocab)
    train_correct = [0, 0]
    test_correct = [0, 0]
    test_counts = np.zeros(vocab_size)

    def score_train(tok, target):
        pred = int(model.logits(tok).argmax())
        train_correct[0] += int(pred == target)
        train_correct[1] += 1

    def score_test(tok, target):
        pred = int(model.logits(tok).argmax())
        test_correct[0] += int(pred == target)
        test_correct[1] += 1
        test_counts[target] += 1

    drive_articles(
        model, dataset, washout,
        drive_full=True, on_train_pair=score_train, on_test_pair=score_test,
        desc="eval", rng=rng,
    )

    return TrainResult(
        train_acc=train_correct[0] / train_correct[1],
        test_acc=test_correct[0] / test_correct[1],
        test_baseline=float(test_counts.max() / test_counts.sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="?", default=None,
                        help="checkpoint path; defaults to latest checkpoint-*.pkl in cwd")
    parser.add_argument("--data", type=str, default="data.json")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="fraction of each article held out as test")
    parser.add_argument("--articles", type=int, default=10,
                        help="number of articles to load from --data")
    parser.add_argument("--washout", type=int, default=0)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = args.checkpoint or latest_checkpoint()
    if ckpt_path is None:
        parser.error("no checkpoint specified and none found in cwd")
    ckpt = load_checkpoint(ckpt_path)
    print(f"# loaded {ckpt_path}", file=sys.stderr)
    model = ckpt["model"]

    dataset = load_dataset(args.data, args.articles, args.holdout, vocab=model.vocab)
    n_train = count_train_pairs(dataset, args.washout)
    n_test = count_test_pairs(dataset)
    print(
        f"# articles: {len(dataset.articles_idx)}  "
        f"train_pairs: {n_train}  test_pairs: {n_test}",
        file=sys.stderr,
    )

    rng = np.random.default_rng(args.rng_seed)
    result = evaluate(model, dataset, args.washout, rng)
    print(
        f"train_acc={result.train_acc:.3f}  "
        f"test_acc={result.test_acc:.3f}  "
        f"test_baseline={result.test_baseline:.3f}"
    )


if __name__ == "__main__":
    main()
