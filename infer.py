"""Load a trained ESN checkpoint and generate text from a seed prompt."""
import argparse
import sys

import cupy as cp
import numpy as np

from common import latest_checkpoint, load_checkpoint


def softmax(x):
    e = cp.exp(x - cp.max(x))
    return e / e.sum()


def generate(model, seed, num_chars, temperature, rng, rand_init):
    char_to_index = {c: i for i, c in enumerate(model.vocab)}
    vocab_size = len(model.vocab)
    if rand_init:
        model.randomize(rng)
    else:
        model.reset()

    last_idx = None
    for c in seed:
        idx = char_to_index.get(c)
        if idx is None:
            continue
        model.step(idx)
        last_idx = idx

    output_chars = list(seed)
    for _ in range(num_chars):
        if last_idx is None:
            break
        ls = model.logits(last_idx)
        if temperature <= 0:
            next_index = int(ls.argmax())
        else:
            probs = cp.asnumpy(softmax(ls / temperature))
            next_index = int(rng.choice(vocab_size, p=probs))
        output_chars.append(model.vocab[next_index])
        model.step(next_index)
        last_idx = next_index

    return "".join(output_chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="?", default=None,
                        help="checkpoint path; defaults to latest checkpoint-*.pkl in cwd")
    parser.add_argument("--seed", type=str, default=" ")
    parser.add_argument("--num-chars", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = argmax (greedy); higher = more random")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--rand-init", action="store_true",
                        help="randomize reservoir state before priming (default: zero)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint or latest_checkpoint()
    if ckpt_path is None:
        parser.error("no checkpoint specified and none found in cwd")
    ckpt = load_checkpoint(ckpt_path)
    print(f"# loaded {ckpt_path}", file=sys.stderr)
    model = ckpt["model"]
    result = ckpt.get("result")
    if result is None:
        print(f"# checkpoint: model={type(model).__name__}", file=sys.stderr)
    else:
        print(
            f"# checkpoint: model={type(model).__name__} "
            f"train_acc={result['train_acc']:.3f} "
            f"test_acc={result['test_acc']:.3f} baseline={result['test_baseline']:.3f}",
            file=sys.stderr,
        )

    rng = np.random.default_rng(args.rng_seed)
    text = generate(model, args.seed, args.num_chars, args.temperature, rng, args.rand_init)
    print(text)


if __name__ == "__main__":
    main()
