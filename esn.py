"""Echo State Network: stateful model with token-indexed step and ridge readout."""
from dataclasses import dataclass
from types import SimpleNamespace

import cupy as cp
import numpy as np


def _build_W(num_neurons, spectral_radius, rng):
    # Circular law: for entries i.i.d. uniform on [-1, 1] (variance 1/3),
    # ρ(W) → √(N/3) as N → ∞ (empirically within ~2% by N=2000).
    W_np = rng.uniform(-1.0, 1.0, size=(num_neurons, num_neurons))
    W_np *= spectral_radius / np.sqrt(num_neurons / 3.0)
    return cp.asarray(W_np)
    # W_np = rng.uniform(-1.0, 1.0, size=(num_neurons, num_neurons))
    # W_np *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W_np)))
    # return cp.asarray(W_np)


@dataclass
class Reservoir:
    leak: float
    W_in: cp.ndarray   # (num_neurons, vocab_size)
    W: cp.ndarray      # (num_neurons, num_neurons)
    x: cp.ndarray      # (num_neurons,) current activations


@dataclass
class Model:
    vocab: list                          # list[str], one char per index
    reservoirs: list                     # list[Reservoir]
    W_out: cp.ndarray | None = None      # (vocab_size, feature_dim); set after training

    @classmethod
    def build(cls, vocab, num_neurons, leaks, input_scale, spectral_radius, rng):
        vocab_size = len(vocab)
        W = _build_W(num_neurons, spectral_radius, rng)
        reservoirs = [
            Reservoir(
                leak=leak,
                W_in=cp.asarray(rng.uniform(-input_scale, input_scale, size=(num_neurons, vocab_size))),
                W=W,
                x=cp.zeros(num_neurons),
            )
            for leak in leaks
        ]
        return cls(vocab=vocab, reservoirs=reservoirs)

    def reset(self):
        for r in self.reservoirs:
            r.x[:] = 0.0

    def randomize(self, rng):
        for r in self.reservoirs:
            r.x[:] = cp.asarray(rng.uniform(-1.0, 1.0, size=r.x.shape))

    def step(self, tok):
        # W_in @ onehot(tok) is just W_in[:, tok]; both terms on the RHS read r.x before assignment.
        for r in self.reservoirs:
            r.x[:] = (1 - r.leak) * r.x + r.leak * cp.tanh(r.W_in[:, tok] + r.W @ r.x)

    @property
    def feature_dim(self):
        vocab_size = len(self.vocab)
        return 1 + vocab_size + sum(r.x.shape[0] for r in self.reservoirs)

    def logits(self, tok):
        vocab_size = len(self.vocab)
        onehot = cp.zeros(vocab_size)
        onehot[tok] = 1.0
        feature = cp.concatenate([cp.ones(1), onehot] + [r.x for r in self.reservoirs])
        return self.W_out @ feature

    def ridge_solver(self, chunk_size):
        vocab_size = len(self.vocab)
        num_reservoirs = len(self.reservoirs)
        num_neurons = self.reservoirs[0].x.shape[0]
        fdim = self.feature_dim

        state_bufs = [cp.zeros((chunk_size, num_neurons)) for _ in range(num_reservoirs)]
        inputs_buf = cp.zeros((chunk_size, vocab_size))
        targets_buf = cp.zeros((chunk_size, vocab_size))
        ones_buf = cp.ones((chunk_size, 1))

        gram = cp.zeros((fdim, fdim))
        gram_targets = cp.zeros((fdim, vocab_size))
        count = [0]

        def accumulate(n):
            if n == 0:
                return
            F = cp.concatenate(
                [ones_buf[:n], inputs_buf[:n]] + [sb[:n] for sb in state_bufs], axis=1
            )
            gram[:] += F.T @ F
            gram_targets[:] += F.T @ targets_buf[:n]

        def collect(tok, target):
            i = count[0]
            inputs_buf[i] = 0.0
            inputs_buf[i, tok] = 1.0
            for k, r in enumerate(self.reservoirs):
                state_bufs[k][i] = r.x
            targets_buf[i] = 0.0
            targets_buf[i, target] = 1.0
            count[0] = i + 1
            if count[0] == chunk_size:
                accumulate(chunk_size)
                count[0] = 0

        def finalize(ridge):
            accumulate(count[0])
            count[0] = 0
            return cp.linalg.solve(gram + ridge * cp.eye(fdim), gram_targets).T

        return SimpleNamespace(collect=collect, finalize=finalize)


class StackModel(Model):
    """Stacked ESN: layer 0 reads the input token; each later layer reads the
    just-updated state of the previous layer."""

    @classmethod
    def build(cls, vocab, num_neurons, leaks, input_scale, spectral_radius, rng):
        vocab_size = len(vocab)
        W = _build_W(num_neurons, spectral_radius, rng)
        reservoirs = []
        for k, leak in enumerate(leaks):
            in_dim = vocab_size if k == 0 else num_neurons
            reservoirs.append(Reservoir(
                leak=leak,
                W_in=cp.asarray(rng.uniform(-input_scale, input_scale, size=(num_neurons, in_dim))),
                W=W,
                x=cp.zeros(num_neurons),
            ))
        return cls(vocab=vocab, reservoirs=reservoirs)

    def step(self, tok):
        prev = None
        for k, r in enumerate(self.reservoirs):
            drive = r.W_in[:, tok] if k == 0 else r.W_in @ prev
            r.x[:] = (1 - r.leak) * r.x + r.leak * cp.tanh(drive + r.W @ r.x)
            prev = r.x


def geometric_leaks(num_reservoirs, min_leak, max_leak):
    if num_reservoirs == 1:
        return [min_leak]
    ratio = (max_leak / min_leak) ** (1.0 / (num_reservoirs - 1))
    return [min_leak * (ratio ** k) for k in range(num_reservoirs)]


def random_leaks(num_reservoirs, min_leak, max_leak, rng):
    log_lo, log_hi = np.log(min_leak), np.log(max_leak)
    return sorted(np.exp(rng.uniform(log_lo, log_hi, size=num_reservoirs)).tolist())


def uniform_leaks(num_reservoirs, min_leak, max_leak, rng):
    return sorted(rng.uniform(min_leak, max_leak, size=num_reservoirs).tolist())
