#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic-Backprop — a real, trainable neural network version
===================================================================

The original V18-CUDA script computes everything with hand-tuned kernels
under @torch.no_grad(): rho/theta/sigma "geometry" comes from fixed
trigonometric formulas, and the "DNN layers" (sigma layer, theta layer,
dim2-relu layer, rho layer) just apply those fixed weights — nothing is
learned, so there is nothing to backpropagate through.

This file keeps the same *shape* of the pipeline —

    token -> geometric features (rho, theta, sigma)
          -> sigma layer -> theta layer -> dim2-relu layer -> rho layer
          -> synaptic (self-attention) mixing across the sequence
          -> vocab logits

— but every one of those stages is now an nn.Module with real trainable
parameters, and the whole thing is trained with cross-entropy loss and
optimizer.step() / loss.backward(), i.e. actual backprop, on next-token
prediction over a text corpus.

Usage:
    python neurosymbolic_backprop.py --corpus mytext.txt --epochs 10
    python neurosymbolic_backprop.py --corpus mytext.txt --generate "once upon a time"
"""

from __future__ import annotations
import argparse, math, re
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = best_device()


# ─────────────────────────────────────────────────────────────────────────
# Exact curve functions from the original script — same math, unchanged.
# These are ordinary differentiable torch ops, so gradients flow through
# them fine; nothing here needs @torch.no_grad().
# ─────────────────────────────────────────────────────────────────────────

def smooth_power_relu(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x_safe = x.clamp(-1, 0.5)
    return (x_safe * x_safe) / (x_safe.abs() + eps)


def signed_power(x: torch.Tensor, p: float, cap: float = 50.0) -> torch.Tensor:
    # cap lowered from the original's 3011.0 since features here are
    # already small (post layer-norm / bounded heads); the curve shape
    # (sign(x)*|x|^p) is identical.
    return x.sign() * (x.abs().clamp(max=cap) + 1e-12).pow(p)


def raised_cosine(delta_theta: torch.Tensor) -> torch.Tensor:
    """k_ori(theta_a, theta_b) = 0.5*(1+cos(theta_b - theta_a)) — identical
    to ThebaultKernels.k_ori in the original."""
    return 0.5 * (1.0 + torch.cos(delta_theta))


def gaussian_kernel(delta: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """exp(-scale * delta^2) — identical curve shape to k_reg / k_side in
    the original (lambda_reg / gamma_side become learnable here)."""
    return torch.exp((-scale * delta * delta).clamp(min=-50.0))


def theta_weight_curve(theta: torch.Tensor) -> torch.Tensor:
    """DNNArrayPipeline._theta_weights: 0.5*(1+cos(theta)) — same formula."""
    return 0.5 * (1.0 + torch.cos(theta))


def sigma_weight_curve(sigma_norm: torch.Tensor) -> torch.Tensor:
    """DNNArrayPipeline._sigma_weights: 0.7 + 0.3*sigma_norm — same affine
    curve, bounded to [0.7, 1.0]."""
    return 0.7 + 0.3 * sigma_norm


def rho_weight_curve(rho: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """DNNArrayPipeline._rho_weights: z-score the batch of rho, clamp to
    [-2.5, 2.5], map through 1 + 0.5*z — same formula, same clamp range.

    unbiased=False (population std, divide by N not N-1) is important here:
    during generation the sequence length can be 1 (e.g. a one-token
    prompt), and the default Bessel-corrected std is 0/0 = NaN for N=1,
    which silently poisons everything downstream. Population std of a
    single element is a well-defined 0.
    """
    mu = rho.mean(dim=-1, keepdim=True)
    std = rho.std(dim=-1, keepdim=True, unbiased=False) + eps
    z = ((rho - mu) / std).clamp(-2.5, 2.5)
    return 1.0 + 0.5 * z


# ─────────────────────────────────────────────────────────────────────────
# Tokenizer + vocab (simple word-level, mirrors the spirit of the original)
# ─────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    text = text.replace("\n", " \n ")
    return re.findall(r"\w+|[.,!?;:]|\n", text.lower())


def build_trigrams(tokens: List[str]) -> "OrderedDict[Tuple[str, str, str], int]":
    """(w1, w2, w3) -> count, in first-seen order. Same idea as the original
    ThebaultCompositionLM.ingest()'s tri_raw table, just standalone here."""
    from collections import OrderedDict
    tri: "OrderedDict[Tuple[str, str, str], int]" = OrderedDict()
    for i in range(len(tokens) - 2):
        key = (tokens[i], tokens[i + 1], tokens[i + 2])
        tri[key] = tri.get(key, 0) + 1
    return tri


def top_trigrams(tri: "Dict[Tuple[str, str, str], int]", k: int = 15) -> List[Tuple[Tuple[str, str, str], int]]:
    return sorted(tri.items(), key=lambda kv: -kv[1])[:k]


class Vocab:
    def __init__(self, tokens: List[str]):
        uniq = sorted(set(tokens))
        self.itos = ["<pad>", "<unk>"] + uniq
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> torch.Tensor:
        return torch.tensor([self.stoi.get(t, 1) for t in tokens], dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        toks = [self.itos[i] for i in ids]
        out = []
        for t in toks:
            if t in {".", ",", "!", "?", ";", ":"}:
                if out:
                    out[-1] += t
            else:
                out.append(t)
        return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────
# Learnable geometric embedding — replaces the fixed Thebault-triple math
# with three small trainable heads (rho, theta, sigma) on top of a token
# embedding. These are ordinary parameters trained by backprop, not the
# closed-form trig formulas from the original script.
# ─────────────────────────────────────────────────────────────────────────

class GeometricEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, max_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb   = nn.Embedding(vocab_size, dim)
        self.pos_emb   = nn.Embedding(max_len, dim)
        self.rho_head  = nn.Linear(dim, 1)
        self.theta_head = nn.Linear(dim, 1)
        self.sigma_head = nn.Linear(dim, 1)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape

        # Fail with a clear CPU-side error instead of a CUDA device-side
        # assert deep in an embedding kernel. This forces a sync (cheap —
        # it's just two reductions), which is worth it: without it, an
        # out-of-range token id or a sequence longer than max_len shows up
        # as an opaque "CUDA error: device-side assert triggered" that
        # points at the wrong line and often corrupts the CUDA context for
        # the rest of the process.
        if T > self.max_len:
            raise ValueError(
                f"sequence length {T} exceeds GeometricEmbedding.max_len={self.max_len}; "
                f"pass a larger max_len to NeuroSymbolicNet (>= your --block-size), "
                f"or reduce --block-size."
            )
        tok_min, tok_max = idx.min().item(), idx.max().item()
        if tok_min < 0 or tok_max >= self.vocab_size:
            raise ValueError(
                f"token id out of range: min={tok_min}, max={tok_max}, "
                f"but vocab_size={self.vocab_size}. This usually means the "
                f"Vocab used to encode ids doesn't match the vocab the model "
                f"was built/loaded with."
            )

        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        e = self.tok_emb(idx) + self.pos_emb(pos)
        rho   = torch.sigmoid(self.rho_head(e)).squeeze(-1)          # [0,1]
        theta = torch.tanh(self.theta_head(e)).squeeze(-1) * math.pi  # (-pi, pi)
        sigma = torch.sigmoid(self.sigma_head(e)).squeeze(-1)         # [0,1]
        return e, rho, theta, sigma


# ─────────────────────────────────────────────────────────────────────────
# Trainable "dim-2 relu" layer — same gating idea as the original
# (_dim2_relu_layer), but the gate is now derived from a learnable theta
# projection rather than a fixed cosine kernel, and gradients flow through
# it normally (no torch.no_grad()).
# ─────────────────────────────────────────────────────────────────────────

class Dim2ReluLayer(nn.Module):
    """Identical curve to the original _dim2_relu_layer:
        theta_w  = 0.5*(1+cos(theta))
        gate_raw = relu(theta_w - mean(theta_w))
        gate     = gate_raw / max(gate_raw)
        out      = x*gate + relu(x)*(1-gate)
    """
    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        theta_w  = theta_weight_curve(theta)                      # [B,T] in [0,1]
        gate_raw = F.relu(theta_w - theta_w.mean(dim=-1, keepdim=True))
        g_max    = gate_raw.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        gate     = (gate_raw / g_max).unsqueeze(-1)                # [B,T,1]
        return x * gate + F.relu(x) * (1.0 - gate)


# ─────────────────────────────────────────────────────────────────────────
# Trainable "synaptic sum" — the differentiable analogue of
# CrossSynapticNeuronSum / build_synaptic_weight_matrix: instead of a
# fixed RBF-kernel weight matrix, this is ordinary learned causal
# self-attention, which plays the same structural role (mixing
# information across candidate/context tokens) but with real gradients.
# ─────────────────────────────────────────────────────────────────────────

class SynapticSelfAttention(nn.Module):
    """Differentiable analogue of build_synaptic_weight_matrix / CSNS.

    The original computed a fixed weight matrix
        W = k_reg(rho) * k_ori(theta) * k_side(sigma)
    with k_reg/k_side Gaussian kernels and k_ori a raised cosine, using
    fixed lambda_reg=811.0 / gamma_side=411.0. Here those same three curve
    shapes are computed per pair of positions and added (in log-space, so
    the product becomes a sum) as a bias to ordinary learned Q/K attention
    scores — same curves, but lambda_reg/gamma_side are now nn.Parameters
    trained by backprop instead of hand-picked constants.
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # learnable analogues of lambda_reg / gamma_side, softplus-ed to
        # stay positive (same role, same curve family as the original)
        self.raw_lambda_reg = nn.Parameter(torch.tensor(2.0))
        self.raw_gamma_side = nn.Parameter(torch.tensor(2.0))
        self.kernel_weight  = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, rho: torch.Tensor, theta: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # [B,H,T,hd]

        lambda_reg = F.softplus(self.raw_lambda_reg).clamp(max=50.0)
        gamma_side = F.softplus(self.raw_gamma_side).clamp(max=50.0)

        d_rho   = rho.unsqueeze(-1)   - rho.unsqueeze(-2)     # [B,T,T]
        d_theta = theta.unsqueeze(-1) - theta.unsqueeze(-2)
        d_sigma = sigma.unsqueeze(-1) - sigma.unsqueeze(-2)

        k_reg  = gaussian_kernel(d_rho, lambda_reg)     # same curve as original k_reg
        k_ori  = raised_cosine(d_theta)                 # same curve as original k_ori
        k_side = gaussian_kernel(d_sigma, gamma_side)    # same curve as original k_side

        kernel_bias = (k_reg * k_ori * k_side).clamp(min=1e-8).log()  # [B,T,T]
        kernel_bias = kernel_bias.unsqueeze(1)                         # [B,1,T,T]

        kw = self.kernel_weight.clamp(-10.0, 10.0)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + kw * kernel_bias
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                       # [B,H,T,hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────────────────
# The network: sigma layer -> theta layer -> dim2-relu -> rho layer ->
# synaptic self-attention -> vocab logits. Mirrors the reversed layer
# order from DNNArrayPipeline.forward, but every weight here is learned.
# ─────────────────────────────────────────────────────────────────────────

class NeuroSymbolicNet(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, hidden: int = 256,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1,
                 max_len: int = 2048):
        super().__init__()
        self.geo = GeometricEmbedding(vocab_size, dim, max_len=max_len)

        self.sigma_layer = nn.Linear(dim, hidden)
        self.theta_layer = nn.Linear(hidden, hidden)
        self.dim2_relu   = Dim2ReluLayer()
        self.rho_layer   = nn.Linear(hidden, hidden)

        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=SynapticSelfAttention(hidden, n_heads),
                ln1=nn.LayerNorm(hidden),
                ff=nn.Sequential(
                    nn.Linear(hidden, hidden * 4), nn.GELU(),
                    nn.Linear(hidden * 4, hidden),
                ),
                ln2=nn.LayerNorm(hidden),
            ))
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(hidden)
        self.out  = nn.Linear(hidden, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        e, rho, theta, sigma = self.geo(idx)

        # Layer 1 — sigma modulation: same curve as _sigma_weights (0.7+0.3*sigma),
        # then signed_power(., p=1.0) as in the original z1.
        sig_w = sigma_weight_curve(sigma)                       # [B,T] in [0.7,1.0]
        z1 = signed_power(self.sigma_layer(e) * sig_w.unsqueeze(-1), p=1.0)

        # Layer 2 — theta modulation: same curve as _theta_weights
        # (0.5*(1+cos(theta))), residual + raw features * 0.3, signed_power(p=1.5).
        theta_w = theta_weight_curve(theta)                     # [B,T] in [0,1]
        z2 = signed_power(self.theta_layer(z1) * theta_w.unsqueeze(-1) + z1 * 0.3, p=1.5)

        # Layer 2b — dim-2 relu, identical gate formula to the original.
        z2b = self.dim2_relu(z2, theta)

        # Layer 3 — rho amplification: same curve as _rho_weights
        # (z-score clamp to [-2.5,2.5], 1+0.5*z), signed_power(p=2.0).
        rho_w = rho_weight_curve(rho)                            # [B,T]
        z3 = signed_power(self.rho_layer(z2b) * rho_w.unsqueeze(-1), p=2.0)
        z3 = self.dropout(z3)

        x = z3
        for blk in self.blocks:
            x = x + blk["attn"](blk["ln1"](x), rho, theta, sigma)
            x = x + blk["ff"](blk["ln2"](x))

        x = self.ln_f(x)
        logits = self.out(x)
        # Defensive: nothing upstream should be able to produce NaN/Inf given
        # the clamps in signed_power / gaussian_kernel / rho_weight_curve,
        # but if training has diverged (e.g. too high --lr) we want a clean
        # signal rather than letting garbage reach softmax/multinomial where
        # it shows up as an opaque CUDA device-side assert.
        return torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)


# ─────────────────────────────────────────────────────────────────────────
# Training loop — real backprop: forward pass -> cross-entropy loss ->
# loss.backward() -> optimizer.step().
# ─────────────────────────────────────────────────────────────────────────

def make_batches(ids: torch.Tensor, block_size: int, batch_size: int, device):
    n = ids.shape[0] - block_size - 1
    while True:
        starts = torch.randint(0, n, (batch_size,))
        x = torch.stack([ids[s:s + block_size] for s in starts]).to(device)
        y = torch.stack([ids[s + 1:s + 1 + block_size] for s in starts]).to(device)
        yield x, y


def train(model: NeuroSymbolicNet, ids: torch.Tensor, vocab_size: int,
          steps: int = 2000, block_size: int = 64, batch_size: int = 32,
          lr: float = 3e-4, device=DEVICE, log_every: int = 100):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batches = make_batches(ids, block_size, batch_size, device)

    model.train()
    for step in range(1, steps + 1):
        x, y = next(batches)

        logits = model(x)                                    # forward pass
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        if not torch.isfinite(loss):
            # With the clamps in the model this shouldn't happen, but if it
            # does, stop now with a clear message rather than continuing to
            # train on garbage (which is what eventually surfaces as a
            # confusing CUDA assert deep inside torch.multinomial later,
            # during generation).
            raise RuntimeError(
                f"loss became non-finite ({loss.item()}) at step {step} — "
                f"training has diverged. Try a smaller --lr (current={lr}), "
                f"a smaller --dim/--hidden, or fewer --layers."
            )

        opt.zero_grad()
        loss.backward()                                       # <-- backprop
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                                            # <-- weight update

        if step % log_every == 0 or step == 1:
            print(f"[step {step:5d}] loss={loss.item():.4f}  ppl={math.exp(min(loss.item(), 20)):.2f}")
    return model


@torch.no_grad()
def generate(model: NeuroSymbolicNet, vocab: Vocab, prompt: str,
             max_new_tokens: int = 60, temperature: float = 1.0,
             block_size: int = 64, device=DEVICE) -> str:
    model.eval()
    toks = tokenize(prompt) or ["<unk>"]
    ids = vocab.encode(toks).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -block_size:]
        # temperature floor raised from 1e-6 to 0.05: dividing already-large
        # logits by something near 1e-6 can overflow to inf, which turns
        # into NaN after softmax and crashes multinomial with the assert
        # you hit ("Assertion `input[0] != 0` failed").
        logits = model(ids_cond)[:, -1, :] / max(temperature, 0.05)
        probs = F.softmax(logits, dim=-1)

        # Last line of defense: if anything upstream still produced a
        # degenerate (all-zero / NaN) distribution, fall back to uniform
        # over the vocab instead of letting torch.multinomial hard-crash.
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = probs.sum(dim=-1, keepdim=True)
        if (row_sum <= 0).any():
            probs = torch.where(row_sum <= 0, torch.full_like(probs, 1.0 / probs.shape[-1]), probs)
            row_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / row_sum

        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)

    return vocab.decode(ids[0].tolist())


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True, help="path to a .txt corpus")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--generate", type=str, default=None, help="prompt to sample from after training")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--save", type=str, default="neurosymbolic_backprop.pt")
    args = ap.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    toks = tokenize(text)
    vocab = Vocab(toks)
    ids = vocab.encode(toks)
    print(f"[*] corpus: {len(toks)} tokens, vocab size {len(vocab)}, device={DEVICE}")

    trigrams = build_trigrams(toks)
    print(f"[*] trigrams: {len(trigrams)} unique (w1,w2,w3) windows")
    print("[*] top trigrams:")
    for (w1, w2, w3), cnt in top_trigrams(trigrams, k=10):
        print(f"      {cnt:4d}x  {w1!r} {w2!r} {w3!r}")

    if len(ids) <= args.block_size + 1:
        raise ValueError(
            f"corpus has only {len(ids)} tokens, but --block-size={args.block_size} "
            f"needs at least {args.block_size + 2}. Use a longer corpus or a smaller --block-size."
        )

    model = NeuroSymbolicNet(
        vocab_size=len(vocab), dim=args.dim, hidden=args.hidden, n_layers=args.layers,
        max_len=args.block_size,   # tie positional capacity exactly to --block-size
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] model params: {n_params:,}")

    train(model, ids, len(vocab), steps=args.steps, block_size=args.block_size,
          batch_size=args.batch_size, lr=args.lr)

    torch.save({"model_state": model.state_dict(), "vocab_itos": vocab.itos,
                "dim": args.dim, "hidden": args.hidden, "layers": args.layers,
                "block_size": args.block_size}, args.save)
    print(f"[+] saved checkpoint to {args.save}")
    while True:
        prompt = input("USER: ")
        sample = generate(model, vocab, prompt, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, block_size=args.block_size)
        print("\n--- SAMPLE ---")
        print(sample)


if __name__ == "__main__":
    main()
