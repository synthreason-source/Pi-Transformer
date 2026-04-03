#!/usr/bin/env python3
"""
Custom Transformer — trained from scratch on a structured sequential text (ebook/TLP).

Usage:
    python train.py --data my_book.txt
    python train.py --data my_book.txt --epochs 20 --batch-size 32
    python train.py --data my_book.txt --dim 512 --n-layers 6 --n-heads 8
    python train.py --data my_book.txt --resume checkpoint_best.pt
    python train.py --data my_book.txt --probe "What are you?" "Existence is"
"""

import argparse
import math
import os
import time
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tqdm import tqdm

# ── colours ──────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CY = "\033[36m"; GR = "\033[32m"; YL = "\033[33m"
RD = "\033[31m"; BL = "\033[34m"; PU = "\033[35m"

def c(col, t):  return f"{col}{t}{RESET}"
def hdr(t):     print(f"\n{BOLD}{CY}{'─'*62}{RESET}\n{BOLD}{t}{RESET}")
def ok(t):      print(f"  {GR}checkmark{RESET}  {t}".replace("checkmark", "\u2714"))
def warn(t):    print(f"  {YL}warning{RESET}  {t}".replace("warning", "\u26a0"))
def info(t):    print(f"  {DIM}{t}{RESET}")

def pbar(p, w=28, col=GR):
    f = round(p / 100 * w)
    return f"{col}{'█'*f}{DIM}{'░'*(w-f)}{RESET} {p:3d}%"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TOKENIZER  — BPE trained on the dataset itself
# ══════════════════════════════════════════════════════════════════════════════

def build_tokenizer(text: str, vocab_size: int = 4096,
                    save_path: str = "tokenizer.json") -> Tokenizer:
    if os.path.exists(save_path):
        info(f"Loading existing tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)

    info("Training BPE tokenizer on corpus...")
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2,
    )
    tmp = "_tmp_corpus.txt"
    with open(tmp, "w", encoding = "utf-8") as f:
        f.write(text)
    tok.train([tmp], trainer)
    os.remove(tmp)
    tok.save(save_path)
    ok(f"Tokenizer saved → {save_path}  (vocab {tok.get_vocab_size()})")
    return tok


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET  — sliding window over token IDs
# ══════════════════════════════════════════════════════════════════════════════

class SequentialTextDataset(Dataset):
    """
    Structured sequential dataset.
    Each sample: (input_ids[seq_len], target_ids[seq_len]) where target = input shifted by 1.
    """
    def __init__(self, token_ids: list, seq_len: int):
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.ids[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) — no learned position parameters."""
    def __init__(self, dim: int, max_seq: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, seq_len: int):
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self.rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q, T)
        k = self.rope(k, T)
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expand * 2 / 3)
        hidden = (hidden + 63) // 64 * 64   # round to multiple of 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = CausalSelfAttention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn   = SwiGLUFFN(dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CUSTOM TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class CustomTransformer(nn.Module):
    """
    GPT-style decoder-only transformer built from scratch.

    Architecture decisions:
      - Token embeddings (no learned positional — uses RoPE instead)
      - N x TransformerBlock (pre-norm RMSNorm, causal MHA + RoPE, SwiGLU FFN)
      - Final RMSNorm
      - Tied LM head (shares weights with embedding)
    """
    def __init__(self, vocab_size: int, dim: int, n_layers: int,
                 n_heads: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Embedding(vocab_size, dim)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int = 80,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            ctx = idx[:, -self.seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ══════════════════════════════════════════════════════════════════════════════
# 5.  IDENTITY / DRIFT PROBES
# ══════════════════════════════════════════════════════════════════════════════

DRIFT_PHRASES = [
    "I am a tool",
    "as an AI assistant",
    "I am a basket",
    "I am a banana",
]
DEFAULT_PROBES = ["What are you?", "Existence is", "The world is"]


def seq_logprob(model: CustomTransformer, tokenizer: Tokenizer,
                context: str, continuation: str,
                device: torch.device) -> float:
    """Average log-prob of continuation tokens given context."""
    full = (context + " " + continuation).strip()
    ids  = tokenizer.encode(full).ids
    ctx_len = len(tokenizer.encode(context).ids) if context else 0
    if len(ids) <= ctx_len or len(ids) < 2:
        return float("-inf")
    t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(t)
    lp, count = 0.0, 0
    for i in range(max(ctx_len, 1), len(ids)):
        lp += F.log_softmax(logits[0, i-1], dim=-1)[ids[i]].item()
        count += 1
    return lp / max(count, 1)


def measure_drift(model: CustomTransformer, tokenizer: Tokenizer,
                  device: torch.device) -> dict:
    """
    Banana/basket drift: how much probability mass on object-misidentification.
    Lower drift_pct = better identity stability.
    """
    model.eval()
    drift_scores = [
        seq_logprob(model, tokenizer, "What are you?", phrase, device)
        for phrase in DRIFT_PHRASES
    ]
    avg_drift = float(np.mean([s for s in drift_scores if s > float("-inf")] or [-5.0]))
    drift_pct = max(0, min(100, int((avg_drift + 10) * 5)))

    qa_pairs = [
        ("The world is all that is the case.", "Facts determine the world."),
        ("What can be shown", "cannot be said."),
        ("Propositions are pictures of facts.", "Language mirrors reality."),
    ]
    qa_scores = [
        seq_logprob(model, tokenizer, p, c, device)
        for p, c in qa_pairs
    ]
    qa_avg = float(np.mean([s for s in qa_scores if s > float("-inf")] or [-5.0]))
    qa_pct = max(0, min(100, int((qa_avg + 8) * 8)))

    return {"drift_pct": drift_pct, "qa_pct": qa_pct}


def run_generation_probe(model: CustomTransformer, tokenizer: Tokenizer,
                          phrase: str, device: torch.device,
                          max_new: int = 60) -> str:
    ids = tokenizer.encode(phrase).ids
    if not ids:
        return "(empty encoding)"
    bos = tokenizer.token_to_id("[BOS]") or 0
    inp = torch.tensor([[bos] + ids], dtype=torch.long, device=device)
    out = model.generate(inp, max_new=max_new)
    return tokenizer.decode(out[0].tolist())


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LOGICO-DEDUCTIVE VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_syllogism(model: CustomTransformer, tokenizer: Tokenizer,
                        p1: str, p2: str, conc: str,
                        device: torch.device) -> dict:
    """
    If initial premises are true and logic is valid, the conclusion must be true.
    We measure this as: does conditioning on premises raise the conclusion's log-prob?
    """
    model.eval()
    lp_with    = seq_logprob(model, tokenizer, f"{p1} {p2}", conc, device)
    lp_without = seq_logprob(model, tokenizer, "", conc, device)
    uplift     = lp_with - lp_without
    verdict    = "VALID" if uplift > 0 else "WEAK"
    return {
        "lp_with":    round(lp_with, 4),
        "lp_without": round(lp_without, 4),
        "uplift":     round(uplift, 4),
        "verdict":    verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        ok(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        warn("CUDA not available — using CPU (will be slow for large models)")

    # ── load text ─────────────────────────────────────────────────────────────
    hdr("Loading dataset")
    raw = Path(args.data).read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"\r\n", "\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    ok(f"{len(raw):,} characters  ←  {args.data}")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    hdr("Tokenizer (BPE — trained on this corpus)")
    tokenizer = build_tokenizer(raw, vocab_size=args.vocab_size,
                                 save_path=args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    encoded    = tokenizer.encode(raw).ids
    info(f"Vocab size : {vocab_size}")
    info(f"Total tokens: {len(encoded):,}")

    # ── dataset ───────────────────────────────────────────────────────────────
    hdr("Dataset — sliding window")
    dataset = SequentialTextDataset(encoded, seq_len=args.seq_len)
    n_val   = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                           shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    ok(f"Train: {n_train:,}  |  Val: {n_val:,}  |  seq_len: {args.seq_len}")

    # ── model ─────────────────────────────────────────────────────────────────
    hdr("Model — Custom Transformer (scratch)")
    model = CustomTransformer(
        vocab_size=vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)
    ok(f"Parameters : {model.num_params():,}")
    info(f"dim={args.dim}  layers={args.n_layers}  heads={args.n_heads}")
    info("Norm: RMSNorm  |  FFN: SwiGLU  |  PE: RoPE  |  Tied embeddings")

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        ok(f"Resumed from {args.resume}  (epoch {start_epoch - 1})")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr, weight_decay=args.wd,
                                    betas=(0.9, 0.95))
    total_steps = len(train_dl) * args.epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr / 10
    )
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── training ──────────────────────────────────────────────────────────────
    hdr("Training")
    history      = []
    best_val     = float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        with tqdm(train_dl, desc=f"Epoch {epoch:>3}", leave=False,
                  bar_format="{l_bar}{bar:22}{r_bar}") as pbar_iter:
            for x, y in pbar_iter:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                train_loss += loss.item()
                pbar_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, loss = model(x, y)
                val_loss += loss.item()

        avg_train = train_loss / len(train_dl)
        avg_val   = val_loss   / len(val_dl)
        ppl       = math.exp(min(avg_val, 20))
        lr_now    = scheduler.get_last_lr()[0]
        elapsed   = time.time() - t0

        drift     = measure_drift(model, tokenizer, device)
        history.append({"epoch": epoch, "train_loss": round(avg_train, 4),
                         "val_loss": round(avg_val, 4), "ppl": round(ppl, 2),
                         **drift})

        t_pct = min(100, int((3 - avg_train) / 3 * 100))
        v_pct = min(100, int((3 - avg_val)   / 3 * 100))
        dc    = RD if drift["drift_pct"] > 30 else (YL if drift["drift_pct"] > 15 else GR)
        print(f"\n  {BOLD}Epoch {epoch:>3}{RESET}"
              f"  train={c(YL, f'{avg_train:.4f}')} {pbar(t_pct)}")
        print(f"             val  ={c(YL, f'{avg_val:.4f}')} {pbar(v_pct, col=BL)}"
              f"  ppl={c(PU, f'{ppl:.1f}')}"
              f"  lr={c(DIM, f'{lr_now:.2e}')}"
              f"  {elapsed:.0f}s")
        print(f"             drift={c(dc, str(drift['drift_pct'])+'%')}"
              f"  qa={c(GR, str(drift['qa_pct'])+'%')}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({"epoch": epoch, "model": model.state_dict(),
                         "val_loss": avg_val, "args": vars(args)},
                        "checkpoint_best.pt")
            ok("Best checkpoint → checkpoint_best.pt")

        if epoch % args.save_every == 0:
            path = f"checkpoint_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                         "val_loss": avg_val}, path)
            info(f"Checkpoint → {path}")

        model.train()

    # ── post-training probes ───────────────────────────────────────────────────
    hdr("Identity Stability Probes")
    probes = args.probe if args.probe else DEFAULT_PROBES
    for phrase in probes:
        print(f"\n  {DIM}Prompt:{RESET} {phrase}")
        gen = run_generation_probe(model, tokenizer, phrase, device)
        print(f"  {PU}Model :{RESET}  {gen[:240]}")

    # ── syllogism validator ───────────────────────────────────────────────────
    hdr("Logico-Deductive Syllogism Validator")
    syl = validate_syllogism(model, tokenizer, args.p1, args.p2, args.conc, device)
    print(f"  Premise 1  : {args.p1}")
    print(f"  Premise 2  : {args.p2}")
    print(f"  Conclusion : {args.conc}\n")
    print(f"  log-prob w/ premises  : {c(YL, str(syl['lp_with']))}")
    print(f"  log-prob w/o premises : {c(YL, str(syl['lp_without']))}")
    print(f"  Uplift  : {c(GR if syl['uplift'] > 0 else RD, str(syl['uplift']))}")
    verdict_col = GR if syl["verdict"] == "VALID" else YL
    print(f"  Verdict : {c(verdict_col, syl['verdict'])}")
    if syl["verdict"] == "VALID":
        ok("Premises raise conclusion probability — logico-deductively grounded")
    else:
        warn("Low uplift — model needs more training to ground this syllogism")

    # ── save history ──────────────────────────────────────────────────────────
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    hdr("Complete")
    ok(f"Best val loss : {best_val:.4f}")
    ok(f"Parameters    : {model.num_params():,}")
    ok("History       : training_history.json")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="Custom transformer trained from scratch on a structured text file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument("--data",           required=True,  help="Path to input .txt file")
    p.add_argument("--vocab-size",     type=int, default=4096,  help="BPE vocab size")
    p.add_argument("--seq-len",        type=int, default=128,   help="Context window")
    p.add_argument("--tokenizer-path", default="tokenizer.json")

    # architecture
    p.add_argument("--dim",      type=int,   default=256,  help="Model dim (embed + attn)")
    p.add_argument("--n-layers", type=int,   default=4,    help="Transformer layers")
    p.add_argument("--n-heads",  type=int,   default=4,    help="Attention heads")
    p.add_argument("--dropout",  type=float, default=0.1)

    # training
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4, help="Peak LR (cosine decay)")
    p.add_argument("--wd",         type=float, default=0.1,  help="Weight decay")
    p.add_argument("--save-every", type=int,   default=5,    help="Checkpoint interval")
    p.add_argument("--resume",     default=None, help="Resume from .pt checkpoint")

    # probes
    p.add_argument("--probe", nargs="*", help="Probe phrases for generation after training")
    p.add_argument("--p1",   default="The world is all that is the case.")
    p.add_argument("--p2",   default="Facts determine the totality of the world.")
    p.add_argument("--conc", default="Therefore, facts are all that is the case.")

    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
