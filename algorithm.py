#!/usr/bin/env python3
"""Markov text model with PyTorch neural constructors and dynamic tensor execution."""

import argparse
import heapq
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", ""
TOKEN_RE = re.compile(r"[a-z0-9']+|[.,!?;:]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORD_RE = re.compile(r"[a-z0-9']")


def detok(tokens):
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))


def load_sentences(path):
    text = open(path, encoding="utf-8").read().lower()
    sents = [TOKEN_RE.findall(c) for c in SENT_SPLIT.split(text)]
    return [s for s in sents if len(s) >= 2]


class Vocab:
    def __init__(self, sents, min_count=1):
        counts = Counter(t for s in sents for t in s)
        kept = sorted((t for t, c in counts.items() if c >= min_count),
                      key=lambda t: (-counts[t], t))
        self.itos = [PAD, UNK, BOS, EOS] + kept
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.V = len(self.itos)
        self.pad, self.unk, self.bos, self.eos = range(4)
        self.counts = np.zeros(self.V)
        for t in kept:
            self.counts[self.stoi[t]] = counts[t]

    def encode(self, toks):
        return [self.stoi.get(t, self.unk) for t in toks]


class NGram:
    def __init__(self, vocab, order=2, lam=0.7, k=0.01, base_clip=100.0):
        self.v, self.order, self.lam, self.k = vocab, order, lam, k
        self.base_clip = base_clip
        self.tables = [defaultdict(Counter) for _ in range(order + 1)]

    def fit(self, seqs):
        for seq in seqs:
            for t in range(1, len(seq)):
                for n in range(min(self.order, t) + 1):
                    self.tables[n][tuple(seq[t - n:t])][seq[t]] += 1
        base = self.v.counts.copy()
        base[self.v.eos] = len(seqs)
        if self.base_clip < 100:
            pos = base[base > 0]
            base = np.minimum(base, np.percentile(pos, self.base_clip))
        base += self.k
        base[[self.v.pad, self.v.bos, self.v.unk]] = 0.0
        self.base = base / base.sum()

    def dist(self, ctx):
        p = self.base
        ctx = tuple(ctx[-self.order:]) if self.order else ()
        for n in range(len(ctx) + 1):
            c = self.tables[n].get(ctx[len(ctx) - n:] if n else ())
            if c:
                vec = np.zeros(self.v.V)
                for tok, cnt in c.items():
                    vec[tok] = cnt
                p = self.lam * vec / vec.sum() + (1 - self.lam) * p
        return p

    def logits(self, ctx):
        lg = np.log(np.maximum(self.dist(ctx), 1e-300))
        lg[[self.v.pad, self.v.bos, self.v.unk]] = -np.inf
        return torch.tensor(lg, dtype=torch.float32)


class EntropyController(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, logits, min_temp=0.05, max_temp=3.0):
        safe_logits = torch.where(torch.isinf(logits), torch.full_like(logits, -100.0), logits)
        h = self.act(self.fc1(safe_logits))
        temp = torch.sigmoid(self.fc2(h)) * (max_temp - min_temp) + min_temp
        return temp.squeeze(-1)


class LeakyAnchorBias(nn.Module):
    def __init__(self, vocab_size, decay=0.9, interval=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.decay = decay
        self.interval = interval
        self.register_buffer("bias_state", torch.zeros(vocab_size))

    def reset(self):
        self.bias_state.zero_()

    def step(self, step_idx, anchor_ids, amp):
        if len(anchor_ids) == 0 or amp <= 0.0:
            return torch.zeros(self.vocab_size)
        self.bias_state.mul_(self.decay)
        if step_idx % self.interval == 0:
            self.bias_state[anchor_ids] += amp
        return self.bias_state.clone()


class NeuralMarkovEngine(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.entropy_ctrl = EntropyController(vocab_size)
        self.anchor_layer = LeakyAnchorBias(vocab_size)

    def forward(self, ngram_logits, seq_tensor, step_idx, anchor_ids, gen_settings):
        logits = ngram_logits.clone()
        if seq_tensor.numel() > 0:
            offsets = torch.tensor([0], device=seq_tensor.device)
            ctx_vec = self.ctx_embed(seq_tensor.unsqueeze(0), offsets)
            ctx_logits = torch.matmul(ctx_vec, self.ctx_embed.weight.T).squeeze(0)
            logits = logits + 0.1 * ctx_logits

        if gen_settings.anchor_amp > 0.0:
            anchor_bias = self.anchor_layer.step(step_idx, anchor_ids, gen_settings.anchor_amp)
            logits = logits + anchor_bias

        if gen_settings.target_entropy > 0.0:
            temp = self.entropy_ctrl(logits)
        else:
            temp = torch.tensor(max(gen_settings.temp, 1e-3))

        return logits, temp


@dataclass
class Gen:
    max_len: int = 40
    temp: float = 0.9
    top_k: int = 40
    target_entropy: float = 0.0
    beam: int = 0
    min_len: int = 0
    length_alpha: float = 0.7
    anchor_amp: float = 0.0
    anchor_decay: float = 0.9
    anchor_interval: int = 16
    slack: int = 20


def anchor_ids(vocab, toks, skip_top=20):
    common = set(np.argsort(-vocab.counts)[:skip_top].tolist())
    ids = sorted({vocab.stoi[t] for t in toks
                  if t in vocab.stoi and WORD_RE.match(t) and vocab.stoi[t] not in common})
    return torch.tensor(ids, dtype=torch.long)


def top_k_mask_torch(logits, k):
    if 0 < k < logits.size(-1):
        v, _ = torch.topk(logits, k)
        min_val = v[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, -float("inf")), logits)
    return logits


def pick_torch(logits, temp, top_k):
    masked_logits = top_k_mask_torch(logits, top_k)
    probs = F.softmax(masked_logits / temp, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1).item()
    return nxt, temp.item()


def materialize(node):
    out = []
    while node is not None:
        tok, node = node
        out.append(tok)
    return out[::-1]


def beam_search(engine, model, ctx0, eos, max_new, width, top_k, alpha, anchor_ids_tensor, gen):
    path0 = None
    for t in ctx0:
        path0 = (t, path0)
    state_len = max(model.order, 1)
    beam, done = [(0.0, path0)], []

    for step in range(max_new):
        cands = {}
        for score, path in beam:
            seq = materialize(path)
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            raw_logits = model.logits(seq)
            
            if len(seq) - len(ctx0) < gen.min_len:
                raw_logits[eos] = -float("inf")

            logits, _ = engine(raw_logits, seq_tensor, step, anchor_ids_tensor, gen)
            logp = F.log_softmax(logits, dim=-1)

            k = min(top_k if top_k > 0 else 16, len(logp) - 1)
            top_vals, top_indices = torch.topk(logp, k)

            for val, idx in zip(top_vals, top_indices):
                if torch.isinf(val):
                    continue
                tok = idx.item()
                s = score + val.item()
                newp = (tok, path)
                if tok == eos:
                    done.append((s / (step + 1) ** alpha, newp))
                    continue
                tail = tuple(seq[-(state_len - 1):] + [tok]) if state_len > 1 else (tok,)
                if tail not in cands or s > cands[tail][0]:
                    cands[tail] = (s, newp)
        if not cands or len(done) >= width:
            break
        beam = heapq.nlargest(width, cands.values(), key=lambda c: c[0])

    pool = done or [(s / max_new ** alpha, p) for s, p in beam]
    out = materialize(max(pool, key=lambda c: c[0])[1])[len(ctx0):]
    return out[:-1] if out and out[-1] == eos else out


def generate(engine, model, vocab, toks, gen):
    ctx0 = [vocab.bos] + vocab.encode(toks)
    anchors = anchor_ids(vocab, toks) if gen.anchor_amp > 0 else torch.tensor([], dtype=torch.long)
    engine.anchor_layer.reset()

    if gen.beam > 0:
        out_ids = beam_search(engine, model, ctx0, vocab.eos, gen.max_len, gen.beam,
                              gen.top_k, gen.length_alpha, anchors, gen)
        return out_ids, []

    ctx, out, temps = list(ctx0), [], []
    for step_idx in range(gen.max_len):
        seq_tensor = torch.tensor(ctx, dtype=torch.long)
        raw_logits = model.logits(ctx)
        
        if step_idx < gen.min_len:
            raw_logits[vocab.eos] = -float("inf")

        logits, temp = engine(raw_logits, seq_tensor, step_idx, anchors, gen)
        nxt, t = pick_torch(logits, temp, gen.top_k)
        
        if nxt == vocab.eos:
            break
        out.append(nxt)
        ctx.append(nxt)
        temps.append(t)

    return out, temps


def min_steps_to_eos(model, vocab, start):
    if model.order < 1:
        return None
    graph = model.tables[1]
    seen, q = {start}, deque([(start, 0)])
    while q:
        s, d = q.popleft()
        nxt = graph.get((s,))
        if not nxt:
            continue
        if vocab.eos in nxt:
            return d + 1
        for n in nxt:
            if n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return None


def complete(engine, model, vocab, prompt, gen):
    toks = TOKEN_RE.findall(prompt.lower())
    if not toks:
        raise ValueError("prompt has no usable tokens")
    ids = vocab.encode(toks)
    oov = [t for t, i in zip(toks, ids) if i == vocab.unk]
    d = min_steps_to_eos(model, vocab, ids[-1])
    note = None
    if d is None:
        note = "no observed path to end-of-text from the last prompt token"
    else:
        gen = replace(gen, max_len=max(gen.max_len, d + gen.slack))
    out, _ = generate(engine, model, vocab, toks, gen)
    return detok(toks + [vocab.itos[i] for i in out]), oov, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--min-len", type=int, default=0)
    ap.add_argument("--target-entropy", type=float, default=2.0)
    ap.add_argument("--anchor", type=float, default=2.0)
    ap.add_argument("--anchor-decay", type=float, default=0.9)
    ap.add_argument("--anchor-interval", type=int, default=16)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--length-alpha", type=float, default=0.7)
    ap.add_argument("--slack", type=int, default=20)
    ap.add_argument("--base-clip", type=float, default=100.0)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sents = load_sentences(args.path)
    vocab = Vocab(sents, args.min_count)
    model = NGram(vocab, args.order, base_clip=args.base_clip)
    model.fit([[vocab.bos] + vocab.encode(s) + [vocab.eos] for s in sents])

    engine = NeuralMarkovEngine(vocab.V)

    gen = Gen(max_len=args.max_len, temp=args.temp, top_k=args.top_k,
              target_entropy=args.target_entropy, beam=args.beam,
              min_len=args.min_len, length_alpha=args.length_alpha,
              anchor_amp=args.anchor, anchor_decay=args.anchor_decay,
              anchor_interval=args.anchor_interval, slack=args.slack)

    def run(prompt):
        oov, note = set(), None
        for _ in range(args.n):
            text, o, note = complete(engine, model, vocab, prompt, gen)
            oov.update(o)
            print(" ", text)
        if note:
            print(f"  ({note})")
        if oov:
            print(f"  (not in vocab, treated as <unk>: {sorted(oov)})")

    if args.prompt:
        run(args.prompt)
    if args.interactive or not args.prompt:
        print("Type a prompt (empty line to quit).")
        while True:
            try:
                p = input("> ").strip()
            except EOFError:
                break
            if not p:
                break
            try:
                run(p)
            except ValueError as e:
                print(" ", e)


if __name__ == "__main__":
    main()
