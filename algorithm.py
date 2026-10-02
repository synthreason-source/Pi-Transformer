#!/usr/bin/env python3
"""Markov text model with prompt completion.

  python markov_gen.py data.txt --prompt "once upon a" --n 3
  python markov_gen.py data.txt --interactive
"""
import argparse
import random
import re
from collections import Counter, defaultdict

import numpy as np

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", ""
TOKEN_RE = re.compile(r"[a-z0-9']+|[.,!?;:]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


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
    """Interpolated n-gram; order = number of previous tokens used."""

    def __init__(self, vocab, order=2, lam=0.7, k=0.01):
        self.v, self.order, self.lam, self.k = vocab, order, lam, k
        self.tables = [defaultdict(Counter) for _ in range(order + 1)]

    def fit(self, seqs):
        for seq in seqs:
            for t in range(1, len(seq)):
                for n in range(min(self.order, t) + 1):
                    self.tables[n][tuple(seq[t - n:t])][seq[t]] += 1
        base = self.v.counts.copy()
        base[self.v.eos] = len(seqs)
        base += self.k
        base[[self.v.pad, self.v.bos, self.v.unk]] = 0.0
        self.base = base / base.sum()

    def dist(self, ctx):
        p = self.base
        ctx = tuple(ctx[-self.order:]) if self.order else ()
        for n in range(len(ctx) + 1):  # shortest -> longest context
            c = self.tables[n].get(ctx[len(ctx) - n:] if n else ())
            if c:
                vec = np.zeros(self.v.V)
                for tok, cnt in c.items():
                    vec[tok] = cnt
                p = self.lam * vec / vec.sum() + (1 - self.lam) * p
        return p

    def sample_next(self, ctx, temp, top_k):
        p = self.dist(ctx).copy()
        p[[self.v.pad, self.v.bos, self.v.unk]] = 0.0
        if 0 < top_k < len(p):
            p[p < np.partition(p, -top_k)[-top_k]] = 0.0
        p **= 1.0 / max(temp, 1e-3)
        return int(np.random.choice(len(p), p=p / p.sum()))


def complete(model, vocab, prompt, max_len=40, temp=0.9, top_k=40):
    """Continue a prompt. Returns (text, oov_words)."""
    toks = TOKEN_RE.findall(prompt.lower())
    if not toks:
        raise ValueError("prompt has no usable tokens")
    ids = vocab.encode(toks)
    oov = [t for t, i in zip(toks, ids) if i == vocab.unk]
    ctx, out = [vocab.bos] + ids, []
    for _ in range(max_len):
        nxt = model.sample_next(ctx, temp, top_k)

        out.append(vocab.itos[nxt])
        ctx.append(nxt)
    return detok(toks + out), oov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--n", type=int, default=1, help="completions per prompt")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sents = load_sentences(args.path)
    vocab = Vocab(sents, args.min_count)
    model = NGram(vocab, args.order)
    model.fit([[vocab.bos] + vocab.encode(s) + [vocab.eos] for s in sents])

    def run(prompt):
        oov = set()
        for _ in range(args.n):
            text, o = complete(model, vocab, prompt, args.max_len, args.temp, args.top_k)
            oov.update(o)
            print(" ", text)
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
