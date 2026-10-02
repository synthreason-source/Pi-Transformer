#!/usr/bin/env python3
"""
Markov text model from a .txt file.

  - vocab: frequency-sorted ids, stable tie-break, special tokens
  - forward chain (static order) + backward chain (reversed view)
  - interpolated n-gram probabilities (proper distributions, add-k base)
  - order-invariant sum encoding vs position-aware encoding
  - held-out perplexity vs the unigram baseline / entropy
  - generation: forward, backward, "around a seed word", and prompt completion

Usage:
  python markov_gen.py data.txt --order 2 --n 5 --seed-word "the"
  python markov_gen.py data.txt --prompt "once upon a" --n 3
  python markov_gen.py data.txt --interactive
"""
import argparse
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", ""
SPECIALS = [PAD, UNK, BOS, EOS]
TOKEN_RE = re.compile(r"[a-z0-9']+|[.,!?;:]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def detok(tokens):
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))


# --------------------------------------------------------------------- data
def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().lower()
    sents = []
    for chunk in SENT_SPLIT.split(text):
        toks = TOKEN_RE.findall(chunk)
        if len(toks) >= 2:
            sents.append(toks)
    return sents


class Vocab:
    """token -> id built once; never re-sorted or reversed afterwards."""

    def __init__(self, sentences, min_count=1):
        counts = Counter(t for s in sentences for t in s)
        kept = [t for t, c in counts.items() if c >= min_count]
        kept.sort(key=lambda t: (-counts[t], t))  # frequency, then alphabetical
        self.itos = SPECIALS + kept
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.V = len(self.itos)
        self.pad, self.unk, self.bos, self.eos = (self.stoi[s] for s in SPECIALS)
        # unigram counts over ids (specials get 0 here; handled in the model)
        self.counts = np.zeros(self.V)
        for t in kept:
            self.counts[self.stoi[t]] = counts[t]

    def encode(self, sent):
        return [self.stoi.get(t, self.unk) for t in sent]

    def decode(self, ids):
        out = [self.itos[i] for i in ids if i not in (self.pad, self.bos, self.eos)]
        return detok(out)

    def unigram_entropy(self):
        p = self.counts[self.counts > 0]
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())


# -------------------------------------------------------------------- model
class NGram:
    """
    Interpolated n-gram. `order` = number of previous tokens used.
    order=1 is a classic Markov chain (bigram), order=2 is trigram, etc.

    `start` / `stop` are the boundary tokens of the sequences this model is
    trained on: forward = (<bos>, <eos>), backward = (<eos>, <bos>).
    The start token is never predicted; the stop token is.
    """

    def __init__(self, vocab, order=2, lam=0.7, k=0.01, start=None, stop=None):
        self.v, self.order, self.lam, self.k = vocab, order, lam, k
        self.start = vocab.bos if start is None else start
        self.stop = vocab.eos if stop is None else stop
        # tables[n][ctx_tuple_of_len_n] -> Counter(next_id)
        self.tables = [defaultdict(Counter) for _ in range(order + 1)]
        self.base = None

    def fit(self, id_seqs):
        for seq in id_seqs:
            for t in range(1, len(seq)):
                for n in range(0, self.order + 1):
                    if t - n < 0:
                        break
                    self.tables[n][tuple(seq[t - n:t])][seq[t]] += 1
        # base: add-k smoothed unigram. Mass on real tokens, <unk>, and the
        # stop token; none on <pad> or the start token (never predicted).
        base = self.v.counts.copy()
        base[self.stop] = len(id_seqs)
        base = base + self.k
        base[[self.v.pad, self.start]] = 0.0
        self.base = base / base.sum()
        return self

    def dist(self, ctx):
        """Full probability vector over the vocab given a context list."""
        p = self.base
        ctx = tuple(ctx[-self.order:]) if self.order else ()
        for n in range(0, len(ctx) + 1):          # shortest -> longest context
            c = self.tables[n].get(ctx[len(ctx) - n:] if n else ())
            if not c:
                continue
            vec = np.zeros(self.v.V)
            for tok, cnt in c.items():
                vec[tok] = cnt
            vec /= vec.sum()
            p = self.lam * vec + (1 - self.lam) * p
        return p

    def logprob(self, seq):
        lp = 0.0
        for t in range(1, len(seq)):
            lp += math.log(self.dist(seq[:t])[seq[t]] + 1e-12)
        return lp

    def sample_next(self, ctx, temp=1.0, top_k=40, banned=()):
        p = self.dist(ctx).copy()
        for b in banned:
            p[b] = 0.0
        if top_k and top_k < len(p):
            cut = np.partition(p, -top_k)[-top_k]
            p[p < cut] = 0.0
        p = p ** (1.0 / max(temp, 1e-3))
        s = p.sum()
        if s == 0:
            return self.stop
        return int(np.random.choice(len(p), p=p / s))


def generate(model, start_ctx, stop_id, max_len=40, temp=0.9, top_k=40):
    ctx, out = list(start_ctx), []
    banned = tuple(b for b in (model.v.pad, model.v.bos, model.v.unk, model.v.eos)
                   if b != stop_id)
    for _ in range(max_len):
        nxt = model.sample_next(ctx, temp, top_k, banned)

        out.append(nxt)
        ctx.append(nxt)
    return out

def complete(model, vocab, prompt, max_len=40, temp=0.9, top_k=40):
    """Continue a free-text prompt. Returns (full_text, oov_words)."""
    toks = TOKEN_RE.findall(prompt.lower())
    if not toks:
        raise ValueError("prompt has no usable tokens")
    ids = vocab.encode(toks)
    oov = [t for t, i in zip(toks, ids) if i == vocab.unk]
    # condition on <bos> + prompt, so the model treats it as a sentence start
    cont = generate(model, [vocab.bos] + ids, vocab.eos,
                    max_len=max_len, temp=temp, top_k=top_k)
    return detok(toks + [vocab.itos[i] for i in cont]), oov


# --------------------------------------------------------------- evaluation
def perplexity(model, seqs):
    lp, n = 0.0, 0
    for s in seqs:
        lp += model.logprob(s)
        n += len(s) - 1
    return math.exp(-lp / n) if n else float("nan")

# ---------------------------------------------------------------- encodings
class Encoder:
    """Sum encoding (order-invariant) vs positional-weighted (order-aware)."""

    def __init__(self, V, dim=16, decay=0.9, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 1, (V, dim))
        self.decay = decay

    def bag_sum(self, ids):
        return self.E[ids].sum(0)

    def positional(self, ids):
        w = self.decay ** np.arange(len(ids))
        return (self.E[ids] * w[:, None]).sum(0)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--n", type=int, default=1, help="sentences to generate")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--lam", type=float, default=0.7, help="interpolation weight")
    ap.add_argument("--k", type=float, default=0.01, help="add-k for the base")
    ap.add_argument("--seed-word", default=None)
    ap.add_argument("--prompt", default=None, help="text to continue")
    ap.add_argument("--interactive", default=True, action="store_true", help="prompt loop")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sents = load_sentences(args.path)
    if not sents:
        raise SystemExit("no usable sentences found in input file")
    random.shuffle(sents)
    split = max(1, int(0.9 * len(sents)))
    train, test = sents[:split], sents[split:]

    vocab = Vocab(train, args.min_count)
    print(f"sentences: {len(train)} train / {len(test)} test | vocab: {vocab.V}")

    def wrap(s):
        return [vocab.bos] + vocab.encode(s) + [vocab.eos]

    fwd_seqs = [wrap(s) for s in train]              # static order
    bwd_seqs = [s[::-1] for s in fwd_seqs]           # reversed view, same ids

    gen_kw = dict(max_len=args.max_len, temp=args.temp, top_k=args.top_k)
    model_kw = dict(order=args.order, lam=args.lam, k=args.k)

    fwd = NGram(vocab, **model_kw).fit(fwd_seqs)
    bwd = NGram(vocab, start=vocab.eos, stop=vocab.bos, **model_kw).fit(bwd_seqs)

    # ---- prompt completion
    def run(prompt):
        oov_all = set()
        for _ in range(args.n):
            text, oov = complete(fwd, vocab, prompt, **gen_kw)
            oov_all.update(oov)
            print(" ", text)
        if oov_all:
            print(f"  (not in vocab, treated as <unk>: {sorted(oov_all)})")

    if args.prompt:
        print(f"\n--- continuing: {args.prompt!r} ---")
        try:
            run(args.prompt)
        except ValueError as e:
            print(" ", e)

    if args.interactive:
        print("\nType a prompt (empty line to quit).")
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
