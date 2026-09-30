#!/usr/bin/env python3
"""Assign a value to the words of a prompt; it unfolds THROUGH the byte
encoding, and the text stream continues the prompt.

Two graphs, joined by the spelling of each word:

  word graph   Sn   words compared by the contexts they live in (like-minded
                    by use): cosine of (left ++ right) bigram signatures,
                    top-n, symmetrized.
  byte graph   Sb   byte 3-gram states (with word-boundary marks) linked when
                    they follow each other inside a word: like-minded by form.
  interface    B    word <-> the byte states along its spelling.

One unfolding hop from a word-value vector `cur`:
    via use   = Sn @ cur                       # cross comparison of words
    via bytes = B @ ( Sb @ (B.T @ cur) )       # deposit on byte states,
                                               # spread on the byte graph,
                                               # read back onto words
    cur       = decay * (mix*via_use + (1-mix)*via_bytes)
The byte-state field also reads out ANY string, including words that are not
in the data.

The prompt: every prompt word that appears in the data gets a value, scaled by
rarity (log(N/count), the rarest prompt word gets --value), so "the" in a
prompt doesn't swamp "moon". Prompt words not in the data are reported and
get no value.
"""
import argparse
import math
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

SEP = 256                                   # word-boundary symbol
SENTENCE_END = {".", "!", "?"}


def normalize_rows(M):
    s = np.asarray(M.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return sp.diags(1.0 / s) @ M if sp.issparse(M) else M / s[:, None]


def word_states(word, k=3):
    seq = (SEP,) + tuple(word.encode("utf-8")) + (SEP,)
    seq = seq + (SEP,) * max(0, k - len(seq))
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def show_state(st):
    out, buf = [], []
    for b in st:
        if b == SEP:
            if buf:
                out.append(bytes(buf).decode("utf-8", "replace"))
                buf = []
            out.append("|")
        else:
            buf.append(b)
    if buf:
        out.append(bytes(buf).decode("utf-8", "replace"))
    return "".join(out)


class ValueField:
    def __init__(self, tokens, values, top=5, hops=3, decay=0.5, mix=0.5, k=3):
        self.k = k
        self.vocab = sorted(set(tokens))
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(self.vocab)

        # word graph: cross comparison by context
        C = np.zeros((n, n))
        for a, b in zip(tokens, tokens[1:]):
            C[self.idx[a], self.idx[b]] += 1.0
        X = np.hstack([normalize_rows(C), normalize_rows(C.T)])
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        S = X @ X.T
        np.fill_diagonal(S, 0.0)
        K = np.zeros_like(S)
        for i in range(n):
            j = np.argsort(S[i])[-top:]
            K[i, j] = S[i, j]
        K = np.maximum(K, K.T)
        d = np.sqrt(np.maximum(K.sum(axis=1), 1e-12))
        Sn = K / np.outer(d, d)

        # byte graph + word/byte interface
        self.states = {}
        rows, cols, er, ec = [], [], [], []
        for wi, w in enumerate(self.vocab):
            path = [self.states.setdefault(st, len(self.states)) for st in word_states(w, k)]
            rows += [wi] * len(path)
            cols += path
            for a, b in zip(path, path[1:]):
                if a != b:
                    er += [a, b]
                    ec += [b, a]
        ns = len(self.states)
        B = normalize_rows(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, ns)))
        A = sp.csr_matrix((np.ones(len(er)), (er, ec)), shape=(ns, ns))
        A.data[:] = 1.0
        dd = 1.0 / np.sqrt(np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-12))
        Sb = sp.diags(dd) @ A @ sp.diags(dd)
        self.state_list = list(self.states)

        # unfolding: every hop crosses the interface
        v0 = np.zeros(n)
        for w, x in values.items():
            v0[self.idx[w]] = x
        cur, self.field = v0, v0.copy()
        self.tiers = [v0]
        self.state_field = B.T @ v0
        for _ in range(hops):
            via_use = Sn @ cur
            st = Sb @ (B.T @ cur)
            via_bytes = B @ st.T
            cur = decay * (mix * via_use + (1 - mix) * via_bytes)
            self.state_field = self.state_field + (1 - mix) * decay * st
            self.field = self.field + cur
            self.tiers.append(cur)

    def read(self, word):
        """Value of ANY string from the byte-state field (works out of vocabulary)."""
        ids = [self.states[s] for s in word_states(word, self.k) if s in self.states]
        return float(np.mean(self.state_field[ids])) if ids else 0.0


class Trigrams:
    """Word trigram rule R: (w1, w2) -> next word, backing off to the bigram
    (w2 -> next word). The value field only reranks what R allows."""

    def __init__(self, tokens):
        self.tri, self.bi = defaultdict(Counter), defaultdict(Counter)
        self.starts = Counter(tokens[:1])                    # words that open a sentence
        for a, b in zip(tokens, tokens[1:]):
            if a in SENTENCE_END:
                self.starts[b] += 1
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            self.tri[(a, b)][c] += 1
        for a, b in zip(tokens, tokens[1:]):
            self.bi[a][b] += 1

    def table(self, out):
        if len(out) >= 2 and (out[-2], out[-1]) in self.tri:
            return self.tri[(out[-2], out[-1])]
        return self.bi.get(out[-1])


def generate(vf, tg, start, n=40, beta=3.0, rng=None):
    """logit(next) = log P_trigram(next | prev two words) + beta * value(next)."""
    rng = rng or np.random.default_rng()
    out = [start]
    while len(out) < n:
        table = tg.table(out)
        if not table:                       # dead end: restart from the start word
            out.append(start)
            continue
        words = list(table)
        p = np.array([table[w] for w in words], dtype=np.float64)
        logits = np.log(p / p.sum()) + beta * vf.field[[vf.idx[w] for w in words]]
        w = np.exp(logits - logits.max())
        out.append(words[rng.choice(len(words), p=w / w.sum())])
    return sorted(out)


def _pick(vf, table, beta, rng):
    words = list(table)
    p = np.array([table[w] for w in words], dtype=np.float64)
    logits = np.log(p / p.sum()) + beta * np.exp(vf.field[[vf.idx[w] for w in words]])
    w = np.exp(logits - logits.max())
    return words[rng.choice(len(words), p=w / w.sum())]


def stream_tokens(vf, tg, beta=3.0, rng=None, ctx=None):
    """Endless token stream. Inside a sentence: trigram rule + value bias.
    After a sentence end the context resets and the next sentence opens from
    the corpus's sentence-openers, again biased by value, so the stream keeps
    returning to what was valued instead of drifting or looping on a dead end.
    `ctx` seeds the first context (the last two words of the prompt)."""
    rng = rng or np.random.default_rng()
    ctx = list(ctx or [])
    while True:
        table = tg.table(ctx) if ctx else tg.starts
        if not table:
            ctx = []
            continue
        w = _pick(vf, table, beta, rng)
        yield w
        ctx = [] if w in SENTENCE_END else (ctx + [w])[-2:]


def stream_text(tokens, first=True, cap=True):
    """Token stream -> text chunks: spacing, punctuation, sentence case.
    first=False continues after text that is already on screen."""
    for t in tokens:
        if re.match(r"[.,!?;:]", t):
            chunk = t
        else:
            chunk = ("" if first else " ") + (t[:1].upper() + t[1:] if cap else t)
        cap, first = t in SENTENCE_END, False
        yield chunk


def detokenize(tokens):
    text = ""
    for t in tokens:
        text += t if (not text or re.match(r"[.,!?;:]", t)) else " " + t
    return text[:1].upper() + text[1:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt", default="the moon",
                    help="text to continue; its words are valued (rarer words count more)")
    ap.add_argument("--value", type=float, default=3.0, help="value of the rarest prompt word")
    ap.add_argument("--mix", type=float, default=0.5, help="1 = by use only, 0 = by bytes only")
    ap.add_argument("--k", type=int, default=3, help="byte n-gram length of a state")
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--beta", type=float, default=3.0, help="weight of the unfolded value")
    gen_seed = int(np.random.SeedSequence().entropy % 2**32)

    ap.add_argument("--words", type=int, default=600, help="stop the stream after N words (0 = endless)")
    ap.add_argument("--delay", type=float, default=0.03, help="seconds between words")
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8", errors="ignore").read().lower()
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)

    # prompt -> values
    counts = Counter(tokens)
    ptoks = re.findall(r"[\w']+|[.,!?;:]", input("USER: ").lower())
    is_word = lambda t: re.match(r"[\w']+", t)
    known = [w for w in dict.fromkeys(ptoks) if is_word(w) and w in counts]
    unknown = [w for w in dict.fromkeys(ptoks) if is_word(w) and w not in counts]
    if unknown:
        print(f"[warn] not in the data, no value assigned: {unknown}", file=sys.stderr)
    if not known:
        raise SystemExit("no prompt word appears in the data")
    rarity = {w: max(math.log(len(tokens) / counts[w]), 1e-9) for w in known}
    values = {w: args.value * rarity[w] / max(rarity.values()) for w in known}

    vf = ValueField(tokens, values, mix=args.mix, k=args.k)
    order = np.argsort(-vf.field)


    # stream: print the prompt, then continue it
    tg = Trigrams(tokens)
    ended = bool(ptoks) and ptoks[-1] in SENTENCE_END
    ctx0 = [] if (not ptoks or ended) else ptoks[-2:]
    toks = stream_tokens(vf, tg, args.beta, np.random.default_rng(gen_seed), ctx=ctx0)

    for n, chunk in enumerate(stream_text(toks, first=False, cap=ended), 1):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        if n >= args.words:
            break
        
    print()
