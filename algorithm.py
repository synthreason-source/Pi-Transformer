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
    via use   = Sn @ cur
    via bytes = B @ ( Sb @ (B.T @ cur) )
    cur       = decay * (mix*via_use + (1-mix)*via_bytes)

NEW  logic mask  (after all the math, before the sampler)
  The value field says WHAT is worth saying; the mask says whether it may be
  said HERE. It is learned from the same data, no external grammar:
    1. roles      words are clustered by their context signature (the same
                  X used for the word graph) into R roles, plus END (. ! ?),
                  PUNCT (, ; :) and BOS (start of sentence).
    2. template   role trigram / bigram tables: which role may follow the
                  last two roles (a learned sentence frame).
    3. rules      explicit logical constraints, checked at every step:
                    R1 role-legality     next role must be licensed by the frame
                    R2 no dangling end   END only after a role that has ended
                                         sentences before, and only after
                                         `min_len` words
                    R3 no stutter        no punctuation right after punctuation,
                                         none opening a sentence
                    R4 no loop           a word may not recur inside `window`
  Each rule contributes a log-bias; the sampler uses
      logit = log P_trigram + beta * value + gamma * mask_bias
  and the mask never dead-ends (penalties are large but finite).
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
SOFT_PUNCT = {",", ";", ":"}
HARD = -30.0                                # "forbidden" that never yields NaN


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
        self.X = X                                   # kept: the mask reuses it
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
        rows, cols, edges = [], [], set()
        for wi, w in enumerate(self.vocab):
            path = [self.states.setdefault(st, len(self.states)) for st in word_states(w, k)]
            rows += [wi] * len(path)
            cols += path
            for a, b in zip(path, path[1:]):
                edges.add((a, b))
                edges.add((b, a))
        ns = len(self.states)
        B = normalize_rows(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, ns)))
        er, ec = zip(*edges) if edges else ([], [])
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
            via_bytes = B @ st
            cur = decay * (mix * via_use + (1 - mix) * via_bytes)
            self.state_field = self.state_field + (1 - mix) * decay * st
            self.field = self.field + cur
            self.tiers.append(cur)

    def read(self, word):
        """Value of ANY string from the byte-state field (works out of vocabulary)."""
        ids = [self.states[s] for s in word_states(word, self.k) if s in self.states]
        return float(np.mean(self.state_field[ids])) if ids else 0.0


class LogicMask:
    """Learned logical template + explicit rules that gate every next word.

    Roles 0..k-1 : clusters of words by context signature
    role END     : . ! ?        role PUNCT : , ; :        role BOS : sentence start
    """

    def __init__(self, vf, tokens, n_classes=12, min_len=4, window=6,
                 floor=1e-3, seed=0, iters=15):
        self.vf, self.min_len, self.window, self.floor = vf, min_len, window, floor
        rng = np.random.default_rng(seed)
        n = len(vf.vocab)

        # 1. roles: spherical k-means on a random projection of X
        Z = vf.X @ rng.standard_normal((vf.X.shape[1], min(32, vf.X.shape[1])))
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        k = max(1, min(n_classes, n))
        cent = Z[rng.choice(n, k, replace=False)].copy()
        lab = np.zeros(n, dtype=int)
        for _ in range(iters):
            lab = np.argmax(Z @ cent.T, axis=1)
            for c in range(k):
                m = lab == c
                if m.any():
                    v = Z[m].mean(axis=0)
                    cent[c] = v / max(np.linalg.norm(v), 1e-12)
        self.END, self.PUNCT, self.BOS = k, k + 1, k + 2
        self.R = k + 2                                   # emit-able roles
        self.role = lab.copy()
        for w, i in vf.idx.items():
            if w in SENTENCE_END:
                self.role[i] = self.END
            elif w in SOFT_PUNCT:
                self.role[i] = self.PUNCT

        # 2. template: role bigram / trigram over sentences (BOS resets)
        self.bi = np.zeros((self.BOS + 1, self.R))
        self.tri = defaultdict(lambda: np.zeros(self.R))
        prev = [self.BOS]
        for t in tokens:
            r = self.role[vf.idx[t]]
            self.bi[prev[-1], r] += 1
            if len(prev) >= 2:
                self.tri[(prev[-2], prev[-1])][r] += 1
            prev = [self.BOS] if t in SENTENCE_END else (prev + [r])[-2:]
        rs = self.bi.sum(axis=1, keepdims=True)
        self.p_end = np.where(rs[:, 0] > 0, self.bi[:, self.END] / np.maximum(rs[:, 0], 1), 0.0)

    def roles_of(self, words):
        return [self.role[self.vf.idx[w]] for w in words if w in self.vf.idx]

    def bias(self, sent_words, cand_words):
        """Log-bias per candidate given the words of the current sentence."""
        vf, fl = self.vf, self.floor
        roles = [self.BOS] + self.roles_of(sent_words)
        last = roles[-1]
        dist = self.tri.get((roles[-2], roles[-1])) if len(roles) >= 2 else None
        if dist is None or dist.sum() == 0:
            dist = self.bi[last]
        dist = dist / max(dist.sum(), 1e-12)
        recent = set(sent_words[-self.window:])
        n_words = sum(1 for w in sent_words if w not in SOFT_PUNCT)
        out = np.zeros(len(cand_words))
        for i, w in enumerate(cand_words):
            r = self.role[vf.idx[w]]
            b = math.log(dist[r] + fl)                                   # R1 legality
            if r == self.END:                                            # R2 no dangling end
                b += math.log(self.p_end[last] + fl)
                if n_words < self.min_len or last in (self.BOS, self.PUNCT):
                    b += HARD
            if r in (self.END, self.PUNCT) and last in (self.BOS, self.END, self.PUNCT):
                b += HARD                                                # R3 no stutter
            if r < self.END and w in recent:                             # R4 no loop
                b += HARD / 3
            out[i] = b
        return out


class Trigrams:
    """Word trigram rule R: (w1, w2) -> next word, backing off to the bigram
    (w2 -> next word). The value field only reranks what R allows."""

    def __init__(self, tokens):
        self.tri, self.bi = defaultdict(Counter), defaultdict(Counter)
        self.starts = Counter(tokens[:1])
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


def _pick(vf, table, beta, rng, mask=None, sent=(), gamma=1.0):
    words = list(table)
    p = np.array([table[w] for w in words], dtype=np.float64)
    logits = np.log(p / p.sum()) + beta * np.exp(vf.field[[vf.idx[w] for w in words]])
    if mask is not None:
        logits = logits + gamma * mask.bias(list(sent), words)
    w = np.exp(logits - logits.max())
    return words[rng.choice(len(words), p=w / w.sum())]


def stream_tokens(vf, tg, beta=3.0, rng=None, ctx=None, mask=None, gamma=1.0, trace=False):
    """Endless token stream. Inside a sentence: trigram rule + value bias +
    logic mask. After a sentence end the context resets."""
    rng = rng or np.random.default_rng()
    ctx = list(ctx or [])
    sent = list(ctx)                         # words of the current sentence
    while True:
        table = tg.table(ctx) if ctx else tg.starts
        if not table:
            ctx, sent = [], []
            continue
        w = _pick(vf, table, beta, rng, mask, sent, gamma)
        if trace and mask is not None:
            print(f"[mask] {w!r} role={mask.role[vf.idx[w]]}", file=sys.stderr)
        yield w
        if w in SENTENCE_END:
            ctx, sent = [], []
        else:
            ctx = (ctx + [w])[-2:]
            sent.append(w)


def stream_text(tokens, first=True, cap=True):
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
    ap.add_argument("--gamma", type=float, default=1.0, help="weight of the logic mask")
    ap.add_argument("--classes", type=int, default=12, help="number of learned roles")
    ap.add_argument("--min-len", type=int, default=4, help="min words before a sentence may end")
    ap.add_argument("--no-mask", action="store_true", help="disable the logic mask")
    ap.add_argument("--trace", action="store_true", help="print mask role decisions to stderr")
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

    # logic mask: learned after all the math, gates the sampler
    mask = None if args.no_mask else LogicMask(vf, tokens, n_classes=args.classes,
                                               min_len=args.min_len)

    # stream: print the prompt, then continue it
    tg = Trigrams(tokens)
    ended = bool(ptoks) and ptoks[-1] in SENTENCE_END
    ctx0 = [] if (not ptoks or ended) else ptoks[-2:]
    toks = stream_tokens(vf, tg, args.beta, np.random.default_rng(gen_seed), ctx=ctx0,
                         mask=mask, gamma=args.gamma, trace=args.trace)

    for n, chunk in enumerate(stream_text(toks, first=False, cap=ended), 1):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        if args.delay:
            time.sleep(args.delay)
        if args.words and n >= args.words:
            break

    print()
