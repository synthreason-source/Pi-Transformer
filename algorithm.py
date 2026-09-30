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

NEW  duplicated vocabulary, partitioned linkwise   (--copies N, default 2)
  Every word exists as N identities: "moon", "moon#1", ... (a copy mark that
  the tokenizer can never produce). The dataset is partitioned LINKWISE: each
  distinct link (word-pair transition, e.g. the -> moon) is assigned a class
  by a stable hash, and crossing a link of class c moves the running copy
  forward by c (mod N). Class-0 links stay inside a copy; other classes
  cross between copies. So the trigram table, the word graph, the roles and
  the mask all see the N copies as separate words, while everything that has
  to do with the surface (spelling/bytes, punctuation, printing, rules R2-R4,
  the prompt) uses the surface word. --copies 1 is the program without this.

NEW  the copies solve an NP-hard problem on each other   (agenda)
  Problem: 0/1 KNAPSACK. From the most valuable words, choose the set with the
  largest total value whose spelling fits a character budget (one sentence).
  Brute force is 2^n subsets. Meet in the middle (Horowitz-Sahni) makes it
  2 * 2^(n/2):
      half A  = candidate words that live in the even copies   (sorted ascending)
      half B  = candidate words that live in the odd copies    (sorted REVERSED,
                descending), so that one two-pointer sweep pairs every subset
                of A with the best subset of B that still fits.
  A word lives in the copy where the linkwise partition put most of its
  occurrences. The winning set is the sentence AGENDA: words the stream gets a
  bonus for saying (once each per sentence). --check-solver proves the answer
  against brute force.

NEW  curved context push until the context is pinned to the dataset   (--context curved)
  The dataset is indexed by position at several context lengths ("dimensions").
  At every step the context is pushed up a CURVED schedule of lengths
      k_j = 1 + round(j ** curve)      e.g. 1, 2, 4, 7, 10     (--curve 1.6)
  and each length asks: where in the dataset does this exact context occur?
      - no position         stop: the previous length is the deepest match
      - <= --pin positions  PINNED: the context matches the dataset
                            positionally; the next word is what the dataset
                            itself says at that position
      - several positions   keep pushing
  Whatever length it stops at gives the candidate next words (the positions'
  successors); value, mask and agenda still rerank them. The context is the
  running history, so it carries across sentence ends and can follow the
  dataset from one sentence into the next. --context trigram is the old rule.
"""
import argparse
import hashlib
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
COPY_MARK = "#"                             # never produced by the tokenizer


# ---------------------------------------------------------------------------
# duplicated vocabulary + linkwise partition
# ---------------------------------------------------------------------------

def surf(token):
    """Surface word of a (possibly copy-tagged) token."""
    return token.partition(COPY_MARK)[0]


def copy_of(token):
    _, _, c = token.partition(COPY_MARK)
    return int(c) if c else 0


def tag(word, copy):
    return word if copy == 0 else f"{word}{COPY_MARK}{copy}"


def shift_token(token, offset, copies):
    return tag(surf(token), (copy_of(token) + offset) % copies)


def link_class(first, second, copies, salt=""):
    """Partition the LINKS into `copies` classes with a stable hash: the
    same word-pair transition always lands in the same class."""
    digest = hashlib.blake2b(
        f"{salt}\x00{first}\x00{second}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % copies


def lift_tokens(tokens, copies, salt=""):
    """Surface tokens -> copy-tagged tokens. Crossing a link of class c moves
    the copy forward by c (mod copies). Projecting with surf() returns the
    original dataset exactly."""
    out, copy, previous = [], 0, None
    for t in tokens:
        if previous is not None:
            copy = (copy + link_class(previous, t, copies, salt)) % copies
        out.append(tag(t, copy))
        previous = t
    return out


def lift_prompt(ptoks, tg, copies, salt, rng):
    """A prompt's links fix its copy DIFFERENCES; the corpus supplies the
    absolute copy. Try every offset and pick among those the trigram table
    knows, weighted by how much data supports them."""
    rel = lift_tokens(ptoks, copies, salt)
    if copies == 1 or not rel:
        return rel
    weights = []
    for o in range(copies):
        shifted = [shift_token(t, o, copies) for t in rel]
        w = 0.0
        if len(shifted) >= 2 and (shifted[-2], shifted[-1]) in tg.tri:
            w = float(sum(tg.tri[(shifted[-2], shifted[-1])].values()))
        elif shifted[-1] in tg.bi:
            w = 0.01 * float(sum(tg.bi[shifted[-1]].values()))
        weights.append(w)
    weights = np.array(weights)
    o = 0 if weights.sum() == 0 else int(rng.choice(copies, p=weights / weights.sum()))
    return [shift_token(t, o, copies) for t in rel]


def partition_report(tokens, lifted, copies, salt):
    seen, distinct, occ = set(), Counter(), Counter()
    for a, b in zip(tokens, tokens[1:]):
        c = link_class(a, b, copies, salt)
        occ[c] += 1
        if (a, b) not in seen:
            seen.add((a, b))
            distinct[c] += 1
    return {
        "copies": copies,
        "vocab": len(set(lifted)),
        "distinct_links_per_class": dict(sorted(distinct.items())),
        "occurrences_per_class": dict(sorted(occ.items())),
        "projects_back_to_dataset": [surf(t) for t in lifted] == tokens,
        "copy_moves_follow_link_class": all(
            copy_of(y) == (copy_of(x) + link_class(a, b, copies, salt)) % copies
            for x, y, a, b in zip(lifted, lifted[1:], tokens, tokens[1:])
        ),
    }


# ---------------------------------------------------------------------------
# NP-hard problem between the copies: knapsack by meet in the middle
# ---------------------------------------------------------------------------

def subset_table(items):
    """All 2^n subsets of [(weight, value), ...] as arrays (W, V, bitmask)."""
    W = np.zeros(1, dtype=np.int64)
    V = np.zeros(1)
    M = np.zeros(1, dtype=np.int64)
    for i, (w, v) in enumerate(items):
        W = np.concatenate([W, W + w])
        V = np.concatenate([V, V + v])
        M = np.concatenate([M, M | (1 << i)])
    return W, V, M


def knapsack_mitm(set_a, set_b, budget):
    """Best total value with total weight <= budget, choosing from both sets.
    A is scanned ascending, B REVERSED (descending): as A's weight grows the
    room left for B only shrinks, so one pointer into B moves forward once and
    a suffix maximum answers "best B-subset that still fits"."""
    WA, VA, MA = subset_table(set_a)
    WB, VB, MB = subset_table(set_b)
    oa = np.argsort(WA, kind="stable")                   # A ascending
    ob = np.argsort(-WB, kind="stable")                  # B reversed
    WA, VA, MA = WA[oa], VA[oa], MA[oa]
    WB, VB, MB = WB[ob], VB[ob], MB[ob]

    nb = len(WB)
    sufv = np.empty(nb)
    sufi = np.empty(nb, dtype=np.int64)
    best_v, best_i = -1.0, -1
    for j in range(nb - 1, -1, -1):
        if VB[j] > best_v:
            best_v, best_i = VB[j], j
        sufv[j], sufi[j] = best_v, best_i

    best = (-1.0, 0, 0)
    j = 0
    for a in range(len(WA)):
        room = budget - WA[a]
        if room < 0:
            break                                        # nothing later in A fits either
        while j < nb and WB[j] > room:
            j += 1                                       # pointer only moves forward
        if j == nb:
            break
        total = VA[a] + sufv[j]
        if total > best[0]:
            best = (total, int(MA[a]), int(MB[sufi[j]]))
    return best[0], best[1], best[2], len(WA) + len(WB)


def knapsack_brute(items, budget):
    W, V, _ = subset_table(items)
    return float(V[W <= budget].max())


def solve_agenda(vf, lcounts, copies, pool=18, budget=30, check=False):
    """Candidates = the `pool` most valuable surface words (value taken at the
    word's home copy). Even-copy words form half A, odd-copy words half B
    (with one copy, halves alternate by rank)."""
    by_surface = defaultdict(list)
    for t in vf.vocab:
        if re.fullmatch(r"[\w']+", surf(t)):
            by_surface[surf(t)].append(t)
    cand = []
    for word, toks in by_surface.items():
        home = max(toks, key=lambda t: (lcounts[t], -copy_of(t)))
        cand.append((word, home, float(vf.field[vf.idx[home]])))
    cand = [c for c in sorted(cand, key=lambda c: -c[2])[:pool] if c[2] > 1e-6]

    halves = ([], [])
    for rank, (word, home, val) in enumerate(cand):
        h = copy_of(home) % 2 if copies >= 2 else rank % 2
        halves[h].append((word, len(word) + 1, val))
    value, ma, mb, ops = knapsack_mitm([(w, v) for _, w, v in halves[0]],
                                       [(w, v) for _, w, v in halves[1]], budget)
    words = [halves[0][i][0] for i in range(len(halves[0])) if ma >> i & 1] + \
            [halves[1][i][0] for i in range(len(halves[1])) if mb >> i & 1]
    chars = sum(len(w) + 1 for w in words)
    report = {"candidates": len(cand), "half_A": len(halves[0]), "half_B": len(halves[1]),
              "agenda": words, "chars": chars, "budget": budget, "value": round(value, 3),
              "subsets_examined": ops, "brute_force_would_examine": 2 ** len(cand)}
    if check and len(cand) <= 22:
        brute = knapsack_brute([(len(w) + 1, v) for w, _, v in cand], budget)
        report["matches_brute_force"] = bool(abs(brute - max(value, 0.0)) < 1e-9)
    return words, report


# ---------------------------------------------------------------------------

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

        # byte graph + word/byte interface (all copies of a word share the
        # byte states of its surface spelling)
        self.states = {}
        rows, cols, edges = [], [], set()
        for wi, w in enumerate(self.vocab):
            path = [self.states.setdefault(st, len(self.states)) for st in word_states(surf(w), k)]
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
        ids = [self.states[s] for s in word_states(surf(word), self.k) if s in self.states]
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
            if surf(w) in SENTENCE_END:
                self.role[i] = self.END
            elif surf(w) in SOFT_PUNCT:
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
            prev = [self.BOS] if surf(t) in SENTENCE_END else (prev + [r])[-2:]
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
        recent = {surf(x) for x in sent_words[-self.window:]}            # surface: any copy counts
        n_words = sum(1 for w in sent_words if surf(w) not in SOFT_PUNCT)
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
            if r < self.END and surf(w) in recent:                       # R4 no loop
                b += HARD / 3
            out[i] = b
        return out


class Trigrams:
    """Word trigram rule R: (w1, w2) -> next word, backing off to the bigram
    (w2 -> next word). The value field only reranks what R allows. Words are
    copy-tagged tokens, so the same surface word in another copy is another
    context."""

    def __init__(self, tokens):
        self.tri, self.bi = defaultdict(Counter), defaultdict(Counter)
        self.starts = Counter(tokens[:1])
        for a, b in zip(tokens, tokens[1:]):
            if surf(a) in SENTENCE_END:
                self.starts[b] += 1
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            self.tri[(a, b)][c] += 1
        for a, b in zip(tokens, tokens[1:]):
            self.bi[a][b] += 1

    def table(self, out):
        if len(out) >= 2 and (out[-2], out[-1]) in self.tri:
            return self.tri[(out[-2], out[-1])]
        return self.bi.get(out[-1])


def curved_schedule(max_order, power):
    """Context lengths 1 + round(j**power): dense at first, then ever wider."""
    ks, j = [], 0
    while True:
        k = 1 + int(round(j ** power))
        if k > max_order:
            break
        if not ks or k > ks[-1]:
            ks.append(k)
        j += 1
    return ks


class CurvedContext(Trigrams):
    """Variable-order context. Keeps Trigrams' tables (tri/bi/starts) for the
    prompt lift and as the fallback, and adds a positional index per length."""

    curved = True

    def __init__(self, tokens, max_order=10, power=1.6, pin=1):
        super().__init__(tokens)
        self.tokens, self.pin = tokens, pin
        self.schedule = curved_schedule(max_order, power)
        self.index = {k: defaultdict(list) for k in self.schedule}
        for k in self.schedule:
            for i in range(k - 1, len(tokens) - 1):      # a next word must exist
                self.index[k][tuple(tokens[i - k + 1:i + 1])].append(i)
        self.last = {"order": 0, "positions": 0, "pinned": False}

    def table(self, history):
        best = None
        for k in self.schedule:
            if k > len(history):
                break
            positions = self.index[k].get(tuple(history[-k:]))
            if not positions:
                break
            best = (k, positions)
            if len(positions) <= self.pin:
                break                                     # pinned to the dataset
        if best is None:
            self.last = {"order": 0, "positions": 0, "pinned": False}
            return Trigrams.table(self, history)
        k, positions = best
        self.last = {"order": k, "positions": len(positions),
                     "pinned": len(positions) <= self.pin}
        return Counter(self.tokens[p + 1] for p in positions)


def _pick(vf, table, beta, rng, mask=None, sent=(), gamma=1.0, favor=(), bonus=0.0):
    words = list(table)
    p = np.array([table[w] for w in words], dtype=np.float64)
    logits = np.log(p / p.sum()) + beta * np.exp(vf.field[[vf.idx[w] for w in words]])
    if mask is not None:
        logits = logits + gamma * mask.bias(list(sent), words)
    if bonus and favor:                                   # the knapsack agenda
        # scaled by the spread of the candidates' logits, so `bonus` means
        # "this fraction of the gap between best and worst candidate"
        spread = max(float(logits.max() - logits.min()), 1.0)
        logits = logits + bonus * spread * np.array([surf(w) in favor for w in words], dtype=np.float64)
    w = np.exp(logits - logits.max())
    return words[rng.choice(len(words), p=w / w.sum())]


def stream_tokens(vf, tg, beta=3.0, rng=None, ctx=None, mask=None, gamma=1.0, trace=False,
                  agenda=(), bonus=0.0, history=None):
    """Endless token stream. Inside a sentence: trigram rule + value bias +
    logic mask. After a sentence end the context resets."""
    rng = rng or np.random.default_rng()
    ctx = list(ctx or [])
    sent = list(ctx)                         # words of the current sentence
    pending = set(agenda) - {surf(x) for x in sent}      # agenda words still to say
    history = list(history or [])                        # every word so far (curved context)
    curved = getattr(tg, "curved", False)
    while True:
        if curved:
            table = tg.table(history) if history else tg.starts
        else:
            table = tg.table(ctx) if ctx else tg.starts
        if not table:
            ctx, sent, history = [], [], []
            continue
        w = _pick(vf, table, beta, rng, mask, sent, gamma, pending, bonus)
        if trace and mask is not None:
            print(f"[mask] {w!r} role={mask.role[vf.idx[w]]}", file=sys.stderr)
        if trace and curved:
            print(f"[context] {w!r} {tg.last}", file=sys.stderr)
        yield w
        history = (history + [w])[-64:]
        pending.discard(surf(w))
        if surf(w) in SENTENCE_END:
            ctx, sent = [], []
            pending = set(agenda)                         # a new sentence, a fresh agenda
        else:
            ctx = (ctx + [w])[-2:]
            sent.append(w)


def stream_text(tokens, first=True, cap=True):
    for t in tokens:
        s = surf(t)
        if re.match(r"[.,!?;:]", s):
            chunk = s
        else:
            chunk = ("" if first else " ") + (s[:1].upper() + s[1:] if cap else s)
        cap, first = s in SENTENCE_END, False
        yield chunk


def detokenize(tokens):
    text = ""
    for t in tokens:
        t = surf(t)
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
    ap.add_argument("--copies", type=int, default=2,
                    help="how many times the vocabulary is duplicated (1 = off)")
    ap.add_argument("--link-seed", type=int, default=0, help="salt for the link -> class partition")
    ap.add_argument("--context", choices=("curved", "trigram"), default="curved",
                    help="curved = push the context up a curved schedule until it is pinned to the dataset")
    ap.add_argument("--max-order", type=int, default=10, help="longest context the curve may reach")
    ap.add_argument("--curve", type=float, default=10.6, help="exponent of the context-length curve")
    ap.add_argument("--pin", type=int, default=3, help="a context with <= this many positions is pinned")
    ap.add_argument("--agenda-pool", type=int, default=32, help="candidate words for the knapsack")
    ap.add_argument("--agenda-chars", type=int, default=64, help="character budget per sentence")
    ap.add_argument("--agenda-bonus", type=float, default=0.9,
                    help="bonus for agenda words, as a fraction of the candidates' logit spread")
    ap.add_argument("--no-agenda", action="store_true", help="skip the knapsack")
    ap.add_argument("--check-solver", action="store_true", help="verify the solver against brute force")
    gen_seed = int(np.random.SeedSequence().entropy % 2**32)

    ap.add_argument("--words", type=int, default=600, help="stop the stream after N words (0 = endless)")
    ap.add_argument("--delay", type=float, default=0.03, help="seconds between words")
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8", errors="ignore").read().lower()
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)

    # duplicate the vocabulary, partition the dataset linkwise
    copies, salt = max(1, args.copies), str(args.link_seed)
    lifted = lift_tokens(tokens, copies, salt)
    lifted_set = set(lifted)
    lcounts = Counter(lifted)
    print(f"[partition] {partition_report(tokens, lifted, copies, salt)}", file=sys.stderr)

    # prompt -> values
    counts = Counter(tokens)                                   # surface counts
    while True:
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
        # a value on a word applies to every copy of it
        values = {tag(w, c): x for w, x in values.items() for c in range(copies)
                  if tag(w, c) in lifted_set}

        vf = ValueField(lifted, values, mix=args.mix, k=args.k)

        # the copies solve knapsack on each other -> sentence agenda
        agenda = []
        if not args.no_agenda:
            agenda, rep_ = solve_agenda(vf, lcounts, copies, args.agenda_pool,
                                        args.agenda_chars, args.check_solver)
            print(f"[agenda] {rep_}", file=sys.stderr)

        # logic mask: learned after all the math, gates the sampler
        mask = None if args.no_mask else LogicMask(vf, lifted, n_classes=args.classes,
                                                   min_len=args.min_len)

        # stream: continue the prompt
        tg = (CurvedContext(lifted, args.max_order, args.curve, args.pin)
              if args.context == "curved" else Trigrams(lifted))
        rng = np.random.default_rng(gen_seed)
        ended = bool(ptoks) and ptoks[-1] in SENTENCE_END
        lp = lift_prompt(ptoks, tg, copies, salt, rng)
        ctx0 = [] if (not ptoks or ended) else lp[-2:]
        toks = stream_tokens(vf, tg, args.beta, rng, ctx=ctx0,
                             mask=mask, gamma=args.gamma, trace=args.trace,
                             agenda=agenda, bonus=args.agenda_bonus, history=lp)

        for n, chunk in enumerate(stream_text(toks, first=False, cap=ended), 1):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            if args.delay:
                time.sleep(args.delay)
            if args.words and n >= args.words:
                break

        print()
