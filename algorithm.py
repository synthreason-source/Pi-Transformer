#!/usr/bin/env python3
"""Dynamic frequency-decay text generator.

Extends the supplied graph/byte/context/mask/copy/knapsack generator with
one important change: during generation, every emitted surface word receives
a growing frequency penalty. The penalty is applied at each sampling step,
so the probability landscape changes continuously as the stream moves on.

The generated-count penalty is reset at sentence boundaries, while the
optional global mode can keep counts for the whole generation.
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

import nltk
from nltk.corpus import wordnet

SEP = 256
SENTENCE_END = {".", "!", "?"}
SOFT_PUNCT = {",", ";", ":"}
HARD = -30.0
COPY_MARK = "#"
ANCHOR_SYNSETS = [wordnet.synset("entity.n.01")]


def nltk_word_value(word, anchor_synsets, default=1e-6):
    synsets = wordnet.synsets(word)
    if not synsets or not anchor_synsets:
        return default
    best = 0.0
    for s in synsets:
        for a in anchor_synsets:
            try:
                sim = s.path_similarity(a)
            except Exception:
                sim = 0.0
            if sim is None:
                sim = 0.0
            best = max(best, sim)
    return max(best, 1e-6)


def surf(token):
    return token.partition(COPY_MARK)[0]


def copy_of(token):
    _, _, c = token.partition(COPY_MARK)
    return int(c) if c else 0


def tag(word, copy):
    return word if copy == 0 else f"{word}{COPY_MARK}{copy}"


def shift_token(token, offset, copies):
    return tag(surf(token), (copy_of(token) + offset) % copies)


def link_class(first, second, copies, salt=""):
    digest = hashlib.blake2b(f"{salt}\x00{first}\x00{second}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % copies


def lift_tokens(tokens, copies, salt=""):
    out, copy, previous = [], 0, None
    for t in tokens:
        if previous is not None:
            copy = (copy + link_class(previous, t, copies, salt)) % copies
        out.append(tag(t, copy))
        previous = t
    return out


def lift_prompt(ptoks, tg, copies, salt, rng):
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
    return {"copies": copies, "vocab": len(set(lifted)), "distinct_links_per_class": dict(sorted(distinct.items())), "occurrences_per_class": dict(sorted(occ.items())), "projects_back_to_dataset": [surf(t) for t in lifted] == tokens}


def subset_table(items):
    W = np.zeros(1, dtype=np.int64)
    V = np.zeros(1)
    M = np.zeros(1, dtype=np.int64)
    for i, (w, v) in enumerate(items):
        W = np.concatenate([W, W + w])
        V = np.concatenate([V, V + v])
        M = np.concatenate([M, M | (1 << i)])
    return W, V, M


def knapsack_mitm(set_a, set_b, budget):
    WA, VA, MA = subset_table(set_a)
    WB, VB, MB = subset_table(set_b)
    oa = np.argsort(WA, kind="stable")
    ob = np.argsort(-WB, kind="stable")
    WA, VA, MA = WA[oa], VA[oa], MA[oa]
    WB, VB, MB = WB[ob], VB[ob], MB[ob]
    nb = len(WB)
    sufv = np.empty(nb)
    sufi = np.empty(nb, dtype=np.int64)
    best_v, best_i = -1.0, 0
    for j in range(nb - 1, -1, -1):
        if VB[j] > best_v:
            best_v, best_i = VB[j], j
        sufv[j], sufi[j] = best_v, best_i
    best = (-1.0, 0, 0)
    j = 0
    for a in range(len(WA)):
        room = budget - WA[a]
        if room < 0:
            break
        while j < nb and WB[j] > room:
            j += 1
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
    value, ma, mb, ops = knapsack_mitm([(w, v) for _, w, v in halves[0]], [(w, v) for _, w, v in halves[1]], budget)
    words = [halves[0][i][0] for i in range(len(halves[0])) if ma >> i & 1] + [halves[1][i][0] for i in range(len(halves[1])) if mb >> i & 1]
    report = {"candidates": len(cand), "half_A": len(halves[0]), "half_B": len(halves[1]), "agenda": words, "chars": sum(len(w) + 1 for w in words), "budget": budget, "value": round(value, 3), "subsets_examined": ops, "brute_force_would_examine": 2 ** len(cand)}
    if check and len(cand) <= 22:
        brute = knapsack_brute([(len(w) + 1, v) for w, _, v in cand], budget)
        report["matches_brute_force"] = bool(abs(brute - max(value, 0.0)) < 1e-9)
    return words, report


def normalize_rows(M):
    s = np.asarray(M.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return sp.diags(1.0 / s) @ M if sp.issparse(M) else M / s[:, None]


def word_states(word, k=3):
    seq = (SEP,) + tuple(word.encode("utf-8")) + (SEP,)
    seq = seq + (SEP,) * max(0, k - len(seq))
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


class ValueField:
    def __init__(self, tokens, values, top=5, hops=3, decay=0.5, mix=0.5, k=3):
        self.k = k
        self.vocab = sorted(set(tokens))
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(self.vocab)
        C = np.zeros((n, n))
        for a, b in zip(tokens, tokens[1:]):
            C[self.idx[a], self.idx[b]] += 1.0
        X = np.hstack([normalize_rows(C), normalize_rows(C.T)])
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self.X = X
        S = X @ X.T
        np.fill_diagonal(S, 0.0)
        K = np.zeros_like(S)
        for i in range(n):
            j = np.argsort(S[i])[-top:]
            K[i, j] = S[i, j]
        K = np.maximum(K, K.T)
        d = np.sqrt(np.maximum(K.sum(axis=1), 1e-12))
        Sn = K / np.outer(d, d)
        self.states = {}
        rows, cols, edges = [], [], set()
        for wi, w in enumerate(self.vocab):
            path = [self.states.setdefault(st, len(self.states)) for st in word_states(surf(w), k)]
            rows += [wi] * len(path)
            cols += path
            for a, b in zip(path, path[1:]):
                edges.add((a, b)); edges.add((b, a))
        ns = len(self.states)
        B = normalize_rows(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, ns)))
        er, ec = zip(*edges) if edges else ([], [])
        A = sp.csr_matrix((np.ones(len(er)), (er, ec)), shape=(ns, ns))
        A.data[:] = 1.0
        dd = 1.0 / np.sqrt(np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-12))
        Sb = sp.diags(dd) @ A @ sp.diags(dd)
        v0 = np.zeros(n)
        for w, x in values.items():
            if w in self.idx:
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
        ids = [self.states[s] for s in word_states(surf(word), self.k) if s in self.states]
        return float(np.mean(self.state_field[ids])) if ids else 0.0


class LogicMask:
    def __init__(self, vf, tokens, n_classes=12, min_len=4, window=6, floor=1e-3, seed=0, iters=15):
        self.vf, self.min_len, self.window, self.floor = vf, min_len, window, floor
        rng = np.random.default_rng(seed)
        n = len(vf.vocab)
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
                    v = Z[m].mean(axis=0); cent[c] = v / max(np.linalg.norm(v), 1e-12)
        self.END, self.PUNCT, self.BOS = k, k + 1, k + 2
        self.R = k + 2
        self.role = lab.copy()
        for w, i in vf.idx.items():
            if surf(w) in SENTENCE_END: self.role[i] = self.END
            elif surf(w) in SOFT_PUNCT: self.role[i] = self.PUNCT
        self.bi = np.zeros((self.BOS + 1, self.R))
        self.tri = defaultdict(lambda: np.zeros(self.R))
        prev = [self.BOS]
        for t in tokens:
            r = self.role[vf.idx[t]]
            self.bi[prev[-1], r] += 1
            if len(prev) >= 2: self.tri[(prev[-2], prev[-1])][r] += 1
            prev = [self.BOS] if surf(t) in SENTENCE_END else (prev + [r])[-2:]
        rs = self.bi.sum(axis=1, keepdims=True)
        self.p_end = np.where(rs[:, 0] > 0, self.bi[:, self.END] / np.maximum(rs[:, 0], 1), 0.0)

    def roles_of(self, words):
        return [self.role[self.vf.idx[w]] for w in words if w in self.vf.idx]

    def bias(self, sent_words, cand_words):
        roles = [self.BOS] + self.roles_of(sent_words)
        last = roles[-1]
        dist = self.tri.get((roles[-2], roles[-1])) if len(roles) >= 2 else None
        if dist is None or dist.sum() == 0: dist = self.bi[last]
        dist = dist / max(dist.sum(), 1e-12)
        recent = {surf(x) for x in sent_words[-self.window:]}
        n_words = sum(1 for w in sent_words if surf(w) not in SOFT_PUNCT)
        out = np.zeros(len(cand_words))
        for i, w in enumerate(cand_words):
            r = self.role[self.vf.idx[w]]
            b = math.log(dist[r] + self.floor)
            if r == self.END: b += math.log(self.p_end[last] + self.floor)
            if n_words < self.min_len or last in (self.BOS, self.PUNCT): b += HARD
            if r in (self.END, self.PUNCT) and last in (self.BOS, self.END, self.PUNCT): b += HARD
            if r < self.END and surf(w) in recent: b += HARD / 3
            out[i] = b
        return out


class Trigrams:
    def __init__(self, tokens):
        self.tri, self.bi = defaultdict(Counter), defaultdict(Counter)
        self.starts = Counter(tokens[:1])
        for a, b in zip(tokens, tokens[1:]):
            if surf(a) in SENTENCE_END: self.starts[b] += 1
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]): self.tri[(a, b)][c] += 1
        for a, b in zip(tokens, tokens[1:]): self.bi[a][b] += 1

    def table(self, out):
        if len(out) >= 2 and (out[-2], out[-1]) in self.tri: return self.tri[(out[-2], out[-1])]
        return self.bi.get(out[-1])


def curved_schedule(max_order, power):
    ks, j = [], 0
    while True:
        k = 1 + int(round(j ** power))
        if k > max_order: break
        if not ks or k > ks[-1]: ks.append(k)
        j += 1
    return ks


class CurvedContext(Trigrams):
    curved = True
    def __init__(self, tokens, max_order=10, power=1.6, pin=1):
        super().__init__(tokens)
        self.tokens, self.pin = tokens, pin
        self.schedule = curved_schedule(max_order, power)
        self.index = {k: defaultdict(list) for k in self.schedule}
        for k in self.schedule:
            for i in range(k - 1, len(tokens) - 1): self.index[k][tuple(tokens[i-k+1:i+1])].append(i)
        self.last = {"order": 0, "positions": 0, "pinned": False}

    def table(self, history):
        best = None
        for k in self.schedule:
            if k > len(history): break
            positions = self.index[k].get(tuple(history[-k:]))
            if not positions: break
            best = (k, positions)
            if len(positions) <= self.pin: break
        if best is None:
            self.last = {"order": 0, "positions": 0, "pinned": False}
            return Trigrams.table(self, history)
        k, positions = best
        self.last = {"order": k, "positions": len(positions), "pinned": len(positions) <= self.pin}
        return Counter(self.tokens[p + 1] for p in positions)


def _pick(vf, table, beta, rng, mask=None, sent=(), gamma=1.0, favor=(), bonus=0.0, generated_counts=None, freq_strength=0.0, decay_mode="linear"):
    words = list(table)
    p = np.array([table[w] for w in words], dtype=np.float64)
    logits = np.log(p / p.sum()) + beta * np.exp(vf.field[[vf.idx[w] for w in words]])
    if mask is not None: logits += gamma * mask.bias(list(sent), words)
    if generated_counts is not None and freq_strength:
        counts = np.array([generated_counts[surf(w)] for w in words], dtype=np.float64)
        if decay_mode == "sqrt":
            penalty = np.sqrt(counts)
        elif decay_mode == "log":
            penalty = np.log1p(counts)
        elif decay_mode == "quadratic":
            penalty = counts * counts
        else:
            penalty = counts
        logits -= freq_strength * penalty
    if bonus and favor:
        spread = max(float(logits.max() - logits.min()), 1.0)
        logits += bonus * spread * np.array([surf(w) in favor for w in words], dtype=np.float64)
    probs = np.exp(logits - logits.max())
    return words[rng.choice(len(words), p=probs / probs.sum())]


def stream_tokens(vf, tg, beta=3.0, rng=None, ctx=None, mask=None, gamma=1.0, trace=False, agenda=(), bonus=0.0, history=None, freq_strength=0.0, freq_reset="sentence", decay_mode="linear"):
    rng = rng or np.random.default_rng()
    ctx, sent, history = list(ctx or []), list(ctx or []), list(history or [])
    pending = set(agenda) - {surf(x) for x in sent}
    generated_counts = Counter()
    curved = getattr(tg, "curved", False)
    step = 0
    while True:
        table = tg.table(history) if curved and history else (tg.starts if curved else (tg.table(ctx) if ctx else tg.starts))
        if not table:
            ctx, sent, history = [], [], []
            if freq_reset == "sentence": generated_counts.clear()
            continue
        w = _pick(vf, table, beta, rng, mask, sent, gamma, pending, bonus, generated_counts, freq_strength, decay_mode)
        step += 1
        if trace:
            print(f"[step {step}] {w!r} surface={surf(w)!r} generated_count={generated_counts[surf(w)]}", file=sys.stderr)
            if mask is not None: print(f"[mask] role={mask.role[vf.idx[w]]}", file=sys.stderr)
            if curved: print(f"[context] {tg.last}", file=sys.stderr)
        yield w
        generated_counts[surf(w)] += 1
        pending.discard(surf(w))
        history = (history + [w])[-64:]
        if surf(w) in SENTENCE_END:
            ctx, sent = [], []
            pending = set(agenda)
            if freq_reset == "sentence": generated_counts.clear()
        else:
            ctx = (ctx + [w])[-2:]
            sent.append(w)


def stream_text(tokens, first=True, cap=True):
    for t in tokens:
        s = surf(t)
        if re.match(r"[.,!?;:]", s): chunk = s
        else: chunk = ("" if first else " ") + (s[:1].upper() + s[1:] if cap else s)
        cap, first = s in SENTENCE_END, False
        yield chunk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--value", type=float, default=3.0)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--link-seed", type=int, default=0)
    ap.add_argument("--context", choices=("curved", "trigram"), default="curved")
    ap.add_argument("--max-order", type=int, default=10)
    ap.add_argument("--curve", type=float, default=1.6)
    ap.add_argument("--pin", type=int, default=3)
    ap.add_argument("--agenda-pool", type=int, default=32)
    ap.add_argument("--agenda-chars", type=int, default=64)
    ap.add_argument("--agenda-bonus", type=float, default=0.9)
    ap.add_argument("--no-agenda", action="store_true")
    ap.add_argument("--check-solver", action="store_true")
    ap.add_argument("--semantic-weight", type=float, default=1.0)
    ap.add_argument("--words", type=int, default=600)
    ap.add_argument("--delay", type=float, default=0.03)
    ap.add_argument("--freq-strength", type=float, default=0.75, help="dynamic probability reduction per generated occurrence")
    ap.add_argument("--freq-reset", choices=("sentence", "generation"), default="sentence", help="reset generated frequency counts at sentence boundaries or keep them for the whole generation")
    ap.add_argument("--freq-decay", choices=("linear", "sqrt", "log", "quadratic"), default="linear", help="shape of the dynamic frequency penalty")
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8", errors="ignore").read().lower()
    tokens = re.findall(r"[\w']+|[.,!?;:]", text)
    copies, salt = max(1, args.copies), str(args.link_seed)
    lifted = lift_tokens(tokens, copies, salt)
    lifted_set = set(lifted)
    lcounts = Counter(lifted)
    print(f"[partition] {partition_report(tokens, lifted, copies, salt)}", file=sys.stderr)
    counts = Counter(tokens)

    while True:
        ptoks = re.findall(r"[\w']+|[.,!?;:]", input("USER: ").lower())
        is_word = lambda t: re.match(r"[\w']+", t)
        known = [w for w in dict.fromkeys(ptoks) if is_word(w) and w in counts]
        unknown = [w for w in dict.fromkeys(ptoks) if is_word(w) and w not in counts]
        if unknown: print(f"[warn] not in the data, no value assigned: {unknown}", file=sys.stderr)
        if not known: raise SystemExit("no prompt word appears in the data")

        rarity = {w: max(math.log(len(tokens) / counts[w]), 1e-9) for w in known}
        max_rarity = max(rarity.values()) or 1.0
        rarity_scaled = {w: rarity[w] / max_rarity for w in known}
        semantic = {w: nltk_word_value(w, ANCHOR_SYNSETS) for w in known}
        max_sem = max(semantic.values()) or 1.0
        semantic_scaled = {w: semantic[w] / max_sem for w in known}
        w_sem = float(args.semantic_weight)
        w_rar = 1.0 - w_sem
        values = {w: args.value * (w_sem * semantic_scaled[w] + w_rar * rarity_scaled[w]) for w in known}
        values = {tag(w, c): x for w, x in values.items() for c in range(copies) if tag(w, c) in lifted_set}
        vf = ValueField(lifted, values, mix=args.mix, k=args.k)

        agenda = []
        if not args.no_agenda:
            agenda, rep_ = solve_agenda(vf, lcounts, copies, args.agenda_pool, args.agenda_chars, args.check_solver)
            print(f"[agenda] {rep_}", file=sys.stderr)

        mask = None if args.no_mask else LogicMask(vf, lifted, n_classes=args.classes, min_len=args.min_len)
        tg = CurvedContext(lifted, args.max_order, args.curve, args.pin) if args.context == "curved" else Trigrams(lifted)
        gen_seed = int(np.random.SeedSequence().entropy % 2**32)
        rng = np.random.default_rng(gen_seed)
        ended = bool(ptoks) and ptoks[-1] in SENTENCE_END
        lp = lift_prompt(ptoks, tg, copies, salt, rng)
        ctx0 = [] if (not ptoks or ended) else lp[-2:]
        toks = stream_tokens(vf, tg, args.beta, rng, ctx=ctx0, mask=mask, gamma=args.gamma, trace=args.trace, agenda=agenda, bonus=args.agenda_bonus, history=lp, freq_strength=args.freq_strength, freq_reset=args.freq_reset, decay_mode=args.freq_decay)
        for n, chunk in enumerate(stream_text(toks, first=False, cap=ended), 1):
            sys.stdout.write(chunk); sys.stdout.flush()
            if args.delay: time.sleep(args.delay)
            if args.words and n >= args.words: break
        print()
