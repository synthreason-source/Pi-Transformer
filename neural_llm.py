#!/usr/bin/env python3
"""neural_llm.py -- MRI neural-circuit simulator + Markov text generator in one file.

Pipeline
  1. MRI volume -> voxel graph (edges streamed to disk, read back via np.memmap)
  2. Corpus co-occurrence (PPMI) -> 3-D coordinates -> each content token is
     placed on a brain node (nearest node in the largest connected component)
  3. A pulse at each token's node is propagated through the circuit
     (leaky threshold-gated dynamics from the simulator). The accumulated
     activity is that token's response vector r_i.
  4. DOT PRODUCTS:  G[i, j] = r_i . r_j   (unit-norm rows, so cosine similarity
     through the brain's wiring). G is extracted once and cached.
  5. PER-WORD MRI CONTEXT: every word keeps its own context read off the scan:
     where its node sits (region), the local tissue intensity, how far its
     pulse spreads (footprint), and its nearest words through the wiring.
     `--explain WORD` prints a detailed circuit intent breakdown.
  6. NOODLE CONTEXT CURVES: noodles (streamlines) are traced through the scan from
     each word's node, following the local structure-tensor orientation.
"""
import argparse
import difflib
import heapq
import json
import os
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds
from scipy.spatial import cKDTree

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", ""
TOKEN_RE = re.compile(r"[a-z0-9']+|[.,!?;:]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORD_RE = re.compile(r"[a-z0-9']")


def get_lemma_root(w):
    if len(w) <= 3:
        return w
    for suffix in ("ing", "ed", "es", "s", "ly", "er", "est", "al", "ic"):
        if suffix == "s" and w.endswith("ss"):
            continue
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def explain_words(values):
    words = []
    for value in values or []:
        words.extend(TOKEN_RE.findall(value.lower()))
    return words


EDGE_DTYPE = np.dtype([("src", "<u4"), ("dst", "<u4"), ("w", "<f4")])


# ===========================================================================
# PART 1 -- NEURAL CIRCUIT: MRI -> graph -> dynamics
# ===========================================================================
def synthesize_volume(shape=(60, 60, 40), seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    body = (x ** 2 / 0.8 + y ** 2 / 0.9 + z ** 2 / 0.8) < 1
    tex = rng.normal(size=shape).astype(np.float32)
    for _ in range(2):
        tex = (tex + np.roll(tex, 1, 0) + np.roll(tex, 1, 1) + np.roll(tex, 1, 2)) / 4
    return np.clip(0.6 + 0.5 * tex, 0, 1) * body


def load_volume(path, downsample=4):
    if path and os.path.exists(path):
        import nibabel as nib
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)[::downsample, ::downsample, ::downsample]
        codes = tuple(nib.aff2axcodes(img.affine))
        return np.clip(data / np.percentile(data[data > 0], 99), 0, 1), codes
    return synthesize_volume(), ("R", "A", "S")


def build_graph(vol, threshold, edges_path):
    mask = vol > threshold
    n = int(mask.sum())
    idx = np.full(vol.shape, -1, np.int64)
    idx[mask] = np.arange(n)
    coords = np.argwhere(mask)
    src, dst, wts = [], [], []
    for ax in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        ia, ib = idx[tuple(a)], idx[tuple(b)]
        ok = (ia >= 0) & (ib >= 0)
        w = 1.0 - np.abs(vol[tuple(a)] - vol[tuple(b)])
        src.append(ia[ok]); dst.append(ib[ok]); wts.append(w[ok])
    s, d, w = np.concatenate(src), np.concatenate(dst), np.concatenate(wts)
    rec = np.empty(2 * len(s), EDGE_DTYPE)
    rec["src"], rec["dst"], rec["w"] = np.r_[s, d], np.r_[d, s], np.r_[w, w]
    rec.tofile(edges_path)
    return coords, n


def load_adjacency(edges_path, n):
    rec = np.memmap(edges_path, dtype=EDGE_DTYPE, mode="r")
    return sp.csr_matrix((np.asarray(rec["w"], np.float32),
                          (np.asarray(rec["src"]), np.asarray(rec["dst"]))), shape=(n, n))


def step_dynamics(A, W, decay, gain, thr):
    fire = np.where(A > thr, A, 0.0)
    return np.minimum(decay * A + gain * (W @ fire), 3.0).astype(np.float32)


def circuit_responses(W, seed_nodes, steps, decay, gain, thr, batch=64, prune=1e-3):
    N, chunks = W.shape[0], []
    for s in range(0, len(seed_nodes), batch):
        sn = seed_nodes[s:s + batch]
        A = np.zeros((N, len(sn)), np.float32)
        A[sn, np.arange(len(sn))] = 1.0
        acc = A.copy()
        for _ in range(steps):
            A = step_dynamics(A, W, decay, gain, thr)
            acc += A
        acc[acc < prune] = 0
        chunks.append(sp.csr_matrix(acc.T))
    R = sp.vstack(chunks).tocsr()
    norm = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel()) + 1e-8
    return sp.diags(1.0 / norm) @ R


# ===========================================================================
# PART 2 -- CORPUS + VOCAB + N-GRAM
# ===========================================================================
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
        self.by_root = {}
        for i in range(4, self.V):
            self.by_root.setdefault(get_lemma_root(self.itos[i]), i)

    def encode(self, toks):
        return [self.stoi.get(t, self.unk) for t in toks]

    def resolve(self, tok):
        if tok in self.stoi:
            return tok
        if not WORD_RE.match(tok) or len(tok) < 3:
            return None
        root = get_lemma_root(tok)
        i = self.by_root.get(root)
        if i is not None:
            return self.itos[i]
        matches = difflib.get_close_matches(tok, self.itos[4:], n=1, cutoff=0.80)
        return matches[0] if matches else None


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
        return lg


# ===========================================================================
# PART 3 -- DOT-PRODUCT EXTRACTION & NOODLE FIELDS
# ===========================================================================
@dataclass
class Circuit:
    ids: np.ndarray
    row_of: np.ndarray
    G: np.ndarray
    Gp: np.ndarray
    ppmi: np.ndarray
    stats: dict
    info: dict = None


def rank01(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(len(x) - 1, 1)


def noodle_field(vol, mask, tube_pct=25.0, sigma_grad=1.0, sigma_int=2.0):
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(vol, sigma_grad)
    g = np.stack(np.gradient(sm), -1)
    J = np.empty(vol.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            J[..., i, j] = J[..., j, i] = gaussian_filter(g[..., i] * g[..., j], sigma_int)
    w, v = np.linalg.eigh(J[mask])
    direction = np.zeros(vol.shape + (3,), np.float32)
    direction[mask] = v[:, :, 0]
    tube = np.zeros(vol.shape, np.float32)
    tube[mask] = (w[:, 1] - w[:, 0]) / (w[:, 2] + 1e-9)
    return dict(dir=direction, tube=tube, mask=mask,
                thr=float(np.percentile(tube[mask], tube_pct)))


def trace_noodles(field, seeds, K=16, L=24, max_angle=70.0, seed=0, paths=False):
    rng = np.random.default_rng(seed)
    n, shape = len(seeds), np.array(field["mask"].shape)
    pos = np.repeat(np.asarray(seeds, np.float64), K, 0) + rng.normal(0, 0.7, (n * K, 3))
    sign = np.tile([1.0, -1.0], (n * K + 1) // 2)[:n * K]
    cos_lim = np.cos(np.deg2rad(max_angle))
    alive, prev = np.ones(n * K, bool), None
    surv = [alive.reshape(n, K).mean(1)]
    hist, ahist = [pos.copy()], [alive.copy()]
    for _ in range(L):
        idt = tuple(np.clip(np.rint(pos).astype(int), 0, shape - 1).T)
        d = field["dir"][idt].astype(np.float64) * sign[:, None]
        ok = field["mask"][idt] & (field["tube"][idt] >= field["thr"])
        if prev is not None:
            dot = (d * prev).sum(1)
            d[dot < 0] *= -1
            ok &= np.abs(dot) >= cos_lim
        alive &= ok
        pos, prev = pos + d, d
        surv.append(alive.reshape(n, K).mean(1))
        hist.append(pos.copy()); ahist.append(alive.copy())
    S = np.array(surv).T.astype(np.float32)
    if paths:
        return S, dict(pos=np.array(hist).reshape(L + 1, n, K, 3),
                       alive=np.array(ahist).reshape(L + 1, n, K))
    return S


_SPARK = " ▁▂▃▄▅▆▇█"


def sparkline(v):
    return "".join(_SPARK[int(round(min(max(x, 0), 1) * 8))] for x in v)


_AXIS_NAMES = {"R": ("right", "left"), "L": ("left", "right"),
               "A": ("anterior", "posterior"), "P": ("posterior", "anterior"),
               "S": ("superior", "inferior"), "I": ("inferior", "superior")}
_AXIS_ORDER = {"right": 0, "left": 0, "anterior": 1, "posterior": 1, "superior": 2, "inferior": 2}


def region_label(xyz, mid, codes):
    parts = [_AXIS_NAMES[c][0] if xyz[k] > mid[k] else _AXIS_NAMES[c][1]
             for k, c in enumerate(codes)]
    return "-".join(sorted(parts, key=_AXIS_ORDER.get))


def word_context_table(R, seeds, coords_all, vals, keep_cc, codes):
    mid = (keep_cc.min(0) + keep_cc.max(0)) / 2
    lo, hi = np.percentile(vals[vals > 0], [33, 67])
    seed_int = vals[seeds]
    band = np.where(seed_int < lo, "low intensity",
                    np.where(seed_int < hi, "mid intensity", "high intensity"))
    mass = np.asarray(R.sum(axis=1)).ravel() + 1e-8
    return dict(
        xyz=coords_all[seeds].astype(np.int32),
        seed_int=seed_int.astype(np.float32),
        band=band.astype("U32"),
        region=np.array([region_label(coords_all[n], mid, codes) for n in seeds], dtype="U40"),
        foot_n=np.diff(R.indptr).astype(np.int32),
        foot_cent=((R @ coords_all.astype(np.float64)) / mass[:, None]).astype(np.float32),
        foot_int=((R @ vals.astype(np.float64)) / mass).astype(np.float32))


def word_context(circ, vocab, word, top=5):
    i = vocab.stoi.get(word, -1)
    r = circ.row_of[i] if i >= 0 else -1
    if r < 0 or not circ.info:
        return None
    f = circ.info
    nb = np.argsort(-circ.G[r])[:top]
    curve = f["curve"][r] if "curve" in f else None
    below = np.flatnonzero(curve < 0.5) if curve is not None else []
    return dict(word=word, curve=curve,
                half_life=(int(below[0]) if len(below) else len(curve) - 1) if curve is not None else None,
                noodle_len=float(curve[1:].sum()) if curve is not None else None,
                word_=word, xyz=tuple(int(v) for v in f["xyz"][r]),
                intensity=float(f["seed_int"][r]), band=str(f["band"][r]),
                region=str(f["region"][r]), footprint=int(f["foot_n"][r]),
                centroid=tuple(round(float(v), 1) for v in f["foot_cent"][r]),
                footprint_intensity=float(f["foot_int"][r]),
                neighbours=[(vocab.itos[circ.ids[j]], float(circ.G[r, j])) for j in nb if circ.G[r, j] > 0])


def build_circuit(sents, vocab, a):
    skip = set(np.argsort(-vocab.counts)[:a.skip_top].tolist())
    cand = [i for i in range(4, vocab.V)
            if i not in skip and WORD_RE.match(vocab.itos[i])
            and vocab.counts[i] >= a.circuit_min_count]
    ids = np.array(sorted(cand, key=lambda i: -vocab.counts[i])[:a.circuit_vocab])
    if len(ids) < 8:
        return None
    Vc = len(ids)
    row_of = np.full(vocab.V, -1, np.int64)
    row_of[ids] = np.arange(Vc)

    r, c = [], []
    for si, s in enumerate(sents):
        for t in set(vocab.encode(s)):
            if row_of[t] >= 0:
                r.append(si); c.append(row_of[t])
    X = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(sents), Vc))
    C = (X.T @ X).toarray().astype(np.float32)
    np.fill_diagonal(C, 0)
    rs = C.sum(1, keepdims=True)
    tot = np.float32(C.sum(dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C * tot / (rs * rs.T))
    del C
    P = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0).astype(np.float32)
    del pmi
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    Gp = (Pn @ Pn.T).astype(np.float32)
    np.fill_diagonal(Gp, 0)
    del Pn

    by_root = {}
    for i in ids:
        by_root.setdefault(get_lemma_root(vocab.itos[i]), row_of[i])
    for i in range(4, vocab.V):
        if row_of[i] < 0 and i not in skip and WORD_RE.match(vocab.itos[i]):
            rr = by_root.get(get_lemma_root(vocab.itos[i]))
            if rr is not None:
                row_of[i] = rr

    vol, codes = load_volume(a.mri, a.downsample)
    coords, n = build_graph(vol, a.voxel_threshold, a.edges_path)
    vals = vol[tuple(coords.T)]
    W = load_adjacency(a.edges_path, n)
    ncomp, lab = connected_components(W, directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    wbar = float(W.data.mean())
    gain = a.gain if a.gain > 0 else 1.5 * a.fire_threshold / wbar

    Ps = sp.csr_matrix(P).astype(np.float64)
    U, S_vals, _ = svds(Ps, k=3, v0=np.random.default_rng(0).normal(size=Vc))
    pos = np.stack([rank01(v) for v in (U * S_vals).T], axis=1)
    cc = coords[keep].astype(np.float64)
    target = cc.min(0) + pos * (cc.max(0) - cc.min(0))
    seeds = keep[cKDTree(cc).query(target)[1]]

    R = circuit_responses(W, seeds, a.steps, a.decay, gain, a.fire_threshold)
    G = (R @ R.T).toarray().astype(np.float32)
    np.fill_diagonal(G, 0)
    info = word_context_table(R, seeds, coords, vals, cc, codes)
    field = noodle_field(vol, vol > a.voxel_threshold, getattr(a, "noodle_tube_pct", 25.0))
    info["curve"] = trace_noodles(field, coords[seeds], getattr(a, "noodle_count", 16),
                                   getattr(a, "noodle_steps", 24), getattr(a, "noodle_angle", 70.0))
    stats = dict(nodes=n, tokens=Vc)
    return Circuit(ids, row_of, G, Gp, P, stats, info)


def expand_bias(circ, vocab, s):
    pos = s[s > 0]
    if not pos.size:
        return None
    x = np.clip(s / np.percentile(pos, 99), 0, 1)
    full = np.zeros(vocab.V, np.float32)
    has = circ.row_of >= 0
    full[has] = x[circ.row_of[has]]
    return full


def prompt_bias(circ, vocab, toks, source, intent_boost=0.3, top_n=5):
    if circ is None or source == "none":
        return None
    M = circ.G if source == "circuit" else circ.Gp
    rows = [circ.row_of[vocab.stoi[t]] for t in toks
            if t in vocab.stoi and circ.row_of[vocab.stoi[t]] >= 0]
    if not rows:
        return None
    s = M[:, rows].sum(axis=1)
    if source == "circuit" and intent_boost > 0:
        intent_vector = np.zeros_like(s)
        for t in toks:
            i = vocab.stoi.get(t, -1)
            r = circ.row_of[i] if i >= 0 else -1
            if r >= 0:
                nb_indices = np.argsort(-circ.G[r])[:top_n]
                for j in nb_indices:
                    w_sim = float(circ.G[r, j])
                    if w_sim > 0:
                        intent_vector[j] += w_sim
        if intent_vector.any():
            s = s + intent_boost * intent_vector
    return expand_bias(circ, vocab, s)


# ===========================================================================
# PART 4 -- DECODING & CLI
# ===========================================================================
@dataclass
class Gen:
    max_len: int = 40
    temp: float = 0.9
    top_k: int = 40
    dots: str = "circuit"
    beta: float = 6.0
    intent_boost: float = 0.3


def generate(model, vocab, toks, gen, circ=None):
    ctx0 = [vocab.bos] + vocab.encode(toks)
    bp = prompt_bias(circ, vocab, toks, gen.dots, intent_boost=gen.intent_boost)
    
    def step(seq):
        lg = model.logits(seq)
        if bp is not None:
            lg += gen.beta * bp
        return lg

    ctx, out = list(ctx0), []
    for _ in range(gen.max_len):
        p = np.exp(step(ctx) / gen.temp)
        nxt = int(np.random.choice(len(p), p=p / p.sum()))
        if nxt == vocab.eos:
            break
        out.append(nxt); ctx.append(nxt)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="training text")
    
    g = ap.add_argument_group("language model")
    g.add_argument("--order", type=int, default=2)
    g.add_argument("--min-count", type=int, default=1)
    g.add_argument("--base-clip", type=float, default=100.0)
    
    g = ap.add_argument_group("circuit")
    g.add_argument("--mri", default="real_brain_mri_t1.nii.gz")
    g.add_argument("--downsample", type=int, default=4)
    g.add_argument("--voxel-threshold", type=float, default=0.2)
    g.add_argument("--edges-path", default="connectivity_edges.bin")
    g.add_argument("--steps", type=int, default=4)
    g.add_argument("--decay", type=float, default=0.6)
    g.add_argument("--gain", type=float, default=0.0)
    g.add_argument("--fire-threshold", type=float, default=0.55)
    g.add_argument("--circuit-vocab", type=int, default=8000)
    g.add_argument("--circuit-min-count", type=int, default=2)
    g.add_argument("--skip-top", type=int, default=20)
    g.add_argument("--noodle-steps", type=int, default=24)
    g.add_argument("--noodle-count", type=int, default=16)
    g.add_argument("--noodle-angle", type=float, default=70.0)
    g.add_argument("--noodle-tube-pct", type=float, default=25.0)

    g = ap.add_argument_group("decoding")
    g.add_argument("--dots", choices=["none", "circuit", "ppmi"], default="circuit")
    g.add_argument("--beta", type=float, default=6.0)
    g.add_argument("--intent-boost", type=float, default=0.3)
    
    ap.add_argument("--explain", nargs="+", metavar="WORD", help="print each word's MRI circuit intent & context")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    sents = load_sentences(a.path)
    vocab = Vocab(sents, a.min_count)
    model = NGram(vocab, a.order, base_clip=a.base_clip)
    model.fit([[vocab.bos] + vocab.encode(s) + [vocab.eos] for s in sents])

    need_circuit = a.dots != "none" or a.explain or a.prompt
    circ = build_circuit(sents, vocab, a) if need_circuit else None

    def show_circuit_intent_explanation(c):
        print(f"  Word: '{c['word']}'")
        print(f"    - Brain Node Coordinates: {c['xyz']}")
        print(f"    - Anatomical Region / Octant: {c['region']}")
        print(f"    - Local Tissue Band: {c['band']} (intensity {c['intensity']:.2f})")
        print(f"    - Pulse Footprint: {c['footprint']} nodes (mean intensity {c['footprint_intensity']:.2f})")
        print(f"    - Nearest Circuit Wiring & Intent Neighbors:")
        if c["neighbours"]:
            for w, sim in c["neighbours"]:
                print(f"        -> {w:12s} (wiring similarity {sim:.3f})")
        else:
            print("        -> None found")
        if c["curve"] is not None:
            print(f"    - Noodle Context Curve |{sparkline(c['curve'][::2])}| (half-life {c['half_life']} steps)")
        print()

    if a.explain:
        explain_tokens = explain_words(a.explain)
        print("[explain] Detailed MRI Circuit Intent & Context Analysis:")
        for word in explain_tokens:
            c = word_context(circ, vocab, word, top=50) if circ else None
            if c:
                show_circuit_intent_explanation(c)
            else:
                print(f"  {word:12s} -> No circuit vector found (function word or out of vocabulary)\n")

    if a.prompt:
        toks = TOKEN_RE.findall(a.prompt.lower())
        gen = Gen(dots=a.dots, beta=a.beta, intent_boost=a.intent_boost)
        out = generate(model, vocab, toks, gen, circ)
        print("Completion:", detok(toks + [vocab.itos[i] for i in out]))


if __name__ == "__main__":
    main()
