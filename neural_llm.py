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
     `--explain WORD` prints it; `--trace` prints it for each generated word.
  6. NOODLE CONTEXT CURVES: noodles (streamlines) are traced through the scan from
     each word's node, following the local structure-tensor orientation. A
     noodle ends when it leaves tissue, turns too sharply, or hits a region with
     no tube-like structure. The share of a word's noodles still alive after k
     steps is that word's context curve S_w(k): how long its context carries
     forward. NOTE: this is a T1 scan (no diffusion data), so the noodles follow
     image structure and are NOT anatomical fibre tracts.
  7. Decoding: the prompt's content words act as a self-stimulating seed
     (bias = beta * level(t) * normalise(G @ prompt), re-pulsed every few steps).
     With `--fb`, every word generated so far also injects its OWN response
     vector into the circuit state, weighted by that word's noodle curve at its
     age (`--curves exp` swaps in a plain exponential decay), so the context
     evolves word by word instead of being fixed by the prompt.

Word recognition
  * Up to --circuit-vocab content words get their own brain vector.
  * Inflections of those words (rivers/river, flowing/flow) are aliased onto
    their sibling's vector via a crude suffix stemmer.
  * Unseen prompt words are matched to a known word by shared stem, then by
    close spelling (difflib), before falling back to <unk>.

Other generation controls: entropy-targeted temperature, direct leaky anchor,
beam search (heapq + linked-list paths + exact state dedup), min-len,
feasibility-based length floor, beam calibration, and an ablation `--diagnose`
that compares: no bias / direct anchor / PPMI dots (no brain) / circuit dots.

  python neural_llm.py data.txt --prompt "the old river" --dots circuit --beta 6
  python neural_llm.py data.txt --prompt "the old river" --diagnose --beta 6
  python neural_llm.py data.txt --explain river ocean --plot-curves curves.png
  python neural_llm.py data.txt --prompt "the old river" --dots circuit --fb 0.5 --trace
  python neural_llm.py data.txt --sim-report --calibrate
  python neural_llm.py data.txt --interactive --dots circuit --target-entropy 1.5
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

_SUFFIXES = ("ing", "ed", "es", "s", "ly")


def stem(w):
    """Crude suffix stripper: enough to link rivers/river, flowing/flow."""
    for suf in _SUFFIXES:
        if suf == "s" and w.endswith("ss"):      # glass != glas
            continue
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def explain_words(values):
    """Expand one or more command-line values into individual explainable words."""
    words = []
    for value in values or []:
        words.extend(TOKEN_RE.findall(value.lower()))
    return words


EDGE_DTYPE = np.dtype([("src", "<u4"), ("dst", "<u4"), ("w", "<f4")])


# ===========================================================================
# PART 1 -- NEURAL CIRCUIT: MRI -> graph -> dynamics
# ===========================================================================
def synthesize_volume(shape=(60, 60, 40), seed=0):
    """Stand-in brain (ellipsoid + smooth texture) when no MRI file is present."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    body = (x ** 2 / 0.8 + y ** 2 / 0.9 + z ** 2 / 0.8) < 1
    tex = rng.normal(size=shape).astype(np.float32)
    for _ in range(2):
        tex = (tex + np.roll(tex, 1, 0) + np.roll(tex, 1, 1) + np.roll(tex, 1, 2)) / 4
    return np.clip(0.6 + 0.5 * tex, 0, 1) * body


def load_volume(path, downsample=4):
    """Real NIfTI scan, downsampled and robustly normalised (99th percentile).
    Returns (volume, axis_codes); codes come from the scan's affine, e.g. ('R','A','S')."""
    if path and os.path.exists(path):
        import nibabel as nib
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)[::downsample, ::downsample, ::downsample]
        codes = tuple(nib.aff2axcodes(img.affine))
        return np.clip(data / np.percentile(data[data > 0], 99), 0, 1), codes
    print(f"[circuit] MRI '{path}' not found; using a synthetic volume (region labels nominal)")
    return synthesize_volume(), ("R", "A", "S")


def build_graph(vol, threshold, edges_path):
    """Vectorised: tissue voxels -> nodes; 6-neighbour edges written to disk."""
    mask = vol > threshold
    n = int(mask.sum())
    idx = np.full(vol.shape, -1, np.int64)
    idx[mask] = np.arange(n)
    coords = np.argwhere(mask)                    # same C-order as idx assignment
    src, dst, wts = [], [], []
    for ax in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        ia, ib = idx[tuple(a)], idx[tuple(b)]
        ok = (ia >= 0) & (ib >= 0)
        w = 1.0 - np.abs(vol[tuple(a)] - vol[tuple(b)])
        src.append(ia[ok]); dst.append(ib[ok]); wts.append(w[ok])
    s, d, w = np.concatenate(src), np.concatenate(dst), np.concatenate(wts)
    rec = np.empty(2 * len(s), EDGE_DTYPE)        # both directions
    rec["src"], rec["dst"], rec["w"] = np.r_[s, d], np.r_[d, s], np.r_[w, w]
    rec.tofile(edges_path)
    return coords, n


def load_adjacency(edges_path, n):
    rec = np.memmap(edges_path, dtype=EDGE_DTYPE, mode="r")   # disk-backed, lazy
    return sp.csr_matrix((np.asarray(rec["w"], np.float32),
                          (np.asarray(rec["src"]), np.asarray(rec["dst"]))), shape=(n, n))


def step_dynamics(A, W, decay, gain, thr):
    """One step of the simulator rule: nodes above `thr` fire and push
    gain * w * activity to neighbours; everything leaks by `decay`."""
    fire = np.where(A > thr, A, 0.0)
    return np.minimum(decay * A + gain * (W @ fire), 3.0).astype(np.float32)


def simulate_activity(W, seed, steps=60, decay=0.6, gain=0.4, thr=0.55,
                      stim_interval=0, amp=1.0):
    """Single pulse (stim_interval=0) or periodic self-stimulation at `seed`.
    Returns (active node counts, firing node counts) per step."""
    a = np.zeros((W.shape[0], 1), np.float32)
    a[seed] = amp
    active, firing = [int((a > 0).sum())], [int((a > thr).sum())]
    for t in range(steps):
        a = step_dynamics(a, W, decay, gain, thr)
        if stim_interval and (t + 1) % stim_interval == 0:
            a[seed] += amp
        active.append(int((a > 0).sum())); firing.append(int((a > thr).sum()))
    return active, firing


def circuit_responses(W, seed_nodes, steps, decay, gain, thr, batch=64, prune=1e-3):
    """Response vector of each seed: activity integrated over `steps`, as sparse
    unit-norm rows. Batched dense propagation keeps memory flat."""
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
        self.by_stem = {}
        for i in range(4, self.V):                # itos is frequency-sorted
            self.by_stem.setdefault(stem(self.itos[i]), i)

    def encode(self, toks):
        return [self.stoi.get(t, self.unk) for t in toks]

    def resolve(self, tok):
        """Known word -> itself. Otherwise try a shared stem, then a close spelling.
        Returns a vocabulary token or None."""
        if tok in self.stoi:
            return tok
        if not WORD_RE.match(tok) or len(tok) < 4:
            return None
        i = self.by_stem.get(stem(tok))
        if i is not None:
            return self.itos[i]
        m = difflib.get_close_matches(tok, self.itos[4:], n=1, cutoff=0.85)
        return m[0] if m else None


class NGram:
    """Interpolated n-gram; order = number of previous tokens used."""

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
# PART 3 -- DOT-PRODUCT EXTRACTION: token placement on the brain + G matrix
# ===========================================================================
@dataclass
class Circuit:
    ids: np.ndarray        # vocab ids that have their own circuit vector
    row_of: np.ndarray     # vocab id -> row in G (or -1); inflections alias a sibling's row
    G: np.ndarray          # circuit dot products  r_i . r_j   (diag = 0)
    Gp: np.ndarray         # ablation: PPMI-cosine dots, no brain (diag = 0)
    ppmi: np.ndarray       # raw PPMI (used only by the diagnostic metric)
    stats: dict
    info: dict = None      # per-word MRI context arrays (see word_context)


def rank01(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(len(x) - 1, 1)


def noodle_field(vol, mask, tube_pct=25.0, sigma_grad=1.0, sigma_int=2.0):
    """Structure-tensor orientation field. For each tissue voxel: the direction of least
    intensity variation (along a noodle) and a tube-ness score (~1 tube-like, ~0 plane or
    isotropic). The stopping threshold is a percentile of the scan's own tube-ness, because
    the absolute scale depends on resolution and contrast."""
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(vol, sigma_grad)
    g = np.stack(np.gradient(sm), -1)
    J = np.empty(vol.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            J[..., i, j] = J[..., j, i] = gaussian_filter(g[..., i] * g[..., j], sigma_int)
    w, v = np.linalg.eigh(J[mask])                       # eigenvalues ascending
    direction = np.zeros(vol.shape + (3,), np.float32)
    direction[mask] = v[:, :, 0]
    tube = np.zeros(vol.shape, np.float32)
    tube[mask] = (w[:, 1] - w[:, 0]) / (w[:, 2] + 1e-9)
    return dict(dir=direction, tube=tube, mask=mask,
                thr=float(np.percentile(tube[mask], tube_pct)))


def trace_noodles(field, seeds, K=16, L=24, max_angle=70.0, seed=0, paths=False):
    """Trace K noodles from each seed (grid coords, shape (n,3)) for up to L unit steps.
    Returns survival S of shape (n, L+1): S[w, k] = share of word w's noodles still alive
    after k steps (S[:,0] = 1). With paths=True also returns positions/alive for plotting."""
    rng = np.random.default_rng(seed)
    n, shape = len(seeds), np.array(field["mask"].shape)
    pos = np.repeat(np.asarray(seeds, np.float64), K, 0) + rng.normal(0, 0.7, (n * K, 3))
    sign = np.tile([1.0, -1.0], (n * K + 1) // 2)[:n * K]     # half go each way
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
            d[dot < 0] *= -1                                   # orientation has no sign
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
    """Coarse octant name from voxel position and the scan's axis codes."""
    parts = [_AXIS_NAMES[c][0] if xyz[k] > mid[k] else _AXIS_NAMES[c][1]
             for k, c in enumerate(codes)]
    return "-".join(sorted(parts, key=_AXIS_ORDER.get))


def word_context_table(R, seeds, coords_all, vals, keep_cc, codes):
    """Per-word context read from the scan: seed position/region/tissue band and
    the footprint of the word's pulse (size, centroid, mean intensity)."""
    mid = (keep_cc.min(0) + keep_cc.max(0)) / 2
    lo, hi = np.percentile(vals[vals > 0], [33, 67])
    seed_int = vals[seeds]
    # relative thirds of this scan's tissue intensities (not a tissue classification)
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
    """Everything the MRI says about one word, or None if it has no circuit vector.
    Inflections share their sibling's vector, so they report the sibling's context."""
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
        print("[circuit] too few content tokens; circuit disabled")
        return None
    Vc = len(ids)
    row_of = np.full(vocab.V, -1, np.int64)
    row_of[ids] = np.arange(Vc)

    # --- PPMI over sentence co-occurrence (float32 to keep the dense Vc x Vc matrices small)
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

    # --- alias inflections of circuit words onto their sibling's row (done after PPMI so
    #     co-occurrence counts aren't merged; function words in `skip` are never aliased)
    by_stem = {}
    for i in ids:
        by_stem.setdefault(stem(vocab.itos[i]), row_of[i])
    for i in range(4, vocab.V):
        if row_of[i] < 0 and i not in skip and WORD_RE.match(vocab.itos[i]):
            rr = by_stem.get(stem(vocab.itos[i]))
            if rr is not None:
                row_of[i] = rr
    print(f"[circuit] {int((row_of >= 0).sum())} vocab words resolve to a circuit vector "
          f"({Vc} direct, rest via stems)")

    # cache keyed by every parameter that affects G
    nd = dict(L=getattr(a, "noodle_steps", 24), K=getattr(a, "noodle_count", 16),
              ang=getattr(a, "noodle_angle", 70.0), pct=getattr(a, "noodle_tube_pct", 25.0))
    sig = json.dumps(dict(v=4, nd=nd, mri=a.mri, ds=a.downsample, vt=a.voxel_threshold, steps=a.steps,
                          decay=a.decay, gain=a.gain, fire=a.fire_threshold,
                          toks=[vocab.itos[i] for i in ids]), sort_keys=True)
    if a.dots_cache and os.path.exists(a.dots_cache):
        z = np.load(a.dots_cache, allow_pickle=False)
        if str(z["sig"]) == sig:
            print(f"[circuit] loaded cached dot products from {a.dots_cache}")
            info = {k[5:]: z[k] for k in z.files if k.startswith("info_")}
            return Circuit(ids, row_of, z["G"], Gp, P, dict(cached=True, tokens=Vc), info)

    # --- brain graph
    vol, codes = load_volume(a.mri, a.downsample)
    coords, n = build_graph(vol, a.voxel_threshold, a.edges_path)
    vals = vol[tuple(coords.T)]
    W = load_adjacency(a.edges_path, n)
    ncomp, lab = connected_components(W, directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    wbar = float(W.data.mean())
    gain = a.gain if a.gain > 0 else 1.5 * a.fire_threshold / wbar   # sharp-transition rule
    print(f"[circuit] volume {vol.shape}: {n} nodes, {W.nnz} directed edges, "
          f"largest component {len(keep)}/{n}, mean edge weight {wbar:.2f}, "
          f"gain {gain:.2f} (spreads only if gain*w > threshold {a.fire_threshold})")

    # --- place tokens on the brain: PPMI -> 3 SVD dims -> uniform [0,1] -> bbox -> nearest node
    Ps = sp.csr_matrix(P).astype(np.float64)       # PPMI is sparse; avoids a dense float64 copy
    U, S, _ = svds(Ps, k=3, v0=np.random.default_rng(0).normal(size=Vc))  # seeded: reproducible placement
    pos = np.stack([rank01(v) for v in (U * S).T], axis=1)
    cc = coords[keep].astype(np.float64)
    target = cc.min(0) + pos * (cc.max(0) - cc.min(0))
    seeds = keep[cKDTree(cc).query(target)[1]]

    # --- circuit responses -> dot products
    R = circuit_responses(W, seeds, a.steps, a.decay, gain, a.fire_threshold)
    G = (R @ R.T).toarray().astype(np.float32)
    np.fill_diagonal(G, 0)
    info = word_context_table(R, seeds, coords, vals, cc, codes)
    field = noodle_field(vol, vol > a.voxel_threshold, nd["pct"])
    info["curve"] = trace_noodles(field, coords[seeds], nd["K"], nd["L"], nd["ang"])
    hl = [(np.flatnonzero(c < 0.5)[:1].tolist() or [nd["L"]])[0] for c in info["curve"]]
    print(f"[noodles] {nd['K']} noodles/word, {nd['L']} steps: context half-life "
          f"min/median/max = {min(hl)}/{int(np.median(hl))}/{max(hl)} steps "
          f"(tube-ness stop threshold {field['thr']:.3f})")
    nnz = np.diff(R.indptr)
    stats = dict(nodes=n, tokens=Vc, median_active_nodes=int(np.median(nnz)),
                 frac_of_brain=float(np.median(nnz) / n), gain=gain)
    print(f"[circuit] extracted {Vc}x{Vc} dot products; each token's response covers "
          f"~{stats['median_active_nodes']} nodes ({100 * stats['frac_of_brain']:.2f}% of brain)")
    if stats["median_active_nodes"] < 10:
        print("[circuit] warning: pulses are not spreading; raise --gain or lower --fire-threshold")
    if a.dots_cache:
        np.savez(a.dots_cache, G=G, sig=sig, **{"info_" + k: v for k, v in info.items()})
    return Circuit(ids, row_of, G, Gp, P, stats, info)


def expand_bias(circ, vocab, s):
    """Row scores -> full-vocab bias, robustly normalised to [0, 1].
    Aliased inflections share their sibling's score."""
    pos = s[s > 0]
    if not pos.size:
        return None
    x = np.clip(s / np.percentile(pos, 99), 0, 1)
    full = np.zeros(vocab.V, np.float32)
    has = circ.row_of >= 0
    full[has] = x[circ.row_of[has]]
    return full


def prompt_bias(circ, vocab, toks, source):
    """Dot products between the prompt's content tokens and every token,
    robustly normalised to [0, 1] and expanded to the full vocabulary."""
    if circ is None or source == "none":
        return None
    M = circ.G if source == "circuit" else circ.Gp
    rows = [circ.row_of[vocab.stoi[t]] for t in toks
            if t in vocab.stoi and circ.row_of[vocab.stoi[t]] >= 0]
    if not rows:
        return None
    return expand_bias(circ, vocab, M[:, rows].sum(axis=1))


def context_bias(circ, M, vocab, gen_ids, decay, curves=False, stride=3):
    """Word-by-word context: each recently generated word injects its own circuit
    response; its weight at age `a` is its noodle curve S_w(a*stride) (curves=True)
    or decay**a. Returns the normalised dot-product bias."""
    u = np.zeros(len(circ.ids), np.float32)
    C = circ.info["curve"] if curves else None
    for age, tok in enumerate(reversed(gen_ids)):
        r = circ.row_of[tok]
        if r >= 0:
            u[r] += C[r, min(age * stride, C.shape[1] - 1)] if curves else decay ** age
    if not u.any():
        return None
    return expand_bias(circ, vocab, M @ u)


# ===========================================================================
# PART 4 -- DECODING
# ===========================================================================
@dataclass
class Gen:
    max_len: int = 40
    temp: float = 0.9
    top_k: int = 40
    target_entropy: float = 0.0
    beam: int = 0
    min_len: int = 0
    length_alpha: float = 0.7
    anchor_amp: float = 0.0       # direct leaky logit bias on prompt words
    anchor_decay: float = 0.9
    anchor_interval: int = 16
    dots: str = "none"            # none | circuit | ppmi
    beta: float = 6.0             # logit strength of the dot-product bias
    fb: float = 1.0               # >0: generated words feed their own MRI context back in
    fb_window: int = 4            # only the last N generated words contribute
    fb_decay: float = 0.7         # newest word weight 1, then decay^age
    curves: str = "noodle"        # noodle | exp  (per-word MRI curve vs plain exponential decay)
    curve_stride: int = 3         # noodle steps per word of age
    eos_guard: bool = True        # keep P(end-of-text) unchanged when a bias boosts other words
    slack: int = 20


def entropy(logits, temp):
    z = logits / temp
    p = np.exp(z - z.max())
    p /= p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def temperature_for_entropy(logits, target, lo=0.05, hi=3.0, iters=24):
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if entropy(logits, mid) < target else (lo, mid)
    return (lo + hi) / 2


def top_k_mask(lg, k):
    if 0 < k < len(lg):
        lg = np.where(lg < np.partition(lg, -k)[-k], -np.inf, lg)
    return lg


def pick(lg, gen):
    lg = top_k_mask(lg, gen.top_k)
    t = (temperature_for_entropy(lg, gen.target_entropy)
         if gen.target_entropy > 0 else max(gen.temp, 1e-3))
    p = np.exp(lg / t - (lg / t).max())
    return int(np.random.choice(len(p), p=p / p.sum())), t


class LeakySchedule:
    """Self-stimulation: level leaks by `decay` each step and is re-pulsed by
    `amp` every `interval` steps (precomputed so beams share it)."""

    def __init__(self, amp, decay, interval, n):
        level, self.levels = amp, []
        for t in range(n):
            self.levels.append(level)
            level *= decay
            if (t + 1) % interval == 0:
                level += amp

    def level(self, t):
        return self.levels[min(t, len(self.levels) - 1)]


def anchor_ids(vocab, toks, skip_top=20):
    common = set(np.argsort(-vocab.counts)[:skip_top].tolist())
    return sorted({vocab.stoi[t] for t in toks if t in vocab.stoi and WORD_RE.match(t)
                   and vocab.stoi[t] not in common})


def materialize(node):
    out = []
    while node is not None:
        tok, node = node
        out.append(tok)
    return out[::-1]


def log_softmax(lg):
    m = lg.max()
    return lg - (m + np.log(np.exp(lg - m).sum()))


def beam_search(step_fn, ctx0, eos, max_new, width, top_k, alpha, state_len):
    """state_len = how many trailing tokens fully determine the future (model order,
    or the feedback window if larger), so merging beams on that tail is exact."""
    path0 = None
    for t in ctx0:
        path0 = (t, path0)
    beam, done = [(0.0, path0)], []
    for step in range(max_new):
        cands = {}
        for score, path in beam:
            seq = materialize(path)
            logp = log_softmax(step_fn(seq))
            k = min(top_k if top_k > 0 else 16, len(logp) - 1)
            for tok in np.argpartition(-logp, k)[:k]:
                if not np.isfinite(logp[tok]):
                    continue
                tok = int(tok)
                s, newp = score + float(logp[tok]), (tok, path)
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


def min_steps_to_eos(model, vocab, start):
    """Shortest observed path from `start` to end-of-text (feasibility + length floor)."""
    if model.order < 1:
        return None
    graph, seen, q = model.tables[1], {start}, deque([(start, 0)])
    while q:
        s, d = q.popleft()
        nxt = graph.get((s,))
        if not nxt:
            continue
        if vocab.eos in nxt:
            return d + 1
        for n in nxt:
            if n not in seen:
                seen.add(n); q.append((n, d + 1))
    return None


def generate(model, vocab, toks, gen, circ=None, trace=None):
    """trace (list, sampling mode only) receives (token_id, bias_applied_to_it)."""
    ctx0 = [vocab.bos] + vocab.encode(toks)
    anchors = anchor_ids(vocab, toks) if gen.anchor_amp > 0 else []
    a_sched = (LeakySchedule(gen.anchor_amp, gen.anchor_decay, gen.anchor_interval,
                             gen.max_len + 1) if anchors else None)
    M = None if circ is None else {"circuit": circ.G, "ppmi": circ.Gp}.get(gen.dots)
    bp = prompt_bias(circ, vocab, toks, gen.dots)
    use_fb = M is not None and gen.fb > 0
    use_curves = (gen.curves == "noodle" and circ is not None and bool(circ.info)
                  and "curve" in circ.info)
    d_sched = (LeakySchedule(1.0, gen.anchor_decay, gen.anchor_interval, gen.max_len + 1)
               if bp is not None else None)
    last = {}

    def step(seq):
        t = len(seq) - len(ctx0)
        lg = model.logits(seq)
        if t < gen.min_len:
            lg[vocab.eos] = -np.inf
        if a_sched:
            lg[anchors] += a_sched.level(t)
        b = None
        if bp is not None:
            b = d_sched.level(t) * bp
        if use_fb:
            f = context_bias(circ, M, vocab, seq[len(ctx0):][-gen.fb_window:], gen.fb_decay,
                             use_curves, gen.curve_stride)
            if f is not None:
                b = gen.fb * f if b is None else b + gen.fb * f
        if b is not None:
            if gen.eos_guard and t >= gen.min_len:
                # keep P(end-of-text) unchanged: shift it by the log-ratio of the other
                # words' probability mass after vs before the boost
                p = np.exp(lg - lg.max())
                p[vocab.eos] = 0.0
                if p.sum() > 0:
                    lg[vocab.eos] += np.log((p * np.exp(gen.beta * b)).sum() / p.sum())
            lg += gen.beta * b
        last["b"] = b
        return lg

    if gen.beam > 0:
        state_len = max(model.order, 1, gen.fb_window if use_fb else 0)
        return beam_search(step, ctx0, vocab.eos, gen.max_len, gen.beam,
                           gen.top_k, gen.length_alpha, state_len), []
    ctx, out, temps = list(ctx0), [], []
    for _ in range(gen.max_len):
        nxt, t = pick(step(ctx), gen)
        if nxt == vocab.eos:
            break
        if trace is not None:
            trace.append((nxt, 0.0 if last["b"] is None else float(last["b"][nxt])))
        out.append(nxt); ctx.append(nxt); temps.append(t)
    return out, temps


def resolve_prompt(vocab, toks):
    """Map each prompt token to a known word where possible.
    Returns (resolved tokens, list of 'typed→matched' strings)."""
    resolved, mapped = [], []
    for t in toks:
        r = vocab.resolve(t)
        if r is not None and r != t:
            mapped.append(f"{t}→{r}")
        resolved.append(r if r is not None else t)
    return resolved, mapped


def complete(model, vocab, prompt, gen, circ=None, trace=None):
    toks = TOKEN_RE.findall(prompt.lower())
    if not toks:
        raise ValueError("prompt has no usable tokens")
    resolved, mapped = resolve_prompt(vocab, toks)
    ids = vocab.encode(resolved)
    oov = [t for t, i in zip(toks, ids) if i == vocab.unk]
    d = min_steps_to_eos(model, vocab, ids[-1])
    note = None
    if d is None:
        note = "no observed path to end-of-text from the last prompt token"
    else:
        gen = replace(gen, max_len=max(gen.max_len, d + gen.slack))
    out, _ = generate(model, vocab, resolved, gen, circ, trace)
    if mapped:
        note = (note + "; " if note else "") + "matched unseen words: " + ", ".join(mapped)
    return detok(toks + [vocab.itos[i] for i in out]), oov, note


def calibrate_beam(model, vocab, sents, circ=None, widths=(1, 2, 4, 8, 16, 32), ref=64,
                   n_prompts=6, max_len=25, top_k=20):
    rng = random.Random(0)
    starts = [s[:2] for s in rng.sample(sents, min(n_prompts, len(sents)))]

    def run(w):
        g = Gen(max_len=max_len, top_k=top_k, beam=w)
        return [tuple(generate(model, vocab, s, g, circ)[0]) for s in starts]

    reference = run(ref)
    return next((w for w in widths if run(w) == reference), ref)


def diagnose(model, vocab, prompt, gen, circ, n=30):
    """Ablation. Metric = mean PPMI between generated content words and the prompt's
    content words (a direct corpus statistic). Caveat: PPMI also builds both dot-product
    sources, so the 'ppmi' arm is favoured by construction; use a labelled-topic corpus
    for an independent check."""
    toks, _ = resolve_prompt(vocab, TOKEN_RE.findall(prompt.lower()))
    prompt_rows = [circ.row_of[vocab.stoi[t]] for t in toks
                   if t in vocab.stoi and circ.row_of[vocab.stoi[t]] >= 0] if circ else []
    anchors = set(anchor_ids(vocab, toks))
    base = replace(gen, anchor_amp=0.0, dots="none", beam=0)
    arms = [("none", base), ("anchor", replace(base, anchor_amp=gen.anchor_amp or 2.0))]
    if circ is not None and prompt_rows:
        fb = gen.fb or 0.5
        arms += [("ppmi dots", replace(base, dots="ppmi")),
                 ("circuit dots", replace(base, dots="circuit")),
                 ("circuit+exp ctx", replace(base, dots="circuit", fb=fb, curves="exp")),
                 ("circuit+noodle ctx", replace(base, dots="circuit", fb=fb, curves="noodle"))]
    print(f"  {'arm':18s} {'prompt-PMI':>10s} {'anchor-rate':>11s} {'mean len':>8s}")
    for label, g in arms:
        pm, hits, total = [], 0, 0
        for _ in range(n):
            out, _ = generate(model, vocab, toks, g, circ)
            hits += sum(o in anchors for o in out)
            total += len(out)
            if circ is not None and prompt_rows:
                rows = [circ.row_of[o] for o in out if circ.row_of[o] >= 0]
                if rows:
                    pm.append(circ.ppmi[np.ix_(rows, prompt_rows)].mean())
        print(f"  {label:18s} {np.mean(pm) if pm else float('nan'):10.3f} "
              f"{hits / max(total, 1):11.4f} {total / n:8.1f}")


def plot_curves(path, circ, vocab, words, a, K=12):
    """Left: each word's context curve S_w(k). Right: its noodles in 3-D over the brain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ctxs = [c for c in (word_context(circ, vocab, w) for w in words) if c]
    if not ctxs:
        print("[plot] none of the words has a circuit vector")
        return
    vol, _ = load_volume(a.mri, a.downsample)
    mask = vol > a.voxel_threshold
    field = noodle_field(vol, mask, getattr(a, "noodle_tube_pct", 25.0))
    seeds = np.array([c["xyz"] for c in ctxs])
    L = len(ctxs[0]["curve"]) - 1
    _, tr = trace_noodles(field, seeds, K, L, getattr(a, "noodle_angle", 70.0), paths=True)
    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_subplot(1, 2, 1)
    for c in ctxs:
        ax.plot(c["curve"], label=f"{c['word']} (half-life {c['half_life']})", lw=2)
    ax.set_xlabel("noodle step k"); ax.set_ylabel("share of noodles alive  S(k)")
    ax.set_title("per-word context curves"); ax.legend(fontsize=8); ax.set_ylim(0, 1.02)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    cloud = np.argwhere(mask)
    cloud = cloud[np.random.default_rng(0).choice(len(cloud), min(3000, len(cloud)), replace=False)]
    ax3.scatter(cloud[:, 2], cloud[:, 1], cloud[:, 0], c="lightgray", s=1, alpha=0.12)
    for wi, c in enumerate(ctxs):
        color = f"C{wi}"
        for k in range(K):
            live = tr["alive"][:, wi, k]
            pts = tr["pos"][live, wi, k]
            if len(pts) > 1:
                ax3.plot(pts[:, 2], pts[:, 1], pts[:, 0], color=color, lw=1.2, alpha=0.8)
        ax3.scatter(*seeds[wi][[2, 1, 0]], color=color, s=40, edgecolors="k")
    ax3.set_title("noodles from each word's node"); ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    plt.tight_layout(); plt.savefig(path, dpi=140)
    print(f"[plot] saved {path}")


# ===========================================================================
# CLI
# ===========================================================================
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
    g.add_argument("--steps", type=int, default=4, help="propagation steps per token pulse")
    g.add_argument("--decay", type=float, default=0.6)
    g.add_argument("--gain", type=float, default=0.0, help="0 = auto (1.5 x threshold / mean weight)")
    g.add_argument("--fire-threshold", type=float, default=0.55)
    g.add_argument("--circuit-vocab", type=int, default=8000,
                   help="max content words with their own brain vector (dense Vc x Vc matrices: memory ~ Vc^2)")
    g.add_argument("--circuit-min-count", type=int, default=2)
    g.add_argument("--skip-top", type=int, default=20)
    g.add_argument("--dots-cache", default=None)
    g.add_argument("--sim-report", action="store_true")
    g.add_argument("--noodle-steps", type=int, default=24)
    g.add_argument("--noodle-count", type=int, default=16, help="noodles traced per word")
    g.add_argument("--noodle-angle", type=float, default=70.0, help="max turn per step, degrees")
    g.add_argument("--noodle-tube-pct", type=float, default=25.0,
                   help="noodles stop where tube-ness is below this percentile of the scan")
    g = ap.add_argument_group("decoding")
    g.add_argument("--dots", choices=["none", "circuit", "ppmi"], default="none")
    g.add_argument("--beta", type=float, default=6.0)
    g.add_argument("--fb", type=float, default=1.0, help="word-by-word MRI context strength (0=off)")
    g.add_argument("--fb-window", type=int, default=4)
    g.add_argument("--fb-decay", type=float, default=0.7, help="decay for --curves exp")
    g.add_argument("--curves", choices=["noodle", "exp"], default="noodle")
    g.add_argument("--curve-stride", type=int, default=3, help="noodle steps per word of age")
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--temp", type=float, default=0.9)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--max-len", type=int, default=400)
    g.add_argument("--min-len", type=int, default=0)
    g.add_argument("--target-entropy", type=float, default=0.0)
    g.add_argument("--anchor", type=float, default=0.0)
    g.add_argument("--anchor-decay", type=float, default=0.9)
    g.add_argument("--anchor-interval", type=int, default=16)
    g.add_argument("--beam", type=int, default=0)
    g.add_argument("--length-alpha", type=float, default=0.7)
    g.add_argument("--slack", type=int, default=20)
    ap.add_argument("--explain", nargs="+", metavar="WORD", help="print each word's MRI context")
    ap.add_argument("--plot-curves", metavar="PNG", help="plot context curves + noodles for --explain words (or prompt words)")
    ap.add_argument("--trace", action="store_true", help="per-word MRI context of the first completion")
    ap.add_argument("--no-eos-guard", action="store_true", help="let the bias change the stopping probability")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    sents = load_sentences(a.path)
    vocab = Vocab(sents, a.min_count)
    model = NGram(vocab, a.order, base_clip=a.base_clip)
    model.fit([[vocab.bos] + vocab.encode(s) + [vocab.eos] for s in sents])

    need_circuit = a.dots != "none" or a.diagnose or a.sim_report or a.explain or a.trace or a.plot_curves
    circ = build_circuit(sents, vocab, a) if need_circuit else None

    if a.sim_report:   # the original simulator's regime check, on this graph
        vol, _ = load_volume(a.mri, a.downsample)
        coords, n = build_graph(vol, a.voxel_threshold, a.edges_path + ".sim")
        W = load_adjacency(a.edges_path + ".sim", n)
        seed = int(np.argmin(coords[:, 0]))
        gain = a.gain if a.gain > 0 else 1.5 * a.fire_threshold / float(W.data.mean())
        for name, kw in (("repo params (gain 0.4)", dict(gain=0.4)),
                         (f"auto gain {gain:.2f}", dict(gain=gain))):
            for label, si in (("single pulse", 0), ("self-stim", 6)):
                act, fire = simulate_activity(W, seed, steps=30, decay=a.decay,
                                              thr=a.fire_threshold, stim_interval=si, **kw)
                print(f"[sim] {name:22s} {label:12s} peak active {max(act):6d}  peak firing {max(fire):6d}")

    gen = Gen(max_len=a.max_len, temp=a.temp, top_k=a.top_k, target_entropy=a.target_entropy,
              beam=a.beam, min_len=a.min_len, length_alpha=a.length_alpha,
              anchor_amp=a.anchor, anchor_decay=a.anchor_decay,
              anchor_interval=a.anchor_interval, dots=a.dots, beta=a.beta, slack=a.slack,
              fb=a.fb, fb_window=a.fb_window, fb_decay=a.fb_decay,
              curves=a.curves, curve_stride=a.curve_stride,
              eos_guard=not a.no_eos_guard)

    def show_context(c):
        print(f"  {c['word']:12s} node {c['xyz']}  {c['region']}, {c['band']}  "
              f"intensity {c['intensity']:.2f}  footprint {c['footprint']} nodes "
              f"(mean intensity {c['footprint_intensity']:.2f})")
        print("               nearest through the wiring: " +
              (", ".join(f"{w} {v:.2f}" for w, v in c["neighbours"]) or "none"))
        if c["curve"] is not None:
            print(f"               noodle context curve |{sparkline(c['curve'][::2])}|  "
                  f"half-life {c['half_life']} steps, mean length {c['noodle_len']:.1f}")

    if a.explain:
        explain_tokens = explain_words(a.explain)
        print("[explain] per-word MRI context (node = grid coordinates after downsampling)")
        for word in explain_tokens:
            c = word_context(circ, vocab, word, top=5) if circ else None
            if c:
                show_context(c)
            else:
                print(f"  {word:12s} no circuit vector (function word, rare, or out of vocabulary)")

    if a.plot_curves and circ is not None:
        explain_tokens = explain_words(a.explain)
        plot_words = (
            explain_tokens
            if explain_tokens
            else [
                t for t in resolve_prompt(vocab, TOKEN_RE.findall((a.prompt or "").lower()))[0]
                if t in vocab.stoi
            ][:6]
        )
        plot_curves(a.plot_curves, circ, vocab, plot_words, a)

    if a.calibrate:
        w = calibrate_beam(model, vocab, sents, circ)
        print(f"calibrated beam width: {w} (matches a width-64 reference); use --beam {w}")

    def run(prompt):
        oov, note = set(), None
        for k in range(a.n):
            tr = [] if (a.trace and k == 0 and a.beam == 0) else None
            text, o, note = complete(model, vocab, prompt, gen, circ, tr)
            oov.update(o)
            print(" ", text)
            if tr:
                print("  per-word MRI context (bias = dot-product logit boost this word received):")
                for tok, b in tr:
                    c = word_context(circ, vocab, vocab.itos[tok]) if circ else None
                    where = (f"{c['region']}, {c['band'].split(' ')[0]}, footprint {c['footprint']}"
                             + (f", noodle half-life {c['half_life']}" if c["half_life"] is not None else "")
                             if c else "no circuit vector")
                    print(f"    {vocab.itos[tok]:12s} bias {b:5.2f}   {where}")
        if note:
            print(f"  ({note})")
        if oov:
            print(f"  (not in vocab, treated as <unk>: {sorted(oov)})")
        if a.diagnose:
            diagnose(model, vocab, prompt, gen, circ)

    if a.prompt:
        run(a.prompt)
    if a.interactive:
        print("interactive mode: type a prompt, empty line to quit")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            try:
                run(line)
            except ValueError as e:
                print(f"  ({e})")


if __name__ == "__main__":
    main()
