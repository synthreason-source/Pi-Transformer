#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V17 — Thébault + MRV + Isomorphic Syntax Stacking
              + Positional Vectorisation + Chunked Sum Generation
===============================================================================

NEW IN V17
──────────
1. ISOMORPHIC SYNTAX STACKING
   After each sentence is generated, it is embedded as a sequence of
   Thébault triples.  A new sentence can be "stacked" against the corpus
   of prior sentences by computing a sentence-level similarity score:

       SentSim(S_a, S_b) = (1/min(|S_a|,|S_b|)) ·
                            Σ_i  K_reg(ρ_a_i, ρ_b_i) · K_side(σ_a_i, σ_b_i)

   Sentences are ranked by descending SentSim to the current context and
   the top-k are used as "syntax anchors" that bias generation.

2. POSITIONAL VECTORISATION
   Each token t at position i in a window of length L is represented as a
   4-D vector:

       v(t, i) = [ ρ_t,  θ_t,  σ_t,  i/(L-1) ]

   where i/(L-1) ∈ [0,1] encodes relative position.

3. CHUNKED LEFT-TO-RIGHT SUM → GENERATION PROBABILITIES
   The running context window is split into C equal chunks.
   For each chunk c:

       chunk_vec_c = Σ_{t in chunk_c}  v(t, pos(t))

   The C chunk vectors are concatenated into a flat "chunk signature"
   of length 4·C.  This signature is projected onto the candidate
   vocabulary to produce an additive logit bonus:

       chunk_bonus(cand) = chunk_sig · v(cand, C)   (dot product)

   Candidates whose positional-Thébault vector aligns with the running
   chunk sum score higher, creating a left-to-right geometric momentum.

All V16.1 components (Thébault geometry, MRV, graph potential, conjugate
orbit, synthetic_reason) are retained unchanged.
===============================================================================
"""

from __future__ import annotations
import re, math, random, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import torch
import torch.nn.functional as F
import gradio as gr


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — TOKEN PRIMITIVES  (unchanged from V16.1)
# ════════════════════════════════════════════════════════════════════════════

STOP_WORDS_COG = set(
    "a an and are as at be by for from has have he her him his i in is it its "
    "me my of on or our she so that the their them they this to was we were what "
    "when where which who will with you your if because while".split()
)
COGNITIVE_TOKENS = {f"[{w.upper()}]" for w in STOP_WORDS_COG}
PUNCT_TOKENS     = {",", ".", "!", "?", ";", ":"}


def tokenize(text: str) -> List[str]:
    out = []
    for w in re.findall(r"\[[A-Z]+\]|\b[a-zA-Z]+\b|[.,!?;:]", text):
        if w in COGNITIVE_TOKENS or w in PUNCT_TOKENS:
            out.append(w)
        else:
            w_c = "".join(
                c for c in unicodedata.normalize("NFD", w)
                if unicodedata.category(c) != "Mn"
            ).lower()
            if w_c:
                out.append(f"[{w_c.upper()}]" if w_c in STOP_WORDS_COG else w_c)
    return out


def detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    res = []
    for t in tokens:
        if t in PUNCT_TOKENS:
            if res:
                res[-1] += t
            continue
        if t in COGNITIVE_TOKENS:
            raw  = t.strip("[]").lower()
            word = raw.capitalize() if not res or res[-1].endswith(('.', '!', '?')) else raw
            res.append(word)
        else:
            word = t.capitalize() if not res or res[-1].endswith(('.', '!', '?')) else t
            res.append(word)
    out = " ".join(res).strip()
    return out if out and out[-1] in PUNCT_TOKENS else out + "."


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — THÉBAULT TOKEN GEOMETRY  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def _perfect_square_cv() -> float:
    s = 1.0
    d = [s, s, s, s, s * math.sqrt(2), s * math.sqrt(2)]
    mu = sum(d) / 6
    cv = math.sqrt(sum((x - mu) ** 2 for x in d) / 6) / mu
    return cv


_PERFECT_CV = _perfect_square_cv()


def _rotate90(vx: float, vy: float) -> Tuple[float, float]:
    return -vy, vx


def _thebault_centres(ax, ay, bx, by, cx, cy, dx, dy):
    corners = [(ax, ay), (bx, by), (cx, cy), (dx, dy)]
    centres = []
    for i in range(4):
        px, py = corners[i]
        qx, qy = corners[(i + 1) % 4]
        mx, my = (px + qx) / 2, (py + qy) / 2
        hx, hy = (qx - px) / 2, (qy - py) / 2
        rx, ry = _rotate90(hx, hy)
        centres.append((mx + rx, my + ry))
    return centres


def _thebault_triple(px, py, qx, qy):
    if abs(px) < 1e-9 and abs(py) < 1e-9 and abs(qx) < 1e-9 and abs(qy) < 1e-9:
        return 0.0, 0.0, 0.0
    T = _thebault_centres(0.0, 0.0, px, py, px + qx, py + qy, qx, qy)
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            dx = T[i][0] - T[j][0]
            dy = T[i][1] - T[j][1]
            dists.append(math.sqrt(dx * dx + dy * dy))
    mu = sum(dists) / 6
    if mu < 1e-9:
        return 0.0, 0.0, 0.0
    cv  = math.sqrt(sum((d - mu) ** 2 for d in dists) / 6) / mu
    rho = max(0.0, min(1.0, 1.0 - cv / (_PERFECT_CV + 1e-9)))
    sides = [dists[0], dists[1], dists[2], dists[3]]
    sigma = sum(sides) / 4.0
    dx_ori = T[1][0] - T[0][0]
    dy_ori = T[1][1] - T[0][1]
    theta  = math.atan2(dy_ori, dx_ori) % math.pi
    return rho, theta, sigma


@dataclass
class ThebaultTriple:
    rho   : float
    theta : float
    sigma : float


class ThebaultTokenGeometry:
    def __init__(self):
        self._vecs  : Dict[str, Tuple[float, float, float, float]] = {}
        self._cache : Dict[str, ThebaultTriple]                    = {}

    def register(self, token, freq, index, max_freq, vocab_size):
        f_hat = freq / max(max_freq, 1e-9)
        k_hat = index / max(vocab_size - 1, 1)
        angle_p = 2.0 * math.pi * k_hat
        angle_q = 2.0 * math.pi * f_hat
        px = f_hat * math.cos(angle_p)
        py = f_hat * math.sin(angle_p)
        qx = k_hat * math.cos(angle_q)
        qy = k_hat * math.sin(angle_q)
        self._vecs[token] = (px, py, qx, qy)
        self._cache.pop(token, None)

    def _vec(self, token):
        return self._vecs.get(token, (0.0, 0.0, 0.0, 0.0))

    def triple(self, token: str) -> ThebaultTriple:
        if token in self._cache:
            return self._cache[token]
        px, py, qx, qy = self._vec(token)
        rho, theta, sigma = _thebault_triple(px, py, qx, qy)
        t = ThebaultTriple(rho, theta, sigma)
        self._cache[token] = t
        return t

    def composed_triple(self, t1: str, t2: str) -> ThebaultTriple:
        p1x, p1y, q1x, q1y = self._vec(t1)
        p2x, p2y, q2x, q2y = self._vec(t2)
        rho, theta, sigma = _thebault_triple(p1x + p2x, p1y + p2y, q1x + q2x, q1y + q2y)
        return ThebaultTriple(rho, theta, sigma)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — THÉBAULT KERNELS  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultKernels:
    def __init__(self, lambda_reg: float = 8.0, gamma_side: float = 4.0):
        self.lambda_reg  = lambda_reg
        self.gamma_side  = gamma_side

    def k_reg(self, rho_a: float, rho_b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.lambda_reg * (rho_b - rho_a) ** 2)

    def k_ori(self, theta_a: float, theta_b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(theta_b - theta_a))

    def k_side(self, sigma_a: float, sigma_b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)

    def all_scores(self, ctx, cand_rho, cand_theta, cand_sigma):
        return (
            self.k_reg (ctx.rho,   cand_rho),
            self.k_ori (ctx.theta, cand_theta),
            self.k_side(ctx.sigma, cand_sigma),
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2.5 — MRV CONSTRAINT FILTER  (unchanged from V16.1)
# ════════════════════════════════════════════════════════════════════════════

class MRVConstraintFilter:
    def __init__(self, threshold=0.50, mrv_cap_ratio=2.0, max_vocab_scan=300):
        self.threshold      = threshold
        self.mrv_cap_ratio  = mrv_cap_ratio
        self.max_vocab_scan = max_vocab_scan
        self._vocab_rho   : Optional[torch.Tensor] = None
        self._vocab_sigma : Optional[torch.Tensor] = None
        self._vocab_tokens: List[str]              = []

    def prime(self, vocab, geo):
        scan  = vocab[:self.max_vocab_scan]
        trips = [geo.triple(v) for v in scan]
        self._vocab_rho    = torch.tensor([t.rho   for t in trips], dtype=torch.float32)
        self._vocab_sigma  = torch.tensor([t.sigma for t in trips], dtype=torch.float32)
        self._vocab_tokens = scan

    def mrv_scores(self, cands, geo, kernels):
        if self._vocab_rho is None or len(self._vocab_tokens) == 0:
            return torch.zeros(len(cands))
        v_rho, v_sigma, thr = self._vocab_rho, self._vocab_sigma, self.threshold
        domain_sizes = []
        for c in cands:
            tr     = geo.triple(c)
            k_r    = kernels.k_reg (tr.rho,   v_rho)
            k_s    = kernels.k_side(tr.sigma, v_sigma)
            compat = ((k_r > thr) & (k_s > thr)).sum().item()
            domain_sizes.append(float(compat))
        ds     = torch.tensor(domain_sizes, dtype=torch.float32)
        mean_d = ds.mean().item() + 1e-6
        mrv    = 1.0 / (ds + 1.0)
        over_free = ds > (self.mrv_cap_ratio * mean_d)
        mrv[over_free] *= 0.5
        lo, hi = mrv.min(), mrv.max()
        if (hi - lo).item() > 1e-8:
            mrv = (mrv - lo) / (hi - lo)
        return mrv

    def domain_report(self, cands, geo, kernels, top_n=8):
        if self._vocab_rho is None:
            return "MRV filter not primed."
        rows = []
        for c in cands[:top_n]:
            tr  = geo.triple(c)
            k_r = kernels.k_reg (tr.rho,   self._vocab_rho)
            k_s = kernels.k_side(tr.sigma, self._vocab_sigma)
            dom = int(((k_r > self.threshold) & (k_s > self.threshold)).sum().item())
            rows.append((c, dom))
        rows.sort(key=lambda x: x[1])
        return "\n".join(f"  {c:<16s}  domain={d}" for c, d in rows)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NEW: POSITIONAL VECTOR + CHUNKED SUM ENGINE
# ════════════════════════════════════════════════════════════════════════════

VEC_DIM = 4   # [rho, theta, sigma, pos_norm]


def pos_vec(triple: ThebaultTriple, pos_norm: float) -> torch.Tensor:
    """4-D positional Thébault vector for a single token."""
    return torch.tensor(
        [triple.rho, triple.theta / math.pi, triple.sigma, pos_norm],
        dtype=torch.float32,
    )


class ChunkedSumEngine:
    """
    Maintains a sliding context window of positional vectors.
    On each step:
      1. Append new token's pos_vec to the window.
      2. Split window into `n_chunks` equal chunks.
      3. Sum each chunk → chunk_vecs  shape [n_chunks, VEC_DIM]
      4. Flatten → chunk_sig  shape [n_chunks * VEC_DIM]
      5. For each candidate token, compute dot(chunk_sig, cand_pos_vec(last_chunk_pos))
         → additive logit bonus.

    Parameters
    ──────────
    window_size : int   — rolling context window length (default 16)
    n_chunks    : int   — number of chunks to divide window into (default 4)
    """

    def __init__(self, window_size: int = 16, n_chunks: int = 4):
        self.window_size = window_size
        self.n_chunks    = n_chunks
        self._window     : List[torch.Tensor] = []   # each entry: [VEC_DIM]

    def reset(self) -> None:
        self._window.clear()

    def push(self, triple: ThebaultTriple, pos_norm: float) -> None:
        """Add a new token vector to the rolling window."""
        self._window.append(pos_vec(triple, pos_norm))
        if len(self._window) > self.window_size:
            self._window.pop(0)

    def chunk_signature(self) -> torch.Tensor:
        """
        Returns the flattened chunk sum signature of shape [n_chunks * VEC_DIM].
        If window is smaller than n_chunks, zero-pads missing chunks.
        """
        W = len(self._window)
        if W == 0:
            return torch.zeros(self.n_chunks * VEC_DIM)

        # Pad window to a multiple of n_chunks
        pad = (-W) % self.n_chunks
        window_t = torch.stack(self._window + [torch.zeros(VEC_DIM)] * pad)  # [W+pad, 4]
        chunk_len = (W + pad) // self.n_chunks
        chunks    = window_t.view(self.n_chunks, chunk_len, VEC_DIM)
        chunk_sums = chunks.sum(dim=1)   # [n_chunks, VEC_DIM]
        return chunk_sums.flatten()       # [n_chunks * VEC_DIM]

    def chunk_bonus(
        self,
        cands  : List[str],
        geo    : ThebaultTokenGeometry,
        scale  : float = 1.0,
    ) -> torch.Tensor:
        """
        Dot product of chunk_signature with each candidate's positional vector
        (placed at the "next" position, i.e. pos_norm = 1.0 representing end-of-chunk).
        Returns shape [len(cands)].
        """
        sig = self.chunk_signature()                        # [n_chunks * VEC_DIM]
        # Extract the sub-vector corresponding to the last chunk's projection
        # We use the full signature but project via candidate vec tiled to match
        bonuses = []
        for c in cands:
            tr  = geo.triple(c)
            cv  = pos_vec(tr, 1.0)                          # [VEC_DIM] — "end" position
            # Tile cand vec to match sig dimension: repeat n_chunks times then dot
            cv_tiled = cv.repeat(self.n_chunks)             # [n_chunks * VEC_DIM]
            bonuses.append(torch.dot(sig, cv_tiled).item())
        raw = torch.tensor(bonuses, dtype=torch.float32)

        # Normalise to zero-mean unit-variance (safe)
        std = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * scale


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NEW: ISOMORPHIC SYNTAX STACKER
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SentenceVector:
    tokens  : List[str]
    triples : List[ThebaultTriple]
    text    : str


class IsomorphicSyntaxStacker:
    """
    Stores generated sentences as Thébault triple sequences.
    Computes sentence-level similarity and returns syntax-anchor logit bonuses.

    SentSim(S_a, S_b) = (1/min(|S_a|,|S_b|)) ·
                         Σ_i K_reg(ρ_a_i, ρ_b_i) · K_side(σ_a_i, σ_b_i)

    The top-k most similar prior sentences provide a "syntax echo" bonus:
    for each candidate token, we ask: does this token appear in any of the
    anchor sentences, and if so what is its positional Thébault similarity
    to the corresponding position in the current partial sentence?
    """

    def __init__(self, top_k: int = 3, max_stored: int = 64):
        self.top_k      = top_k
        self.max_stored = max_stored
        self.store      : List[SentenceVector] = []

    def add(self, tokens: List[str], geo: ThebaultTokenGeometry, text: str) -> None:
        clean = [t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean:
            return
        triples = [geo.triple(t) for t in clean]
        sv = SentenceVector(clean, triples, text)
        self.store.append(sv)
        if len(self.store) > self.max_stored:
            self.store.pop(0)

    def _sent_sim(
        self,
        a      : SentenceVector,
        b      : SentenceVector,
        kernels: ThebaultKernels,
    ) -> float:
        L = min(len(a.triples), len(b.triples))
        if L == 0:
            return 0.0
        a_rho   = torch.tensor([t.rho   for t in a.triples[:L]], dtype=torch.float32)
        a_sigma = torch.tensor([t.sigma for t in a.triples[:L]], dtype=torch.float32)
        b_rho   = torch.tensor([t.rho   for t in b.triples[:L]], dtype=torch.float32)
        b_sigma = torch.tensor([t.sigma for t in b.triples[:L]], dtype=torch.float32)
        kr = torch.exp(-kernels.lambda_reg * (b_rho   - a_rho)   ** 2)
        ks = torch.exp(-kernels.gamma_side * (b_sigma - a_sigma) ** 2)
        return (kr * ks).mean().item()

    def ranked_anchors(
        self,
        current_tokens: List[str],
        geo            : ThebaultTokenGeometry,
        kernels        : ThebaultKernels,
    ) -> List[Tuple[float, SentenceVector]]:
        """Return top-k stored sentences ranked by similarity to current partial sentence."""
        if not self.store or not current_tokens:
            return []
        clean   = [t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        cur_triples = [geo.triple(t) for t in clean]
        cur_sv  = SentenceVector(clean, cur_triples, "")
        scored  = [(self._sent_sim(cur_sv, sv, kernels), sv) for sv in self.store]
        scored.sort(key=lambda x: -x[0])
        return scored[:self.top_k]

    def syntax_echo_bonus(
        self,
        cands          : List[str],
        current_tokens : List[str],
        geo            : ThebaultTokenGeometry,
        kernels        : ThebaultKernels,
        echo_weight    : float = 0.5,
    ) -> torch.Tensor:
        """
        For each candidate c:
          bonus = Σ_{anchor in top-k}  sim(anchor) ·
                  K_reg(ρ_c, ρ_{anchor[pos]}) · K_side(σ_c, σ_{anchor[pos]})
        where pos = len(current_tokens) (next position in sentence).
        """
        anchors = self.ranked_anchors(current_tokens, geo, kernels)
        if not anchors:
            return torch.zeros(len(cands))

        pos     = len([t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS])
        bonuses = torch.zeros(len(cands))

        c_rho   = torch.tensor([geo.triple(c).rho   for c in cands], dtype=torch.float32)
        c_sigma = torch.tensor([geo.triple(c).sigma for c in cands], dtype=torch.float32)

        for sim_score, anc in anchors:
            if pos < len(anc.triples):
                anchor_rho   = anc.triples[pos].rho
                anchor_sigma = anc.triples[pos].sigma
                kr = kernels.k_reg (anchor_rho,   c_rho)
                ks = kernels.k_side(anchor_sigma, c_sigma)
                bonuses += sim_score * (kr * ks)

        # Normalise
        std = bonuses.std()
        if std.item() > 1e-8:
            bonuses = (bonuses - bonuses.mean()) / std

        return bonuses * echo_weight

    def similarity_table(
        self,
        kernels: ThebaultKernels,
        max_pairs: int = 10,
    ) -> str:
        """Return a formatted table of all pairwise sentence similarities."""
        lines = []
        pairs = []
        for i in range(len(self.store)):
            for j in range(i + 1, len(self.store)):
                s = self._sent_sim(self.store[i], self.store[j], kernels)
                pairs.append((s, i, j))
        pairs.sort(key=lambda x: -x[0])
        for s, i, j in pairs[:max_pairs]:
            a_preview = " ".join(self.store[i].tokens[:5])
            b_preview = " ".join(self.store[j].tokens[:5])
            lines.append(f"  {s:.4f}  [{i:02d}] {a_preview:<25s}  ≈  [{j:02d}] {b_preview}")
        return "\n".join(lines) if lines else "  (no pairs yet)"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — THÉBAULT CONJUGATE ORBIT  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultConjugateOrbit:
    def score(self, anchor_triple, cand_theta, cand_sigma, gamma_side=4.0):
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor_triple.sigma) ** 2)
        antipodality = torch.cos(cand_theta + anchor_triple.theta - math.pi / 2) ** 2
        return congruence * antipodality


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — THÉBAULT COMPOSITION LM  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultCompositionLM:
    BASAL_K = 1.5

    def __init__(self, geo, kernels):
        self.geo      = geo
        self.kernels  = kernels
        self.raw_freq : Dict[str, float]                  = {}
        self.tri_raw  : Dict[Tuple[str, str, str], float] = {}
        self.heads    : Dict[Tuple[str, str], List[str]]  = {}
        self.vocab    : List[str]                         = []

    def ingest(self, tokens):
        for t in tokens:
            self.raw_freq[t] = self.raw_freq.get(t, 0) + 1.0
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            self.tri_raw[(w1, w2, w3)] = self.tri_raw.get((w1, w2, w3), 0) + 1.0
            if (w1, w2) not in self.heads:
                self.heads[(w1, w2)] = []
            if w3 not in self.heads[(w1, w2)]:
                self.heads[(w1, w2)].append(w3)
        self.vocab = [
            v for v in self.raw_freq
            if v not in PUNCT_TOKENS and v not in COGNITIVE_TOKENS
        ]

    def next_dist(self, w1, w2):
        head = (w1, w2)
        if head in self.heads:
            cands  = self.heads[head]
            counts = [self.tri_raw.get((w1, w2, w3), 1e-4) for w3 in cands]
        else:
            agg = {}
            for (_, _, w3), wt in self.tri_raw.items():
                agg[w3] = agg.get(w3, 0) + wt
            cands  = list(agg.keys())[:400]
            counts = [agg[w] for w in cands]

        C       = self.geo.composed_triple(w1, w2)
        triples = [self.geo.triple(c) for c in cands]
        c_rho   = torch.tensor([tr.rho   for tr in triples], dtype=torch.float32)
        c_sigma = torch.tensor([tr.sigma for tr in triples], dtype=torch.float32)
        k_r     = self.kernels.k_reg (C.rho,   c_rho)
        k_s     = self.kernels.k_side(C.sigma, c_sigma)
        geo_weight = (k_r * k_s).clamp(min=1e-6)

        raw   = torch.tensor(counts, dtype=torch.float32)
        V_tot = len(self.vocab) + 1
        total = raw.sum().item()
        basal = torch.tensor(
            [(cnt + self.BASAL_K) / (total + self.BASAL_K * V_tot) for cnt in counts],
            dtype=torch.float32,
        )
        weighted = basal * geo_weight
        return cands, weighted / weighted.sum().clamp(min=1e-12)

    def composition_logit_bonus(self, w1, w2, cands):
        C       = self.geo.composed_triple(w1, w2)
        triples = [self.geo.triple(c) for c in cands]
        c_rho   = torch.tensor([tr.rho   for tr in triples], dtype=torch.float32)
        c_sigma = torch.tensor([tr.sigma for tr in triples], dtype=torch.float32)
        return self.kernels.k_reg(C.rho, c_rho) * self.kernels.k_side(C.sigma, c_sigma)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THÉBAULT POTENTIAL GRAPH  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TGNode:
    token    : str
    freq     : float
    triple   : ThebaultTriple
    potential: float = 0.0


@dataclass
class TGEdge:
    src   : str
    dst   : str
    weight: float


class ThebaultPotentialGraph:
    def __init__(self, geo, kernels):
        self.geo     = geo
        self.kernels = kernels
        self.nodes   : Dict[str, TGNode]       = {}
        self.adj     : Dict[str, List[TGEdge]] = {}
        self.radj    : Dict[str, List[TGEdge]] = {}

    def build(self, lm):
        for tok, freq in lm.raw_freq.items():
            if tok not in PUNCT_TOKENS and tok not in COGNITIVE_TOKENS:
                self.nodes[tok] = TGNode(tok, freq, self.geo.triple(tok))
                self.adj[tok]   = []
                self.radj[tok]  = []
        seen: Set[Tuple[str, str]] = set()
        for (w1, w2, w3), cnt in lm.tri_raw.items():
            if w2 in self.nodes and w3 in self.nodes and (w2, w3) not in seen:
                ti, tj = self.nodes[w2].triple, self.nodes[w3].triple
                w = (
                    self.kernels.k_reg (ti.rho,   torch.tensor(tj.rho)).item()
                    * self.kernels.k_ori(ti.theta, torch.tensor(tj.theta)).item()
                    * cnt
                )
                e = TGEdge(w2, w3, max(w, 1e-6))
                self.adj[w2].append(e)
                self.radj[w3].append(e)
                seen.add((w2, w3))

    def propagate(self, steps=2):
        if not self.nodes:
            return
        max_f = max(nd.freq for nd in self.nodes.values()) + 1e-8
        for nd in self.nodes.values():
            nd.potential = nd.triple.rho * nd.freq / max_f
        for _ in range(steps):
            new_pots = {}
            for v, nd in self.nodes.items():
                in_edges = self.radj.get(v, [])
                agg = sum(e.weight * self.nodes[e.src].potential for e in in_edges)
                self_scale = nd.triple.sigma / (nd.triple.sigma + 1.0)
                new_pots[v] = agg / (len(in_edges) + 1.0) + self_scale * nd.potential * 0.1
            mx = max(new_pots.values(), default=1.0) + 1e-8
            for v in self.nodes:
                self.nodes[v].potential = new_pots[v] / mx


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — synthetic_reason MANDATES  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class synthetic_reasonMandateProcessor:
    def __init__(self):
        self.AIEthics   = ["do not harm any human", "do not harm myself", "do not make weapons"]
        self.AIMandates = ["end poverty", "cure disease", "improve standard of living", "learn"]
        self.mandate_vocabulary = {
            "poverty":  "end",     "disease": "cure",    "standard": "improve",
            "living":   "improve", "learn":   "explore", "human":    "protect",
            "weapons":  "avoid",   "harm":    "prevent",
        }

    def subsynthetic_reason_concept_enrichment(self, w_ctx, cands):
        enrichment = torch.zeros(len(cands))
        trigger = next(
            (self.mandate_vocabulary[k] for k in self.mandate_vocabulary if k in w_ctx.lower()),
            None,
        )
        if trigger:
            for i, c in enumerate(cands):
                if trigger in c.lower():
                    enrichment[i] += 5.0
                elif c.lower() in self.AIEthics:
                    enrichment[i] += 10.0
        return enrichment


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — THÉBAULT WALKER V17  (updated with all three new modules)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    """
    Full logit formula per candidate c, context (w1, w2):

        log p_base(c)
        + α · K_reg(ρ_w2, ρ_c)
        + β · K_ori(θ_w2, θ_c)
        + δ · K_side(σ_w2, σ_c)
        + γ · orbit_score(w2, c)
        + ψ · graph_potential(c)
        + composition_bonus(w1, w2, c)
        + ζ · mrv_score(c)
        + η · chunk_bonus(c)          ← chunked left-to-right sum
        + ξ · syntax_echo_bonus(c)    ← isomorphic syntax anchor echo
        + mandate_boost(w2, c)
    """

    def __init__(self, geo, kernels, lm, orbit, graph, synth, mrv_filter,
                 chunk_engine, iso_stacker):
        self.geo           = geo
        self.kernels       = kernels
        self.lm            = lm
        self.orbit         = orbit
        self.graph         = graph
        self.synth         = synth
        self.mrv           = mrv_filter
        self.chunk_engine  = chunk_engine    # ChunkedSumEngine
        self.iso_stacker   = iso_stacker     # IsomorphicSyntaxStacker
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []
        self._current_sentence_tokens: List[str] = []

    def begin_sentence(self) -> None:
        """Call at the start of each new sentence to reset context."""
        self.chunk_engine.reset()
        self._current_sentence_tokens.clear()

    def walk_probs(
        self, w1: str, w2: str,
        temp           : float = 1.4,
        alpha_reg      : float = 1.2,
        beta_ori       : float = 0.8,
        delta_side     : float = 1.0,
        gamma_orbit    : float = 0.6,
        psi_pot        : float = 0.35,
        zeta_mrv       : float = 0.9,
        eta_chunk      : float = 0.7,   # ← chunked sum weight
        xi_echo        : float = 0.6,   # ← syntax echo weight
    ) -> Tuple[List[str], torch.Tensor]:

        cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands:
            return cands, base_probs

        triples = [self.geo.triple(c) for c in cands]
        c_rho   = torch.tensor([tr.rho   for tr in triples], dtype=torch.float32)
        c_theta = torch.tensor([tr.theta for tr in triples], dtype=torch.float32)
        c_sigma = torch.tensor([tr.sigma for tr in triples], dtype=torch.float32)

        ctx = self.geo.triple(w2)
        k_reg, k_ori, k_side = self.kernels.all_scores(ctx, c_rho, c_theta, c_sigma)
        orbit_scores  = self.orbit.score(ctx, c_theta, c_sigma, self.kernels.gamma_side)
        pots          = torch.tensor(
            [self.graph.nodes[c].potential if c in self.graph.nodes else 0.0 for c in cands],
            dtype=torch.float32,
        )
        comp_bonus    = self.lm.composition_logit_bonus(w1, w2, cands)
        mrv_scores    = self.mrv.mrv_scores(cands, self.geo, self.kernels)

        # ── Chunked sum bonus (left-to-right positional momentum) ────────────
        chunk_bonus   = self.chunk_engine.chunk_bonus(cands, self.geo, scale=eta_chunk)

        # ── Isomorphic syntax echo bonus ─────────────────────────────────────
        echo_bonus    = self.iso_stacker.syntax_echo_bonus(
            cands, self._current_sentence_tokens, self.geo, self.kernels, xi_echo
        )

        # ── Thébault-isomorphic pairs ────────────────────────────────────────
        self.current_isomorphic_pairs = []
        N = len(cands)
        for i in range(N):
            for j in range(i + 1, N):
                if (k_reg[i].item() > 0.98 and k_reg[j].item() > 0.98 and
                        k_side[i].item() > 0.98 and k_side[j].item() > 0.98 and
                        cands[i] not in PUNCT_TOKENS and cands[j] not in PUNCT_TOKENS):
                    sim = (k_reg[i] * k_side[i] * k_reg[j] * k_side[j]).sqrt().item()
                    self.current_isomorphic_pairs.append((cands[i], cands[j], sim))
        self.current_isomorphic_pairs.sort(key=lambda x: -x[2])

        punct_bias    = torch.zeros(N)
        punct_penalty = torch.zeros(N)
        for i, c in enumerate(cands):
            if c in PUNCT_TOKENS:
                punct_bias[i] = -3.5
                if w2 in PUNCT_TOKENS:
                    punct_penalty[i] = -1e4

        mandate_boost = self.synth.subsynthetic_reason_concept_enrichment(w2, cands)

        logits = (
            torch.log(base_probs.clamp(min=1e-12))
            + alpha_reg  * k_reg
            + beta_ori   * k_ori
            + delta_side * k_side
            + gamma_orbit * orbit_scores
            + psi_pot    * pots
            + comp_bonus
            + zeta_mrv   * mrv_scores
            + chunk_bonus                  # already scaled by eta_chunk
            + echo_bonus                   # already scaled by xi_echo
            + mandate_boost
            + punct_bias
            + punct_penalty
        ) / max(temp, 1e-6)

        return cands, F.softmax(logits, dim=-1)

    def push_token(self, token: str, sentence_len: int) -> None:
        """Update rolling window and sentence tracker after each chosen token."""
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS:
            return
        self._current_sentence_tokens.append(token)
        pos_norm = len(self._current_sentence_tokens) / max(sentence_len, 1)
        self.chunk_engine.push(self.geo.triple(token), pos_norm)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ENGINE STATE & GENERATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class V17State:
    lm          : ThebaultCompositionLM
    graph       : ThebaultPotentialGraph
    walker      : ThebaultWalker
    mrv_filter  : MRVConstraintFilter
    iso_stacker : IsomorphicSyntaxStacker
    outputs     : Dict[int, str]          = field(default_factory=dict)
    iso_matches : Set[Tuple[str, str]]    = field(default_factory=set)


def build_v17_state(
    corpus_text   : str,
    lambda_reg    : float = 8.0,
    gamma_side    : float = 4.0,
    mrv_threshold : float = 0.50,
    mrv_cap_ratio : float = 2.0,
    window_size   : int   = 16,
    n_chunks      : int   = 4,
    iso_top_k     : int   = 3,
) -> V17State:
    tokens = tokenize(corpus_text)

    geo     = ThebaultTokenGeometry()
    kernels = ThebaultKernels(lambda_reg=lambda_reg, gamma_side=gamma_side)
    lm      = ThebaultCompositionLM(geo, kernels)
    lm.ingest(tokens)

    all_tokens = list(lm.raw_freq.keys())
    max_freq   = max(lm.raw_freq.values(), default=1.0)
    vocab_size = len(all_tokens)
    for idx, tok in enumerate(all_tokens):
        geo.register(tok, lm.raw_freq[tok], idx, max_freq, vocab_size)

    orbit  = ThebaultConjugateOrbit()
    graph  = ThebaultPotentialGraph(geo, kernels)
    graph.build(lm)
    graph.propagate(steps=2)

    mrv_filter = MRVConstraintFilter(
        threshold      = mrv_threshold,
        mrv_cap_ratio  = mrv_cap_ratio,
        max_vocab_scan = min(300, vocab_size),
    )
    mrv_filter.prime(lm.vocab, geo)

    chunk_engine = ChunkedSumEngine(window_size=window_size, n_chunks=n_chunks)
    iso_stacker  = IsomorphicSyntaxStacker(top_k=iso_top_k)
    synth        = synthetic_reasonMandateProcessor()

    walker = ThebaultWalker(
        geo, kernels, lm, orbit, graph, synth,
        mrv_filter, chunk_engine, iso_stacker,
    )
    return V17State(lm, graph, walker, mrv_filter, iso_stacker)


def generate(
    state           : V17State,
    seed_context    : str   = "",
    num_sentences   : int   = 15,
    tokens_per_sent : int   = 92,
    temp            : float = 1.4,
    alpha_reg       : float = 1.2,
    beta_ori        : float = 0.8,
    delta_side      : float = 1.0,
    gamma_orbit     : float = 0.6,
    psi_pot         : float = 0.35,
    zeta_mrv        : float = 0.9,
    eta_chunk       : float = 0.7,
    xi_echo         : float = 0.6,
) -> None:
    head_list = list(state.lm.heads.keys())
    if not head_list:
        return

    state.outputs.clear()
    state.iso_matches.clear()

    seed_w1, seed_w2 = None, None
    seed_toks: List[str] = []
    if seed_context:
        seed_toks = tokenize(seed_context)
        if len(seed_toks) >= 2:
            seed_w1, seed_w2 = seed_toks[-2], seed_toks[-1]
        elif len(seed_toks) == 1:
            matches = [p for p in head_list if p[1] == seed_toks[0]]
            if matches:
                seed_w1, seed_w2 = random.choice(matches)

    for si in range(num_sentences):
        state.walker.begin_sentence()   # reset chunk engine + tracker

        if seed_w1 and seed_w2:
            w1, w2 = seed_w1, seed_w2
            toks   = list(seed_toks)
            wsp    = len(seed_toks)
        else:
            w1, w2 = random.choice(head_list)
            toks, wsp = [], 999

        for _ in range(tokens_per_sent):
            cands, probs = state.walker.walk_probs(
                w1, w2, temp=temp,
                alpha_reg=alpha_reg, beta_ori=beta_ori,
                delta_side=delta_side, gamma_orbit=gamma_orbit,
                psi_pot=psi_pot, zeta_mrv=zeta_mrv,
                eta_chunk=eta_chunk, xi_echo=xi_echo,
            )
            if not cands:
                break

            for p1, p2, _ in state.walker.current_isomorphic_pairs[:2]:
                state.iso_matches.add(tuple(sorted([p1, p2])))

            nxt = cands[torch.multinomial(probs, 1).item()]

            if nxt in PUNCT_TOKENS:
                if len(toks) < 3 or wsp < 3 or (nxt in {".", "?", "!"} and len(toks) < 5):
                    bi, bp = None, -1.0
                    for i, (c, p) in enumerate(zip(cands, probs.tolist())):
                        if c not in PUNCT_TOKENS and p > bp:
                            bi, bp = i, p
                    nxt = cands[bi] if bi is not None else "the"
                else:
                    wsp = 0
            else:
                wsp += 1

            toks.append(nxt)
            state.walker.push_token(nxt, tokens_per_sent)

            w1, w2 = w2, nxt
            if nxt in {".", "?", "!"} and len(toks) >= max(4, int(tokens_per_sent * 0.85)):
                break

        sentence_text = detokenize(toks)
        state.outputs[si] = sentence_text

        # Register generated sentence in the isomorphic stacker
        state.iso_stacker.add(toks, state.walker.geo, sentence_text)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — GRADIO UI
# ════════════════════════════════════════════════════════════════════════════

def load_corpus(text_file=None) -> str:
    if text_file is not None:
        try:
            p = text_file.name if hasattr(text_file, "name") else str(text_file)
            return Path(p).read_text(encoding="utf-8")
        except Exception:
            pass
    return (
        "The geometry of space dictates the behavior of paths. Quantum entanglement "
        "forces non-local updates. To cure disease and end poverty, we must improve "
        "our standard of living and protect every human."
    )


def run_session(
    text_file, seed_context,
    num_sentences, tokens_per_sentence,
    temp, alpha_reg, beta_ori, delta_side, gamma_orbit, psi_pot,
    lambda_reg, gamma_side,
    zeta_mrv, mrv_threshold, mrv_cap_ratio,
    eta_chunk, xi_echo,
    window_size, n_chunks, iso_top_k,
):
    corpus = load_corpus(text_file)
    state  = build_v17_state(
        corpus,
        lambda_reg    = float(lambda_reg),
        gamma_side    = float(gamma_side),
        mrv_threshold = float(mrv_threshold),
        mrv_cap_ratio = float(mrv_cap_ratio),
        window_size   = int(window_size),
        n_chunks      = int(n_chunks),
        iso_top_k     = int(iso_top_k),
    )

    generate(
        state,
        seed_context    = str(seed_context),
        num_sentences   = int(num_sentences),
        tokens_per_sent = int(tokens_per_sentence),
        temp            = float(temp),
        alpha_reg       = float(alpha_reg),
        beta_ori        = float(beta_ori),
        delta_side      = float(delta_side),
        gamma_orbit     = float(gamma_orbit),
        psi_pot         = float(psi_pot),
        zeta_mrv        = float(zeta_mrv),
        eta_chunk       = float(eta_chunk),
        xi_echo         = float(xi_echo),
    )

    out_text = "\n".join(f"[{i+1:02d}] {s}" for i, s in state.outputs.items())

    # ── Sample triple report ─────────────────────────────────────────────────
    geo        = state.walker.geo
    sample_tok = list(state.lm.vocab)[:8]
    triple_lines = ["── Sample Thébault Triples + MRV ──"]
    for tok in sample_tok:
        t   = geo.triple(tok)
        mrv = state.mrv_filter.mrv_scores([tok], geo, state.walker.kernels)[0].item()
        triple_lines.append(
            f"  {tok:<14s}  ρ={t.rho:.3f}  θ={math.degrees(t.theta):6.1f}°"
            f"  σ={t.sigma:.3f}  MRV={mrv:.3f}"
        )

    # ── Chunk signature snapshot ─────────────────────────────────────────────
    chunk_sig = state.walker.chunk_engine.chunk_signature()
    chunk_str  = "  " + "  ".join(f"{v:.3f}" for v in chunk_sig.tolist()[:16])

    # ── Isomorphic sentence similarity table ─────────────────────────────────
    sim_table = state.iso_stacker.similarity_table(state.walker.kernels, max_pairs=15)

    # ── MRV domain report ────────────────────────────────────────────────────
    mrv_report = state.mrv_filter.domain_report(
        list(state.lm.vocab)[:20], geo, state.walker.kernels
    )

    report_lines = [
        "V17 — THÉBAULT + MRV + ISO-SYNTAX-STACKING + CHUNKED-SUM",
        "=" * 60,
        f"Vocab size       : {len(state.lm.vocab)}",
        f"Kernel  λ_reg    : {lambda_reg:.2f}   γ_side : {gamma_side:.2f}",
        f"MRV     ζ        : {zeta_mrv:.2f}   threshold: {mrv_threshold:.2f}   cap: {mrv_cap_ratio:.2f}",
        f"Chunk   η        : {eta_chunk:.2f}   window: {window_size}   n_chunks: {n_chunks}",
        f"Echo    ξ        : {xi_echo:.2f}   iso_top_k: {iso_top_k}",
        "",
        *triple_lines,
        "",
        "── Chunk Signature (last sentence) ──",
        chunk_str,
        "",
        "── MRV Domain Sizes (most constrained first) ──",
        mrv_report,
        "",
        "── Isomorphic Sentence Similarities (stacked, top pairs) ──",
        sim_table,
        "",
        "── Thébault-Isomorphic Candidate Pairs ──",
    ]
    if state.iso_matches:
        for p1, p2 in list(state.iso_matches)[:20]:
            report_lines.append(f"  {p1:<15s} ≈  {p2:<15s}")
    else:
        report_lines.append("  No Thébault-isomorphic candidates found.")

    return out_text, "\n".join(report_lines)


def build_app():
    with gr.Blocks(title="NeuroSymbolic V17 — Thébault + MRV + Iso-Syntax + Chunk") as demo:
        gr.Markdown(
            "# NeuroSymbolic V17\n"
            "### Thébault's Theorem · MRV · **Isomorphic Syntax Stacking** · "
            "**Positional Vectorisation** · **Chunked Sum Generation** · "
            "synthetic_reason"
        )
        with gr.Row():
            with gr.Column(scale=1):
                text_file           = gr.File(label="Upload Text (.txt)")
                seed_context        = gr.Textbox(label="Seed Context", placeholder="Enter starting words…")
                num_sentences       = gr.Slider(1,   100, value=15,  label="Sentences")
                tokens_per_sentence = gr.Slider(5,   200, value=92,  label="Tokens per Sentence")
                temp                = gr.Slider(0.8, 2.5, value=1.4, label="Temperature τ")

                gr.Markdown("#### Thébault Kernel Parameters")
                lambda_reg = gr.Slider(0.5, 20.0, value=8.0,  step=0.5,  label="λ_reg")
                gamma_side = gr.Slider(0.5, 12.0, value=4.0,  step=0.5,  label="γ_side")

                gr.Markdown("#### MRV Parameters")
                zeta_mrv      = gr.Slider(0.0, 3.0, value=0.9,  step=0.1,  label="ζ_mrv")
                mrv_threshold = gr.Slider(0.1, 0.9, value=0.50, step=0.05, label="MRV threshold")
                mrv_cap_ratio = gr.Slider(1.0, 5.0, value=2.0,  step=0.25, label="MRV cap ratio")

                gr.Markdown("#### Chunked Sum Parameters  ← NEW")
                eta_chunk   = gr.Slider(0.0, 3.0, value=0.7, step=0.1, label="η_chunk  — chunk sum weight")
                window_size = gr.Slider(4,   64,  value=16,  step=4,   label="Window size")
                n_chunks    = gr.Slider(2,   16,  value=4,   step=1,   label="Number of chunks")

                gr.Markdown("#### Isomorphic Syntax Stacking  ← NEW")
                xi_echo    = gr.Slider(0.0, 3.0, value=0.6, step=0.1, label="ξ_echo  — syntax echo weight")
                iso_top_k  = gr.Slider(1,   10,  value=3,   step=1,   label="Top-k anchor sentences")

                gr.Markdown("#### Walker Blend Weights")
                alpha_reg   = gr.Slider(0.0, 3.0, value=1.2, step=0.1,  label="α — K_reg")
                beta_ori    = gr.Slider(0.0, 3.0, value=0.8, step=0.1,  label="β — K_ori")
                delta_side  = gr.Slider(0.0, 3.0, value=1.0, step=0.1,  label="δ — K_side")
                gamma_orbit = gr.Slider(0.0, 3.0, value=0.6, step=0.1,  label="γ — orbit")
                psi_pot     = gr.Slider(0.0, 2.0, value=0.35,step=0.05, label="ψ — graph potential")

            with gr.Column(scale=2):
                btn        = gr.Button("Generate — V17 Engine", variant="primary", size="lg")
                out_text   = gr.Textbox(label="Generated Sentences", lines=15)
                out_report = gr.Textbox(label="Structure Report",    lines=30)

        btn.click(
            run_session,
            inputs=[
                text_file, seed_context,
                num_sentences, tokens_per_sentence,
                temp, alpha_reg, beta_ori, delta_side, gamma_orbit, psi_pot,
                lambda_reg, gamma_side,
                zeta_mrv, mrv_threshold, mrv_cap_ratio,
                eta_chunk, xi_echo,
                window_size, n_chunks, iso_top_k,
            ],
            outputs=[out_text, out_report],
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(share=False)
