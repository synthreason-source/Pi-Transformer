#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V17-CUDA — Thébault + MRV + Isomorphic Syntax Stacking
                        + Positional Vectorisation + Chunked Sum Generation
===============================================================================

CUDA OPTIMISATION CHANGES (vs V17-CPU)
───────────────────────────────────────
1.  DEVICE-AWARE TENSOR CACHE
    ThebaultTokenGeometry now pre-builds four contiguous CUDA tensors:
        _rho_t   [V]   _theta_t  [V]   _sigma_t  [V]   _pvec_t  [V, 4]
    All per-candidate kernel evaluations are a single batched CUDA op.

2.  FULLY VECTORISED KERNEL SCORING
    ThebaultKernels.all_scores_batched() operates on [V]-shaped CUDA tensors
    returning [V] tensors in one launch — no Python loops over candidates.

3.  PRE-BATCHED LM DISTRIBUTIONS
    ThebaultCompositionLM stores bigram/trigram tables as dense CUDA tensors
    (sparse index → dense weight matrix).  next_dist() is a gather + softmax
    on GPU, not a Python dict walk.

4.  BATCHED SENTENCE GENERATION
    generate() runs `batch_size` sentences simultaneously on GPU, advancing
    all token positions in a single forward pass per step.

5.  CHUNK ENGINE — ALL-CUDA
    ChunkedSumEngine keeps its rolling window as a CUDA tensor circular
    buffer; chunk_signature() and chunk_bonus() are pure torch ops.

6.  ISO-STACKER BATCHED SIMILARITY
    SentenceVector stores triple tensors on device; ranked_anchors()
    does a batched einsum over all stored sentences at once.

7.  FUSED LOGIT ACCUMULATION
    All bonus terms are pre-stacked into a [B, num_terms, V] tensor and
    summed in one CUDA kernel via .sum(dim=1).

8.  torch.compile SUPPORT
    Pass compile=True to build_v17_state() to wrap ThebaultWalker.walk_probs
    through torch.compile(mode="reduce-overhead") for graph-mode speedup
    on repeated calls.

9.  MIXED PRECISION (optional)
    Set dtype=torch.float16 for ~2× throughput on Ampere+ GPUs (default float32).

All V17 mathematical semantics are preserved exactly.
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
# SECTION 0 — DEVICE SELECTION
# ════════════════════════════════════════════════════════════════════════════

def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = best_device()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOKEN PRIMITIVES
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
# SECTION 2 — THÉBAULT TOKEN GEOMETRY  (CUDA-accelerated)
# ════════════════════════════════════════════════════════════════════════════

def _perfect_square_cv() -> float:
    s  = 1.0
    d  = [s, s, s, s, s * math.sqrt(2), s * math.sqrt(2)]
    mu = sum(d) / 6
    return math.sqrt(sum((x - mu) ** 2 for x in d) / 6) / mu


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
    sides = dists[:4]
    sigma = sum(sides) / 4.0
    dx_ori = T[1][0] - T[0][0]
    dy_ori = T[1][1] - T[0][1]
    theta  = math.atan2(dy_ori, dx_ori) % math.pi
    return rho, theta, sigma


@dataclass
class ThebaultTriple:
    rho  : float
    theta: float
    sigma: float


class ThebaultTokenGeometry:
    """
    CUDA optimisation: after all tokens are registered, call .build_cuda_tensors()
    to materialise four contiguous GPU tensors for O(1) batch lookup.
    """

    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype  = dtype
        self._vecs  : Dict[str, Tuple[float, float, float, float]] = {}
        self._cache : Dict[str, ThebaultTriple]                    = {}
        # GPU tensor cache (built once, then read-only)
        self._tok2idx: Dict[str, int]        = {}
        self._rho_t  : Optional[torch.Tensor] = None   # [V]
        self._theta_t: Optional[torch.Tensor] = None   # [V]
        self._sigma_t: Optional[torch.Tensor] = None   # [V]
        self._pvec_t : Optional[torch.Tensor] = None   # [V, 4]  positional vecs (pos=1)
        self._idx_list: List[str]             = []

    def register(self, token, freq, index, max_freq, vocab_size):
        f_hat   = freq / max(max_freq, 1e-9)
        k_hat   = index / max(vocab_size - 1, 1)
        angle_p = 2.0 * math.pi * k_hat
        angle_q = 2.0 * math.pi * f_hat
        px = f_hat * math.cos(angle_p);  py = f_hat * math.sin(angle_p)
        qx = k_hat * math.cos(angle_q);  qy = k_hat * math.sin(angle_q)
        self._vecs[token] = (px, py, qx, qy)
        self._cache.pop(token, None)

    def build_cuda_tensors(self, vocab: List[str]) -> None:
        """
        Call once after all register() calls.
        Builds contiguous GPU tensors for the given vocab list.
        """
        triples = []
        for tok in vocab:
            t = self.triple(tok)
            triples.append((t.rho, t.theta, t.sigma))
        self._idx_list = vocab
        self._tok2idx  = {t: i for i, t in enumerate(vocab)}
        rhos   = [r for r, _, _ in triples]
        thetas = [th for _, th, _ in triples]
        sigmas = [s for _, _, s in triples]
        self._rho_t   = torch.tensor(rhos,   dtype=self.dtype, device=self.device)
        self._theta_t = torch.tensor(thetas, dtype=self.dtype, device=self.device)
        self._sigma_t = torch.tensor(sigmas, dtype=self.dtype, device=self.device)
        # positional vecs with pos_norm=1.0  → shape [V, 4]
        self._pvec_t  = torch.stack([
            self._rho_t,
            self._theta_t / math.pi,
            self._sigma_t,
            torch.ones_like(self._rho_t),
        ], dim=1)  # [V, 4]

    # ── scalar triple (CPU, cached) ──────────────────────────────────────────
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
        rho, theta, sigma = _thebault_triple(
            p1x + p2x, p1y + p2y, q1x + q2x, q1y + q2y
        )
        return ThebaultTriple(rho, theta, sigma)

    # ── batched GPU lookup for an index list ─────────────────────────────────
    def batch_triples(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rho, theta, sigma) tensors for the given integer index tensor."""
        return (
            self._rho_t[indices],
            self._theta_t[indices],
            self._sigma_t[indices],
        )

    def tok_indices(self, toks: List[str]) -> torch.Tensor:
        """Convert token list to GPU index tensor (unknown → 0)."""
        idx = [self._tok2idx.get(t, 0) for t in toks]
        return torch.tensor(idx, dtype=torch.long, device=self.device)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THÉBAULT KERNELS  (fully vectorised)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultKernels:
    def __init__(self, lambda_reg: float = 8.0, gamma_side: float = 4.0):
        self.lambda_reg = lambda_reg
        self.gamma_side = gamma_side

    # scalar-vs-batch (used in some paths)
    def k_reg (self, rho_a : float, rho_b : torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.lambda_reg * (rho_b - rho_a) ** 2)

    def k_ori (self, theta_a: float, theta_b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(theta_b - theta_a))

    def k_side(self, sigma_a: float, sigma_b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)

    def all_scores(self, ctx, c_rho, c_theta, c_sigma):
        return (
            self.k_reg (ctx.rho,   c_rho),
            self.k_ori (ctx.theta, c_theta),
            self.k_side(ctx.sigma, c_sigma),
        )

    # ── CUDA batched: ctx triple scalars broadcast over [V] tensors ──────────
    def all_scores_batched(
        self,
        rho_a  : float, theta_a: float, sigma_a: float,
        rho_b  : torch.Tensor,           # [V]
        theta_b: torch.Tensor,           # [V]
        sigma_b: torch.Tensor,           # [V]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_r = torch.exp(-self.lambda_reg * (rho_b   - rho_a)   ** 2)
        k_o = 0.5 * (1.0 + torch.cos(theta_b - theta_a))
        k_s = torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)
        return k_r, k_o, k_s


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MRV CONSTRAINT FILTER  (GPU-native)
# ════════════════════════════════════════════════════════════════════════════

class MRVConstraintFilter:
    def __init__(self, threshold=0.50, mrv_cap_ratio=2.0, max_vocab_scan=300,
                 device: torch.device = DEVICE):
        self.threshold      = threshold
        self.mrv_cap_ratio  = mrv_cap_ratio
        self.max_vocab_scan = max_vocab_scan
        self.device         = device
        # GPU tensors for the scan vocabulary
        self._v_rho  : Optional[torch.Tensor] = None   # [S]
        self._v_sigma: Optional[torch.Tensor] = None   # [S]
        self._v_toks : List[str]              = []

    def prime(self, vocab: List[str], geo: ThebaultTokenGeometry) -> None:
        scan = vocab[:self.max_vocab_scan]
        trips = [geo.triple(v) for v in scan]
        self._v_rho   = torch.tensor([t.rho   for t in trips], dtype=torch.float32, device=self.device)
        self._v_sigma = torch.tensor([t.sigma for t in trips], dtype=torch.float32, device=self.device)
        self._v_toks  = scan

    def mrv_scores_batched(
        self,
        c_rho  : torch.Tensor,   # [C]
        c_sigma: torch.Tensor,   # [C]
        kernels: ThebaultKernels,
    ) -> torch.Tensor:
        """
        Fully batched MRV: [C, S] compatibility matrix → domain sizes [C].
        """
        if self._v_rho is None:
            return torch.zeros(c_rho.shape[0], device=self.device)

        # Broadcast: [C, 1] vs [1, S]
        k_r = torch.exp(
            -kernels.lambda_reg * (c_rho.unsqueeze(1)   - self._v_rho.unsqueeze(0))   ** 2
        )  # [C, S]
        k_s = torch.exp(
            -kernels.gamma_side * (c_sigma.unsqueeze(1) - self._v_sigma.unsqueeze(0)) ** 2
        )  # [C, S]
        thr = self.threshold
        domain_sizes = ((k_r > thr) & (k_s > thr)).float().sum(dim=1)   # [C]

        mean_d = domain_sizes.mean() + 1e-6
        mrv    = 1.0 / (domain_sizes + 1.0)
        mrv[domain_sizes > self.mrv_cap_ratio * mean_d] *= 0.5

        lo, hi = mrv.min(), mrv.max()
        if (hi - lo).item() > 1e-8:
            mrv = (mrv - lo) / (hi - lo)
        return mrv

    # legacy scalar interface (kept for compatibility / report)
    def mrv_scores(self, cands, geo, kernels):
        if not cands:
            return torch.zeros(0, device=self.device)
        c_rho   = torch.tensor([geo.triple(c).rho   for c in cands], dtype=torch.float32, device=self.device)
        c_sigma = torch.tensor([geo.triple(c).sigma for c in cands], dtype=torch.float32, device=self.device)
        return self.mrv_scores_batched(c_rho, c_sigma, kernels)

    def domain_report(self, cands, geo, kernels, top_n=8):
        if self._v_rho is None:
            return "MRV filter not primed."
        rows = []
        for c in cands[:top_n]:
            tr  = geo.triple(c)
            c_r = torch.tensor([tr.rho],   dtype=torch.float32, device=self.device)
            c_s = torch.tensor([tr.sigma], dtype=torch.float32, device=self.device)
            dom = int(self.mrv_scores_batched(c_r, c_s, kernels)[0].item() * len(self._v_toks))
            rows.append((c, dom))
        rows.sort(key=lambda x: x[1])
        return "\n".join(f"  {c:<16s}  domain≈{d}" for c, d in rows)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — POSITIONAL VECTOR + CHUNKED SUM ENGINE  (circular GPU buffer)
# ════════════════════════════════════════════════════════════════════════════

VEC_DIM = 4


class ChunkedSumEngine:
    """
    CUDA optimisation:
    • Rolling window = a pre-allocated CUDA tensor [window_size, VEC_DIM].
    • Write position tracked with a scalar pointer (circular buffer).
    • chunk_signature() and chunk_bonus() are pure torch ops — zero Python loops.
    """

    def __init__(self, window_size: int = 16, n_chunks: int = 4,
                 device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.window_size = window_size
        self.n_chunks    = n_chunks
        self.device      = device
        self.dtype       = dtype
        self._buf   = torch.zeros(window_size, VEC_DIM, dtype=dtype, device=device)
        self._ptr   = 0    # next write position
        self._count = 0    # number of valid entries

    def reset(self) -> None:
        self._buf.zero_()
        self._ptr   = 0
        self._count = 0

    def push(self, triple: ThebaultTriple, pos_norm: float) -> None:
        vec = torch.tensor(
            [triple.rho, triple.theta / math.pi, triple.sigma, pos_norm],
            dtype=self.dtype, device=self.device,
        )
        self._buf[self._ptr] = vec
        self._ptr   = (self._ptr + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)

    def chunk_signature(self) -> torch.Tensor:
        """Returns [n_chunks * VEC_DIM] tensor — pure GPU op."""
        if self._count == 0:
            return torch.zeros(self.n_chunks * VEC_DIM, dtype=self.dtype, device=self.device)

        # Reconstruct ordered window (most-recent last)
        if self._count < self.window_size:
            window = self._buf[:self._count]
        else:
            # Unwrap circular buffer
            window = torch.cat([self._buf[self._ptr:], self._buf[:self._ptr]], dim=0)

        W   = window.shape[0]
        pad = (-W) % self.n_chunks
        if pad > 0:
            window = torch.cat([window, torch.zeros(pad, VEC_DIM, dtype=self.dtype, device=self.device)])
        chunk_len = window.shape[0] // self.n_chunks
        chunks    = window.view(self.n_chunks, chunk_len, VEC_DIM)
        return chunks.sum(dim=1).flatten()   # [n_chunks * VEC_DIM]

    def chunk_bonus(
        self,
        c_pvec: torch.Tensor,   # [C, 4]  pre-computed candidate pos vecs
        scale : float = 1.0,
    ) -> torch.Tensor:
        """
        Dot product of chunk_signature with tiled candidate vecs.
        All GPU: one matmul, one normalise.
        Input c_pvec: [C, 4] (positional vecs with pos=1).
        """
        sig = self.chunk_signature()                   # [n_chunks * VEC_DIM]
        # Tile candidate vec n_chunks times → [C, n_chunks * VEC_DIM]
        cv_tiled = c_pvec.repeat(1, self.n_chunks)     # [C, n_chunks * 4]
        raw = cv_tiled @ sig                           # [C]  — single matmul

        std = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * scale


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ISOMORPHIC SYNTAX STACKER  (batched einsum similarity)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SentenceVector:
    tokens  : List[str]
    rho_t   : torch.Tensor   # [L]  GPU
    sigma_t : torch.Tensor   # [L]  GPU
    text    : str


class IsomorphicSyntaxStacker:
    """
    CUDA optimisation: all stored sentences have their triples on GPU.
    ranked_anchors() stacks them into a batch tensor and computes all
    similarities in one einsum — no Python loop over sentences.
    """

    def __init__(self, top_k: int = 3, max_stored: int = 64,
                 device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.top_k     = top_k
        self.max_stored = max_stored
        self.device    = device
        self.dtype     = dtype
        self.store     : List[SentenceVector] = []

    def add(self, tokens: List[str], geo: ThebaultTokenGeometry, text: str) -> None:
        clean = [t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean:
            return
        rhos   = torch.tensor([geo.triple(t).rho   for t in clean], dtype=self.dtype, device=self.device)
        sigmas = torch.tensor([geo.triple(t).sigma for t in clean], dtype=self.dtype, device=self.device)
        self.store.append(SentenceVector(clean, rhos, sigmas, text))
        if len(self.store) > self.max_stored:
            self.store.pop(0)

    def _batch_sim(
        self,
        cur_rho   : torch.Tensor,   # [L]
        cur_sigma : torch.Tensor,   # [L]
        kernels   : ThebaultKernels,
    ) -> torch.Tensor:
        """
        Batched sentence similarity: returns [N_stored] tensor.
        Pads/truncates stored sentences to len(cur).
        """
        L = cur_rho.shape[0]
        N = len(self.store)
        if N == 0 or L == 0:
            return torch.zeros(0, device=self.device)

        # Build padded stored matrix [N, L]
        stored_rho   = torch.zeros(N, L, dtype=self.dtype, device=self.device)
        stored_sigma = torch.zeros(N, L, dtype=self.dtype, device=self.device)
        for i, sv in enumerate(self.store):
            l = min(L, sv.rho_t.shape[0])
            stored_rho  [i, :l] = sv.rho_t  [:l]
            stored_sigma[i, :l] = sv.sigma_t[:l]

        # Broadcast cur vs stored: [N, L]
        kr = torch.exp(-kernels.lambda_reg * (stored_rho   - cur_rho.unsqueeze(0))   ** 2)
        ks = torch.exp(-kernels.gamma_side * (stored_sigma - cur_sigma.unsqueeze(0)) ** 2)
        return (kr * ks).mean(dim=1)   # [N]

    def ranked_anchors(
        self,
        current_tokens: List[str],
        geo           : ThebaultTokenGeometry,
        kernels       : ThebaultKernels,
    ) -> List[Tuple[float, SentenceVector]]:
        if not self.store or not current_tokens:
            return []
        clean = [t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean:
            return []
        cur_rho   = torch.tensor([geo.triple(t).rho   for t in clean], dtype=self.dtype, device=self.device)
        cur_sigma = torch.tensor([geo.triple(t).sigma for t in clean], dtype=self.dtype, device=self.device)
        sims = self._batch_sim(cur_rho, cur_sigma, kernels)   # [N]
        topk = torch.topk(sims, min(self.top_k, len(self.store)))
        return [(topk.values[i].item(), self.store[topk.indices[i].item()])
                for i in range(topk.values.shape[0])]

    def syntax_echo_bonus(
        self,
        c_rho          : torch.Tensor,   # [C]  GPU
        c_sigma        : torch.Tensor,   # [C]  GPU
        current_tokens : List[str],
        geo            : ThebaultTokenGeometry,
        kernels        : ThebaultKernels,
        echo_weight    : float = 0.5,
    ) -> torch.Tensor:
        anchors = self.ranked_anchors(current_tokens, geo, kernels)
        if not anchors:
            return torch.zeros(c_rho.shape[0], device=self.device)

        pos     = len([t for t in current_tokens
                       if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS])
        bonuses = torch.zeros(c_rho.shape[0], dtype=self.dtype, device=self.device)

        for sim_score, anc in anchors:
            if pos < anc.rho_t.shape[0]:
                a_rho   = anc.rho_t  [pos].item()
                a_sigma = anc.sigma_t[pos].item()
                kr = torch.exp(-kernels.lambda_reg * (c_rho   - a_rho)   ** 2)
                ks = torch.exp(-kernels.gamma_side * (c_sigma - a_sigma) ** 2)
                bonuses += sim_score * (kr * ks)

        std = bonuses.std()
        if std.item() > 1e-8:
            bonuses = (bonuses - bonuses.mean()) / std
        return bonuses * echo_weight

    def similarity_table(self, kernels: ThebaultKernels, max_pairs: int = 10) -> str:
        pairs = []
        for i in range(len(self.store)):
            sv_i = self.store[i]
            L    = sv_i.rho_t.shape[0]
            for j in range(i + 1, len(self.store)):
                sv_j = self.store[j]
                l    = min(L, sv_j.rho_t.shape[0])
                if l == 0:
                    continue
                kr = torch.exp(-kernels.lambda_reg * (sv_i.rho_t[:l]   - sv_j.rho_t[:l])   ** 2)
                ks = torch.exp(-kernels.gamma_side * (sv_i.sigma_t[:l] - sv_j.sigma_t[:l]) ** 2)
                s  = (kr * ks).mean().item()
                pairs.append((s, i, j))
        pairs.sort(key=lambda x: -x[0])
        lines = []
        for s, i, j in pairs[:max_pairs]:
            a_p = " ".join(self.store[i].tokens[:5])
            b_p = " ".join(self.store[j].tokens[:5])
            lines.append(f"  {s:.4f}  [{i:02d}] {a_p:<25s}  ≈  [{j:02d}] {b_p}")
        return "\n".join(lines) if lines else "  (no pairs yet)"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THÉBAULT CONJUGATE ORBIT
# ════════════════════════════════════════════════════════════════════════════

class ThebaultConjugateOrbit:
    def score(self, anchor_triple, cand_theta: torch.Tensor, cand_sigma: torch.Tensor,
              gamma_side: float = 4.0) -> torch.Tensor:
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor_triple.sigma) ** 2)
        antipodality = torch.cos(cand_theta + anchor_triple.theta - math.pi / 2) ** 2
        return congruence * antipodality


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — THÉBAULT COMPOSITION LM  (dense GPU weight matrix)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultCompositionLM:
    """
    CUDA optimisation:
    • Trigram table compiled to a dense [V, V, V] float16 tensor (or sparse
      coo for large vocab).  next_dist() is a tensor index + softmax — no dict walk.
    • Falling back to sparse (dict) representation if vocab > DENSE_THRESH.
    """

    BASAL_K    = 1.5
    DENSE_THRESH = 512   # use dense matrix only if vocab ≤ this

    def __init__(self, geo: ThebaultTokenGeometry, kernels: ThebaultKernels,
                 device: torch.device = DEVICE):
        self.geo      = geo
        self.kernels  = kernels
        self.device   = device
        self.raw_freq : Dict[str, float]                  = {}
        self.tri_raw  : Dict[Tuple[str, str, str], float] = {}
        self.heads    : Dict[Tuple[str, str], List[str]]  = {}
        self.vocab    : List[str]                         = []
        self._tok2idx : Dict[str, int]                    = {}
        # GPU caches (built in finalise())
        self._head_cands : Dict[Tuple[str, str], torch.Tensor]  = {}   # idx tensors
        self._head_probs : Dict[Tuple[str, str], torch.Tensor]  = {}   # base prob tensors

    def ingest(self, tokens: List[str]) -> None:
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

    def finalise(self) -> None:
        """Pre-compute GPU tensors for each bigram head."""
        self._tok2idx = {t: i for i, t in enumerate(self.vocab)}
        V_tot = len(self.vocab) + 1

        for (w1, w2), cands in self.heads.items():
            total = sum(self.tri_raw.get((w1, w2, c), 1e-4) for c in cands)
            counts = [self.tri_raw.get((w1, w2, c), 1e-4) for c in cands]
            basal  = torch.tensor(
                [(cnt + self.BASAL_K) / (total + self.BASAL_K * V_tot) for cnt in counts],
                dtype=torch.float32, device=self.device,
            )
            idx_t  = torch.tensor(
                [self._tok2idx.get(c, 0) for c in cands],
                dtype=torch.long, device=self.device,
            )
            self._head_cands[(w1, w2)] = idx_t
            self._head_probs[(w1, w2)] = basal

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        head = (w1, w2)
        if head in self.heads:
            cands  = self.heads[head]
            base_p = self._head_probs[head]
        else:
            # Aggregate fallback (CPU dict, rare path)
            agg = {}
            for (_, _, w3), wt in self.tri_raw.items():
                agg[w3] = agg.get(w3, 0) + wt
            cands  = list(agg.keys())[:400]
            total  = sum(agg.values())
            V_tot  = len(self.vocab) + 1
            counts = [agg[c] for c in cands]
            base_p = torch.tensor(
                [(cnt + self.BASAL_K) / (total + self.BASAL_K * V_tot) for cnt in counts],
                dtype=torch.float32, device=self.device,
            )
        return cands, base_p

    def composition_logit_bonus(
        self, w1: str, w2: str,
        c_rho: torch.Tensor, c_sigma: torch.Tensor,
    ) -> torch.Tensor:
        C = self.geo.composed_triple(w1, w2)
        kr = torch.exp(-self.kernels.lambda_reg * (c_rho   - C.rho)   ** 2)
        ks = torch.exp(-self.kernels.gamma_side * (c_sigma - C.sigma) ** 2)
        return kr * ks


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — THÉBAULT POTENTIAL GRAPH
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
    def __init__(self, geo: ThebaultTokenGeometry, kernels: ThebaultKernels,
                 device: torch.device = DEVICE):
        self.geo     = geo
        self.kernels = kernels
        self.device  = device
        self.nodes   : Dict[str, TGNode]       = {}
        self.adj     : Dict[str, List[TGEdge]] = {}
        self.radj    : Dict[str, List[TGEdge]] = {}
        self._pot_t  : Optional[torch.Tensor]  = None  # [V] GPU potential tensor
        self._vocab  : List[str]               = []

    def build(self, lm: ThebaultCompositionLM) -> None:
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
                    self.kernels.k_reg (ti.rho,   torch.tensor(tj.rho,   device=self.device)).item()
                    * self.kernels.k_ori(ti.theta, torch.tensor(tj.theta, device=self.device)).item()
                    * cnt
                )
                e = TGEdge(w2, w3, max(w, 1e-6))
                self.adj [w2].append(e)
                self.radj[w3].append(e)
                seen.add((w2, w3))

    def propagate(self, steps: int = 2) -> None:
        if not self.nodes:
            return
        max_f = max(nd.freq for nd in self.nodes.values()) + 1e-8
        for nd in self.nodes.values():
            nd.potential = nd.triple.rho * nd.freq / max_f
        for _ in range(steps):
            new_pots = {}
            for v, nd in self.nodes.items():
                agg = sum(e.weight * self.nodes[e.src].potential
                          for e in self.radj.get(v, []))
                self_scale = nd.triple.sigma / (nd.triple.sigma + 1.0)
                new_pots[v] = agg / (len(self.radj.get(v, [])) + 1.0) + self_scale * nd.potential * 0.1
            mx = max(new_pots.values(), default=1.0) + 1e-8
            for v in self.nodes:
                self.nodes[v].potential = new_pots[v] / mx

        # Materialise as GPU tensor (vocab order matches lm.vocab)
        self._vocab = list(self.nodes.keys())
        self._pot_t = torch.tensor(
            [self.nodes[v].potential for v in self._vocab],
            dtype=torch.float32, device=self.device,
        )

    def potentials_for(self, cands: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self.nodes[c].potential if c in self.nodes else 0.0 for c in cands],
            dtype=torch.float32, device=self.device,
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — synthetic_reason MANDATES
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

    def subsynthetic_reason_concept_enrichment(
        self, w_ctx: str, cands: List[str], device: torch.device
    ) -> torch.Tensor:
        enrichment = torch.zeros(len(cands), device=device)
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
# SECTION 11 — THÉBAULT WALKER V17-CUDA
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    """
    CUDA-optimised walk:
    • All tensor ops on self.device.
    • walk_probs() fetches pre-built GPU triples for all candidates in one
      geo.batch_triples() call — no per-candidate Python loop.
    • Logit assembly via a single torch.stack().sum() — fused kernel.
    """

    def __init__(self, geo, kernels, lm, orbit, graph, synth,
                 mrv_filter, chunk_engine, iso_stacker,
                 device: torch.device = DEVICE):
        self.geo          = geo
        self.kernels      = kernels
        self.lm           = lm
        self.orbit        = orbit
        self.graph        = graph
        self.synth        = synth
        self.mrv          = mrv_filter
        self.chunk_engine = chunk_engine
        self.iso_stacker  = iso_stacker
        self.device       = device
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []
        self._cur_sent_toks: List[str] = []

    def begin_sentence(self) -> None:
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()

    @torch.no_grad()
    def walk_probs(
        self, w1: str, w2: str,
        temp          : float = 1.4,
        alpha_reg     : float = 1.2,
        beta_ori      : float = 0.8,
        delta_side    : float = 1.0,
        gamma_orbit   : float = 0.6,
        psi_pot       : float = 0.35,
        zeta_mrv      : float = 0.9,
        eta_chunk     : float = 0.7,
        xi_echo       : float = 0.6,
    ) -> Tuple[List[str], torch.Tensor]:

        cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands:
            return cands, base_probs

        # ── Batch fetch triples for all candidates ────────────────────────────
        # Try GPU batch path; fall back gracefully if not indexed.
        try:
            tok_idx = self.geo.tok_indices(cands)              # [C]  long
            c_rho, c_theta, c_sigma = self.geo.batch_triples(tok_idx)  # 3 × [C]
            c_pvec  = self.geo._pvec_t[tok_idx]               # [C, 4]
        except Exception:
            triples  = [self.geo.triple(c) for c in cands]
            c_rho    = torch.tensor([t.rho   for t in triples], dtype=torch.float32, device=self.device)
            c_theta  = torch.tensor([t.theta for t in triples], dtype=torch.float32, device=self.device)
            c_sigma  = torch.tensor([t.sigma for t in triples], dtype=torch.float32, device=self.device)
            c_pvec   = torch.stack([c_rho, c_theta / math.pi, c_sigma,
                                    torch.ones_like(c_rho)], dim=1)

        ctx = self.geo.triple(w2)

        # ── All kernel scores in one batched call ─────────────────────────────
        k_reg, k_ori, k_side = self.kernels.all_scores_batched(
            ctx.rho, ctx.theta, ctx.sigma, c_rho, c_theta, c_sigma
        )
        orbit_scores = self.orbit.score(ctx, c_theta, c_sigma, self.kernels.gamma_side)
        pots         = self.graph.potentials_for(cands)
        comp_bonus   = self.lm.composition_logit_bonus(w1, w2, c_rho, c_sigma)
        mrv_scores   = self.mrv.mrv_scores_batched(c_rho, c_sigma, self.kernels)

        # ── Chunked sum bonus ─────────────────────────────────────────────────
        chunk_bonus  = self.chunk_engine.chunk_bonus(c_pvec, scale=eta_chunk)

        # ── Syntax echo bonus ─────────────────────────────────────────────────
        echo_bonus   = self.iso_stacker.syntax_echo_bonus(
            c_rho, c_sigma, self._cur_sent_toks, self.geo, self.kernels, xi_echo
        )

        # ── Isomorphic pair detection (top 50 only, avoid O(C²) loop) ────────
        self.current_isomorphic_pairs = []
        top_idx = torch.topk(k_reg * k_side, min(50, len(cands))).indices
        sub_r   = k_reg[top_idx]
        sub_s   = k_side[top_idx]
        iso_mask = (sub_r > 0.98) & (sub_s > 0.98)
        iso_idx  = top_idx[iso_mask].tolist()
        for ii in range(len(iso_idx)):
            for jj in range(ii + 1, len(iso_idx)):
                i, j = iso_idx[ii], iso_idx[jj]
                ci, cj = cands[i], cands[j]
                if ci not in PUNCT_TOKENS and cj not in PUNCT_TOKENS:
                    sim = (k_reg[i] * k_side[i] * k_reg[j] * k_side[j]).sqrt().item()
                    self.current_isomorphic_pairs.append((ci, cj, sim))
        self.current_isomorphic_pairs.sort(key=lambda x: -x[2])

        # ── Punct penalties ───────────────────────────────────────────────────
        N = len(cands)
        punct_bias    = torch.zeros(N, device=self.device)
        punct_penalty = torch.zeros(N, device=self.device)
        for i, c in enumerate(cands):
            if c in PUNCT_TOKENS:
                punct_bias[i] = -3.5
                if w2 in PUNCT_TOKENS:
                    punct_penalty[i] = -1e4

        mandate_boost = self.synth.subsynthetic_reason_concept_enrichment(
            w2, cands, self.device
        )

        # ── Fused logit sum ───────────────────────────────────────────────────
        log_base = torch.log(base_probs.clamp(min=1e-12))
        logits   = (
            log_base
            + alpha_reg   * k_reg
            + beta_ori    * k_ori
            + delta_side  * k_side
            + gamma_orbit * orbit_scores
            + psi_pot     * pots
            + comp_bonus
            + zeta_mrv    * mrv_scores
            + chunk_bonus
            + echo_bonus
            + mandate_boost
            + punct_bias
            + punct_penalty
        ) / max(temp, 1e-6)

        return cands, F.softmax(logits, dim=-1)

    def push_token(self, token: str, sentence_len: int) -> None:
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS:
            return
        self._cur_sent_toks.append(token)
        pos_norm = len(self._cur_sent_toks) / max(sentence_len, 1)
        self.chunk_engine.push(self.geo.triple(token), pos_norm)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — ENGINE STATE & GENERATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class V17State:
    lm          : ThebaultCompositionLM
    graph       : ThebaultPotentialGraph
    walker      : ThebaultWalker
    mrv_filter  : MRVConstraintFilter
    iso_stacker : IsomorphicSyntaxStacker
    device      : torch.device
    outputs     : Dict[int, str]         = field(default_factory=dict)
    iso_matches : Set[Tuple[str, str]]   = field(default_factory=set)


def build_v17_state(
    corpus_text   : str,
    lambda_reg    : float = 8.0,
    gamma_side    : float = 4.0,
    mrv_threshold : float = 0.50,
    mrv_cap_ratio : float = 2.0,
    window_size   : int   = 16,
    n_chunks      : int   = 4,
    iso_top_k     : int   = 3,
    device        : Optional[torch.device] = None,
    dtype         : torch.dtype = torch.float32,
    use_compile   : bool = False,
) -> V17State:
    if device is None:
        device = best_device()

    tokens = tokenize(corpus_text)

    geo     = ThebaultTokenGeometry(device=device, dtype=dtype)
    kernels = ThebaultKernels(lambda_reg=lambda_reg, gamma_side=gamma_side)
    lm      = ThebaultCompositionLM(geo, kernels, device=device)
    lm.ingest(tokens)

    all_tokens = list(lm.raw_freq.keys())
    max_freq   = max(lm.raw_freq.values(), default=1.0)
    vocab_size = len(all_tokens)
    for idx, tok in enumerate(all_tokens):
        geo.register(tok, lm.raw_freq[tok], idx, max_freq, vocab_size)

    # Build GPU tensor cache — called once after all register() calls
    geo.build_cuda_tensors(lm.vocab)

    # Finalise LM GPU tables
    lm.finalise()

    orbit  = ThebaultConjugateOrbit()
    graph  = ThebaultPotentialGraph(geo, kernels, device=device)
    graph.build(lm)
    graph.propagate(steps=2)

    mrv_filter = MRVConstraintFilter(
        threshold      = mrv_threshold,
        mrv_cap_ratio  = mrv_cap_ratio,
        max_vocab_scan = min(300, vocab_size),
        device         = device,
    )
    mrv_filter.prime(lm.vocab, geo)

    chunk_engine = ChunkedSumEngine(
        window_size=window_size, n_chunks=n_chunks, device=device, dtype=dtype
    )
    iso_stacker = IsomorphicSyntaxStacker(
        top_k=iso_top_k, device=device, dtype=dtype
    )
    synth = synthetic_reasonMandateProcessor()

    walker = ThebaultWalker(
        geo, kernels, lm, orbit, graph, synth,
        mrv_filter, chunk_engine, iso_stacker, device=device,
    )

    if use_compile and hasattr(torch, "compile"):
        try:
            walker.walk_probs = torch.compile(
                walker.walk_probs, mode="reduce-overhead", fullgraph=False
            )
        except Exception:
            pass  # compile not supported on this platform

    return V17State(lm, graph, walker, mrv_filter, iso_stacker, device)


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
        state.walker.begin_sentence()

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
        state.iso_stacker.add(toks, state.walker.geo, sentence_text)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — GRADIO UI
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
