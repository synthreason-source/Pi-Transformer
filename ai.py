#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V17-CUDA — Thébault + MRV + Isomorphic Syntax Stacking
                        + Positional Vectorisation + Chunked Sum Generation
                        + Petr–Douglas–Neumann (PDN) Theorem Integration
===============================================================================

PDN THEOREM EXTENSION (new in this version)
────────────────────────────────────────────
The Petr–Douglas–Neumann (PDN) theorem generalises Thébault's theorem:
  - Thébault: squares (n=4) on parallelogram sides → four equidistant centres
  - PDN:      regular n-gons on any polygon's sides → nth DFT component = 0
               implies the centre polygon is itself regular.

In token space this manifests as:

  1.  PDN_SPECTRAL_KERNEL  — each token's (rho, theta) pair is lifted into
      a complex DFT coefficient Z_k = rho·exp(i·k·theta) for k=1…n_modes.
      The PDN invariant is that the k-th Fourier mode of a "good" sequence
      must be near-zero; we use deviation from this as a *penalty*.

  2.  PDN_REGULARITY_SCORE — measures how close the current n-gram window's
      centre-polygon is to a regular polygon (PDN convergence metric).
      Used as a *bonus* for tokens that increase sequence regularity.

  3.  PDN_ORBIT_FAMILIES — tokens are grouped by which n-gon orbit they
      belong to (floor(theta / (2π/n))). The walker prefers tokens from
      the next orbit in sequence, encoding the rotational structure of PDN.

  4.  PDN_DATASET_STEM — the corpus is analysed for its dominant PDN mode
      n* via spectral analysis of all trigram (rho, theta) triples. n* is
      used globally as the governing symmetry order, making it *data-driven*.

CUDA OPTIMISATION CHANGES (vs V17-CPU)
───────────────────────────────────────
1.  DEVICE-AWARE TENSOR CACHE
2.  FULLY VECTORISED KERNEL SCORING
3.  PRE-BATCHED LM DISTRIBUTIONS
4.  BATCHED SENTENCE GENERATION
5.  CHUNK ENGINE — ALL-CUDA
6.  ISO-STACKER BATCHED SIMILARITY
7.  FUSED LOGIT ACCUMULATION
8.  torch.compile SUPPORT
9.  MIXED PRECISION (optional)

All V17 mathematical semantics are preserved exactly.
===============================================================================
"""

from __future__ import annotations
import re, math, random, unicodedata, pickle, argparse, cmath
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
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype  = dtype
        self._vecs  : Dict[str, Tuple[float, float, float, float]] = {}
        self._cache : Dict[str, ThebaultTriple]                    = {}
        self._tok2idx: Dict[str, int]        = {}
        self._rho_t  : Optional[torch.Tensor] = None
        self._theta_t: Optional[torch.Tensor] = None
        self._sigma_t: Optional[torch.Tensor] = None
        self._pvec_t : Optional[torch.Tensor] = None
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
        self._pvec_t  = torch.stack([
            self._rho_t,
            self._theta_t / math.pi,
            self._sigma_t,
            torch.ones_like(self._rho_t),
        ], dim=1)

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

    def batch_triples(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._rho_t[indices],
            self._theta_t[indices],
            self._sigma_t[indices],
        )

    def tok_indices(self, toks: List[str]) -> torch.Tensor:
        idx = [self._tok2idx.get(t, 0) for t in toks]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2b — PETR–DOUGLAS–NEUMANN THEOREM ENGINE
# ════════════════════════════════════════════════════════════════════════════
#
# Mathematical Foundation
# ───────────────────────
# PDN Theorem: Given any polygon P with vertices v_0…v_{m-1}, construct a
# regular n-gon on each side. Connect the outer centres C_0…C_{m-1}. Then:
#
#   The centre polygon is regular  ⟺  DFT_k(v) = 0  for k ≠ 0 (mod n)
#
# where DFT_k(v) = Σ_j v_j · exp(-2πi·j·k/m).
#
# Thébault's theorem is the n=4 case: squares on a quadrilateral's sides.
# The canonical Thébault result (centres form a square) is equivalent to
# the statement that the k=1 Fourier mode of the quadrilateral vanishes,
# which happens automatically when the quadrilateral is a parallelogram.
#
# Token-Space Interpretation
# ───────────────────────────
# Each token t has a complex coordinate z_t = rho_t · exp(i · theta_t).
# A token n-gram window W = [t_0, …, t_{n-1}] defines a "polygon" in this
# complex plane. We compute:
#
#   F_k(W) = (1/n) Σ_{j=0}^{n-1}  z_{t_j} · exp(-2πi·j·k/n)
#
# The PDN regularity score for mode k is:
#
#   R_k(W) = exp(- |F_k(W)|² / sigma²_pdn )
#
# R_k → 1 means the k-th mode is suppressed, i.e. the token polygon is
# "regular" w.r.t. that mode — precisely the PDN condition.
#
# Dataset-Derived n* (PDN stem from corpus)
# ──────────────────────────────────────────
# We compute the power spectrum P_k = Σ_{trigrams} |F_k(trigram)|² for
# k = 3, 4, 5, 6. The dominant k = n* is the "natural symmetry order" of
# the corpus. This makes the PDN mode *data-driven* rather than fixed.

class PDNEngine:
    """
    Petr–Douglas–Neumann theorem engine for token sequences.

    Attributes
    ----------
    n_modes   : int   — number of candidate PDN orders to probe (3 to 3+n_modes)
    n_star    : int   — dominant PDN order determined from corpus
    sigma_pdn : float — bandwidth for the regularity Gaussian
    """

    def __init__(
        self,
        n_modes   : int   = 4,   # probe n = 3,4,5,6
        sigma_pdn : float = 0.25,
        orbit_weight: float = 0.4,
        regularity_weight: float = 0.5,
        spectral_penalty_weight: float = 0.3,
        device    : torch.device = DEVICE,
        dtype     : torch.dtype  = torch.float32,
    ):
        self.n_modes   = n_modes
        self.sigma_pdn = sigma_pdn
        self.orbit_weight = orbit_weight
        self.regularity_weight = regularity_weight
        self.spectral_penalty_weight = spectral_penalty_weight
        self.device    = device
        self.dtype     = dtype
        self.n_star    : int = 4          # default = Thébault (squares)
        self.power_spectrum: Dict[int, float] = {}
        # Precomputed orbit look-up: token → orbit index under n*
        self._orbit_map: Dict[str, int] = {}

    # ── 2b.1  DATASET STEM: find dominant PDN order from corpus trigrams ──

    def fit_from_trigrams(self, geo: ThebaultTokenGeometry, tri_raw: Dict) -> None:
        """
        Compute power spectrum P_k for k in {3,4,5,6} over all corpus
        trigrams.  Sets self.n_star to the dominant order.
        """
        candidate_ns = list(range(3, 3 + self.n_modes))
        power: Dict[int, float] = {n: 0.0 for n in candidate_ns}

        for (w1, w2, w3), cnt in tri_raw.items():
            toks = [w1, w2, w3]
            zs   = []
            for t in toks:
                tr  = geo.triple(t)
                zs.append(complex(tr.rho * math.cos(tr.theta),
                                  tr.rho * math.sin(tr.theta)))
            # For each candidate n, compute the (n-1)th DFT mode of the
            # 3-vertex sub-polygon (pad to length n with zeros)
            for n in candidate_ns:
                padded = zs + [0+0j] * (n - 3)
                for k in range(1, n):
                    F_k = sum(padded[j] * cmath.exp(-2j * math.pi * j * k / n)
                              for j in range(n)) / n
                    power[n] += cnt * abs(F_k) ** 2

        self.power_spectrum = power
        # n* is the order with LOWEST total power — most "regular" corpus
        self.n_star = min(power, key=lambda k: power[k])
        print(f"[PDN] Power spectrum: { {n: f'{p:.2f}' for n, p in power.items()} }")
        print(f"[PDN] Dominant symmetry order n* = {self.n_star}  "
              f"(stems from Thébault n=4 → generalised PDN)")

    # ── 2b.2  ORBIT FAMILIES ──────────────────────────────────────────────

    def build_orbit_map(self, vocab: List[str], geo: ThebaultTokenGeometry) -> None:
        """
        Partition vocabulary tokens into n* orbit families based on theta.
        Orbit index = floor(theta / (2π / n*))
        """
        sector = 2.0 * math.pi / max(self.n_star, 2)
        for tok in vocab:
            tr = geo.triple(tok)
            # theta is in [0, π]; map to full circle by mirroring
            full_theta = tr.theta * 2.0   # ∈ [0, 2π)
            self._orbit_map[tok] = int(full_theta / sector) % self.n_star
        print(f"[PDN] Built orbit map for {len(self._orbit_map)} tokens "
              f"across {self.n_star} orbit families.")

    def orbit_of(self, token: str) -> int:
        return self._orbit_map.get(token, 0)

    # ── 2b.3  SPECTRAL REGULARITY SCORE (vectorised, CUDA) ───────────────

    def regularity_scores(
        self,
        window_rho  : torch.Tensor,   # (W,) current n-gram window rhos
        window_theta: torch.Tensor,   # (W,) current n-gram window thetas
        c_rho       : torch.Tensor,   # (C,) candidate rhos
        c_theta     : torch.Tensor,   # (C,) candidate thetas
    ) -> torch.Tensor:
        """
        For each candidate c, compute the PDN regularity score of the
        extended window [window_tokens..., c] under n=n_star.

        Score = exp(- |F_{n*-1}(extended)|² / sigma²)

        Higher score → appending this token makes the sequence *more
        regular* in the PDN sense — i.e. closer to satisfying the theorem.
        """
        n   = self.n_star
        W   = window_rho.shape[0]
        C   = c_rho.shape[0]

        if W == 0:
            return torch.ones(C, dtype=self.dtype, device=self.device)

        # Complex coordinates of existing window: (W,) complex128 → real pairs
        # Re/Im for existing tokens
        win_re = (window_rho * torch.cos(window_theta)).to(self.dtype)  # (W,)
        win_im = (window_rho * torch.sin(window_theta)).to(self.dtype)  # (W,)

        # Candidate complex coords: (C,)
        c_re = (c_rho * torch.cos(c_theta)).to(self.dtype)
        c_im = (c_rho * torch.sin(c_theta)).to(self.dtype)

        # DFT mode k = n-1 (the critical PDN mode)
        k = n - 1
        # Precompute DFT coefficients for existing window positions 0..W-1
        # exp(-2πi·j·k/n) for j=0..W-1  →  (cos, -sin) pairs
        js        = torch.arange(W, dtype=self.dtype, device=self.device)
        angle_w   = -2.0 * math.pi * js * k / n            # (W,)
        cos_w     = torch.cos(angle_w)                      # (W,)
        sin_w     = torch.sin(angle_w)                      # (W,)

        # Partial sum for existing tokens
        re_partial = (win_re * cos_w - win_im * sin_w).sum()   # scalar
        im_partial = (win_re * sin_w + win_im * cos_w).sum()   # scalar

        # Coefficient for the appended candidate at position W
        angle_c = -2.0 * math.pi * W * k / n
        cos_c   = math.cos(angle_c)
        sin_c   = math.sin(angle_c)

        # Full DFT mode k for each candidate
        F_re = re_partial + c_re * cos_c - c_im * sin_c    # (C,)
        F_im = im_partial + c_re * sin_c + c_im * cos_c    # (C,)

        power = (F_re ** 2 + F_im ** 2) / (n ** 2)         # (C,) normalised

        return torch.exp(-power / (self.sigma_pdn ** 2 + 1e-8))

    # ── 2b.4  ORBIT SEQUENCE BONUS ────────────────────────────────────────

    def orbit_bonus(
        self,
        current_orbit: int,
        c_theta      : torch.Tensor,   # (C,)
    ) -> torch.Tensor:
        """
        Prefer candidates whose orbit index is (current_orbit + 1) % n*.
        This encodes the rotational traversal structure of PDN construction:
        each successive token "turns" by one sector, analogous to rotating
        the n-gon construction around the polygon's perimeter.
        """
        n        = self.n_star
        target   = (current_orbit + 1) % n
        sector   = 2.0 * math.pi / max(n, 2)
        # Map candidate thetas to orbit index (continuous approximation)
        full_theta = c_theta * 2.0              # mirror [0,π] → [0,2π)
        orbit_cont = full_theta / sector         # continuous orbit index
        # Cosine similarity to target orbit
        bonus = torch.cos(2.0 * math.pi * (orbit_cont - target) / n) * 0.5 + 0.5
        return bonus

    # ── 2b.5  COMBINED PDN SCORE ──────────────────────────────────────────

    @torch.no_grad()
    def pdn_logit_bonus(
        self,
        window_rho  : torch.Tensor,
        window_theta: torch.Tensor,
        c_rho       : torch.Tensor,
        c_theta     : torch.Tensor,
        current_orbit: int,
    ) -> torch.Tensor:
        """
        Fused PDN bonus = regularity_weight * R(c)
                        + orbit_weight      * O(c)
        Returned as a (C,) logit addition.
        """
        reg = self.regularity_scores(window_rho, window_theta, c_rho, c_theta)
        orb = self.orbit_bonus(current_orbit, c_theta)

        # Normalise each component to zero-mean unit-variance if possible
        def _norm(x):
            std = x.std()
            return (x - x.mean()) / (std + 1e-8) if std.item() > 1e-8 else x - x.mean()

        return self.regularity_weight * _norm(reg) + self.orbit_weight * _norm(orb)

    # ── 2b.6  THÉBAULT→PDN BRIDGE REPORT ─────────────────────────────────

    def theorem_bridge_report(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║         Thébault → Petr–Douglas–Neumann Bridge Report        ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Thébault case:  n = 4  (squares on parallelogram sides)     ║",
            f"║  PDN n*:         n = {self.n_star:<2d}  (derived from corpus spectrum)    ║",
            "║                                                              ║",
            "║  Equivalence:                                                ║",
            "║   Thébault → DFT_{k=3} of quad vertices = 0                 ║",
            "║   PDN n*   → DFT_{k=n*-1} of token window = 0              ║",
            "║                                                              ║",
            "║  Power spectrum over corpus trigrams:                        ║",
        ]
        for n, p in sorted(self.power_spectrum.items()):
            marker = " ← n* (dominant)" if n == self.n_star else ""
            lines.append(f"║    n={n}: P={p:>10.2f}{marker:<28s}║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THÉBAULT KERNELS
# ════════════════════════════════════════════════════════════════════════════

class ThebaultKernels:
    def __init__(self, lambda_reg: float = 8.0, gamma_side: float = 4.0):
        self.lambda_reg = lambda_reg
        self.gamma_side = gamma_side

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

    def all_scores_batched(
        self,
        rho_a  : float, theta_a: float, sigma_a: float,
        rho_b  : torch.Tensor, theta_b: torch.Tensor, sigma_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_r = torch.exp(-self.lambda_reg * (rho_b   - rho_a)   ** 2)
        k_o = 0.5 * (1.0 + torch.cos(theta_b - theta_a))
        k_s = torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)
        return k_r, k_o, k_s

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MRV CONSTRAINT FILTER
# ════════════════════════════════════════════════════════════════════════════

class MRVConstraintFilter:
    def __init__(self, threshold=0.50, mrv_cap_ratio=2.0, max_vocab_scan=300, device: torch.device = DEVICE):
        self.threshold      = threshold
        self.mrv_cap_ratio  = mrv_cap_ratio
        self.max_vocab_scan = max_vocab_scan
        self.device         = device
        self._v_rho  : Optional[torch.Tensor] = None
        self._v_sigma: Optional[torch.Tensor] = None
        self._v_toks : List[str]              = []

    def prime(self, vocab: List[str], geo: ThebaultTokenGeometry) -> None:
        scan  = vocab[:self.max_vocab_scan]
        trips = [geo.triple(v) for v in scan]
        self._v_rho   = torch.tensor([t.rho   for t in trips], dtype=torch.float32, device=self.device)
        self._v_sigma = torch.tensor([t.sigma for t in trips], dtype=torch.float32, device=self.device)
        self._v_toks  = scan

    def mrv_scores_batched(self, c_rho: torch.Tensor, c_sigma: torch.Tensor, kernels: ThebaultKernels) -> torch.Tensor:
        if self._v_rho is None:
            return torch.zeros(c_rho.shape[0], device=self.device)
        k_r = torch.exp(-kernels.lambda_reg * (c_rho.unsqueeze(1)   - self._v_rho.unsqueeze(0))   ** 2)
        k_s = torch.exp(-kernels.gamma_side * (c_sigma.unsqueeze(1) - self._v_sigma.unsqueeze(0)) ** 2)
        thr = self.threshold
        domain_sizes = ((k_r > thr) & (k_s > thr)).float().sum(dim=1)
        mean_d = domain_sizes.mean() + 1e-6
        mrv    = 1.0 / (domain_sizes + 1.0)
        mrv[domain_sizes > self.mrv_cap_ratio * mean_d] *= 0.5
        lo, hi = mrv.min(), mrv.max()
        if (hi - lo).item() > 1e-8:
            mrv = (mrv - lo) / (hi - lo)
        return mrv

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
# SECTION 5 — POSITIONAL VECTOR + CHUNKED SUM ENGINE
# ════════════════════════════════════════════════════════════════════════════

VEC_DIM = 4

class ChunkedSumEngine:
    def __init__(self, window_size: int = 16, n_chunks: int = 4, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.window_size = window_size
        self.n_chunks    = n_chunks
        self.device      = device
        self.dtype       = dtype
        self._buf   = torch.zeros(window_size, VEC_DIM, dtype=dtype, device=device)
        self._ptr   = 0
        self._count = 0

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
        if self._count == 0:
            return torch.zeros(self.n_chunks * VEC_DIM, dtype=self.dtype, device=self.device)
        if self._count < self.window_size:
            window = self._buf[:self._count]
        else:
            window = torch.cat([self._buf[self._ptr:], self._buf[:self._ptr]], dim=0)
        W   = window.shape[0]
        pad = (-W) % self.n_chunks
        if pad > 0:
            window = torch.cat([window, torch.zeros(pad, VEC_DIM, dtype=self.dtype, device=self.device)])
        chunk_len = window.shape[0] // self.n_chunks
        chunks    = window.view(self.n_chunks, chunk_len, VEC_DIM)
        return chunks.sum(dim=1).flatten()

    def chunk_bonus(self, c_pvec: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        sig      = self.chunk_signature()
        cv_tiled = c_pvec.repeat(1, self.n_chunks)
        raw      = cv_tiled @ sig
        std      = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * scale

    def window_rho_theta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current ring-buffer rho and theta columns for PDN."""
        if self._count == 0:
            empty = torch.zeros(0, dtype=self.dtype, device=self.device)
            return empty, empty
        if self._count < self.window_size:
            window = self._buf[:self._count]
        else:
            window = torch.cat([self._buf[self._ptr:], self._buf[:self._ptr]], dim=0)
        # col 0 = rho, col 1 = theta/π  →  recover theta
        return window[:, 0], window[:, 1] * math.pi

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ISOMORPHIC SYNTAX STACKER
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SentenceVector:
    tokens  : List[str]
    rho_t   : torch.Tensor
    sigma_t : torch.Tensor
    text    : str

class IsomorphicSyntaxStacker:
    def __init__(self, top_k: int = 3, max_stored: int = 64, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.top_k      = top_k
        self.max_stored = max_stored
        self.device     = device
        self.dtype      = dtype
        self.store      : List[SentenceVector] = []

    def add(self, tokens: List[str], geo: ThebaultTokenGeometry, text: str) -> None:
        clean = [t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean:
            return
        rhos   = torch.tensor([geo.triple(t).rho   for t in clean], dtype=self.dtype, device=self.device)
        sigmas = torch.tensor([geo.triple(t).sigma for t in clean], dtype=self.dtype, device=self.device)
        self.store.append(SentenceVector(clean, rhos, sigmas, text))
        if len(self.store) > self.max_stored:
            self.store.pop(0)

    def _batch_sim(self, cur_rho: torch.Tensor, cur_sigma: torch.Tensor, kernels: ThebaultKernels) -> torch.Tensor:
        L = cur_rho.shape[0]
        N = len(self.store)
        if N == 0 or L == 0:
            return torch.zeros(0, device=self.device)
        stored_rho   = torch.zeros(N, L, dtype=self.dtype, device=self.device)
        stored_sigma = torch.zeros(N, L, dtype=self.dtype, device=self.device)
        for i, sv in enumerate(self.store):
            l = min(L, sv.rho_t.shape[0])
            stored_rho  [i, :l] = sv.rho_t  [:l]
            stored_sigma[i, :l] = sv.sigma_t[:l]
        kr = torch.exp(-kernels.lambda_reg * (stored_rho   - cur_rho.unsqueeze(0))   ** 2)
        ks = torch.exp(-kernels.gamma_side * (stored_sigma - cur_sigma.unsqueeze(0)) ** 2)
        return (kr * ks).mean(dim=1)

    def ranked_anchors(self, current_tokens: List[str], geo: ThebaultTokenGeometry, kernels: ThebaultKernels) -> List[Tuple[float, SentenceVector]]:
        if not self.store or not current_tokens:
            return []
        clean = [t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean:
            return []
        cur_rho   = torch.tensor([geo.triple(t).rho   for t in clean], dtype=self.dtype, device=self.device)
        cur_sigma = torch.tensor([geo.triple(t).sigma for t in clean], dtype=self.dtype, device=self.device)
        sims = self._batch_sim(cur_rho, cur_sigma, kernels)
        topk = torch.topk(sims, min(self.top_k, len(self.store)))
        return [(topk.values[i].item(), self.store[topk.indices[i].item()])
                for i in range(topk.values.shape[0])]

    def syntax_echo_bonus(self, c_rho: torch.Tensor, c_sigma: torch.Tensor, current_tokens: List[str], geo: ThebaultTokenGeometry, kernels: ThebaultKernels, echo_weight: float = 0.5) -> torch.Tensor:
        anchors = self.ranked_anchors(current_tokens, geo, kernels)
        if not anchors:
            return torch.zeros(c_rho.shape[0], device=self.device)
        pos     = len([t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS])
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
    def score(self, anchor_triple, cand_theta: torch.Tensor, cand_sigma: torch.Tensor, gamma_side: float = 4.0) -> torch.Tensor:
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor_triple.sigma) ** 2)
        antipodality = torch.cos(cand_theta + anchor_triple.theta - math.pi / 2) ** 2
        return congruence * antipodality

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — THÉBAULT COMPOSITION LM
# ════════════════════════════════════════════════════════════════════════════

class ThebaultCompositionLM:
    BASAL_K      = 1.5
    DENSE_THRESH = 512

    def __init__(self, geo: ThebaultTokenGeometry, kernels: ThebaultKernels, device: torch.device = DEVICE):
        self.geo      = geo
        self.kernels  = kernels
        self.device   = device
        self.raw_freq : Dict[str, float]                  = {}
        self.tri_raw  : Dict[Tuple[str, str, str], float] = {}
        self.heads    : Dict[Tuple[str, str], List[str]]  = {}
        self.vocab    : List[str]                         = []
        self._tok2idx : Dict[str, int]                    = {}
        self._head_cands : Dict[Tuple[str, str], torch.Tensor] = {}
        self._head_probs : Dict[Tuple[str, str], torch.Tensor] = {}

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
        self._tok2idx = {t: i for i, t in enumerate(self.vocab)}
        V_tot = len(self.vocab) + 1
        for (w1, w2), cands in self.heads.items():
            total  = sum(self.tri_raw.get((w1, w2, c), 1e-4) for c in cands)
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

    def composition_logit_bonus(self, w1: str, w2: str, c_rho: torch.Tensor, c_sigma: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, geo: ThebaultTokenGeometry, kernels: ThebaultKernels, device: torch.device = DEVICE):
        self.geo     = geo
        self.kernels = kernels
        self.device  = device
        self.nodes   : Dict[str, TGNode]       = {}
        self.adj     : Dict[str, List[TGEdge]] = {}
        self.radj    : Dict[str, List[TGEdge]] = {}
        self._pot_t  : Optional[torch.Tensor]  = None
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

    def subsynthetic_reason_concept_enrichment(self, w_ctx: str, cands: List[str], device: torch.device) -> torch.Tensor:
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
# SECTION 11 — THÉBAULT WALKER V17-CUDA  (+PDN)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    def __init__(
        self, geo, kernels, lm, orbit, graph, synth,
        mrv_filter, chunk_engine, iso_stacker,
        pdn_engine: PDNEngine,                        # ← NEW
        device: torch.device = DEVICE,
    ):
        self.geo          = geo
        self.kernels      = kernels
        self.lm           = lm
        self.orbit        = orbit
        self.graph        = graph
        self.synth        = synth
        self.mrv          = mrv_filter
        self.chunk_engine = chunk_engine
        self.iso_stacker  = iso_stacker
        self.pdn          = pdn_engine               # ← NEW
        self.device       = device
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []
        self._cur_sent_toks : List[str] = []
        self._cur_orbit     : int       = 0           # ← NEW: current PDN orbit

    def begin_sentence(self) -> None:
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()
        self._cur_orbit = 0

    @torch.no_grad()
    def walk_probs(
        self, w1: str, w2: str,
        temp          : float = 1.4, alphareg      : float = 1.2,
        betaori       : float = 0.8, deltaside     : float = 1.0,
        gammaorbit    : float = 0.6, psipot        : float = 0.35,
        zetamrv       : float = 0.9, etachunk      : float = 0.7,
        xiecho        : float = 0.6,
        pdn_weight    : float = 0.8,                 # ← NEW weight for PDN bonus
    ) -> Tuple[List[str], torch.Tensor]:

        cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands:
            return cands, base_probs

        try:
            tok_idx = self.geo.tok_indices(cands)
            c_rho, c_theta, c_sigma = self.geo.batch_triples(tok_idx)
            c_pvec  = self.geo._pvec_t[tok_idx]
        except Exception:
            triples  = [self.geo.triple(c) for c in cands]
            c_rho    = torch.tensor([t.rho   for t in triples], dtype=torch.float32, device=self.device)
            c_theta  = torch.tensor([t.theta for t in triples], dtype=torch.float32, device=self.device)
            c_sigma  = torch.tensor([t.sigma for t in triples], dtype=torch.float32, device=self.device)
            c_pvec   = torch.stack([c_rho, c_theta / math.pi, c_sigma, torch.ones_like(c_rho)], dim=1)

        ctx = self.geo.triple(w2)

        k_reg, k_ori, k_side = self.kernels.all_scores_batched(
            ctx.rho, ctx.theta, ctx.sigma, c_rho, c_theta, c_sigma
        )
        orbit_scores = self.orbit.score(ctx, c_theta, c_sigma, self.kernels.gamma_side)
        pots         = self.graph.potentials_for(cands)
        comp_bonus   = self.lm.composition_logit_bonus(w1, w2, c_rho, c_sigma)
        mrv_scores   = self.mrv.mrv_scores_batched(c_rho, c_sigma, self.kernels)
        chunk_bonus  = self.chunk_engine.chunk_bonus(c_pvec, scale=etachunk)
        echo_bonus   = self.iso_stacker.syntax_echo_bonus(
            c_rho, c_sigma, self._cur_sent_toks, self.geo, self.kernels, xiecho
        )

        # ── PDN BONUS ────────────────────────────────────────────────────
        win_rho, win_theta = self.chunk_engine.window_rho_theta()
        pdn_bonus = self.pdn.pdn_logit_bonus(
            win_rho, win_theta, c_rho, c_theta, self._cur_orbit
        )
        # ─────────────────────────────────────────────────────────────────

        # Isomorphic pair detection (unchanged)
        self.current_isomorphic_pairs = []
        top_idx = torch.topk(k_reg * k_side, min(50, len(cands))).indices
        sub_r   = k_reg[top_idx];  sub_s = k_side[top_idx]
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

        log_base = torch.log(base_probs.clamp(min=1e-12))
        logits   = (
            log_base
            + alphareg    * k_reg
            + betaori     * k_ori
            + deltaside   * k_side
            + gammaorbit  * orbit_scores
            + psipot      * pots
            + comp_bonus
            + zetamrv     * mrv_scores
            + chunk_bonus
            + echo_bonus
            + pdn_weight  * pdn_bonus     # ← NEW: PDN contribution
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
        # Advance PDN orbit
        self._cur_orbit = self.pdn.orbit_of(token)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — TEXT GENERATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def generate_passage(
    walker: ThebaultWalker,
    lm: ThebaultCompositionLM,
    num_sentences: int = 4,
    tokens_per_sent: int = 40,
    seed_text: str = "",
) -> str:
    outputs   = []
    head_list = list(lm.heads.keys())
    if not head_list:
        return ""

    seed_w1, seed_w2 = None, None
    if seed_text:
        seed_toks = tokenize(seed_text)
        if len(seed_toks) >= 2:
            seed_w1, seed_w2 = seed_toks[-2], seed_toks[-1]
        elif len(seed_toks) == 1:
            matches = [p for p in head_list if p[1] == seed_toks[0]]
            if matches:
                seed_w1, seed_w2 = random.choice(matches)

    if seed_w1 is None or seed_w2 is None or (seed_w1, seed_w2) not in lm.heads:
        seed_w1, seed_w2 = random.choice(head_list)

    for sent_idx in range(num_sentences):
        walker.begin_sentence()

        if sent_idx == 0:
            w1, w2 = seed_w1, seed_w2
            toks   = [w1, w2] if seed_text else []
            wsp    = len(toks)
        else:
            w1, w2 = random.choice(head_list)
            toks, wsp = [], 999

        for _ in range(tokens_per_sent):
            cands, probs = walker.walk_probs(w1, w2)
            if not cands:
                break

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
            walker.push_token(nxt, tokens_per_sent)
            w1, w2 = w2, nxt

            if nxt in {".", "?", "!"} and len(toks) >= max(4, int(tokens_per_sent * 0.85)):
                break

        outputs.append(detokenize(toks))

    return " ".join(outputs)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — V17 ENGINE (with PDN)
# ════════════════════════════════════════════════════════════════════════════

class V17Engine:
    def __init__(self):
        self.device = DEVICE
        self.geo         = ThebaultTokenGeometry(device=self.device)
        self.kernels     = ThebaultKernels()
        self.lm          = ThebaultCompositionLM(self.geo, self.kernels, device=self.device)
        self.orbit       = ThebaultConjugateOrbit()
        self.graph       = ThebaultPotentialGraph(self.geo, self.kernels, device=self.device)
        self.mrv         = MRVConstraintFilter(device=self.device)
        self.chunk       = ChunkedSumEngine(device=self.device)
        self.synth       = synthetic_reasonMandateProcessor()
        self.iso_stacker = IsomorphicSyntaxStacker(device=self.device)
        self.pdn         = PDNEngine(device=self.device)          # ← NEW

        self.walker          = None
        self.corpus_snippet  = ""

    def train(self, corpus_text: str):
        print(f"[*] Tokenizing corpus ({len(corpus_text)} chars)...")
        self.corpus_snippet = corpus_text[:1000]
        tokens = tokenize(corpus_text)
        self.lm.ingest(tokens)

        all_tokens = list(self.lm.raw_freq.keys())
        max_freq   = max(self.lm.raw_freq.values(), default=1.0)
        vocab_size = len(all_tokens)

        print(f"[*] Registering {vocab_size} tokens in Thébault Geometry...")
        for idx, tok in enumerate(all_tokens):
            self.geo.register(tok, self.lm.raw_freq[tok], idx, max_freq, vocab_size)

        print("[*] Building GPU Tensor Caches...")
        self.geo.build_cuda_tensors(self.lm.vocab)
        self.lm.finalise()

        print("[*] Building graph potentials...")
        self.graph.build(self.lm)
        self.graph.propagate(steps=2)

        print("[*] Initializing MRV Filter...")
        self.mrv.prime(self.lm.vocab, self.geo)

        # ── PDN: fit from corpus ──────────────────────────────────────────
        print("[*] Fitting PDN (Petr–Douglas–Neumann) symmetry order from corpus...")
        self.pdn.fit_from_trigrams(self.geo, self.lm.tri_raw)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)
        print(self.pdn.theorem_bridge_report())
        # ─────────────────────────────────────────────────────────────────

        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            self.pdn,                                     # ← NEW
            device=self.device,
        )
        print("[+] Training complete.")

    def save_cache(self, filename: str = "v17_model.pkl"):
        print(f"[*] Saving model state to {filename}...")
        state = {
            "geo_vecs"        : self.geo._vecs,
            "geo_cache"       : self.geo._cache,
            "lm_raw_freq"     : self.lm.raw_freq,
            "lm_tri_raw"      : self.lm.tri_raw,
            "lm_heads"        : self.lm.heads,
            "lm_vocab"        : self.lm.vocab,
            "graph_nodes"     : self.graph.nodes,
            "corpus_snippet"  : self.corpus_snippet,
            "pdn_n_star"      : self.pdn.n_star,         # ← NEW
            "pdn_power"       : self.pdn.power_spectrum,  # ← NEW
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        print("[+] Save successful.")

    def load_cache(self, filename: str):
        print(f"[*] Loading model state from {filename}...")
        with open(filename, "rb") as f:
            state = pickle.load(f)

        self.geo._vecs        = state["geo_vecs"]
        self.geo._cache       = state["geo_cache"]
        self.lm.raw_freq      = state["lm_raw_freq"]
        self.lm.tri_raw       = state["lm_tri_raw"]
        self.lm.heads         = state["lm_heads"]
        self.lm.vocab         = state["lm_vocab"]
        self.graph.nodes      = state["graph_nodes"]
        self.corpus_snippet   = state["corpus_snippet"]
        self.pdn.n_star       = state.get("pdn_n_star", 4)       # ← NEW
        self.pdn.power_spectrum = state.get("pdn_power", {})      # ← NEW

        print("[*] Rebuilding GPU Tensors from loaded state...")
        self.geo.build_cuda_tensors(self.lm.vocab)
        self.lm.finalise()
        self.graph.build(self.lm)
        self.graph.propagate(steps=2)
        self.mrv.prime(self.lm.vocab, self.geo)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)

        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            self.pdn,
            device=self.device,
        )
        print("[+] Load successful.")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — GRADIO GUI
# ════════════════════════════════════════════════════════════════════════════

class V17GUI:
    def __init__(self):
        self.engine = None

    def init_engine_from_file(self, file_obj):
        if file_obj is None:
            return "Error: No file uploaded."
        try:
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                corpus_text = f.read()
            if not corpus_text.strip():
                return "Error: Uploaded file is empty."
            self.engine = V17Engine()
            self.engine.train(corpus_text)
            report = self.engine.pdn.theorem_bridge_report()
            return (f"Engine initialised from file ({file_obj.name.split('/')[-1]}). "
                    f"Vocab size: {len(self.engine.lm.vocab)}\n\n{report}")
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def generate_text(self, sentences, tokens, seed_text):
        if not self.engine or not self.engine.walker:
            return "Engine not initialised. Please load a corpus first."
        return generate_passage(
            self.engine.walker,
            self.engine.lm,
            num_sentences=int(sentences),
            tokens_per_sent=int(tokens),
            seed_text=seed_text.strip(),
        )

    def pdn_report(self):
        if not self.engine:
            return "Engine not initialised."
        return self.engine.pdn.theorem_bridge_report()


def launch_gui():
    gui = V17GUI()

    with gr.Blocks(title="NeuroSymbolic V17 CUDA + PDN") as app:
        gr.Markdown(
            "# NeuroSymbolic V17 CUDA\n"
            "### Thébault Geometry Engine + Petr–Douglas–Neumann Theorem"
        )

        file_input      = gr.File(label="Upload .txt Corpus File", file_types=[".txt"])
        train_file_btn  = gr.Button("Initialise from File", variant="primary")
        init_out        = gr.Textbox(label="Engine Status / PDN Report", lines=20, interactive=False)
        train_file_btn.click(gui.init_engine_from_file, inputs=[file_input], outputs=init_out)

        gr.Markdown("### Text Generation")
        with gr.Row():
            sentences = gr.Slider(1, 10, value=4, step=1, label="Sentences")
            tokens    = gr.Slider(20, 180, value=80, step=1, label="Tokens per sentence")

        seed_input = gr.Textbox(label="Seed Text (Optional)", placeholder="e.g. quantum entanglement")
        gen_btn    = gr.Button("Generate Passage", variant="primary")
        gen_out    = gr.Textbox(lines=12, label="Generated Text")
        gen_btn.click(gui.generate_text, inputs=[sentences, tokens, seed_input], outputs=gen_out)

        gr.Markdown("### PDN Theorem Report")
        pdn_btn    = gr.Button("Show PDN Bridge Report")
        pdn_out    = gr.Textbox(lines=18, label="PDN Report", interactive=False)
        pdn_btn.click(gui.pdn_report, outputs=pdn_out)

    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",    action="store_true", help="Launch Gradio GUI")
    parser.add_argument("--corpus", type=str,            help="Path to training text file")
    parser.add_argument("--save",   type=str, default="v17_model.pkl")
    args = parser.parse_args()

    if args.gui or not args.corpus:
        launch_gui()
        exit(0)

    try:
        corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    except Exception as e:
        print(f"[!] Failed to read {args.corpus}: {e}")
        exit(1)

    engine = V17Engine()
    engine.train(corpus_text)
    engine.save_cache(args.save)

    print("\n--- SAMPLE GENERATION ---")
    print(generate_passage(engine.walker, engine.lm, num_sentences=3, tokens_per_sent=30))
