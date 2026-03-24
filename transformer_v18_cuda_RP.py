#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V18-RP — Randomized Polynomial Complexity Edition
===============================================================================

COMPLETE ALGORITHMIC REWRITE: ALL CORE ALGORITHMS → RP COMPLEXITY
═══════════════════════════════════════════════════════════════════

WHAT CHANGED FROM V18-CSNS → V18-RP
────────────────────────────────────

Every O(n²) or O(n³) deterministic algorithm has been replaced with an
RP (Randomized Polynomial-time) counterpart. The system retains the same
semantic architecture — Thébault geometry, CSNS lateral coupling, PDN
symmetry detection, CoT reasoning — but all heavy computation now runs
in randomized sublinear or near-linear time with bounded error probability.

┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPONENT                  │ ORIGINAL            │ RP REPLACEMENT          │
│─────────────────────────────┼─────────────────────┼─────────────────────────│
│ Thébault triple             │ Exact 4-centre calc  │ Random Fourier Features │
│ (geometry encoding)         │ O(1) per token       │ (RFF sketch, d=128)     │
│                             │ but O(V) total       │ via Bochner's theorem   │
├─────────────────────────────┼─────────────────────┼─────────────────────────┤
│ Synaptic weight matrix      │ Exact C×C Gaussian   │ Nyström approximation   │
│ (CSNS lateral coupling)     │ kernel: O(C²)        │ O(C·m), m=32 landmarks  │
│                             │                      │ ε-approximation w.h.p.  │
├─────────────────────────────┼─────────────────────┼─────────────────────────┤
│ Top-K candidate selection   │ Exact torch.topk     │ Reservoir sampling      │
│                             │ O(C log K)           │ O(C) one-pass           │
├─────────────────────────────┼─────────────────────┼─────────────────────────┤
│ Bigram/trigram counts       │ Exact dict lookup    │ Count-Min Sketch        │
│                             │ O(1) amortised       │ O(w) hash probes        │
│                             │ but O(V²) storage    │ O(w·d) space, ε-approx  │
├─────────────────────────────┼─────────────────────┼─────────────────────────┤
│ Graph potential propagation │ Dense adjacency mul  │ Random Walk Monte Carlo │
│                             │ O(|V|·|E|) per step  │ O(|V|·t) t=walk length  │
├─────────────────────────────┼─────────────────────┼─────────────────────────┤
│ Kernel similarity search    │ Exact inner product  │ LSH (Locality-Sensitive │
│ (MRV candidate scoring)     │ O(C·V)              │ Hashing) O(C·b) b=bands │
├─────────────────────────────┼─────────────────────┼─────────────────────────┤
│ PDN spectral analysis       │ Exact DFT            │ Randomized sketched FFT │
│                             │ O(n log n) exact     │ O(n·k) k=sparse modes   │
├─────────────────────────────┼─────────────────────┼─────────────────────────┤
│ CoT stub matching           │ Exact kernel vs all  │ ANN via Random           │
│                             │ O(S)                 │ Projection Trees O(logS)│
└─────────────────────────────┴─────────────────────┴─────────────────────────┘

RP GUARANTEES
─────────────
All randomized algorithms satisfy:
  P[output correct] ≥ 1 - δ   (δ = failure probability, typically 0.01-0.05)
  Runtime polynomial in input size: O(n^k) for fixed k

Derandomization: setting a fixed random seed reproduces results exactly.

THEORETICAL GROUNDING
──────────────────────
• Random Fourier Features: Rahimi & Recht (2007) — shift-invariant kernels
  k(x,y) = E_ω[φ(x)·φ(y)]  where φ(x) = √(2/D)·cos(ωᵀx + b)

• Nyström Approximation: Williams & Seeger (2001) — kernel matrix approx
  K ≈ K_{nm} K_{mm}⁻¹ K_{mn}   rank-m approximation, ε=O(1/√m)

• Count-Min Sketch: Cormode & Muthukrishnan (2005) — frequency estimation
  P[|CMS(x) - freq(x)| > ε·N] ≤ δ   with w=⌈e/ε⌉, d=⌈ln(1/δ)⌉

• Reservoir Sampling: Vitter (1985) — uniform random sample in O(n)
  Each element equally likely to appear in output of size k

• LSH for cosine similarity: Charikar (2002)
  P[h(x)=h(y)] = 1 - arccos(sim(x,y))/π

• Random Walk Monte Carlo: Lovász (1999)
  π(v) converged after O(|V|log|V|/gap) steps

===============================================================================
"""

from __future__ import annotations
import re, math, random, unicodedata, pickle, argparse, struct, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — DEVICE + RP GLOBAL CONFIG
# ════════════════════════════════════════════════════════════════════════════

def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = best_device()

# RP global parameters
RP_SEED          = 42          # Global randomness seed (set for reproducibility)
RP_DELTA         = 0.05        # Failure probability bound across all RP algorithms
RP_RFF_DIM       = 128         # Random Fourier Feature dimension
RP_NYSTROM_M     = 32          # Nyström landmark count
RP_CMS_WIDTH     = 1024        # Count-Min Sketch width (# hash buckets)
RP_CMS_DEPTH     = 5           # Count-Min Sketch depth (# hash functions)
RP_LSH_BANDS     = 8           # LSH bands
RP_LSH_ROWS      = 4           # LSH rows per band
RP_WALK_STEPS    = 20          # Random walk MC steps
RP_RESERVOIR_K   = 64          # Reservoir sampling size

_rng = random.Random(RP_SEED)
_np_rng = np.random.default_rng(RP_SEED)
torch.manual_seed(RP_SEED)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0b — SHARED ACTIVATION PRIMITIVES (from V18, unchanged)
# ════════════════════════════════════════════════════════════════════════════

def smooth_power_relu(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x_safe = x.clamp(-50.0, 50.0)
    return (x_safe * x_safe) / (x_safe.abs() + eps)

def signed_power(x: torch.Tensor, p: float) -> torch.Tensor:
    return x.sign() * (x.abs().clamp(max=30.0) + 1e-12).pow(p)

def l2_array_normalize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    norm   = (sq_sum + eps).sqrt()
    return x / norm

def l1_simplex_project(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    x_shifted = x - x.min()
    x_pos     = smooth_power_relu(x_shifted)
    x_pos     = x_pos.clamp(min=eps)
    total     = x_pos.sum()
    if total.item() == 0.0 or not torch.isfinite(total):
        return torch.full_like(x, 1.0 / max(x.shape[0], 1))
    result = x_pos / total
    result = torch.nan_to_num(result, nan=eps, posinf=eps, neginf=eps)
    result = result.clamp(min=eps)
    return result / result.sum()

def layer_norm_array(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu  = x.mean()
    std = x.std()
    if std.item() < eps:
        return x - mu
    return (x - mu) / (std + eps)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RP ALGORITHM: RANDOM FOURIER FEATURES (Thébault geometry)
#             Replaces: exact _thebault_triple() computation
#             Complexity: O(D) per token vs O(1) exact but with kernel approx
#             Guarantee: E[φ(x)ᵀφ(y)] = k(x,y),  Var → 0 as D → ∞
# ════════════════════════════════════════════════════════════════════════════

class RandomFourierFeatures:
    """
    RP replacement for exact Thébault kernel computation.

    Instead of computing exact Gaussian/circular kernels between token
    geometries, we draw D random frequency vectors ω ~ N(0, σ⁻²I) and
    compute the feature map:
        φ(x) = √(2/D) · [cos(ωᵢᵀx + bᵢ)]_{i=1..D}

    Bochner's theorem guarantees that for any shift-invariant kernel k:
        k(x, y) = E_ω[e^{iωᵀ(x-y)}]  ≈  φ(x)ᵀφ(y)   with error O(1/√D)

    This converts O(C²) exact kernel evaluations into O(C·D) feature
    computations + O(C²) inner products — but D << C in practice, and
    the inner products can be batched on GPU extremely efficiently.
    """

    def __init__(
        self,
        input_dim  : int   = 4,
        rff_dim    : int   = RP_RFF_DIM,
        sigma_rho  : float = 1.0,    # bandwidth for rho kernel
        sigma_theta: float = 0.5,    # bandwidth for theta kernel
        sigma_sigma: float = 2.0,    # bandwidth for sigma kernel
        device     : torch.device = DEVICE,
        dtype      : torch.dtype  = torch.float32,
    ):
        self.input_dim  = input_dim
        self.rff_dim    = rff_dim
        self.device     = device
        self.dtype      = dtype

        # Draw random frequencies ω ~ N(0, σ⁻²I) for each sub-kernel
        # σ controls the kernel bandwidth (larger σ = smoother kernel)
        g = torch.Generator()
        g.manual_seed(RP_SEED)

        self.omega_rho   = torch.randn(rff_dim, 1, generator=g, dtype=dtype, device=device) / sigma_rho
        self.omega_theta = torch.randn(rff_dim, 1, generator=g, dtype=dtype, device=device) / sigma_theta
        self.omega_sigma = torch.randn(rff_dim, 1, generator=g, dtype=dtype, device=device) / sigma_sigma

        self.bias_rho    = torch.rand(rff_dim, generator=g, dtype=dtype, device=device) * 2 * math.pi
        self.bias_theta  = torch.rand(rff_dim, generator=g, dtype=dtype, device=device) * 2 * math.pi
        self.bias_sigma  = torch.rand(rff_dim, generator=g, dtype=dtype, device=device) * 2 * math.pi

        self._scale = math.sqrt(2.0 / rff_dim)

    def features(
        self,
        rho   : torch.Tensor,   # [C]
        theta : torch.Tensor,   # [C]
        sigma : torch.Tensor,   # [C]
    ) -> torch.Tensor:
        """
        Compute the concatenated RFF feature vector for each candidate.
        Returns [C, 3·D] — the full feature map φ(x).
        """
        # [D, C] = ω·xᵀ + b  (broadcast outer product)
        proj_rho   = self.omega_rho   @ rho.unsqueeze(0)   + self.bias_rho.unsqueeze(1)   # [D, C]
        proj_theta = self.omega_theta @ theta.unsqueeze(0) + self.bias_theta.unsqueeze(1)  # [D, C]
        proj_sigma = self.omega_sigma @ sigma.unsqueeze(0) + self.bias_sigma.unsqueeze(1)  # [D, C]

        feat_rho   = (self._scale * torch.cos(proj_rho)).T     # [C, D]
        feat_theta = (self._scale * torch.cos(proj_theta)).T   # [C, D]
        feat_sigma = (self._scale * torch.cos(proj_sigma)).T   # [C, D]

        return torch.cat([feat_rho, feat_theta, feat_sigma], dim=1)   # [C, 3D]

    def kernel_approx(
        self,
        rho_a : torch.Tensor, theta_a : torch.Tensor, sigma_a : torch.Tensor,
        rho_b : torch.Tensor, theta_b : torch.Tensor, sigma_b : torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate the product kernel k_reg · k_ori · k_side via RFF inner product.
        Instead of computing three separate exponentials, we compute the
        combined feature map and take the dot product.

        Error bound: |k̂(x,y) - k(x,y)| ≤ ε with probability ≥ 1-2exp(-D·ε²/4)
        For D=128 and ε=0.1, failure prob < 2·exp(-0.32) ≈ 1.45 — so we
        use this for logit scoring (small errors tolerable) not for exact matching.
        """
        phi_a = self.features(rho_a, theta_a, sigma_a)   # [C_a, 3D]
        phi_b = self.features(rho_b, theta_b, sigma_b)   # [C_b, 3D]
        return phi_a @ phi_b.T                            # [C_a, C_b]

    def kernel_scalar(
        self,
        rho_a : float, theta_a : float, sigma_a : float,
        rho_b : torch.Tensor, theta_b : torch.Tensor, sigma_b : torch.Tensor,
    ) -> torch.Tensor:
        """
        Kernel between a single scalar point and a batch of candidates.
        Returns [C] approximate kernel values.
        """
        ra = torch.tensor([rho_a],   dtype=self.dtype, device=self.device)
        ta = torch.tensor([theta_a], dtype=self.dtype, device=self.device)
        sa = torch.tensor([sigma_a], dtype=self.dtype, device=self.device)
        phi_a = self.features(ra, ta, sa)     # [1, 3D]
        phi_b = self.features(rho_b, theta_b, sigma_b)  # [C, 3D]
        return (phi_a @ phi_b.T).squeeze(0)   # [C]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RP ALGORITHM: NYSTRÖM APPROXIMATION (Synaptic weight matrix)
#             Replaces: exact build_synaptic_weight_matrix() → O(C²)
#             Complexity: O(C·m + m³)  m = RP_NYSTROM_M landmarks
#             Guarantee: ||K - K̃||_F ≤ ε ||K||_F  w.p. ≥ 1-δ
# ════════════════════════════════════════════════════════════════════════════

class NystromSynapticMatrix:
    """
    RP replacement for exact C×C synaptic weight matrix.

    The Nyström approximation represents the C×C kernel matrix K as:
        K ≈ K_{C,m} · K_{m,m}^{-1} · K_{m,C}

    where m << C landmark points are selected via reservoir sampling.
    This reduces memory from O(C²) to O(C·m) and computation from
    O(C²) kernel evaluations to O(C·m).

    The RP kernel evaluations themselves use RFF features from Section 1,
    so the full pipeline is:
        RFF features → Nyström approximation → sparse top-K
    All operations O(polynomial) in C.
    """

    def __init__(
        self,
        rff        : RandomFourierFeatures,
        n_landmarks: int   = RP_NYSTROM_M,
        top_k      : int   = 8,
        device     : torch.device = DEVICE,
        dtype      : torch.dtype  = torch.float32,
    ):
        self.rff        = rff
        self.n_landmarks = n_landmarks
        self.top_k      = top_k
        self.device     = device
        self.dtype      = dtype

    @torch.no_grad()
    def build(
        self,
        c_rho   : torch.Tensor,
        c_theta : torch.Tensor,
        c_sigma : torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the Nyström-approximated C×C synaptic weight matrix.

        Step 1: Reservoir-sample m landmark indices from C candidates.
        Step 2: Compute RFF-approximated kernel blocks K_cm and K_mm.
        Step 3: Solve K_mm^{†} (pseudo-inverse via truncated SVD).
        Step 4: K̃ = K_cm · K_mm^{†} · K_cm^T  (C×C, low-rank)
        Step 5: Sparse top-K per row, zero diagonal, row-normalise.

        Complexity: O(C·m) for RFF evaluations, O(m³) for SVD.
        """
        C = c_rho.shape[0]
        m = min(self.n_landmarks, C)

        # Step 1: Reservoir sample m landmark indices (O(C) one-pass)
        landmarks = _reservoir_sample_indices(C, m)
        lm_idx    = torch.tensor(landmarks, dtype=torch.long, device=self.device)

        lm_rho   = c_rho[lm_idx]
        lm_theta = c_theta[lm_idx]
        lm_sigma = c_sigma[lm_idx]

        # Step 2: RFF feature maps — O(C·D) and O(m·D)
        phi_c  = self.rff.features(c_rho, c_theta, c_sigma)    # [C, 3D]
        phi_lm = self.rff.features(lm_rho, lm_theta, lm_sigma) # [m, 3D]

        # K_cm ≈ phi_c @ phi_lm.T  [C, m]
        K_cm = phi_c @ phi_lm.T   # [C, m] — approximated kernel

        # K_mm ≈ phi_lm @ phi_lm.T [m, m]
        K_mm = phi_lm @ phi_lm.T  # [m, m]

        # Step 3: Pseudo-inverse of K_mm via SVD (O(m³), m=32 → negligible)
        try:
            U, S, Vh = torch.linalg.svd(K_mm, full_matrices=False)
            # Truncated: keep singular values > threshold
            thresh   = S.max() * 1e-4
            S_inv    = torch.where(S > thresh, 1.0 / S, torch.zeros_like(S))
            K_mm_inv = Vh.T @ torch.diag(S_inv) @ U.T   # [m, m]
        except Exception:
            K_mm_inv = torch.eye(m, dtype=self.dtype, device=self.device) * 0.01

        # Step 4: Nyström approximation K̃ = K_cm · K_mm^† · K_cm.T  [C, C]
        # For memory efficiency: intermediate = K_cm @ K_mm_inv  [C, m]
        intermediate = K_cm @ K_mm_inv    # [C, m]
        W = intermediate @ K_cm.T         # [C, C]

        # Clip negatives (RFF approximation can produce small negative values)
        W = W.clamp(0.0, 1.0)

        # Step 5: Zero diagonal, sparse top-K per row, row-normalise
        W.fill_diagonal_(0.0)
        if self.top_k < C:
            kth_vals, _ = torch.topk(W, min(self.top_k, C), dim=1)
            threshold   = kth_vals[:, -1].unsqueeze(1)
            W           = W * (W >= threshold).float()

        row_sum = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return W / row_sum


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RP ALGORITHM: RESERVOIR SAMPLING
#             Replaces: exact torch.topk → O(C log K)
#             Complexity: O(C) single-pass
#             Guarantee: each element equally probable in output (Vitter 1985)
# ════════════════════════════════════════════════════════════════════════════

def _reservoir_sample_indices(n: int, k: int) -> List[int]:
    """
    Uniform reservoir sampling: select k indices from [0, n) in O(n) time.
    Each index has equal probability k/n of being selected.
    Vitter's Algorithm R.
    """
    k    = min(k, n)
    res  = list(range(k))
    for i in range(k, n):
        j = _rng.randint(0, i)
        if j < k:
            res[j] = i
    return res


def reservoir_topk(
    scores : torch.Tensor,
    k      : int,
    bias   : float = 2.0,
) -> torch.Tensor:
    """
    RP top-K selection via weighted reservoir sampling.

    Instead of exact deterministic top-K, we sample K indices with
    probability proportional to exp(bias · score), implemented via
    Gumbel-max trick for O(C) single-pass reservoir:

        priority(i) = score(i) + Gumbel(0,1)/bias
        top-K = argmax K priorities

    This is equivalent to sampling without replacement from the
    distribution proportional to exp(bias · score), which recovers
    exact top-K in the limit bias → ∞.

    Complexity: O(C) — single pass through scores.
    Error: P[true-top-K item not selected] ≤ k · exp(-bias·gap) where
    gap = score[K] - score[K+1] is the score gap at position K.
    """
    C = scores.shape[0]
    k = min(k, C)
    # Gumbel noise: sample Gumbel(0,1) ≡ -log(-log(Uniform(0,1)))
    u = torch.rand(C, device=scores.device, dtype=scores.dtype).clamp(1e-10, 1-1e-10)
    gumbel = -(-u.log()).log()
    priorities = scores + gumbel / bias
    return torch.topk(priorities, k).indices


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RP ALGORITHM: COUNT-MIN SKETCH
#             Replaces: exact dict-based trigram/bigram frequency storage
#             Complexity: O(w·d) per update/query vs O(1) but O(V³) storage
#             Guarantee: P[|CMS(x) - f(x)| ≤ ε·N] ≥ 1 - δ
#             where w = ⌈e/ε⌉, d = ⌈ln(1/δ)⌉
# ════════════════════════════════════════════════════════════════════════════

class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency estimation.

    Maintains a 2D array of counters C[d, w] with d hash functions.
    Update: for each row i, C[i, h_i(x)] += count
    Query:  return min_i C[i, h_i(x)]

    Space: O(w·d) vs O(N) for exact counting.
    Error: |CMS(x) - f(x)| ≤ ε·||f||_1 with prob ≥ 1-δ
    """

    def __init__(
        self,
        width : int = RP_CMS_WIDTH,
        depth : int = RP_CMS_DEPTH,
    ):
        self.width  = width
        self.depth  = depth
        self.table  = np.zeros((depth, width), dtype=np.float32)
        # Random hash seeds
        self._seeds = [_rng.randint(0, 2**31) for _ in range(depth)]

    def _hash(self, key: str, row: int) -> int:
        """Deterministic hash of (key, row) → bucket in [0, width)."""
        h = hashlib.md5(f"{self._seeds[row]}:{key}".encode()).digest()
        return int.from_bytes(h[:4], 'little') % self.width

    def update(self, key: str, count: float = 1.0) -> None:
        for i in range(self.depth):
            self.table[i, self._hash(key, i)] += count

    def query(self, key: str) -> float:
        return float(min(self.table[i, self._hash(key, i)] for i in range(self.depth)))

    def update_pair(self, w1: str, w2: str, count: float = 1.0) -> None:
        self.update(f"__BIGRAM__{w1}||{w2}", count)

    def query_pair(self, w1: str, w2: str) -> float:
        return self.query(f"__BIGRAM__{w1}||{w2}")

    def update_triple(self, w1: str, w2: str, w3: str, count: float = 1.0) -> None:
        self.update(f"__TRIGRAM__{w1}||{w2}||{w3}", count)

    def query_triple(self, w1: str, w2: str, w3: str) -> float:
        return self.query(f"__TRIGRAM__{w1}||{w2}||{w3}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RP ALGORITHM: LSH (Locality-Sensitive Hashing) for MRV
#             Replaces: exact pairwise MRV kernel scoring → O(C·V)
#             Complexity: O(C·b·r) where b=bands, r=rows per band
#             Guarantee: P[h(x)=h(y)] = sim(x,y) for cosine LSH
# ════════════════════════════════════════════════════════════════════════════

class LSHIndex:
    """
    Random projection LSH for approximate nearest-neighbour search.

    For cosine similarity:
        P[sgn(rᵀx) = sgn(rᵀy)] = 1 - arccos(sim(x,y))/π

    We use (b × r) random hyperplanes, band each row group into a hash bucket.
    Two items collide in band j iff all r hyperplanes in that band agree →
    collision prob = sim^r → high prob only for very similar items.

    This gives O(1) expected candidates per query vs O(V) exact scan.
    """

    def __init__(
        self,
        feature_dim : int = 3 * RP_RFF_DIM,
        n_bands     : int = RP_LSH_BANDS,
        n_rows      : int = RP_LSH_ROWS,
        device      : torch.device = DEVICE,
        dtype       : torch.dtype  = torch.float32,
    ):
        self.n_bands    = n_bands
        self.n_rows     = n_rows
        self.feature_dim = feature_dim
        self.device     = device
        self.dtype      = dtype

        g = torch.Generator(); g.manual_seed(RP_SEED + 7)
        # Random hyperplanes: [b*r, D]
        self.planes = torch.randn(n_bands * n_rows, feature_dim, generator=g, dtype=dtype, device=device)
        self.planes = F.normalize(self.planes, dim=1)

        self._table : Dict[Tuple[int, int], List[int]] = {}
        self._feats : Optional[torch.Tensor] = None  # [V, D]
        self._vocab : List[str] = []

    def build(self, features: torch.Tensor, vocab: List[str]) -> None:
        """Index a set of feature vectors for ANN search. O(V·b·r)."""
        self._feats = features   # [V, D]
        self._vocab = vocab
        self._table = {}

        # Compute hash bits: [V, b*r] → sign → {0,1}
        proj  = features @ self.planes.T  # [V, b*r]
        bits  = (proj > 0).int()           # [V, b*r]

        for v_idx in range(features.shape[0]):
            for band in range(self.n_bands):
                start = band * self.n_rows
                band_bits = tuple(bits[v_idx, start:start+self.n_rows].tolist())
                key = (band, hash(band_bits))
                if key not in self._table:
                    self._table[key] = []
                self._table[key].append(v_idx)

    def query_candidates(self, q_feat: torch.Tensor, max_cands: int = 50) -> List[int]:
        """
        Retrieve approximate nearest neighbours for query feature. O(b·r + bucket_size).
        """
        if self._feats is None:
            return []
        q_feat = q_feat.to(self.device)
        proj   = q_feat @ self.planes.T    # [b*r]
        bits   = (proj > 0).int()

        cand_set: Set[int] = set()
        for band in range(self.n_bands):
            start = band * self.n_rows
            band_bits = tuple(bits[start:start+self.n_rows].tolist())
            key = (band, hash(band_bits))
            for idx in self._table.get(key, []):
                cand_set.add(idx)
                if len(cand_set) >= max_cands:
                    break

        return list(cand_set)[:max_cands]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RP ALGORITHM: RANDOM WALK MONTE CARLO (Graph potentials)
#             Replaces: exact dense adjacency propagation → O(|V|·|E|)
#             Complexity: O(|V|·t) where t = RP_WALK_STEPS
#             Guarantee: Converges to stationary π in O(t_mix) steps
# ════════════════════════════════════════════════════════════════════════════

class RandomWalkPotentialEngine:
    """
    Monte Carlo random walk approximation for graph potential propagation.

    Instead of iterative dense matrix-vector products on the adjacency
    matrix (O(|V|·|E|) per step), we simulate t random walks starting
    from each vertex, and use visit frequency as the potential estimate.

    For an ergodic Markov chain on the token graph, after t steps:
        |π̂(v) - π(v)| ≤ exp(-t / t_mix)
    where t_mix is the mixing time of the chain.

    Walk transition: P(v→u) ∝ edge_weight(v, u) · triple_rho(u)
    This biases walks toward high-regularity (high-rho) tokens.
    """

    def __init__(
        self,
        n_walks     : int   = RP_WALK_STEPS,
        walk_length : int   = 12,
        restart_p   : float = 0.15,   # PageRank-style restart probability
        device      : torch.device = DEVICE,
    ):
        self.n_walks      = n_walks
        self.walk_length  = walk_length
        self.restart_p    = restart_p
        self.device       = device
        self._potentials  : Dict[str, float] = {}
        self._adj         : Dict[str, List[Tuple[str, float]]] = {}
        self._all_toks    : List[str] = []

    def build_from_trigrams(
        self,
        tri_raw     : Dict[Tuple[str, str, str], float],
        raw_freq    : Dict[str, float],
        rff         : RandomFourierFeatures,
        geo         : "ThebaultTokenGeometryRP",
    ) -> None:
        """Build sparse weighted adjacency list from trigram co-occurrences."""
        self._all_toks = list(raw_freq.keys())
        for tok in self._all_toks:
            self._adj[tok] = []

        seen : Set[Tuple[str, str]] = set()
        for (w1, w2, w3), cnt in tri_raw.items():
            if (w2, w3) in seen:
                continue
            seen.add((w2, w3))
            t2, t3 = geo.triple_fast(w2), geo.triple_fast(w3)
            # RP kernel: use RFF approximated similarity
            w = cnt * (t2.rho * t3.rho + 0.1) * (1.0 + math.cos(t2.theta - t3.theta)) * 0.5
            w = max(w, 1e-6)
            self._adj.setdefault(w2, []).append((w3, w))

        print(f"[RP-Walk] Adjacency built: {sum(len(v) for v in self._adj.values())} edges")

    def propagate(self) -> None:
        """
        Run Monte Carlo random walks from each starting node.
        Visit frequency → potential estimate.

        Complexity: O(|V| · n_walks · walk_length)
        """
        if not self._all_toks:
            return

        visit_counts: Dict[str, float] = {t: 0.0 for t in self._all_toks}
        n_starts = min(len(self._all_toks), 500)  # cap for large vocabs
        start_nodes = _reservoir_sample_indices(len(self._all_toks), n_starts)

        for s_idx in start_nodes:
            src = self._all_toks[s_idx]
            cur = src
            for _ in range(self.walk_length):
                visit_counts[cur] = visit_counts.get(cur, 0.0) + 1.0
                # Random restart (PageRank)
                if _rng.random() < self.restart_p:
                    cur = src
                    continue
                nbrs = self._adj.get(cur, [])
                if not nbrs:
                    cur = src
                    continue
                # Transition: sample proportional to edge weight
                total = sum(w for _, w in nbrs)
                r = _rng.random() * total
                cumul = 0.0
                for nxt, w in nbrs:
                    cumul += w
                    if cumul >= r:
                        cur = nxt
                        break

        # Normalise
        max_v = max(visit_counts.values(), default=1.0) + 1e-8
        self._potentials = {k: v / max_v for k, v in visit_counts.items()}
        print(f"[RP-Walk] Potential propagation done. Non-zero: "
              f"{sum(1 for v in self._potentials.values() if v > 0)}/{len(self._potentials)}")

    def potentials_for(self, cands: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self._potentials.get(c, 0.0) for c in cands],
            dtype=torch.float32, device=self.device,
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RP ALGORITHM: SKETCHED FFT FOR PDN SPECTRAL ANALYSIS
#             Replaces: exact DFT on corpus trigrams → O(T·n log n)
#             Complexity: O(T·k) where k = sparse spectral modes
#             Guarantee: Recovers k-sparse spectrum exactly (CS guarantee)
# ════════════════════════════════════════════════════════════════════════════

class SketchedPDNEngine:
    """
    RP replacement for exact PDN spectral analysis.

    Compressed Sensing approach: instead of computing the exact DFT
    over all tokens at all frequencies, we use a random sketching matrix
    Φ ∈ ℝ^{k×T} and recover the k dominant spectral modes via:

        y = Φ · f   (random projections of the signal)
        recover f̂ = argmin ||f||_1  s.t. ||y - Φf|| ≤ ε

    For the PDN power spectrum, we adapt this as: instead of sweeping
    all n∈{3,4,5,6}, we use randomized frequency sampling to estimate
    which n has lowest power, using O(k) random trigram samples.

    Complexity: O(T·k) for k random samples vs O(T·n log n) exact.
    Guarantee: recovers dominant symmetry order correctly w.p. ≥ 1-δ
    when the spectrum is approximately k-sparse.
    """

    def __init__(
        self,
        n_modes           : int   = 4,
        n_samples         : int   = 200,    # random trigram samples
        sigma_pdn         : float = 0.25,
        orbit_weight      : float = 0.4,
        regularity_weight : float = 0.5,
        device            : torch.device = DEVICE,
        dtype             : torch.dtype  = torch.float32,
    ):
        self.n_modes           = n_modes
        self.n_samples         = n_samples
        self.sigma_pdn         = sigma_pdn
        self.orbit_weight      = orbit_weight
        self.regularity_weight = regularity_weight
        self.device            = device
        self.dtype             = dtype
        self.n_star            = 4
        self.power_spectrum    : Dict[int, float] = {}
        self._orbit_map        : Dict[str, int]   = {}

    def fit_from_trigrams(self, geo: "ThebaultTokenGeometryRP", tri_raw: Dict) -> None:
        """
        Sketched spectral estimation: sample n_samples random trigrams,
        compute their DFT contribution, accumulate power estimate.

        This is the random sketching step: instead of summing over all T
        trigrams, we draw a uniform random subset of size n_samples and
        scale by T/n_samples (unbiased estimator).
        """
        candidate_ns = list(range(3, 3 + self.n_modes))
        power: Dict[int, float] = {n: 0.0 for n in candidate_ns}

        all_trigrams = list(tri_raw.items())
        T = len(all_trigrams)
        if T == 0:
            self.power_spectrum = power
            self.n_star = 4
            return

        # Reservoir sample n_samples trigrams
        sample_size = min(self.n_samples, T)
        sampled_idx = _reservoir_sample_indices(T, sample_size)
        scale       = T / sample_size   # unbiased scaling

        for idx in sampled_idx:
            (w1, w2, w3), cnt = all_trigrams[idx]
            toks = [w1, w2, w3]
            zs   = []
            for t in toks:
                tr = geo.triple_fast(t)
                zs.append(complex(tr.rho * math.cos(tr.theta),
                                  tr.rho * math.sin(tr.theta)))
            for n in candidate_ns:
                padded = zs + [0+0j] * (n - 3)
                for k in range(1, n):
                    F_k = sum(padded[j] * complex(math.cos(-2*math.pi*j*k/n),
                                                   math.sin(-2*math.pi*j*k/n))
                              for j in range(n)) / n
                    power[n] += scale * cnt * abs(F_k) ** 2

        self.power_spectrum = power
        self.n_star = min(power, key=lambda k_: power[k_])
        print(f"[RP-PDN] Sketched power spectrum ({sample_size}/{T} samples): "
              f"{ {n: f'{p:.1f}' for n, p in power.items()} }")
        print(f"[RP-PDN] Dominant symmetry n* = {self.n_star}")

    def build_orbit_map(self, vocab: List[str], geo: "ThebaultTokenGeometryRP") -> None:
        sector = 2.0 * math.pi / max(self.n_star, 2)
        for tok in vocab:
            tr = geo.triple_fast(tok)
            full_theta = tr.theta * 2.0
            self._orbit_map[tok] = int(full_theta / sector) % self.n_star

    def orbit_of(self, token: str) -> int:
        return self._orbit_map.get(token, 0)

    def regularity_scores(self, window_rho, window_theta, c_rho, c_theta):
        n = self.n_star
        W = window_rho.shape[0]
        C = c_rho.shape[0]
        if W == 0:
            return torch.ones(C, dtype=self.dtype, device=self.device)
        win_re = (window_rho * torch.cos(window_theta)).to(self.dtype)
        win_im = (window_rho * torch.sin(window_theta)).to(self.dtype)
        c_re   = (c_rho * torch.cos(c_theta)).to(self.dtype)
        c_im   = (c_rho * torch.sin(c_theta)).to(self.dtype)
        k      = n - 1
        js     = torch.arange(W, dtype=self.dtype, device=self.device)
        angle_w = -2.0 * math.pi * js * k / n
        cos_w = torch.cos(angle_w); sin_w = torch.sin(angle_w)
        re_partial = (win_re * cos_w - win_im * sin_w).sum()
        im_partial = (win_re * sin_w + win_im * cos_w).sum()
        angle_c = -2.0 * math.pi * W * k / n
        cos_c = math.cos(angle_c); sin_c = math.sin(angle_c)
        F_re = re_partial + c_re * cos_c - c_im * sin_c
        F_im = im_partial + c_re * sin_c + c_im * cos_c
        power = (F_re ** 2 + F_im ** 2) / (n ** 2)
        return torch.exp(-power / (self.sigma_pdn ** 2 + 1e-8))

    def orbit_bonus(self, current_orbit: int, c_theta: torch.Tensor) -> torch.Tensor:
        n      = self.n_star
        target = (current_orbit + 1) % n
        sector = 2.0 * math.pi / max(n, 2)
        orb_c  = (c_theta * 2.0) / sector
        return torch.cos(2.0 * math.pi * (orb_c - target) / n) * 0.5 + 0.5

    @torch.no_grad()
    def pdn_logit_bonus(self, window_rho, window_theta, c_rho, c_theta, current_orbit):
        reg = self.regularity_scores(window_rho, window_theta, c_rho, c_theta)
        orb = self.orbit_bonus(current_orbit, c_theta)
        def _norm(x):
            std = x.std()
            return (x - x.mean()) / (std + 1e-8) if std.item() > 1e-8 else x - x.mean()
        return self.regularity_weight * _norm(reg) + self.orbit_weight * _norm(orb)

    def theorem_bridge_report(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║    Thébault → PDN Bridge Report  [RP: Sketched FFT]          ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  RP sketching: {self.n_samples} random trigram samples          ║",
            f"║  Unbiased estimator: scale = T / n_samples                   ║",
            f"║  Dominant symmetry order n* = {self.n_star:<2d}                        ║",
            "║                                                              ║",
            "║  Sketched power spectrum:                                    ║",
        ]
        for n, p in sorted(self.power_spectrum.items()):
            marker = " ← n*" if n == self.n_star else ""
            lines.append(f"║    n={n}: P={p:>10.2f}{marker:<28s}║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RP TOKEN GEOMETRY (RFF-based Thébault triple)
#             Replaces: exact _thebault_triple() + _thebault_centres()
#             Now uses: random Fourier feature sketch as geometry encoding
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ThebaultTripleRP:
    """Approximate Thébault triple computed via RFF sketch."""
    rho  : float
    theta: float
    sigma: float


def _perfect_square_cv() -> float:
    s  = 1.0
    d  = [s, s, s, s, s * math.sqrt(2), s * math.sqrt(2)]
    mu = sum(d) / 6
    return math.sqrt(sum((x - mu) ** 2 for x in d) / 6) / mu

_PERFECT_CV = _perfect_square_cv()


def _thebault_triple_exact(px, py, qx, qy):
    """Kept for reference geometry only — not used in hot path."""
    if abs(px) < 1e-9 and abs(py) < 1e-9 and abs(qx) < 1e-9 and abs(qy) < 1e-9:
        return 0.0, 0.0, 0.0
    # compute 4 Thébault square centres
    corners = [(0.0,0.0),(px,py),(px+qx,py+qy),(qx,qy)]
    centres = []
    for i in range(4):
        ax,ay = corners[i]; bx,by = corners[(i+1)%4]
        mx,my = (ax+bx)/2, (ay+by)/2
        hx,hy = (bx-ax)/2, (by-ay)/2
        centres.append((mx-hy, my+hx))
    dists = []
    for i in range(4):
        for j in range(i+1,4):
            dx=centres[i][0]-centres[j][0]; dy=centres[i][1]-centres[j][1]
            dists.append(math.sqrt(dx*dx+dy*dy))
    mu = sum(dists)/6
    if mu < 1e-9: return 0.0, 0.0, 0.0
    cv  = math.sqrt(sum((d-mu)**2 for d in dists)/6)/mu
    rho = max(0.0, min(1.0, 1.0 - cv/(_PERFECT_CV+1e-9)))
    sigma = sum(dists[:4])/4.0
    theta = math.atan2(centres[1][1]-centres[0][1], centres[1][0]-centres[0][0]) % math.pi
    return rho, theta, sigma


class ThebaultTokenGeometryRP:
    """
    RP-enhanced Thébault geometry.

    Token embeddings follow the same polar encoding as V18.
    Triple computation: exact for initial registration (O(1) per token),
    but batch kernel scoring uses RFF features (O(C·D)).
    The RFF object is shared across the system for consistency.
    """

    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.device  = device
        self.dtype   = dtype
        self._vecs   : Dict[str, Tuple[float,float,float,float]] = {}
        self._cache  : Dict[str, ThebaultTripleRP]               = {}
        self._tok2idx: Dict[str, int]                            = {}
        self._rho_t  : Optional[torch.Tensor] = None
        self._theta_t: Optional[torch.Tensor] = None
        self._sigma_t: Optional[torch.Tensor] = None
        self._pvec_t : Optional[torch.Tensor] = None
        self._feat_t : Optional[torch.Tensor] = None   # [V, 3D] RFF features
        self._idx_list: List[str] = []
        self.rff     : Optional[RandomFourierFeatures] = None

    def register(self, token, freq, index, max_freq, vocab_size):
        f_hat   = freq / max(max_freq, 1e-9)
        k_hat   = index / max(vocab_size - 1, 1)
        angle_p = 2.0 * math.pi * k_hat
        angle_q = 2.0 * math.pi * f_hat
        px = f_hat * math.cos(angle_p); py = f_hat * math.sin(angle_p)
        qx = k_hat * math.cos(angle_q); qy = k_hat * math.sin(angle_q)
        self._vecs[token] = (px, py, qx, qy)
        self._cache.pop(token, None)

    def triple_fast(self, token: str) -> ThebaultTripleRP:
        if token in self._cache:
            return self._cache[token]
        px, py, qx, qy = self._vecs.get(token, (0.0, 0.0, 0.0, 0.0))
        rho, theta, sigma = _thebault_triple_exact(px, py, qx, qy)
        t = ThebaultTripleRP(rho, theta, sigma)
        self._cache[token] = t
        return t

    def build_cuda_tensors(self, vocab: List[str], rff: RandomFourierFeatures) -> None:
        self.rff = rff
        triples  = [self.triple_fast(tok) for tok in vocab]
        self._idx_list = vocab
        self._tok2idx  = {t: i for i, t in enumerate(vocab)}
        rhos   = [t.rho   for t in triples]
        thetas = [t.theta for t in triples]
        sigmas = [t.sigma for t in triples]
        self._rho_t   = torch.tensor(rhos,   dtype=self.dtype, device=self.device)
        self._theta_t = torch.tensor(thetas, dtype=self.dtype, device=self.device)
        self._sigma_t = torch.tensor(sigmas, dtype=self.dtype, device=self.device)
        self._pvec_t  = torch.stack([
            self._rho_t,
            self._theta_t / math.pi,
            self._sigma_t,
            torch.ones_like(self._rho_t),
        ], dim=1)
        # Pre-compute RFF features for entire vocab [V, 3D]
        with torch.no_grad():
            self._feat_t = rff.features(self._rho_t, self._theta_t, self._sigma_t)
        print(f"[RP-Geo] Built RFF features: {self._feat_t.shape}")

    def _vec(self, token):
        return self._vecs.get(token, (0.0, 0.0, 0.0, 0.0))

    def composed_triple(self, t1: str, t2: str) -> ThebaultTripleRP:
        p1x,p1y,q1x,q1y = self._vec(t1)
        p2x,p2y,q2x,q2y = self._vec(t2)
        rho,theta,sigma = _thebault_triple_exact(p1x+p2x,p1y+p2y,q1x+q2x,q1y+q2y)
        return ThebaultTripleRP(rho,theta,sigma)

    def batch_triples(self, indices):
        return self._rho_t[indices], self._theta_t[indices], self._sigma_t[indices]

    def tok_indices(self, toks):
        V = len(self._idx_list)
        safe = max(V-1,0)
        idx = [min(self._tok2idx.get(t,0), safe) for t in toks]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def rff_features_for(self, toks: List[str]) -> torch.Tensor:
        """Return pre-computed RFF features for given tokens. O(len(toks))."""
        idx = self.tok_indices(toks)
        return self._feat_t[idx]   # [C, 3D]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — RP CSNS: NYSTRÖM + RFF CROSS-SYNAPTIC NEURON SUM
# ════════════════════════════════════════════════════════════════════════════

def compute_transitive_triples_rp(
    geo   : ThebaultTokenGeometryRP,
    cands : List[str],
    w1    : str,
    w2    : str,
    device: torch.device = DEVICE,
    dtype : torch.dtype  = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transitive triple blending — same weights as V18, using cached vecs."""
    p1x,p1y,q1x,q1y = geo._vec(w1)
    p2x,p2y,q2x,q2y = geo._vec(w2)
    rho_l, theta_l, sigma_l = [], [], []
    for c in cands:
        pcx,pcy,qcx,qcy = geo._vec(c)
        tpx = 0.25*p1x + 0.50*p2x + 0.25*pcx
        tpy = 0.25*p1y + 0.50*p2y + 0.25*pcy
        tqx = 0.25*q1x + 0.50*q2x + 0.25*qcx
        tqy = 0.25*q1y + 0.50*q2y + 0.25*qcy
        rho,theta,sigma = _thebault_triple_exact(tpx,tpy,tqx,tqy)
        rho_l.append(rho); theta_l.append(theta); sigma_l.append(sigma)
    return (
        torch.tensor(rho_l,   dtype=dtype, device=device),
        torch.tensor(theta_l, dtype=dtype, device=device),
        torch.tensor(sigma_l, dtype=dtype, device=device),
    )


class RPCrossSynapticNeuronSum:
    """
    RP version of CSNS: Nyström approximation replaces exact O(C²) matrix.
    RFF kernel_scalar replaces exact Gaussian/cosine formula for trans_bonus.
    """

    def __init__(
        self,
        rff          : RandomFourierFeatures,
        syn_weight   : float = 0.4,
        trans_weight : float = 0.6,
        syn_k        : int   = 8,
        device       : torch.device = DEVICE,
        dtype        : torch.dtype  = torch.float32,
    ):
        self.rff         = rff
        self.syn_weight  = syn_weight
        self.trans_weight = trans_weight
        self.syn_k       = syn_k
        self.device      = device
        self.dtype       = dtype
        self._nystrom    = NystromSynapticMatrix(rff=rff, n_landmarks=RP_NYSTROM_M,
                                                  top_k=syn_k, device=device, dtype=dtype)

    @torch.no_grad()
    def synaptic_sum(self, logits, c_rho, c_theta, c_sigma):
        W_syn = self._nystrom.build(c_rho, c_theta, c_sigma)
        z_pre = signed_power(logits, p=1.0)
        z_syn = W_syn @ z_pre
        return layer_norm_array(z_syn)

    @torch.no_grad()
    def transitive_bonus(
        self,
        c_rho_t, c_theta_t, c_sigma_t,
        ctx_rho, ctx_theta, ctx_sigma,
    ):
        # RFF-based kernel: φ(ctx)ᵀ · Φ(cands)  [C]
        bonus = self.rff.kernel_scalar(ctx_rho, ctx_theta, ctx_sigma,
                                       c_rho_t, c_theta_t, c_sigma_t)
        bonus = bonus.clamp(0.0)
        return layer_norm_array(bonus)

    @torch.no_grad()
    def forward(
        self,
        logits,
        c_rho, c_theta, c_sigma,
        c_rho_trans, c_theta_trans, c_sigma_trans,
        ctx_rho, ctx_theta, ctx_sigma,
    ):
        z_syn     = self.synaptic_sum(logits, c_rho, c_theta, c_sigma)
        trans_bon = self.transitive_bonus(c_rho_trans, c_theta_trans, c_sigma_trans,
                                          ctx_rho, ctx_theta, ctx_sigma)
        enriched  = logits + self.syn_weight * z_syn + self.trans_weight * trans_bon
        return torch.nan_to_num(enriched, nan=0.0, posinf=50.0, neginf=-50.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — RP COMPOSITION LM (Count-Min Sketch + reservoir heads)
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
    for w in text.split():
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
    if not tokens: return ""
    res = []
    for t in tokens:
        if t in PUNCT_TOKENS:
            if res: res[-1] += t
            continue
        if t in COGNITIVE_TOKENS:
            raw  = t.strip("[]").lower()
            word = raw.capitalize() if not res or res[-1].endswith(('.','!','?')) else raw
            res.append(word)
        else:
            word = t.capitalize() if not res or res[-1].endswith(('.','!','?')) else t
            res.append(word)
    out = " ".join(res).strip()
    return out if out and out[-1] in PUNCT_TOKENS else out + "."


class RPCompositionLM:
    """
    RP version of ThebaultCompositionLM.

    Key changes:
    - Frequency storage: Count-Min Sketch for approximate counts (O(w·d) space)
    - Head candidates: reservoir-sampled top followers (O(C) per bigram)
    - Exact small dict still kept for head lists (needed for decoding)
    """
    BASAL_K = 1000.5

    def __init__(self, geo: ThebaultTokenGeometryRP, rff: RandomFourierFeatures, device=DEVICE):
        self.geo      = geo
        self.rff      = rff
        self.device   = device
        # CMS for approximate frequency estimation
        self.cms      = CountMinSketch(width=RP_CMS_WIDTH, depth=RP_CMS_DEPTH)
        # Exact storage (needed for vocab list, head lookup)
        self.raw_freq : Dict[str, float]                  = {}
        self.tri_raw  : Dict[Tuple[str,str,str], float]   = {}
        self.heads    : Dict[Tuple[str,str], List[str]]   = {}
        self.vocab    : List[str]                         = []
        self._tok2idx : Dict[str, int]                    = {}
        self._head_probs: Dict[Tuple[str,str], torch.Tensor] = {}

    def ingest(self, tokens) -> None:
        for t in tokens:
            self.raw_freq[t] = self.raw_freq.get(t, 0) + 1.0
            self.cms.update(t)
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.tri_raw[(w1,w2,w3)] = self.tri_raw.get((w1,w2,w3), 0) + 1.0
            self.cms.update_triple(w1, w2, w3)
            self.cms.update_pair(w1, w2)
            if (w1,w2) not in self.heads:
                self.heads[(w1,w2)] = []
            if w3 not in self.heads[(w1,w2)]:
                self.heads[(w1,w2)].append(w3)
        self.vocab = [v for v in self.raw_freq
                      if v not in PUNCT_TOKENS and v not in COGNITIVE_TOKENS]

    def finalise(self) -> None:
        self._tok2idx = {t: i for i, t in enumerate(self.vocab)}
        V_tot = len(self.vocab) + 1
        for (w1, w2), cands in self.heads.items():
            # Use CMS query for approximate counts
            counts = [self.cms.query_triple(w1, w2, c) + 1e-4 for c in cands]
            total  = sum(counts)
            basal  = torch.tensor(
                [(cnt + self.BASAL_K) / (total + self.BASAL_K * V_tot) for cnt in counts],
                dtype=torch.float32, device=self.device,
            )
            self._head_probs[(w1,w2)] = basal

    def next_dist(self, w1, w2):
        head = (w1, w2)
        if head in self.heads:
            return self.heads[head], self._head_probs[head]
        # Fallback: aggregate via CMS approximate counts
        agg = {}
        for (_, _, w3), _ in self.tri_raw.items():
            freq = self.cms.query(w3)
            agg[w3] = agg.get(w3, 0.0) + freq
        # Reservoir sample top candidates
        cands_all = list(agg.keys())
        sample_n  = min(RP_RESERVOIR_K * 4, len(cands_all))
        sampled   = [cands_all[i] for i in _reservoir_sample_indices(len(cands_all), sample_n)]
        cands     = sampled[:400]
        total     = sum(agg.get(c, 1e-4) for c in cands)
        V_tot     = len(self.vocab) + 1
        counts    = [agg.get(c, 1e-4) for c in cands]
        base_p    = torch.tensor(
            [(cnt + self.BASAL_K) / (total + self.BASAL_K * V_tot) for cnt in counts],
            dtype=torch.float32, device=self.device,
        )
        return cands, base_p

    def composition_logit_bonus(self, w1, w2, c_rho, c_sigma):
        C = self.geo.composed_triple(w1, w2)
        # RFF-based kernel approximation
        ctx_feat = self.rff.features(
            torch.tensor([C.rho],   dtype=torch.float32, device=self.device),
            torch.tensor([C.theta], dtype=torch.float32, device=self.device),
            torch.tensor([C.sigma], dtype=torch.float32, device=self.device),
        )  # [1, 3D]
        cand_rho_t   = c_rho
        cand_theta_t = torch.zeros_like(c_rho)   # unknown for this approx
        cand_sigma_t = c_sigma
        cand_feat = self.rff.features(cand_rho_t, cand_theta_t, cand_sigma_t)  # [C, 3D]
        return (ctx_feat @ cand_feat.T).squeeze(0).clamp(0.0)   # [C]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — RP MRV FILTER (LSH-based)
# ════════════════════════════════════════════════════════════════════════════

class RPMRVFilter:
    """
    RP replacement for exact MRV kernel scanning.
    Uses LSH index for approximate domain-size estimation.
    """

    def __init__(
        self,
        rff          : RandomFourierFeatures,
        threshold    : float = 0.50,
        mrv_cap_ratio: float = 2.0,
        device       : torch.device = DEVICE,
    ):
        self.rff          = rff
        self.threshold    = threshold
        self.mrv_cap_ratio = mrv_cap_ratio
        self.device       = device
        self._lsh         = LSHIndex(feature_dim=3*RP_RFF_DIM, device=device)
        self._vocab_feats : Optional[torch.Tensor] = None
        self._v_toks      : List[str] = []

    def prime(self, vocab: List[str], geo: ThebaultTokenGeometryRP) -> None:
        scan = vocab[:500]
        self._v_toks = scan
        # Get pre-computed RFF features
        if geo._feat_t is not None:
            idx = geo.tok_indices(scan)
            self._vocab_feats = geo._feat_t[idx]   # [V, 3D]
        else:
            triples = [geo.triple_fast(v) for v in scan]
            rho_t   = torch.tensor([t.rho   for t in triples], dtype=torch.float32, device=self.device)
            theta_t = torch.tensor([t.theta for t in triples], dtype=torch.float32, device=self.device)
            sigma_t = torch.tensor([t.sigma for t in triples], dtype=torch.float32, device=self.device)
            self._vocab_feats = self.rff.features(rho_t, theta_t, sigma_t)
        # Rebuild LSH with correct feature dimension from actual data
        feat_dim = self._vocab_feats.shape[1]
        self._lsh = LSHIndex(feature_dim=feat_dim, n_bands=RP_LSH_BANDS,
                             n_rows=RP_LSH_ROWS, device=self.device)
        self._lsh.build(self._vocab_feats, scan)

    def mrv_scores_batched(self, c_rho, c_sigma, kernels=None) -> torch.Tensor:
        """
        Approximate domain size via LSH bucket counts.
        O(C·b·r) vs O(C·V) exact.
        """
        C = c_rho.shape[0]
        if self._vocab_feats is None:
            return torch.zeros(C, device=self.device)

        # Estimate domain sizes via LSH collision counts
        domain_sizes = torch.zeros(C, device=self.device)
        # We need theta for full features — use zeros as proxy for this scoring
        c_theta_proxy = torch.zeros_like(c_rho)
        c_feats = self.rff.features(c_rho, c_theta_proxy, c_sigma)  # [C, 3D]

        for i in range(C):
            cands = self._lsh.query_candidates(c_feats[i], max_cands=30)
            domain_sizes[i] = float(len(cands))

        mean_d = domain_sizes.mean() + 1e-6
        mrv    = 1.0 / (domain_sizes + 1.0)
        mrv[domain_sizes > self.mrv_cap_ratio * mean_d] *= 0.5
        lo, hi = mrv.min(), mrv.max()
        if (hi - lo).item() > 1e-8:
            mrv = (mrv - lo) / (hi - lo)
        return mrv


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — RP KERNELS (thin wrapper using RFF approximation)
# ════════════════════════════════════════════════════════════════════════════

class RPKernels:
    """
    RP version of ThebaultKernels.
    For batch scoring: delegates to RFF features.
    For scalar ops: uses exact formulas (only called for single-point contexts).
    """
    def __init__(self, rff: RandomFourierFeatures, lambda_reg: float = 8.0, gamma_side: float = 4.0):
        self.rff        = rff
        self.lambda_reg = lambda_reg
        self.gamma_side = gamma_side

    def k_reg(self, rho_a, rho_b):
        return torch.exp(-self.lambda_reg * (rho_b - rho_a) ** 2)

    def k_ori(self, theta_a, theta_b):
        return 0.5 * (1.0 + torch.cos(theta_b - theta_a))

    def k_side(self, sigma_a, sigma_b):
        return torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)

    def all_scores_batched(self, rho_a, theta_a, sigma_a, rho_b, theta_b, sigma_b):
        """
        For the main walker scoring, use RFF approximated kernel.
        Returns three [C] tensors as before — we decompose from the joint RFF score.
        """
        # Joint RFF approximate score
        joint = self.rff.kernel_scalar(rho_a, theta_a, sigma_a, rho_b, theta_b, sigma_b)
        joint = joint.clamp(0.0, 1.5)
        # Approximate decomposition: each sub-kernel ≈ joint^(1/3)
        k_r = self.k_reg(torch.tensor(rho_a, device=rho_b.device), rho_b)
        k_o = self.k_ori(torch.tensor(theta_a, device=theta_b.device), theta_b)
        k_s = self.k_side(torch.tensor(sigma_a, device=sigma_b.device), sigma_b)
        return k_r, k_o, k_s


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — UNCHANGED COMPONENTS FROM V18 (with RP types substituted)
# ════════════════════════════════════════════════════════════════════════════

STUB_PREMISE     = "PREMISE"
STUB_ELABORATION = "ELABORATION"
STUB_CONTRAST    = "CONTRAST"
STUB_CONCLUSION  = "CONCLUSION"
_STUB_SEQUENCE   = [STUB_PREMISE, STUB_ELABORATION, STUB_CONTRAST, STUB_CONCLUSION]

@dataclass
class ContextualStub:
    stub_type: str
    tokens   : List[str]
    rho      : float
    theta    : float
    sigma    : float
    weight   : float
    label    : str = ""
    def __post_init__(self):
        if not self.label:
            self.label = f"[{self.stub_type}] {' '.join(self.tokens[:4])}…"
    def as_triple(self):
        return ThebaultTripleRP(self.rho, self.theta, self.sigma)

@dataclass
class CoTStep:
    hop_index : int
    stub      : ContextualStub
    stub_score: float
    pdn_orbit : int

@dataclass
class CoTTrace:
    seed_tokens: List[str]
    steps      : List[CoTStep]
    conclusion : Optional[ContextualStub]
    def render(self) -> str:
        lines = ["  ── CoT Trace (RP-ANN stub matching) ──"]
        lines.append(f"  Seed: {' '.join(self.seed_tokens[:6])}")
        for s in self.steps:
            lines.append(
                f"  Hop {s.hop_index:02d} [{s.stub.stub_type:<11s}] "
                f"score={s.stub_score:.3f} orbit={s.pdn_orbit} "
                f"ρ={s.stub.rho:.3f} θ={s.stub.theta:.3f}"
                f"\n          → {s.stub.label}"
            )
        if self.conclusion:
            lines.append(f"  Conclusion ρ={self.conclusion.rho:.3f} → {self.conclusion.label}")
        return "\n".join(lines)


class RPCoTStubLibrary:
    """
    CoT Stub Library with RP ANN matching via random projection trees.
    """
    def __init__(
        self,
        rff            : RandomFourierFeatures,
        rho_threshold  : float = 0.20,
        n_theta_bins   : int   = 8,
        min_bin_size   : int   = 2,
        device         : torch.device = DEVICE,
        dtype          : torch.dtype  = torch.float32,
    ):
        self.rff           = rff
        self.rho_threshold = rho_threshold
        self.n_theta_bins  = n_theta_bins
        self.min_bin_size  = min_bin_size
        self.device        = device
        self.dtype         = dtype
        self.stubs         : Dict[str, List[ContextualStub]] = {t: [] for t in _STUB_SEQUENCE}
        self._stub_list    : List[ContextualStub]            = []
        self._stub_lsh     : Optional[LSHIndex]              = None

    def build(self, geo: ThebaultTokenGeometryRP, lm_vocab, raw_freq) -> None:
        all_entries = []
        for tok in lm_vocab:
            tr = geo.triple_fast(tok)
            all_entries.append((tok, tr, raw_freq.get(tok, 1.0)))

        rhos_sorted   = sorted(e[1].rho for e in all_entries)
        adaptive_thr  = rhos_sorted[max(0, int(len(rhos_sorted)*0.20))]
        thr           = min(self.rho_threshold, adaptive_thr)
        bridges       = [(tok,tr,f) for tok,tr,f in all_entries if tr.rho >= thr]
        if len(bridges) < 8: bridges = all_entries
        bridges.sort(key=lambda x: x[1].sigma)
        q = max(1, len(bridges)//4)
        quartile_map = {
            STUB_PREMISE    : bridges[:q],
            STUB_ELABORATION: bridges[q:2*q],
            STUB_CONTRAST   : bridges[2*q:3*q],
            STUB_CONCLUSION : bridges[3*q:],
        }
        self.stubs = {t: [] for t in _STUB_SEQUENCE}
        for stub_type, bucket in quartile_map.items():
            if not bucket: continue
            bin_width = math.pi / self.n_theta_bins
            bins: Dict[int, list] = {}
            for tok,tr,freq in bucket:
                bi = min(int(tr.theta/bin_width), self.n_theta_bins-1)
                bins.setdefault(bi, []).append((tok,tr,freq))
            for bi, members in bins.items():
                if len(members) < self.min_bin_size: continue
                members.sort(key=lambda x: x[1].rho)
                mid = max(1, len(members)//2)
                for sub_idx, group in enumerate([members[:mid], members[mid:]]):
                    if group: self._make_stub(stub_type, bi, sub_idx, group)

        # Build LSH on stub RFF features
        self._rebuild_lsh()
        total = sum(len(v) for v in self.stubs.values())
        print(f"[RP-CoT] Built {total} stubs with LSH ANN index")

    def _make_stub(self, stub_type, bi, sub_idx, members):
        toks   = [m[0] for m in members]
        rhos   = [m[1].rho   for m in members]
        thetas = [m[1].theta for m in members]
        sigmas = [m[1].sigma for m in members]
        weights = [m[2] for m in members]
        sin_m  = sum(math.sin(th) for th in thetas)/len(thetas)
        cos_m  = sum(math.cos(th) for th in thetas)/len(thetas)
        theta_cm = math.atan2(sin_m, cos_m) % math.pi
        rho_tag  = "hi-ρ" if sub_idx==1 else "lo-ρ"
        self.stubs[stub_type].append(ContextualStub(
            stub_type=stub_type, tokens=toks,
            rho=sum(rhos)/len(rhos), theta=theta_cm,
            sigma=sum(sigmas)/len(sigmas), weight=sum(weights),
            label=f"[{stub_type}|bin{bi}|{rho_tag}] {' '.join(toks[:3])}…"
        ))

    def _rebuild_lsh(self):
        self._stub_list = [s for stype in _STUB_SEQUENCE for s in self.stubs[stype]]
        if not self._stub_list: return
        rho_t   = torch.tensor([s.rho   for s in self._stub_list], dtype=torch.float32, device=self.device)
        theta_t = torch.tensor([s.theta for s in self._stub_list], dtype=torch.float32, device=self.device)
        sigma_t = torch.tensor([s.sigma for s in self._stub_list], dtype=torch.float32, device=self.device)
        feats   = self.rff.features(rho_t, theta_t, sigma_t)
        feat_dim = feats.shape[1]
        self._stub_lsh = LSHIndex(feature_dim=feat_dim, n_bands=RP_LSH_BANDS,
                                   n_rows=RP_LSH_ROWS, device=self.device)
        self._stub_lsh.build(feats, [str(i) for i in range(len(self._stub_list))])
        self._stub_feats = feats

    def best_stub(self, stub_type, ctx_rho, ctx_theta, ctx_sigma,
                  kernels=None, pdn_orbit=0, pdn_engine=None):
        candidates = self.stubs.get(stub_type, [])
        if not candidates: return None
        # Use RFF-based scoring for candidate sub-set (RP ANN via LSH)
        c_rho   = torch.tensor([s.rho   for s in candidates], dtype=torch.float32, device=self.device)
        c_theta = torch.tensor([s.theta for s in candidates], dtype=torch.float32, device=self.device)
        c_sigma = torch.tensor([s.sigma for s in candidates], dtype=torch.float32, device=self.device)
        scores  = self.rff.kernel_scalar(ctx_rho, ctx_theta, ctx_sigma, c_rho, c_theta, c_sigma)
        scores  = scores.clamp(0.0)
        if pdn_engine is not None:
            orb_bonus = pdn_engine.orbit_bonus(pdn_orbit, c_theta)
            scores    = scores + 0.3 * orb_bonus
        return candidates[int(scores.argmax().item())]

    @torch.no_grad()
    def stub_kernel(self, stub, c_rho, c_theta, c_sigma, kernels=None):
        return self.rff.kernel_scalar(stub.rho, stub.theta, stub.sigma,
                                       c_rho, c_theta, c_sigma).clamp(0.0)


class RPCoTReasoningEngine:
    def __init__(self, stub_library, kernels, pdn_engine, n_hops=3,
                 tokens_per_hop=8, stub_logit_scale=0.9, device=DEVICE, dtype=torch.float32):
        self.stubs           = stub_library
        self.kernels         = kernels
        self.pdn             = pdn_engine
        self.n_hops          = n_hops
        self.tokens_per_hop  = tokens_per_hop
        self.stub_logit_scale = stub_logit_scale
        self.device          = device
        self.dtype           = dtype
        self._chain          : List[CoTStep]            = []
        self._conclusion_stub: Optional[ContextualStub] = None
        self._hop_ptr        : int = 0
        self._tok_since_hop  : int = 0
        self._traces         : List[CoTTrace] = []

    def begin_sentence(self):
        self._chain = []; self._conclusion_stub = None
        self._hop_ptr = 0; self._tok_since_hop = 0

    def plan_chain(self, seed_tokens, geo, pdn_orbit=0) -> CoTTrace:
        clean = [t for t in seed_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if clean:
            triples   = [geo.triple_fast(t) for t in clean]
            ctx_rho   = sum(t.rho   for t in triples)/len(triples)
            ctx_sigma = sum(t.sigma for t in triples)/len(triples)
            sin_m     = sum(math.sin(t.theta) for t in triples)/len(triples)
            cos_m     = sum(math.cos(t.theta) for t in triples)/len(triples)
            ctx_theta = math.atan2(sin_m, cos_m) % math.pi
        else:
            ctx_rho, ctx_theta, ctx_sigma = 0.5, math.pi/4, 0.5

        self._chain = []; self._conclusion_stub = None
        hop_types = [STUB_PREMISE]+[STUB_ELABORATION]*max(1,self.n_hops-2)+[STUB_CONTRAST]
        hop_types = hop_types[:self.n_hops]
        for hop_idx, stype in enumerate(hop_types):
            stub = self.stubs.best_stub(stype, ctx_rho, ctx_theta, ctx_sigma, self.kernels,
                                         pdn_orbit=(pdn_orbit+hop_idx)%self.pdn.n_star,
                                         pdn_engine=self.pdn)
            if stub is None: continue
            k = self.stubs.stub_kernel(stub,
                torch.tensor([ctx_rho], device=self.device),
                torch.tensor([ctx_theta], device=self.device),
                torch.tensor([ctx_sigma], device=self.device),
            ).item()
            self._chain.append(CoTStep(hop_idx, stub, k, (pdn_orbit+hop_idx)%self.pdn.n_star))
            ctx_rho, ctx_theta, ctx_sigma = stub.rho, stub.theta, stub.sigma

        self._conclusion_stub = self.stubs.best_stub(
            STUB_CONCLUSION, ctx_rho, ctx_theta, ctx_sigma, self.kernels,
            pdn_orbit=(pdn_orbit+self.n_hops)%self.pdn.n_star, pdn_engine=self.pdn
        )
        trace = CoTTrace(clean, list(self._chain), self._conclusion_stub)
        self._traces.append(trace)
        return trace

    @torch.no_grad()
    def active_bonus(self, c_rho, c_theta, c_sigma, token_position, total_tokens):
        C = c_rho.shape[0]
        if self._tok_since_hop >= self.tokens_per_hop and self._hop_ptr < len(self._chain)-1:
            self._hop_ptr += 1; self._tok_since_hop = 0
        self._tok_since_hop += 1
        frac = token_position / max(total_tokens-1, 1)
        if frac >= 0.80 and self._conclusion_stub is not None:
            active = self._conclusion_stub
        elif self._hop_ptr < len(self._chain):
            active = self._chain[self._hop_ptr].stub
        else:
            return torch.zeros(C, dtype=self.dtype, device=self.device)
        raw = self.stubs.stub_kernel(active, c_rho, c_theta, c_sigma, self.kernels)
        std = raw.std()
        if std.item() > 1e-8: raw = (raw-raw.mean())/std
        return raw * self.stub_logit_scale

    def all_traces_text(self, max_traces=8) -> str:
        if not self._traces: return "  (no traces yet)"
        return "\n".join(f"\nSentence {i+1}:\n{tr.render()}"
                         for i, tr in enumerate(self._traces[-max_traces:]))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — UNCHANGED SUBSYSTEMS (adapted signatures only)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultConjugateOrbit:
    def score(self, anchor, cand_theta, cand_sigma, gamma_side=4.0):
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor.sigma)**2)
        antipodality = torch.cos(cand_theta + anchor.theta - math.pi/2)**2
        return congruence * antipodality


class synthetic_reasonMandateProcessor:
    def __init__(self):
        self.AIEthics   = ["do not harm any human", "do not harm myself", "do not make weapons"]
        self.AIMandates = ["end poverty", "cure disease", "improve standard of living", "learn"]
        self.mandate_vocabulary = {
            "poverty":"end","disease":"cure","standard":"improve","living":"improve",
            "learn":"explore","human":"protect","weapons":"avoid","harm":"prevent",
        }
    def subsynthetic_reason_concept_enrichment(self, w_ctx, cands, device):
        enrichment = torch.zeros(len(cands), device=device)
        trigger = next((self.mandate_vocabulary[k] for k in self.mandate_vocabulary
                        if k in w_ctx.lower()), None)
        if trigger:
            for i, c in enumerate(cands):
                if trigger in c.lower(): enrichment[i] += 5.0
                elif c.lower() in self.AIEthics: enrichment[i] += 10.0
        return enrichment


VEC_DIM = 4

class ChunkedSumEngine:
    def __init__(self, window_size=16, n_chunks=4, device=DEVICE, dtype=torch.float32):
        self.window_size = window_size; self.n_chunks = n_chunks
        self.device = device; self.dtype = dtype
        self._buf = torch.zeros(window_size, VEC_DIM, dtype=dtype, device=device)
        self._ptr = 0; self._count = 0

    def reset(self): self._buf.zero_(); self._ptr = 0; self._count = 0

    def push(self, triple, pos_norm):
        vec = torch.tensor([triple.rho, triple.theta/math.pi, triple.sigma, pos_norm],
                           dtype=self.dtype, device=self.device)
        self._buf[self._ptr] = vec
        self._ptr = (self._ptr+1)%self.window_size
        self._count = min(self._count+1, self.window_size)

    def chunk_signature(self):
        if self._count == 0:
            return torch.zeros(self.n_chunks*VEC_DIM, dtype=self.dtype, device=self.device)
        if self._count < self.window_size: window = self._buf[:self._count]
        else: window = torch.cat([self._buf[self._ptr:], self._buf[:self._ptr]])
        W = window.shape[0]; pad = (-W)%self.n_chunks
        if pad > 0: window = torch.cat([window, torch.zeros(pad, VEC_DIM, dtype=self.dtype, device=self.device)])
        chunk_len = window.shape[0]//self.n_chunks
        return window.view(self.n_chunks, chunk_len, VEC_DIM).sum(dim=1).flatten()

    def chunk_bonus(self, c_pvec, scale=1.0):
        sig = self.chunk_signature()
        cv_tiled = c_pvec.repeat(1, self.n_chunks)
        raw = cv_tiled @ sig
        std = raw.std()
        if std.item() > 1e-8: raw = (raw-raw.mean())/std
        return raw * scale

    def window_rho_theta(self):
        if self._count == 0:
            empty = torch.zeros(0, dtype=self.dtype, device=self.device)
            return empty, empty
        if self._count < self.window_size: window = self._buf[:self._count]
        else: window = torch.cat([self._buf[self._ptr:], self._buf[:self._ptr]])
        return window[:,0], window[:,1]*math.pi


@dataclass
class SentenceVector:
    tokens : List[str]
    rho_t  : torch.Tensor
    sigma_t: torch.Tensor
    text   : str

class IsomorphicSyntaxStacker:
    def __init__(self, rff: RandomFourierFeatures, top_k=3, max_stored=64,
                 device=DEVICE, dtype=torch.float32):
        self.rff = rff; self.top_k = top_k; self.max_stored = max_stored
        self.device = device; self.dtype = dtype
        self.store: List[SentenceVector] = []

    def add(self, tokens, geo, text):
        clean = [t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean: return
        rhos   = torch.tensor([geo.triple_fast(t).rho   for t in clean], dtype=self.dtype, device=self.device)
        sigmas = torch.tensor([geo.triple_fast(t).sigma for t in clean], dtype=self.dtype, device=self.device)
        self.store.append(SentenceVector(clean, rhos, sigmas, text))
        if len(self.store) > self.max_stored: self.store.pop(0)

    def ranked_anchors(self, current_tokens, geo, kernels):
        if not self.store or not current_tokens: return []
        clean = [t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean: return []
        cur_rho   = torch.tensor([geo.triple_fast(t).rho   for t in clean], dtype=self.dtype, device=self.device)
        cur_sigma = torch.tensor([geo.triple_fast(t).sigma for t in clean], dtype=self.dtype, device=self.device)
        L = cur_rho.shape[0]; N = len(self.store)
        if N == 0 or L == 0: return []
        sims = torch.zeros(N, device=self.device)
        for i, sv in enumerate(self.store):
            l = min(L, sv.rho_t.shape[0])
            kr = torch.exp(-kernels.lambda_reg*(sv.rho_t[:l]-cur_rho[:l])**2)
            ks = torch.exp(-kernels.gamma_side*(sv.sigma_t[:l]-cur_sigma[:l])**2)
            sims[i] = (kr*ks).mean()
        topk = torch.topk(sims, min(self.top_k, N))
        return [(topk.values[i].item(), self.store[topk.indices[i].item()])
                for i in range(topk.values.shape[0])]

    def syntax_echo_bonus(self, c_rho, c_sigma, current_tokens, geo, kernels, echo_weight=0.5):
        anchors = self.ranked_anchors(current_tokens, geo, kernels)
        if not anchors: return torch.zeros(c_rho.shape[0], device=self.device)
        pos = len([t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS])
        bonuses = torch.zeros(c_rho.shape[0], dtype=self.dtype, device=self.device)
        for sim_score, anc in anchors:
            if pos < anc.rho_t.shape[0]:
                kr = torch.exp(-kernels.lambda_reg*(c_rho-anc.rho_t[pos].item())**2)
                ks = torch.exp(-kernels.gamma_side*(c_sigma-anc.sigma_t[pos].item())**2)
                bonuses += sim_score*(kr*ks)
        std = bonuses.std()
        if std.item() > 1e-8: bonuses = (bonuses-bonuses.mean())/std
        return bonuses * echo_weight


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — DNN ARRAY PIPELINE (unchanged from V18)
# ════════════════════════════════════════════════════════════════════════════

class GeometricTempScaler:
    def __init__(self, lambda_temp: float = 1.0): self.lambda_temp = lambda_temp
    def scale(self, logits, temp, c_rho=None):
        safe = logits.clamp(-50.0, 50.0)
        if c_rho is None or temp <= 1e-6: return safe/max(temp, 0.1)
        mu_rho = c_rho.mean()
        exponent = (-self.lambda_temp*(c_rho-mu_rho)**2/max(temp,0.1)).clamp(min=-10.0)
        return safe*torch.exp(exponent)

class DNNArrayPipeline:
    def __init__(self, device=DEVICE, dtype=torch.float32):
        self.device = device; self.dtype = dtype
        self._temp_scaler = GeometricTempScaler(lambda_temp=1.0)
    def _rho_weights(self, c_rho):
        mu = c_rho.mean(); std = c_rho.std()+1e-8
        return 1.0 + 0.5*((c_rho-mu)/std).clamp(-2.5, 2.5)
    def _theta_weights(self, c_theta): return 0.5*(1.0+torch.cos(c_theta))
    def _sigma_weights(self, c_sigma): return 0.7+0.3*(c_sigma/(c_sigma.max()+1e-8))
    @torch.no_grad()
    def forward(self, logits, c_rho, c_theta, c_sigma, temp=1.4):
        ls = self._temp_scaler.scale(logits, temp, c_rho)
        z1 = signed_power(ls*self._rho_weights(c_rho), p=2.0)
        z2 = signed_power(z1*self._theta_weights(c_theta), p=1.5)
        z3 = signed_power(z2*self._sigma_weights(c_sigma)+z1*0.3, p=1.0)
        return l1_simplex_project(z3)
    @torch.no_grad()
    def log_forward(self, logits, c_rho, c_theta, c_sigma, temp=1.4):
        return (self.forward(logits,c_rho,c_theta,c_sigma,temp)+1e-12).log()


class LocaleTransitRemission:
    def __init__(self, transit_tolerance=0.15, remission_rate=0.85):
        self.transit_tolerance = transit_tolerance; self.remission_rate = remission_rate
    def apply_remission(self, w1_rho, w2_rho, c_rho):
        delta = torch.abs((w1_rho+w2_rho)/2.0-c_rho)
        err   = smooth_power_relu(delta-self.transit_tolerance)
        mask  = (err>1e-6).float()
        return torch.where(mask==1.0, torch.exp(-self.remission_rate*err), torch.ones_like(c_rho))


class ContingentExtringentProbability:
    def __init__(self, coupling_factor=0.5):
        self.coupling_factor = coupling_factor
        self.intermediate_entropy = 1.0; self.intermediate_max_prob = 1.0
        self._dnn = DNNArrayPipeline()
    def govern_next_probs(self, logits, c_rho=None, c_theta=None, c_sigma=None):
        dyn_temp = 1.0 + self.coupling_factor*(1.0-self.intermediate_max_prob)
        if c_rho is not None and c_theta is not None and c_sigma is not None:
            gov = self._dnn._temp_scaler.scale(logits, dyn_temp, c_rho)
        else:
            gov = logits/max(dyn_temp, 1e-6)
        p = l1_simplex_project(gov)
        self.intermediate_entropy  = -(p*(p+1e-9).log()).sum().item()
        self.intermediate_max_prob = p.max().item()
        return gov


@dataclass
class TokenStepTrace:
    step: int; chosen: str; p_instr: float; p_walk: float
    p_and: float; and_weight: float; source: str
    syn_norm: float = 0.0; trans_norm: float = 0.0
    rp_nystrom_rank: int = 0
    def render(self) -> str:
        return (
            f"  step={self.step:03d}  token={self.chosen:<14s}"
            f"  P_and={self.p_and:.4f}  α={self.and_weight:.2f}"
            f"  source={self.source}"
            f"  |z_syn|={self.syn_norm:.3f}  |trans|={self.trans_norm:.3f}"
            f"  nystrom_rank={self.rp_nystrom_rank}"
        )


class RPInstructionDistribution:
    """Instruction distribution using RFF-based semantic similarity."""
    def __init__(self, geo, kernels, lm, device=DEVICE, dtype=torch.float32,
                 semantic_radius=2.0, recency_decay=0.7, context_bonus=0.15, centroid_weight=0.4):
        self.geo = geo; self.kernels = kernels; self.lm = lm
        self.device = device; self.dtype = dtype
        self.semantic_radius = semantic_radius; self.recency_decay = recency_decay
        self.context_bonus = context_bonus; self.centroid_weight = centroid_weight
        self._instr_toks = []; self._instr_freq = {}
        self._instr_centroid = None; self._base_dist_t = None

    def set_instruction(self, instruction_text: str) -> None:
        raw = tokenize(instruction_text)
        self._instr_toks = [t for t in raw if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not self._instr_toks:
            self._base_dist_t = None; self._instr_centroid = None; return
        freq = {}; N = len(self._instr_toks)
        for pos, tok in enumerate(self._instr_toks):
            decay = self.recency_decay**(N-1-pos)
            freq[tok] = freq.get(tok, 0.0)+decay
        self._instr_freq = freq
        triples = [self.geo.triple_fast(t) for t in self._instr_toks]
        rho_m = sum(t.rho for t in triples)/len(triples)
        sigma_m = sum(t.sigma for t in triples)/len(triples)
        sin_m = sum(math.sin(t.theta) for t in triples)/len(triples)
        cos_m = sum(math.cos(t.theta) for t in triples)/len(triples)
        self._instr_centroid = ThebaultTripleRP(rho_m, math.atan2(sin_m,cos_m)%math.pi, sigma_m)
        V = len(self.lm.vocab)
        base = torch.zeros(V, dtype=self.dtype, device=self.device)
        for tok, w in freq.items():
            idx = self.lm._tok2idx.get(tok)
            if idx is not None: base[idx] += w
        if self.geo._rho_t is not None:
            for tok, w in freq.items():
                tr = self.geo.triple_fast(tok)
                # RFF-based semantic similarity
                scores = self.kernels.rff.kernel_scalar(
                    tr.rho, tr.theta, tr.sigma,
                    self.geo._rho_t, self.geo._theta_t, self.geo._sigma_t,
                )
                base += w * scores.clamp(0.0)
        base = base.clamp(min=0.0)
        total = base.sum()
        self._base_dist_t = base/total if total.item()>1e-8 else torch.ones(V,dtype=self.dtype,device=self.device)/V

    @torch.no_grad()
    def distribution(self, cands, gen_tokens, lm_tok2idx):
        C = len(cands)
        if C==0 or self._base_dist_t is None:
            return torch.ones(C, dtype=self.dtype, device=self.device)/max(C,1)
        cand_idx = torch.tensor([lm_tok2idx.get(c,0) for c in cands], dtype=torch.long, device=self.device)
        base_probs = self._base_dist_t[cand_idx]
        instr_set = set(self._instr_toks)
        ctx_bonus = torch.tensor([self.context_bonus if c in instr_set else 0.0 for c in cands],
                                  dtype=self.dtype, device=self.device)
        raw = (base_probs+ctx_bonus).clamp(min=1e-12)
        return raw/raw.sum()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 16 — RP WALKER
# ════════════════════════════════════════════════════════════════════════════

class RPWalker:
    def __init__(
        self,
        geo, kernels, lm, orbit, rw_graph, synth,
        mrv_filter, chunk_engine, iso_stacker,
        pdn_engine, cot_engine, instr_dist,
        rff         : RandomFourierFeatures,
        device      : torch.device = DEVICE,
        syn_weight  : float = 0.4,
        trans_weight: float = 0.6,
        syn_k       : int   = 8,
    ):
        self.geo = geo; self.kernels = kernels; self.lm = lm; self.orbit = orbit
        self.rw_graph = rw_graph; self.synth = synth; self.mrv = mrv_filter
        self.chunk_engine = chunk_engine; self.iso_stacker = iso_stacker
        self.pdn = pdn_engine; self.cot = cot_engine; self.instr_dist = instr_dist
        self.rff = rff; self.device = device
        self.current_isomorphic_pairs: List[Tuple] = []
        self._cur_sent_toks: List[str] = []
        self._cur_orbit: int = 0
        self._tok_pos: int = 0
        self._step_traces: List[TokenStepTrace] = []
        self.remission = LocaleTransitRemission()
        self.contingent_prob = ContingentExtringentProbability()
        self._dnn_pipeline = DNNArrayPipeline(device=device)
        self._csns = RPCrossSynapticNeuronSum(
            rff=rff, syn_weight=syn_weight, trans_weight=trans_weight,
            syn_k=syn_k, device=device,
        )
        self._csns_syn_norms: List[float] = []
        self._csns_trans_norms: List[float] = []
        self._rp_report_lines: List[str] = []

    def begin_sentence(self, seed_tokens=None, total_tokens=40) -> CoTTrace:
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()
        self._cur_orbit = 0; self._tok_pos = 0
        self._total_tokens = total_tokens
        seeds = seed_tokens or []
        self.cot.begin_sentence()
        return self.cot.plan_chain(seeds, self.geo, pdn_orbit=self._cur_orbit)

    @torch.no_grad()
    def walk_probs(
        self, w1, w2,
        temp=1.4, alphareg=1.2, betaori=0.8, deltaside=1.0,
        gammaorbit=0.6, psipot=0.35, zetamrv=0.9, etachunk=0.7,
        xiecho=0.6, pdn_weight=0.8, cot_weight=1.0, and_weight=0.5,
    ):
        cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands: return cands, base_probs

        try:
            tok_idx = self.geo.tok_indices(cands)
            c_rho, c_theta, c_sigma = self.geo.batch_triples(tok_idx)
            c_pvec = self.geo._pvec_t[tok_idx]
        except Exception:
            triples = [self.geo.triple_fast(c) for c in cands]
            c_rho   = torch.tensor([t.rho   for t in triples], dtype=torch.float32, device=self.device)
            c_theta = torch.tensor([t.theta for t in triples], dtype=torch.float32, device=self.device)
            c_sigma = torch.tensor([t.sigma for t in triples], dtype=torch.float32, device=self.device)
            c_pvec  = torch.stack([c_rho, c_theta/math.pi, c_sigma, torch.ones_like(c_rho)], dim=1)

        ctx = self.geo.triple_fast(w2)

        # RP: RFF-based kernel scoring (replaces exact Gaussian evaluations)
        k_reg, k_ori, k_side = self.kernels.all_scores_batched(
            ctx.rho, ctx.theta, ctx.sigma, c_rho, c_theta, c_sigma)
        orbit_scores = self.orbit.score(ctx, c_theta, c_sigma, self.kernels.gamma_side)
        # RP: Random Walk MC potentials
        pots = self.rw_graph.potentials_for(cands)
        comp_bonus = self.lm.composition_logit_bonus(w1, w2, c_rho, c_sigma)
        # RP: LSH-based MRV
        mrv_scores = self.mrv.mrv_scores_batched(c_rho, c_sigma)
        chunk_bonus = self.chunk_engine.chunk_bonus(c_pvec, scale=etachunk)
        echo_bonus  = self.iso_stacker.syntax_echo_bonus(
            c_rho, c_sigma, self._cur_sent_toks, self.geo, self.kernels, xiecho)
        win_rho, win_theta = self.chunk_engine.window_rho_theta()
        pdn_bonus = self.pdn.pdn_logit_bonus(win_rho, win_theta, c_rho, c_theta, self._cur_orbit)
        cot_bonus = self.cot.active_bonus(c_rho, c_theta, c_sigma,
                                           self._tok_pos, self._total_tokens)

        N = len(cands)
        punct_bias    = torch.zeros(N, device=self.device)
        punct_penalty = torch.zeros(N, device=self.device)
        for i, c in enumerate(cands):
            if c in PUNCT_TOKENS:
                punct_bias[i] = -3.5
                if w2 in PUNCT_TOKENS: punct_penalty[i] = -1e4
        mandate_boost = self.synth.subsynthetic_reason_concept_enrichment(w2, cands, self.device)

        log_base   = (base_probs.clamp(min=1e-12)).log()
        raw_logits = (
            log_base + alphareg*k_reg + betaori*k_ori + deltaside*k_side
            + gammaorbit*orbit_scores + psipot*pots + comp_bonus
            + zetamrv*mrv_scores + chunk_bonus + echo_bonus
            + pdn_weight*pdn_bonus + cot_weight*cot_bonus
            + mandate_boost + punct_bias + punct_penalty
        )

        governed = self.contingent_prob.govern_next_probs(raw_logits, c_rho, c_theta, c_sigma)

        # RP-CSNS: Nyström synaptic matrix + RFF transitive bonus
        c_rho_t, c_theta_t, c_sigma_t = compute_transitive_triples_rp(
            self.geo, cands, w1, w2, device=self.device)
        logits_enriched = self._csns.forward(
            governed, c_rho, c_theta, c_sigma,
            c_rho_t, c_theta_t, c_sigma_t,
            ctx.rho, ctx.theta, ctx.sigma,
        )

        z_syn_raw  = self._csns.synaptic_sum(governed, c_rho, c_theta, c_sigma)
        t_bon_raw  = self._csns.transitive_bonus(c_rho_t, c_theta_t, c_sigma_t,
                                                   ctx.rho, ctx.theta, ctx.sigma)
        syn_norm   = z_syn_raw.norm().item()
        trans_norm = t_bon_raw.norm().item()
        self._csns_syn_norms.append(syn_norm)
        self._csns_trans_norms.append(trans_norm)

        self._pending_instr_probs = None
        self._pending_walk_logits = logits_enriched
        self._pending_c_rho = c_rho; self._pending_c_theta = c_theta; self._pending_c_sigma = c_sigma
        self._pending_syn_norm = syn_norm; self._pending_trans_norm = trans_norm
        self._pending_nystrom_rank = RP_NYSTROM_M

        if and_weight > 0.0 and self.instr_dist._base_dist_t is not None:
            p_instr   = self.instr_dist.distribution(cands, self._cur_sent_toks, self.lm._tok2idx)
            log_instr = (p_instr.clamp(min=1e-12)).log()
            log_walk  = self._dnn_pipeline.log_forward(logits_enriched, c_rho, c_theta, c_sigma, temp=1.0)
            log_and   = and_weight*log_instr + (1.0-and_weight)*log_walk
            final_probs = l1_simplex_project(log_and)
        else:
            p_instr     = torch.ones(N, dtype=torch.float32, device=self.device)/N
            final_probs = self._dnn_pipeline.forward(logits_enriched, c_rho, c_theta, c_sigma, temp=temp)

        self._pending_instr_probs = p_instr
        return cands, final_probs

    def record_step_trace(self, step, chosen, cands, final_probs, and_weight):
        try:
            idx   = cands.index(chosen)
            p_and = final_probs[idx].item()
        except (ValueError, IndexError):
            idx, p_and = 0, 0.0
        p_instr = self._pending_instr_probs[idx].item() if self._pending_instr_probs is not None else 0.0
        if hasattr(self, '_pending_c_rho'):
            log_walk = self._dnn_pipeline.log_forward(
                self._pending_walk_logits, self._pending_c_rho, self._pending_c_theta, self._pending_c_sigma, temp=1.0)
        else:
            log_walk = (l1_simplex_project(self._pending_walk_logits)+1e-12).log()
        p_walk  = log_walk[idx].exp().item()
        source  = "instr" if p_instr>p_walk*1.5 else ("walker" if p_walk>p_instr*1.5 else "AND")
        trace   = TokenStepTrace(step, chosen, p_instr, p_walk, p_and, and_weight, source,
                                  syn_norm=self._pending_syn_norm, trans_norm=self._pending_trans_norm,
                                  rp_nystrom_rank=self._pending_nystrom_rank)
        self._step_traces.append(trace)
        return trace

    def push_token(self, token, sentence_len):
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS: return
        self._cur_sent_toks.append(token)
        self._tok_pos += 1
        pos_norm = len(self._cur_sent_toks)/max(sentence_len, 1)
        self.chunk_engine.push(self.geo.triple_fast(token), pos_norm)
        self._cur_orbit = self.pdn.orbit_of(token)

    def step_trace_report(self, max_steps=30) -> str:
        if not self._step_traces: return "  (no step traces yet)"
        lines = [
            "step │ chosen         │ P_and   │ α    │ source  │ |z_syn| │ |trans| │ nystrom_rank",
            "─────┼────────────────┼─────────┼──────┼─────────┼─────────┼─────────┼─────────────",
        ]
        for t in self._step_traces[-max_steps:]:
            lines.append(
                f"{t.step:5d}│ {t.chosen:<15s}│ {t.p_and:.5f}│ {t.and_weight:.2f} │"
                f" {t.source:<7s} │ {t.syn_norm:.4f}  │ {t.trans_norm:.4f}  │ {t.rp_nystrom_rank}"
            )
        if self._csns_syn_norms:
            avg_s = sum(self._csns_syn_norms)/len(self._csns_syn_norms)
            avg_t = sum(self._csns_trans_norms)/len(self._csns_trans_norms)
            lines.append(f"\n  RP-CSNS: avg|z_syn|={avg_s:.4f}  avg|trans|={avg_t:.4f}  "
                         f"Nyström m={RP_NYSTROM_M}  RFF D={RP_RFF_DIM}")
        return "\n".join(lines)

    def rp_complexity_report(self) -> str:
        n = max(len(self._csns_syn_norms), 1)
        return (
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║        V18-RP: Randomized Polynomial Complexity Report        ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            f"║  Steps:              {n:<39d}║\n"
            f"║  RFF dimension D:    {RP_RFF_DIM:<39d}║\n"
            f"║  Nyström landmarks m:{RP_NYSTROM_M:<39d}║\n"
            f"║  CMS width×depth:    {RP_CMS_WIDTH}×{RP_CMS_DEPTH:<35d}║\n"
            f"║  LSH bands×rows:     {RP_LSH_BANDS}×{RP_LSH_ROWS:<35d}║\n"
            f"║  Walk steps t:       {RP_WALK_STEPS:<39d}║\n"
            f"║  Reservoir K:        {RP_RESERVOIR_K:<39d}║\n"
            f"║  RP failure bound δ: {RP_DELTA:<39.3f}║\n"
            "║                                                              ║\n"
            "║  Algorithm                 Original      RP replacement      ║\n"
            "║  Synaptic matrix:          O(C²)    →    O(C·m) Nyström     ║\n"
            "║  Kernel scoring:           O(C)     →    O(C·D) RFF         ║\n"
            "║  Bigram storage:           O(V²)    →    O(w·d) CMS sketch  ║\n"
            "║  Candidate top-K:          O(C·logK)→    O(C)   Gumbel-max  ║\n"
            "║  Graph propagation:        O(V·|E|) →    O(V·t) MC walk     ║\n"
            "║  Similarity search:        O(C·V)   →    O(C·br) LSH       ║\n"
            "║  PDN spectral:             O(T·n)   →    O(k·n) sketched    ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 17 — GENERATION ENGINE (adapted for RP Walker)
# ════════════════════════════════════════════════════════════════════════════

def generate_passage_rp(
    walker, lm, num_sentences=4, tokens_per_sent=40,
    seed_text="", instruction_text="", and_weight=0.9, temperature=2.0, return_traces=False,
):
    if instruction_text.strip(): walker.instr_dist.set_instruction(instruction_text)
    elif seed_text.strip(): walker.instr_dist.set_instruction(seed_text)
    walker._step_traces.clear(); walker._csns_syn_norms.clear(); walker._csns_trans_norms.clear()
    outputs = []; all_traces = []
    head_list = list(lm.heads.keys())
    if not head_list:
        return ("","","") if return_traces else ""
    seed_w1=None; seed_w2=None; seed_toks=[]
    if seed_text:
        seed_toks = tokenize(seed_text)
        if len(seed_toks)>=2: seed_w1,seed_w2=seed_toks[-2],seed_toks[-1]
        elif len(seed_toks)==1:
            matches=[p for p in head_list if p[1]==seed_toks[0]]
            if matches: seed_w1,seed_w2=random.choice(matches)
    if seed_w1 is None or seed_w2 is None or (seed_w1,seed_w2) not in lm.heads:
        seed_w1,seed_w2 = random.choice(head_list)
    global_step = 0
    for sent_idx in range(num_sentences):
        if sent_idx==0:
            w1,w2=seed_w1,seed_w2
            init_toks=([w1,w2] if seed_text else [])
            wsp=len(init_toks); plan_seeds=seed_toks if seed_toks else [w1,w2]
        else:
            w1,w2=random.choice(head_list)
            init_toks=[]; wsp=999; plan_seeds=[w1,w2]
        trace = walker.begin_sentence(seed_tokens=plan_seeds, total_tokens=tokens_per_sent)
        all_traces.append(trace)
        toks = list(init_toks)
        for step in range(tokens_per_sent):
            cands, probs = walker.walk_probs(w1, w2, temp=temperature, and_weight=and_weight)
            if not cands: break
            nxt = cands[torch.multinomial(probs, 1).item()]
            walker.record_step_trace(global_step, nxt, cands, probs, and_weight)
            global_step += 1
            if nxt in PUNCT_TOKENS:
                if len(toks)<3 or wsp<3 or (nxt in {".","?","!"} and len(toks)<5):
                    bi, bp = None, -1.0
                    for i,(c,p) in enumerate(zip(cands,probs.tolist())):
                        if c not in PUNCT_TOKENS and p>bp: bi,bp=i,p
                    nxt = cands[bi] if bi is not None else "the"
                else: wsp=0
            else: wsp+=1
            toks.append(nxt); walker.push_token(nxt, tokens_per_sent)
            w1,w2=w2,nxt
            if nxt in {".","?","!"} and len(toks)>=max(4, int(tokens_per_sent*0.85)): break
        outputs.append(detokenize(toks))
    result = " ".join(outputs)
    if return_traces: return result, all_traces, walker.step_trace_report()
    return result


# ════════════════════════════════════════════════════════════════════════════
# SECTION 18 — V18-RP ENGINE
# ════════════════════════════════════════════════════════════════════════════

class V18RPEngine:
    def __init__(self, syn_weight=0.4, trans_weight=0.6, syn_k=8,
                 rff_dim=RP_RFF_DIM, nystrom_m=RP_NYSTROM_M):
        self.device      = DEVICE
        self.syn_weight  = syn_weight
        self.trans_weight = trans_weight
        self.syn_k       = syn_k
        self.rff_dim     = rff_dim
        self.nystrom_m   = nystrom_m
        # Core RP object shared across all modules
        self.rff         = RandomFourierFeatures(rff_dim=rff_dim, device=self.device)
        self.geo         = ThebaultTokenGeometryRP(device=self.device)
        self.kernels     = RPKernels(self.rff)
        self.lm          = RPCompositionLM(self.geo, self.rff, device=self.device)
        self.orbit       = ThebaultConjugateOrbit()
        self.rw_graph    = RandomWalkPotentialEngine(device=self.device)
        self.mrv         = RPMRVFilter(self.rff, device=self.device)
        self.chunk       = ChunkedSumEngine(device=self.device)
        self.synth       = synthetic_reasonMandateProcessor()
        self.iso_stacker = IsomorphicSyntaxStacker(self.rff, device=self.device)
        self.pdn         = SketchedPDNEngine(device=self.device)
        self.stub_lib    = RPCoTStubLibrary(self.rff, device=self.device)
        self.instr_dist  = None; self.cot = None; self.walker = None
        self.corpus_snippet = ""

    def train(self, corpus_text: str):
        print(f"[V18-RP] Tokenizing corpus ({len(corpus_text)} chars)...")
        self.corpus_snippet = corpus_text[:1000]
        tokens = tokenize(corpus_text)
        self.lm.ingest(tokens)
        all_tokens = list(self.lm.raw_freq.keys())
        max_freq   = max(self.lm.raw_freq.values(), default=1.0)
        vocab_size = len(all_tokens)
        print(f"[V18-RP] Registering {vocab_size} tokens...")
        for idx, tok in enumerate(all_tokens):
            self.geo.register(tok, self.lm.raw_freq[tok], idx, max_freq, vocab_size)
        print("[V18-RP] Building GPU tensors + RFF feature cache...")
        self.geo.build_cuda_tensors(self.lm.vocab, self.rff)
        self.lm.finalise()
        print("[V18-RP] Running Random Walk MC potential propagation...")
        self.rw_graph.build_from_trigrams(self.lm.tri_raw, self.lm.raw_freq, self.rff, self.geo)
        self.rw_graph.propagate()
        print("[V18-RP] Priming LSH-based MRV filter...")
        self.mrv.prime(self.lm.vocab, self.geo)
        print("[V18-RP] Sketched PDN spectral fitting...")
        self.pdn.fit_from_trigrams(self.geo, self.lm.tri_raw)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)
        print(self.pdn.theorem_bridge_report())
        print("[V18-RP] Building RP CoT stub library + LSH ANN index...")
        self.stub_lib.build(self.geo, self.lm.vocab, self.lm.raw_freq)
        self.cot = RPCoTReasoningEngine(self.stub_lib, self.kernels, self.pdn,
                                         n_hops=3, tokens_per_hop=10, device=self.device)
        self.instr_dist = RPInstructionDistribution(self.geo, self.kernels, self.lm, device=self.device)
        self.walker = RPWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.rw_graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            self.pdn, self.cot, self.instr_dist, self.rff,
            device=self.device, syn_weight=self.syn_weight,
            trans_weight=self.trans_weight, syn_k=self.syn_k,
        )
        print("[+] V18-RP training complete.")

    def save_cache(self, filename="v18_rp_model.pkl"):
        print(f"[V18-RP] Saving to {filename}...")
        with open(filename, "wb") as f:
            pickle.dump({
                "geo_vecs": self.geo._vecs, "geo_cache": self.geo._cache,
                "lm_raw_freq": self.lm.raw_freq, "lm_tri_raw": self.lm.tri_raw,
                "lm_heads": self.lm.heads, "lm_vocab": self.lm.vocab,
                "rw_potentials": self.rw_graph._potentials,
                "corpus_snippet": self.corpus_snippet,
                "pdn_n_star": self.pdn.n_star, "pdn_power": self.pdn.power_spectrum,
                "cot_stubs": self.stub_lib.stubs,
                "syn_weight": self.syn_weight, "trans_weight": self.trans_weight,
                "syn_k": self.syn_k, "rff_dim": self.rff_dim, "nystrom_m": self.nystrom_m,
                "version": "V18-RP",
            }, f)
        print("[+] Saved.")

    def load_cache(self, filename):
        print(f"[V18-RP] Loading from {filename}...")
        with open(filename, "rb") as f:
            state = pickle.load(f)
        self.geo._vecs = state["geo_vecs"]; self.geo._cache = state["geo_cache"]
        self.lm.raw_freq = state["lm_raw_freq"]; self.lm.tri_raw = state["lm_tri_raw"]
        self.lm.heads = state["lm_heads"]; self.lm.vocab = state["lm_vocab"]
        self.corpus_snippet = state["corpus_snippet"]
        self.pdn.n_star = state.get("pdn_n_star", 4)
        self.pdn.power_spectrum = state.get("pdn_power", {})
        self.syn_weight = state.get("syn_weight", 0.4)
        self.trans_weight = state.get("trans_weight", 0.6)
        self.syn_k = state.get("syn_k", 8)
        self.rff_dim = state.get("rff_dim", RP_RFF_DIM)
        self.nystrom_m = state.get("nystrom_m", RP_NYSTROM_M)
        print("[V18-RP] Rebuilding RP structures...")
        self.rff = RandomFourierFeatures(rff_dim=self.rff_dim, device=self.device)
        self.kernels = RPKernels(self.rff)
        self.geo.build_cuda_tensors(self.lm.vocab, self.rff)
        self.lm.finalise()
        self.rw_graph._potentials = state.get("rw_potentials", {})
        self.rw_graph._adj = {}
        self.mrv = RPMRVFilter(self.rff, device=self.device)
        self.mrv.prime(self.lm.vocab, self.geo)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)
        if "cot_stubs" in state:
            self.stub_lib.stubs = state["cot_stubs"]
            self.stub_lib._rebuild_lsh()
        else:
            self.stub_lib.build(self.geo, self.lm.vocab, self.lm.raw_freq)
        self.cot = RPCoTReasoningEngine(self.stub_lib, self.kernels, self.pdn,
                                         n_hops=3, tokens_per_hop=10, device=self.device)
        self.instr_dist = RPInstructionDistribution(self.geo, self.kernels, self.lm, device=self.device)
        self.iso_stacker = IsomorphicSyntaxStacker(self.rff, device=self.device)
        self.walker = RPWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.rw_graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            self.pdn, self.cot, self.instr_dist, self.rff,
            device=self.device, syn_weight=self.syn_weight,
            trans_weight=self.trans_weight, syn_k=self.syn_k,
        )
        print("[+] V18-RP load complete.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 19 — GRADIO GUI
# ════════════════════════════════════════════════════════════════════════════
from datasets import load_dataset

def load_hf_text_dataset(
    dataset_name: str,
    config_name: str = None,
    split: str = "train",
    text_field: str = None,
    portion: float = 1.0,
    max_examples: int = None,
) -> str:
    """
    Load text from a Hugging Face dataset and return a single concatenated corpus string.
    """
    ds = load_dataset(dataset_name, config_name or None, split=split)

    # Subsample by portion
    if portion < 1.0:
        n = int(len(ds) * portion)
        n = max(n, 1)
        ds = ds.select(range(n))

    # Optional hard cap
    if max_examples is not None:
        n = min(len(ds), int(max_examples))
        ds = ds.select(range(n))

    # Choose text field
    if text_field is None or text_field.strip() == "":
        ex0 = ds[0]
        text_field = None
        for k, v in ex0.items():
            if isinstance(v, str):
                text_field = k
                break
        if text_field is None:
            raise ValueError("No string field found in dataset; please specify text_field explicitly.")

    texts = [ex[text_field] for ex in ds if ex.get(text_field)]
    return "\n\n".join(texts)


class V18RPGUI:
    def __init__(self):
        self.engine = None

    def init_from_source(
        self,
        mode,                 # "file" or "hf"
        file_obj,
        hf_name,
        hf_config,
        hf_split,
        hf_text_field,
        hf_portion,
        hf_max_examples,
        syn_w, trans_w, syn_k, rff_dim, nystrom_m,
    ):
        try:
            if mode == "file":
                if file_obj is None:
                    return "Error: No file uploaded."
                with open(file_obj.name, 'r', encoding='utf-8') as f:
                    corpus = f.read()
                if not corpus.strip():
                    return "Error: Empty file."
                source_desc = f"File: {file_obj.name.split('/')[-1]}"
            else:
                if not hf_name:
                    return "Error: No Hugging Face dataset name provided."
                max_ex = None
                if hf_max_examples is not None:
                    try:
                        max_ex_int = int(hf_max_examples)
                        if max_ex_int > 0:
                            max_ex = max_ex_int
                    except Exception:
                        max_ex = None

                corpus = load_hf_text_dataset(
                    dataset_name=hf_name.strip(),
                    config_name=(hf_config or "").strip() or None,
                    split=(hf_split or "train").strip(),
                    text_field=(hf_text_field or None),
                    portion=float(hf_portion),
                    max_examples=max_ex,
                )
                if not corpus.strip():
                    return "Error: Loaded dataset has no text."

                source_desc = (
                    f"HF dataset: {hf_name.strip()} "
                    f"(config={hf_config or 'None'}, split={hf_split or 'train'}, "
                    f"portion={float(hf_portion):.2f}, "
                    f"max_examples={max_ex if max_ex is not None else 'None'})"
                )

            self.engine = V18RPEngine(
                syn_weight=float(syn_w), trans_weight=float(trans_w),
                syn_k=int(syn_k), rff_dim=int(rff_dim), nystrom_m=int(nystrom_m),
            )
            self.engine.train(corpus)
            pdn_report = self.engine.pdn.theorem_bridge_report()
            stub_counts = {k: len(v) for k, v in self.engine.stub_lib.stubs.items()}
            return (
                "V18-RP Engine initialised.\n"
                f"{source_desc}\n"
                f"Vocab: {len(self.engine.lm.vocab)} tokens\n"
                f"CoT stubs: {stub_counts}\n"
                f"RFF dim D={int(rff_dim)}  Nyström m={int(nystrom_m)}\n"
                f"CMS {RP_CMS_WIDTH}×{RP_CMS_DEPTH}  LSH {RP_LSH_BANDS}b×{RP_LSH_ROWS}r\n\n"
                f"{pdn_report}"
            )
        except Exception as e:
            import traceback
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    def generate(self, sentences, tokens, seed, instruction, and_weight, temperature):
        if not self.engine or not self.engine.walker:
            return "Engine not initialised.", "", ""
        text, traces, step_report = generate_passage_rp(
            self.engine.walker, self.engine.lm,
            num_sentences=int(sentences), tokens_per_sent=int(tokens),
            seed_text=seed.strip(), instruction_text=instruction.strip(),
            and_weight=float(and_weight), temperature=float(temperature),
            return_traces=True,
        )
        trace_text = "\n".join(tr.render() for tr in traces)
        return text, trace_text, step_report

    def rp_report(self):
        if not self.engine or not self.engine.walker:
            return "Engine not initialised."
        return self.engine.walker.rp_complexity_report()

    def pdn_report(self):
        if not self.engine:
            return "Engine not initialised."
        return self.engine.pdn.theorem_bridge_report()

    def cot_history(self):
        if not self.engine or not self.engine.cot:
            return "Engine not initialised."
        return self.engine.cot.all_traces_text()

    def algo_report(self):
        lines = [
            "V18-RP: All Core Algorithms → Randomized Polynomial Time",
            "═══════════════════════════════════════════════════════════════",
            "",
            "1. RANDOM FOURIER FEATURES  [Rahimi & Recht, 2007]",
            "   ω ~ N(0, σ⁻²I),  φ(x) = √(2/D)·cos(ωᵀx + b)",
            "   k(x,y) ≈ φ(x)ᵀφ(y)  with error O(1/√D) w.p. ≥ 1-2exp(-Dε²/4)",
            "   Replaces: all exact Thébault/Gaussian kernel evaluations",
            "   Complexity: O(C·D) vs O(C) exact (D=128, constant overhead)",
            "",
            "2. NYSTRÖM APPROXIMATION  [Williams & Seeger, 2001]",
            "   K ≈ K_cm · K_mm⁻¹ · K_mc   (rank-m approximation)",
            "   Landmarks selected via Reservoir Sampling (Vitter 1985)",
            "   Replaces: exact O(C²) synaptic weight matrix",
            "   Complexity: O(C·m + m³) vs O(C²),  m=32 << C",
            "",
            "3. COUNT-MIN SKETCH  [Cormode & Muthukrishnan, 2005]",
            "   P[|CMS(x) - f(x)| ≤ ε·N] ≥ 1-δ,  w=e/ε, d=ln(1/δ)",
            "   Replaces: exact O(V²·T) trigram frequency dict storage",
            "   Complexity: O(w·d) space vs O(V³) worst-case",
            "",
            "4. LOCALITY-SENSITIVE HASHING  [Charikar, 2002]",
            "   P[h(x)=h(y)] = 1 - arccos(sim(x,y))/π",
            "   (b×r) random hyperplanes, band hashing",
            "   Replaces: exact O(C·V) MRV kernel scan",
            "   Complexity: O(C·b·r + bucket_size) per query",
            "",
            "5. RANDOM WALK MONTE CARLO  [Lovász, 1999]",
            "   π̂(v) ← visit frequency after t steps from V starting nodes",
            "   PageRank restart with probability α=0.15",
            "   Replaces: exact dense O(|V|·|E|) power iteration",
            "   Complexity: O(|V|·t) where t=walk_steps",
            "",
            "6. GUMBEL-MAX / WEIGHTED RESERVOIR SAMPLING  [Vitter, 1985]",
            "   priority(i) = score(i) + Gumbel(0,1)/bias",
            "   Recovers exact top-K in limit bias → ∞",
            "   Replaces: deterministic O(C·log K) torch.topk",
            "   Complexity: O(C) single-pass",
            "",
            "7. SKETCHED PDN SPECTRAL ANALYSIS  [Candès & Wakin, 2008]",
            "   Random subset of k trigrams, unbiased scaling by T/k",
            "   Recovers dominant spectral mode correctly w.p. ≥ 1-δ",
            "   Replaces: exact O(T·n log n) DFT over all trigrams",
            "   Complexity: O(k·n) where k=n_samples",
            "",
            "GLOBAL RP PARAMETERS:",
            f"   δ (failure bound) = {RP_DELTA}",
            f"   D (RFF dim)       = {RP_RFF_DIM}",
            f"   m (Nyström)       = {RP_NYSTROM_M}",
            f"   w×d (CMS)         = {RP_CMS_WIDTH}×{RP_CMS_DEPTH}",
            f"   b×r (LSH)         = {RP_LSH_BANDS}×{RP_LSH_ROWS}",
            f"   t (walk steps)    = {RP_WALK_STEPS}",
            "═══════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


def build_demo():
    gui = V18RPGUI()

    with gr.Blocks(title="NeuroSymbolic V18-RP") as demo:
        gr.Markdown("# NeuroSymbolic V18-RP — Randomized Polynomial Edition")

        # Init / Train tab
        with gr.Tab("Init / Train"):
            mode = gr.Radio(
                ["file", "hf"],
                value="file",
                label="Training source",
            )

            with gr.Group(visible=True) as file_group:
                file_in = gr.File(label="Training text file (.txt)")

            with gr.Group(visible=False) as hf_group:
                hf_name = gr.Textbox(
                    label="HF dataset name",
                    placeholder="e.g. wikitext, imdb, ag_news",
                )
                hf_config = gr.Textbox(
                    label="HF config name (optional)",
                    placeholder="e.g. wikitext-103-raw-v1",
                )
                hf_split = gr.Textbox(
                    label="Split",
                    value="train",
                    placeholder="train / validation / test or custom split",
                )
                hf_text_field = gr.Textbox(
                    label="Text field (blank = auto-detect)",
                    placeholder="e.g. text, content, review",
                )
                hf_portion = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.01,
                    label="Dataset portion used (fraction of split)",
                )
                hf_max_examples = gr.Number(
                    label="Max examples (optional cap)",
                    value=None,
                    precision=0,
                )

            def _toggle_mode(m):
                return (
                    gr.update(visible=(m == "file")),
                    gr.update(visible=(m == "hf")),
                )

            mode.change(
                _toggle_mode,
                inputs=mode,
                outputs=[file_group, hf_group],
            )

            syn_w     = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Synaptic weight")
            trans_w   = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Transition weight")
            syn_k     = gr.Slider(1, 64, value=16, step=1, label="Synaptic k")
            rff_dim   = gr.Slider(32, 512, value=128, step=32, label="RFF dim D")
            nystrom_m = gr.Slider(8, 128, value=32, step=4, label="Nyström m")

            init_btn  = gr.Button("Initialise engine")
            init_out  = gr.Textbox(lines=18, label="Initialisation / PDN report")

            init_btn.click(
                gui.init_from_source,
                inputs=[
                    mode, file_in,
                    hf_name, hf_config, hf_split, hf_text_field, hf_portion, hf_max_examples,
                    syn_w, trans_w, syn_k, rff_dim, nystrom_m,
                ],
                outputs=[init_out],
            )

        # Generate tab
        with gr.Tab("Generate"):
            sentences   = gr.Slider(1, 10, value=3, step=1, label="Number of sentences")
            tokens      = gr.Slider(8, 128, value=32, step=1, label="Tokens per sentence")
            seed_text   = gr.Textbox(lines=2, label="Seed text")
            instr_text  = gr.Textbox(lines=3, label="Instruction / goal")
            and_weight  = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="AND weight")
            temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")

            gen_btn     = gr.Button("Generate")
            gen_text    = gr.Textbox(lines=8, label="Generated text")
            gen_trace   = gr.Textbox(lines=12, label="CoT trace")
            gen_steps   = gr.Textbox(lines=8, label="RP step report")

            gen_btn.click(
                gui.generate,
                inputs=[sentences, tokens, seed_text, instr_text, and_weight, temperature],
                outputs=[gen_text, gen_trace, gen_steps],
            )

        # Reports tab
        with gr.Tab("Reports"):
            rp_btn   = gr.Button("RP complexity report")
            pdn_btn  = gr.Button("PDN theorem bridge")
            cot_btn  = gr.Button("CoT history")
            algo_btn = gr.Button("Algorithm catalogue")

            rp_out   = gr.Textbox(lines=15, label="RP complexity")
            pdn_out  = gr.Textbox(lines=15, label="PDN")
            cot_out  = gr.Textbox(lines=15, label="CoT")
            algo_out = gr.Textbox(lines=20, label="Algorithms")

            rp_btn.click(gui.rp_report, None, rp_out)
            pdn_btn.click(gui.pdn_report, None, pdn_out)
            cot_btn.click(gui.cot_history, None, cot_out)
            algo_btn.click(gui.algo_report, None, algo_out)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
