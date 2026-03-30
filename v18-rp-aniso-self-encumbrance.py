#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V18-RP-ANISO — Non-Isotropic Inter-Candidate Edition
===============================================================================

WHAT CHANGED FROM V18-RP → V18-RP-ANISO
─────────────────────────────────────────

The core addition is NON-ISOTROPIC PREDICTION WEIGHTS.

In V18-RP every candidate token was scored only against the generation context
(w1, w2 bigram centroid). That made the kernel isotropic: two candidates that
are geometrically close to each other in Thébault space received the same
bonus even though they are substitutable synonyms.

V18-RP-ANISO adds two interacting mechanisms:

┌──────────────────────────────────────────────────────────────────────────┐
│ 1. SENTENCE OBJECT-OF-INTEREST (OOI) TRACKER                            │
│    Each sentence maintains a live set of "objects of interest" — the      │
│    geometrically salient tokens already generated (high rho, mid theta). │
│    Every new candidate is scored against EACH OOI via the anisotropic    │
│    directional kernel (see below), not just against the context bigram.  │
│                                                                           │
│ 2. ANISOTROPIC DIRECTIONAL KERNEL  (AnisoDirKernel)                      │
│    k_aniso(a, b) = exp(-λ_rho·Δρ² - λ_theta·Δθ_aniso² - λ_sigma·Δσ²)   │
│    where Δθ_aniso is the SIGNED angular difference modulated by ρ:       │
│      Δθ_aniso = (θ_b - θ_a) · (1 + α·ρ_a)                               │
│    High-rho tokens stretch the angular axis → tokens that are similar    │
│    in rho but differ slightly in theta are pushed further apart in        │
│    kernel space.  This breaks the isotropy of the standard cosine kernel. │
│                                                                           │
│ 3. INTER-CANDIDATE REPULSION (DPP-LITE)                                  │
│    After scoring each candidate against the OOI set, we compute a        │
│    lightweight diversity penalty:                                          │
│      repulsion_i = Σ_j≠i  k_aniso(c_i, c_j) · final_prob_j              │
│    This is subtracted (scaled) from the logits so that a cluster of      │
│    near-synonym candidates is forced to differentiate — only the best    │
│    representative of each geometric cluster survives with high weight.   │
│                                                                           │
│ 4. FEATURE DIMENSION EXTENDED 17 → 19                                    │
│    Feature 18: ooi_affinity   — mean aniso kernel score vs OOI set       │
│    Feature 19: inter_repulsion — DPP-lite diversity penalty               │
└──────────────────────────────────────────────────────────────────────────┘

All other algorithms (RFF, Nyström, CMS, reservoir, LSH, random walk, PDN,
CoT, fitted line) are unchanged from V18-RP.

THEORETICAL NOTE
────────────────
The anisotropic kernel belongs to the family of Matern-style directional
kernels studied in:
  Paciorek & Schervish (2006) "Spatial modelling using a new class of
  nonstationary covariance functions."
The ρ-dependent axis stretching is equivalent to a locally-adaptive
length-scale in the θ direction, giving a non-stationary (non-isotropic)
Gaussian process prior over token geometry.

The DPP-lite repulsion is a first-order approximation of Determinantal
Point Process diversity (Kulesza & Taskar 2012) that runs in O(C²) but
with C typically ≤ 200 so it remains practical.
"""

from __future__ import annotations
import re, math, random, unicodedata, pickle, argparse, struct, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

RP_SEED          = 42
RP_DELTA         = 0.05
RP_RFF_DIM       = 128
RP_NYSTROM_M     = 32
RP_CMS_WIDTH     = 1024
RP_CMS_DEPTH     = 5
RP_LSH_BANDS     = 8
RP_LSH_ROWS      = 4
RP_WALK_STEPS    = 20
RP_RESERVOIR_K   = 64

# ── ANISO CONFIG ─────────────────────────────────────────────────────────
ANISO_LAMBDA_RHO   = 8.0   # Mahalanobis stretch along ρ axis
ANISO_LAMBDA_THETA = 4.0   # Mahalanobis stretch along θ axis
ANISO_LAMBDA_SIGMA = 2.0   # Mahalanobis stretch along σ axis
ANISO_ALPHA        = 1.5   # ρ-dependent θ-axis anisotropy factor
ANISO_OOI_MAX      = 12    # max objects-of-interest tracked per sentence
ANISO_OOI_RHO_THR  = 0.25  # minimum ρ for a token to be OOI-eligible
ANISO_REPULSION_W  = 0.55  # weight of inter-candidate repulsion in final logits
ANISO_OOI_W        = 0.70  # weight of OOI affinity bonus in final logits
SELF_ENCUMBRANCE_W = 0.35  # post-ANISO self-encumbrance weight

_rng    = random.Random(RP_SEED)
_np_rng = np.random.default_rng(RP_SEED)
torch.manual_seed(RP_SEED)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0b — SHARED ACTIVATION PRIMITIVES  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def smooth_power_relu(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x_safe = x.clamp(-50.0, 50.0)
    return (x_safe * x_safe) / (x_safe.abs() + eps)

def signed_power(x: torch.Tensor, p: float) -> torch.Tensor:
    return x.sign() * (x.abs().clamp(max=30.0) + 1e-12).pow(p)

def l2_array_normalize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    return x / (sq_sum + eps).sqrt()

def l1_simplex_project(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    x_shifted = x - x.min()
    x_pos     = smooth_power_relu(x_shifted).clamp(min=eps)
    total     = x_pos.sum()
    if total.item() == 0.0 or not torch.isfinite(total):
        return torch.full_like(x, 1.0 / max(x.shape[0], 1))
    result = (x_pos / total).clamp(min=eps)
    return result / result.sum()

def layer_norm_array(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(); std = x.std()
    return (x - mu) / (std + eps) if std.item() >= eps else x - mu


def self_encumbrance_from_probs(prob_vec: torch.Tensor, strength: float = SELF_ENCUMBRANCE_W) -> torch.Tensor:
    """
    Post-ANISO self-encumbrance over a candidate distribution.

    Sort probabilities descending, build forward and backward cumulative mass,
    then map the signed encumbrance back to original candidate order.
    High-probability candidates receive a negative correction, while lower-mass
    candidates receive a relative lift. The output is z-normalised so it can be
    added directly to logits as a stable post-feature adjustment.
    """
    C = prob_vec.shape[0]
    if C <= 1:
        return torch.zeros_like(prob_vec)
    p = prob_vec.clamp(min=1e-12)
    p = p / p.sum().clamp(min=1e-12)
    sort_idx = torch.argsort(p, descending=True)
    inv_idx = torch.empty_like(sort_idx)
    inv_idx[sort_idx] = torch.arange(C, device=p.device)
    ps = p[sort_idx]
    c_fwd = torch.cumsum(ps, dim=0)
    c_rev = torch.cumsum(ps.flip(0), dim=0).flip(0)
    rank = torch.linspace(1.0, 0.0, C, device=p.device, dtype=p.dtype)
    raw = -(c_fwd - c_rev) * (0.5 + 0.5 * rank)
    std = raw.std(unbiased=False)
    raw = (raw - raw.mean()) / (std + 1e-8) if std.item() > 1e-8 else raw - raw.mean()
    return raw[inv_idx] * strength


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0c — ANISOTROPIC DIRECTIONAL KERNEL  ← NEW
# ════════════════════════════════════════════════════════════════════════════

class AnisoDirKernel:
    """
    Non-isotropic kernel over Thébault space (ρ, θ, σ).

    k_aniso(a, b) = exp(
        -λ_rho   · (ρ_b - ρ_a)²
        -λ_theta · Δθ_aniso(a,b)²
        -λ_sigma · (σ_b - σ_a)²
    )

    where   Δθ_aniso = (θ_b - θ_a) · (1 + α · ρ_a)

    Effect: high-ρ "anchor" tokens stretch the angular dimension, so two
    candidates that differ only slightly in θ are treated as more distinct
    when the anchor is geometrically confident (high ρ).  This breaks the
    isotropy of the plain cosine kernel used in V18-RP.
    """

    def __init__(self,
                 lambda_rho:   float = ANISO_LAMBDA_RHO,
                 lambda_theta: float = ANISO_LAMBDA_THETA,
                 lambda_sigma: float = ANISO_LAMBDA_SIGMA,
                 alpha:        float = ANISO_ALPHA,
                 device: torch.device = DEVICE,
                 dtype:  torch.dtype  = torch.float32):
        self.lr  = lambda_rho
        self.lt  = lambda_theta
        self.ls  = lambda_sigma
        self.a   = alpha
        self.device = device
        self.dtype  = dtype

    # ── scalar anchor × batch candidates ────────────────────────────────
    def score_anchor_vs_batch(
        self,
        anc_rho:   float, anc_theta: float, anc_sigma: float,
        c_rho:   torch.Tensor, c_theta: torch.Tensor, c_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (C,) scores; anchor is a single triple."""
        d_rho   = c_rho   - anc_rho
        d_theta = (c_theta - anc_theta) * (1.0 + self.a * anc_rho)
        d_sigma = c_sigma  - anc_sigma
        return torch.exp(
            -self.lr * d_rho**2
            -self.lt * d_theta**2
            -self.ls * d_sigma**2
        )

    # ── batch × batch  (C × C)  for inter-candidate repulsion ───────────
    def gram_matrix(
        self,
        c_rho:   torch.Tensor,   # (C,)
        c_theta: torch.Tensor,   # (C,)
        c_sigma: torch.Tensor,   # (C,)
    ) -> torch.Tensor:
        """
        Returns (C, C) Gram matrix K where K[i,j] = k_aniso(c_i, c_j).
        The diagonal is 1 by definition (self-similarity).
        Because anisotropy is anchor-dependent, K is NOT necessarily symmetric;
        K[i,j] uses c_i as anchor: Δθ_aniso = (θ_j-θ_i)·(1+α·ρ_i).
        """
        C = c_rho.shape[0]
        # broadcast: rows = anchors (i), cols = candidates (j)
        rho_i   = c_rho.unsqueeze(1)    # (C,1)
        theta_i = c_theta.unsqueeze(1)  # (C,1)
        sigma_i = c_sigma.unsqueeze(1)  # (C,1)

        rho_j   = c_rho.unsqueeze(0)    # (1,C)
        theta_j = c_theta.unsqueeze(0)  # (1,C)
        sigma_j = c_sigma.unsqueeze(0)  # (1,C)

        d_rho   = rho_j   - rho_i
        d_theta = (theta_j - theta_i) * (1.0 + self.a * rho_i)
        d_sigma = sigma_j  - sigma_i

        K = torch.exp(
            -self.lr * d_rho**2
            -self.lt * d_theta**2
            -self.ls * d_sigma**2
        )
        return K  # (C, C)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0d — SENTENCE OOI TRACKER  ← NEW
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class OOIEntry:
    token: str
    rho:   float
    theta: float
    sigma: float


class SentenceOOITracker:
    """
    Tracks "objects of interest" within the current sentence.

    A token is OOI-eligible if:
      • ρ ≥ ANISO_OOI_RHO_THR  (geometrically confident)
      • not a stop-word / punctuation
      • not already in the OOI set (deduplication by token)

    The OOI set is maintained as a fixed-size ring (newest evicts oldest once
    full), so it always reflects the most recent salient context.

    At each generation step, every candidate is scored against the full OOI
    set via AnisoDirKernel, giving a richer, sentence-aware affinity signal
    than the plain bigram context used elsewhere.
    """

    def __init__(self,
                 aniso_kernel: AnisoDirKernel,
                 max_ooi:   int   = ANISO_OOI_MAX,
                 rho_thr:   float = ANISO_OOI_RHO_THR,
                 device: torch.device = DEVICE):
        self.kernel  = aniso_kernel
        self.max_ooi = max_ooi
        self.rho_thr = rho_thr
        self.device  = device
        self._ooi: List[OOIEntry] = []

    def reset(self):
        self._ooi.clear()

    def push(self, token: str, triple) -> bool:
        """
        Offer a token to the OOI set.
        Returns True if the token was accepted.
        """
        if triple.rho < self.rho_thr:
            return False
        # Deduplicate
        if any(e.token == token for e in self._ooi):
            return False
        entry = OOIEntry(token, triple.rho, triple.theta, triple.sigma)
        if len(self._ooi) >= self.max_ooi:
            self._ooi.pop(0)          # evict oldest
        self._ooi.append(entry)
        return True

    @property
    def size(self) -> int:
        return len(self._ooi)

    # ── main scoring: (C,) ooi_affinity for each candidate ──────────────
    @torch.no_grad()
    def ooi_affinity(
        self,
        c_rho:   torch.Tensor,
        c_theta: torch.Tensor,
        c_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each candidate c_i, compute the mean anisotropic kernel score
        against all OOI entries.  Returns (C,) tensor in [0, 1].

        If the OOI set is empty, returns zeros.
        """
        C = c_rho.shape[0]
        if not self._ooi:
            return torch.zeros(C, device=self.device)

        # Accumulate scores from each OOI anchor
        agg = torch.zeros(C, device=self.device)
        for entry in self._ooi:
            agg += self.kernel.score_anchor_vs_batch(
                entry.rho, entry.theta, entry.sigma,
                c_rho, c_theta, c_sigma)

        return agg / len(self._ooi)   # mean in [0,1]

    # ── inter-candidate repulsion: (C,) DPP-lite penalty ────────────────
    @torch.no_grad()
    def inter_candidate_repulsion(
        self,
        c_rho:    torch.Tensor,
        c_theta:  torch.Tensor,
        c_sigma:  torch.Tensor,
        prob_vec: torch.Tensor,        # current soft probability (C,)
    ) -> torch.Tensor:
        """
        DPP-lite repulsion:
            repulsion_i = Σ_{j≠i} K[i,j] · prob_j

        This penalises a candidate proportionally to how similar it is
        (in anisotropic kernel space) to other high-probability candidates,
        breaking isotropy across the candidate set.

        Returns (C,) repulsion scores (higher = more redundant = more penalised).
        The caller should SUBTRACT this from logits.
        """
        C = c_rho.shape[0]
        if C < 2:
            return torch.zeros(C, device=self.device)

        K = self.kernel.gram_matrix(c_rho, c_theta, c_sigma)  # (C, C)

        # Zero diagonal so a token doesn't repel itself
        K = K * (1.0 - torch.eye(C, device=self.device))

        p = prob_vec.to(self.device).clamp(min=0.0)
        if p.sum().item() < 1e-12:
            p = torch.ones(C, device=self.device) / C
        else:
            p = p / p.sum()

        repulsion = K @ p   # (C,) weighted sum of similarity to other cands
        return repulsion


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RANDOM FOURIER FEATURES  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class RandomFourierFeatures:
    def __init__(self, input_dim=4, rff_dim=RP_RFF_DIM,
                 sigma_rho=1.0, sigma_theta=0.5, sigma_sigma=2.0,
                 device=DEVICE, dtype=torch.float32):
        self.rff_dim = rff_dim; self.device = device; self.dtype = dtype
        g = torch.Generator(); g.manual_seed(RP_SEED)
        self.omega_rho   = torch.randn(rff_dim,1,generator=g,dtype=dtype,device=device)/sigma_rho
        self.omega_theta = torch.randn(rff_dim,1,generator=g,dtype=dtype,device=device)/sigma_theta
        self.omega_sigma = torch.randn(rff_dim,1,generator=g,dtype=dtype,device=device)/sigma_sigma
        self.bias_rho    = torch.rand(rff_dim,generator=g,dtype=dtype,device=device)*2*math.pi
        self.bias_theta  = torch.rand(rff_dim,generator=g,dtype=dtype,device=device)*2*math.pi
        self.bias_sigma  = torch.rand(rff_dim,generator=g,dtype=dtype,device=device)*2*math.pi
        self._scale = math.sqrt(2.0/rff_dim)

    def features(self, rho, theta, sigma) -> torch.Tensor:
        pr = self.omega_rho   @ rho.unsqueeze(0)   + self.bias_rho.unsqueeze(1)
        pt = self.omega_theta @ theta.unsqueeze(0) + self.bias_theta.unsqueeze(1)
        ps = self.omega_sigma @ sigma.unsqueeze(0) + self.bias_sigma.unsqueeze(1)
        return torch.cat([(self._scale*torch.cos(pr)).T,
                          (self._scale*torch.cos(pt)).T,
                          (self._scale*torch.cos(ps)).T], dim=1)

    def kernel_approx(self, rho_a,theta_a,sigma_a, rho_b,theta_b,sigma_b):
        return self.features(rho_a,theta_a,sigma_a) @ self.features(rho_b,theta_b,sigma_b).T

    def kernel_scalar(self, rho_a,theta_a,sigma_a, rho_b,theta_b,sigma_b):
        ra = torch.tensor([rho_a],dtype=self.dtype,device=self.device)
        ta = torch.tensor([theta_a],dtype=self.dtype,device=self.device)
        sa = torch.tensor([sigma_a],dtype=self.dtype,device=self.device)
        return (self.features(ra,ta,sa) @ self.features(rho_b,theta_b,sigma_b).T).squeeze(0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NYSTRÖM APPROXIMATION  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class NystromSynapticMatrix:
    def __init__(self, rff, n_landmarks=RP_NYSTROM_M, top_k=8, device=DEVICE, dtype=torch.float32):
        self.rff=rff; self.n_landmarks=n_landmarks; self.top_k=top_k
        self.device=device; self.dtype=dtype

    @torch.no_grad()
    def build(self, c_rho, c_theta, c_sigma) -> torch.Tensor:
        C = c_rho.shape[0]; m = min(self.n_landmarks, C)
        lm_idx = torch.tensor(_reservoir_sample_indices(C,m), dtype=torch.long, device=self.device)
        phi_c  = self.rff.features(c_rho, c_theta, c_sigma)
        phi_lm = self.rff.features(c_rho[lm_idx], c_theta[lm_idx], c_sigma[lm_idx])
        K_cm   = phi_c @ phi_lm.T
        K_mm   = phi_lm @ phi_lm.T
        try:
            U,S,Vh = torch.linalg.svd(K_mm, full_matrices=False)
            S_inv  = torch.where(S > S.max()*1e-4, 1.0/S, torch.zeros_like(S))
            K_mm_inv = Vh.T @ torch.diag(S_inv) @ U.T
        except Exception:
            K_mm_inv = torch.eye(m, dtype=self.dtype, device=self.device)*0.01
        W = (K_cm @ K_mm_inv @ K_cm.T).clamp(0.0, 1.0)
        W.fill_diagonal_(0.0)
        if self.top_k < C:
            kth,_ = torch.topk(W, min(self.top_k,C), dim=1)
            W = W * (W >= kth[:,-1].unsqueeze(1)).float()
        return W / W.sum(dim=1, keepdim=True).clamp(min=1e-8)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RESERVOIR SAMPLING  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def _reservoir_sample_indices(n: int, k: int) -> List[int]:
    k = min(k,n); res = list(range(k))
    for i in range(k,n):
        j = _rng.randint(0,i)
        if j < k: res[j] = i
    return res

def reservoir_topk(scores: torch.Tensor, k: int, bias: float = 2.0) -> torch.Tensor:
    C = scores.shape[0]; k = min(k,C)
    u = torch.rand(C, device=scores.device, dtype=scores.dtype).clamp(1e-10, 1-1e-10)
    return torch.topk(scores + -(-u.log()).log()/bias, k).indices


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COUNT-MIN SKETCH  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class CountMinSketch:
    def __init__(self, width=RP_CMS_WIDTH, depth=RP_CMS_DEPTH):
        self.width=width; self.depth=depth
        self.table = np.zeros((depth,width), dtype=np.float32)
        self._seeds = [_rng.randint(0,2**31) for _ in range(depth)]

    def _hash(self, key, row):
        h = hashlib.md5(f"{self._seeds[row]}:{key}".encode()).digest()
        return int.from_bytes(h[:4],'little') % self.width

    def update(self, key, count=1.0):
        for i in range(self.depth): self.table[i, self._hash(key,i)] += count

    def query(self, key):
        return float(min(self.table[i, self._hash(key,i)] for i in range(self.depth)))

    def update_pair(self,w1,w2,count=1.0):   self.update(f"__BIGRAM__{w1}||{w2}",count)
    def query_pair(self,w1,w2):               return self.query(f"__BIGRAM__{w1}||{w2}")
    def update_triple(self,w1,w2,w3,count=1.0): self.update(f"__TRIGRAM__{w1}||{w2}||{w3}",count)
    def query_triple(self,w1,w2,w3):          return self.query(f"__TRIGRAM__{w1}||{w2}||{w3}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LSH INDEX  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class LSHIndex:
    def __init__(self, feature_dim=3*RP_RFF_DIM, n_bands=RP_LSH_BANDS, n_rows=RP_LSH_ROWS,
                 device=DEVICE, dtype=torch.float32):
        self.n_bands=n_bands; self.n_rows=n_rows; self.device=device; self.dtype=dtype
        g = torch.Generator(); g.manual_seed(RP_SEED+7)
        self.planes = F.normalize(
            torch.randn(n_bands*n_rows, feature_dim, generator=g, dtype=dtype, device=device), dim=1)
        self._table: Dict[Tuple[int,int],List[int]] = {}
        self._feats: Optional[torch.Tensor] = None
        self._vocab: List[str] = []

    def build(self, features, vocab):
        self._feats=features; self._vocab=vocab; self._table={}
        bits = (features @ self.planes.T > 0).int()
        for v in range(features.shape[0]):
            for b in range(self.n_bands):
                s = b*self.n_rows
                key = (b, hash(tuple(bits[v,s:s+self.n_rows].tolist())))
                self._table.setdefault(key,[]).append(v)

    def query_candidates(self, q_feat, max_cands=50):
        if self._feats is None: return []
        bits = (q_feat.to(self.device) @ self.planes.T > 0).int()
        cands: Set[int] = set()
        for b in range(self.n_bands):
            s = b*self.n_rows
            key = (b, hash(tuple(bits[s:s+self.n_rows].tolist())))
            for idx in self._table.get(key,[]):
                cands.add(idx)
                if len(cands) >= max_cands: break
        return list(cands)[:max_cands]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RANDOM WALK POTENTIAL ENGINE  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class RandomWalkPotentialEngine:
    def __init__(self, n_walks=RP_WALK_STEPS, walk_length=12, restart_p=0.15, device=DEVICE):
        self.n_walks=n_walks; self.walk_length=walk_length; self.restart_p=restart_p
        self.device=device
        self._potentials: Dict[str,float] = {}
        self._adj: Dict[str,List[Tuple[str,float]]] = {}
        self._all_toks: List[str] = []

    def build_from_trigrams(self, tri_raw, raw_freq, rff, geo):
        self._all_toks = list(raw_freq.keys())
        for tok in self._all_toks: self._adj[tok] = []
        seen: Set[Tuple[str,str]] = set()
        for (w1,w2,w3),cnt in tri_raw.items():
            if (w2,w3) in seen: continue
            seen.add((w2,w3))
            t2,t3 = geo.triple_fast(w2), geo.triple_fast(w3)
            w = max(cnt*(t2.rho*t3.rho+0.1)*(1.0+math.cos(t2.theta-t3.theta))*0.5, 1e-6)
            self._adj.setdefault(w2,[]).append((w3,w))
        print(f"[RP-Walk] Adjacency built: {sum(len(v) for v in self._adj.values())} edges")

    def propagate(self):
        if not self._all_toks: return
        visit: Dict[str,float] = {t:0.0 for t in self._all_toks}
        starts = [self._all_toks[i] for i in _reservoir_sample_indices(len(self._all_toks),500)]
        for src in starts:
            cur = src
            for _ in range(self.walk_length):
                visit[cur] = visit.get(cur,0.0)+1.0
                if _rng.random() < self.restart_p: cur=src; continue
                nbrs = self._adj.get(cur,[])
                if not nbrs: cur=src; continue
                total = sum(w for _,w in nbrs); r = _rng.random()*total; cumul=0.0
                for nxt,w in nbrs:
                    cumul+=w
                    if cumul>=r: cur=nxt; break
        maxv = max(visit.values(), default=1.0)+1e-8
        self._potentials = {k:v/maxv for k,v in visit.items()}
        print(f"[RP-Walk] Done. Non-zero: {sum(1 for v in self._potentials.values() if v>0)}/{len(self._potentials)}")

    def potentials_for(self, cands):
        return torch.tensor([self._potentials.get(c,0.0) for c in cands],
                             dtype=torch.float32, device=self.device)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SKETCHED PDN ENGINE  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class SketchedPDNEngine:
    def __init__(self, n_modes=4, n_samples=200, sigma_pdn=0.25,
                 orbit_weight=0.4, regularity_weight=0.5, device=DEVICE, dtype=torch.float32):
        self.n_modes=n_modes; self.n_samples=n_samples; self.sigma_pdn=sigma_pdn
        self.orbit_weight=orbit_weight; self.regularity_weight=regularity_weight
        self.device=device; self.dtype=dtype
        self.n_star=4; self.power_spectrum: Dict[int,float]={}; self._orbit_map: Dict[str,int]={}

    def fit_from_trigrams(self, geo, tri_raw):
        cns = list(range(3, 3+self.n_modes)); power={n:0.0 for n in cns}
        all_tri = list(tri_raw.items()); T = len(all_tri)
        if T==0: self.power_spectrum=power; self.n_star=4; return
        sample_size = min(self.n_samples,T); scale = T/sample_size
        for idx in _reservoir_sample_indices(T, sample_size):
            (w1,w2,w3),cnt = all_tri[idx]
            zs = [complex(geo.triple_fast(t).rho*math.cos(geo.triple_fast(t).theta),
                          geo.triple_fast(t).rho*math.sin(geo.triple_fast(t).theta))
                  for t in (w1,w2,w3)]
            for n in cns:
                padded = zs+[0+0j]*(n-3)
                for k in range(1,n):
                    Fk = sum(padded[j]*complex(math.cos(-2*math.pi*j*k/n),
                                               math.sin(-2*math.pi*j*k/n))
                             for j in range(n))/n
                    power[n] += scale*cnt*abs(Fk)**2
        self.power_spectrum=power; self.n_star=min(power,key=lambda k_:power[k_])
        print(f"[RP-PDN] n*={self.n_star}, spectrum={{{', '.join(f'n{n}:{p:.1f}' for n,p in power.items())}}}")

    def build_orbit_map(self, vocab, geo):
        sector = 2.0*math.pi/max(self.n_star,2)
        for tok in vocab:
            self._orbit_map[tok] = int(geo.triple_fast(tok).theta*2.0/sector)%self.n_star

    def orbit_of(self, token): return self._orbit_map.get(token,0)

    def regularity_scores(self, window_rho, window_theta, c_rho, c_theta):
        n=self.n_star; W=window_rho.shape[0]; C=c_rho.shape[0]
        if W==0: return torch.ones(C,dtype=self.dtype,device=self.device)
        win_re=(window_rho*torch.cos(window_theta)).to(self.dtype)
        win_im=(window_rho*torch.sin(window_theta)).to(self.dtype)
        c_re=(c_rho*torch.cos(c_theta)).to(self.dtype); c_im=(c_rho*torch.sin(c_theta)).to(self.dtype)
        k=n-1; js=torch.arange(W,dtype=self.dtype,device=self.device)
        aw=-2.0*math.pi*js*k/n
        re_p=(win_re*torch.cos(aw)-win_im*torch.sin(aw)).sum()
        im_p=(win_re*torch.sin(aw)+win_im*torch.cos(aw)).sum()
        ac=-2.0*math.pi*W*k/n
        F_re=re_p+c_re*math.cos(ac)-c_im*math.sin(ac)
        F_im=im_p+c_re*math.sin(ac)+c_im*math.cos(ac)
        return torch.exp(-(F_re**2+F_im**2)/(n**2)/(self.sigma_pdn**2+1e-8))

    def orbit_bonus(self, current_orbit, c_theta):
        n=self.n_star; target=(current_orbit+1)%n; sector=2.0*math.pi/max(n,2)
        return torch.cos(2.0*math.pi*(c_theta*2.0/sector-target)/n)*0.5+0.5

    @torch.no_grad()
    def pdn_logit_bonus(self, window_rho, window_theta, c_rho, c_theta, current_orbit):
        reg=self.regularity_scores(window_rho,window_theta,c_rho,c_theta)
        orb=self.orbit_bonus(current_orbit,c_theta)
        def _n(x): std=x.std(); return (x-x.mean())/(std+1e-8) if std.item()>1e-8 else x-x.mean()
        return self.regularity_weight*_n(reg)+self.orbit_weight*_n(orb)

    def theorem_bridge_report(self):
        lines=["╔══════════════════════════════════════════════════════════════╗",
               "║    Thébault → PDN Bridge Report  [RP: Sketched FFT]          ║",
               "╠══════════════════════════════════════════════════════════════╣",
               f"║  RP sketching: {self.n_samples} random trigram samples          ║",
               f"║  Unbiased estimator: scale = T / n_samples                   ║",
               f"║  Dominant symmetry order n* = {self.n_star:<2d}                        ║",
               "║                                                              ║",
               "║  Sketched power spectrum:                                    ║"]
        for n,p in sorted(self.power_spectrum.items()):
            marker=" ← n*" if n==self.n_star else ""
            lines.append(f"║    n={n}: P={p:>10.2f}{marker:<28s}║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — THÉBAULT TOKEN GEOMETRY  (unchanged)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class BolyaiTripleRP:
    rho: float; theta: float; sigma: float

def _hyp_dist(x1, y1, x2, y2, eps=1e-8):
    n1 = min(x1*x1 + y1*y1, 1.0 - eps)
    n2 = min(x2*x2 + y2*y2, 1.0 - eps)
    dx = (x1 - x2)**2 + (y1 - y2)**2
    z = 1.0 + 2.0 * dx / max((1.0 - n1) * (1.0 - n2), eps)
    z = max(z, 1.0 + eps)
    return math.acosh(z)

class BolyaiTokenGeometryRP:
    def __init__(self, device=DEVICE, dtype=torch.float32):
        self.device=device; self.dtype=dtype
        self._vecs: Dict[str,Tuple]={}; self._cache: Dict[str,BolyaiTripleRP]={}
        self._tok2idx: Dict[str,int]={}; self._idx_list: List[str]=[]
        self._rho_t=self._theta_t=self._sigma_t=self._pvec_t=self._feat_t=None
        self.rff: Optional['RandomFourierFeatures']=None

    def register(self, token, freq, index, max_freq, vocab_size):
        f = freq / max(max_freq, 1e-9)
        k = index / max(vocab_size - 1, 1)
        r = 0.92 * math.sqrt(max(f, 1e-9))
        ang = 2.0 * math.pi * k
        x = r * math.cos(ang)
        y = r * math.sin(ang)
        self._vecs[token] = (x, y)
        self._cache.pop(token, None)

    def triple_fast(self, token) -> BolyaiTripleRP:
        if token in self._cache: return self._cache[token]
        x, y = self._vecs.get(token, (0.0, 0.0))
        eu = min(math.sqrt(x*x + y*y), 1.0 - 1e-8)
        hyp_r = 2.0 * math.atanh(eu)
        rho = math.tanh(0.5 * hyp_r)
        theta = math.atan2(y, x) % math.pi
        sigma = 2.0 / max(1.0 - eu*eu, 1e-8)
        t = BolyaiTripleRP(rho, theta, sigma)
        self._cache[token] = t
        return t

    def build_cuda_tensors(self, vocab, rff):
        self.rff=rff; triples=[self.triple_fast(t) for t in vocab]
        self._idx_list=vocab; self._tok2idx={t:i for i,t in enumerate(vocab)}
        self._rho_t   = torch.tensor([t.rho   for t in triples],dtype=self.dtype,device=self.device)
        self._theta_t = torch.tensor([t.theta for t in triples],dtype=self.dtype,device=self.device)
        self._sigma_t = torch.tensor([t.sigma for t in triples],dtype=self.dtype,device=self.device)
        self._pvec_t  = torch.stack([self._rho_t,self._theta_t/math.pi,self._sigma_t, torch.ones_like(self._rho_t)],dim=1)
        with torch.no_grad(): self._feat_t=rff.features(self._rho_t,self._theta_t,self._sigma_t)
        print(f"[RP-Geo-Bolyai] RFF features: {self._feat_t.shape}")

    def _vec(self, token):
        return self._vecs.get(token, (0.0, 0.0))

    def composed_triple(self, t1, t2):
        x1, y1 = self._vec(t1)
        x2, y2 = self._vec(t2)
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        n = math.sqrt(x*x + y*y)
        if n >= 0.98:
            s = 0.98 / max(n, 1e-8)
            x *= s; y *= s
        eu = min(math.sqrt(x*x + y*y), 1.0 - 1e-8)
        hyp_r = 2.0 * math.atanh(eu)
        rho = math.tanh(0.5 * hyp_r)
        theta = math.atan2(y, x) % math.pi
        sigma = 2.0 / max(1.0 - eu*eu, 1e-8)
        return BolyaiTripleRP(rho, theta, sigma)

    def batch_triples(self, idx):
        return self._rho_t[idx], self._theta_t[idx], self._sigma_t[idx]

    def tok_indices(self, toks):
        safe = max(len(self._idx_list) - 1, 0)
        return torch.tensor([min(self._tok2idx.get(t, 0), safe) for t in toks], dtype=torch.long, device=self.device)

    def rff_features_for(self, toks):
        return self._feat_t[self.tok_indices(toks)]

def compute_transitive_triples_rp(geo, cands, w1, w2, device=DEVICE, dtype=torch.float32):
    p1x, p1y = geo._vec(w1)
    p2x, p2y = geo._vec(w2)
    rl, tl, sl = [], [], []
    for c in cands:
        pcx, pcy = geo._vec(c)
        x = .25 * p1x + .5 * p2x + .25 * pcx
        y = .25 * p1y + .5 * p2y + .25 * pcy
        n = math.sqrt(x*x + y*y)
        if n >= 0.98:
            s = 0.98 / max(n, 1e-8)
            x *= s; y *= s
        eu = min(math.sqrt(x*x + y*y), 1.0 - 1e-8)
        hyp_r = 2.0 * math.atanh(eu)
        rho = math.tanh(0.5 * hyp_r)
        theta = math.atan2(y, x) % math.pi
        sigma = 2.0 / max(1.0 - eu*eu, 1e-8)
        rl.append(rho); tl.append(theta); sl.append(sigma)
    return (torch.tensor(rl,dtype=dtype,device=device),
            torch.tensor(tl,dtype=dtype,device=device),
            torch.tensor(sl,dtype=dtype,device=device))

class RPCrossSynapticNeuronSum:
    def __init__(self, rff, syn_weight=0.4, trans_weight=0.6, syn_k=8, device=DEVICE, dtype=torch.float32):
        self.rff=rff; self.syn_weight=syn_weight; self.trans_weight=trans_weight
        self.syn_k=syn_k; self.device=device; self.dtype=dtype
        self._nystrom=NystromSynapticMatrix(rff=rff,n_landmarks=RP_NYSTROM_M,top_k=syn_k,
                                             device=device,dtype=dtype)

    @torch.no_grad()
    def synaptic_sum(self, logits, c_rho, c_theta, c_sigma):
        W=self._nystrom.build(c_rho,c_theta,c_sigma)
        return layer_norm_array(W @ signed_power(logits,p=1.0))

    @torch.no_grad()
    def transitive_bonus(self, c_rho_t,c_theta_t,c_sigma_t, ctx_rho,ctx_theta,ctx_sigma):
        return layer_norm_array(
            self.rff.kernel_scalar(ctx_rho,ctx_theta,ctx_sigma,c_rho_t,c_theta_t,c_sigma_t).clamp(0.0))

    @torch.no_grad()
    def forward(self, logits, c_rho,c_theta,c_sigma, c_rho_t,c_theta_t,c_sigma_t, ctx_rho,ctx_theta,ctx_sigma):
        z_syn=self.synaptic_sum(logits,c_rho,c_theta,c_sigma)
        tb=self.transitive_bonus(c_rho_t,c_theta_t,c_sigma_t,ctx_rho,ctx_theta,ctx_sigma)
        return torch.nan_to_num(logits+self.syn_weight*z_syn+self.trans_weight*tb,
                                nan=0.0,posinf=50.0,neginf=-50.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — RP COMPOSITION LM  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

STOP_WORDS_COG = set(
    "a an and are as at be by for from has have he her him his i in is it its "
    "me my of on or our she so that the their them they this to was we were what "
    "when where which who will with you your if because while".split()
)
COGNITIVE_TOKENS = {f"[{w.upper()}]" for w in STOP_WORDS_COG}
PUNCT_TOKENS     = {",", ".", "!", "?", ";", ":"}

def tokenize(text: str) -> List[str]:
    out=[]
    for w in text.split():
        if w in COGNITIVE_TOKENS or w in PUNCT_TOKENS: out.append(w); continue
        wc="".join(c for c in unicodedata.normalize("NFD",w)
                   if unicodedata.category(c)!="Mn").lower()
        if wc: out.append(f"[{wc.upper()}]" if wc in STOP_WORDS_COG else wc)
    return out

def detokenize(tokens: List[str]) -> str:
    if not tokens: return ""
    res=[]
    for t in tokens:
        if t in PUNCT_TOKENS:
            if res: res[-1]+=t; continue
        if t in COGNITIVE_TOKENS:
            raw=t.strip("[]").lower()
            res.append(raw.capitalize() if not res or res[-1].endswith(('.','!','?')) else raw)
        else:
            res.append(t.capitalize() if not res or res[-1].endswith(('.','!','?')) else t)
    out=" ".join(res).strip()
    return out if out and out[-1] in PUNCT_TOKENS else out+"."

class RPCompositionLM:
    BASAL_K = 1000.5
    def __init__(self, geo, rff, device=DEVICE):
        self.geo=geo; self.rff=rff; self.device=device
        self.cms=CountMinSketch()
        self.raw_freq: Dict[str,float]={}
        self.tri_raw: Dict[Tuple[str,str,str],float]={}
        self.heads: Dict[Tuple[str,str],List[str]]={}
        self.vocab: List[str]=[]; self._tok2idx: Dict[str,int]={}
        self._head_probs: Dict[Tuple[str,str],torch.Tensor]={}

    def ingest(self, tokens):
        for t in tokens:
            self.raw_freq[t]=self.raw_freq.get(t,0)+1.0; self.cms.update(t)
        for i in range(len(tokens)-2):
            w1,w2,w3=tokens[i],tokens[i+1],tokens[i+2]
            self.tri_raw[(w1,w2,w3)]=self.tri_raw.get((w1,w2,w3),0)+1.0
            self.cms.update_triple(w1,w2,w3); self.cms.update_pair(w1,w2)
            if (w1,w2) not in self.heads: self.heads[(w1,w2)]=[]
            if w3 not in self.heads[(w1,w2)]: self.heads[(w1,w2)].append(w3)
        self.vocab=[v for v in self.raw_freq if v not in PUNCT_TOKENS and v not in COGNITIVE_TOKENS]

    def finalise(self):
        self._tok2idx={t:i for i,t in enumerate(self.vocab)}
        V=len(self.vocab)+1
        for (w1,w2),cands in self.heads.items():
            counts=[self.cms.query_triple(w1,w2,c)+1e-4 for c in cands]
            total=sum(counts)
            self._head_probs[(w1,w2)]=torch.tensor(
                [(c+self.BASAL_K)/(total+self.BASAL_K*V) for c in counts],
                dtype=torch.float32,device=self.device)

    def next_dist(self, w1, w2):
        if (w1,w2) in self.heads: return self.heads[(w1,w2)],self._head_probs[(w1,w2)]
        agg={}
        for (_,_,w3),_ in self.tri_raw.items(): agg[w3]=agg.get(w3,0)+self.cms.query(w3)
        cands_all=list(agg.keys())
        sampled=[cands_all[i] for i in _reservoir_sample_indices(len(cands_all),
                                                                   min(RP_RESERVOIR_K*4,len(cands_all)))]
        cands=sampled; total=sum(agg.get(c,1e-4) for c in cands); V=len(self.vocab)+1
        return cands,torch.tensor(
            [(agg.get(c,1e-4)+self.BASAL_K)/(total+self.BASAL_K*V) for c in cands],
            dtype=torch.float32,device=self.device)

    def composition_logit_bonus(self,w1,w2,c_rho,c_sigma):
        C=self.geo.composed_triple(w1,w2)
        ctx_feat=self.rff.features(
            torch.tensor([C.rho],dtype=torch.float32,device=self.device),
            torch.tensor([C.theta],dtype=torch.float32,device=self.device),
            torch.tensor([C.sigma],dtype=torch.float32,device=self.device))
        return (ctx_feat @ self.rff.features(c_rho,torch.zeros_like(c_rho),c_sigma).T).squeeze(0).clamp(0.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — RP MRV FILTER  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class RPMRVFilter:
    def __init__(self, rff, threshold=0.50, mrv_cap_ratio=2.0, device=DEVICE):
        self.rff=rff; self.threshold=threshold; self.mrv_cap_ratio=mrv_cap_ratio
        self.device=device; self._lsh=LSHIndex(feature_dim=3*RP_RFF_DIM,device=device)
        self._vocab_feats: Optional[torch.Tensor]=None; self._v_toks: List[str]=[]

    def prime(self, vocab, geo):
        scan=vocab; self._v_toks=scan
        if geo._feat_t is not None:
            self._vocab_feats=geo._feat_t[geo.tok_indices(scan)]
        else:
            triples=[geo.triple_fast(v) for v in scan]
            rho_t=torch.tensor([t.rho for t in triples],dtype=torch.float32,device=self.device)
            theta_t=torch.tensor([t.theta for t in triples],dtype=torch.float32,device=self.device)
            sigma_t=torch.tensor([t.sigma for t in triples],dtype=torch.float32,device=self.device)
            self._vocab_feats=self.rff.features(rho_t,theta_t,sigma_t)
        fd=self._vocab_feats.shape[1]
        self._lsh=LSHIndex(feature_dim=fd,n_bands=RP_LSH_BANDS,n_rows=RP_LSH_ROWS,device=self.device)
        self._lsh.build(self._vocab_feats,scan)

    def mrv_scores_batched(self, c_rho, c_sigma, kernels=None):
        C=c_rho.shape[0]
        if self._vocab_feats is None: return torch.zeros(C,device=self.device)
        c_feats=self.rff.features(c_rho,torch.zeros_like(c_rho),c_sigma)
        domain=torch.zeros(C,device=self.device)
        for i in range(C): domain[i]=float(len(self._lsh.query_candidates(c_feats[i],max_cands=30)))
        mrv=1.0/(domain+1.0); mean_d=domain.mean()+1e-6
        mrv[domain>self.mrv_cap_ratio*mean_d]*=0.5
        lo,hi=mrv.min(),mrv.max()
        return (mrv-lo)/(hi-lo) if (hi-lo).item()>1e-8 else mrv


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — RP KERNELS  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class RPKernels:
    def __init__(self, rff, lambda_reg=8.0, gamma_side=4.0):
        self.rff=rff; self.lambda_reg=lambda_reg; self.gamma_side=gamma_side

    def k_reg(self,ra,rb):   return torch.exp(-self.lambda_reg*(rb-ra)**2)
    def k_ori(self,ta,tb):   return 0.5*(1.0+torch.cos(tb-ta))
    def k_side(self,sa,sb):  return torch.exp(-self.gamma_side*(sb-sa)**2)

    def all_scores_batched(self,rho_a,theta_a,sigma_a,rho_b,theta_b,sigma_b):
        kr=self.k_reg(torch.tensor(rho_a,device=rho_b.device),rho_b)
        ko=self.k_ori(torch.tensor(theta_a,device=theta_b.device),theta_b)
        ks=self.k_side(torch.tensor(sigma_a,device=sigma_b.device),sigma_b)
        return kr,ko,ks


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — CoT STUBS + REASONING ENGINE  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

STUB_PREMISE="PREMISE"; STUB_ELABORATION="ELABORATION"
STUB_CONTRAST="CONTRAST"; STUB_CONCLUSION="CONCLUSION"
_STUB_SEQUENCE=[STUB_PREMISE,STUB_ELABORATION,STUB_CONTRAST,STUB_CONCLUSION]

@dataclass
class ContextualStub:
    stub_type:str; tokens:List[str]; rho:float; theta:float; sigma:float; weight:float; label:str=""
    def __post_init__(self):
        if not self.label: self.label=f"[{self.stub_type}] {' '.join(self.tokens[:4])}…"
    def as_triple(self): return BolyaiTripleRP(self.rho,self.theta,self.sigma)

@dataclass
class CoTStep:
    hop_index:int; stub:ContextualStub; stub_score:float; pdn_orbit:int

@dataclass
class CoTTrace:
    seed_tokens:List[str]; steps:List[CoTStep]; conclusion:Optional[ContextualStub]
    def render(self):
        lines=["  ── CoT Trace (RP-ANN) ──",f"  Seed: {' '.join(self.seed_tokens[:6])}"]
        return "\n".join(lines)

class RPCoTStubLibrary:
    def __init__(self, rff, rho_threshold=0.20, n_theta_bins=8, min_bin_size=2, device=DEVICE, dtype=torch.float32):
        self.rff=rff; self.rho_threshold=rho_threshold; self.n_theta_bins=n_theta_bins
        self.min_bin_size=min_bin_size; self.device=device; self.dtype=dtype
        self.stubs={t:[] for t in _STUB_SEQUENCE}
        self._stub_list: List[ContextualStub]=[]; self._stub_lsh: Optional[LSHIndex]=None

    def build(self, geo, lm_vocab, raw_freq):
        all_entries=[(tok,geo.triple_fast(tok),raw_freq.get(tok,1.0)) for tok in lm_vocab]
        rhos_sorted=sorted(e[1].rho for e in all_entries)
        thr=min(self.rho_threshold, rhos_sorted[max(0,int(len(rhos_sorted)*0.20))])
        bridges=[(t,tr,f) for t,tr,f in all_entries if tr.rho>=thr]
        if len(bridges)<8: bridges=all_entries
        bridges.sort(key=lambda x:x[1].sigma)
        q=max(1,len(bridges)//4)
        for stub_type,bucket in zip(_STUB_SEQUENCE[:4],[bridges[:q],bridges[q:2*q],bridges[2*q:3*q],bridges[3*q:]]):
            if not bucket: continue
            bw=math.pi/self.n_theta_bins; bins={}
            for tok,tr,freq in bucket:
                bi=min(int(tr.theta/bw),self.n_theta_bins-1)
                bins.setdefault(bi,[]).append((tok,tr,freq))
            for bi,members in bins.items():
                if len(members)<self.min_bin_size: continue
                members.sort(key=lambda x:x[1].rho); mid=max(1,len(members)//2)
                for si,group in enumerate([members[:mid],members[mid:]]):
                    if group: self._make_stub(stub_type,bi,si,group)
        self._rebuild_lsh()
        print(f"[RP-CoT] Built {sum(len(v) for v in self.stubs.values())} stubs")

    def _make_stub(self,stub_type,bi,sub_idx,members):
        toks=[m[0] for m in members]; rhos=[m[1].rho for m in members]
        thetas=[m[1].theta for m in members]; sigmas=[m[1].sigma for m in members]
        sin_m=sum(math.sin(t) for t in thetas)/len(thetas)
        cos_m=sum(math.cos(t) for t in thetas)/len(thetas)
        self.stubs[stub_type].append(ContextualStub(
            stub_type=stub_type,tokens=toks,rho=sum(rhos)/len(rhos),
            theta=math.atan2(sin_m,cos_m)%math.pi,sigma=sum(sigmas)/len(sigmas),
            weight=sum(m[2] for m in members),
            label=f"[{stub_type}|bin{bi}|{'hi' if sub_idx else 'lo'}-ρ] {' '.join(toks[:3])}…"))

    def _rebuild_lsh(self):
        for _stype in _STUB_SEQUENCE:
            self.stubs[_stype].sort(key=lambda s: s.rho)
        self._stub_list=[s for st in _STUB_SEQUENCE for s in self.stubs[st]]
        if not self._stub_list: return
        rt=torch.tensor([s.rho for s in self._stub_list],dtype=torch.float32,device=self.device)
        tt=torch.tensor([s.theta for s in self._stub_list],dtype=torch.float32,device=self.device)
        st=torch.tensor([s.sigma for s in self._stub_list],dtype=torch.float32,device=self.device)
        feats=self.rff.features(rt,tt,st); fd=feats.shape[1]
        self._stub_lsh=LSHIndex(feature_dim=fd,n_bands=RP_LSH_BANDS,n_rows=RP_LSH_ROWS,device=self.device)
        self._stub_lsh.build(feats,[str(i) for i in range(len(self._stub_list))])
        self._stub_feats=feats

    def best_stub(self,stub_type,ctx_rho,ctx_theta,ctx_sigma,kernels=None,pdn_orbit=0,pdn_engine=None):
        cands=self.stubs.get(stub_type,[])
        if not cands: return None
        cr=torch.tensor([s.rho for s in cands],dtype=torch.float32,device=self.device)
        ct=torch.tensor([s.theta for s in cands],dtype=torch.float32,device=self.device)
        cs=torch.tensor([s.sigma for s in cands],dtype=torch.float32,device=self.device)
        scores=self.rff.kernel_scalar(ctx_rho,ctx_theta,ctx_sigma,cr,ct,cs).clamp(0.0)
        if pdn_engine is not None: scores=scores+0.3*pdn_engine.orbit_bonus(pdn_orbit,ct)
        return cands[int(scores.argmax().item())]

    @torch.no_grad()
    def stub_kernel(self,stub,c_rho,c_theta,c_sigma,kernels=None):
        return self.rff.kernel_scalar(stub.rho,stub.theta,stub.sigma,c_rho,c_theta,c_sigma).clamp(0.0)

class RPCoTReasoningEngine:
    def __init__(self,stub_library,kernels,pdn_engine,n_hops=3,tokens_per_hop=8,
                 stub_logit_scale=0.9,device=DEVICE,dtype=torch.float32):
        self.stubs=stub_library; self.kernels=kernels; self.pdn=pdn_engine
        self.n_hops=n_hops; self.tokens_per_hop=tokens_per_hop
        self.stub_logit_scale=stub_logit_scale; self.device=device; self.dtype=dtype
        self._chain: List[CoTStep]=[]; self._conclusion_stub: Optional[ContextualStub]=None
        self._hop_ptr=0; self._tok_since_hop=0; self._traces: List[CoTTrace]=[]

    def begin_sentence(self):
        self._chain=[]; self._conclusion_stub=None; self._hop_ptr=0; self._tok_since_hop=0

    def plan_chain(self,seed_tokens,geo,pdn_orbit=0) -> CoTTrace:
        clean=[t for t in seed_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if clean:
            triples=[geo.triple_fast(t) for t in clean]
            ctx_rho=sum(t.rho for t in triples)/len(triples)
            ctx_sigma=sum(t.sigma for t in triples)/len(triples)
            sin_m=sum(math.sin(t.theta) for t in triples)/len(triples)
            cos_m=sum(math.cos(t.theta) for t in triples)/len(triples)
            ctx_theta=math.atan2(sin_m,cos_m)%math.pi
        else: ctx_rho,ctx_theta,ctx_sigma=0.5,math.pi/4,0.5
        self._chain=[]; self._conclusion_stub=None
        _seq = _STUB_SEQUENCE
        if self.n_hops >= len(_seq):
            hops = list(_seq) + [_seq[-2]] * (self.n_hops - len(_seq))
        else:
            _step = (len(_seq) - 1) / max(self.n_hops - 1, 1)
            hops = [_seq[min(int(round(i * _step)), len(_seq)-1)]
                    for i in range(self.n_hops)]
        for i,stype in enumerate(hops[:self.n_hops]):
            stub=self.stubs.best_stub(stype,ctx_rho,ctx_theta,ctx_sigma,self.kernels,
                                       pdn_orbit=(pdn_orbit+i)%self.pdn.n_star,pdn_engine=self.pdn)
            if stub is None: continue
            k=self.stubs.stub_kernel(stub,
                torch.tensor([ctx_rho],device=self.device),
                torch.tensor([ctx_theta],device=self.device),
                torch.tensor([ctx_sigma],device=self.device)).item()
            self._chain.append(CoTStep(i,stub,k,(pdn_orbit+i)%self.pdn.n_star))
            ctx_rho,ctx_theta,ctx_sigma=stub.rho,stub.theta,stub.sigma
        self._conclusion_stub=self.stubs.best_stub(STUB_CONCLUSION,ctx_rho,ctx_theta,ctx_sigma,
                                                    self.kernels,pdn_orbit=(pdn_orbit+self.n_hops)%self.pdn.n_star,
                                                    pdn_engine=self.pdn)
        trace=CoTTrace(clean,list(self._chain),self._conclusion_stub)
        self._traces.append(trace); return trace

    @torch.no_grad()
    def active_bonus(self,c_rho,c_theta,c_sigma,token_position,total_tokens):
        C=c_rho.shape[0]
        if self._tok_since_hop>=self.tokens_per_hop and self._hop_ptr<len(self._chain)-1:
            self._hop_ptr+=1; self._tok_since_hop=0
        self._tok_since_hop+=1
        frac=token_position/max(total_tokens-1,1)
        if frac>=0.80 and self._conclusion_stub is not None: active=self._conclusion_stub
        elif self._hop_ptr<len(self._chain): active=self._chain[self._hop_ptr].stub
        else: return torch.zeros(C,dtype=self.dtype,device=self.device)
        raw=self.stubs.stub_kernel(active,c_rho,c_theta,c_sigma,self.kernels)
        std=raw.std()
        if std.item()>1e-8: raw=(raw-raw.mean())/std
        return raw*self.stub_logit_scale

    def all_traces_text(self,max_traces=8):
        if not self._traces: return "  (no traces yet)"
        return "\n".join(f"\nSentence {i+1}:\n{tr.render()}"
                         for i,tr in enumerate(self._traces[-max_traces:]))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — ANCILLARY SUBSYSTEMS  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class BolyaiConjugateOrbit:
    def score(self,anchor,cand_theta,cand_sigma,gamma_side=4.0):
        return torch.exp(-gamma_side*(cand_sigma-anchor.sigma)**2) * \
               torch.cos(cand_theta+anchor.theta-math.pi/2)**2

class synthetic_reasonMandateProcessor:
    def __init__(self):
        self.AIEthics=["do not harm any human","do not harm myself","do not make weapons"]
        self.AIMandates=["end poverty","cure disease","improve standard of living","learn"]
        self.mandate_vocabulary={"poverty":"end","disease":"cure","standard":"improve",
                                  "living":"improve","learn":"explore","human":"protect",
                                  "weapons":"avoid","harm":"prevent"}
    def subsynthetic_reason_concept_enrichment(self,w_ctx,cands,device):
        enrichment=torch.zeros(len(cands),device=device)
        trigger=next((self.mandate_vocabulary[k] for k in self.mandate_vocabulary
                      if k in w_ctx.lower()),None)
        if trigger:
            for i,c in enumerate(cands):
                if trigger in c.lower(): enrichment[i]+=5.0
                elif c.lower() in self.AIEthics: enrichment[i]+=10.0
        return enrichment

VEC_DIM=4
class ChunkedSumEngine:
    def __init__(self,window_size=16,n_chunks=4,device=DEVICE,dtype=torch.float32):
        self.window_size=window_size; self.n_chunks=n_chunks; self.device=device; self.dtype=dtype
        self._buf=torch.zeros(window_size,VEC_DIM,dtype=dtype,device=device); self._ptr=0; self._count=0
    def reset(self): self._buf.zero_(); self._ptr=0; self._count=0
    def push(self,triple,pos_norm):
        self._buf[self._ptr]=torch.tensor([triple.rho,triple.theta/math.pi,triple.sigma,pos_norm],
                                           dtype=self.dtype,device=self.device)
        self._ptr=(self._ptr+1)%self.window_size; self._count=min(self._count+1,self.window_size)
    def chunk_signature(self):
        if self._count==0: return torch.zeros(self.n_chunks*VEC_DIM,dtype=self.dtype,device=self.device)
        w=self._buf[:self._count] if self._count<self.window_size \
          else torch.cat([self._buf[self._ptr:],self._buf[:self._ptr]])
        pad=(-w.shape[0])%self.n_chunks
        if pad>0: w=torch.cat([w,torch.zeros(pad,VEC_DIM,dtype=self.dtype,device=self.device)])
        cl=w.shape[0]//self.n_chunks
        return w.view(self.n_chunks,cl,VEC_DIM).sum(dim=1).flatten()
    def chunk_bonus(self,c_pvec,scale=1.0):
        sig=self.chunk_signature(); raw=c_pvec.repeat(1,self.n_chunks)@sig
        std=raw.std()
        if std.item()>1e-8: raw=(raw-raw.mean())/std
        return raw*scale
    def window_rho_theta(self):
        if self._count==0:
            empty=torch.zeros(0,dtype=self.dtype,device=self.device); return empty,empty
        w=self._buf[:self._count] if self._count<self.window_size \
          else torch.cat([self._buf[self._ptr:],self._buf[:self._ptr]])
        return w[:,0],w[:,1]*math.pi

@dataclass
class SentenceVector:
    tokens:List[str]; rho_t:torch.Tensor; sigma_t:torch.Tensor; text:str

class IsomorphicSyntaxStacker:
    def __init__(self,rff,top_k=3,max_stored=64,device=DEVICE,dtype=torch.float32):
        self.rff=rff; self.top_k=top_k; self.max_stored=max_stored
        self.device=device; self.dtype=dtype; self.store: List[SentenceVector]=[]
    def add(self,tokens,geo,text):
        clean=[t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean: return
        rhos=torch.tensor([geo.triple_fast(t).rho for t in clean],dtype=self.dtype,device=self.device)
        sigs=torch.tensor([geo.triple_fast(t).sigma for t in clean],dtype=self.dtype,device=self.device)
        self.store.append(SentenceVector(clean,rhos,sigs,text))
        if len(self.store)>self.max_stored: self.store.pop(0)
    def ranked_anchors(self,current_tokens,geo,kernels):
        if not self.store or not current_tokens: return []
        clean=[t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean: return []
        cr=torch.tensor([geo.triple_fast(t).rho for t in clean],dtype=self.dtype,device=self.device)
        cs=torch.tensor([geo.triple_fast(t).sigma for t in clean],dtype=self.dtype,device=self.device)
        L=cr.shape[0]; N=len(self.store)
        sims=torch.zeros(N,device=self.device)
        for i,sv in enumerate(self.store):
            l=min(L,sv.rho_t.shape[0])
            sims[i]=(torch.exp(-kernels.lambda_reg*(sv.rho_t[:l]-cr[:l])**2)*
                     torch.exp(-kernels.gamma_side*(sv.sigma_t[:l]-cs[:l])**2)).mean()
        topk=torch.topk(sims,min(self.top_k,N))
        return [(topk.values[i].item(),self.store[topk.indices[i].item()]) for i in range(topk.values.shape[0])]
    def syntax_echo_bonus(self,c_rho,c_sigma,current_tokens,geo,kernels,echo_weight=0.5):
        anchors=self.ranked_anchors(current_tokens,geo,kernels)
        if not anchors: return torch.zeros(c_rho.shape[0],device=self.device)
        pos=len([t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS])
        bonuses=torch.zeros(c_rho.shape[0],dtype=self.dtype,device=self.device)
        for sim_score,anc in anchors:
            if pos<anc.rho_t.shape[0]:
                kr=torch.exp(-kernels.lambda_reg*(c_rho-anc.rho_t[pos].item())**2)
                ks=torch.exp(-kernels.gamma_side*(c_sigma-anc.sigma_t[pos].item())**2)
                bonuses+=sim_score*kr*ks
        if bonuses.shape[0] > 1:
            std = bonuses.std(unbiased=False)
            if std.item() > 1e-8:
                bonuses = (bonuses - bonuses.mean()) / std
        return bonuses * echo_weight

class GeometricTempScaler:
    def __init__(self,lambda_temp=1.0): self.lambda_temp=lambda_temp
    def scale(self,logits,temp,c_rho=None):
        safe=logits.clamp(-50.0,50.0)
        if c_rho is None or temp<1e-6: return safe/max(temp,0.1)
        mu_rho=c_rho.mean()
        exp=(-self.lambda_temp*(c_rho-mu_rho)**2/max(temp,0.1)).clamp(min=-10.0)
        return safe*torch.exp(exp)

class DNNArrayPipeline:
    def __init__(self,device=DEVICE,dtype=torch.float32):
        self.device=device; self.dtype=dtype; self.temp_scaler=GeometricTempScaler(lambda_temp=1.0)
    def rho_weights(self,c_rho):
        mu=c_rho.mean(); std=c_rho.std()+1e-8
        return 1.0+0.5*((c_rho-mu)/std).clamp(-2.5,2.5)
    def theta_weights(self,c_theta): return 0.5*(1.0+torch.cos(c_theta))
    def sigma_weights(self,c_sigma): return 0.7+0.3*c_sigma/(c_sigma.max()+1e-8)
    @torch.no_grad()
    def forward(self,logits,c_rho,c_theta,c_sigma,temp=1.4):
        ls=self.temp_scaler.scale(logits,temp,c_rho)
        z1=signed_power(ls*self.rho_weights(c_rho),p=2.0)
        z2=signed_power(z1*self.theta_weights(c_theta),p=1.5)
        z3=signed_power(z2*self.sigma_weights(c_sigma)+z1*0.3,p=1.0)
        return l1_simplex_project(z3)
    @torch.no_grad()
    def log_forward(self,logits,c_rho,c_theta,c_sigma,temp=1.4):
        return (self.forward(logits,c_rho,c_theta,c_sigma,temp)+1e-12).log()

class LocaleTransitRemission:
    def __init__(self,transit_tolerance=0.15,remission_rate=0.85):
        self.transit_tolerance=transit_tolerance; self.remission_rate=remission_rate
    def apply_remission(self,w1_rho,w2_rho,c_rho):
        delta=torch.abs(w1_rho+w2_rho)/2.0-c_rho
        err=smooth_power_relu(delta-self.transit_tolerance)
        mask=(err>1e-6).float()
        return torch.where(mask>0,torch.exp(-self.remission_rate*err),torch.ones_like(c_rho))

class ContingentExtringentProbability:
    def __init__(self,coupling_factor=0.5):
        self.coupling_factor=coupling_factor; self.intermediate_entropy=1.0
        self.intermediate_max_prob=1.0; self.dnn=DNNArrayPipeline()
    def govern_next_probs(self,logits,c_rho=None,c_theta=None,c_sigma=None):
        dyn_temp=1.0+self.coupling_factor*(1.0-self.intermediate_max_prob)
        if c_rho is not None and c_theta is not None and c_sigma is not None:
            gov=self.dnn.temp_scaler.scale(logits,dyn_temp,c_rho)
        else: gov=logits/max(dyn_temp,1e-6)
        p=l1_simplex_project(gov)
        self.intermediate_entropy=-(p*(p+1e-9).log()).sum().item()
        self.intermediate_max_prob=p.max().item()
        return gov

@dataclass
class TokenStepTrace:
    step:int; chosen:str; p_instr:float; p_walk:float; p_and:float; and_weight:float
    source:str; syn_norm:float=0.0; trans_norm:float=0.0; rp_nystrom_rank:int=0
    ooi_size:int=0; repulsion_mean:float=0.0
    def render(self):
        return (f"  {self.step:03d} {self.chosen:<14s} Pand={self.p_and:.4f} ({self.and_weight:.2f}) "
                f"[{self.source:>7s}] zsyn={self.syn_norm:.3f} trans={self.trans_norm:.3f} "
                f"nystrom={self.rp_nystrom_rank} ooi={self.ooi_size} rep={self.repulsion_mean:.3f}")



# ════════════════════════════════════════════════════════════════════════════
# SECTION 14.5 — PROPOSITIONAL SURJECTION ENGINE (NEW)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PropositionalStatement:
    subj: str
    pred: str
    obj: str
    confidence: float

    def render(self):
        return f"  ⟨ {self.subj} → {self.pred} → {self.obj} ⟩  (conf: {self.confidence:.3f})"

class PropositionalSurjectionEngine:
    """
    Performs surjective mapping of the dense generative manifold (token sequences)
    into a small, discrete set of propositional statements using geometric properties.
    """
    def __init__(self, geo, rho_threshold=0.20):
        self.geo = geo
        self.rho_threshold = rho_threshold

    def surject_sentence(self, tokens: List[str]) -> List[PropositionalStatement]:
        clean_toks = [t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if len(clean_toks) < 3:
            return []

        triples = [self.geo.triple_fast(t) for t in clean_toks]
        statements = []

        for i in range(len(clean_toks) - 2):
            w1, w2, w3 = clean_toks[i], clean_toks[i+1], clean_toks[i+2]
            t1, t2, t3 = triples[i], triples[i+1], triples[i+2]

            # Map high-density entities via intermediate relation
            if t1.rho > self.rho_threshold and t3.rho > self.rho_threshold:
                conf = (t1.rho * t2.sigma * t3.rho) ** (1/3)
                stmt = PropositionalStatement(
                    subj=w1.upper(), pred=w2.lower(), obj=w3.upper(), confidence=conf
                )
                statements.append(stmt)

        statements.sort(key=lambda x: x.confidence, reverse=True)

        # Deduplicate to minimal cover
        seen_entities = set()
        minimal_cover = []
        for stmt in statements:
            if stmt.subj not in seen_entities or stmt.obj not in seen_entities:
                minimal_cover.append(stmt)
                seen_entities.add(stmt.subj)
                seen_entities.add(stmt.obj)

        return minimal_cover[:3]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — RP INSTRUCTION DISTRIBUTION  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class RPInstructionDistribution:
    def __init__(self,geo,kernels,lm,device=DEVICE,dtype=torch.float32,
                 semantic_radius=2.0,recency_decay=0.7,context_bonus=0.15,centroid_weight=0.4):
        self.geo=geo; self.kernels=kernels; self.lm=lm; self.device=device; self.dtype=dtype
        self.semantic_radius=semantic_radius; self.recency_decay=recency_decay
        self.context_bonus=context_bonus; self.centroid_weight=centroid_weight
        self.instr_toks=[]; self.instr_freq={}; self.instr_centroid=None; self.base_dist_t=None

    def set_instruction(self,instruction_text):
        raw=tokenize(instruction_text)
        self.instr_toks=[t for t in raw if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not self.instr_toks: self.base_dist_t=None; self.instr_centroid=None; return
        freq={}; N=len(self.instr_toks)
        for pos,tok in enumerate(self.instr_toks):
            freq[tok]=freq.get(tok,0)+self.recency_decay**(N-1-pos)
        self.instr_freq=freq
        triples=[self.geo.triple_fast(t) for t in self.instr_toks]
        ctx_rho=sum(t.rho for t in triples)/len(triples)
        ctx_sigma=sum(t.sigma for t in triples)/len(triples)
        sin_m=sum(math.sin(t.theta) for t in triples)/len(triples)
        cos_m=sum(math.cos(t.theta) for t in triples)/len(triples)
        self.instr_centroid=BolyaiTripleRP(ctx_rho,math.atan2(sin_m,cos_m)%math.pi,ctx_sigma)
        V=len(self.lm.vocab); base=torch.zeros(V,dtype=self.dtype,device=self.device)
        for tok,w in freq.items():
            idx=self.lm._tok2idx.get(tok)
            if idx is not None: base[idx]+=w
        if self.geo._rho_t is not None:
            for tok,w in freq.items():
                tr=self.geo.triple_fast(tok)
                scores=self.kernels.rff.kernel_scalar(tr.rho,tr.theta,tr.sigma,
                                                       self.geo._rho_t,self.geo._theta_t,self.geo._sigma_t)
                base+=w*scores.clamp(0.0)
        base=base.clamp(min=0.0); total=base.sum()
        self.base_dist_t=base/total if total.item()>1e-8 else torch.ones(V,dtype=self.dtype,device=self.device)/V

    @torch.no_grad()
    def distribution(self,cands,gen_tokens,lm_tok2idx):
        C=len(cands)
        if C==0 or self.base_dist_t is None: return torch.ones(C,dtype=self.dtype,device=self.device)/max(C,1)
        cand_idx=torch.tensor([lm_tok2idx.get(c,0) for c in cands],dtype=torch.long,device=self.device)
        base_probs=self.base_dist_t[cand_idx]
        instr_set=set(self.instr_toks)
        ctx_bonus=torch.tensor([self.context_bonus if c in instr_set else 0.0 for c in cands],
                                dtype=self.dtype,device=self.device)
        raw=(base_probs+ctx_bonus).clamp(min=1e-12)
        return raw/raw.sum()



# ════════════════════════════════════════════════════════════════════════════
# SECTION 14.5 — PROPOSITIONAL SURJECTION ENGINE (NEW)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PropositionalStatement:
    subj: str
    pred: str
    obj: str
    confidence: float

    def render(self):
        return f"  ⟨ {self.subj} → {self.pred} → {self.obj} ⟩  (conf: {self.confidence:.3f})"

class PropositionalSurjectionEngine:
    """
    Performs surjective mapping of the dense generative manifold (token sequences)
    into a small, discrete set of propositional statements using geometric properties.
    """
    def __init__(self, geo, rho_threshold=0.20):
        self.geo = geo
        self.rho_threshold = rho_threshold

    def surject_sentence(self, tokens: List[str]) -> List[PropositionalStatement]:
        clean_toks = [t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if len(clean_toks) < 3:
            return []

        triples = [self.geo.triple_fast(t) for t in clean_toks]
        statements = []

        for i in range(len(clean_toks) - 2):
            w1, w2, w3 = clean_toks[i], clean_toks[i+1], clean_toks[i+2]
            t1, t2, t3 = triples[i], triples[i+1], triples[i+2]

            # Map high-density entities via intermediate relation
            if t1.rho > self.rho_threshold and t3.rho > self.rho_threshold:
                conf = (t1.rho * t2.sigma * t3.rho) ** (1/3)
                stmt = PropositionalStatement(
                    subj=w1.upper(), pred=w2.lower(), obj=w3.upper(), confidence=conf
                )
                statements.append(stmt)

        statements.sort(key=lambda x: x.confidence, reverse=True)

        # Deduplicate to minimal cover
        seen_entities = set()
        minimal_cover = []
        for stmt in statements:
            if stmt.subj not in seen_entities or stmt.obj not in seen_entities:
                minimal_cover.append(stmt)
                seen_entities.add(stmt.subj)
                seen_entities.add(stmt.obj)

        return minimal_cover[:3]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15.5 — FITTED LINE REGRESSION  (extended to 19 features)
# ════════════════════════════════════════════════════════════════════════════

class FittedLineRegression(nn.Module):
    """
    Single 'fitted line' (rank-1) over ALL 19 RP+ANISO bonus signals.

    Features 1-17: identical to V18-RP
    Feature 18: ooi_affinity      — mean aniso kernel score vs OOI set
    Feature 19: inter_repulsion   — DPP-lite diversity penalty (negated for fitting)
    """
    FEATURE_NAMES = [
        "k_reg","k_ori","k_side","orbit","potential","mrv",
        "chunk","echo","pdn","cot","instr","syn_norm","trans_norm",
        "rho_mean","sigma_mean","composition","sorted_impulse",
        "ooi_affinity",        # NEW
        "inter_repulsion_neg", # NEW (stored negated so + weight = more repulsion)
    ]
    FEATURE_DIM = 19

    def __init__(self, feature_dim: int = FEATURE_DIM, rank: int = 1):
        super().__init__()
        self.feature_dim  = feature_dim
        self.W            = nn.Parameter(torch.randn(feature_dim, rank) * 0.05)
        self.b            = nn.Parameter(torch.randn(rank) * 0.05)
        self.feature_scale= nn.Parameter(torch.ones(feature_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features * self.feature_scale.unsqueeze(0)
        return torch.tanh(torch.matmul(x, self.W).sum(-1) + self.b.sum()) * 2.0

    def loss(self, features: torch.Tensor, gold_indices: torch.Tensor) -> torch.Tensor:
        B, C, D = features.shape
        deltas  = self(features.view(B*C, D)).view(B, C)
        probs   = F.softmax(deltas, dim=-1)
        targets = F.one_hot(gold_indices, C).float()
        return F.binary_cross_entropy(probs, targets)

    def feature_report(self) -> str:
        lines = ["  Fitted Line Feature Weights (V18-RP-ANISO):"]
        w = (self.W.squeeze(-1) * self.feature_scale).detach().cpu()
        for name, wi in zip(self.FEATURE_NAMES, w):
            bar = "█" * int(abs(wi.item())*10)
            sign = "+" if wi.item() >= 0 else "-"
            lines.append(f"    {name:<18s} {sign}{abs(wi.item()):.4f}  {bar}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 16 — RP WALKER  (extended with ANISO signals)
# ════════════════════════════════════════════════════════════════════════════

class RPWalker:
    def __init__(self, geo, kernels, lm, orbit, rw_graph, synth, mrv_filter,
                 chunk_engine, iso_stacker, pdn_engine, cot_engine, instr_dist, rff,
                 device=DEVICE, syn_weight=0.4, trans_weight=0.6, syn_k=8,
                 aniso_ooi_weight: float = ANISO_OOI_W,
                 aniso_repulsion_weight: float = ANISO_REPULSION_W):
        self.geo=geo; self.kernels=kernels; self.lm=lm; self.orbit=orbit
        self.rw_graph=rw_graph; self.synth=synth; self.mrv=mrv_filter
        self.chunk_engine=chunk_engine; self.iso_stacker=iso_stacker
        self.pdn=pdn_engine; self.cot=cot_engine; self.instr_dist=instr_dist
        self.rff=rff; self.device=device
        self.aniso_ooi_weight      = aniso_ooi_weight
        self.aniso_repulsion_weight= aniso_repulsion_weight

        self._current_isomorphic_pairs=[]; self._cur_sent_toks: List[str]=[]
        self._cur_orbit=0; self._tok_pos=0; self._step_traces: List[TokenStepTrace]=[]
        self._total_tokens=40
        self._remission=LocaleTransitRemission()
        self._contingent=ContingentExtringentProbability()
        self._dnn=DNNArrayPipeline(device=device)
        self._csns=RPCrossSynapticNeuronSum(rff=rff,syn_weight=syn_weight,
                                             trans_weight=trans_weight,syn_k=syn_k,device=device)
        self._csns_syn_norms: List[float]=[]; self._csns_trans_norms: List[float]=[]

        # ── ANISO & Surjection subsystems ────────────────────────────────
        self.surjector = PropositionalSurjectionEngine(geo)
        # ── ANISO subsystems ─────────────────────────────────────────────
        self._aniso_kernel = AnisoDirKernel(device=device)
        self._ooi_tracker  = SentenceOOITracker(self._aniso_kernel, device=device)

        # Pending state for step-trace recording
        self._pending_instr_probs=None; self._pending_walk_logits=None
        self._pending_crho=self._pending_ctheta=self._pending_csigma=None
        self._pending_syn_norm=self._pending_trans_norm=0.0
        self._pending_nystrom_rank=RP_NYSTROM_M
        self._pending_ooi_size=0; self._pending_repulsion_mean=0.0

        # Fitted line (None until train_fitted_line() is called)
        self.fitted_model: Optional[FittedLineRegression] = None
        self._fl_replay_buf: List[Tuple[torch.Tensor, int]] = []

    def begin_sentence(self, seed_tokens=None, total_tokens=40) -> CoTTrace:
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()
        self._cur_orbit=0; self._tok_pos=0; self._total_tokens=total_tokens
        # Reset OOI tracker for the new sentence
        self._ooi_tracker.reset()
        seeds = seed_tokens or []
        self.cot.begin_sentence()
        return self.cot.plan_chain(seeds, self.geo, pdn_orbit=self._cur_orbit)

    # ── ANISO feature extraction helper ─────────────────────────────────
    def _aniso_features(
        self,
        c_rho: torch.Tensor, c_theta: torch.Tensor, c_sigma: torch.Tensor,
        pre_softmax_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the two ANISO-specific features for the candidate set:
          ooi_affinity   (C,) in [0,1]
          inter_repulsion(C,) in [0,1]  (before negation)

        pre_softmax_probs: current soft probability estimate used for
        DPP-lite repulsion weighting.  Any distribution will do; we use the
        LM base probs so the repulsion is well-defined even on the first call.
        """
        ooi_aff = self._ooi_tracker.ooi_affinity(c_rho, c_theta, c_sigma)
        repulsion = self._ooi_tracker.inter_candidate_repulsion(
            c_rho, c_theta, c_sigma, pre_softmax_probs)
        return ooi_aff, repulsion

    # ── full feature vector: 17 (V18-RP) + 2 (ANISO) = 19 ───────────────
    def _extract_features(
        self, C,
        k_reg, k_ori, k_side,
        orbit_scores, pot_bonus, mrv_scores,
        chunk_bonus, echo_bonus,
        pdn_bonus, cot_bonus,
        instr_probs, syn_norm_vec, trans_norm_vec,
        c_rho, c_sigma, comp_bonus,
        sorted_impulse,
        ooi_affinity,          # NEW feature 18
        inter_repulsion,       # NEW feature 19 (stored negated)
    ) -> torch.Tensor:
        def _safe(t):
            if t.shape[0] != C:
                t = torch.zeros(C, device=self.device)
            return t.clamp(-10, 10)
        return torch.stack([
            _safe(k_reg), _safe(k_ori), _safe(k_side),
            _safe(orbit_scores), _safe(pot_bonus), _safe(mrv_scores),
            _safe(chunk_bonus), _safe(echo_bonus),
            _safe(pdn_bonus), _safe(cot_bonus),
            _safe(instr_probs), _safe(syn_norm_vec), _safe(trans_norm_vec),
            _safe(c_rho), _safe(c_sigma), _safe(comp_bonus),
            _safe(sorted_impulse),
            _safe(ooi_affinity),
            _safe(-inter_repulsion),   # negated: high repulsion → negative feature
        ], dim=-1)   # (C, 19)

    @torch.no_grad()
    def walk_probs(self, w1, w2, temp=1.4,
                   alpha_reg=1.2, beta_ori=0.8, delta_side=1.0, gamma_orbit=0.6,
                   psi_pot=4.35, zeta_mrv=10.9, eta_chunk=40.7, xi_echo=80.6,
                   pdn_weight=10.8, cot_weight=51.0, and_weight=210.5,
                   cands=None, base_probs=None):

        if cands is None or base_probs is None:
            cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands: return cands, base_probs

        C = len(cands)
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
        k_reg, k_ori, k_side = self.kernels.all_scores_batched(
            ctx.rho, ctx.theta, ctx.sigma, c_rho, c_theta, c_sigma)
        orbit_scores  = self.orbit.score(ctx, c_theta, c_sigma, self.kernels.gamma_side)
        pot_bonus     = self.rw_graph.potentials_for(cands) * psi_pot
        mrv_scores    = self.mrv.mrv_scores_batched(c_rho, c_sigma, self.kernels) * zeta_mrv
        window_rho, window_theta = self.chunk_engine.window_rho_theta()
        chunk_bonus   = self.chunk_engine.chunk_bonus(c_pvec, scale=eta_chunk)
        echo_bonus    = self.iso_stacker.syntax_echo_bonus(
            c_rho, c_sigma, self._cur_sent_toks, self.geo, self.kernels, echo_weight=xi_echo)
        pdn_bonus     = self.pdn.pdn_logit_bonus(
            window_rho, window_theta, c_rho, c_theta, self._cur_orbit) * pdn_weight
        cot_bonus     = self.cot.active_bonus(
            c_rho, c_theta, c_sigma, self._tok_pos, self._total_tokens) * cot_weight
        mandate_boost = self.synth.subsynthetic_reason_concept_enrichment(
            w2, cands, self.device)
        punct_bias    = torch.tensor(
            [2.0 if c in PUNCT_TOKENS else 0.0 for c in cands],
            dtype=torch.float32, device=self.device)
        punct_penalty = torch.tensor(
            [-3.0 if (c in PUNCT_TOKENS and len(self._cur_sent_toks) < 6) else 0.0
             for c in cands], dtype=torch.float32, device=self.device)
        base_logits   = torch.log(base_probs.clamp(min=1e-12))

        # CSNS forward
        c_rho_t, c_theta_t, c_sigma_t = compute_transitive_triples_rp(
            self.geo, cands, w1, w2, device=self.device)
        governed = self._contingent.govern_next_probs(base_logits, c_rho, c_theta, c_sigma)
        logits_enriched = self._csns.forward(
            governed, c_rho, c_theta, c_sigma,
            c_rho_t, c_theta_t, c_sigma_t,
            ctx.rho, ctx.theta, ctx.sigma)

        z_syn_raw  = self._csns.synaptic_sum(governed, c_rho, c_theta, c_sigma)
        t_bon_raw  = self._csns.transitive_bonus(
            c_rho_t, c_theta_t, c_sigma_t, ctx.rho, ctx.theta, ctx.sigma)
        syn_norm   = z_syn_raw.norm().item()
        trans_norm = t_bon_raw.norm().item()
        self._csns_syn_norms.append(syn_norm)
        self._csns_trans_norms.append(trans_norm)
        syn_norm_vec   = z_syn_raw.clamp(-5, 5)
        trans_norm_vec = t_bon_raw.clamp(-5, 5)
        comp_bonus     = self.lm.composition_logit_bonus(w1, w2, c_rho, c_sigma)

        # Sorted impulse (rho-rank directional, unchanged from V18-RP)
        _rho_rank    = torch.argsort(torch.argsort(c_rho)).float()
        _rank_norm   = _rho_rank / max(float(C - 1), 1.0)
        _hop_frac    = self._tok_pos / max(self._total_tokens - 1, 1)
        _impulse_dir = math.cos(math.pi * _hop_frac)
        sorted_impulse = layer_norm_array(_rank_norm * _impulse_dir)

        # Instruction distribution
        if and_weight > 0.0 and self.instr_dist.base_dist_t is not None:
            p_instr = self.instr_dist.distribution(cands, self._cur_sent_toks, self.lm._tok2idx)
        else:
            p_instr = torch.ones(C, dtype=torch.float32, device=self.device) / C

        # ════════════════════════════════════════════════════════════════
        # ANISO SIGNALS  ← NEW
        # Use LM base probs as the probability weight for DPP-lite repulsion.
        # This is evaluated BEFORE the fitted-line / hand-tuned correction so
        # that similar candidates are repelled based on the prior, not the
        # post-hoc adjusted distribution (avoids circular dependency).
        # ════════════════════════════════════════════════════════════════
        ooi_affinity, inter_repulsion = self._aniso_features(
            c_rho, c_theta, c_sigma, base_probs)

        # Normalise for stable logit addition
        def _znorm(t: torch.Tensor) -> torch.Tensor:
            s = t.std()
            return (t - t.mean()) / (s + 1e-8) if s.item() > 1e-8 else t - t.mean()

        ooi_aff_norm = _znorm(ooi_affinity)
        rep_norm     = _znorm(inter_repulsion)

        self._pending_ooi_size       = self._ooi_tracker.size
        self._pending_repulsion_mean = inter_repulsion.mean().item()

        # Build full 19-d feature tensor
        features = self._extract_features(
            C, k_reg, k_ori, k_side, orbit_scores, pot_bonus, mrv_scores,
            chunk_bonus, echo_bonus, pdn_bonus, cot_bonus,
            p_instr, syn_norm_vec, trans_norm_vec, c_rho, c_sigma, comp_bonus,
            sorted_impulse,
            ooi_affinity,
            inter_repulsion)

        # ── FITTED LINE or hand-tuned fallback ──────────────────────────
        if self.fitted_model is not None:
            _fd = self.fitted_model.W.shape[0]
            if features.shape[1] != _fd:
                if features.shape[1] < _fd:
                    _pad = torch.zeros(C, _fd - features.shape[1], device=self.device)
                    features = torch.cat([features, _pad], dim=1)
                else:
                    features = features[:, :_fd]
            delta      = self.fitted_model(features)
            raw_logits = logits_enriched + delta
        else:
            # Hand-tuned fallback — ANISO terms appended after V18-RP sum
            raw_logits = (logits_enriched
                          + alpha_reg   * k_reg
                          + beta_ori    * k_ori
                          + delta_side  * k_side
                          + gamma_orbit * orbit_scores
                          + psi_pot     * pot_bonus
                          + zeta_mrv    * mrv_scores
                          + eta_chunk   * chunk_bonus
                          + xi_echo     * echo_bonus
                          + pdn_weight  * pdn_bonus
                          + cot_weight  * cot_bonus
                          + mandate_boost
                          + punct_bias
                          + punct_penalty
                          + 0.4   * comp_bonus
                          + 0.25 * sorted_impulse
                          # ── ANISO additions ──────────────────────────
                          + self.aniso_ooi_weight       * ooi_aff_norm
                          - self.aniso_repulsion_weight * rep_norm)

        # Remission gating
        w1_rho = self.geo.triple_fast(w1).rho
        w2_rho = self.geo.triple_fast(w2).rho
        remission = self._remission.apply_remission(
            torch.tensor(w1_rho, device=self.device),
            torch.tensor(w2_rho, device=self.device),
            c_rho)
        raw_logits = raw_logits * remission

        # Save for replay / training
        self._fl_replay_buf.append((features.cpu().clone(), -1))
        self._pending_instr_probs   = p_instr
        self._pending_walk_logits   = raw_logits
        self._pending_crho          = c_rho
        self._pending_ctheta        = c_theta
        self._pending_csigma        = c_sigma
        self._pending_syn_norm      = syn_norm
        self._pending_trans_norm    = trans_norm
        self._pending_nystrom_rank  = RP_NYSTROM_M

        # Post-ANISO self-encumbrance
        prelim_probs = self._dnn.forward(raw_logits, c_rho, c_theta, c_sigma, temp=temp)
        encumb_bonus = self_encumbrance_from_probs(
            prelim_probs,
            strength=getattr(self, "self_encumbrance_weight", SELF_ENCUMBRANCE_W)
        )
        raw_logits = raw_logits + encumb_bonus
        self._pending_self_encumbrance = encumb_bonus

        # Final distribution
        if and_weight > 0.0 and self.instr_dist.base_dist_t is not None:
            log_instr = p_instr.clamp(min=1e-12).log()
            log_walk  = self._dnn.log_forward(raw_logits, c_rho, c_theta, c_sigma, temp=1.0)
            log_and   = and_weight * log_instr + (1.0 - and_weight) * log_walk
            final_probs = l1_simplex_project(log_and)
        else:
            final_probs = self._dnn.forward(raw_logits, c_rho, c_theta, c_sigma, temp=temp)

        return cands, final_probs

    def record_step_trace(self, step, chosen, cands, final_probs, and_weight):
        try:
            idx   = cands.index(chosen)
            p_and = final_probs[idx].item()
        except (ValueError, IndexError):
            idx, p_and = 0, 0.0

        p_instr = self._pending_instr_probs[idx].item() \
                  if self._pending_instr_probs is not None else 0.0

        if hasattr(self, '_pending_walk_logits') and self._pending_walk_logits is not None:
            log_walk = self._dnn.log_forward(
                self._pending_walk_logits,
                self._pending_crho, self._pending_ctheta, self._pending_csigma, temp=1.0)
            p_walk = log_walk[idx].exp().item()
        else:
            p_walk = 0.0

        source = ("instr" if p_instr > p_walk * 1.5 else
                  "walker" if p_walk > p_instr * 1.5 else "AND")

        if self._fl_replay_buf:
            feats, _ = self._fl_replay_buf[-1]
            self._fl_replay_buf[-1] = (feats, idx)

        trace = TokenStepTrace(
            step=step, chosen=chosen, p_instr=p_instr, p_walk=p_walk,
            p_and=p_and, and_weight=and_weight, source=source,
            syn_norm=self._pending_syn_norm, trans_norm=self._pending_trans_norm,
            rp_nystrom_rank=self._pending_nystrom_rank,
            ooi_size=self._pending_ooi_size,
            repulsion_mean=self._pending_repulsion_mean)
        self._step_traces.append(trace)
        return trace

    def push_token(self, token: str, sentence_len: int):
        """
        Push a newly generated token into the sentence state.
        Also offers the token to the OOI tracker for the sentence-level
        non-isotropic comparison set.
        """
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS:
            return
        self._cur_sent_toks.append(token)
        self._tok_pos += 1
        pos_norm = len(self._cur_sent_toks) / max(sentence_len, 1)
        triple   = self.geo.triple_fast(token)
        self.chunk_engine.push(triple, pos_norm)
        self._cur_orbit = self.pdn.orbit_of(token)
        # ── ANISO: register as object-of-interest if eligible ───────────
        self._ooi_tracker.push(token, triple)

    def step_trace_report(self, max_steps=30) -> str:
        if not self._step_traces: return "  (no step traces)"
        lines = ["  step  chosen          Pand   wt   source  zsyn   trans  nystrom  ooi  rep"]
        for t in self._step_traces[-max_steps:]:
            lines.append(t.render())
        if self._csns_syn_norms:
            avg_s = sum(self._csns_syn_norms) / len(self._csns_syn_norms)
            avg_t = sum(self._csns_trans_norms) / len(self._csns_trans_norms)
            lines.append(f"  [RP-CSNS avg] zsyn={avg_s:.4f}  trans={avg_t:.4f}"
                         f"  Nyström m={RP_NYSTROM_M}  RFF D={RP_RFF_DIM}")
        return "\n".join(lines)

    def algo_report(self) -> str:
        return "\n".join([
            "V18-RP-ANISO — Non-Isotropic Inter-Candidate Vectorisation",
            "",
            "1. RANDOM FOURIER FEATURES    O(C·D)   D=128",
            "2. NYSTRÖM APPROXIMATION      O(C·m)   m=32",
            "3. COUNT-MIN SKETCH           O(w·d)   w=1024 d=5",
            "4. RESERVOIR SAMPLING         O(C)     one-pass",
            "5. LSH ANN SEARCH             O(C·b·r) b=8 r=4",
            "6. RANDOM WALK MC             O(V·t)   t=20",
            "7. SKETCHED FFT (PDN)         O(T·k)   k=200",
            "8. FITTED LINE REGRESSION     O(C·19)  learnable delta",
            "9. ANISO DIR KERNEL           O(C²)    ρ-dependent θ stretch",
            "   – OOI affinity             O(C·|OOI|) sentence-aware",
            "   – DPP-lite repulsion       O(C²)    inter-candidate diversity",
            "",
            f"Fitted line active: {self.fitted_model is not None}",
            f"OOI tracker size:   {self._ooi_tracker.size}/{ANISO_OOI_MAX}",
            f"λ_rho={ANISO_LAMBDA_RHO}  λ_theta={ANISO_LAMBDA_THETA}  "
            f"λ_sigma={ANISO_LAMBDA_SIGMA}  α={ANISO_ALPHA}",
            self.fitted_model.feature_report() if self.fitted_model
            else "  (train via engine.train_fitted_line())",
        ])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 17 — FITTED LINE TRAINING PIPELINE  (updated for 19 features)
# ════════════════════════════════════════════════════════════════════════════

def train_fitted_line(walker: RPWalker, corpus_tokens: List[str],
                      batch_size=64, epochs=200, lr=3e-4,
                      max_replay_steps=50000, device=DEVICE) -> FittedLineRegression:
    print(f"[FittedLine] Replaying {len(corpus_tokens)} tokens, up to {max_replay_steps} steps…")

    if walker.fitted_model is None:
        walker.fitted_model = FittedLineRegression(FittedLineRegression.FEATURE_DIM).to(device)

    walker._fl_replay_buf.clear()
    features_list: List[torch.Tensor] = []
    gold_list:     List[int]          = []

    w1, w2 = corpus_tokens[0], corpus_tokens[1]
    steps_done = 0

    for t_pos in range(2, len(corpus_tokens)):
        if steps_done >= max_replay_steps:
            break
        gold_tok = corpus_tokens[t_pos]

        cands, probs = walker.walk_probs(w1, w2, temp=1e-9)
        if not cands:
            w1, w2 = w2, gold_tok; continue

        if gold_tok in cands:
            gold_idx = cands.index(gold_tok)
        else:
            w1, w2 = w2, gold_tok; continue

        if walker._fl_replay_buf:
            feats, _ = walker._fl_replay_buf.pop(0)
            features_list.append(feats)
            gold_list.append(gold_idx)
            steps_done += 1

        w1, w2 = w2, gold_tok

        if steps_done % 5000 == 0 and steps_done > 0:
            print(f"[FittedLine]   …replayed {steps_done} steps")

    if not features_list:
        print("[FittedLine] No training data collected.")
        return walker.fitted_model

    max_C = max(f.shape[0] for f in features_list)
    FD    = FittedLineRegression.FEATURE_DIM
    padded_feats, padded_golds = [], []
    for feats, gold_idx in zip(features_list, gold_list):
        C = feats.shape[0]
        # Pad or trim feature dim
        if feats.shape[1] < FD:
            feats = torch.cat([feats, torch.zeros(C, FD - feats.shape[1])], dim=1)
        elif feats.shape[1] > FD:
            feats = feats[:, :FD]
        if C < max_C:
            feats = torch.cat([feats, torch.zeros(max_C - C, FD)], dim=0)
        padded_feats.append(feats)
        padded_golds.append(min(gold_idx, max_C - 1))

    features_t = torch.stack(padded_feats).to(device)
    golds_t    = torch.tensor(padded_golds, dtype=torch.long, device=device)

    print(f"[FittedLine] Training on {len(features_t):,} steps, C={max_C}, D={FD}, epochs={epochs}")

    model = FittedLineRegression(feature_dim=FD, rank=1).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    from torch.utils.data import TensorDataset, DataLoader
    ds     = TensorDataset(features_t, golds_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for feat_b, gold_b in loader:
            loss = model.loss(feat_b, gold_b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[FittedLine]   Epoch {epoch:3d}/{epochs}  "
                  f"CE={epoch_loss/max(len(loader),1):.5f}")

    model.eval()
    walker.fitted_model = model
    print(f"[FittedLine] Done.\n{model.feature_report()}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# SECTION 18 — GENERATION  (unchanged interface)
# ════════════════════════════════════════════════════════════════════════════

def generate_passage_rp(walker: RPWalker, lm: RPCompositionLM,
                        num_sentences=4, tokens_per_sent=40,
                        seed_text="", instruction_text="",
                        and_weight=0.9, temperature=2.0,
                        return_traces=False):
    if instruction_text.strip():
        walker.instr_dist.set_instruction(instruction_text)
    elif seed_text.strip():
        walker.instr_dist.set_instruction(seed_text)

    walker._step_traces.clear()
    walker._csns_syn_norms.clear()
    walker._csns_trans_norms.clear()

    outputs, all_traces = [], []
    head_list = list(lm.heads.keys())
    if not head_list:
        return ("", "", "") if return_traces else ""

    seed_w1 = seed_w2 = None
    seed_toks = tokenize(seed_text) if seed_text else []
    if len(seed_toks) >= 2:
        seed_w1, seed_w2 = seed_toks[-2], seed_toks[-1]
    elif len(seed_toks) == 1:
        matches = [p for p in head_list if p[1] == seed_toks[0]]
        if matches: seed_w1, seed_w2 = random.choice(matches)
    if seed_w1 is None or (seed_w1, seed_w2) not in lm.heads:
        seed_w1, seed_w2 = random.choice(head_list)

    global_step = 0
    for sent_idx in range(num_sentences):
        if sent_idx == 0:
            w1, w2 = seed_w1, seed_w2
            init_toks = [w1, w2]
        else:
            w1, w2 = random.choice(head_list)
            init_toks = []

        plan_seeds = seed_toks if seed_toks else [w1, w2]
        trace = walker.begin_sentence(seed_tokens=plan_seeds, total_tokens=tokens_per_sent)
        all_traces.append(trace)

        toks = list(init_toks)
        for step in range(tokens_per_sent):
            cands, probs = walker.walk_probs(w1, w2, temp=temperature, and_weight=and_weight)
            if not cands: break
            nxt = cands[torch.multinomial(probs, 1).item()]
            walker.record_step_trace(global_step, nxt, cands, probs, and_weight)
            walker.push_token(nxt, tokens_per_sent)
            global_step += 1
            toks.append(nxt)
            if nxt in PUNCT_TOKENS and len(toks) > 8: break
            w1, w2 = w2, nxt

        sent_text = detokenize(toks)
        outputs.append(sent_text)
        walker.iso_stacker.add(toks, walker.geo, sent_text)

        # Surject into propositions
        props = walker.surjector.surject_sentence(toks)
        if props:
            walker._step_traces.append(TokenStepTrace(
                step=global_step, chosen="[SURJECTION]", p_instr=0, p_walk=0, p_and=0,
                and_weight=0, source="PROP", syn_norm=0, trans_norm=0,
                rp_nystrom_rank=0, ooi_size=0, repulsion_mean=0
            ))
            # Hack: We inject the proposition trace directly into the CoT traces string later
            if not hasattr(walker, '_prop_traces'): walker._prop_traces = []
            walker._prop_traces.append((sent_idx, props))

    full_text = " ".join(outputs)

    # Format propositions
    prop_text = "Small Propositional Statements:\n"
    if hasattr(walker, '_prop_traces') and walker._prop_traces:
        for s_idx, props in walker._prop_traces:
            prop_text += f"\nSentence {s_idx+1}:\n"
            for p in props:
                prop_text += p.render() + "\n"
        walker._prop_traces.clear()
    else:
        prop_text += "  (no confident propositions found)\n"

    if return_traces:
        cot_text  = walker.cot.all_traces_text()
        step_text = walker.step_trace_report()
        return full_text, cot_text, step_text, prop_text
    return full_text


# ════════════════════════════════════════════════════════════════════════════
# SECTION 19 — V18-RP-ANISO ENGINE
# ════════════════════════════════════════════════════════════════════════════

class V18RPEngine:
    def __init__(self, syn_weight=0.4, trans_weight=0.6, syn_k=8,
                 rff_dim=RP_RFF_DIM, nystrom_m=RP_NYSTROM_M,
                 aniso_ooi_weight=ANISO_OOI_W,
                 aniso_repulsion_weight=ANISO_REPULSION_W):
        self.device              = DEVICE
        self.syn_weight          = syn_weight
        self.trans_weight        = trans_weight
        self.syn_k               = syn_k
        self.rff_dim             = rff_dim
        self.nystrom_m           = nystrom_m
        self.aniso_ooi_weight    = aniso_ooi_weight
        self.aniso_repulsion_weight = aniso_repulsion_weight
        self._corpus_snippet     = ""
        self._initialised        = False

        self.rff        = RandomFourierFeatures(rff_dim=rff_dim, device=self.device)
        self.geo        = BolyaiTokenGeometryRP(device=self.device)
        self.lm         = RPCompositionLM(self.geo, self.rff, device=self.device)
        self.kernels    = RPKernels(self.rff)
        self.orbit      = BolyaiConjugateOrbit()
        self.rw_graph   = RandomWalkPotentialEngine(device=self.device)
        self.synth      = synthetic_reasonMandateProcessor()
        self.mrv        = RPMRVFilter(self.rff, device=self.device)
        self.chunk      = ChunkedSumEngine(device=self.device)
        self.isostacker = IsomorphicSyntaxStacker(self.rff, device=self.device)
        self.pdn        = SketchedPDNEngine(device=self.device)
        self.stublib    = RPCoTStubLibrary(self.rff, device=self.device)
        self.cot        = None
        self.instrdist  = None
        self.walker     = None

    def train(self, corpus_text: str):
        self._corpus_snippet = corpus_text
        print(f"[V18-RP-ANISO] Tokenising {len(corpus_text)} chars…")
        tokens = tokenize(corpus_text)
        self.lm.ingest(tokens)
        all_tokens = list(self.lm.raw_freq.keys())
        max_freq   = max(self.lm.raw_freq.values(), default=1.0)
        vocab_size = len(all_tokens)
        print(f"[V18-RP-ANISO] Registering {vocab_size} tokens…")
        for idx, tok in enumerate(all_tokens):
            self.geo.register(tok, self.lm.raw_freq[tok], idx, max_freq, vocab_size)
        print("[V18-RP-ANISO] Building GPU tensors + RFF feature cache…")
        self.geo.build_cuda_tensors(self.lm.vocab, self.rff)
        self.lm.finalise()
        print("[V18-RP-ANISO] Random Walk MC potential propagation…")
        self.rw_graph.build_from_trigrams(self.lm.tri_raw, self.lm.raw_freq, self.rff, self.geo)
        self.rw_graph.propagate()
        print("[V18-RP-ANISO] Priming LSH-based MRV filter…")
        self.mrv.prime(self.lm.vocab, self.geo)
        print("[V18-RP-ANISO] Sketched PDN spectral fitting…")
        self.pdn.fit_from_trigrams(self.geo, self.lm.tri_raw)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)
        print(self.pdn.theorem_bridge_report())
        print("[V18-RP-ANISO] Building RP CoT stub library + LSH ANN index…")
        self.stublib.build(self.geo, self.lm.vocab, self.lm.raw_freq)
        self.cot = RPCoTReasoningEngine(
            self.stublib, self.kernels, self.pdn,
            n_hops=3, tokens_per_hop=10, device=self.device)
        self.instrdist = RPInstructionDistribution(
            self.geo, self.kernels, self.lm, device=self.device)
        self.walker = RPWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.rw_graph, self.synth, self.mrv, self.chunk,
            self.isostacker, self.pdn, self.cot, self.instrdist,
            self.rff, device=self.device,
            syn_weight=self.syn_weight,
            trans_weight=self.trans_weight,
            syn_k=self.syn_k,
            aniso_ooi_weight=self.aniso_ooi_weight,
            aniso_repulsion_weight=self.aniso_repulsion_weight)
        self._initialised = True
        print("[V18-RP-ANISO] Engine ready.")

    def train_fitted_line(self, corpus_text: str = "",
                          epochs=200, lr=3e-4, max_steps=50000) -> FittedLineRegression:
        assert self._initialised, "Call .train() first."
        text   = corpus_text if corpus_text.strip() else self._corpus_snippet
        tokens = tokenize(text)
        if len(tokens) < 10:
            print("[FittedLine] Corpus too short!")
            return self.walker.fitted_model
        model = train_fitted_line(
            self.walker, tokens,
            epochs=epochs, lr=lr,
            max_replay_steps=max_steps,
            device=self.device)
        torch.save(model.state_dict(), "fitted_line_v18rp_aniso.pt")
        print("[FittedLine] Weights saved to fitted_line_v18rp_aniso.pt")
        return model

    def load_fitted_line(self, path="fitted_line_v18rp_aniso.pt"):
        assert self._initialised, "Call .train() first."
        model = FittedLineRegression(FittedLineRegression.FEATURE_DIM).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        self.walker.fitted_model = model
        print(f"[FittedLine] Loaded from {path}\n{model.feature_report()}")

    def generate(self, seed_text="", instruction_text="", num_sentences=4,
                 tokens_per_sent=40, and_weight=0.9, temperature=2.0,
                 return_traces=False):
        assert getattr(self, "_initialised", False), "Call .train() first."
        return generate_passage_rp(
            self.walker, self.lm,
            num_sentences=num_sentences, tokens_per_sent=tokens_per_sent,
            seed_text=seed_text, instruction_text=instruction_text,
            and_weight=and_weight, temperature=temperature,
            return_traces=return_traces)

    def save(self, path="v18rp_aniso_engine.pkl"):
        with open(path, "wb") as f: pickle.dump(self, f)
        print(f"[V18-RP-ANISO] Engine saved to {path}")

    @staticmethod
    def load(path="v18rp_aniso_engine.pkl") -> "V18RPEngine":
        with open(path, "rb") as f: eng = pickle.load(f)
        print(f"[V18-RP-ANISO] Engine loaded from {path}")
        return eng


# ════════════════════════════════════════════════════════════════════════════
# SECTION 20 — GRADIO GUI
# ════════════════════════════════════════════════════════════════════════════

_engine: Optional[V18RPEngine] = None

def _gui_init(mode, file_in, hf_name, hf_config, hf_split, hf_field,
              hf_portion, hf_max, syn_w, trans_w, syn_k, rff_dim, nystrom_m,
              ooi_w, rep_w):
    global _engine
    try:
        _engine = V18RPEngine(
            syn_weight=float(syn_w), trans_weight=float(trans_w),
            syn_k=int(syn_k), rff_dim=int(rff_dim), nystrom_m=int(nystrom_m),
            aniso_ooi_weight=float(ooi_w),
            aniso_repulsion_weight=float(rep_w))

        if mode == "Text file":
            if file_in is None: return "❌ No file uploaded."
            text = Path(file_in.name).read_text(encoding="utf-8", errors="replace")
        elif mode == "HuggingFace dataset":
            from datasets import load_dataset
            ds = load_dataset(hf_name, hf_config or None, split=hf_split)
            field = hf_field or "text"
            rows = int(len(ds) * max(0.01, min(1.0, float(hf_portion))))
            if hf_max and int(hf_max) > 0: rows = min(rows, int(hf_max))
            text = "\n".join(str(ds[i].get(field, "")) for i in range(rows))
        else:
            return "❌ Unknown mode."

        _engine.train(text)
        return (f"✅ Engine initialised (V18-RP-ANISO).\n"
                f"Vocab: {len(_engine.lm.vocab):,}  "
                f"Trigrams: {len(_engine.lm.tri_raw):,}  "
                f"Device: {_engine.device}\n"
                f"ANISO: OOI_w={ooi_w}  repulsion_w={rep_w}  "
                f"λρ={ANISO_LAMBDA_RHO}  λθ={ANISO_LAMBDA_THETA}  α={ANISO_ALPHA}\n\n"
                + _engine.pdn.theorem_bridge_report())
    except Exception as e:
        import traceback; return f"❌ Error:\n{traceback.format_exc()}"

def _gui_fit_line(epochs, lr, max_steps):
    global _engine
    if _engine is None or not _engine._initialised:
        return "❌ Initialise engine first."
    try:
        model = _engine.train_fitted_line(
            epochs=int(epochs), lr=float(lr), max_steps=int(max_steps))
        return f"✅ Fitted line trained.\n\n{model.feature_report()}"
    except Exception as e:
        import traceback; return f"❌ Error:\n{traceback.format_exc()}"

def _gui_generate(seed, instruction, n_sents, toks_per_sent,
                  and_weight, temperature, show_traces):
    global _engine
    if _engine is None or not _engine._initialised:
        return "❌ Initialise engine first.", "", "", ""
    try:
        text, cot, steps, props = _engine.generate(
            seed_text=seed, instruction_text=instruction,
            num_sentences=int(n_sents), tokens_per_sent=int(toks_per_sent),
            and_weight=float(and_weight), temperature=float(temperature),
            return_traces=True)
        cot_out   = cot   if show_traces else "(traces disabled)"
        steps_out = steps if show_traces else "(traces disabled)"
        prop_out  = props if show_traces else "(traces disabled)"
        return text, cot_out, steps_out, prop_out
    except Exception as e:
        import traceback; return f"❌ Error:\n{traceback.format_exc()}", "", "", ""

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="NeuroSymbolic V18-RP-ANISO") as demo:
        gr.Markdown("# NeuroSymbolic V18-RP-ANISO — Non-Isotropic Inter-Candidate Edition")
        gr.Markdown(
            "**New in ANISO**: Each sentence maintains a live *Object-of-Interest* (OOI) set of "
            "geometrically salient tokens. Candidates are scored against this set via the "
            "**anisotropic directional kernel** (ρ-stretched θ axis) and a **DPP-lite repulsion** "
            "term that penalises near-synonym clusters — making prediction weights non-isotropic "
            "to similar words.")

        with gr.Tab("Init / Train"):
            mode    = gr.Radio(["Text file","HuggingFace dataset"], value="Text file", label="Source")
            file_in = gr.File(label="Upload .txt")
            with gr.Row():
                hf_name    = gr.Textbox(label="HF dataset name")
                hf_config  = gr.Textbox(label="Config")
                hf_split   = gr.Textbox(value="train", label="Split")
                hf_field   = gr.Textbox(value="text", label="Text field")
                hf_portion = gr.Slider(0.01,1.0,value=1.0,label="Portion")
                hf_max     = gr.Textbox(value="", label="Max examples")
            with gr.Row():
                syn_w    = gr.Slider(0.0,5.0,value=1.0,step=0.1,label="Synaptic weight")
                trans_w  = gr.Slider(0.0,5.0,value=1.0,step=0.1,label="Transition weight")
                syn_k    = gr.Slider(1,64,value=16,step=1,label="Synaptic k")
                rff_dim  = gr.Slider(32,512,value=128,step=32,label="RFF dim D")
                nystrom_m= gr.Slider(8,128,value=32,step=4,label="Nyström m")
            gr.Markdown("### ANISO Hyperparameters")
            with gr.Row():
                ooi_w = gr.Slider(0.0,3.0,value=ANISO_OOI_W,step=0.05,
                                   label="OOI affinity weight")
                rep_w = gr.Slider(0.0,3.0,value=ANISO_REPULSION_W,step=0.05,
                                   label="Inter-candidate repulsion weight")
            init_btn = gr.Button("Initialise + Train")
            init_out = gr.Textbox(lines=20, label="Init output")
            init_btn.click(_gui_init,
                inputs=[mode,file_in,hf_name,hf_config,hf_split,hf_field,
                        hf_portion,hf_max,syn_w,trans_w,syn_k,rff_dim,nystrom_m,
                        ooi_w,rep_w],
                outputs=init_out)

        with gr.Tab("Fit Line"):
            gr.Markdown("Train the FittedLineRegression (19 features incl. OOI affinity + repulsion).")
            with gr.Row():
                fl_epochs   = gr.Slider(10,500,value=200,step=10,label="Epochs")
                fl_lr       = gr.Slider(1e-5,1e-2,value=3e-4,step=1e-5,label="Learning rate")
                fl_maxsteps = gr.Slider(1000,200000,value=50000,step=1000,label="Max replay steps")
            fl_btn = gr.Button("Train Fitted Line")
            fl_out = gr.Textbox(lines=22, label="Fitted line report")
            fl_btn.click(_gui_fit_line, inputs=[fl_epochs,fl_lr,fl_maxsteps], outputs=fl_out)

        with gr.Tab("Generate"):
            with gr.Row():
                seed_txt  = gr.Textbox(label="Seed text")
                instr_txt = gr.Textbox(label="Instruction text")
            with gr.Row():
                n_sents   = gr.Slider(1,16,value=4,step=1,label="Sentences")
                toks_sent = gr.Slider(10,120,value=80,step=5,label="Tokens/sentence")
                and_w     = gr.Slider(0.0,1.0,value=0.9,step=0.05,label="AND weight")
                temp      = gr.Slider(0.5,5.0,value=2.0,step=0.1,label="Temperature")
                show_tr   = gr.Checkbox(value=True,label="Show traces")
            gen_btn  = gr.Button("Generate")
            gen_out  = gr.Textbox(lines=8,  label="Generated text")
            prop_out = gr.Textbox(lines=6,  label="Surjected Propositions")
            cot_out  = gr.Textbox(lines=12, label="CoT traces")
            step_out = gr.Textbox(lines=14, label="Step traces (incl. OOI/repulsion)")
            gen_btn.click(_gui_generate,
                inputs=[seed_txt,instr_txt,n_sents,toks_sent,and_w,temp,show_tr],
                outputs=[gen_out,cot_out,step_out,prop_out])

    return demo


# ════════════════════════════════════════════════════════════════════════════
# SECTION 21 — ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V18-RP-ANISO FittedLine Edition")
    parser.add_argument("--corpus",      default="",    help="Path to corpus .txt")
    parser.add_argument("--fit_line",    action="store_true")
    parser.add_argument("--fit_epochs",  type=int,   default=200)
    parser.add_argument("--fit_lr",      type=float, default=3e-4)
    parser.add_argument("--fit_steps",   type=int,   default=50000)
    parser.add_argument("--seed",        default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--sentences",   type=int,   default=4)
    parser.add_argument("--save",        default="")
    parser.add_argument("--load",        default="")
    parser.add_argument("--gui",         action="store_true")
    parser.add_argument("--ooi_weight",  type=float, default=ANISO_OOI_W)
    parser.add_argument("--rep_weight",  type=float, default=ANISO_REPULSION_W)
    args = parser.parse_args()

    build_gradio_app().launch(share=False)