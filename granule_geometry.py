"""
granule_geometry.py
===================
Discrete vectorised geometry kernels for granularisation.

Exports
-------
CandidateMatrix          – vectorised candidate representation
membership_matrix        – (G x W) bool membership tensor
granule_centroids        – centre-of-mass per granule
gaussian_granule_masks   – soft Gaussian coverage mask (G x W)
granule_areas            – scalar area per granule  (weighted / unweighted)
word_granule_area        – per-word area summed across granules  (W,)
zone_dist_from_granules  – normalised probability vector from granule geometry
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EPS: float = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 1a. CANDIDATE MATRIX  –  dense word × feature representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateMatrix:
    """
    Compact vectorised view of a (word, prob) candidate list.

    Attributes
    ----------
    words   : list of W word strings
    probs   : (W,) float64 normalised probability vector
    idx     : (W,) float64 rank index  0 … W-1
    rank    : (W,) float64 normalised rank  0 … 1
    logp    : (W,) float64  log probabilities
    cumsum  : (W,) float64  cumulative probability
    feats   : (W, 4)  [rank | probs | logp | cumsum]  — intrinsic feature matrix
    """
    words:  List[str]
    probs:  torch.Tensor   # (W,) float64
    idx:    torch.Tensor   # (W,) float64
    rank:   torch.Tensor   # (W,) float64
    logp:   torch.Tensor   # (W,) float64
    cumsum: torch.Tensor   # (W,) float64
    feats:  torch.Tensor   # (W, 4) float64

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[Tuple[str, float]],
        device: Optional[torch.device] = None,
        dtype:  torch.dtype = torch.float64,
    ) -> "CandidateMatrix":
        W = len(pairs)
        if W == 0:
            empty = torch.zeros(0, dtype=dtype, device=device)
            return cls([], empty, empty, empty, empty, empty,
                       torch.zeros((0, 4), dtype=dtype, device=device))

        words = [w for w, _ in pairs]
        raw   = torch.tensor([float(v) for _, v in pairs], dtype=dtype, device=device)
        probs = raw / raw.sum().clamp_min(EPS)

        idx    = torch.arange(W, dtype=dtype, device=device)
        rank   = idx / max(1, W - 1)
        logp   = torch.log(probs.clamp_min(EPS))
        cumsum = torch.cumsum(probs, dim=0)
        feats  = torch.stack([rank, probs, logp, cumsum], dim=1)  # (W,4)

        return cls(words, probs, idx, rank, logp, cumsum, feats)

    @property
    def W(self) -> int:
        return len(self.words)

    def word_to_col(self) -> Dict[str, int]:
        return {w: i for i, w in enumerate(self.words)}

    def to(self, device: torch.device) -> "CandidateMatrix":
        return CandidateMatrix(
            self.words,
            self.probs.to(device), self.idx.to(device),
            self.rank.to(device),  self.logp.to(device),
            self.cumsum.to(device), self.feats.to(device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1b. MEMBERSHIP MATRIX  –  (G x W) bool
# ─────────────────────────────────────────────────────────────────────────────

def membership_matrix(
    cand:     CandidateMatrix,
    granules: Dict[Any, Sequence[str]],
    device:   Optional[torch.device] = None,
) -> Tuple[List[Any], torch.Tensor]:
    """
    Build a (G, W) boolean membership tensor.

    Parameters
    ----------
    cand     : CandidateMatrix – provides word list and column lookup
    granules : {key: [word, ...]}  – G granule sets

    Returns
    -------
    keys : list of G granule keys (same order as rows)
    M    : (G, W) bool tensor  –  M[g, w] = True iff word_w ∈ granule_g
    """
    keys  = list(granules.keys())
    G, W  = len(keys), cand.W
    lut   = cand.word_to_col()
    M     = torch.zeros((G, W), dtype=torch.bool, device=device)
    for r, k in enumerate(keys):
        for word in granules[k]:
            c = lut.get(word)
            if c is not None:
                M[r, c] = True
    return keys, M


# ─────────────────────────────────────────────────────────────────────────────
# 1c. GRANULE CENTROIDS  –  (G,) float64
# ─────────────────────────────────────────────────────────────────────────────

def granule_centroids(
    M:   torch.Tensor,   # (G, W) bool
    idx: torch.Tensor,   # (W,)   float64 candidate rank-indices
) -> torch.Tensor:
    """
    Compute the centre-of-mass rank for each granule.

    ctr[g] = mean( idx[w]  for w  in granule_g )

    Returns (G,) float64.
    """
    M_f   = M.to(idx.dtype)                   # (G, W)
    count = M_f.sum(dim=1).clamp_min(1.0)     # (G,)
    return (M_f * idx.unsqueeze(0)).sum(dim=1) / count   # (G,)


# ─────────────────────────────────────────────────────────────────────────────
# 1d. GAUSSIAN GRANULE MASKS  –  (G x W) float64
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_granule_masks(
    M:     torch.Tensor,    # (G, W) bool
    idx:   torch.Tensor,    # (W,)   float64  candidate indices
    sigma: torch.Tensor,    # scalar learnable std-dev
    floor: torch.Tensor,    # scalar learnable probability floor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft Gaussian coverage mask for each granule centred at its centre of mass.

    mask[g, w] = floor + (1 - floor) * exp( -0.5 * ((idx[w] - ctr[g]) / sigma)^2 )

    Returns
    -------
    mask : (G, W) float64
    ctr  : (G,)   float64  centroid rank-indices
    """
    sigma = sigma.clamp_min(1e-6)
    floor = floor.clamp(0.0, 1.0 - 1e-6)
    ctr   = granule_centroids(M, idx)               # (G,)
    diff  = idx.unsqueeze(0) - ctr.unsqueeze(1)     # (G, W)  broadcast
    mask  = floor + (1.0 - floor) * torch.exp(-0.5 * (diff / sigma) ** 2)
    return mask, ctr


# ─────────────────────────────────────────────────────────────────────────────
# 1e. GRANULE AREAS  –  scalar area per granule
# ─────────────────────────────────────────────────────────────────────────────

def granule_areas(
    mask:  torch.Tensor,                      # (G, W)
    probs: Optional[torch.Tensor] = None,     # (W,)  optional probability weights
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-granule area as the integral of the mask.

    Unweighted:  area[g] = sum_w  mask[g, w]
    Weighted:    area[g] = sum_w  mask[g, w] * probs[w]

    Returns
    -------
    area     : (G,) area per granule
    weighted : (G, W) element-wise contribution  (= mask if unweighted)
    """
    if probs is None:
        return mask.sum(dim=1), mask
    weighted = mask * probs.unsqueeze(0)    # (G, W) broadcast
    return weighted.sum(dim=1), weighted


# ─────────────────────────────────────────────────────────────────────────────
# 1f. PER-WORD GRANULE AREA  –  (W,) float64
# ─────────────────────────────────────────────────────────────────────────────

def word_granule_area(
    cand:    CandidateMatrix,
    granules: Dict[Any, Sequence[str]],
    sigma:   torch.Tensor,
    floor:   torch.Tensor,
    device:  Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    For every word in `cand`, compute its total area contribution
    across all granules.

    Steps
    -----
    1. Build membership matrix  M  (G x W)
    2. Compute Gaussian granule masks  (G x W)
    3. Weight by candidate probs  → per-element contributions  (G x W)
    4. Sum over granules  → per-word area  (W,)

    Returns a dict with all intermediate tensors for inspection / loss use.
    """
    cand_d = cand.to(device) if device else cand
    keys, M = membership_matrix(cand_d, granules, device=device)
    mask, ctr = gaussian_granule_masks(M, cand_d.idx, sigma, floor)
    g_area, contrib = granule_areas(mask, cand_d.probs)
    w_area = contrib.sum(dim=0)     # (W,)  sum over granules per word

    return {
        "granule_keys":           keys,
        "membership":             M,           # (G, W) bool
        "centroids":              ctr,          # (G,)
        "mask":                   mask,         # (G, W) float64
        "granule_area":           g_area,       # (G,)
        "per_element_contrib":    contrib,      # (G, W)
        "word_area":              w_area,       # (W,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1g. ZONE DISTRIBUTION FROM GRANULES  –  normalised (W,) probability
# ─────────────────────────────────────────────────────────────────────────────

def zone_dist_from_granules(
    layer_name: str,
    cand:       CandidateMatrix,
    granules:   Dict[Any, Sequence[str]],
    sigma:      torch.Tensor,
    floor:      torch.Tensor,
    device:     Optional[torch.device] = None,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    Build a normalised zone probability distribution from granule geometry.

    1. Compute word_granule_area  (W,)
    2. Normalise to a probability simplex
    3. Return (word, prob) pairs  +  layerdict with all geometry info

    This replaces the hand-coded Gaussian zone layers in the original pipeline.
    """
    info  = word_granule_area(cand, granules, sigma, floor, device=device)
    w_area = info["word_area"]                          # (W,)
    probs  = w_area / w_area.sum().clamp_min(EPS)       # normalise

    pairs = list(zip(cand.words, probs.tolist()))
    ldict = {
        "name":   layer_name,
        "words":  cand.words,
        "probs":  probs.detach().cpu().numpy(),
        "probs_t": probs,
        **info,
    }
    return pairs, ldict
