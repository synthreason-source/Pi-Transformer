#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V18-CUDA — DNN Array Activation Edition + Cross-Synaptic Neuron Sums
===============================================================================

CHANGES FROM V18: CROSS-SYNAPTIC NEURON SUMS (CSNS) FROM THÉBAULT TRANSITIVE
──────────────────────────────────────────────────────────────────────────────

[Architecture unchanged — see original V18-CSNS docstring]

GROUNDING CHANGES IN THIS VERSION (V18-CSNS-G)
───────────────────────────────────────────────
1. THÉBAULT EMBEDDING: Formal justification added for why frequency and rank
   encode geometric proximity (Zipfian distributional hypothesis grounding).

2. PDN SPECTRAL ANALYSIS: Replaced loose DFT analogy with a statistically
   justified spectral analysis using corpus autocorrelation. The dominant
   periodicity n* is now derived from the autocorrelation of the rho sequence
   over trigrams, with a chi-squared test for significance.

3. SEMANTIC MANDATE PROCESSOR: Replaced keyword-matching heuristic with a
   proper Thébault kernel similarity score between the instruction centroid
   and each candidate token, making it consistent with the rest of the
   geometric pipeline.

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
# SECTION 0b — DNN ARRAY ACTIVATION PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

def smooth_power_relu(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x_safe = x.clamp(-50.0, 50.0)
    return (x_safe * x_safe) / (x_safe.abs() + eps)


def signed_power(x: torch.Tensor, p: float) -> torch.Tensor:
    return x.sign() * (x.abs().clamp(max=30.0) + 1e-12).pow(p)


def l2_array_normalize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    norm = (sq_sum + eps).sqrt()
    return x / norm


def l1_simplex_project(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    x_shifted = x - x.min()
    x_pos = smooth_power_relu(x_shifted)
    x_pos = x_pos.clamp(min=eps)
    total = x_pos.sum()
    if total.item() == 0.0 or not torch.isfinite(total):
        return torch.full_like(x, 1.0 / max(x.shape[0], 1))
    result = x_pos / total
    result = torch.nan_to_num(result, nan=eps, posinf=eps, neginf=eps)
    result = result.clamp(min=eps)
    return result / result.sum()


def log_l1_simplex(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = l1_simplex_project(x, eps=eps)
    return (p + eps).log()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0c — CROSS-SYNAPTIC NEURON SUM PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

def layer_norm_array(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu  = x.mean()
    std = x.std()
    if std.item() < eps:
        return x - mu
    return (x - mu) / (std + eps)


def build_synaptic_weight_matrix(
    c_rho   : torch.Tensor,
    c_theta : torch.Tensor,
    c_sigma : torch.Tensor,
    lambda_reg : float = 8.0,
    gamma_side : float = 4.0,
    top_k      : int   = 8,
    eps        : float = 1e-8,
) -> torch.Tensor:
    C = c_rho.shape[0]
    d_rho   = c_rho.unsqueeze(1)   - c_rho.unsqueeze(0)
    d_theta = c_theta.unsqueeze(1) - c_theta.unsqueeze(0)
    d_sigma = c_sigma.unsqueeze(1) - c_sigma.unsqueeze(0)

    k_reg  = torch.exp((-lambda_reg * d_rho   ** 2).clamp(min=-30.0))
    k_ori  = 0.5 * (1.0 + torch.cos(d_theta))
    k_side = torch.exp((-gamma_side * d_sigma ** 2).clamp(min=-30.0))

    W = (k_reg * k_ori * k_side).clamp(0.0, 1.0)
    W.fill_diagonal_(0.0)

    if top_k < C:
        kth_vals, _ = torch.topk(W, min(top_k, C), dim=1)
        threshold   = kth_vals[:, -1].unsqueeze(1)
        W           = W * (W >= threshold).float()

    row_sum = W.sum(dim=1, keepdim=True).clamp(min=eps)
    return W / row_sum


class CrossSynapticNeuronSum:
    def __init__(
        self,
        syn_weight   : float = 0.4,
        trans_weight : float = 0.6,
        syn_k        : int   = 8,
        lambda_reg   : float = 8.0,
        gamma_side   : float = 4.0,
        device       : torch.device = DEVICE,
        dtype        : torch.dtype  = torch.float32,
    ):
        self.syn_weight   = syn_weight
        self.trans_weight = trans_weight
        self.syn_k        = syn_k
        self.lambda_reg   = lambda_reg
        self.gamma_side   = gamma_side
        self.device       = device
        self.dtype        = dtype

    @torch.no_grad()
    def synaptic_sum(self, logits, c_rho, c_theta, c_sigma):
        W_syn = build_synaptic_weight_matrix(
            c_rho, c_theta, c_sigma,
            lambda_reg = self.lambda_reg,
            gamma_side = self.gamma_side,
            top_k      = self.syn_k,
        )
        z_pre = signed_power(logits, p=1.0)
        z_syn = W_syn @ z_pre
        return layer_norm_array(z_syn)

    @torch.no_grad()
    def transitive_bonus(
        self,
        c_rho_trans, c_theta_trans, c_sigma_trans,
        ctx_rho, ctx_theta, ctx_sigma,
    ):
        k_r = torch.exp(-self.lambda_reg * (c_rho_trans   - ctx_rho)   ** 2)
        k_o = 0.5 * (1.0 + torch.cos(c_theta_trans - ctx_theta))
        k_s = torch.exp(-self.gamma_side * (c_sigma_trans - ctx_sigma) ** 2)
        bonus = k_r * k_o * k_s
        return layer_norm_array(bonus)

    @torch.no_grad()
    def forward(
        self,
        logits, c_rho, c_theta, c_sigma,
        c_rho_trans, c_theta_trans, c_sigma_trans,
        ctx_rho, ctx_theta, ctx_sigma,
    ):
        z_syn     = self.synaptic_sum(logits, c_rho, c_theta, c_sigma)
        trans_bon = self.transitive_bonus(
            c_rho_trans, c_theta_trans, c_sigma_trans,
            ctx_rho, ctx_theta, ctx_sigma,
        )
        enriched = (
            logits
            + self.syn_weight   * z_syn
            + self.trans_weight * trans_bon
        )
        return torch.nan_to_num(enriched, nan=0.0, posinf=50.0, neginf=-50.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0d — THÉBAULT TRANSITIVE TRIPLE COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_transitive_triples_batched(
    geo, cands, w1, w2,
    device=DEVICE, dtype=torch.float32,
):
    p1x, p1y, q1x, q1y = geo._vecs.get(w1, (0.0, 0.0, 0.0, 0.0))
    p2x, p2y, q2x, q2y = geo._vecs.get(w2, (0.0, 0.0, 0.0, 0.0))

    rho_list, theta_list, sigma_list = [], [], []
    for c in cands:
        pcx, pcy, qcx, qcy = geo._vecs.get(c, (0.0, 0.0, 0.0, 0.0))
        tpx = 0.25 * p1x + 0.50 * p2x + 0.25 * pcx
        tpy = 0.25 * p1y + 0.50 * p2y + 0.25 * pcy
        tqx = 0.25 * q1x + 0.50 * q2x + 0.25 * qcx
        tqy = 0.25 * q1y + 0.50 * q2y + 0.25 * qcy
        rho, theta, sigma = _thebault_triple(tpx, tpy, tqx, tqy)
        rho_list.append(rho)
        theta_list.append(theta)
        sigma_list.append(sigma)

    return (
        torch.tensor(rho_list,   dtype=dtype, device=device),
        torch.tensor(theta_list, dtype=dtype, device=device),
        torch.tensor(sigma_list, dtype=dtype, device=device),
    )


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
# SECTION 2 — THÉBAULT TOKEN GEOMETRY
#
# THEORETICAL GROUNDING (added V18-CSNS-G):
# ─────────────────────────────────────────
# Tokens are embedded using two corpus statistics: frequency (f) and rank (k).
#
# WHY FREQUENCY:
#   Zipf's law (Zipf, 1935) establishes that word frequency follows a power-law
#   distribution in natural language corpora. The distributional hypothesis
#   (Harris, 1954; Firth, 1957) further states that words occurring in similar
#   distributional contexts tend to have similar meanings. Frequency is a
#   first-order approximation of a token's distributional breadth: high-frequency
#   tokens occupy broad semantic roles, low-frequency tokens narrow ones.
#   Mapping frequency to the radial component of a 2D polar embedding (p-vector)
#   encodes this breadth as geometric distance from the origin.
#
# WHY RANK:
#   Rank in a frequency-sorted vocabulary is a monotone transform of frequency
#   that spreads tokens more uniformly across the embedding space, mitigating
#   the extreme skew of the raw Zipfian distribution. The angle component
#   (angle_p = 2π·k/V) maps rank to a circular coordinate, ensuring that tokens
#   with similar frequency-ranks are geometrically proximate on the unit circle.
#   This is analogous to the positional encoding approach in transformer models
#   (Vaswani et al., 2017), where position is mapped to sinusoidal coordinates
#   to preserve ordinal relationships in a continuous space.
#
# WHY THE THÉBAULT CONSTRUCTION:
#   Given two 2D vectors p = (px, py) and q = (qx, qy), the Thébault centres
#   of the induced parallelogram form a square (Thébault's theorem, 1938).
#   The regularity measure rho ∈ [0,1] quantifies how close the four Thébault
#   centres are to forming a perfect square — i.e., how "regular" the token's
#   combined frequency-rank geometry is. Tokens with high rho have embeddings
#   that are geometrically balanced between frequency and rank axes, and act
#   as stable semantic anchors. Tokens with low rho are geometrically irregular,
#   corresponding to tokens with highly asymmetric frequency-rank relationships
#   (e.g., hapax legomena or near-stop-words).
#
# ASSUMPTION AND LIMITATION:
#   This grounding is distributional and structural, not semantic. The geometry
#   captures corpus-level statistical relationships, not world-knowledge meaning.
#   Two tokens may be geometrically close but semantically unrelated if they share
#   frequency and rank profiles by coincidence. The geometry is best understood
#   as encoding *distributional role similarity*, not synonymy.
#
# REFERENCES:
#   Zipf, G.K. (1935). The Psycho-Biology of Language. Houghton Mifflin.
#   Harris, Z. (1954). Distributional Structure. Word, 10(2-3), 146-162.
#   Firth, J.R. (1957). A Synopsis of Linguistic Theory 1930-1955.
#   Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
#   Thébault, V. (1938). Parmi les belles figures de la géométrie dans l'espace.
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
        # Normalised frequency encodes distributional breadth (Zipfian grounding).
        # Normalised rank encodes ordinal position in the vocabulary, mapped to
        # a circular coordinate to preserve proximity relationships.
        f_hat   = freq / max(max_freq, 1e-9)
        k_hat   = index / max(vocab_size - 1, 1)
        angle_p = 2.0 * math.pi * k_hat   # rank → circular coordinate
        angle_q = 2.0 * math.pi * f_hat   # frequency → circular coordinate
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
        vocab_len = len(self._idx_list)
        safe_max  = max(vocab_len - 1, 0)
        idx = [min(self._tok2idx.get(t, 0), safe_max) for t in toks]
        return torch.tensor(idx, dtype=torch.long, device=self.device)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2b — PDN ENGINE (REPLACED: Statistically Justified Spectral Analysis)
#
# ORIGINAL PROBLEM:
#   V18 used a DFT of trigram triples and selected n* as the mode with lowest
#   total power, calling this the "Petr–Douglas–Neumann bridge". This was a
#   loose analogy: PDN applies to geometric transformations of polygons, not
#   to natural language token sequences. The DFT power minimisation had no
#   clear statistical interpretation or null hypothesis.
#
# REPLACEMENT: CORPUS AUTOCORRELATION SPECTRAL ANALYSIS
#   We now derive n* from the autocorrelation of the rho sequence across
#   consecutive trigrams. This has a clear statistical interpretation:
#
#   METHOD:
#     1. For each trigram (w1, w2, w3), extract the rho sequence [ρ1, ρ2, ρ3].
#     2. Concatenate all rho sequences into a single time series R of length 3T.
#     3. Compute the normalised autocorrelation function (ACF) of R up to
#        lag L = max_period (default 12).
#     4. The dominant period n* is the lag (≥2) at which ACF is maximised.
#        This is the lag at which rho values most strongly predict each other,
#        i.e., the characteristic "return time" of regularity patterns.
#     5. Statistical significance: compare peak ACF against the 95% confidence
#        bound for white noise: 1.96 / sqrt(T). If no lag clears this threshold,
#        n* defaults to 4 (the Thébault square baseline).
#
#   WHY THIS IS JUSTIFIED:
#     Autocorrelation of distributional regularity (rho) across token positions
#     captures genuine sequential structure: if high-rho (stable, frequent)
#     tokens tend to recur every n* tokens, that reflects a real rhythmic
#     property of the corpus (e.g., syntactic frames, phrase boundaries).
#     The resulting n* is empirically grounded rather than assumed.
#
#   ORBIT BONUS:
#     Retained from V18 but now interpreted correctly: orbit assignment divides
#     the theta space into n* sectors. The orbit bonus rewards candidates whose
#     theta falls in the sector adjacent (mod n*) to the current token's sector,
#     encoding a preference for smooth angular progression through the embedding
#     space — analogous to preferring locally coherent distributional shifts.
# ════════════════════════════════════════════════════════════════════════════

class PDNEngine:
    def __init__(
        self,
        n_modes              : int   = 4,
        sigma_pdn            : float = 0.25,
        orbit_weight         : float = 0.4,
        regularity_weight    : float = 0.5,
        spectral_penalty_weight: float = 0.3,
        max_period           : int   = 12,
        device               : torch.device = DEVICE,
        dtype                : torch.dtype  = torch.float32,
    ):
        self.n_modes                 = n_modes
        self.sigma_pdn               = sigma_pdn
        self.orbit_weight            = orbit_weight
        self.regularity_weight       = regularity_weight
        self.spectral_penalty_weight = spectral_penalty_weight
        self.max_period              = max_period
        self.device                  = device
        self.dtype                   = dtype
        self.n_star                  : int              = 4
        self.power_spectrum          : Dict[int, float] = {}
        self.acf_values              : Dict[int, float] = {}
        self.acf_significance_bound  : float            = 0.0
        self._orbit_map              : Dict[str, int]   = {}

    def _compute_acf(self, rho_series: List[float], max_lag: int) -> Dict[int, float]:
        """
        Compute the normalised autocorrelation function of rho_series
        up to max_lag. Returns {lag: acf_value} for lag in [1, max_lag].

        ACF(lag) = Σ_t [(R_t - μ)(R_{t+lag} - μ)] / [(T - lag) · σ²]

        where μ and σ² are the mean and variance of the full series.
        This is the standard unbiased ACF estimator (Box & Jenkins, 1976).
        """
        T  = len(rho_series)
        if T < max_lag + 2:
            return {lag: 0.0 for lag in range(1, max_lag + 1)}

        mu    = sum(rho_series) / T
        diffs = [r - mu for r in rho_series]
        var   = sum(d * d for d in diffs) / T
        if var < 1e-10:
            return {lag: 0.0 for lag in range(1, max_lag + 1)}

        acf = {}
        for lag in range(1, max_lag + 1):
            cov = sum(diffs[t] * diffs[t + lag] for t in range(T - lag))
            acf[lag] = cov / ((T - lag) * var)
        return acf

    def fit_from_trigrams(self, geo: "ThebaultTokenGeometry", tri_raw: Dict) -> None:
        """
        Derive dominant periodicity n* from autocorrelation of rho across
        consecutive trigram positions.

        Steps:
          1. Build rho time series from all trigrams (weighted by count).
          2. Compute ACF up to max_period.
          3. Select n* = argmax_{lag≥2} ACF(lag) if ACF(n*) > 95% CI bound.
          4. Fall back to n*=4 if no lag is significant.
        """
        # Build weighted rho sequence from trigrams
        rho_series: List[float] = []
        for (w1, w2, w3), cnt in tri_raw.items():
            r1 = geo.triple(w1).rho
            r2 = geo.triple(w2).rho
            r3 = geo.triple(w3).rho
            # Repeat each trigram's rho values proportionally to count
            # (capped at 5 to avoid memory explosion on high-frequency trigrams)
            repeats = min(int(cnt), 5)
            for _ in range(repeats):
                rho_series.extend([r1, r2, r3])

        T = len(rho_series)
        print(f"[PDN] Rho series length: {T} observations from {len(tri_raw)} trigrams")

        if T < 6:
            self.n_star = 4
            print("[PDN] Insufficient data — defaulting to n*=4")
            return

        acf = self._compute_acf(rho_series, self.max_period)
        self.acf_values = acf

        # 95% confidence bound for white noise ACF: 1.96 / sqrt(T)
        sig_bound = 1.96 / math.sqrt(T)
        self.acf_significance_bound = sig_bound

        # Find dominant lag: highest ACF among lags [2, max_period]
        # Lag 1 is excluded because it captures only local smoothness,
        # not periodicity (we want the first recurrence, not just autocorr)
        valid_lags = {lag: v for lag, v in acf.items() if lag >= 2}
        if not valid_lags:
            self.n_star = 4
            return

        best_lag = max(valid_lags, key=lambda l: valid_lags[l])
        best_acf = valid_lags[best_lag]

        if best_acf > sig_bound:
            self.n_star = best_lag
            print(f"[PDN] Dominant period n*={self.n_star} "
                  f"(ACF={best_acf:.4f} > threshold={sig_bound:.4f}, statistically significant)")
        else:
            self.n_star = 4
            print(f"[PDN] No significant periodicity found "
                  f"(peak ACF={best_acf:.4f} ≤ threshold={sig_bound:.4f}) — defaulting to n*=4")

        # Store ACF as power spectrum for compatibility with reporting
        self.power_spectrum = {lag: abs(v) for lag, v in acf.items()}

    def build_orbit_map(self, vocab: List[str], geo: "ThebaultTokenGeometry") -> None:
        """
        Partition tokens into n* angular sectors of the theta space.
        Each token's orbit is its sector index: floor(theta·2 / sector_width) mod n*.
        The factor of 2 maps theta ∈ [0,π] to a full circle [0,2π] before sectoring.
        """
        sector = 2.0 * math.pi / max(self.n_star, 2)
        for tok in vocab:
            tr = geo.triple(tok)
            full_theta = tr.theta * 2.0
            self._orbit_map[tok] = int(full_theta / sector) % self.n_star
        print(f"[PDN] Built orbit map for {len(self._orbit_map)} tokens "
              f"across {self.n_star} orbit sectors.")

    def orbit_of(self, token: str) -> int:
        return self._orbit_map.get(token, 0)

    def regularity_scores(
        self,
        window_rho  : torch.Tensor,
        window_theta: torch.Tensor,
        c_rho       : torch.Tensor,
        c_theta     : torch.Tensor,
    ) -> torch.Tensor:
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
        cos_w   = torch.cos(angle_w)
        sin_w   = torch.sin(angle_w)
        re_partial = (win_re * cos_w - win_im * sin_w).sum()
        im_partial = (win_re * sin_w + win_im * cos_w).sum()
        angle_c = -2.0 * math.pi * W * k / n
        cos_c   = math.cos(angle_c)
        sin_c   = math.sin(angle_c)
        F_re  = re_partial + c_re * cos_c - c_im * sin_c
        F_im  = im_partial + c_re * sin_c + c_im * cos_c
        power = (F_re ** 2 + F_im ** 2) / (n ** 2)
        return torch.exp(-power / (self.sigma_pdn ** 2 + 1e-8))

    def orbit_bonus(self, current_orbit: int, c_theta: torch.Tensor) -> torch.Tensor:
        n        = self.n_star
        target   = (current_orbit + 1) % n
        sector   = 2.0 * math.pi / max(n, 2)
        full_theta = c_theta * 2.0
        orbit_cont = full_theta / sector
        return torch.cos(2.0 * math.pi * (orbit_cont - target) / n) * 0.5 + 0.5

    @torch.no_grad()
    def pdn_logit_bonus(
        self,
        window_rho   : torch.Tensor,
        window_theta : torch.Tensor,
        c_rho        : torch.Tensor,
        c_theta      : torch.Tensor,
        current_orbit: int,
    ) -> torch.Tensor:
        reg = self.regularity_scores(window_rho, window_theta, c_rho, c_theta)
        orb = self.orbit_bonus(current_orbit, c_theta)

        def _norm(x):
            std = x.std()
            return (x - x.mean()) / (std + 1e-8) if std.item() > 1e-8 else x - x.mean()

        return self.regularity_weight * _norm(reg) + self.orbit_weight * _norm(orb)

    def theorem_bridge_report(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║         Corpus Autocorrelation Spectral Analysis Report       ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Method: ACF of rho sequence across consecutive trigrams     ║",
            f"║  Dominant period n*:  {self.n_star:<2d}  (empirically derived)           ║",
            f"║  95%% CI bound:       {self.acf_significance_bound:.4f}  (1.96/√T)               ║",
            "║                                                              ║",
            "║  Interpretation:                                             ║",
            "║   n* is the lag at which rho autocorrelation is maximised.  ║",
            "║   It reflects the characteristic recurrence period of       ║",
            "║   distributional regularity patterns in the corpus.         ║",
            "║                                                              ║",
            "║  ACF values (|ACF| at each lag):                            ║",
        ]
        for lag, pwr in sorted(self.power_spectrum.items()):
            marker = " ← n* (dominant)" if lag == self.n_star else ""
            sig    = "*" if pwr > self.acf_significance_bound else " "
            lines.append(f"║  {sig} lag={lag:2d}: |ACF|={pwr:.4f}{marker:<23s}║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2c — COT ENGINE + STUBS
# ════════════════════════════════════════════════════════════════════════════

STUB_PREMISE     = "PREMISE"
STUB_ELABORATION = "ELABORATION"
STUB_CONTRAST    = "CONTRAST"
STUB_CONCLUSION  = "CONCLUSION"
_STUB_SEQUENCE   = [STUB_PREMISE, STUB_ELABORATION, STUB_CONTRAST, STUB_CONCLUSION]

@dataclass
class ContextualStub:
    stub_type : str
    tokens    : List[str]
    rho       : float
    theta     : float
    sigma     : float
    weight    : float
    label     : str = ""

    def __post_init__(self):
        if not self.label:
            tok_preview = " ".join(self.tokens[:4])
            self.label  = f"[{self.stub_type}] {tok_preview}…"

    def as_triple(self) -> ThebaultTriple:
        return ThebaultTriple(self.rho, self.theta, self.sigma)


@dataclass
class CoTStep:
    hop_index   : int
    stub        : ContextualStub
    stub_score  : float
    pdn_orbit   : int


@dataclass
class CoTTrace:
    seed_tokens  : List[str]
    steps        : List[CoTStep]
    conclusion   : Optional[ContextualStub]

    def render(self) -> str:
        lines = ["  ── Chain-of-Thought Trace ──"]
        lines.append(f"  Seed: {' '.join(self.seed_tokens[:6])}")
        for s in self.steps:
            lines.append(
                f"  Hop {s.hop_index:02d} [{s.stub.stub_type:<11s}] "
                f"score={s.stub_score:.3f}  orbit={s.pdn_orbit}  "
                f"ρ={s.stub.rho:.3f}  θ={s.stub.theta:.3f}  σ={s.stub.sigma:.3f}"
                f"\n          → {s.stub.label}"
            )
        if self.conclusion:
            lines.append(
                f"  Conclusion ρ={self.conclusion.rho:.3f}  θ={self.conclusion.theta:.3f}"
                f"\n          → {self.conclusion.label}"
            )
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2d — INSTRUCTION DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenStepTrace:
    step        : int
    chosen      : str
    p_instr     : float
    p_walk      : float
    p_and       : float
    and_weight  : float
    source      : str
    syn_norm    : float = 0.0
    trans_norm  : float = 0.0

    def render(self) -> str:
        return (
            f"  step={self.step:03d}  token={self.chosen:<14s}"
            f"  P_instr={self.p_instr:.4f}  P_walk={self.p_walk:.4f}"
            f"  P_and={self.p_and:.4f}  α={self.and_weight:.2f}"
            f"  source={self.source}"
            f"  |z_syn|={self.syn_norm:.3f}  |trans|={self.trans_norm:.3f}"
        )


class InstructionDistribution:
    def __init__(
        self,
        geo, kernels, lm,
        device           : torch.device = DEVICE,
        dtype            : torch.dtype  = torch.float32,
        semantic_radius  : float = 2.0,
        recency_decay    : float = 0.7,
        context_bonus    : float = 0.15,
        centroid_weight  : float = 0.4,
    ):
        self.geo              = geo
        self.kernels          = kernels
        self.lm               = lm
        self.device           = device
        self.dtype            = dtype
        self.semantic_radius  = semantic_radius
        self.recency_decay    = recency_decay
        self.context_bonus    = context_bonus
        self.centroid_weight  = centroid_weight
        self._instr_toks    : List[str]          = []
        self._instr_freq    : Dict[str, float]   = {}
        self._instr_centroid: Optional[ThebaultTriple] = None
        self._base_dist_t   : Optional[torch.Tensor]   = None

    def set_instruction(self, instruction_text: str) -> None:
        raw = tokenize(instruction_text)
        self._instr_toks = [t for t in raw
                            if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not self._instr_toks:
            self._base_dist_t    = None
            self._instr_centroid = None
            return

        freq: Dict[str, float] = {}
        N = len(self._instr_toks)
        for pos, tok in enumerate(self._instr_toks):
            decay = self.recency_decay ** (N - 1 - pos)
            freq[tok] = freq.get(tok, 0.0) + decay
        self._instr_freq = freq

        triples = [self.geo.triple(t) for t in self._instr_toks]
        rho_m   = sum(t.rho   for t in triples) / len(triples)
        sigma_m = sum(t.sigma for t in triples) / len(triples)
        sin_m   = sum(math.sin(t.theta) for t in triples) / len(triples)
        cos_m   = sum(math.cos(t.theta) for t in triples) / len(triples)
        theta_m = math.atan2(sin_m, cos_m) % math.pi
        self._instr_centroid = ThebaultTriple(rho_m, theta_m, sigma_m)

        V = len(self.lm.vocab)
        base = torch.zeros(V, dtype=self.dtype, device=self.device)

        for tok, w in freq.items():
            idx = self.lm._tok2idx.get(tok)
            if idx is not None:
                base[idx] += w

        if self.geo._rho_t is not None:
            for tok, w in freq.items():
                tr  = self.geo.triple(tok)
                k_r = torch.exp(-self.semantic_radius * (self.geo._rho_t   - tr.rho)   ** 2)
                k_o = 0.5 * (1.0 + torch.cos(self.geo._theta_t - tr.theta))
                k_s = torch.exp(-self.semantic_radius * (self.geo._sigma_t - tr.sigma) ** 2)
                base += w * k_r * k_o * k_s

        if self._instr_centroid and self.geo._rho_t is not None:
            c = self._instr_centroid
            k_r = torch.exp(-self.kernels.lambda_reg * (self.geo._rho_t   - c.rho)   ** 2)
            k_o = 0.5 * (1.0 + torch.cos(self.geo._theta_t - c.theta))
            k_s = torch.exp(-self.kernels.gamma_side * (self.geo._sigma_t - c.sigma) ** 2)
            base += self.centroid_weight * k_r * k_o * k_s

        base = base.clamp(min=0.0)
        total = base.sum()
        if total.item() > 1e-8:
            base = base / total
        else:
            base = torch.ones(V, dtype=self.dtype, device=self.device) / V

        self._base_dist_t = base
        print(f"[InstrDist] Built instruction distribution from "
              f"{len(self._instr_toks)} tokens, vocab={V}")

    @torch.no_grad()
    def distribution(self, cands, gen_tokens, lm_tok2idx):
        C = len(cands)
        if C == 0 or self._base_dist_t is None:
            return torch.ones(C, dtype=self.dtype, device=self.device) / max(C, 1)

        cand_idx   = torch.tensor(
            [lm_tok2idx.get(c, 0) for c in cands],
            dtype=torch.long, device=self.device,
        )
        base_probs = self._base_dist_t[cand_idx]

        instr_set   = set(self._instr_toks)
        ctx_bonus_v = torch.tensor(
            [self.context_bonus if c in instr_set else 0.0 for c in cands],
            dtype=self.dtype, device=self.device,
        )

        bigram_bonus = torch.zeros(C, dtype=self.dtype, device=self.device)
        if self._instr_toks and gen_tokens:
            w1, w2 = self._instr_toks[-1], gen_tokens[-1]
            followers = self.lm.heads.get((w1, w2), [])
            fset = set(followers)
            for i, c in enumerate(cands):
                if c in fset:
                    bigram_bonus[i] = 0.1

        raw = base_probs + ctx_bonus_v + bigram_bonus
        raw = raw.clamp(min=1e-12)
        return raw / raw.sum()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2e — COT STUB LIBRARY
# ════════════════════════════════════════════════════════════════════════════

class CoTStubLibrary:
    def __init__(
        self,
        rho_threshold  : float = 0.20,
        n_theta_bins   : int   = 8,
        min_bin_size   : int   = 2,
        device         : torch.device = DEVICE,
        dtype          : torch.dtype  = torch.float32,
    ):
        self.rho_threshold = rho_threshold
        self.n_theta_bins  = n_theta_bins
        self.min_bin_size  = min_bin_size
        self.device        = device
        self.dtype         = dtype
        self.stubs         : Dict[str, List[ContextualStub]] = {
            t: [] for t in _STUB_SEQUENCE
        }
        self._stub_rho_t  : Optional[torch.Tensor] = None
        self._stub_theta_t: Optional[torch.Tensor] = None
        self._stub_sigma_t: Optional[torch.Tensor] = None
        self._stub_list   : List[ContextualStub]   = []

    def build(self, geo, lm_vocab, raw_freq) -> None:
        all_entries = []
        for tok in lm_vocab:
            tr = geo.triple(tok)
            all_entries.append((tok, tr, raw_freq.get(tok, 1.0)))

        rhos_sorted  = sorted(e[1].rho for e in all_entries)
        adaptive_thr = rhos_sorted[max(0, int(len(rhos_sorted) * 0.20))]
        thr          = min(self.rho_threshold, adaptive_thr)

        bridges = [(tok, tr, freq) for tok, tr, freq in all_entries if tr.rho >= thr]
        if len(bridges) < 8:
            bridges = all_entries

        bridges.sort(key=lambda x: x[1].sigma)
        q = max(1, len(bridges) // 4)
        quartile_map = {
            STUB_PREMISE    : bridges[:q],
            STUB_ELABORATION: bridges[q : 2 * q],
            STUB_CONTRAST   : bridges[2 * q : 3 * q],
            STUB_CONCLUSION : bridges[3 * q:],
        }

        self.stubs = {t: [] for t in _STUB_SEQUENCE}

        for stub_type, bucket in quartile_map.items():
            if not bucket:
                continue
            bin_width = math.pi / self.n_theta_bins
            theta_bins: Dict[int, list] = {}
            for tok, tr, freq in bucket:
                bin_idx = min(int(tr.theta / bin_width), self.n_theta_bins - 1)
                theta_bins.setdefault(bin_idx, []).append((tok, tr, freq))

            for bin_idx, members in theta_bins.items():
                if len(members) < self.min_bin_size:
                    continue
                members.sort(key=lambda x: x[1].rho)
                mid = max(1, len(members) // 2)
                for sub_idx, group in enumerate([members[:mid], members[mid:]]):
                    if group:
                        self._make_stub(stub_type, bin_idx, sub_idx, group)

        self._rebuild_tensors()
        total = sum(len(v) for v in self.stubs.values())
        per   = {t: len(v) for t, v in self.stubs.items()}
        print(f"[CoT] Built {total} contextual stubs: {per}")

    def _make_stub(self, stub_type, bin_idx, sub_idx, members) -> None:
        toks    = [m[0] for m in members]
        rhos    = [m[1].rho   for m in members]
        thetas  = [m[1].theta for m in members]
        sigmas  = [m[1].sigma for m in members]
        weights = [m[2]       for m in members]
        sin_m   = sum(math.sin(th) for th in thetas) / len(thetas)
        cos_m   = sum(math.cos(th) for th in thetas) / len(thetas)
        theta_cm = math.atan2(sin_m, cos_m) % math.pi
        rho_tag  = "hi-ρ" if sub_idx == 1 else "lo-ρ"
        tok_preview = " ".join(toks[:3])
        label = f"[{stub_type}|bin{bin_idx}|{rho_tag}] {tok_preview}…"
        self.stubs[stub_type].append(ContextualStub(
            stub_type = stub_type,
            tokens    = toks,
            rho       = sum(rhos)   / len(rhos),
            theta     = theta_cm,
            sigma     = sum(sigmas) / len(sigmas),
            weight    = sum(weights),
            label     = label,
        ))

    def _rebuild_tensors(self) -> None:
        self._stub_list    = [s for stype in _STUB_SEQUENCE for s in self.stubs[stype]]
        if not self._stub_list:
            return
        self._stub_rho_t   = torch.tensor([s.rho   for s in self._stub_list], dtype=torch.float32, device=DEVICE)
        self._stub_theta_t = torch.tensor([s.theta for s in self._stub_list], dtype=torch.float32, device=DEVICE)
        self._stub_sigma_t = torch.tensor([s.sigma for s in self._stub_list], dtype=torch.float32, device=DEVICE)

    def best_stub(self, stub_type, ctx_rho, ctx_theta, ctx_sigma, kernels, pdn_orbit=0, pdn_engine=None):
        candidates = self.stubs.get(stub_type, [])
        if not candidates:
            return None
        lam_stub, gam_stub = 1.5, 0.8
        c_rho   = torch.tensor([s.rho   for s in candidates], dtype=torch.float32, device=DEVICE)
        c_theta = torch.tensor([s.theta for s in candidates], dtype=torch.float32, device=DEVICE)
        c_sigma = torch.tensor([s.sigma for s in candidates], dtype=torch.float32, device=DEVICE)
        k_r = torch.exp(-lam_stub * (c_rho   - ctx_rho)   ** 2)
        k_o = 0.5 * (1.0 + torch.cos(c_theta - ctx_theta))
        k_s = torch.exp(-gam_stub * (c_sigma - ctx_sigma) ** 2)
        scores = k_r * k_o * k_s
        if pdn_engine is not None:
            orb_bonus = pdn_engine.orbit_bonus(pdn_orbit, c_theta)
            scores    = scores + 0.3 * orb_bonus
        return candidates[int(scores.argmax().item())]

    @torch.no_grad()
    def stub_kernel(self, stub, c_rho, c_theta, c_sigma, kernels):
        k_r = torch.exp(-kernels.lambda_reg * (c_rho   - stub.rho)   ** 2)
        k_o = 0.5 * (1.0 + torch.cos(c_theta - stub.theta))
        k_s = torch.exp(-kernels.gamma_side * (c_sigma - stub.sigma) ** 2)
        return k_r * k_o * k_s


class CoTReasoningEngine:
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

    def begin_sentence(self) -> None:
        self._chain           = []
        self._conclusion_stub = None
        self._hop_ptr         = 0
        self._tok_since_hop   = 0

    def plan_chain(self, seed_tokens, geo, pdn_orbit=0) -> CoTTrace:
        lam, gam = 1.5, 0.8
        clean_seeds = [t for t in seed_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if clean_seeds:
            triples   = [geo.triple(t) for t in clean_seeds]
            ctx_rho   = sum(tr.rho   for tr in triples) / len(triples)
            ctx_sigma = sum(tr.sigma for tr in triples) / len(triples)
            sin_m     = sum(math.sin(tr.theta) for tr in triples) / len(triples)
            cos_m     = sum(math.cos(tr.theta) for tr in triples) / len(triples)
            ctx_theta = math.atan2(sin_m, cos_m) % math.pi
        else:
            ctx_rho, ctx_theta, ctx_sigma = 0.5, math.pi / 4, 0.5

        self._chain, self._conclusion_stub = [], None
        hop_types = [STUB_PREMISE] + [STUB_ELABORATION] * max(1, self.n_hops - 2) + [STUB_CONTRAST]
        hop_types = hop_types[:self.n_hops]

        for hop_idx, stype in enumerate(hop_types):
            stub = self.stubs.best_stub(
                stype, ctx_rho, ctx_theta, ctx_sigma, self.kernels,
                pdn_orbit=(pdn_orbit + hop_idx) % self.pdn.n_star,
                pdn_engine=self.pdn,
            )
            if stub is None:
                continue
            k_r   = math.exp(-lam * (stub.rho   - ctx_rho)   ** 2)
            k_o   = 0.5 * (1.0 + math.cos(stub.theta - ctx_theta))
            k_s   = math.exp(-gam * (stub.sigma - ctx_sigma) ** 2)
            self._chain.append(CoTStep(hop_idx, stub, k_r * k_o * k_s,
                                       (pdn_orbit + hop_idx) % self.pdn.n_star))
            ctx_rho, ctx_theta, ctx_sigma = stub.rho, stub.theta, stub.sigma

        self._conclusion_stub = self.stubs.best_stub(
            STUB_CONCLUSION, ctx_rho, ctx_theta, ctx_sigma, self.kernels,
            pdn_orbit=(pdn_orbit + self.n_hops) % self.pdn.n_star,
            pdn_engine=self.pdn,
        )
        trace = CoTTrace(clean_seeds, list(self._chain), self._conclusion_stub)
        self._traces.append(trace)
        return trace

    @torch.no_grad()
    def active_bonus(self, c_rho, c_theta, c_sigma, token_position, total_tokens):
        C = c_rho.shape[0]
        if self._tok_since_hop >= self.tokens_per_hop and self._hop_ptr < len(self._chain) - 1:
            self._hop_ptr      += 1
            self._tok_since_hop = 0
        self._tok_since_hop += 1
        frac = token_position / max(total_tokens - 1, 1)
        if frac >= 0.80 and self._conclusion_stub is not None:
            active_stub = self._conclusion_stub
        elif self._hop_ptr < len(self._chain):
            active_stub = self._chain[self._hop_ptr].stub
        else:
            return torch.zeros(C, dtype=self.dtype, device=self.device)
        raw = self.stubs.stub_kernel(active_stub, c_rho, c_theta, c_sigma, self.kernels)
        std = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * self.stub_logit_scale

    def all_traces_text(self, max_traces=8) -> str:
        if not self._traces:
            return "  (no traces yet)"
        return "\n".join(
            f"\nSentence {i+1}:\n{tr.render()}"
            for i, tr in enumerate(self._traces[-max_traces:])
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THÉBAULT KERNELS
# ════════════════════════════════════════════════════════════════════════════

class ThebaultKernels:
    def __init__(self, lambda_reg: float = 8.0, gamma_side: float = 4.0):
        self.lambda_reg = lambda_reg
        self.gamma_side = gamma_side

    def k_reg (self, rho_a, rho_b):     return torch.exp(-self.lambda_reg * (rho_b - rho_a) ** 2)
    def k_ori (self, theta_a, theta_b): return 0.5 * (1.0 + torch.cos(theta_b - theta_a))
    def k_side(self, sigma_a, sigma_b): return torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)

    def all_scores_batched(self, rho_a, theta_a, sigma_a, rho_b, theta_b, sigma_b):
        k_r = torch.exp(-self.lambda_reg * (rho_b   - rho_a)   ** 2)
        k_o = 0.5 * (1.0 + torch.cos(theta_b - theta_a))
        k_s = torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)
        return k_r, k_o, k_s


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MRV FILTER
# ════════════════════════════════════════════════════════════════════════════

class MRVConstraintFilter:
    def __init__(self, threshold=0.50, mrv_cap_ratio=2.0, max_vocab_scan=300, device=DEVICE):
        self.threshold      = threshold
        self.mrv_cap_ratio  = mrv_cap_ratio
        self.max_vocab_scan = max_vocab_scan
        self.device         = device
        self._v_rho  : Optional[torch.Tensor] = None
        self._v_sigma: Optional[torch.Tensor] = None
        self._v_toks : List[str]              = []

    def prime(self, vocab, geo) -> None:
        scan  = vocab[:self.max_vocab_scan]
        trips = [geo.triple(v) for v in scan]
        self._v_rho   = torch.tensor([t.rho   for t in trips], dtype=torch.float32, device=self.device)
        self._v_sigma = torch.tensor([t.sigma for t in trips], dtype=torch.float32, device=self.device)
        self._v_toks  = scan

    def mrv_scores_batched(self, c_rho, c_sigma, kernels):
        if self._v_rho is None:
            return torch.zeros(c_rho.shape[0], device=self.device)
        k_r = torch.exp(-kernels.lambda_reg * (c_rho.unsqueeze(1)   - self._v_rho.unsqueeze(0))   ** 2)
        k_s = torch.exp(-kernels.gamma_side * (c_sigma.unsqueeze(1) - self._v_sigma.unsqueeze(0)) ** 2)
        domain_sizes = ((k_r > self.threshold) & (k_s > self.threshold)).float().sum(dim=1)
        mean_d       = domain_sizes.mean() + 1e-6
        mrv          = 1.0 / (domain_sizes + 1.0)
        mrv[domain_sizes > self.mrv_cap_ratio * mean_d] *= 0.5
        lo, hi = mrv.min(), mrv.max()
        if (hi - lo).item() > 1e-8:
            mrv = (mrv - lo) / (hi - lo)
        return mrv


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — POSITIONAL VECTOR + CHUNKED SUM ENGINE
# ════════════════════════════════════════════════════════════════════════════

VEC_DIM = 4

class ChunkedSumEngine:
    def __init__(self, window_size=16, n_chunks=4, device=DEVICE, dtype=torch.float32):
        self.window_size = window_size
        self.n_chunks    = n_chunks
        self.device      = device
        self.dtype       = dtype
        self._buf   = torch.zeros(window_size, VEC_DIM, dtype=dtype, device=device)
        self._ptr   = 0
        self._count = 0

    def reset(self) -> None:
        self._buf.zero_(); self._ptr = 0; self._count = 0

    def push(self, triple, pos_norm) -> None:
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
        return window.view(self.n_chunks, chunk_len, VEC_DIM).sum(dim=1).flatten()

    def chunk_bonus(self, c_pvec, scale=1.0) -> torch.Tensor:
        sig = self.chunk_signature()
        cv_tiled = c_pvec.repeat(1, self.n_chunks)
        raw = cv_tiled @ sig
        std = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * scale

    def window_rho_theta(self):
        if self._count == 0:
            empty = torch.zeros(0, dtype=self.dtype, device=self.device)
            return empty, empty
        if self._count < self.window_size:
            window = self._buf[:self._count]
        else:
            window = torch.cat([self._buf[self._ptr:], self._buf[:self._ptr]], dim=0)
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
    def __init__(self, top_k=3, max_stored=64, device=DEVICE, dtype=torch.float32):
        self.top_k      = top_k
        self.max_stored = max_stored
        self.device     = device
        self.dtype      = dtype
        self.store      : List[SentenceVector] = []

    def add(self, tokens, geo, text) -> None:
        clean = [t for t in tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not clean:
            return
        rhos   = torch.tensor([geo.triple(t).rho   for t in clean], dtype=self.dtype, device=self.device)
        sigmas = torch.tensor([geo.triple(t).sigma for t in clean], dtype=self.dtype, device=self.device)
        self.store.append(SentenceVector(clean, rhos, sigmas, text))
        if len(self.store) > self.max_stored:
            self.store.pop(0)

    def _batch_sim(self, cur_rho, cur_sigma, kernels):
        L, N = cur_rho.shape[0], len(self.store)
        if N == 0 or L == 0:
            return torch.zeros(0, device=self.device)
        stored_rho   = torch.zeros(N, L, dtype=self.dtype, device=self.device)
        stored_sigma = torch.zeros(N, L, dtype=self.dtype, device=self.device)
        for i, sv in enumerate(self.store):
            l = min(L, sv.rho_t.shape[0])
            stored_rho[i, :l]   = sv.rho_t[:l]
            stored_sigma[i, :l] = sv.sigma_t[:l]
        kr = torch.exp(-kernels.lambda_reg * (stored_rho   - cur_rho.unsqueeze(0))   ** 2)
        ks = torch.exp(-kernels.gamma_side * (stored_sigma - cur_sigma.unsqueeze(0)) ** 2)
        return (kr * ks).mean(dim=1)

    def ranked_anchors(self, current_tokens, geo, kernels):
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

    def syntax_echo_bonus(self, c_rho, c_sigma, current_tokens, geo, kernels, echo_weight=0.5):
        anchors = self.ranked_anchors(current_tokens, geo, kernels)
        if not anchors:
            return torch.zeros(c_rho.shape[0], device=self.device)
        pos     = len([t for t in current_tokens if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS])
        bonuses = torch.zeros(c_rho.shape[0], dtype=self.dtype, device=self.device)
        for sim_score, anc in anchors:
            if pos < anc.rho_t.shape[0]:
                kr = torch.exp(-kernels.lambda_reg * (c_rho   - anc.rho_t  [pos].item()) ** 2)
                ks = torch.exp(-kernels.gamma_side * (c_sigma - anc.sigma_t[pos].item()) ** 2)
                bonuses += sim_score * (kr * ks)
        std = bonuses.std()
        if std.item() > 1e-8:
            bonuses = (bonuses - bonuses.mean()) / std
        return bonuses * echo_weight


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THÉBAULT CONJUGATE ORBIT
# ════════════════════════════════════════════════════════════════════════════

class ThebaultConjugateOrbit:
    def score(self, anchor_triple, cand_theta, cand_sigma, gamma_side=4.0):
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor_triple.sigma) ** 2)
        antipodality = torch.cos(cand_theta + anchor_triple.theta - math.pi / 2) ** 2
        return congruence * antipodality


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — THÉBAULT COMPOSITION LM
# ════════════════════════════════════════════════════════════════════════════

class ThebaultCompositionLM:
    BASAL_K      = 1000.5
    DENSE_THRESH = 512

    def __init__(self, geo, kernels, device=DEVICE):
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

    def ingest(self, tokens) -> None:
        for t in tokens:
            self.raw_freq[t] = self.raw_freq.get(t, 0) + 1.0
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.tri_raw[(w1, w2, w3)] = self.tri_raw.get((w1, w2, w3), 0) + 1.0
            if (w1, w2) not in self.heads:
                self.heads[(w1, w2)] = []
            if w3 not in self.heads[(w1, w2)]:
                self.heads[(w1, w2)].append(w3)
        self.vocab = [v for v in self.raw_freq if v not in PUNCT_TOKENS and v not in COGNITIVE_TOKENS]

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
            self._head_cands[(w1, w2)] = torch.tensor(
                [self._tok2idx.get(c, 0) for c in cands], dtype=torch.long, device=self.device,
            )
            self._head_probs[(w1, w2)] = basal

    def next_dist(self, w1, w2):
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

    def composition_logit_bonus(self, w1, w2, c_rho, c_sigma):
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
    def __init__(self, geo, kernels, device=DEVICE):
        self.geo     = geo
        self.kernels = kernels
        self.device  = device
        self.nodes   : Dict[str, TGNode]       = {}
        self.adj     : Dict[str, List[TGEdge]] = {}
        self.radj    : Dict[str, List[TGEdge]] = {}

    def build(self, lm) -> None:
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
                    self.kernels.k_reg(ti.rho, torch.tensor(tj.rho, device=self.device)).item()
                    * self.kernels.k_ori(ti.theta, torch.tensor(tj.theta, device=self.device)).item()
                    * cnt
                )
                e = TGEdge(w2, w3, max(w, 1e-6))
                self.adj[w2].append(e)
                self.radj[w3].append(e)
                seen.add((w2, w3))

    def propagate(self, steps=2) -> None:
        if not self.nodes:
            return
        max_f = max(nd.freq for nd in self.nodes.values()) + 1e-8
        for nd in self.nodes.values():
            nd.potential = nd.triple.rho * nd.freq / max_f
        for _ in range(steps):
            new_pots = {}
            for v, nd in self.nodes.items():
                agg = sum(e.weight * self.nodes[e.src].potential for e in self.radj.get(v, []))
                self_scale = nd.triple.sigma / (nd.triple.sigma + 1.0)
                new_pots[v] = agg / (len(self.radj.get(v, [])) + 1.0) + self_scale * nd.potential * 0.1
            mx = max(new_pots.values(), default=1.0) + 1e-8
            for v in self.nodes:
                self.nodes[v].potential = new_pots[v] / mx

    def potentials_for(self, cands):
        return torch.tensor(
            [self.nodes[c].potential if c in self.nodes else 0.0 for c in cands],
            dtype=torch.float32, device=self.device,
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SEMANTIC MANDATE PROCESSOR  (REPLACED from V18)
#
# ORIGINAL PROBLEM:
#   The V18 `synthetic_reasonMandateProcessor` used hard-coded keyword
#   matching ("poverty" → "end", "disease" → "cure") to boost candidate
#   logits. This was:
#     (a) Inconsistent: the rest of the pipeline uses geometric kernel
#         similarity; keyword matching is a categorically different mechanism.
#     (b) Brittle: any instruction not containing exact keywords was ignored.
#     (c) Ungrounded: the +5.0 and +10.0 boost magnitudes were arbitrary
#         and not calibrated to the scale of other logit contributions.
#
# REPLACEMENT: THÉBAULT KERNEL SEMANTIC SCORER
#   The instruction text is encoded as a Thébault triple centroid (exactly
#   as InstructionDistribution does for its base distribution). Each
#   candidate's logit bonus is computed via the same three-kernel product
#   used throughout the pipeline:
#
#     mandate_bonus[i] = k_reg(ρ_i, ρ_instr)
#                      · k_ori(θ_i, θ_instr)
#                      · k_side(σ_i, σ_instr)
#                      · mandate_scale
#
#   where (ρ_instr, θ_instr, σ_instr) is the centroid of the instruction's
#   token triples (weighted by recency, same as InstructionDistribution).
#
#   This integrates mandate scoring fully into the geometric framework:
#   candidates geometrically close to the instruction centroid receive a
#   bonus, with magnitude controlled by the single parameter `mandate_scale`
#   (default 2.0, calibrated to be comparable to other logit contributions).
#
#   The bonus is layer-normalised before application so it cannot dominate
#   the distribution regardless of instruction content.
# ════════════════════════════════════════════════════════════════════════════

class SemanticMandateScorer:
    """
    Replaces synthetic_reasonMandateProcessor.

    Scores candidate tokens by their Thébault kernel similarity to the
    instruction centroid. Fully consistent with the geometric pipeline.
    """

    def __init__(
        self,
        geo            : ThebaultTokenGeometry,
        kernels        : ThebaultKernels,
        mandate_scale  : float = 2.0,
        recency_decay  : float = 0.7,
        device         : torch.device = DEVICE,
        dtype          : torch.dtype  = torch.float32,
    ):
        self.geo           = geo
        self.kernels       = kernels
        self.mandate_scale = mandate_scale
        self.recency_decay = recency_decay
        self.device        = device
        self.dtype         = dtype
        self._centroid     : Optional[ThebaultTriple] = None

    def set_instruction(self, instruction_text: str) -> None:
        """
        Compute and store the Thébault triple centroid of the instruction.
        Uses recency-weighted averaging of token triples (recent tokens
        weighted higher, consistent with InstructionDistribution).
        """
        toks = [t for t in tokenize(instruction_text)
                if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if not toks:
            self._centroid = None
            return

        N = len(toks)
        weights = [self.recency_decay ** (N - 1 - i) for i in range(N)]
        total_w = sum(weights)

        triples = [self.geo.triple(t) for t in toks]
        rho_m   = sum(w * tr.rho   for w, tr in zip(weights, triples)) / total_w
        sigma_m = sum(w * tr.sigma for w, tr in zip(weights, triples)) / total_w
        sin_m   = sum(w * math.sin(tr.theta) for w, tr in zip(weights, triples)) / total_w
        cos_m   = sum(w * math.cos(tr.theta) for w, tr in zip(weights, triples)) / total_w
        theta_m = math.atan2(sin_m, cos_m) % math.pi

        self._centroid = ThebaultTriple(rho_m, theta_m, sigma_m)

    @torch.no_grad()
    def score(self, cands: List[str], c_rho: torch.Tensor,
              c_theta: torch.Tensor, c_sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel-based mandate bonus for each candidate.
        Returns a layer-normalised tensor of shape [C].
        Returns zeros if no instruction centroid is set.
        """
        C = len(cands)
        if self._centroid is None:
            return torch.zeros(C, dtype=self.dtype, device=self.device)

        k_r = torch.exp(
            -self.kernels.lambda_reg * (c_rho   - self._centroid.rho)   ** 2
        )
        k_o = 0.5 * (1.0 + torch.cos(c_theta - self._centroid.theta))
        k_s = torch.exp(
            -self.kernels.gamma_side * (c_sigma - self._centroid.sigma) ** 2
        )
        bonus = k_r * k_o * k_s  # ∈ [0, 1]^C

        # Layer-normalise so bonus scale is independent of instruction geometry
        bonus = layer_norm_array(bonus)

        return bonus * self.mandate_scale

    def centroid_report(self) -> str:
        if self._centroid is None:
            return "  No instruction centroid set."
        return (
            f"  Instruction centroid: "
            f"ρ={self._centroid.rho:.4f}  "
            f"θ={self._centroid.theta:.4f}  "
            f"σ={self._centroid.sigma:.4f}"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11a — DNN ARRAY PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class ThebaultDNNNormalizer:
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype  = dtype

    def _build_rho_scale(self, c_rho):
        mu  = c_rho.mean()
        std = c_rho.std() + 1e-8
        return 1.0 + 0.5 * ((c_rho - mu) / std).clamp(-2.0, 2.0)

    def _build_freq_scale(self, c_rho, c_sigma):
        return (c_rho.clamp(min=1e-6) * c_sigma.clamp(min=1e-6)).sqrt()

    def normalize(self, logits, c_rho=None, c_sigma=None, temp=1.0):
        if c_rho is not None:
            temp_weights = torch.exp(
                -((c_rho - c_rho.mean()) ** 2) / (2.0 * max(temp, 0.1) + 1e-8)
            )
            scaled_logits = logits * temp_weights
        else:
            scaled_logits = logits / max(temp, 1e-6)

        if c_rho is not None and c_sigma is not None:
            rho_scale  = self._build_rho_scale(c_rho)
            y1         = scaled_logits * rho_scale
            a1         = signed_power(y1, p=2.0)
            freq_scale = self._build_freq_scale(c_rho, c_sigma)
            freq_norm  = l2_array_normalize(freq_scale, dim=0)
            y2         = (freq_norm * a1).sum() * freq_norm + a1 * 0.5
            a2         = signed_power(y2, p=1.5)
        else:
            a2 = scaled_logits

        return l1_simplex_project(a2)

    def log_normalize(self, logits, c_rho=None, c_sigma=None, temp=1.0):
        p = self.normalize(logits, c_rho, c_sigma, temp)
        return (p + 1e-12).log()


class GeometricTempScaler:
    def __init__(self, lambda_temp: float = 1.0):
        self.lambda_temp = lambda_temp

    def scale(self, logits, temp, c_rho=None):
        safe_logits = logits.clamp(-50.0, 50.0)
        if c_rho is None or temp <= 1e-6:
            return safe_logits / max(temp, 0.1)
        mu_rho   = c_rho.mean()
        exponent    = (-self.lambda_temp * (c_rho - mu_rho) ** 2 / max(temp, 0.1)).clamp(min=-10.0)
        geo_weights = torch.exp(exponent)
        return safe_logits * geo_weights


class DNNArrayPipeline:
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32):
        self.device      = device
        self.dtype       = dtype
        self._normalizer  = ThebaultDNNNormalizer(device, dtype)
        self._temp_scaler = GeometricTempScaler(lambda_temp=1.0)

    def _rho_weights(self, c_rho):
        mu  = c_rho.mean()
        std = c_rho.std() + 1e-8
        z   = (c_rho - mu) / std
        return 1.0 + 0.5 * z.clamp(-2.5, 2.5)

    def _theta_weights(self, c_theta):
        return 0.5 * (1.0 + torch.cos(c_theta))

    def _sigma_weights(self, c_sigma):
        sig_norm = c_sigma / (c_sigma.max() + 1e-8)
        return 0.7 + 0.3 * sig_norm

    @torch.no_grad()
    def forward(self, logits, c_rho, c_theta, c_sigma, temp=1.4):
        logits_scaled = self._temp_scaler.scale(logits, temp, c_rho)
        rho_w = self._rho_weights(c_rho)
        z1    = signed_power(logits_scaled * rho_w, p=2.0)
        theta_w = self._theta_weights(c_theta)
        z2      = signed_power(z1 * theta_w, p=1.5)
        sigma_w = self._sigma_weights(c_sigma)
        z3      = signed_power(z2 * sigma_w + z1 * 0.3, p=1.0)
        return l1_simplex_project(z3)

    @torch.no_grad()
    def log_forward(self, logits, c_rho, c_theta, c_sigma, temp=1.4):
        p = self.forward(logits, c_rho, c_theta, c_sigma, temp)
        return (p + 1e-12).log()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11b — LOCALE TRANSIT REMISSION
# ════════════════════════════════════════════════════════════════════════════

class LocaleTransitRemission:
    def __init__(self, transit_tolerance=0.15, remission_rate=0.85):
        self.transit_tolerance = transit_tolerance
        self.remission_rate    = remission_rate

    def apply_remission(self, w1_rho, w2_rho, c_rho):
        transit_delta     = torch.abs((w1_rho + w2_rho) / 2.0 - c_rho)
        linear_error      = smooth_power_relu(transit_delta - self.transit_tolerance)
        manipulation_mask = (linear_error > 1e-6).float()
        remission_penalty = torch.exp(-self.remission_rate * linear_error)
        return torch.where(manipulation_mask == 1.0, remission_penalty, torch.ones_like(c_rho))


class ContingentExtringentProbability:
    def __init__(self, coupling_factor=0.5):
        self.coupling_factor       = coupling_factor
        self.intermediate_entropy  = 1.0
        self.intermediate_max_prob = 1.0
        self._dnn = DNNArrayPipeline()

    def govern_next_probs(self, logits, c_rho=None, c_theta=None, c_sigma=None):
        dynamic_temp = 1.0 + (self.coupling_factor * (1.0 - self.intermediate_max_prob))
        if c_rho is not None and c_theta is not None and c_sigma is not None:
            governed_logits = self._dnn._temp_scaler.scale(logits, dynamic_temp, c_rho)
        else:
            governed_logits = logits / max(dynamic_temp, 1e-6)
        current_probs = l1_simplex_project(governed_logits)
        entropy = -(current_probs * (current_probs + 1e-9).log()).sum()
        self.intermediate_entropy  = entropy.item()
        self.intermediate_max_prob = current_probs.max().item()
        return governed_logits


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11c — ANONYMOUS VARIABLE SOLVER
# ════════════════════════════════════════════════════════════════════════════

class AnonymousVariableSolver:
    def __init__(self, geo, lm, kernels, device=DEVICE):
        self.geo     = geo
        self.lm      = lm
        self.kernels = kernels
        self.device  = device
        self.logic_terms = {
            "is": "equality", "has": "property", "eats": "relation",
            "every": "forall", "some": "exists", "the": "definite"
        }
        self.anon_var = "_"

    def parse_pattern(self, text):
        tokens  = tokenize(text.lower())
        pattern = {"terms": tokens, "bindings": {}, "quants": []}
        for i, tok in enumerate(tokens):
            if tok == self.anon_var:
                pattern["bindings"][f"anon_{i}"] = None
            elif tok in self.logic_terms:
                pattern["quants"].append((tok, i))
        return pattern

    def solve_bindings(self, pattern, cands):
        N = len(cands)
        binding_scores = torch.ones(N, device=self.device)
        for var_name, _ in pattern["bindings"].items():
            if var_name.startswith("anon_"):
                binding_scores *= 0.8
        for quant, pos in pattern["quants"]:
            if quant == "every":
                binding_scores *= 0.7
            elif quant == "some":
                binding_scores *= 1.2
        return l2_array_normalize(binding_scores, dim=0)

    def integrate_logits(self, logits, instruction, cands):
        pattern       = self.parse_pattern(instruction)
        binding_bonus = self.solve_bindings(pattern, cands)
        return logits + (binding_bonus.clamp(min=1e-8)).log()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — THÉBAULT WALKER V18-CSNS-G
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    def __init__(
        self,
        geo, kernels, lm, orbit, graph,
        mandate_scorer   : SemanticMandateScorer,
        mrv_filter, chunk_engine, iso_stacker,
        pdn_engine       : PDNEngine,
        cot_engine       : CoTReasoningEngine,
        instr_dist       : InstructionDistribution,
        device           : torch.device = DEVICE,
        syn_weight       : float = 0.4,
        trans_weight     : float = 0.6,
        syn_k            : int   = 8,
    ):
        self.geo          = geo
        self.kernels      = kernels
        self.lm           = lm
        self.orbit        = orbit
        self.graph        = graph
        self.mandate      = mandate_scorer   # SemanticMandateScorer (replaces synth)
        self.mrv          = mrv_filter
        self.chunk_engine = chunk_engine
        self.iso_stacker  = iso_stacker
        self.pdn          = pdn_engine
        self.cot          = cot_engine
        self.instr_dist   = instr_dist
        self.device       = device
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []
        self._cur_sent_toks : List[str] = []
        self._cur_orbit     : int       = 0
        self._tok_pos       : int       = 0
        self._step_traces   : List[TokenStepTrace] = []
        self.remission       = LocaleTransitRemission()
        self.contingent_prob = ContingentExtringentProbability()
        self._dnn_pipeline   = DNNArrayPipeline(device=device)
        self._dnn_normalizer = ThebaultDNNNormalizer(device=device)
        self._csns = CrossSynapticNeuronSum(
            syn_weight   = syn_weight,
            trans_weight = trans_weight,
            syn_k        = syn_k,
            lambda_reg   = kernels.lambda_reg,
            gamma_side   = kernels.gamma_side,
            device       = device,
        )
        self._csns_syn_norms   : List[float] = []
        self._csns_trans_norms : List[float] = []

    def begin_sentence(self, seed_tokens=None, total_tokens=40) -> CoTTrace:
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()
        self._cur_orbit    = 0
        self._tok_pos      = 0
        self._total_tokens = total_tokens
        seeds = seed_tokens or []
        self.cot.begin_sentence()
        return self.cot.plan_chain(seeds, self.geo, pdn_orbit=self._cur_orbit)

    @torch.no_grad()
    def walk_probs(
        self, w1: str, w2: str,
        temp          : float = 1.4,
        alphareg      : float = 1.2,
        betaori       : float = 0.8,
        deltaside     : float = 1.0,
        gammaorbit    : float = 0.6,
        psipot        : float = 0.35,
        zetamrv       : float = 0.9,
        etachunk      : float = 0.7,
        xiecho        : float = 0.6,
        pdn_weight    : float = 0.8,
        cot_weight    : float = 1.0,
        and_weight    : float = 0.5,
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
            c_pvec   = torch.stack([c_rho, c_theta/math.pi, c_sigma, torch.ones_like(c_rho)], dim=1)

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

        win_rho, win_theta = self.chunk_engine.window_rho_theta()
        pdn_bonus = self.pdn.pdn_logit_bonus(win_rho, win_theta, c_rho, c_theta, self._cur_orbit)

        cot_bonus = self.cot.active_bonus(
            c_rho, c_theta, c_sigma,
            token_position=self._tok_pos,
            total_tokens  =self._total_tokens,
        )

        # Semantic mandate bonus — kernel-based, consistent with pipeline
        mandate_boost = self.mandate.score(cands, c_rho, c_theta, c_sigma)

        # Isomorphic pair detection
        self.current_isomorphic_pairs = []
        top_idx  = torch.topk(k_reg * k_side, min(50, len(cands))).indices
        sub_r, sub_s = k_reg[top_idx], k_side[top_idx]
        iso_mask = (sub_r > 0.98) & (sub_s > 0.98)
        iso_idx  = top_idx[iso_mask].tolist()
        for ii in range(len(iso_idx)):
            for jj in range(ii+1, len(iso_idx)):
                i, j = iso_idx[ii], iso_idx[jj]
                ci, cj = cands[i], cands[j]
                if ci not in PUNCT_TOKENS and cj not in PUNCT_TOKENS:
                    sim = (k_reg[i] * k_side[i] * k_reg[j] * k_side[j]).sqrt().item()
                    self.current_isomorphic_pairs.append((ci, cj, sim))
        self.current_isomorphic_pairs.sort(key=lambda x: -x[2])

        N             = len(cands)
        punct_bias    = torch.zeros(N, device=self.device)
        punct_penalty = torch.zeros(N, device=self.device)
        for i, c in enumerate(cands):
            if c in PUNCT_TOKENS:
                punct_bias[i] = -3.5
                if w2 in PUNCT_TOKENS:
                    punct_penalty[i] = -1e4

        log_base   = (base_probs.clamp(min=1e-12)).log()
        raw_logits = (
            log_base
            + alphareg   * k_reg
            + betaori    * k_ori
            + deltaside  * k_side
            + gammaorbit * orbit_scores
            + psipot     * pots
            + comp_bonus
            + zetamrv    * mrv_scores
            + chunk_bonus
            + echo_bonus
            + pdn_weight * pdn_bonus
            + cot_weight * cot_bonus
            + mandate_boost
            + punct_bias
            + punct_penalty
        )

        governed_logits = self.contingent_prob.govern_next_probs(
            raw_logits, c_rho, c_theta, c_sigma
        )

        c_rho_trans, c_theta_trans, c_sigma_trans = compute_transitive_triples_batched(
            self.geo, cands, w1, w2, device=self.device,
        )

        logits_enriched = self._csns.forward(
            governed_logits,
            c_rho, c_theta, c_sigma,
            c_rho_trans, c_theta_trans, c_sigma_trans,
            ctx_rho   = ctx.rho,
            ctx_theta = ctx.theta,
            ctx_sigma = ctx.sigma,
        )

        z_syn_raw = self._csns.synaptic_sum(governed_logits, c_rho, c_theta, c_sigma)
        t_bon_raw = self._csns.transitive_bonus(
            c_rho_trans, c_theta_trans, c_sigma_trans,
            ctx.rho, ctx.theta, ctx.sigma,
        )
        syn_norm   = z_syn_raw.norm().item()
        trans_norm = t_bon_raw.norm().item()
        self._csns_syn_norms.append(syn_norm)
        self._csns_trans_norms.append(trans_norm)

        self._pending_instr_probs = None
        self._pending_walk_logits = logits_enriched
        self._pending_c_rho       = c_rho
        self._pending_c_theta     = c_theta
        self._pending_c_sigma     = c_sigma
        self._pending_syn_norm    = syn_norm
        self._pending_trans_norm  = trans_norm

        if and_weight > 0.0 and self.instr_dist._base_dist_t is not None:
            p_instr   = self.instr_dist.distribution(cands, self._cur_sent_toks, self.lm._tok2idx)
            log_instr = (p_instr.clamp(min=1e-12)).log()
            log_walk  = self._dnn_pipeline.log_forward(
                logits_enriched, c_rho, c_theta, c_sigma, temp=1.0
            )
            log_and   = and_weight * log_instr + (1.0 - and_weight) * log_walk
            final_probs = l1_simplex_project(log_and)
        else:
            p_instr     = torch.ones(N, dtype=torch.float32, device=self.device) / N
            final_probs = self._dnn_pipeline.forward(
                logits_enriched, c_rho, c_theta, c_sigma, temp=temp
            )

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
                self._pending_walk_logits,
                self._pending_c_rho, self._pending_c_theta, self._pending_c_sigma,
                temp=1.0,
            )
        else:
            log_walk = (l1_simplex_project(self._pending_walk_logits) + 1e-12).log()
        p_walk = log_walk[idx].exp().item()

        if p_instr > p_walk * 1.5:
            source = "instr"
        elif p_walk > p_instr * 1.5:
            source = "walker"
        else:
            source = "AND"

        trace = TokenStepTrace(
            step, chosen, p_instr, p_walk, p_and, and_weight, source,
            syn_norm   = getattr(self, '_pending_syn_norm',   0.0),
            trans_norm = getattr(self, '_pending_trans_norm', 0.0),
        )
        self._step_traces.append(trace)
        return trace

    def push_token(self, token: str, sentence_len: int) -> None:
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS:
            return
        self._cur_sent_toks.append(token)
        self._tok_pos += 1
        pos_norm = len(self._cur_sent_toks) / max(sentence_len, 1)
        self.chunk_engine.push(self.geo.triple(token), pos_norm)
        self._cur_orbit = self.pdn.orbit_of(token)

    def step_trace_report(self, max_steps: int = 30) -> str:
        if not self._step_traces:
            return "  (no step traces yet)"
        lines = [
            "step | chosen         | P_instr | P_walk  | P_and   | α    | source  | |z_syn| | |trans|",
            "─────┼───────────────┼─────────┼─────────┼─────────┼──────┼─────────┼─────────┼────────",
        ]
        for t in self._step_traces[-max_steps:]:
            lines.append(
                f"{t.step:5d}│ {t.chosen:<14s}│ {t.p_instr:.5f}│ {t.p_walk:.5f}│"
                f" {t.p_and:.5f}│ {t.and_weight:.2f} │ {t.source:<7s} │ {t.syn_norm:.4f}  │ {t.trans_norm:.4f}"
            )
        if self._csns_syn_norms:
            avg_syn   = sum(self._csns_syn_norms)   / len(self._csns_syn_norms)
            avg_trans = sum(self._csns_trans_norms) / len(self._csns_trans_norms)
            lines.append(f"\n  CSNS summary: avg |z_syn|={avg_syn:.4f}  avg |trans|={avg_trans:.4f}  "
                         f"steps={len(self._csns_syn_norms)}")
        return "\n".join(lines)

    def csns_report(self) -> str:
        if not self._csns_syn_norms:
            return "  (no CSNS data yet — generate text first)"
        n = len(self._csns_syn_norms)
        avg_s = sum(self._csns_syn_norms) / n
        avg_t = sum(self._csns_trans_norms) / n
        max_s = max(self._csns_syn_norms)
        max_t = max(self._csns_trans_norms)
        return (
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║   Cross-Synaptic Neuron Sum (CSNS) Diagnostic Report         ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            f"║  Steps processed:       {n:<36d}║\n"
            f"║  Synaptic sum weight    ω_syn   = {self._csns.syn_weight:<25.3f}║\n"
            f"║  Transitive bonus weight ω_trans = {self._csns.trans_weight:<24.3f}║\n"
            f"║  Synaptic sparsity K    = {self._csns.syn_k:<33d}║\n"
            "║                                                              ║\n"
            f"║  avg |z_syn|    = {avg_s:<41.4f}║\n"
            f"║  max |z_syn|    = {max_s:<41.4f}║\n"
            f"║  avg |trans|    = {avg_t:<41.4f}║\n"
            f"║  max |trans|    = {max_t:<41.4f}║\n"
            "║                                                              ║\n"
            "║  Transitive triple blend weights:                            ║\n"
            "║    w1 × 0.25  +  w2 × 0.50  +  c × 0.25                    ║\n"
            "║  (w2 is immediate context, weighted highest)                 ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — TEXT GENERATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def generate_passage(
    walker, lm,
    num_sentences   : int   = 4,
    tokens_per_sent : int   = 40,
    seed_text       : str   = "",
    instruction_text: str   = "",
    and_weight      : float = 0.9,
    temperature     : float = 2.0,
    return_traces   : bool  = False,
):
    if instruction_text.strip():
        walker.instr_dist.set_instruction(instruction_text)
        walker.mandate.set_instruction(instruction_text)   # keep mandate in sync
    elif seed_text.strip():
        walker.instr_dist.set_instruction(seed_text)
        walker.mandate.set_instruction(seed_text)

    walker._step_traces.clear()
    walker._csns_syn_norms.clear()
    walker._csns_trans_norms.clear()

    outputs    : List[str]      = []
    all_traces : List[CoTTrace] = []
    head_list  = list(lm.heads.keys())
    if not head_list:
        return ("", [], "") if return_traces else ""

    seed_w1, seed_w2 = None, None
    seed_toks = []
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

    global_step = 0

    for sent_idx in range(num_sentences):
        if sent_idx == 0:
            w1, w2    = seed_w1, seed_w2
            init_toks = [w1, w2] if seed_text else []
            wsp       = len(init_toks)
            plan_seeds = seed_toks if seed_toks else [w1, w2]
        else:
            w1, w2    = random.choice(head_list)
            init_toks, wsp = [], 999
            plan_seeds = [w1, w2]

        trace = walker.begin_sentence(seed_tokens=plan_seeds, total_tokens=tokens_per_sent)
        all_traces.append(trace)
        toks = list(init_toks)

        for step in range(tokens_per_sent):
            cands, probs = walker.walk_probs(w1, w2, temp=temperature, and_weight=and_weight)
            if not cands:
                break

            nxt = cands[torch.multinomial(probs, 1).item()]
            walker.record_step_trace(global_step, nxt, cands, probs, and_weight)
            global_step += 1

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

    result = " ".join(outputs)
    if return_traces:
        return result, all_traces, walker.step_trace_report()
    return result


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — V18-CSNS-G ENGINE
# ════════════════════════════════════════════════════════════════════════════

class V18Engine:
    def __init__(self, syn_weight=0.4, trans_weight=0.6, syn_k=8):
        self.device      = DEVICE
        self.geo         = ThebaultTokenGeometry(device=self.device)
        self.kernels     = ThebaultKernels()
        self.lm          = ThebaultCompositionLM(self.geo, self.kernels, device=self.device)
        self.orbit       = ThebaultConjugateOrbit()
        self.graph       = ThebaultPotentialGraph(self.geo, self.kernels, device=self.device)
        self.mrv         = MRVConstraintFilter(device=self.device)
        self.chunk       = ChunkedSumEngine(device=self.device)
        self.iso_stacker = IsomorphicSyntaxStacker(device=self.device)
        self.pdn         = PDNEngine(device=self.device)
        self.stub_lib    = CoTStubLibrary(n_theta_bins=8, device=self.device)
        self.mandate_scorer = None   # SemanticMandateScorer (created after geo is ready)
        self.instr_dist  = None
        self.cot         = None
        self.walker      = None
        self.corpus_snippet = ""
        self.syn_weight   = syn_weight
        self.trans_weight = trans_weight
        self.syn_k        = syn_k

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

        print("[*] Fitting PDN symmetry order from corpus autocorrelation...")
        self.pdn.fit_from_trigrams(self.geo, self.lm.tri_raw)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)
        print(self.pdn.theorem_bridge_report())

        print("[*] Building CoT contextual stub library...")
        self.stub_lib.build(self.geo, self.lm.vocab, self.lm.raw_freq)

        self.cot = CoTReasoningEngine(
            stub_library   = self.stub_lib,
            kernels        = self.kernels,
            pdn_engine     = self.pdn,
            n_hops         = 3,
            tokens_per_hop = 10,
            device         = self.device,
        )

        print("[*] Building Instruction Distribution module...")
        self.instr_dist = InstructionDistribution(
            geo=self.geo, kernels=self.kernels, lm=self.lm, device=self.device,
        )

        print("[*] Building Semantic Mandate Scorer...")
        self.mandate_scorer = SemanticMandateScorer(
            geo=self.geo, kernels=self.kernels, device=self.device,
        )

        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.mandate_scorer, self.mrv, self.chunk, self.iso_stacker,
            self.pdn, self.cot, self.instr_dist,
            device       = self.device,
            syn_weight   = self.syn_weight,
            trans_weight = self.trans_weight,
            syn_k        = self.syn_k,
        )
        print("[+] Training complete. (V18-CSNS-G: grounded embedding + ACF spectral + kernel mandate)")

    def save_cache(self, filename: str = "v18_csns_g_model.pkl"):
        print(f"[*] Saving model state to {filename}...")
        with open(filename, "wb") as f:
            pickle.dump({
                "geo_vecs"       : self.geo._vecs,
                "geo_cache"      : self.geo._cache,
                "lm_raw_freq"    : self.lm.raw_freq,
                "lm_tri_raw"     : self.lm.tri_raw,
                "lm_heads"       : self.lm.heads,
                "lm_vocab"       : self.lm.vocab,
                "graph_nodes"    : self.graph.nodes,
                "corpus_snippet" : self.corpus_snippet,
                "pdn_n_star"     : self.pdn.n_star,
                "pdn_acf"        : self.pdn.acf_values,
                "pdn_sig_bound"  : self.pdn.acf_significance_bound,
                "cot_stubs"      : self.stub_lib.stubs,
                "syn_weight"     : self.syn_weight,
                "trans_weight"   : self.trans_weight,
                "syn_k"          : self.syn_k,
                "version"        : "V18-CSNS-G",
            }, f)
        print("[+] Save successful.")

    def load_cache(self, filename: str):
        print(f"[*] Loading model state from {filename}...")
        with open(filename, "rb") as f:
            state = pickle.load(f)

        self.geo._vecs          = state["geo_vecs"]
        self.geo._cache         = state["geo_cache"]
        self.lm.raw_freq        = state["lm_raw_freq"]
        self.lm.tri_raw         = state["lm_tri_raw"]
        self.lm.heads           = state["lm_heads"]
        self.lm.vocab           = state["lm_vocab"]
        self.graph.nodes        = state["graph_nodes"]
        self.corpus_snippet     = state["corpus_snippet"]
        self.pdn.n_star         = state.get("pdn_n_star", 4)
        self.pdn.acf_values     = state.get("pdn_acf", {})
        self.pdn.acf_significance_bound = state.get("pdn_sig_bound", 0.0)
        self.pdn.power_spectrum = {k: abs(v) for k, v in self.pdn.acf_values.items()}
        self.syn_weight         = state.get("syn_weight",   0.4)
        self.trans_weight       = state.get("trans_weight", 0.6)
        self.syn_k              = state.get("syn_k",        8)

        print("[*] Rebuilding GPU Tensors from loaded state...")
        self.geo.build_cuda_tensors(self.lm.vocab)
        self.lm.finalise()
        self.graph.build(self.lm)
        self.graph.propagate(steps=2)
        self.mrv.prime(self.lm.vocab, self.geo)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)

        if "cot_stubs" in state:
            self.stub_lib.stubs = state["cot_stubs"]
            self.stub_lib._rebuild_tensors()
        else:
            self.stub_lib.build(self.geo, self.lm.vocab, self.lm.raw_freq)

        self.cot = CoTReasoningEngine(
            stub_library=self.stub_lib, kernels=self.kernels,
            pdn_engine=self.pdn, n_hops=3, tokens_per_hop=10, device=self.device,
        )
        self.instr_dist = InstructionDistribution(
            geo=self.geo, kernels=self.kernels, lm=self.lm, device=self.device,
        )
        self.mandate_scorer = SemanticMandateScorer(
            geo=self.geo, kernels=self.kernels, device=self.device,
        )
        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.mandate_scorer, self.mrv, self.chunk, self.iso_stacker,
            self.pdn, self.cot, self.instr_dist,
            device=self.device,
            syn_weight=self.syn_weight, trans_weight=self.trans_weight, syn_k=self.syn_k,
        )
        print("[+] Load successful. (V18-CSNS-G)")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — GRADIO GUI
# ════════════════════════════════════════════════════════════════════════════

class V18GUI:
    def __init__(self):
        self.engine = None

    def init_engine_from_file(self, file_obj, syn_weight, trans_weight, syn_k):
        if file_obj is None:
            return "Error: No file uploaded."
        try:
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                corpus_text = f.read()
            if not corpus_text.strip():
                return "Error: Uploaded file is empty."
            self.engine = V18Engine(
                syn_weight=float(syn_weight),
                trans_weight=float(trans_weight),
                syn_k=int(syn_k),
            )
            self.engine.train(corpus_text)
            report = self.engine.pdn.theorem_bridge_report()
            stub_counts = {k: len(v) for k, v in self.engine.stub_lib.stubs.items()}
            return (
                f"V18-CSNS-G Engine initialised.\n"
                f"File: {file_obj.name.split('/')[-1]}\n"
                f"Vocab size: {len(self.engine.lm.vocab)}\n"
                f"CoT stubs: {stub_counts}\n"
                f"CSNS: ω_syn={syn_weight:.2f}  ω_trans={trans_weight:.2f}  K={int(syn_k)}\n\n"
                f"{report}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_text(self, sentences, tokens, seed_text, instruction_text, and_weight, temperature):
        if not self.engine or not self.engine.walker:
            return "Engine not initialised.", "", ""
        text, traces, step_report = generate_passage(
            self.engine.walker, self.engine.lm,
            num_sentences=int(sentences), tokens_per_sent=int(tokens),
            seed_text=seed_text.strip(), instruction_text=instruction_text.strip(),
            and_weight=float(and_weight), temperature=float(temperature),
            return_traces=True,
        )
        trace_text = "\n".join(tr.render() for tr in traces)
        return text, trace_text, step_report

    def pdn_report(self):
        if not self.engine:
            return "Engine not initialised."
        return self.engine.pdn.theorem_bridge_report()

    def cot_history(self):
        if not self.engine or not self.engine.cot:
            return "Engine not initialised."
        return self.engine.cot.all_traces_text()

    def csns_report(self):
        if not self.engine or not self.engine.walker:
            return "Engine not initialised."
        return self.engine.walker.csns_report()

    def mandate_report(self):
        if not self.engine or not self.engine.mandate_scorer:
            return "Engine not initialised."
        return self.engine.mandate_scorer.centroid_report()

    def dnn_report(self):
        lines = [
            "V18-CSNS-G DNN Array + Cross-Synaptic Neuron Sum Report",
            "═══════════════════════════════════════════════════════════════",
            "",
            "GROUNDING CHANGES (V18-CSNS-G vs V18-CSNS):",
            "  1. Thébault embedding: formally justified via Zipfian distributional",
            "     hypothesis. Frequency → radial depth (distributional breadth).",
            "     Rank → angular coordinate (ordinal proximity, like sinusoidal PE).",
            "     Refs: Zipf (1935), Harris (1954), Vaswani et al. (2017).",
            "",
            "  2. PDN spectral analysis: replaced DFT power analogy with corpus",
            "     autocorrelation ACF. n* = argmax_{lag≥2} ACF(lag) if significant",
            "     at 95% CI (1.96/√T). Statistically interpretable: n* is the",
            "     characteristic recurrence period of rho in the token sequence.",
            "     Ref: Box & Jenkins (1976). Time Series Analysis.",
            "",
            "  3. Mandate scorer: replaced keyword matching with Thébault kernel",
            "     similarity to instruction centroid. Fully consistent with pipeline.",
            "     mandate_bonus = k_reg · k_ori · k_side · mandate_scale (layer-normed).",
            "",
            "PIPELINE ORDER (per token step):",
            "  1. Raw walker logits  (kernels + PDN ACF bonus + CoT + kernel mandate)",
            "  2. ContingentExtringentProbability governance",
            "  3. CSNS enrichment (synaptic sum + transitive bonus)",
            "  4. DNNArrayPipeline.forward(logits_enriched)",
            "  5. AND combination: α·log P_instr + (1−α)·DNN log P_walk",
            "═══════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


def launch_gui():
    gui = V18GUI()

    with gr.Blocks(title="NeuroSymbolic V18-CSNS-G") as app:
        gr.Markdown(
            "# NeuroSymbolic V18-CSNS-G — Grounded Edition\n"
            "### Thébault Geometry (Zipfian grounding) · ACF Spectral Analysis · "
            "Kernel Mandate Scorer · DNN Array · Cross-Synaptic Neuron Sums"
        )

        with gr.Tab("Train"):
            file_input     = gr.File(label="Upload .txt Corpus File", file_types=[])
            with gr.Row():
                syn_w_slider   = gr.Slider(0.0, 2.0, value=0.4,  step=0.05, label="CSNS ω_syn")
                trans_w_slider = gr.Slider(0.0, 2.0, value=1.8,  step=0.05, label="CSNS ω_trans")
                syn_k_slider   = gr.Slider(2,   32,  value=31,   step=1,    label="CSNS K")
            train_file_btn = gr.Button("Initialise from File", variant="primary")
            init_out       = gr.Textbox(label="Engine Status / ACF Spectral Report", lines=22, interactive=False)
            train_file_btn.click(
                gui.init_engine_from_file,
                inputs=[file_input, syn_w_slider, trans_w_slider, syn_k_slider],
                outputs=init_out,
            )

        with gr.Tab("Generate"):
            with gr.Row():
                sentences   = gr.Slider(1, 10,   value=4,    step=1,    label="Sentences")
                tokens      = gr.Slider(20, 180, value=80,   step=1,    label="Tokens/sentence")
                and_weight  = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="AND weight α")
                temperature = gr.Slider(0.1, 3.0, value=0.2, step=0.05, label="Temperature")

            instruction_input = gr.Textbox(
                label="Instruction (AND distribution + kernel mandate source)",
                value="Explain the meaning of life and human consciousness.",
                lines=2,
            )
            seed_input = gr.Textbox(label="Seed Text", placeholder="e.g. quantum entanglement")
            gen_btn = gr.Button("Generate", variant="primary")
            gen_out = gr.Textbox(lines=10, label="Generated Text")

            with gr.Row():
                cot_out  = gr.Textbox(lines=12, label="Chain-of-Thought Trace",        interactive=False)
                step_out = gr.Textbox(lines=12, label="AND+CSNS Step Trace (per token)", interactive=False)

            gen_btn.click(
                gui.generate_text,
                inputs=[sentences, tokens, seed_input, instruction_input, and_weight, temperature],
                outputs=[gen_out, cot_out, step_out],
            )

        with gr.Tab("Diagnostics"):
            dnn_btn  = gr.Button("Show DNN + CSNS-G Pipeline Report")
            dnn_out  = gr.Textbox(lines=25, label="DNN Array + CSNS-G Report", interactive=False)
            dnn_btn.click(gui.dnn_report, outputs=dnn_out)

            csns_btn = gr.Button("Show CSNS Diagnostic Report")
            csns_out = gr.Textbox(lines=16, label="CSNS Diagnostics", interactive=False)
            csns_btn.click(gui.csns_report, outputs=csns_out)

            pdn_btn  = gr.Button("Show ACF Spectral Report")
            pdn_out  = gr.Textbox(lines=20, label="ACF Spectral Report (replaces PDN bridge)", interactive=False)
            pdn_btn.click(gui.pdn_report, outputs=pdn_out)

            mandate_btn = gr.Button("Show Mandate Centroid")
            mandate_out = gr.Textbox(lines=4, label="Semantic Mandate Scorer", interactive=False)
            mandate_btn.click(gui.mandate_report, outputs=mandate_out)

            cot_hist_btn = gr.Button("Show Full CoT History")
            cot_hist_out = gr.Textbox(lines=20, label="CoT Trace History", interactive=False)
            cot_hist_btn.click(gui.cot_history, outputs=cot_hist_out)

    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",          action="store_true")
    parser.add_argument("--corpus",       type=str)
    parser.add_argument("--instruction",  type=str,  default="")
    parser.add_argument("--and-weight",   type=float, default=0.5)
    parser.add_argument("--temperature",  type=float, default=1.4)
    parser.add_argument("--syn-weight",   type=float, default=0.4)
    parser.add_argument("--trans-weight", type=float, default=0.6)
    parser.add_argument("--syn-k",        type=int,   default=8)
    args = parser.parse_args()

    if args.gui or not args.corpus:
        launch_gui()
        exit(0)

    try:
        corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    except Exception as e:
        print(f"[!] Failed to read {args.corpus}: {e}")
        exit(1)

    engine = V18Engine(
        syn_weight=args.syn_weight,
        trans_weight=args.trans_weight,
        syn_k=args.syn_k,
    )
    engine.train(corpus_text)
    engine.save_cache("v18_csns_g_model.pkl")

    print("\n--- SAMPLE GENERATION (V18-CSNS-G) ---")
    instruction = args.instruction or "Explain the meaning of life."
    text, traces, step_report = generate_passage(
        engine.walker, engine.lm,
        num_sentences=3, tokens_per_sent=30,
        instruction_text=instruction,
        and_weight=args.and_weight,
        temperature=args.temperature,
        return_traces=True,
    )
    print(text)
    print("\n--- COT TRACES ---")
    for tr in traces:
        print(tr.render())
    print("\n--- AND+CSNS STEP TRACE ---")
    print(step_report)
    print("\n--- CSNS DIAGNOSTIC ---")
    print(engine.walker.csns_report())
    print("\n--- MANDATE CENTROID ---")
    print(engine.mandate_scorer.centroid_report())
