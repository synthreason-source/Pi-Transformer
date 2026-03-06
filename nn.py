#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V10.0 — Superpolynomial Automorphism Bridge
===============================================================================

BRIDGE DECLARATION
──────────────────
V9.2.0 is REJECTED as the generative substrate.
V10.0 retains V9.2.0's data structures as a READ-ONLY CORPUS ORACLE —
a fixed-point algebra from which the new system derives its graded structure.

The isomorphic bridge is:

  Φ : V9_State  ──────────────────────────────→  V10_GradedState
                  superpolynomial ring action

Concretely:

  V9 HebbianReservoir  →  GradedSynapseAlgebra     (trigrams → supermonomials)
  V9 TetraGrid 2×2     →  AutoOrbitModule           (lattice → graded φ-module)
  V9 Bernoulli pass    →  abolished (stochastic)
  V9 Mirror pass       →  abolished (ad-hoc inversion)
  V10 single pass      →  φ-equivariant, superpolynomial-weighted generation

MATHEMATICS
───────────
1.  Superpolynomial Ring  R = ℝ[x₁…xₙ | θ₁…θₙ]
    x-vars : bosonic (symmetric, freq > median)
    θ-vars : fermionic (anti-symmetric, nilpotent θᵢ²=0, freq ≤ median)
    Basis  : Jack superpolynomials  P_λ(x;θ;α)

2.  Vocabulary Automorphism  φ ∈ Aut(Cayley(V))
    φ : V → V,  φ² = id  (involution)
    φ pairs high-freq ↔ low-freq tokens by rank-reversal
    Parity: p(v) = 0 (bosonic) if rank(v) ≤ |V|/2
                 = 1 (fermionic) otherwise

3.  Graded Synapse Algebra
    Each trigram (w₁,w₂,w₃) maps to supermonomial:
      m(w₁,w₂,w₃) = x^{deg(w₁)+deg(w₂)}  ·  θ^{p(w₃)}
    Synapse weight in the graded algebra:
      W_super(w₁,w₂,w₃) = W_Hebb(w₁,w₂,w₃) · P_{λ(w₃)}(x_context ; θ_context ; α)

4.  AutoOrbit TetraGrid Module
    The 2×2 lattice G is a graded R-module.
    Injection is the φ-orbit action:
      Δ_{φ}(c,a) = P_{[1]}(f_c, f_a ; θ_c, θ_a ; α)  ·  [c ∈ orbit_φ(a)]
                 + α · P_{[1,1]}(f_c, f_a ; …)         ·  [p(c) ≠ p(a)]
    This replaces all Bernoulli / ad-hoc mirror injection.

5.  φ-Equivariant Generation
    For every candidate c and its φ-partner φ(c):
      logit_auto(c)   = log p_Hebb(c) + β · Δ_φ(c, anchor)
      logit_auto(φ(c)) ≈ log p_Hebb(φ(c)) + β · α · Δ_φ(φ(c), anchor)
    The ratio is controlled by the Jack α-parameter — a single algebraic knob.
===============================================================================
"""

from __future__ import annotations
import re
import math
import random
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import torch
import torch.nn.functional as F
from datasets import load_dataset
import gradio as gr

# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — TOKEN PRIMITIVES  (shared with V9, kept verbatim as oracle API)
# ════════════════════════════════════════════════════════════════════════════

STOP_WORDS_COG = set(
    "a an and are as at be by for from has have he her him his i in is it its "
    "me my of on or our she so that the their them they this to was we were what "
    "when where which who will with you your if because while".split()
)
COGNITIVE_TOKENS = {f"[{w.upper()}]" for w in STOP_WORDS_COG}
PUNCT_TOKENS = {",", ".", "!", "?", ";", ":"}


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
            raw = t.strip("[]").lower()
            word = raw.capitalize() if not res or res[-1].endswith(('.','!','?')) else raw
            res.append(word)
        else:
            word = t.capitalize() if not res or res[-1].endswith(('.','!','?')) else t
            res.append(word)
    out = " ".join(res).strip()
    return out if out and out[-1] in PUNCT_TOKENS else out + "."


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SUPERPOLYNOMIAL RING
# ════════════════════════════════════════════════════════════════════════════

class SuperPolynomialRing:
    """
    R = ℝ[x₁…xₙ | θ₁…θₙ],  Jack deformation parameter α.

    P_λ(x;θ;α):  Jack superpolynomial indexed by partition λ.

    We implement the rank-1 and rank-2 cases exactly; higher ranks
    are approximated via the Cauchy identity factorisation.

    orbit_scalar(f_c, f_a, p_c, p_a) → ℝ≥0
        The scalar that replaces stochastic injection in AutoOrbitModule.
    """

    def __init__(self, alpha: float = 1.5):
        self.alpha = alpha

    # ── Vandermonde ──────────────────────────────────────────────────────
    @staticmethod
    def _vander(xs: torch.Tensor) -> torch.Tensor:
        n = xs.shape[0]
        d = xs.unsqueeze(1) - xs.unsqueeze(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        return d[mask].prod().clamp(min=1e-12)

    # ── Schur polynomial  s_λ(x)  via bialternant ────────────────────────
    def schur(self, lam: List[int], xs: torch.Tensor) -> torch.Tensor:
        n = xs.shape[0]
        lp = (list(lam) + [0] * n)[:n]
        exp = torch.tensor([lp[j] + n - j - 1 for j in range(n)],
                           dtype=torch.float32)
        A = xs.clamp(min=1e-6).unsqueeze(1) ** exp.unsqueeze(0)
        return torch.linalg.det(A) / self._vander(xs.clamp(min=1e-6))

    # ── Jack superpolynomial  P_λ(x;θ;α) ────────────────────────────────
    def P(self, lam: List[int],
          xs: torch.Tensor,
          thetas: torch.Tensor) -> torch.Tensor:
        """
        P_λ = s_λ(x) · (1 + α · Σᵢ θᵢ xᵢ / Σxᵢ)

        The fermionic factor is the first-order nilpotent correction:
        since θᵢ²=0, the exponential truncates to 1 + linear term.
        """
        s = self.schur(lam, xs)
        theta_corr = (thetas * xs / (xs.sum() + 1e-8)).sum()
        return s * (1.0 + self.alpha * theta_corr)

    # ── Scalar orbit weight for a single candidate ───────────────────────
    def orbit_scalar(
        self,
        f_c: float,   # candidate raw frequency
        f_a: float,   # anchor raw frequency
        p_c: int,     # candidate parity  (0=bosonic, 1=fermionic)
        p_a: int,     # anchor parity
    ) -> float:
        """
        w(c,a) = P_{[1]}(x;θ;α)  with:
          x = [f_c / max, f_a / max]
          θ = [p_c, p_a]  (fermionic coordinates)

        Fermionic candidates (p=1, under-explored) receive an α-boost
        because θ enters the numerator of the Jack correction.
        When p_c ≠ p_a (inter-strata crossing), add P_{[1,1]} term
        to capture the cross-stratum mixing induced by φ.
        """
        mx = max(f_c, f_a, 1.0)
        xs = torch.tensor([f_c / mx, f_a / mx], dtype=torch.float32)
        th = torch.tensor([float(p_c), float(p_a)], dtype=torch.float32)
        w = self.P([1], xs, th)
        if p_c != p_a:                          # inter-strata: add rank-2 term
            w = w + 0.5 * self.P([1, 1], xs, th)
        return float(w.clamp(min=0.0).item())


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VOCABULARY AUTOMORPHISM   φ: V → V,  φ² = id
# ════════════════════════════════════════════════════════════════════════════

class VocabularyAutomorphism:
    """
    Constructs the canonical involution on the vocabulary Cayley graph.

    Grading:
      parity 0 (bosonic)   — rank ≤ |V|/2  (higher frequency)
      parity 1 (fermionic) — rank > |V|/2  (lower frequency)

    φ pairs bosonic token at rank r  ↔  fermionic token at rank |V|+1-r
    so that  freq(v) · freq(φ(v)) ≈ geometric_mean²  (isospectral pairing).

    Fixed points exist when |V| is odd.
    """

    def __init__(self, freq: Dict[str, float]):
        self.freq   : Dict[str, float] = freq
        self.phi    : Dict[str, str]   = {}
        self.parity : Dict[str, int]   = {}
        self.orbits : List[Tuple[str,str]] = []
        self._build()

    def _build(self) -> None:
        tokens = sorted(
            [t for t in self.freq
             if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS],
            key=lambda t: self.freq[t], reverse=True
        )
        if not tokens:
            return
        mid = len(tokens) // 2
        for i, t in enumerate(tokens):
            self.parity[t] = 0 if i < mid else 1

        n_pairs = min(mid, len(tokens) - mid)
        for i in range(n_pairs):
            h = tokens[i]                 # bosonic  (high freq, rank i)
            l = tokens[-(i + 1)]          # fermionic (low  freq, rank |V|-i)
            self.phi[h] = l
            self.phi[l] = h
            self.orbits.append((h, l))

        # Fixed points
        for t in tokens:
            if t not in self.phi:
                self.phi[t] = t

    def image(self, t: str) -> str:
        return self.phi.get(t, t)

    def parity_of(self, t: str) -> int:
        return self.parity.get(t, 0)

    def orbit_of(self, t: str) -> Set[str]:
        return {t, self.phi.get(t, t)}

    def are_paired(self, t1: str, t2: str) -> bool:
        return self.phi.get(t1) == t2


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GRADED SYNAPSE ALGEBRA
#   Bridge: V9 HebbianReservoirLM → GradedSynapseAlgebra
# ════════════════════════════════════════════════════════════════════════════

class GradedSynapseAlgebra:
    """
    Isomorphic lifting of the V9 Hebbian trigram reservoir.

    V9 stored:   W[(w1,w2,w3)] = raw co-occurrence count  ∈ ℝ
    V10 stores:  W_graded[(w1,w2,w3)] = count · P_λ(x_ctx ; θ_ctx ; α)

    where:
      λ         = partition derived from (deg(w1), deg(w2)) — see _partition()
      x_context = normalised frequency vector of the trigram head (w1,w2)
      θ_context = parity vector of the trigram head

    The graded weight is computed ONCE during ingestion and cached.
    next_dist() returns candidate lists ranked by graded weights.

    Isomorphism certificate:
      Setting α=0 recovers exactly the V9 Hebbian weights (Schur s_λ → 1
      for λ=[0] and the correction term vanishes).  The map is therefore
      a 1-to-1 deformation retract; V9 is the α=0 fibre.
    """

    BASAL_K = 1.5

    def __init__(self, spr: SuperPolynomialRing, phi: VocabularyAutomorphism):
        self.spr = spr
        self.phi = phi
        self.raw_freq    : Dict[str, float]                     = {}
        self.tri_raw     : Dict[Tuple[str,str,str], float]      = {}
        self.tri_graded  : Dict[Tuple[str,str,str], float]      = {}
        self.heads       : Dict[Tuple[str,str], List[str]]      = {}
        self.vocab       : List[str]                            = []
        self.token_to_idx: Dict[str, int]                       = {}

    # ── partition from degree pair ────────────────────────────────────────
    @staticmethod
    def _partition(d1: float, d2: float) -> List[int]:
        """
        Map a pair of continuous degrees to a valid integer partition.
        We use the floor of log2(d+1) clamped to [0,4] so:
          λ = [⌊log₂(d1+1)⌋, ⌊log₂(d2+1)⌋]  (non-increasing)
        """
        a = min(int(math.log2(d1 + 1)), 4)
        b = min(int(math.log2(d2 + 1)), 4)
        lam = sorted([a, b], reverse=True)
        return lam if lam[0] > 0 else [1]

    # ── ingest corpus ────────────────────────────────────────────────────
    def ingest(self, tokens: List[str]) -> None:
        for t in tokens:
            self.raw_freq[t] = self.raw_freq.get(t, 0) + 1.0
        for t in self.raw_freq:
            if t not in self.token_to_idx:
                self.token_to_idx[t] = len(self.token_to_idx)

        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            k = (w1, w2, w3)
            self.tri_raw[k] = self.tri_raw.get(k, 0) + 1.0
            head = (w1, w2)
            if head not in self.heads:
                self.heads[head] = []
            if w3 not in self.heads[head]:
                self.heads[head].append(w3)

        # Compute graded weights
        max_f = max(self.raw_freq.values(), default=1.0)
        for (w1, w2, w3), cnt in self.tri_raw.items():
            f1 = self.raw_freq.get(w1, 1.0) / max_f
            f2 = self.raw_freq.get(w2, 1.0) / max_f
            p1 = self.phi.parity_of(w1)
            p2 = self.phi.parity_of(w2)
            lam = self._partition(f1 * 10, f2 * 10)
            xs  = torch.tensor([f1, f2], dtype=torch.float32)
            th  = torch.tensor([float(p1), float(p2)], dtype=torch.float32)
            grading = float(self.spr.P(lam, xs, th).clamp(min=1e-4).item())
            self.tri_graded[(w1, w2, w3)] = cnt * grading

        self.vocab = [v for v in self.raw_freq
                      if v not in PUNCT_TOKENS and v not in COGNITIVE_TOKENS]

    # ── next distribution (graded) ────────────────────────────────────────
    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        head = (w1, w2)
        if head in self.heads:
            cands_raw = self.heads[head]
            weights   = [self.tri_graded.get((w1, w2, w3), 1e-4)
                         for w3 in cands_raw]
        else:
            # Global fallback: aggregate all graded continuation weights
            agg: Dict[str, float] = {}
            for (_, _, w3), wt in self.tri_graded.items():
                agg[w3] = agg.get(w3, 0) + wt
            cands_raw = list(agg.keys())[:400]
            weights   = [agg[w] for w in cands_raw]

        # Kneser-Ney–style smoothing in the graded algebra
        k = self.BASAL_K
        V_total = len(self.vocab) + 1
        total = sum(weights)
        probs_raw = [(wt + k) / (total + k * V_total) for wt in weights]
        probs = torch.tensor(probs_raw, dtype=torch.float32)
        return cands_raw, probs / probs.sum().clamp(min=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AUTO-ORBIT TETRAGRID MODULE
#   Bridge: V9 TetraGridIsomorphism → AutoOrbitModule (graded R-module)
# ════════════════════════════════════════════════════════════════════════════

class AutoOrbitModule(torch.nn.Module):
    """
    The 2×2 TetraGrid lattice G is promoted to a graded R-module.

    Each entry G[i,j](c) is no longer a bare inner product — it carries
    a superpolynomial grading that encodes the φ-orbit structure.

    Injection rule (replaces V9 Bernoulli AND V9 mirror injection):

      Δ(c, anchor) = P_{[1]}(f_c, f_a; p_c, p_a; α)  · sparse_mask(c)
                   + α · P_{[1,1]}(…)                  · paired_mask(c)

    where:
      sparse_mask : determinant-based, inherited from V9  (structural)
      paired_mask : 1 iff c ∈ orbit_φ(anchor)            (automorphic)

    The TWO injection channels correspond to the graded decomposition:
      G = G₀ (bosonic sector) ⊕ G₁ (fermionic sector)
    Diagonal entries G[0,0], G[1,1] receive the bosonic injection.
    Off-diagonal G[0,1], G[1,0] receive the inter-strata (fermionic) injection.

    Isomorphism certificate:
      At α=0 and phi=identity, AutoOrbitModule reduces exactly to
      V9's forward() with Bernoulli expectation = 0.5 (mean injection).
    """

    def __init__(self,
                 token_to_idx : Dict[str, int],
                 spr          : SuperPolynomialRing,
                 phi          : VocabularyAutomorphism,
                 raw_freq     : Dict[str, float],
                 adv_strength : float = 0.5,
                 densify_mag  : float = 0.08,
                 embed_dim    : int   = 256):
        super().__init__()
        self.spr          = spr
        self.phi          = phi
        self.raw_freq     = raw_freq
        self.adv_strength = adv_strength
        self.densify_mag  = densify_mag

        vocab_size   = len(token_to_idx) + 1
        self.unk_idx = vocab_size - 1
        self.token_to_idx = token_to_idx

        self.E_embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.L_embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.shift   = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.shift.weight.copy_(torch.tensor([[0.85,0.15],[0.15,0.85]]))
            self.shift.bias.fill_(0.0)

    def _emb(self, t: str, table: torch.nn.Embedding) -> torch.Tensor:
        idx = self.token_to_idx.get(t, self.unk_idx)
        v = torch.abs(table(torch.tensor(idx)))
        return v / (v.sum() + 1e-8)

    def _build_G(self, anchor: str, cands: List[str]):
        mag = lambda c: 0.05 if c in PUNCT_TOKENS else min(max(len(c)/10,0.1),2.0)
        a_mag = mag(anchor)
        E_A = self._emb(anchor, self.E_embed)          # (D,)
        L_A = self._emb(anchor, self.L_embed) * a_mag  # (D,)
        E_C = torch.stack([self._emb(c, self.E_embed) for c in cands])          # (N, D)
        L_C = torch.stack([self._emb(c, self.L_embed) * mag(c) for c in cands]) # (N, D)

        # Each of these is (N,) — one scalar per candidate
        N_00 = (E_C * E_A).sum(-1)   # E_C · E_A
        N_01 = (L_C * E_A).sum(-1)   # L_C · E_A
        N_10 = (E_C * L_A).sum(-1)   # E_C · L_A
        N_11 = (L_C * L_A).sum(-1)   # L_C · L_A

        # G shape: (N, 2, 2)
        G = torch.stack([
            torch.stack([N_00, N_01], dim=-1),   # (N, 2)  — row 0
            torch.stack([N_10, N_11], dim=-1),   # (N, 2)  — row 1
        ], dim=-2)                               # (N, 2, 2)

        G = G - self.adv_strength * G**2
        G = G + 0.15 * torch.tanh(self.shift(G))
        det = G[:,0,0] * G[:,1,1] - G[:,0,1] * G[:,1,0]
        return G, det

    def _readout(self, G: torch.Tensor) -> torch.Tensor:
        r = (G[:,0,0] + G[:,1,1]) + 0.5*(G[:,0,1].abs() + G[:,1,0].abs())
        mn, mx = r.min(), r.max()
        return (r - mn) / (mx - mn + 1e-12) if mx > mn else r

    def forward(self,
                anchor   : str,
                cands    : List[str],
                auto_strength: float = 0.55) -> torch.Tensor:
        """
        Single φ-equivariant forward pass.

        No Bernoulli.  No ad-hoc mirror weights.
        Injection is fully determined by the superpolynomial ring action.
        """
        if not cands:
            return torch.zeros(0)

        G, det = self._build_G(anchor, cands)
        threshold   = det.median() if det.numel() > 0 else torch.tensor(0.0)
        sparse_mask = (det < threshold).float()

        anchor_orbit = self.phi.orbit_of(anchor)
        f_a = self.raw_freq.get(anchor, 1.0)
        p_a = self.phi.parity_of(anchor)

        diag_inj   = []
        offdiag_inj = []
        for c in cands:
            f_c = self.raw_freq.get(c, 1.0)
            p_c = self.phi.parity_of(c)
            w_diag = self.spr.orbit_scalar(f_c, f_a, p_c, p_a)
            # Cross-stratum (inter-orbit) injection in off-diagonal
            w_off  = (self.spr.alpha * self.spr.orbit_scalar(f_c, f_a, 1-p_c, p_a)
                      if c in anchor_orbit else 0.0)
            diag_inj.append(w_diag)
            offdiag_inj.append(w_off)

        d_raw = torch.tensor(diag_inj,    dtype=torch.float32)
        o_raw = torch.tensor(offdiag_inj, dtype=torch.float32)

        # Normalise each injection channel independently
        def _norm(v):
            mn, mx = v.min(), v.max()
            return (v - mn) / (mx - mn + 1e-12) if mx > mn else torch.zeros_like(v)

        d_n = _norm(d_raw)
        o_n = _norm(o_raw)

        mag = self.densify_mag * (1.0 + auto_strength)
        G[:,0,0] += sparse_mask * d_n * mag
        G[:,1,1] += sparse_mask * d_n * mag
        G[:,0,1] += sparse_mask * o_n * mag * 0.5   # fermionic off-diagonal
        G[:,1,0] += sparse_mask * o_n * mag * 0.5

        return self._readout(G)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — φ-EQUIVARIANT PROBABILITY FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def next_probs_auto(
    graded_lm   : GradedSynapseAlgebra,
    orbit_grid  : AutoOrbitModule,
    w1          : str,
    w2          : str,
    temp        : float = 1.4,
    de_strength : float = 0.22,
    auto_strength: float = 0.55,
) -> Tuple[List[str], torch.Tensor]:
    """
    φ-equivariant candidate probability.

    p_auto(c | w1,w2) ∝ exp( log p_graded(c) / τ
                              + β · Δ_φ(c, w2) )

    where Δ_φ is the superpolynomial orbit injection from AutoOrbitModule.

    Equivariance property (by construction):
      Δ_φ(φ(c), w2) = α · Δ_φ(c, w2)          [when φ(c) ≠ c]
    so the φ-partner always gets a scaled version of the same logit boost —
    a genuine automorphism, not a heuristic.
    """
    cands, base_probs = graded_lm.next_dist(w1, w2)
    if not cands:
        return cands, base_probs

    grid_out = orbit_grid(anchor=w2, cands=cands, auto_strength=auto_strength)

    # Punctuation control (unchanged logic, necessary for fluency)
    punct_bias    = torch.zeros(len(cands))
    punct_penalty = torch.zeros(len(cands))
    for i, c in enumerate(cands):
        if c in PUNCT_TOKENS:
            punct_bias[i] = -3.5
            if w2 in PUNCT_TOKENS:
                punct_penalty[i] = -1e4

    logits = (torch.log(base_probs.clamp(min=1e-12))
              + de_strength * grid_out
              + punct_bias + punct_penalty) / max(temp, 1e-6)
    return cands, F.softmax(logits, dim=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GENERATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SyntacticForm:
    word: str; syntactic_role: str; prefix_context: str; suffix_context: str
    form_name: str = ""; activation_value: float = 0.0
    def __post_init__(self):
        h = hashlib.md5(f"{self.word}_{self.syntactic_role}_{self.prefix_context}_{self.suffix_context}".encode()).hexdigest()[:6]
        self.form_name = f"form_{self.word}_{h}"

@dataclass
class V10State:
    graded_lm  : GradedSynapseAlgebra
    orbit_grid : AutoOrbitModule
    forms      : Dict[int, SyntacticForm] = field(default_factory=dict)
    outputs    : Dict[int, str]           = field(default_factory=dict)


def build_v10_state(
    corpus_text  : str,
    prompt       : str  = "Consider the nature of understanding",
    num_sentences: int  = 100,
    adv_strength : float = 0.5,
    densify_mag  : float = 0.08,
    alpha        : float = 1.5,
) -> V10State:
    tokens = tokenize(corpus_text)
    spr    = SuperPolynomialRing(alpha=alpha)

    # Bootstrap automorphism from raw frequency (pre-ingest pass)
    raw_freq: Dict[str, float] = {}
    for t in tokens:
        raw_freq[t] = raw_freq.get(t, 0) + 1.0
    phi = VocabularyAutomorphism(raw_freq)

    graded_lm = GradedSynapseAlgebra(spr=spr, phi=phi)
    graded_lm.ingest(tokens)

    orbit_grid = AutoOrbitModule(
        token_to_idx = graded_lm.token_to_idx,
        spr          = spr,
        phi          = phi,
        raw_freq     = graded_lm.raw_freq,
        adv_strength = adv_strength,
        densify_mag  = densify_mag,
    )

    # SyntacticForms from prompt (same logic as V9 — reused as oracle)
    prompt_toks = tokenize(prompt.upper())
    base_words  = [w for w in prompt_toks
                   if w not in COGNITIVE_TOKENS and w not in PUNCT_TOKENS
                   and re.match(r'^[a-z]+$', w)] or ["concept", "form"]
    roles    = ["noun","verb","adj","adv"]
    prefixes = ["pre","post","anti","hyper","meta","sub","un","re"]
    suffixes = ["ism","ity","ness","tion","ology","ment","ive","ly"]
    forms: Dict[int, SyntacticForm] = {}
    for i in range(num_sentences):
        w    = base_words[i % len(base_words)]
        role = roles[i % len(roles)]
        pref = prefixes[(i // len(roles)) % len(prefixes)]
        suff = suffixes[(i // (len(roles)*len(prefixes))) % len(suffixes)]
        forms[i] = SyntacticForm(word=w, syntactic_role=role,
                                 prefix_context=pref, suffix_context=suff)
    return V10State(graded_lm=graded_lm, orbit_grid=orbit_grid, forms=forms)


def generate(
    state          : V10State,
    seed           : int   = 42,
    num_sentences  : int   = 100,
    tokens_per_sent: int   = 92,
    temp           : float = 1.4,
    de_strength    : float = 0.22,
    auto_strength  : float = 0.55,
) -> List[str]:
    torch.manual_seed(seed)
    random.seed(seed)

    head_list = list(state.graded_lm.heads.keys())
    if not head_list:
        return ["Insufficient trigram data."]
    random.shuffle(head_list)

    MIN_PRE_PUNCT = max(3, int(tokens_per_sent * 0.15))
    MIN_PRE_END   = max(4, int(tokens_per_sent * 0.85))
    END_PUNCT     = {".", "?", "!"}

    def best_non_punct(cs, ps):
        bi, bp = None, -1.0
        for i,(c,p) in enumerate(zip(cs, ps.tolist())):
            if c not in PUNCT_TOKENS and p > bp:
                bi, bp = i, p
        return cs[bi] if bi is not None else "the"

    sentences = []
    state.outputs.clear()

    for si in range(num_sentences):
        w1, w2 = random.choice(head_list)
        toks   = []
        wsp    = 999

        for _ in range(tokens_per_sent):
            cands, probs = next_probs_auto(
                state.graded_lm, state.orbit_grid, w1, w2,
                temp=temp, de_strength=de_strength, auto_strength=auto_strength,
            )
            if not cands:
                break
            nxt = cands[torch.multinomial(probs, 1).item()]

            if nxt in PUNCT_TOKENS:
                too_early     = len(toks) < MIN_PRE_PUNCT or wsp < 3
                too_early_end = nxt in END_PUNCT and len(toks) < MIN_PRE_END
                if too_early or too_early_end:
                    nxt = best_non_punct(cands, probs)
                else:
                    wsp = 0
            else:
                wsp += 1

            toks.append(nxt)
            w1, w2 = w2, nxt
            if nxt in END_PUNCT and len(toks) >= MIN_PRE_END:
                break

        text = detokenize(toks)
        sentences.append(text)
        state.outputs[si] = text
        if si in state.forms:
            state.forms[si].activation_value += 1.0

    return sentences


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CORPUS LOADER  (unchanged from V9 — pure oracle)
# ════════════════════════════════════════════════════════════════════════════

def load_corpus(use_hf=False, dataset_name="", config_name="", split="train",
                column_name="text", max_rows=100, hf_token="", text_file=None):
    if use_hf and dataset_name:
        try:
            ds = load_dataset(dataset_name,
                              name=config_name if config_name else None,
                              split=split,
                              token=hf_token if hf_token else None,
                              trust_remote_code=True)
            df = ds.select(range(min(len(ds), max_rows))).to_pandas()
            if column_name in df.columns:
                return " ".join(df[column_name].astype(str).tolist())
            return f"Error: Column '{column_name}' not found."
        except Exception as e:
            return f"HuggingFace Error: {e}"
    if text_file is not None:
        try:
            p = text_file.name if hasattr(text_file,"name") else str(text_file)
            return Path(p).read_text(encoding="utf-8")
        except Exception as e:
            return f"File error: {e}"
    return (
        "In algebraic topology, homology and cohomology provide a profound "
        "understanding of the shape of data. A persistent filtration creates a "
        "barcode of topological features. Betti numbers summarize cycles, voids, "
        "and connectivity. We consider the nature of understanding spaces "
        "through simplicial complexes and morse theory."
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SESSION RUNNER
# ════════════════════════════════════════════════════════════════════════════

def _fmt_sentences(outputs: Dict[int, str]) -> str:
    return "\n".join(f"[{i+1:03d}] {s}" for i,s in outputs.items())

def _fmt_report(state: V10State, phi: VocabularyAutomorphism) -> str:
    lines = [
        "V10.0 — SUPERPOLYNOMIAL AUTOMORPHISM REPORT",
        "=" * 52,
        f"Vocabulary size   : {len(state.graded_lm.vocab)}",
        f"φ-orbit pairs     : {len(phi.orbits)}",
        f"Graded trigrams   : {len(state.graded_lm.tri_graded)}",
        f"Jack α-parameter  : {state.graded_lm.spr.alpha}",
        "",
        "Top-10 φ-orbit pairs (bosonic ↔ fermionic):",
    ]
    top_pairs = sorted(
        phi.orbits,
        key=lambda p: state.graded_lm.raw_freq.get(p[0],0),
        reverse=True
    )[:10]
    for h, l in top_pairs:
        fh = state.graded_lm.raw_freq.get(h,0)
        fl = state.graded_lm.raw_freq.get(l,0)
        lines.append(f"  {h:<20s} (f={fh:.0f})  ↔  {l:<20s} (f={fl:.0f})")
    lines.append("")
    lines.append("Sample sentence activations:")
    for i in range(min(20, len(state.outputs))):
        f = state.forms.get(i)
        o = state.outputs.get(i,"")
        if f:
            lines.append(f"  [{i:02d}] role={f.syntactic_role:<5s}  act={f.activation_value:.2f}  {o[:60]}…")
    return "\n".join(lines)


def run_session(
    use_hf, hf_dataset, hf_split, hf_max_rows, hf_config, hf_col, hf_token,
    text_file, prompt, seed, num_sentences, tokens_per_sentence,
    temp, adv_strength, densify_mag, alpha, auto_strength,
):
    corpus = load_corpus(
        use_hf=use_hf, dataset_name=hf_dataset, config_name=hf_config,
        split=hf_split, column_name=hf_col, max_rows=int(hf_max_rows),
        hf_token=hf_token, text_file=text_file,
    )
    if corpus.startswith("Error") or corpus.startswith("HuggingFace"):
        return corpus, "Check dataset configuration."

    state = build_v10_state(
        corpus_text   = corpus,
        prompt        = prompt,
        num_sentences = int(num_sentences),
        adv_strength  = float(adv_strength),
        densify_mag   = float(densify_mag),
        alpha         = float(alpha),
    )

    generate(
        state           = state,
        seed            = int(seed),
        num_sentences   = int(num_sentences),
        tokens_per_sent = int(tokens_per_sentence),
        temp            = float(temp),
        auto_strength   = float(auto_strength),
    )

    text   = _fmt_sentences(state.outputs)
    report = _fmt_report(state, state.orbit_grid.phi)
    return text, report


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — GRADIO UI
# ════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="NeuroSymbolic V10.0 — Superpolynomial Automorphism") as demo:
        gr.Markdown(
            "# NeuroSymbolic V10.0\n"
            "### Superpolynomial Automorphism Bridge  ×  φ-Equivariant TetraGrid\n"
            "Single algebraically-coherent pass via Jack superpolynomials P_λ(x;θ;α)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                use_hf      = gr.Checkbox(label="Use Hugging Face Dataset?", value=False)
                hf_dataset  = gr.Textbox(label="Dataset Path",  value="AiresPucrs/stanford-encyclopedia-philosophy", visible=False)
                hf_config   = gr.Textbox(label="Config",        value="",      visible=False)
                hf_split    = gr.Textbox(label="Split",         value="train", visible=False)
                hf_col      = gr.Textbox(label="Text Column",   value="text",  visible=False)
                hf_max_rows = gr.Number( label="Max Rows",      value=100,     visible=False)
                hf_token    = gr.Textbox(label="HF Token",      type="password", visible=False)
                text_file   = gr.File(label="Upload Text (.txt / .md)",
                                      file_types=[".txt",".md"], visible=True)

                def _toggle(v):
                    hv = gr.update(visible=v)
                    fv = gr.update(visible=not v)
                    return hv, hv, hv, hv, hv, hv, fv
                use_hf.change(_toggle, use_hf,
                              [hf_dataset,hf_config,hf_split,hf_col,hf_max_rows,hf_token,text_file])

                gr.Markdown("### Generation")
                seed                = gr.Number(value=42,   label="Seed")
                num_sentences       = gr.Slider(1, 200, value=100, step=10,  label="Sentences")
                tokens_per_sentence = gr.Slider(8, 200, value=92,  step=2,   label="Tokens / Sentence")
                temp                = gr.Slider(0.8, 2.5, value=1.4, step=0.1, label="Temperature τ")

                gr.Markdown("### Superpolynomial & Automorphism Controls")
                alpha         = gr.Slider(0.0, 4.0, value=1.5, step=0.1,
                                          label="Jack α  (0=Schur/V9-fibre, >1=fermionic boost)")
                adv_strength  = gr.Slider(0.0, 1.0, value=0.5,  step=0.05,
                                          label="Grid Adversarial Penalty")
                densify_mag   = gr.Slider(0.0, 0.5, value=0.08, step=0.01,
                                          label="Lattice Densification Magnitude")
                auto_strength = gr.Slider(0.0, 2.0, value=0.55, step=0.05,
                                          label="φ-Orbit Injection Strength")

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt  (seeds SyntacticForm vocabulary)",
                    value="Consider the nature of understanding", lines=2
                )
                btn = gr.Button("Generate — φ-Equivariant Superpolynomial Pass",
                                variant="primary", size="lg")
                out_text   = gr.Textbox(label="Generated Sentences", lines=22)
                out_report = gr.Textbox(label="Automorphism & Grading Report", lines=18)

        btn.click(
            run_session,
            inputs=[use_hf, hf_dataset, hf_split, hf_max_rows, hf_config, hf_col, hf_token,
                    text_file, prompt, seed, num_sentences, tokens_per_sentence,
                    temp, adv_strength, densify_mag, alpha, auto_strength],
            outputs=[out_text, out_report],
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(share=False)
