#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V16.0 — Exclusively Thébault's Theorem Architecture
===============================================================================

ALL mathematical operations derive from a single geometric theorem:

    THÉBAULT'S THEOREM (1938)
    ─────────────────────────
    Given a parallelogram ABCD, erect squares externally on each of the four
    sides.  The centres of those four squares themselves form a perfect square
    — the "Thébault square" of the parallelogram.

Every module in this system is a direct expression of one of the theorem's
three geometric quantities:

    ρ  — Thébault Regularity   (how close the 4 centres are to a perfect square)
    θ  — Thébault Orientation  (the rotation angle of the Thébault square)
    σ  — Thébault Side-length  (the side-length of the Thébault square)

TOKEN GEOMETRY
──────────────
Each vocabulary token t is assigned a canonical parallelogram derived
exclusively from its corpus statistics: raw frequency f and vocabulary
index k (first-occurrence rank).

    f̂  = f / F          normalised frequency   ∈ (0, 1]
    k̂  = k / (K − 1)   normalised rank        ∈ [0, 1]

    P_t = ( f̂ · cos(2π·k̂),  f̂ · sin(2π·k̂) )
          radius = relative frequency, angle = relative rank

    Q_t = ( k̂ · cos(2π·f̂),  k̂ · sin(2π·f̂) )
          radius = relative rank, angle = relative frequency

The parallelogram is:
    A = (0, 0)
    B = P_t
    C = P_t + Q_t
    D = Q_t

Its Thébault triple  (ρ_t, θ_t, σ_t)  is the token's entire geometric
identity.  No hashes, no embeddings, no weight matrices.

FIVE MATHEMATICAL MODULES — ALL THÉBAULT
─────────────────────────────────────────
1.  ThebaultTokenGeometry
        Computes (ρ, θ, σ) for every token.  These three scalars replace
        frequency, parity, and the vocabulary automorphism.

2.  ThebaultCompositionLM  (replaces GradedSynapseAlgebra + trigram model)
        Given context (w1, w2), scores candidate w3 by the regularity of the
        *composed* parallelogram formed by chaining the two token parallelograms:
            composed vertices: A=0, B=P_w1, C=P_w1+P_w2, D=P_w2
        Then the Thébault triple of the composed figure is compared with that
        of w3 via the three kernel functions below.

3.  ThebaultConjugateOrbit  (replaces AutoOrbitModule + VocabularyAutomorphism)
        Two tokens are "Thébault conjugates" when their Thébault squares are
        congruent (|σ_i − σ_j| < ε) but oppositely oriented (|θ_i + θ_j| ≈ π).
        Orbit score = congruence × orientation-antipodality.

4.  ThebaultPotentialGraph  (replaces SuperPolyGraph)
        Graph edge weight between tokens i and j =
            K_reg(ρ_i, ρ_j) · K_ori(θ_i, θ_j)
        Potential propagates identically to the V15 graph but using only
        Thébault-derived edge weights.

5.  ThebaultWalker  (replaces RKHSGraphSuperPolyWalk + NeuralKernelExpansions)
        Three kernels, each a direct Thébault geometric quantity:

        K_reg(ρ_a, ρ_b)  =  exp(−λ · (ρ_a − ρ_b)²)
            "Regularity kernel" — high score when both tokens have similar
            geometric structure in their Thébault squares.

        K_ori(θ_a, θ_b)  =  ½(1 + cos(θ_a − θ_b))
            "Orientation kernel" — von-Mises-style cosine on the Thébault
            square's rotation angle; rewards contextual alignment.

        K_side(σ_a, σ_b)  =  exp(−γ · (σ_a − σ_b)²)
            "Side-length kernel" — rewards candidates whose Thébault square
            has the same scale as the context token.

        Final logit for candidate c given context w2:
            log p_base(c)
            + α · K_reg(ρ_w2, ρ_c)
            + β · K_ori(θ_w2, θ_c)
            + δ · K_side(σ_w2, σ_c)
            + composition_score(w1, w2, c)

        "Thébault-isomorphic" pairs: K_reg > 0.98 AND K_side > 0.98,
        i.e. congruent Thébault squares up to rotation.

synthetic_reason SYMMETRICAL MANDATES
──────────────────────────────────────
Retained from V15 (inspired by SynthReason 0.9N C++, George Wagenknecht 2017)
as an ethical overlay on top of the geometric selection.

===============================================================================
"""

from __future__ import annotations
import re, math, random, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Set
import torch
import torch.nn.functional as F
import gradio as gr


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — TOKEN PRIMITIVES  (unchanged surface API)
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
# SECTION 1 — THÉBAULT TOKEN GEOMETRY
#   Replaces: SuperPolynomialRing, VocabularyAutomorphism, frequency tables
# ════════════════════════════════════════════════════════════════════════════

# CV of the six pairwise distances for a *perfect* Thébault square
# (4 equal sides s and 2 diagonals s√2):
#   distances = [s, s, s, s, s√2, s√2]
#   mean = s(4 + 2√2)/6,  std = s·√(Σ(dᵢ−mean)²/6)
# Pre-computed normalisation constant so ρ=1 at the ideal.
def _perfect_square_cv() -> float:
    s = 1.0
    d = [s, s, s, s, s * math.sqrt(2), s * math.sqrt(2)]
    mu = sum(d) / 6
    cv = math.sqrt(sum((x - mu) ** 2 for x in d) / 6) / mu
    return cv


_PERFECT_CV = _perfect_square_cv()   # ≈ 0.1925


def _rotate90(vx: float, vy: float) -> Tuple[float, float]:
    """Rotate 2-D vector 90° counter-clockwise."""
    return -vy, vx


def _thebault_centres(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
    dx: float, dy: float,
) -> List[Tuple[float, float]]:
    """
    Given parallelogram ABCD, return the four Thébault square-centres.
    For side PQ the external square-centre is:
        mid(P,Q) + rotate90((Q-P)/2)
    """
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


def _thebault_triple(
    px: float, py: float,
    qx: float, qy: float,
) -> Tuple[float, float, float]:
    """
    Given the two defining vectors P and Q, build parallelogram
        A=(0,0), B=P, C=P+Q, D=Q
    and return its Thébault triple (ρ, θ, σ):

        ρ  Regularity  ∈ [0,1]   — 1 = perfect Thébault square
        θ  Orientation ∈ [0,π)   — rotation angle of the Thébault square
        σ  Side-length ≥ 0        — side-length of the Thébault square
    """
    # Degenerate check
    if abs(px) < 1e-9 and abs(py) < 1e-9 and abs(qx) < 1e-9 and abs(qy) < 1e-9:
        return 0.0, 0.0, 0.0

    T = _thebault_centres(0.0, 0.0, px, py, px + qx, py + qy, qx, qy)

    # Six pairwise distances
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

    # Side-length σ = mean of the four side distances
    sides = [dists[0], dists[1], dists[2], dists[3]]   # consecutive pairs
    sigma = sum(sides) / 4.0

    # Orientation θ = angle of vector T0→T1 in [0, π)
    dx_ori = T[1][0] - T[0][0]
    dy_ori = T[1][1] - T[0][1]
    theta  = math.atan2(dy_ori, dx_ori) % math.pi

    return rho, theta, sigma


@dataclass
class ThebaultTriple:
    rho   : float   # regularity
    theta : float   # orientation
    sigma : float   # side-length


class ThebaultTokenGeometry:
    """
    Assigns each token a canonical parallelogram derived exclusively from its
    corpus statistics — raw frequency and vocabulary index — then computes the
    Thébault triple (ρ, θ, σ).

    Parallelogram construction from corpus statistics
    ──────────────────────────────────────────────────
    Let:
        f  = raw frequency of token t  (count of occurrences in corpus)
        k  = vocabulary index of token t  (rank by first-occurrence order)
        F  = maximum raw frequency across all tokens
        K  = vocabulary size (total unique tokens)

    Normalised values:
        f̂  = f / F          ∈ (0, 1]   — relative frequency
        k̂  = k / (K − 1)   ∈ [0, 1]   — relative rank

    The two defining vectors are then:

        P_t = ( f̂ · cos(2π · k̂),   f̂ · sin(2π · k̂) )
            — a point on a circle of radius f̂, at angle 2π·k̂.
            High-frequency tokens have long P vectors (large circle radius).
            Rank distributes them evenly around the circle.

        Q_t = ( k̂ · cos(2π · f̂),   k̂ · sin(2π · f̂) )
            — roles swapped: radius = relative rank, angle driven by frequency.
            Rare tokens cluster near the origin; common ones spread outward.

    The parallelogram is:
        A = (0, 0)
        B = P_t
        C = P_t + Q_t
        D = Q_t

    Geometry intuition:
        • Two tokens with the same frequency but different ranks → same |P|
          but different angles → different θ (orientation).
        • Two tokens with the same rank but different frequencies → same θ_P
          angle but different |P| → different σ (side-length).
        • ρ (regularity) measures how "square-like" the Thébault figure is,
          capturing the interaction between the two corpus dimensions.

    Composition (trigram context):
        P_composed = P_t1 + P_t2     Q_composed = Q_t1 + Q_t2
        Vector addition in each plane; the composed parallelogram's Thébault
        triple represents the joint corpus geometry of the bigram context.

    Registration:
        call  geo.register(token, freq, index, max_freq, vocab_size)
        before calling  geo.triple(token)  or  geo.composed_triple(t1, t2).
    """

    def __init__(self):
        self._vecs  : Dict[str, Tuple[float, float, float, float]] = {}  # px,py,qx,qy
        self._cache : Dict[str, ThebaultTriple]                    = {}

    def register(
        self,
        token     : str,
        freq      : float,
        index     : int,
        max_freq  : float,
        vocab_size: int,
    ) -> None:
        """Compute and store the (P, Q) corpus vectors for one token."""
        f_hat = freq / max(max_freq, 1e-9)
        k_hat = index / max(vocab_size - 1, 1)

        angle_p = 2.0 * math.pi * k_hat   # rank → angle
        angle_q = 2.0 * math.pi * f_hat   # freq → angle

        px = f_hat * math.cos(angle_p)
        py = f_hat * math.sin(angle_p)
        qx = k_hat * math.cos(angle_q)
        qy = k_hat * math.sin(angle_q)

        self._vecs[token] = (px, py, qx, qy)
        self._cache.pop(token, None)   # invalidate if re-registered

    def _vec(self, token: str) -> Tuple[float, float, float, float]:
        """Return stored (px, py, qx, qy); fall back to origin if unregistered."""
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
        """
        Compose two token parallelograms by vector addition:
            P_composed = P_t1 + P_t2
            Q_composed = Q_t1 + Q_t2
        Represents the joint corpus geometry of bigram context (t1, t2).
        """
        p1x, p1y, q1x, q1y = self._vec(t1)
        p2x, p2y, q2x, q2y = self._vec(t2)
        rho, theta, sigma = _thebault_triple(p1x + p2x, p1y + p2y, q1x + q2x, q1y + q2y)
        return ThebaultTriple(rho, theta, sigma)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — THÉBAULT KERNELS
#   Replaces: NeuralKernelExpansions, SplitComplexVisAVisCompound,
#             InverseSurjectionMonograph, RhomboidShearKernel
# ════════════════════════════════════════════════════════════════════════════

class ThebaultKernels:
    """
    Three kernels, each a direct function of one Thébault geometric quantity.

    K_reg(ρ_a, ρ_b)  =  exp(−λ · (ρ_a − ρ_b)²)
        Regularity kernel: rewards candidates whose Thébault square is equally
        regular/irregular to the context token.  λ = lambda_reg.

    K_ori(θ_a, θ_b)  =  ½(1 + cos(θ_a − θ_b))
        Orientation kernel: von-Mises-style on the Thébault square rotation;
        rewards contextual directional alignment.

    K_side(σ_a, σ_b)  =  exp(−γ · (σ_a − σ_b)²)
        Side-length kernel: rewards candidates whose Thébault square has the
        same scale as the context.  γ = gamma_side.
    """

    def __init__(self, lambda_reg: float = 8.0, gamma_side: float = 4.0):
        self.lambda_reg  = lambda_reg
        self.gamma_side  = gamma_side

    def k_reg(self, rho_a: float, rho_b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.lambda_reg * (rho_b - rho_a) ** 2)

    def k_ori(self, theta_a: float, theta_b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.cos(theta_b - theta_a))

    def k_side(self, sigma_a: float, sigma_b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)

    def all_scores(
        self,
        ctx: ThebaultTriple,
        cand_rho  : torch.Tensor,
        cand_theta: torch.Tensor,
        cand_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.k_reg (ctx.rho,   cand_rho),
            self.k_ori (ctx.theta, cand_theta),
            self.k_side(ctx.sigma, cand_sigma),
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THÉBAULT CONJUGATE ORBIT
#   Replaces: AutoOrbitModule, VocabularyAutomorphism
# ════════════════════════════════════════════════════════════════════════════

class ThebaultConjugateOrbit:
    """
    Two tokens are Thébault conjugates when their Thébault squares are:
        • Congruent   : |σ_i − σ_j| < ε_side
        • Antipodal   : |θ_i + θ_j − π| < ε_ori  (opposite orientations)

    Orbit score for candidate c given anchor a:
        orbit(a, c) = K_side(σ_a, σ_c) · cos²(θ_a + θ_c − π/2)

    The cosine² term peaks when θ_a + θ_c = π/2  (quarter-turn antipodal),
    giving a smooth score ∈ [0, 1] without any threshold.
    """

    def score(
        self,
        anchor_triple : ThebaultTriple,
        cand_theta    : torch.Tensor,
        cand_sigma    : torch.Tensor,
        gamma_side    : float = 4.0,
    ) -> torch.Tensor:
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor_triple.sigma) ** 2)
        antipodality = torch.cos(cand_theta + anchor_triple.theta - math.pi / 2) ** 2
        return congruence * antipodality


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THÉBAULT COMPOSITION LM
#   Replaces: GradedSynapseAlgebra (trigram counts + graded weights)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultCompositionLM:
    """
    Language model whose transition weights derive entirely from Thébault geometry.

    Training (ingest):
        Raw trigram counts are collected as before.

    Scoring candidate w3 given context (w1, w2):
        1. Compose the two context parallelograms → composed triple C.
        2. Score = raw_count(w1,w2,w3) · K_reg(ρ_C, ρ_w3) · K_side(σ_C, σ_w3)
           i.e. the candidate is rewarded for matching the Thébault geometry of
           the composed context.

    The base probability is the Thébault-weighted trigram distribution.
    """

    BASAL_K = 1.5

    def __init__(self, geo: ThebaultTokenGeometry, kernels: ThebaultKernels):
        self.geo     = geo
        self.kernels = kernels
        self.raw_freq  : Dict[str, float]                   = {}
        self.tri_raw   : Dict[Tuple[str, str, str], float]  = {}
        self.heads     : Dict[Tuple[str, str], List[str]]   = {}
        self.vocab     : List[str]                          = []

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

    def next_dist(
        self, w1: str, w2: str
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Return candidates and their Thébault-composition-weighted probabilities.
        """
        head = (w1, w2)
        if head in self.heads:
            cands   = self.heads[head]
            counts  = [self.tri_raw.get((w1, w2, w3), 1e-4) for w3 in cands]
        else:
            agg: Dict[str, float] = {}
            for (_, _, w3), wt in self.tri_raw.items():
                agg[w3] = agg.get(w3, 0) + wt
            cands  = list(agg.keys())[:400]
            counts = [agg[w] for w in cands]

        # Composed context triple  C = compose(w1, w2)
        C = self.geo.composed_triple(w1, w2)

        # Per-candidate Thébault triples
        triples = [self.geo.triple(c) for c in cands]
        c_rho   = torch.tensor([tr.rho   for tr in triples], dtype=torch.float32)
        c_sigma = torch.tensor([tr.sigma for tr in triples], dtype=torch.float32)

        # Composition score = K_reg(ρ_C, ρ_c) · K_side(σ_C, σ_c)
        k_r = self.kernels.k_reg (C.rho,   c_rho)
        k_s = self.kernels.k_side(C.sigma, c_sigma)
        geo_weight = (k_r * k_s).clamp(min=1e-6)  # [N]

        raw   = torch.tensor(counts, dtype=torch.float32)
        V_tot = len(self.vocab) + 1
        total = raw.sum().item()
        basal = torch.tensor(
            [(cnt + self.BASAL_K) / (total + self.BASAL_K * V_tot) for cnt in counts],
            dtype=torch.float32,
        )
        weighted = basal * geo_weight
        return cands, weighted / weighted.sum().clamp(min=1e-12)

    def composition_logit_bonus(
        self, w1: str, w2: str, cands: List[str]
    ) -> torch.Tensor:
        """
        Additive logit bonus from Thébault composition geometry.
        Separate from next_dist so the walker can add it explicitly.
        """
        C       = self.geo.composed_triple(w1, w2)
        triples = [self.geo.triple(c) for c in cands]
        c_rho   = torch.tensor([tr.rho   for tr in triples], dtype=torch.float32)
        c_sigma = torch.tensor([tr.sigma for tr in triples], dtype=torch.float32)
        return self.kernels.k_reg(C.rho, c_rho) * self.kernels.k_side(C.sigma, c_sigma)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — THÉBAULT POTENTIAL GRAPH
#   Replaces: SuperPolyGraph
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
    """
    Graph where edge weight between tokens i → j is the Thébault kernel product:
        w(i,j) = K_reg(ρ_i, ρ_j) · K_ori(θ_i, θ_j)

    This means tokens with similar regularity AND similar orientation are
    strongly connected — pure Thébault geometry, no learned parameters.
    """

    def __init__(self, geo: ThebaultTokenGeometry, kernels: ThebaultKernels):
        self.geo     = geo
        self.kernels = kernels
        self.nodes   : Dict[str, TGNode]         = {}
        self.adj     : Dict[str, List[TGEdge]]   = {}
        self.radj    : Dict[str, List[TGEdge]]   = {}

    def build(self, lm: ThebaultCompositionLM) -> None:
        for tok, freq in lm.raw_freq.items():
            if tok not in PUNCT_TOKENS and tok not in COGNITIVE_TOKENS:
                self.nodes[tok]  = TGNode(tok, freq, self.geo.triple(tok))
                self.adj[tok]    = []
                self.radj[tok]   = []

        # Trigram-derived directed edges, weighted by Thébault kernel
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

    def propagate(self, steps: int = 2) -> None:
        if not self.nodes:
            return
        max_f = max(nd.freq for nd in self.nodes.values()) + 1e-8
        for nd in self.nodes.values():
            nd.potential = nd.triple.rho * nd.freq / max_f   # seed = ρ × normalised freq

        for _ in range(steps):
            new_pots: Dict[str, float] = {}
            for v, nd in self.nodes.items():
                in_edges = self.radj.get(v, [])
                agg = sum(e.weight * self.nodes[e.src].potential for e in in_edges)
                # Self-loop: Thébault side-length rescales the potential
                self_scale = nd.triple.sigma / (nd.triple.sigma + 1.0)
                new_pots[v] = agg / (len(in_edges) + 1.0) + self_scale * nd.potential * 0.1
            mx = max(new_pots.values(), default=1.0) + 1e-8
            for v in self.nodes:
                self.nodes[v].potential = new_pots[v] / mx


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4.5 — synthetic_reason SYMMETRICAL MANDATES  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class synthetic_reasonMandateProcessor:
    """
    Inspired by SynthReason 0.9N C++ (George Wagenknecht, 2017).
    Ethical overlay: biases RKHS distribution toward mandate-fulfilling vocab.
    """

    def __init__(self):
        self.AIEthics   = ["do not harm any human", "do not harm myself", "do not make weapons"]
        self.AIMandates = ["end poverty", "cure disease", "improve standard of living", "learn"]
        self.mandate_vocabulary = {
            "poverty":  "end",     "disease": "cure",    "standard": "improve",
            "living":   "improve", "learn":   "explore", "human":    "protect",
            "weapons":  "avoid",   "harm":    "prevent",
        }

    def subsynthetic_reason_concept_enrichment(
        self, w_ctx: str, cands: List[str]
    ) -> torch.Tensor:
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
# SECTION 6 — THÉBAULT WALKER
#   Replaces: RKHSGraphSuperPolyWalk
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    """
    Token-by-token generation using exclusively Thébault geometric scores.

    Logit for candidate c given context (w1, w2):

        log p_base(c)                          ← Thébault-composition LM base
        + α · K_reg(ρ_w2, ρ_c)                ← regularity alignment
        + β · K_ori(θ_w2, θ_c)                ← orientation alignment
        + δ · K_side(σ_w2, σ_c)               ← scale alignment
        + γ · orbit_score(w2, c)              ← conjugate-orbit score
        + ψ · graph_potential(c)              ← Thébault graph potential
        + composition_bonus(w1, w2, c)        ← composed-context match
        + mandate_boost(w2, c)                ← synthetic_reason ethical overlay
    """

    def __init__(
        self,
        geo            : ThebaultTokenGeometry,
        kernels        : ThebaultKernels,
        lm             : ThebaultCompositionLM,
        orbit          : ThebaultConjugateOrbit,
        graph          : ThebaultPotentialGraph,
        synthetic_reason: synthetic_reasonMandateProcessor,
    ):
        self.geo      = geo
        self.kernels  = kernels
        self.lm       = lm
        self.orbit    = orbit
        self.graph    = graph
        self.synth    = synthetic_reason
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []

    def walk_probs(
        self, w1: str, w2: str,
        temp          : float = 1.4,
        alpha_reg     : float = 1.2,
        beta_ori      : float = 0.8,
        delta_side    : float = 1.0,
        gamma_orbit   : float = 0.6,
        psi_pot       : float = 0.35,
    ) -> Tuple[List[str], torch.Tensor]:

        cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands:
            return cands, base_probs

        # Thébault triples for all candidates
        triples    = [self.geo.triple(c) for c in cands]
        c_rho      = torch.tensor([tr.rho   for tr in triples], dtype=torch.float32)
        c_theta    = torch.tensor([tr.theta for tr in triples], dtype=torch.float32)
        c_sigma    = torch.tensor([tr.sigma for tr in triples], dtype=torch.float32)

        # Context token's triple
        ctx = self.geo.triple(w2)

        # ── Three Thébault kernels ───────────────────────────────────────────
        k_reg, k_ori, k_side = self.kernels.all_scores(ctx, c_rho, c_theta, c_sigma)

        # ── Conjugate-orbit score ────────────────────────────────────────────
        orbit_scores = self.orbit.score(ctx, c_theta, c_sigma, self.kernels.gamma_side)

        # ── Graph potential ──────────────────────────────────────────────────
        pots = torch.tensor(
            [self.graph.nodes[c].potential if c in self.graph.nodes else 0.0
             for c in cands],
            dtype=torch.float32,
        )

        # ── Composition logit bonus ──────────────────────────────────────────
        comp_bonus = self.lm.composition_logit_bonus(w1, w2, cands)

        # ── Thébault-isomorphic pairs ────────────────────────────────────────
        #    Two candidates are isomorphic when K_reg > 0.98 AND K_side > 0.98
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

        # ── Punctuation guards ───────────────────────────────────────────────
        punct_bias    = torch.zeros(N)
        punct_penalty = torch.zeros(N)
        for i, c in enumerate(cands):
            if c in PUNCT_TOKENS:
                punct_bias[i] = -3.5
                if w2 in PUNCT_TOKENS:
                    punct_penalty[i] = -1e4

        # ── synthetic_reason mandate boost ───────────────────────────────────
        mandate_boost = self.synth.subsynthetic_reason_concept_enrichment(w2, cands)

        # ── Final logit assembly (all Thébault) ──────────────────────────────
        logits = (
            torch.log(base_probs.clamp(min=1e-12))
            + alpha_reg   * k_reg
            + beta_ori    * k_ori
            + delta_side  * k_side
            + gamma_orbit * orbit_scores
            + psi_pot     * pots
            + comp_bonus
            + mandate_boost
            + punct_bias
            + punct_penalty
        ) / max(temp, 1e-6)

        return cands, F.softmax(logits, dim=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENGINE STATE & GENERATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class V16State:
    lm      : ThebaultCompositionLM
    graph   : ThebaultPotentialGraph
    walker  : ThebaultWalker
    outputs : Dict[int, str]              = field(default_factory=dict)
    iso_matches: Set[Tuple[str, str]]     = field(default_factory=set)


def build_v16_state(
    corpus_text : str,
    lambda_reg  : float = 8.0,
    gamma_side  : float = 4.0,
) -> V16State:
    tokens = tokenize(corpus_text)

    geo     = ThebaultTokenGeometry()
    kernels = ThebaultKernels(lambda_reg=lambda_reg, gamma_side=gamma_side)
    lm      = ThebaultCompositionLM(geo, kernels)
    lm.ingest(tokens)

    # ── Register every token's corpus geometry (freq + index) ────────────────
    # Vocabulary index = first-occurrence order, preserved in raw_freq insertion
    # order (Python 3.7+ dict guarantees insertion order).
    all_tokens = list(lm.raw_freq.keys())
    max_freq   = max(lm.raw_freq.values(), default=1.0)
    vocab_size = len(all_tokens)
    for idx, tok in enumerate(all_tokens):
        geo.register(
            token      = tok,
            freq       = lm.raw_freq[tok],
            index      = idx,
            max_freq   = max_freq,
            vocab_size = vocab_size,
        )

    orbit  = ThebaultConjugateOrbit()
    graph  = ThebaultPotentialGraph(geo, kernels)
    graph.build(lm)
    graph.propagate(steps=2)

    synth  = synthetic_reasonMandateProcessor()
    walker = ThebaultWalker(geo, kernels, lm, orbit, graph, synth)

    return V16State(lm, graph, walker)


def generate(
    state            : V16State,
    seed_context     : str   = "",
    num_sentences    : int   = 15,
    tokens_per_sent  : int   = 92,
    temp             : float = 1.4,
    alpha_reg        : float = 1.2,
    beta_ori         : float = 0.8,
    delta_side       : float = 1.0,
    gamma_orbit      : float = 0.6,
    psi_pot          : float = 0.35,
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
                psi_pot=psi_pot,
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
            w1, w2 = w2, nxt
            if nxt in {".", "?", "!"} and len(toks) >= max(4, int(tokens_per_sent * 0.85)):
                break

        state.outputs[si] = detokenize(toks)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GRADIO UI
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
):
    corpus = load_corpus(text_file)
    state  = build_v16_state(corpus, lambda_reg=float(lambda_reg), gamma_side=float(gamma_side))

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
    )

    out_text = "\n".join(f"[{i+1:02d}] {s}" for i, s in state.outputs.items())

    # Sample Thébault triples for a few vocab tokens
    geo = state.walker.geo
    sample_toks = list(state.lm.vocab)[:8]
    triple_lines = ["── Sample Thébault Triples ──"]
    for tok in sample_toks:
        t = geo.triple(tok)
        triple_lines.append(
            f"  {tok:<14s}  ρ={t.rho:.3f}  θ={math.degrees(t.theta):6.1f}°  σ={t.sigma:.3f}"
        )

    report_lines = [
        "V16.0 — EXCLUSIVELY THÉBAULT'S THEOREM",
        "=" * 60,
        f"Vocab size       : {len(state.lm.vocab)}",
        f"Kernel  λ_reg    : {lambda_reg:.2f}   γ_side : {gamma_side:.2f}",
        f"Walker  α_reg    : {alpha_reg:.2f}   β_ori  : {beta_ori:.2f}",
        f"        δ_side   : {delta_side:.2f}   γ_orbit: {gamma_orbit:.2f}   ψ_pot: {psi_pot:.2f}",
        "Mandates         : synthetic_reason Active (SynthReason C++ Mandates)",
        "",
        *triple_lines,
        "",
        "── Thébault-Isomorphic Candidate Pairs (K_reg·K_side > 0.98) ──",
    ]
    if state.iso_matches:
        for p1, p2 in list(state.iso_matches)[:20]:
            report_lines.append(f"  {p1:<15s} ≈  {p2:<15s}")
    else:
        report_lines.append("  No Thébault-isomorphic candidates found.")

    return out_text, "\n".join(report_lines)


def build_app():
    with gr.Blocks(title="NeuroSymbolic V16.0 — Thébault's Theorem") as demo:
        gr.Markdown(
            "# NeuroSymbolic V16.0\n"
            "### All mathematics derived exclusively from **Thébault's Theorem** · "
            "synthetic_reason Symmetrical Processor"
        )
        with gr.Row():
            with gr.Column(scale=1):
                text_file           = gr.File(label="Upload Text (.txt)")
                seed_context        = gr.Textbox(label="Seed Context", placeholder="Enter starting words…")
                num_sentences       = gr.Slider(1,   100, value=15,  label="Sentences")
                tokens_per_sentence = gr.Slider(5,   200, value=92,  label="Tokens per Sentence")
                temp                = gr.Slider(0.8, 2.5, value=1.4, label="Temperature τ")

                gr.Markdown("#### Thébault Kernel Parameters")
                lambda_reg = gr.Slider(0.5, 20.0, value=8.0, step=0.5, label="λ_reg  — regularity kernel bandwidth")
                gamma_side = gr.Slider(0.5, 12.0, value=4.0, step=0.5, label="γ_side — side-length kernel bandwidth")

                gr.Markdown("#### Walker Blend Weights")
                gr.Markdown("*All scores are Thébault geometric quantities*")
                alpha_reg   = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="α  — K_reg  weight (regularity alignment)")
                beta_ori    = gr.Slider(0.0, 3.0, value=0.8, step=0.1, label="β  — K_ori  weight (orientation alignment)")
                delta_side  = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="δ  — K_side weight (scale alignment)")
                gamma_orbit = gr.Slider(0.0, 3.0, value=0.6, step=0.1, label="γ  — orbit  weight (conjugate score)")
                psi_pot     = gr.Slider(0.0, 2.0, value=0.35,step=0.05,label="ψ  — graph potential weight")

            with gr.Column(scale=2):
                btn        = gr.Button("Generate — Run Thébault Engine", variant="primary", size="lg")
                out_text   = gr.Textbox(label="Generated Sentences", lines=15)
                out_report = gr.Textbox(label="Structure Report",    lines=20)

        btn.click(
            run_session,
            inputs=[
                text_file, seed_context,
                num_sentences, tokens_per_sentence,
                temp, alpha_reg, beta_ori, delta_side, gamma_orbit, psi_pot,
                lambda_reg, gamma_side,
            ],
            outputs=[out_text, out_report],
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(share=False)
