#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V16.1 — Thébault's Theorem + MRV Constraint Heuristic
===============================================================================

Adds MRVConstraintFilter to the Thébault architecture.

MRV (MINIMUM REMAINING VALUES) HEURISTIC
─────────────────────────────────────────
Borrowed from constraint-satisfaction problems (CSP), MRV prioritises the
candidate token that has the *fewest valid successors* in the Thébault graph —
i.e. the most "constrained" next choice.

Rationale inside this system:
  • A token with many Thébault-compatible successors is "free" — it leaves
    the walk in a large open region of the geometry space.
  • A token with few compatible successors is "tight" — choosing it commits
    the walk to a narrower geometric corridor.
  • MRV biases toward tight tokens, preventing the walk from drifting into
    sparse regions with no coherent continuations (reducing dead-ends and
    degenerate terminal symbol repetition).

Implementation:
  For each candidate c, count how many vocabulary tokens v satisfy:
      K_reg(ρ_c, ρ_v) > mrv_threshold  AND  K_side(σ_c, σ_v) > mrv_threshold
  This is the "domain size" of c.  MRV score = 1 / (domain_size + ε),
  so tokens with smaller domains get higher MRV scores.

  The MRV score is added as an additive logit bonus (weight ζ_mrv).

All other mathematics remain exclusively Thébault-derived.
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
# SECTION 0 — TOKEN PRIMITIVES
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


def _thebault_centres(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
    dx: float, dy: float,
) -> List[Tuple[float, float]]:
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
# SECTION 2 — THÉBAULT KERNELS
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
# SECTION 2.5 — MRV CONSTRAINT FILTER  ← NEW
# ════════════════════════════════════════════════════════════════════════════

class MRVConstraintFilter:
    """
    Minimum Remaining Values (MRV) heuristic for candidate token selection.

    For each candidate token c, the "domain size" is the number of vocabulary
    tokens v that are Thébault-compatible with c:

        compatible(c, v)  iff  K_reg(ρ_c, ρ_v) > threshold
                          AND  K_side(σ_c, σ_v) > threshold

    MRV score  =  1 / (domain_size(c) + ε)

    Tokens with smaller domains score higher — they are more constrained and
    choosing them keeps the generation in a coherent geometric corridor.

    The filter also supports a hard cap: candidates whose domain_size exceeds
    mrv_cap_ratio × mean_domain_size are soft-penalised (not hard-removed) so
    the walk never gets stuck.

    Parameters
    ──────────
    threshold     : float  — K_reg / K_side threshold for compatibility (default 0.5)
    mrv_cap_ratio : float  — soft-penalty multiplier for over-free tokens (default 2.0)
    max_vocab_scan: int    — max vocab tokens scanned per candidate for speed (default 300)
    """

    def __init__(
        self,
        threshold     : float = 0.50,
        mrv_cap_ratio : float = 2.0,
        max_vocab_scan: int   = 300,
    ):
        self.threshold      = threshold
        self.mrv_cap_ratio  = mrv_cap_ratio
        self.max_vocab_scan = max_vocab_scan

        # Cached vocab-level tensors; rebuilt when vocab changes
        self._vocab_rho   : torch.Tensor | None = None
        self._vocab_sigma : torch.Tensor | None = None
        self._vocab_tokens: List[str]            = []

    # ── Cache the vocabulary Thébault arrays ────────────────────────────────
    def prime(
        self,
        vocab  : List[str],
        geo    : ThebaultTokenGeometry,
    ) -> None:
        """Pre-compute vocab rho/sigma tensors (call once after build_v16_state)."""
        scan  = vocab[:self.max_vocab_scan]
        trips = [geo.triple(v) for v in scan]
        self._vocab_rho    = torch.tensor([t.rho   for t in trips], dtype=torch.float32)
        self._vocab_sigma  = torch.tensor([t.sigma for t in trips], dtype=torch.float32)
        self._vocab_tokens = scan

    # ── Compute MRV scores for a list of candidates ─────────────────────────
    def mrv_scores(
        self,
        cands  : List[str],
        geo    : ThebaultTokenGeometry,
        kernels: ThebaultKernels,
    ) -> torch.Tensor:
        """
        Returns a tensor of shape [len(cands)] with MRV logit bonuses.
        Higher = more constrained = preferred by MRV.
        """
        if self._vocab_rho is None or len(self._vocab_tokens) == 0:
            return torch.zeros(len(cands))

        v_rho   = self._vocab_rho    # [V]
        v_sigma = self._vocab_sigma  # [V]
        thr     = self.threshold

        domain_sizes = []
        for c in cands:
            tr      = geo.triple(c)
            k_r     = kernels.k_reg (tr.rho,   v_rho)    # [V]
            k_s     = kernels.k_side(tr.sigma, v_sigma)  # [V]
            compat  = ((k_r > thr) & (k_s > thr)).sum().item()
            domain_sizes.append(float(compat))

        ds     = torch.tensor(domain_sizes, dtype=torch.float32)
        mean_d = ds.mean().item() + 1e-6

        # MRV score: inverse domain size, normalised
        mrv = 1.0 / (ds + 1.0)

        # Soft-penalise tokens whose domain is much larger than average
        over_free = ds > (self.mrv_cap_ratio * mean_d)
        mrv[over_free] *= 0.5   # halve score for unconstrained tokens

        # Normalise to [0, 1]
        lo, hi = mrv.min(), mrv.max()
        if (hi - lo).item() > 1e-8:
            mrv = (mrv - lo) / (hi - lo)

        return mrv   # [N_cands]

    # ── Diagnostic summary ──────────────────────────────────────────────────
    def domain_report(
        self,
        cands  : List[str],
        geo    : ThebaultTokenGeometry,
        kernels: ThebaultKernels,
        top_n  : int = 8,
    ) -> str:
        if self._vocab_rho is None:
            return "MRV filter not primed."
        v_rho   = self._vocab_rho
        v_sigma = self._vocab_sigma
        rows    = []
        for c in cands[:top_n]:
            tr     = geo.triple(c)
            k_r    = kernels.k_reg (tr.rho,   v_rho)
            k_s    = kernels.k_side(tr.sigma, v_sigma)
            dom    = int(((k_r > self.threshold) & (k_s > self.threshold)).sum().item())
            rows.append((c, dom))
        rows.sort(key=lambda x: x[1])
        return "\n".join(f"  {c:<16s}  domain={d}" for c, d in rows)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THÉBAULT CONJUGATE ORBIT
# ════════════════════════════════════════════════════════════════════════════

class ThebaultConjugateOrbit:
    def score(self, anchor_triple, cand_theta, cand_sigma, gamma_side=4.0):
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor_triple.sigma) ** 2)
        antipodality = torch.cos(cand_theta + anchor_triple.theta - math.pi / 2) ** 2
        return congruence * antipodality


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THÉBAULT COMPOSITION LM
# ════════════════════════════════════════════════════════════════════════════

class ThebaultCompositionLM:
    BASAL_K = 1.5

    def __init__(self, geo: ThebaultTokenGeometry, kernels: ThebaultKernels):
        self.geo     = geo
        self.kernels = kernels
        self.raw_freq : Dict[str, float]                  = {}
        self.tri_raw  : Dict[Tuple[str, str, str], float] = {}
        self.heads    : Dict[Tuple[str, str], List[str]]  = {}
        self.vocab    : List[str]                         = []

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

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        head = (w1, w2)
        if head in self.heads:
            cands  = self.heads[head]
            counts = [self.tri_raw.get((w1, w2, w3), 1e-4) for w3 in cands]
        else:
            agg: Dict[str, float] = {}
            for (_, _, w3), wt in self.tri_raw.items():
                agg[w3] = agg.get(w3, 0) + wt
            cands  = list(agg.keys())[:400]
            counts = [agg[w] for w in cands]

        C = self.geo.composed_triple(w1, w2)
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
# SECTION 5 — THÉBAULT POTENTIAL GRAPH
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
        self.geo   = geo
        self.kernels = kernels
        self.nodes : Dict[str, TGNode]       = {}
        self.adj   : Dict[str, List[TGEdge]] = {}
        self.radj  : Dict[str, List[TGEdge]] = {}

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
            nd.potential = nd.triple.rho * nd.freq / max_f
        for _ in range(steps):
            new_pots: Dict[str, float] = {}
            for v, nd in self.nodes.items():
                in_edges = self.radj.get(v, [])
                agg = sum(e.weight * self.nodes[e.src].potential for e in in_edges)
                self_scale = nd.triple.sigma / (nd.triple.sigma + 1.0)
                new_pots[v] = agg / (len(in_edges) + 1.0) + self_scale * nd.potential * 0.1
            mx = max(new_pots.values(), default=1.0) + 1e-8
            for v in self.nodes:
                self.nodes[v].potential = new_pots[v] / mx


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4.5 — synthetic_reason SYMMETRICAL MANDATES
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
# SECTION 6 — THÉBAULT WALKER  (updated with MRV)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    """
    Token-by-token generation using Thébault geometric scores + MRV heuristic.

    Logit for candidate c given context (w1, w2):

        log p_base(c)                          ← Thébault-composition LM base
        + α · K_reg(ρ_w2, ρ_c)                ← regularity alignment
        + β · K_ori(θ_w2, θ_c)                ← orientation alignment
        + δ · K_side(σ_w2, σ_c)               ← scale alignment
        + γ · orbit_score(w2, c)              ← conjugate-orbit score
        + ψ · graph_potential(c)              ← Thébault graph potential
        + composition_bonus(w1, w2, c)        ← composed-context match
        + ζ · mrv_score(c)          ← MRV: prefer constrained tokens  ← NEW
        + mandate_boost(w2, c)                ← synthetic_reason ethical overlay
    """

    def __init__(self, geo, kernels, lm, orbit, graph, synthetic_reason, mrv_filter):
        self.geo      = geo
        self.kernels  = kernels
        self.lm       = lm
        self.orbit    = orbit
        self.graph    = graph
        self.synth    = synthetic_reason
        self.mrv      = mrv_filter          # ← MRV filter
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []

    def walk_probs(
        self, w1: str, w2: str,
        temp          : float = 1.4,
        alpha_reg     : float = 1.2,
        beta_ori      : float = 0.8,
        delta_side    : float = 1.0,
        gamma_orbit   : float = 0.6,
        psi_pot       : float = 0.35,
        zeta_mrv      : float = 0.9,       # ← MRV weight
    ) -> Tuple[List[str], torch.Tensor]:

        cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands:
            return cands, base_probs

        triples    = [self.geo.triple(c) for c in cands]
        c_rho      = torch.tensor([tr.rho   for tr in triples], dtype=torch.float32)
        c_theta    = torch.tensor([tr.theta for tr in triples], dtype=torch.float32)
        c_sigma    = torch.tensor([tr.sigma for tr in triples], dtype=torch.float32)

        ctx = self.geo.triple(w2)

        k_reg, k_ori, k_side = self.kernels.all_scores(ctx, c_rho, c_theta, c_sigma)

        orbit_scores = self.orbit.score(ctx, c_theta, c_sigma, self.kernels.gamma_side)

        pots = torch.tensor(
            [self.graph.nodes[c].potential if c in self.graph.nodes else 0.0
             for c in cands],
            dtype=torch.float32,
        )

        comp_bonus = self.lm.composition_logit_bonus(w1, w2, cands)

        # ── MRV scores ───────────────────────────────────────────────────────
        mrv_scores = self.mrv.mrv_scores(cands, self.geo, self.kernels)

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
            + alpha_reg   * k_reg
            + beta_ori    * k_ori
            + delta_side  * k_side
            + gamma_orbit * orbit_scores
            + psi_pot     * pots
            + comp_bonus
            + zeta_mrv    * mrv_scores      # ← MRV contribution
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
    lm          : ThebaultCompositionLM
    graph       : ThebaultPotentialGraph
    walker      : ThebaultWalker
    mrv_filter  : MRVConstraintFilter
    outputs     : Dict[int, str]          = field(default_factory=dict)
    iso_matches : Set[Tuple[str, str]]    = field(default_factory=set)


def build_v16_state(
    corpus_text  : str,
    lambda_reg   : float = 8.0,
    gamma_side   : float = 4.0,
    mrv_threshold: float = 0.50,
    mrv_cap_ratio: float = 2.0,
) -> V16State:
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

    # ── Build & prime the MRV filter ─────────────────────────────────────────
    mrv_filter = MRVConstraintFilter(
        threshold      = mrv_threshold,
        mrv_cap_ratio  = mrv_cap_ratio,
        max_vocab_scan = min(300, vocab_size),
    )
    mrv_filter.prime(lm.vocab, geo)

    synth  = synthetic_reasonMandateProcessor()
    walker = ThebaultWalker(geo, kernels, lm, orbit, graph, synth, mrv_filter)

    return V16State(lm, graph, walker, mrv_filter)


def generate(
    state           : V16State,
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
                psi_pot=psi_pot, zeta_mrv=zeta_mrv,
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
    zeta_mrv, mrv_threshold, mrv_cap_ratio,
):
    corpus = load_corpus(text_file)
    state  = build_v16_state(
        corpus,
        lambda_reg    = float(lambda_reg),
        gamma_side    = float(gamma_side),
        mrv_threshold = float(mrv_threshold),
        mrv_cap_ratio = float(mrv_cap_ratio),
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
    )

    out_text = "\n".join(f"[{i+1:02d}] {s}" for i, s in state.outputs.items())

    # Sample Thébault triples + MRV domain sizes
    geo        = state.walker.geo
    sample_tok = list(state.lm.vocab)[:8]
    triple_lines = ["── Sample Thébault Triples + MRV Domain Sizes ──"]
    for tok in sample_tok:
        t   = geo.triple(tok)
        mrv = state.mrv_filter.mrv_scores([tok], geo, state.walker.kernels)[0].item()
        triple_lines.append(
            f"  {tok:<14s}  ρ={t.rho:.3f}  θ={math.degrees(t.theta):6.1f}°"
            f"  σ={t.sigma:.3f}  MRV={mrv:.3f}"
        )

    # MRV domain report for most recent candidate set
    mrv_report = state.mrv_filter.domain_report(
        list(state.lm.vocab)[:20], geo, state.walker.kernels
    )

    report_lines = [
        "V16.1 — THÉBAULT'S THEOREM + MRV CONSTRAINT HEURISTIC",
        "=" * 60,
        f"Vocab size       : {len(state.lm.vocab)}",
        f"Kernel  λ_reg    : {lambda_reg:.2f}   γ_side : {gamma_side:.2f}",
        f"Walker  α_reg    : {alpha_reg:.2f}   β_ori  : {beta_ori:.2f}",
        f"        δ_side   : {delta_side:.2f}   γ_orbit: {gamma_orbit:.2f}   ψ_pot: {psi_pot:.2f}",
        f"MRV     ζ_mrv    : {zeta_mrv:.2f}   threshold: {mrv_threshold:.2f}   cap_ratio: {mrv_cap_ratio:.2f}",
        "Mandates         : synthetic_reason Active",
        "",
        *triple_lines,
        "",
        "── MRV Domain Sizes (sorted by most constrained) ──",
        mrv_report,
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
    with gr.Blocks(title="NeuroSymbolic V16.1 — Thébault + MRV") as demo:
        gr.Markdown(
            "# NeuroSymbolic V16.1\n"
            "### Thébault's Theorem · **MRV Constraint Heuristic** · "
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
                lambda_reg = gr.Slider(0.5, 20.0, value=8.0,  step=0.5,  label="λ_reg  — regularity kernel bandwidth")
                gamma_side = gr.Slider(0.5, 12.0, value=4.0,  step=0.5,  label="γ_side — side-length kernel bandwidth")

                gr.Markdown("#### MRV Constraint Parameters  ← NEW")
                zeta_mrv      = gr.Slider(0.0, 3.0, value=0.9,  step=0.1,  label="ζ_mrv       — MRV logit weight")
                mrv_threshold = gr.Slider(0.1, 0.9, value=0.50, step=0.05, label="MRV threshold — K_reg/K_side compatibility cutoff")
                mrv_cap_ratio = gr.Slider(1.0, 5.0, value=2.0,  step=0.25, label="MRV cap ratio — soft-penalise tokens with domain > ratio×mean")

                gr.Markdown("#### Walker Blend Weights")
                alpha_reg   = gr.Slider(0.0, 3.0, value=1.2, step=0.1,  label="α  — K_reg  weight")
                beta_ori    = gr.Slider(0.0, 3.0, value=0.8, step=0.1,  label="β  — K_ori  weight")
                delta_side  = gr.Slider(0.0, 3.0, value=1.0, step=0.1,  label="δ  — K_side weight")
                gamma_orbit = gr.Slider(0.0, 3.0, value=0.6, step=0.1,  label="γ  — orbit  weight")
                psi_pot     = gr.Slider(0.0, 2.0, value=0.35,step=0.05, label="ψ  — graph potential weight")

            with gr.Column(scale=2):
                btn        = gr.Button("Generate — Run Thébault + MRV Engine", variant="primary", size="lg")
                out_text   = gr.Textbox(label="Generated Sentences", lines=15)
                out_report = gr.Textbox(label="Structure Report",    lines=25)

        btn.click(
            run_session,
            inputs=[
                text_file, seed_context,
                num_sentences, tokens_per_sentence,
                temp, alpha_reg, beta_ori, delta_side, gamma_orbit, psi_pot,
                lambda_reg, gamma_side,
                zeta_mrv, mrv_threshold, mrv_cap_ratio,
            ],
            outputs=[out_text, out_report],
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(share=False)