#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V17-CUDA — Thébault + MRV + Isomorphic Syntax Stacking
                        + Positional Vectorisation + Chunked Sum Generation
                        + Petr–Douglas–Neumann (PDN) Theorem Integration
                        + Chain-of-Thought (CoT) Reasoning + Contextual Stubs
===============================================================================

CHAIN-OF-THOUGHT + CONTEXTUAL STUBS  (new in this version)
────────────────────────────────────────────────────────────

MOTIVATION
  Standard token-by-token walking is memoryless between steps — each position
  only looks back two tokens (bigram context).  CoT reasoning forces the
  system to explicitly commit to intermediate *reasoning checkpoints* before
  generating the next surface token, giving it multi-hop lookahead.

ARCHITECTURE

  ┌─────────────────────────────────────────────────────┐
  │                 CoT REASONING LOOP                   │
  │                                                     │
  │  Premise stub  →  Hypothesis stub  →  Conclusion    │
  │       ↓                 ↓                 ↓         │
  │   GeometricStep     GeometricStep    Surface token  │
  │  (Thébault score)  (PDN regularity)  (walker probs) │
  └─────────────────────────────────────────────────────┘

  1. CONTEXTUAL STUBS (CoTStubLibrary)
     ─────────────────────────────────
     A stub is a typed reasoning template extracted from the corpus.  Four
     stub types are defined:

       PREMISE     — "Given that …"    anchors the start of a reasoning chain
       ELABORATION — "This implies …"  extends an established fact
       CONTRAST    — "However …"       introduces counterpoint geometry
       CONCLUSION  — "Therefore …"     closes the chain

     Stubs are populated by scanning the corpus for high-frequency bigram
     *bridge tokens* — tokens whose Thébault rho score is above a threshold
     (i.e. geometrically "stable"), indicating they carry semantic weight.
     Each stub stores the (rho, theta, sigma) centroid of its constituent
     tokens, so the geometry of the stub is directly comparable to candidates.

  2. CHAIN-OF-THOUGHT STEPS (CoTStep / CoTChain)
     ─────────────────────────────────────────────
     Before generating each sentence the CoT engine:

       a) SELECTS a premise stub whose geometry best matches the current
          sentence's seed tokens (via Thébault kernel similarity).

       b) Runs N_HOPS reasoning hops.  Each hop:
            i.  Picks the best elaboration/contrast stub given the *current
                reasoning context* (the accumulated CoT rho/theta centroid).
            ii. Scores every vocabulary candidate against the stub centroid
                using a *stub kernel*:
                    K_stub(c) = k_reg(c.rho, stub.rho) · k_ori(c.theta, stub.theta)
            iii.The stub kernel score is added as a logit bonus during the
                actual token walk for the NEXT surface token.  This means
                the reasoning hop *pre-biases* generation toward tokens that
                are geometrically consistent with the planned stub.

       c) After N_HOPS hops, picks a conclusion stub and closes the chain.
          The conclusion stub's geometry is injected as a final bonus for
          the last few tokens of the sentence.

  3. COT TRACE (interpretability)
     ──────────────────────────────
     Every sentence's reasoning chain is recorded as a CoTTrace object
     containing the sequence of stubs chosen and the geometric distances
     traversed.  The GUI exposes a "CoT Trace" panel that renders these
     traces as human-readable reasoning steps alongside the generated text.

  4. CONTEXTUAL STUB KERNEL (vectorised, CUDA)
     ────────────────────────────────────────────
     For each candidate c and active stub s:

       K_stub(c, s) = exp(-λ·(ρ_c - ρ_s)²) · ½(1 + cos(θ_c - θ_s))
                    · exp(-γ·(σ_c - σ_s)²)

     This is the full Thébault triple kernel evaluated against the stub
     centroid rather than the context token.  The stub thus acts as a
     *geometric attractor* pulling generation toward its reasoning type.

  5. MULTI-HOP INTEGRATION WITH PDN
     ──────────────────────────────
     The CoT hop sequence is constrained to progress through PDN orbit
     families in order.  Hop k must land in orbit k % n*, linking the
     geometric reasoning structure to the PDN symmetry of the corpus.

PDN THEOREM EXTENSION
─────────────────────
(unchanged from previous version — see PDNEngine docstring)

CUDA OPTIMISATION CHANGES (vs V17-CPU)
───────────────────────────────────────
1.  DEVICE-AWARE TENSOR CACHE          6.  ISO-STACKER BATCHED SIMILARITY
2.  FULLY VECTORISED KERNEL SCORING    7.  FUSED LOGIT ACCUMULATION
3.  PRE-BATCHED LM DISTRIBUTIONS       8.  torch.compile SUPPORT
4.  BATCHED SENTENCE GENERATION        9.  MIXED PRECISION (optional)
5.  CHUNK ENGINE — ALL-CUDA           10.  COT STUB KERNELS — ALL-CUDA (new)

All V17 + PDN mathematical semantics are preserved exactly.
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
    # Keep for any legacy callers, but no longer used in rho computation
    s  = 1.0
    d  = [s, s, s, s, s * math.sqrt(2), s * math.sqrt(2)]
    mu = sum(d) / 6
    return math.sqrt(sum((x - mu) ** 2 for x in d) / 6) / mu

_PERFECT_CV = _perfect_square_cv()

def _thebault_triple(px: float, py: float, qx: float, qy: float):
    """
    Compute (rho, theta, sigma) from two 2-D vectors p and q.

    The Thébault theorem guarantees that erecting squares on the four sides
    of parallelogram (0, p, p+q, q) always yields a perfect square — so
    measuring the CV of those centres gives 0 for every input.

    Instead we measure how 'square-like' the INPUT parallelogram is:
      rho   = perpendicularity × equal-length score  ∈ [0, 1]
                1 when p⊥q and |p|=|q|  (square input → maximally ordered)
                0 when p∥q              (degenerate)
      theta = orientation of the Thébault diagonal (T[1]−T[0])
      sigma = mean Thébault diagonal half-length (scale)
    """
    if abs(px) < 1e-9 and abs(py) < 1e-9 and abs(qx) < 1e-9 and abs(qy) < 1e-9:
        return 0.0, 0.0, 0.0

    len_p = math.sqrt(px*px + py*py)
    len_q = math.sqrt(qx*qx + qy*qy)
    if len_p < 1e-9 or len_q < 1e-9:
        return 0.0, 0.0, 0.0

    # Perpendicularity: 1 when p⊥q, 0 when parallel
    cos_pq      = (px*qx + py*qy) / (len_p * len_q)
    perp_score  = 1.0 - abs(cos_pq)

    # Equal-length score: 1 when |p|==|q|, decays toward 0
    ratio        = min(len_p, len_q) / max(len_p, len_q)  # ∈ (0, 1]
    equal_score  = ratio

    rho = perp_score * equal_score  # ∈ [0, 1]

    # Thébault centres (unchanged — used for theta and sigma)
    T = _thebault_centres(0.0, 0.0, px, py, px + qx, py + qy, qx, qy)

    # Diagonal lengths of the Thébault-centre square
    d1x = T[2][0] - T[0][0];  d1y = T[2][1] - T[0][1]
    d2x = T[3][0] - T[1][0];  d2y = T[3][1] - T[1][1]
    len1 = math.sqrt(d1x*d1x + d1y*d1y)
    len2 = math.sqrt(d2x*d2x + d2y*d2y)
    sigma = (len1 + len2) / 2.0

    # Orientation from first Thébault edge
    dx_ori = T[1][0] - T[0][0]
    dy_ori = T[1][1] - T[0][1]
    theta  = math.atan2(dy_ori, dx_ori) % math.pi

    return rho, theta, sigma

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
        self._min_freq : float                = 1.0
        self._max_freq : float                = 1.0

    def register(self, token, freq, index, max_freq, vocab_size):
        if freq < self._min_freq or self._min_freq == 1.0:
            self._min_freq = max(freq, 1e-9)
        self._max_freq = max(max_freq, 1e-9)
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

    def stub_coords(self, token: str) -> tuple:
        """
        Return (lf_norm, k_hat, sigma) for stub clustering.

        lf_norm: min-max normalisation of log(freq) across the corpus.
          lf_norm = (log(freq+1) - log(min_freq+1)) / (log(max_freq+1) - log(min_freq+1))
          This maps [min_freq, max_freq] -> [0, 1] in log-space, spreading
          the Zipf distribution across all bins instead of collapsing to bin 0.

        k_hat: vocab rank / (vocab_size-1), uniform [0,1] by construction,
          recovered as sqrt(qx^2+qy^2) from the stored p/q vectors.

        sigma: Thebault sigma, used for stub TYPE assignment.
        """
        px, py, qx, qy = self._vec(token)
        f_hat = math.sqrt(px*px + py*py)   # = freq / max_freq
        k_hat = math.sqrt(qx*qx + qy*qy)  # = rank / (vocab_size-1)

        freq_raw = f_hat * self._max_freq  # recover raw freq
        min_f    = self._min_freq
        max_f    = self._max_freq
        denom    = math.log(max_f + 1) - math.log(min_f + 1) + 1e-9
        lf_norm  = (math.log(freq_raw + 1) - math.log(min_f + 1)) / denom
        lf_norm  = max(0.0, min(1.0, lf_norm))

        sigma = self.triple(token).sigma
        return lf_norm, k_hat, sigma

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2b — PETR–DOUGLAS–NEUMANN THEOREM ENGINE
# ════════════════════════════════════════════════════════════════════════════

class PDNEngine:
    """
    Petr–Douglas–Neumann theorem engine.
    PDN generalises Thébault (n=4) to regular n-gons on any polygon.
    n* is determined from corpus spectral analysis of trigram DFT modes.
    """

    def __init__(
        self,
        n_modes              : int   = 4,
        sigma_pdn            : float = 0.25,
        orbit_weight         : float = 0.4,
        regularity_weight    : float = 0.5,
        spectral_penalty_weight: float = 0.3,
        device               : torch.device = DEVICE,
        dtype                : torch.dtype  = torch.float32,
    ):
        self.n_modes                 = n_modes
        self.sigma_pdn               = sigma_pdn
        self.orbit_weight            = orbit_weight
        self.regularity_weight       = regularity_weight
        self.spectral_penalty_weight = spectral_penalty_weight
        self.device                  = device
        self.dtype                   = dtype
        self.n_star                  : int              = 4
        self.power_spectrum          : Dict[int, float] = {}
        self._orbit_map              : Dict[str, int]   = {}

    def fit_from_trigrams(self, geo: ThebaultTokenGeometry, tri_raw: Dict) -> None:
        candidate_ns = list(range(3, 3 + self.n_modes))
        power: Dict[int, float] = {n: 0.0 for n in candidate_ns}
        for (w1, w2, w3), cnt in tri_raw.items():
            toks = [w1, w2, w3]
            zs   = []
            for t in toks:
                tr = geo.triple(t)
                zs.append(cmath.exp(1j * tr.theta))   # unit phasor, preserves angular structure
            for n in candidate_ns:
                padded = zs + [0+0j] * (n - 3)
                mean_rho = sum(geo.triple(t).rho for t in toks) / 3
                for k in range(1, n):
                    F_k = sum(padded[j] * cmath.exp(-2j * math.pi * j * k / n)
                              for j in range(n)) / n
                    power[n] += cnt * mean_rho * abs(F_k) ** 2   # weight by rho
        self.power_spectrum = power
        # AFTER (picks strongest mode — correct)
        total_power = sum(power.values())
        if total_power < 1e-10:
            print("[PDN] Warning: power spectrum degenerate, defaulting to n*=4")
            self.n_star = 4
        else:
            self.n_star = max(power, key=lambda k: power[k])
        print(f"[PDN] Power spectrum: { {n: f'{p:.2f}' for n, p in power.items()} }")
        print(f"[PDN] Dominant symmetry order n* = {self.n_star}")

    def build_orbit_map(self, vocab: List[str], geo: ThebaultTokenGeometry) -> None:
        sector = 2.0 * math.pi / max(self.n_star, 2)
        for tok in vocab:
            tr = geo.triple(tok)
            full_theta = tr.theta * 2.0
            self._orbit_map[tok] = int(full_theta / sector) % self.n_star
        print(f"[PDN] Built orbit map for {len(self._orbit_map)} tokens "
              f"across {self.n_star} orbit families.")

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
# SECTION 2c — CHAIN-OF-THOUGHT ENGINE + CONTEXTUAL STUBS
# ════════════════════════════════════════════════════════════════════════════

# ── Stub types ───────────────────────────────────────────────────────────────
STUB_PREMISE     = "PREMISE"       # "Given that …"
STUB_ELABORATION = "ELABORATION"   # "This implies …"
STUB_CONTRAST    = "CONTRAST"      # "However …"
STUB_CONCLUSION  = "CONCLUSION"    # "Therefore …"

_STUB_SEQUENCE = [STUB_PREMISE, STUB_ELABORATION, STUB_CONTRAST, STUB_CONCLUSION]

@dataclass
class ContextualStub:
    """
    A reasoning template extracted from the corpus.

    Fields
    ------
    stub_type : str           — one of the four STUB_* constants
    tokens    : List[str]     — representative content tokens
    rho       : float         — mean Thébault rho of content tokens
    theta     : float         — circular mean of thetas
    sigma     : float         — mean sigma
    weight    : float         — stub importance (corpus frequency proxy)
    label     : str           — human-readable label for the GUI trace
    """
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
    """One hop in the chain of thought."""
    hop_index   : int
    stub        : ContextualStub
    stub_score  : float          # geometric match to current context
    pdn_orbit   : int            # PDN orbit at this hop


@dataclass
class CoTTrace:
    """Full CoT reasoning trace for one generated sentence."""
    seed_tokens  : List[str]
    steps        : List[CoTStep]
    conclusion   : Optional[ContextualStub]

    def render(self) -> str:
        """Return a human-readable reasoning trace string."""
        lines = ["  ── Chain-of-Thought Trace ──"]
        seed_str = ' '.join(self.seed_tokens[:6]) if self.seed_tokens else "(random)"
        lines.append(f"  Seed: {seed_str}")
        for s in self.steps:
            # stub.rho=lf_norm, stub.theta=k_hat, stub.sigma=sigma
            lines.append(
                f"  Hop {s.hop_index:02d} [{s.stub.stub_type:<11s}] "
                f"score={s.stub_score:.3f}  orbit={s.pdn_orbit}  "
                f"lf={s.stub.rho:.3f}  k={s.stub.theta:.3f}  sig={s.stub.sigma:.3f}"
            )
            lines.append(f"          → {s.stub.label}")
        if self.conclusion:
            lines.append(
                f"  Conclusion lf={self.conclusion.rho:.3f}  k={self.conclusion.theta:.3f}  sig={self.conclusion.sigma:.3f}"
            )
            lines.append(f"          → {self.conclusion.label}")
        return "\n".join(lines)


class CoTStubLibrary:
    """
    Contextual stubs clustered in (lf_norm, k_hat, sigma) space.

    Coordinate axes
    ───────────────
    lf_norm  — log-normalised frequency, uniform in [0,1] after transform
    k_hat    — vocab rank / (vocab_size-1), uniform [0,1] by construction
    sigma    — Thebault sigma (side-length scale)

    Clustering
    ──────────
    Level 1 (TYPE)   : sigma quartile -> PREMISE / ELABORATION / CONTRAST / CONCLUSION
    Level 2 (SUBTYPE): uniform grid over (lf_norm × k_hat) in [0,1]^2
                       n_lf_bins × n_k_bins cells, each non-empty cell = one stub

    This guarantees that seeds with different frequency and rank profiles
    activate different stubs, producing varied CoT chains per-seed.

    Selection kernel (in lf_norm × k_hat × sigma space)
    ────────────────────────────────────────────────────
    score(seed, stub) = exp(-al*(lf_s - lf_stub)^2)
                      * exp(-ak*(k_s  - k_stub )^2)
                      * exp(-as*(sig_s - sig_stub)^2)

    All three axes contribute genuine discrimination because all three
    are now spread across [0,1] with real population variance.
    """

    def __init__(
        self,
        n_theta_bins : int   = 8,    # repurposed as n_lf_bins
        n_k_bins     : int   = 6,
        min_bin_size : int   = 1,
        rho_threshold: float = 0.0,  # unused, kept for API compat
        device       : torch.device = DEVICE,
        dtype        : torch.dtype  = torch.float32,
    ):
        self.n_lf_bins   = n_theta_bins
        self.n_k_bins    = n_k_bins
        self.min_bin_size = min_bin_size
        self.device      = device
        self.dtype       = dtype
        self.stubs       : Dict[str, List[ContextualStub]] = {t: [] for t in _STUB_SEQUENCE}
        self._stub_list  : List[ContextualStub] = []
        # Tensor caches: lf_norm (rho field), k_hat (theta field), sigma
        self._stub_lf_t  : Optional[torch.Tensor] = None
        self._stub_k_t   : Optional[torch.Tensor] = None
        self._stub_s_t   : Optional[torch.Tensor] = None

    # ── 2c.1  BUILD FROM CORPUS ──────────────────────────────────────────

    def build(
        self,
        geo      : "ThebaultTokenGeometry",
        lm_vocab : List[str],
        raw_freq : Dict[str, float],
    ) -> None:
        """Two-level grid clustering in (lf_norm, k_hat, sigma) space."""

        entries = []
        for tok in lm_vocab:
            lf, kh, sig = geo.stub_coords(tok)
            entries.append((tok, lf, kh, sig, raw_freq.get(tok, 1.0)))

        if not entries:
            print("[CoT] Warning: empty vocab — no stubs built.")
            return

        # Level 1: sigma quartile -> stub type
        entries.sort(key=lambda x: x[3])
        q = max(1, len(entries) // 4)
        quartile_map = {
            STUB_PREMISE    : entries[:q],
            STUB_ELABORATION: entries[q : 2*q],
            STUB_CONTRAST   : entries[2*q : 3*q],
            STUB_CONCLUSION : entries[3*q:],
        }

        self.stubs = {t: [] for t in _STUB_SEQUENCE}

        for stub_type, bucket in quartile_map.items():
            if not bucket:
                continue

            # Level 2: grid over (lf_norm × k_hat) in [0,1]^2
            nlf, nk = self.n_lf_bins, self.n_k_bins
            grid: Dict[Tuple[int, int], list] = {}
            for tok, lf, kh, sig, freq in bucket:
                li = min(int(lf * nlf), nlf - 1)
                ki = min(int(kh * nk),  nk  - 1)
                grid.setdefault((li, ki), []).append((tok, lf, kh, sig, freq))

            for (li, ki), members in grid.items():
                if len(members) < self.min_bin_size:
                    continue
                self._make_stub(stub_type, li, ki, members)

        self._rebuild_tensors()
        total = sum(len(v) for v in self.stubs.values())
        per   = {t: len(v) for t, v in self.stubs.items()}
        print(f"[CoT] Built {total} contextual stubs in (lf_norm x k_hat) grid: {per}")

    def _make_stub(
        self,
        stub_type : str,
        li        : int,
        ki        : int,
        members   : list,
    ) -> None:
        toks     = [m[0] for m in members]
        lf_vals  = [m[1] for m in members]
        k_vals   = [m[2] for m in members]
        s_vals   = [m[3] for m in members]
        freqs    = [m[4] for m in members]

        lf_mean  = sum(lf_vals) / len(lf_vals)
        k_mean   = sum(k_vals)  / len(k_vals)
        s_mean   = sum(s_vals)  / len(s_vals)

        # Store lf_mean in .rho field, k_mean in .theta field, s_mean in .sigma
        tok_preview = " ".join(toks[:3])
        label = (f"[{stub_type}|lf{li}k{ki}] "
                 f"lf={lf_mean:.2f} k={k_mean:.2f} sig={s_mean:.3f} | {tok_preview}...")

        stub = ContextualStub(
            stub_type = stub_type,
            tokens    = toks,
            rho       = lf_mean,   # lf_norm centroid
            theta     = k_mean,    # k_hat centroid  (repurposed theta field)
            sigma     = s_mean,
            weight    = sum(freqs),
            label     = label,
        )
        self.stubs[stub_type].append(stub)

    def _rebuild_tensors(self) -> None:
        self._stub_list = [s for stype in _STUB_SEQUENCE for s in self.stubs[stype]]
        if not self._stub_list:
            return
        self._stub_lf_t = torch.tensor([s.rho   for s in self._stub_list], dtype=self.dtype, device=self.device)
        self._stub_k_t  = torch.tensor([s.theta for s in self._stub_list], dtype=self.dtype, device=self.device)
        self._stub_s_t  = torch.tensor([s.sigma for s in self._stub_list], dtype=self.dtype, device=self.device)

        # ── 2c.2  STUB SELECTION ─────────────────────────────────────────────

    def best_stub(
        self,
        stub_type  : str,
        ctx_rho    : float,    # lf_norm of seed (stub.rho field)
        ctx_theta  : float,    # k_hat of seed  (stub.theta field)
        ctx_sigma  : float,    # sigma of seed
        kernels    : "ThebaultKernels",
        pdn_orbit  : int = 0,
        pdn_engine : Optional["PDNEngine"] = None,
    ) -> Optional[ContextualStub]:
        """
        Score each candidate stub against the seed context in (lf, k, sigma) space.

        Field mapping (stubs store in Thebault fields for compat):
          stub.rho   = lf_norm centroid  (log-normalised frequency)
          stub.theta = k_hat centroid    (vocab rank)
          stub.sigma = sigma centroid

        Score = exp(-al*(lf_s - lf_stub)^2)
              * exp(-ak*(k_s  - k_stub )^2)
              * exp(-as*(sig_s - sig_stub)^2)

        Bandwidths tuned for [0,1] normalised axes:
          al=5.0  — moderately selective on frequency
          ak=5.0  — moderately selective on rank
          as=3.0  — soft on sigma (wider spread between quartiles)
        """
        candidates = self.stubs.get(stub_type, [])
        if not candidates:
            return None

        al = 5.0
        ak = 5.0
        as_ = 3.0

        c_lf = torch.tensor([s.rho   for s in candidates], dtype=self.dtype, device=self.device)
        c_k  = torch.tensor([s.theta for s in candidates], dtype=self.dtype, device=self.device)
        c_s  = torch.tensor([s.sigma for s in candidates], dtype=self.dtype, device=self.device)

        scores = (
            torch.exp(-al  * (c_lf - ctx_rho)   ** 2)
          * torch.exp(-ak  * (c_k  - ctx_theta)  ** 2)
          * torch.exp(-as_ * (c_s  - ctx_sigma)  ** 2)
        )

        # PDN orbit bonus using k_hat as orbit proxy
        if pdn_engine is not None:
            orb_bonus = pdn_engine.orbit_bonus(pdn_orbit, c_k * math.pi)
            scores    = scores + 0.25 * orb_bonus

        best_idx = int(scores.argmax().item())
        return candidates[best_idx]

    # ── 2c.3  STUB KERNEL (vectorised over vocab candidates) ─────────────

    @torch.no_grad()
    def stub_kernel(
        self,
        stub    : ContextualStub,
        c_rho   : torch.Tensor,
        c_theta : torch.Tensor,
        c_sigma : torch.Tensor,
        kernels : "ThebaultKernels",
        geo     : Optional["ThebaultTokenGeometry"] = None,
        cands   : Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Score every candidate token against the active stub.

        Uses the same (lf_norm, k_hat, sigma) space as best_stub when
        geo+cands are provided, so the tokens most geometrically similar
        to the selected stub receive the highest generation bonus.

          score_c = exp(-al*(lf_c - stub.rho  )^2)
                  * exp(-ak*(k_c  - stub.theta)^2)
                  * exp(-as*(sig_c- stub.sigma )^2)
        """
        C = c_rho.shape[0]

        if geo is not None and cands is not None:
            al  = 5.0
            ak  = 5.0
            as_ = 3.0
            # Batch compute stub_coords for all candidates
            coords = [geo.stub_coords(c) for c in cands]
            lf_vals = torch.tensor([co[0] for co in coords], dtype=self.dtype, device=self.device)
            k_vals  = torch.tensor([co[1] for co in coords], dtype=self.dtype, device=self.device)
            s_vals  = torch.tensor([co[2] for co in coords], dtype=self.dtype, device=self.device)
            return (
                torch.exp(-al  * (lf_vals - stub.rho)   ** 2)
              * torch.exp(-ak  * (k_vals  - stub.theta)  ** 2)
              * torch.exp(-as_ * (s_vals  - stub.sigma)  ** 2)
            )
        else:
            # Fallback: sigma only
            return torch.exp(-kernels.gamma_side * (c_sigma - stub.sigma) ** 2)


class CoTReasoningEngine:
    """
    Chain-of-Thought orchestrator.

    For each sentence to be generated:
      1. plan_chain()  — builds a CoTChain (sequence of stub choices)
      2. active_bonus() — called per token-step, returns the (C,) logit
                          addition from the current active stub
      3. advance()     — moves to the next hop after N tokens emitted

    The engine is stateful per-sentence and must be reset via begin_sentence().
    """

    def __init__(
        self,
        stub_library      : CoTStubLibrary,
        kernels           : "ThebaultKernels",
        pdn_engine        : PDNEngine,
        n_hops            : int   = 3,
        tokens_per_hop    : int   = 8,
        stub_logit_scale  : float = 0.9,
        device            : torch.device = DEVICE,
        dtype             : torch.dtype  = torch.float32,
    ):
        self.stubs           = stub_library
        self.kernels         = kernels
        self.pdn             = pdn_engine
        self.n_hops          = n_hops
        self.tokens_per_hop  = tokens_per_hop
        self.stub_logit_scale = stub_logit_scale
        self.device          = device
        self.dtype           = dtype

        # Per-sentence state
        self._chain          : List[CoTStep]              = []
        self._conclusion_stub: Optional[ContextualStub]   = None
        self._hop_ptr        : int                        = 0
        self._tok_since_hop  : int                        = 0
        self._traces         : List[CoTTrace]             = []

    # ── 2c.4  CHAIN PLANNING ─────────────────────────────────────────────

    def begin_sentence(self) -> None:
        self._chain           = []
        self._conclusion_stub = None
        self._hop_ptr         = 0
        self._tok_since_hop   = 0

    def plan_chain(
        self,
        seed_tokens  : List[str],
        geo          : ThebaultTokenGeometry,
        pdn_orbit    : int = 0,
    ) -> CoTTrace:
        """
        Build a full CoT plan for one sentence.

        Step-by-step reasoning:
          a) Compute seed centroid from seed token triples.
          b) Select a PREMISE stub closest to the seed centroid using relaxed
             kernel widths (matching best_stub's lam=1.5, gam=0.8).
          c) Each subsequent hop selects from its type's stub population
             based on the *previous stub's centroid*, not the seed — encoding
             a genuine geometric reasoning traversal through the manifold.
          d) The conclusion stub is selected from the CONCLUSION population
             closest to where the chain ended up geometrically.
          e) Score stored in each CoTStep uses the same relaxed widths so
             the printed scores reflect genuine geometric proximity.
        """
        # Step a: seed centroid in (lf_norm, k_hat, sigma) space
        clean_seeds = [t for t in seed_tokens
                       if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS]
        if clean_seeds:
            coords    = [geo.stub_coords(t) for t in clean_seeds]
            ctx_rho   = sum(c[0] for c in coords) / len(coords)   # lf_norm mean
            ctx_theta = sum(c[1] for c in coords) / len(coords)   # k_hat mean
            ctx_sigma = sum(c[2] for c in coords) / len(coords)   # sigma mean
        else:
            ctx_rho, ctx_theta, ctx_sigma = 0.5, 0.5, 0.3

        self._chain           = []
        self._conclusion_stub = None

        # Step b+c: build hop chain
        hop_types = [STUB_PREMISE] + [STUB_ELABORATION] * max(1, self.n_hops - 2) + [STUB_CONTRAST]
        hop_types = hop_types[:self.n_hops]

        for hop_idx, stype in enumerate(hop_types):
            stub = self.stubs.best_stub(
                stub_type  = stype,
                ctx_rho    = ctx_rho,
                ctx_theta  = ctx_theta,
                ctx_sigma  = ctx_sigma,
                kernels    = self.kernels,
                pdn_orbit  = (pdn_orbit + hop_idx) % self.pdn.n_star,
                pdn_engine = self.pdn,
            )
            if stub is None:
                continue

            # Score in (lf_norm, k_hat, sigma) space matching best_stub
            score = (
                math.exp(-5.0 * (stub.rho   - ctx_rho)   ** 2)
              * math.exp(-5.0 * (stub.theta - ctx_theta)  ** 2)
              * math.exp(-3.0 * (stub.sigma - ctx_sigma)  ** 2)
            )

            step = CoTStep(
                hop_index  = hop_idx,
                stub       = stub,
                stub_score = score,
                pdn_orbit  = (pdn_orbit + hop_idx) % self.pdn.n_star,
            )
            self._chain.append(step)

            # Next hop context = this stub's centroid (chain propagation)
            # stub.rho = lf_norm, stub.theta = k_hat, stub.sigma = sigma
            ctx_rho, ctx_theta, ctx_sigma = stub.rho, stub.theta, stub.sigma

        # Step d: conclusion stub from end of chain geometry
        self._conclusion_stub = self.stubs.best_stub(
            stub_type  = STUB_CONCLUSION,
            ctx_rho    = ctx_rho,
            ctx_theta  = ctx_theta,
            ctx_sigma  = ctx_sigma,
            kernels    = self.kernels,
            pdn_orbit  = (pdn_orbit + self.n_hops) % self.pdn.n_star,
            pdn_engine = self.pdn,
        )

        trace = CoTTrace(
            seed_tokens = clean_seeds,
            steps       = list(self._chain),
            conclusion  = self._conclusion_stub,
        )
        self._traces.append(trace)
        return trace

    # ── 2c.5  PER-TOKEN BONUS ────────────────────────────────────────────

    @torch.no_grad()
    def active_bonus(
        self,
        c_rho  : torch.Tensor,
        c_theta: torch.Tensor,
        c_sigma: torch.Tensor,
        token_position : int,
        total_tokens   : int,
        geo            : Optional["ThebaultTokenGeometry"] = None,
        cands          : Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Return the (C,) stub logit bonus for the current generation step.

        Passes geo+cands to stub_kernel so it can score in (f_hat, sigma)
        space rather than the degenerate Thebault-rho space.
        """
        C = c_rho.shape[0]

        # Advance hop pointer
        if self._tok_since_hop >= self.tokens_per_hop and self._hop_ptr < len(self._chain) - 1:
            self._hop_ptr      += 1
            self._tok_since_hop = 0
        self._tok_since_hop += 1

        # Decide which stub to apply
        frac = token_position / max(total_tokens - 1, 1)
        if frac >= 0.80 and self._conclusion_stub is not None:
            active_stub = self._conclusion_stub
        elif self._hop_ptr < len(self._chain):
            active_stub = self._chain[self._hop_ptr].stub
        else:
            return torch.zeros(C, dtype=self.dtype, device=self.device)

        raw = self.stubs.stub_kernel(
            active_stub, c_rho, c_theta, c_sigma, self.kernels,
            geo=geo, cands=cands,
        )

        # Normalise
        std = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * self.stub_logit_scale

    # ── 2c.6  TRACE HISTORY ──────────────────────────────────────────────

    def all_traces_text(self, max_traces: int = 8) -> str:
        if not self._traces:
            return "  (no traces yet — generate some text first)"
        lines = []
        for i, tr in enumerate(self._traces[-max_traces:]):
            lines.append(f"\nSentence {i + 1}:")
            lines.append(tr.render())
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
        thr          = self.threshold
        domain_sizes = ((k_r > thr) & (k_s > thr)).float().sum(dim=1)
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
# SECTION 11 — THÉBAULT WALKER V17-CUDA  (+ PDN + CoT)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    def __init__(
        self,
        geo, kernels, lm, orbit, graph, synth,
        mrv_filter, chunk_engine, iso_stacker,
        pdn_engine  : PDNEngine,
        cot_engine  : CoTReasoningEngine,          # ← NEW
        device      : torch.device = DEVICE,
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
        self.pdn          = pdn_engine
        self.cot          = cot_engine             # ← NEW
        self.device       = device
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []
        self._cur_sent_toks : List[str] = []
        self._cur_orbit     : int       = 0
        self._tok_pos       : int       = 0        # ← NEW: position counter for CoT

    def begin_sentence(self, seed_tokens: List[str] = None, total_tokens: int = 40) -> CoTTrace:
        """
        Reset per-sentence state and plan the CoT chain.

        Returns the CoTTrace so the caller can log it.
        """
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()
        self._cur_orbit   = 0
        self._tok_pos     = 0
        self._total_tokens = total_tokens

        # Plan the CoT chain for this sentence
        seeds = seed_tokens or []
        self.cot.begin_sentence()
        trace = self.cot.plan_chain(seeds, self.geo, pdn_orbit=self._cur_orbit)
        return trace

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
        cot_weight    : float = 1.0,               # ← NEW weight for CoT stub bonus
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

        # ── COT STUB BONUS ───────────────────────────────────────────────
        # The active CoT stub acts as a *geometric attractor*, pulling
        # generation toward tokens whose (rho, theta, sigma) match the
        # planned reasoning step for this token position.
        cot_bonus = self.cot.active_bonus(
            c_rho, c_theta, c_sigma,
            token_position = self._tok_pos,
            total_tokens   = self._total_tokens,
            geo            = self.geo,
            cands          = cands,
        )
        # ─────────────────────────────────────────────────────────────────

        # Isomorphic pair detection
        self.current_isomorphic_pairs = []
        top_idx  = torch.topk(k_reg * k_side, min(50, len(cands))).indices
        sub_r    = k_reg[top_idx];  sub_s = k_side[top_idx]
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

        N             = len(cands)
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
            + cot_weight * cot_bonus        # ← CoT stub attractor
            + mandate_boost
            + punct_bias
            + punct_penalty
        ) / max(temp, 1e-6)

        return cands, F.softmax(logits, dim=-1)

    def push_token(self, token: str, sentence_len: int) -> None:
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS:
            return
        self._cur_sent_toks.append(token)
        self._tok_pos += 1
        pos_norm = len(self._cur_sent_toks) / max(sentence_len, 1)
        self.chunk_engine.push(self.geo.triple(token), pos_norm)
        self._cur_orbit = self.pdn.orbit_of(token)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — TEXT GENERATION ENGINE  (CoT-aware)
# ════════════════════════════════════════════════════════════════════════════

def generate_passage(
    walker          : ThebaultWalker,
    lm              : ThebaultCompositionLM,
    num_sentences   : int  = 4,
    tokens_per_sent : int  = 40,
    seed_text       : str  = "",
    return_traces   : bool = False,
) -> str | Tuple[str, List[CoTTrace]]:
    """
    Generate a multi-sentence passage with CoT reasoning.

    Each sentence:
      1. Calls walker.begin_sentence(seed_tokens) → plans the CoT chain.
      2. Walks tokens, injecting stub bonuses at each step.
      3. Advances the CoT hop pointer every tokens_per_hop tokens.

    If return_traces=True, also returns the list of CoTTrace objects.
    """
    outputs    = []
    all_traces : List[CoTTrace] = []
    head_list  = list(lm.heads.keys())
    if not head_list:
        return ("", []) if return_traces else ""

    # Resolve seed
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

    for sent_idx in range(num_sentences):
        # ── CoT PLAN for this sentence ────────────────────────────────────
        if sent_idx == 0:
            w1, w2     = seed_w1, seed_w2
            init_toks  = [w1, w2] if seed_text else []
            wsp        = len(init_toks)
            plan_seeds = seed_toks if seed_toks else [w1, w2]
        else:
            w1, w2     = random.choice(head_list)
            init_toks, wsp = [], 999
            plan_seeds = [w1, w2]

        trace = walker.begin_sentence(
            seed_tokens  = plan_seeds,
            total_tokens = tokens_per_sent,
        )
        all_traces.append(trace)
        # ─────────────────────────────────────────────────────────────────

        toks = list(init_toks)

        for step in range(tokens_per_sent):
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

    result = " ".join(outputs)
    return (result, all_traces) if return_traces else result

# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — V17 ENGINE  (+ CoT stubs)
# ════════════════════════════════════════════════════════════════════════════

class V17Engine:
    def __init__(self):
        self.device      = DEVICE
        self.geo         = ThebaultTokenGeometry(device=self.device)
        self.kernels     = ThebaultKernels()
        self.lm          = ThebaultCompositionLM(self.geo, self.kernels, device=self.device)
        self.orbit       = ThebaultConjugateOrbit()
        self.graph       = ThebaultPotentialGraph(self.geo, self.kernels, device=self.device)
        self.mrv         = MRVConstraintFilter(device=self.device)
        self.chunk       = ChunkedSumEngine(device=self.device)
        self.synth       = synthetic_reasonMandateProcessor()
        self.iso_stacker = IsomorphicSyntaxStacker(device=self.device)
        self.pdn         = PDNEngine(device=self.device)
        self.stub_lib    = CoTStubLibrary(n_theta_bins=8, device=self.device)   # ← multi-stub
        self.cot         = None                                  # ← built after training

        self.walker         = None
        self.corpus_snippet = ""

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

        print("[*] Fitting PDN (Petr–Douglas–Neumann) symmetry order from corpus...")
        self.pdn.fit_from_trigrams(self.geo, self.lm.tri_raw)
        self.pdn.build_orbit_map(self.lm.vocab, self.geo)
        print(self.pdn.theorem_bridge_report())

        # ── CoT stub library ──────────────────────────────────────────────
        print("[*] Building CoT contextual stub library from corpus geometry...")
        self.stub_lib.build(self.geo, self.lm.vocab, self.lm.raw_freq)

        self.cot = CoTReasoningEngine(
            stub_library  = self.stub_lib,
            kernels       = self.kernels,
            pdn_engine    = self.pdn,
            n_hops        = 3,
            tokens_per_hop= 10,
            device        = self.device,
        )
        # ─────────────────────────────────────────────────────────────────

        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            self.pdn, self.cot,
            device=self.device,
        )
        print("[+] Training complete.")

    def save_cache(self, filename: str = "v17_model.pkl"):
        print(f"[*] Saving model state to {filename}...")
        state = {
            "geo_vecs"       : self.geo._vecs,
            "geo_cache"      : self.geo._cache,
            "lm_raw_freq"    : self.lm.raw_freq,
            "lm_tri_raw"     : self.lm.tri_raw,
            "lm_heads"       : self.lm.heads,
            "lm_vocab"       : self.lm.vocab,
            "graph_nodes"    : self.graph.nodes,
            "corpus_snippet" : self.corpus_snippet,
            "pdn_n_star"     : self.pdn.n_star,
            "pdn_power"      : self.pdn.power_spectrum,
            "cot_stubs"      : self.stub_lib.stubs,     # ← NEW
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)
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
        self.pdn.power_spectrum = state.get("pdn_power", {})

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
            stub_library   = self.stub_lib,
            kernels        = self.kernels,
            pdn_engine     = self.pdn,
            n_hops         = 3,
            tokens_per_hop = 10,
            device         = self.device,
        )

        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            self.pdn, self.cot,
            device=self.device,
        )
        print("[+] Load successful.")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — GRADIO GUI  (+ CoT Trace panel)
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
            stub_counts = {k: len(v) for k, v in self.engine.stub_lib.stubs.items()}
            return (
                f"Engine initialised from file ({file_obj.name.split('/')[-1]}).\n"
                f"Vocab size: {len(self.engine.lm.vocab)}\n"
                f"CoT stubs: {stub_counts}\n\n"
                f"{report}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_text(self, sentences, tokens, seed_text):
        if not self.engine or not self.engine.walker:
            return "Engine not initialised.", ""
        text, traces = generate_passage(
            self.engine.walker,
            self.engine.lm,
            num_sentences   = int(sentences),
            tokens_per_sent = int(tokens),
            seed_text       = seed_text.strip(),
            return_traces   = True,
        )
        trace_text = "\n".join(tr.render() for tr in traces)
        return text, trace_text

    def pdn_report(self):
        if not self.engine:
            return "Engine not initialised."
        return self.engine.pdn.theorem_bridge_report()

    def cot_history(self):
        if not self.engine or not self.engine.cot:
            return "Engine not initialised."
        return self.engine.cot.all_traces_text()


def launch_gui():
    gui = V17GUI()

    with gr.Blocks(title="NeuroSymbolic V17 CUDA + PDN + CoT") as app:
        gr.Markdown(
            "# NeuroSymbolic V17 CUDA\n"
            "### Thébault Geometry · Petr–Douglas–Neumann Theorem · Chain-of-Thought Reasoning"
        )

        with gr.Tab("Train"):
            file_input     = gr.File(label="Upload .txt Corpus File", file_types=[".txt"])
            train_file_btn = gr.Button("Initialise from File", variant="primary")
            init_out       = gr.Textbox(label="Engine Status / PDN Report", lines=22, interactive=False)
            train_file_btn.click(gui.init_engine_from_file, inputs=[file_input], outputs=init_out)

        with gr.Tab("Generate"):
            gr.Markdown("### Text Generation with Chain-of-Thought")
            with gr.Row():
                sentences = gr.Slider(1, 10, value=4, step=1, label="Sentences")
                tokens    = gr.Slider(20, 180, value=80, step=1, label="Tokens per sentence")
            seed_input = gr.Textbox(label="Seed Text (Optional)", placeholder="e.g. quantum entanglement")
            gen_btn    = gr.Button("Generate", variant="primary")
            gen_out    = gr.Textbox(lines=12, label="Generated Text")
            cot_out    = gr.Textbox(lines=16, label="Chain-of-Thought Reasoning Trace", interactive=False)
            gen_btn.click(gui.generate_text, inputs=[sentences, tokens, seed_input], outputs=[gen_out, cot_out])

        with gr.Tab("Diagnostics"):
            pdn_btn    = gr.Button("Show PDN Bridge Report")
            pdn_out    = gr.Textbox(lines=18, label="PDN Report", interactive=False)
            pdn_btn.click(gui.pdn_report, outputs=pdn_out)

            cot_hist_btn = gr.Button("Show Full CoT History")
            cot_hist_out = gr.Textbox(lines=20, label="CoT Trace History", interactive=False)
            cot_hist_btn.click(gui.cot_history, outputs=cot_hist_out)

    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",    action="store_true")
    parser.add_argument("--corpus", type=str)
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

    print("\n--- SAMPLE GENERATION (with CoT traces) ---")
    text, traces = generate_passage(
        engine.walker, engine.lm,
        num_sentences=3, tokens_per_sent=30,
        return_traces=True,
    )
    print(text)
    print("\n--- COT TRACES ---")
    for tr in traces:
        print(tr.render())
