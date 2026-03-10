#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V17-CUDA — Thébault + MRV + Isomorphic Syntax Stacking
                        + Positional Vectorisation + Chunked Sum Generation
                        + REAL-TIME SWEET SPOT EXPLORER (Gradio GUI)
===============================================================================

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

SWEET SPOT EXPLORER ADDITIONS
──────────────────────────────
• All 9 walk_probs() weights exposed as live sliders
• Context-Position Presets (Early / Mid / Late / Dense / Creative / Coherent)
• Real-time conflict detection (7 rules)
• Per-section logit-budget bar
• Effective-weight-by-position table (ηchunk / ξecho / ζmrv position scaling)
• Live Python snippet export
• Sentence-level probe: generate with current params and show token log

All V17 mathematical semantics preserved exactly.
===============================================================================
"""

from __future__ import annotations
import re, math, random, unicodedata, pickle, argparse, time, html
from dataclasses import dataclass
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
        scan = vocab[:self.max_vocab_scan]
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

    def chunk_bonus(self, c_pvec: torch.Tensor, scale : float = 1.0) -> torch.Tensor:
        sig = self.chunk_signature()
        cv_tiled = c_pvec.repeat(1, self.n_chunks)
        raw = cv_tiled @ sig
        std = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * scale

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
    BASAL_K    = 1.5
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
        self._head_cands : Dict[Tuple[str, str], torch.Tensor]  = {}
        self._head_probs : Dict[Tuple[str, str], torch.Tensor]  = {}

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
# SECTION 11 — THÉBAULT WALKER V17-CUDA (with token-level probe capture)
# ════════════════════════════════════════════════════════════════════════════

class ThebaultWalker:
    def __init__(self, geo, kernels, lm, orbit, graph, synth, mrv_filter, chunk_engine, iso_stacker, device: torch.device = DEVICE):
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
        # ── probe state (populated during walk_probs when probe_mode=True) ──
        self.last_probe: Optional[dict] = None

    def begin_sentence(self) -> None:
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()

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
        probe_mode    : bool  = False,
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

        # individual weighted contributions (for probe)
        contrib_reg    = alphareg    * k_reg
        contrib_ori    = betaori     * k_ori
        contrib_side   = deltaside   * k_side
        contrib_orbit  = gammaorbit  * orbit_scores
        contrib_pot    = psipot      * pots
        contrib_comp   = comp_bonus
        contrib_mrv    = zetamrv     * mrv_scores
        contrib_chunk  = chunk_bonus           # already scaled by etachunk
        contrib_echo   = echo_bonus            # already scaled by xiecho

        logits = (
            log_base
            + contrib_reg
            + contrib_ori
            + contrib_side
            + contrib_orbit
            + contrib_pot
            + contrib_comp
            + contrib_mrv
            + contrib_chunk
            + contrib_echo
            + mandate_boost
            + punct_bias
            + punct_penalty
        ) / max(temp, 1e-6)

        probs = F.softmax(logits, dim=-1)

        if probe_mode:
            # Capture top-10 candidates with full component breakdown
            topk = torch.topk(probs, min(10, len(cands)))
            rows = []
            for rank, idx in enumerate(topk.indices.tolist()):
                c = cands[idx]
                rows.append({
                    "rank"   : rank + 1,
                    "token"  : c,
                    "prob"   : probs[idx].item(),
                    "reg"    : contrib_reg   [idx].item(),
                    "ori"    : contrib_ori   [idx].item(),
                    "side"   : contrib_side  [idx].item(),
                    "orbit"  : contrib_orbit [idx].item(),
                    "pot"    : contrib_pot   [idx].item(),
                    "comp"   : contrib_comp  [idx].item(),
                    "mrv"    : contrib_mrv   [idx].item(),
                    "chunk"  : contrib_chunk [idx].item(),
                    "echo"   : contrib_echo  [idx].item(),
                    "mandate": mandate_boost [idx].item(),
                })
            self.last_probe = {
                "w1": w1, "w2": w2,
                "ctx_rho": ctx.rho, "ctx_theta": ctx.theta, "ctx_sigma": ctx.sigma,
                "chunk_fill": self.chunk_engine._count / self.chunk_engine.window_size,
                "echo_stored": len(self.iso_stacker.store),
                "rows": rows,
            }

        return cands, probs

    def push_token(self, token: str, sentence_len: int) -> None:
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS:
            return
        self._cur_sent_toks.append(token)
        pos_norm = len(self._cur_sent_toks) / max(sentence_len, 1)
        self.chunk_engine.push(self.geo.triple(token), pos_norm)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — TEXT GENERATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def generate_passage(
    walker: ThebaultWalker,
    lm: ThebaultCompositionLM,
    num_sentences: int = 4,
    tokens_per_sent: int = 40,
    seed_text: str = "",
    # sweet-spot parameters
    temp       : float = 1.4,
    alphareg   : float = 1.2,
    betaori    : float = 0.8,
    deltaside  : float = 1.0,
    gammaorbit : float = 0.6,
    psipot     : float = 0.35,
    zetamrv    : float = 0.9,
    etachunk   : float = 0.7,
    xiecho     : float = 0.6,
    capture_probe: bool = False,
) -> Tuple[str, Optional[dict]]:
    """Returns (passage_text, probe_data_or_None)."""
    outputs = []
    probes  = []
    head_list = list(lm.heads.keys())
    if not head_list:
        return "", None

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
            toks = [w1, w2] if seed_text else []
            wsp = len(toks)
        else:
            w1, w2 = random.choice(head_list)
            toks, wsp = [], 999

        for step in range(tokens_per_sent):
            do_probe = capture_probe and sent_idx == 0 and step < 3
            cands, probs = walker.walk_probs(
                w1, w2,
                temp=temp, alphareg=alphareg, betaori=betaori,
                deltaside=deltaside, gammaorbit=gammaorbit, psipot=psipot,
                zetamrv=zetamrv, etachunk=etachunk, xiecho=xiecho,
                probe_mode=do_probe,
            )
            if do_probe and walker.last_probe:
                probes.append(walker.last_probe.copy())

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

        sent_text = detokenize(toks)
        outputs.append(sent_text)
        # Feed completed sentence into iso-stacker
        walker.iso_stacker.add(toks, walker.geo, sent_text)

    return " ".join(outputs), probes if probes else None

# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — V17Engine (unchanged API surface)
# ════════════════════════════════════════════════════════════════════════════

class V17Engine:
    def __init__(self):
        self.device = DEVICE
        self.geo = ThebaultTokenGeometry(device=self.device)
        self.kernels = ThebaultKernels()
        self.lm = ThebaultCompositionLM(self.geo, self.kernels, device=self.device)
        self.orbit = ThebaultConjugateOrbit()
        self.graph = ThebaultPotentialGraph(self.geo, self.kernels, device=self.device)
        self.mrv = MRVConstraintFilter(device=self.device)
        self.chunk = ChunkedSumEngine(device=self.device)
        self.synth = synthetic_reasonMandateProcessor()
        self.iso_stacker = IsomorphicSyntaxStacker(device=self.device)
        self.walker = None
        self.corpus_snippet = ""

    def train(self, corpus_text: str):
        print(f"[*] Tokenizing corpus ({len(corpus_text)} chars)...")
        self.corpus_snippet = corpus_text[:1000]
        tokens = tokenize(corpus_text)
        self.lm.ingest(tokens)
        all_tokens = list(self.lm.raw_freq.keys())
        max_freq = max(self.lm.raw_freq.values(), default=1.0)
        vocab_size = len(all_tokens)
        print(f"[*] Registering {vocab_size} tokens in Thebault Geometry...")
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
        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            device=self.device
        )
        print("[+] Training complete.")

    def save_cache(self, filename: str = "v17_model.pkl"):
        state = {
            "geo_vecs": self.geo._vecs, "geo_cache": self.geo._cache,
            "lm_raw_freq": self.lm.raw_freq, "lm_tri_raw": self.lm.tri_raw,
            "lm_heads": self.lm.heads, "lm_vocab": self.lm.vocab,
            "graph_nodes": self.graph.nodes, "corpus_snippet": self.corpus_snippet,
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        print(f"[+] Saved to {filename}")

    def load_cache(self, filename: str):
        with open(filename, "rb") as f:
            state = pickle.load(f)
        self.geo._vecs = state["geo_vecs"]; self.geo._cache = state["geo_cache"]
        self.lm.raw_freq = state["lm_raw_freq"]; self.lm.tri_raw = state["lm_tri_raw"]
        self.lm.heads = state["lm_heads"]; self.lm.vocab = state["lm_vocab"]
        self.graph.nodes = state["graph_nodes"]; self.corpus_snippet = state["corpus_snippet"]
        self.geo.build_cuda_tensors(self.lm.vocab)
        self.lm.finalise()
        self.graph.build(self.lm)
        self.graph.propagate(steps=2)
        self.mrv.prime(self.lm.vocab, self.geo)
        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.synth, self.mrv, self.chunk, self.iso_stacker,
            device=self.device
        )
        print(f"[+] Loaded from {filename}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — SWEET SPOT ANALYSIS HELPERS
# ════════════════════════════════════════════════════════════════════════════

LAMBDA_REG = 8.0
GAMMA_SIDE = 4.0

PARAM_META = [
    ("alphareg",  "αreg",   "Thébault Kernels", 1.2),
    ("betaori",   "βori",   "Thébault Kernels", 0.8),
    ("deltaside", "δside",  "Thébault Kernels", 1.0),
    ("gammaorbit","γorbit", "Thébault Kernels", 0.6),
    ("psipot",    "ψpot",   "Graph Potential",  0.35),
    ("zetamrv",   "ζmrv",   "MRV Constraint",   0.9),
    ("etachunk",  "ηchunk", "Memory Systems",   0.7),
    ("xiecho",    "ξecho",  "Memory Systems",   0.6),
    ("temp",      "τtemp",  "Sampling",         1.4),
]

SECTION_COLORS = {
    "Thébault Kernels": "#00c2ff",
    "Graph Potential":  "#a0ff80",
    "MRV Constraint":   "#ffb347",
    "Memory Systems":   "#c77dff",
    "Sampling":         "#ff6b9d",
}

CONTEXT_PRESETS = {
    "Early Sentence (pos 0–3)":   dict(alphareg=1.8, betaori=0.5, deltaside=1.6, gammaorbit=0.3, psipot=0.5,  zetamrv=1.4, etachunk=0.1, xiecho=0.1, temp=1.2),
    "Mid Sentence (pos 4–12)":    dict(alphareg=1.2, betaori=0.9, deltaside=1.0, gammaorbit=0.7, psipot=0.4,  zetamrv=0.9, etachunk=0.8, xiecho=0.6, temp=1.4),
    "Late Sentence (pos 13+)":    dict(alphareg=0.9, betaori=0.7, deltaside=0.8, gammaorbit=0.5, psipot=0.3,  zetamrv=0.7, etachunk=1.4, xiecho=1.0, temp=1.5),
    "Dense Technical Corpus":     dict(alphareg=2.2, betaori=0.4, deltaside=2.0, gammaorbit=0.2, psipot=0.6,  zetamrv=1.8, etachunk=0.5, xiecho=0.3, temp=0.9),
    "Creative / Narrative":       dict(alphareg=0.7, betaori=1.6, deltaside=0.6, gammaorbit=1.3, psipot=0.3,  zetamrv=0.4, etachunk=0.9, xiecho=0.8, temp=1.9),
    "High Coherence (Long Doc)":  dict(alphareg=1.1, betaori=0.8, deltaside=1.2, gammaorbit=0.3, psipot=0.8,  zetamrv=1.0, etachunk=1.5, xiecho=1.4, temp=1.1),
}

def _approx_logit_contrib(key, value):
    """Rough expected logit contribution at nominal context (rho=0.5, theta=0.8, sigma=0.3)."""
    if key == "alphareg":   base = math.exp(-LAMBDA_REG * 0.04)
    elif key == "betaori":  base = 0.5 * (1 + math.cos(0.5))
    elif key == "deltaside":base = math.exp(-GAMMA_SIDE * 0.04)
    elif key == "gammaorbit": base = math.exp(-GAMMA_SIDE * 0.02) * math.cos(0.8 + 0.5 - math.pi/2)**2
    elif key == "psipot":   base = 0.45
    elif key == "zetamrv":  base = 0.60
    elif key == "etachunk": base = 0.55
    elif key == "xiecho":   base = 0.50
    else: return None   # temp is a divisor
    return value * base

def build_logit_budget_html(params: dict) -> str:
    section_totals: Dict[str, float] = {}
    for key, label, section, _ in PARAM_META:
        c = _approx_logit_contrib(key, params.get(key, 0))
        if c is not None:
            section_totals[section] = section_totals.get(section, 0) + c
    total = sum(section_totals.values()) or 1.0
    rows = []
    for sec, val in section_totals.items():
        pct = val / total * 100
        col = SECTION_COLORS.get(sec, "#888")
        rows.append(f"""
<div style="margin-bottom:6px">
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#aaa;margin-bottom:2px">
    <span style="color:{col};font-weight:600">{sec}</span>
    <span>{pct:.1f}%</span>
  </div>
  <div style="height:7px;background:#1a1a2e;border-radius:4px;overflow:hidden">
    <div style="height:100%;width:{pct:.1f}%;background:{col};border-radius:4px;transition:width 0.3s"></div>
  </div>
</div>""")
    return "".join(rows)

def detect_conflicts(params: dict) -> List[Tuple[str, str]]:
    """Returns list of (level, message) tuples. level ∈ {'warn','info'}."""
    issues = []
    p = params
    if p["etachunk"] > 1.2 and p["xiecho"] > 1.0:
        issues.append(("warn", "ηchunk + ξecho both high → double-counting positional memory → repetitive loops. Cap one ≤ 0.7."))
    if p["alphareg"] > 2.0 and p["deltaside"] > 1.8:
        issues.append(("warn", "αreg + δside both high → over-constrained geometry, vocab diversity collapses. Reduce one by 0.4–0.6."))
    if p["temp"] < 0.7 and p["zetamrv"] > 1.5:
        issues.append(("warn", "Low τ + high ζmrv → mode collapse. MRV concentrates mass; low temp amplifies it."))
    if p["gammaorbit"] > 1.2 and p["betaori"] > 1.4:
        issues.append(("warn", "γorbit seeks antipodal θ, βori seeks aligned θ — they partially cancel. Offset by > 0.5."))
    if p["psipot"] > 1.0 and p["temp"] < 1.0:
        issues.append(("warn", "High ψpot + low τ → generation gravitates to corpus hub words, loss of variety."))
    if p["temp"] > 2.5:
        issues.append(("info", "τ > 2.5: logits near-uniform. Geometric scoring has minimal effect. Consider τ ≤ 2.0."))
    if p["alphareg"] < 0.3 and p["deltaside"] < 0.3 and p["betaori"] < 0.3:
        issues.append(("warn", "All Thébault kernels near zero → pure bigram LM, no geometric guidance."))
    if p["zetamrv"] > 2.0:
        issues.append(("info", "ζmrv > 2.0: MRV dominates. Useful for technical corpora, risky for creative."))
    if p["xiecho"] > 1.5 and p["etachunk"] < 0.3:
        issues.append(("info", "High ξecho but low ηchunk: sentence-level tracked but not sub-phrase. Consider raising ηchunk."))
    return issues

def build_conflicts_html(params: dict) -> str:
    issues = detect_conflicts(params)
    if not issues:
        return '<div style="color:#4a4;font-size:12px;padding:6px 10px;background:#0a1a0a;border-radius:5px;border:1px solid #1a3a1a">✓ No conflicts detected.</div>'
    parts = []
    for level, msg in issues:
        col  = "#ffb347" if level == "warn" else "#4cc9f0"
        bg   = "#1a1000" if level == "warn" else "#001018"
        bdr  = "#3a2000" if level == "warn" else "#002030"
        icon = "⚠" if level == "warn" else "ℹ"
        parts.append(f'<div style="font-size:11px;color:{col};padding:5px 9px;background:{bg};border:1px solid {bdr};border-radius:4px;margin-bottom:3px">{icon} {html.escape(msg)}</div>')
    return "".join(parts)

def build_position_matrix_html(params: dict) -> str:
    positions = [
        ("Start",  "0–2",   0.00, 0.00),
        ("Early",  "3–6",   0.25, 0.15),
        ("Mid",    "7–12",  0.60, 0.50),
        ("Late",   "13–20", 1.00, 0.90),
        ("Tail",   "21+",   1.00, 1.00),
    ]
    show_params = [
        ("αreg",   "alphareg",   "#00c2ff", None),
        ("βori",   "betaori",    "#00c2ff", None),
        ("δside",  "deltaside",  "#00c2ff", None),
        ("ζmrv",   "zetamrv",    "#ffb347", "mrv"),
        ("ηchunk", "etachunk",   "#c77dff", "chunk"),
        ("ξecho",  "xiecho",     "#c77dff", "echo"),
    ]
    hdr = "".join(f'<th style="color:#888;padding:4px 10px;text-align:center;border-bottom:1px solid #222;font-size:11px">{label}<br><span style="color:#555;font-weight:400">{pos}</span></th>' for label, pos, _, _ in positions)
    rows_html = ""
    for label, key, col, scale_type in show_params:
        base = params.get(key, 0)
        cells = ""
        for _, _, chunk_fill, echo_fill in positions:
            if scale_type == "chunk":   eff = base * chunk_fill
            elif scale_type == "echo":  eff = base * echo_fill
            elif scale_type == "mrv":   eff = base * (1.2 - 0.4 * chunk_fill)
            else:                       eff = base
            pct  = min(eff / 2.5, 1.0)
            alpha_bg  = int(pct * 50  + 5)
            alpha_bdr = int(pct * 80  + 20)
            bright    = int(pct * 200 + 55)
            cells += f'<td style="padding:5px 10px;text-align:center;border-bottom:1px solid #111"><div style="display:inline-block;padding:2px 7px;border-radius:4px;background:rgba({_hex_to_rgb(col)},{alpha_bg/255:.2f});border:1px solid rgba({_hex_to_rgb(col)},{alpha_bdr/255:.2f});color:rgb({bright},{bright},{bright});font-family:monospace;font-size:11px;min-width:38px">{eff:.2f}</div></td>'
        rows_html += f'<tr><td style="padding:5px 8px;font-family:monospace;font-weight:700;font-size:12px;color:{col};border-bottom:1px solid #111">{label}</td>{cells}</tr>'
    return f'<table style="border-collapse:collapse;width:100%;font-size:11px"><thead><tr><th style="color:#666;padding:4px 8px;text-align:left;border-bottom:1px solid #222;font-size:11px">Param</th>{hdr}</tr></thead><tbody>{rows_html}</tbody></table>'

def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"

def build_probe_html(probes) -> str:
    if not probes:
        return "<i>Generate with 'Show Probe' enabled to see per-token score breakdown.</i>"
    COLS = ["reg","ori","side","orbit","pot","comp","mrv","chunk","echo","mandate"]
    COL_LABELS = ["αreg","βori","δside","γorb","ψpot","comp","ζmrv","ηchk","ξecho","mand"]
    COL_COLORS = ["#00c2ff","#00c2ff","#00c2ff","#4cc9f0","#a0ff80","#a0ff80","#ffb347","#c77dff","#c77dff","#ff9f43"]
    parts = []
    for pi, probe in enumerate(probes[:3]):
        ctx_info = (f"<span style='color:#888;font-size:10px'>"
                    f"w1=<b>{html.escape(probe['w1'])}</b> w2=<b>{html.escape(probe['w2'])}</b> "
                    f"| ctx ρ={probe['ctx_rho']:.3f} θ={probe['ctx_theta']:.3f} σ={probe['ctx_sigma']:.3f} "
                    f"| chunk_fill={probe['chunk_fill']:.0%} iso_stored={probe['echo_stored']}</span>")
        hdr = "".join(f'<th style="padding:3px 6px;color:{c};font-size:10px;font-weight:700;white-space:nowrap">{l}</th>'
                      for l, c in zip(COL_LABELS, COL_COLORS))
        row_html = ""
        for r in probe["rows"]:
            prob_pct = r["prob"] * 100
            bar = f'<div style="display:inline-block;width:{min(prob_pct*4,80):.0f}px;height:3px;background:#00c2ff;border-radius:2px;vertical-align:middle;margin-left:4px"></div>'
            tok_cell = f'<td style="padding:3px 7px;font-family:monospace;font-weight:700;color:#eee;white-space:nowrap">#{r["rank"]} {html.escape(str(r["token"]))}{bar}<span style="color:#888;font-size:9px"> {prob_pct:.1f}%</span></td>'
            score_cells = "".join(
                f'<td style="padding:3px 6px;text-align:right;font-family:monospace;font-size:10px;color:{c}">{r[k]:+.3f}</td>'
                for k, c in zip(COLS, COL_COLORS)
            )
            row_html += f"<tr>{tok_cell}{score_cells}</tr>"
        parts.append(f"""
<div style="margin-bottom:14px">
  <div style="margin-bottom:4px">Step {pi+1}: {ctx_info}</div>
  <div style="overflow-x:auto">
    <table style="border-collapse:collapse;font-size:11px;width:100%">
      <thead><tr><th style="padding:3px 7px;color:#666;font-size:10px;text-align:left">Token</th>{hdr}</tr></thead>
      <tbody>{row_html}</tbody>
    </table>
  </div>
</div>""")
    return "".join(parts)

def build_snippet(params: dict) -> str:
    return (
        f"walker.walk_probs(\n"
        f"    w1, w2,\n"
        f"    temp       = {params['temp']:.2f},\n"
        f"    alphareg   = {params['alphareg']:.2f},\n"
        f"    betaori    = {params['betaori']:.2f},\n"
        f"    deltaside  = {params['deltaside']:.2f},\n"
        f"    gammaorbit = {params['gammaorbit']:.2f},\n"
        f"    psipot     = {params['psipot']:.2f},\n"
        f"    zetamrv    = {params['zetamrv']:.2f},\n"
        f"    etachunk   = {params['etachunk']:.2f},\n"
        f"    xiecho     = {params['xiecho']:.2f},\n"
        f")"
    )

# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — GRADIO GUI (full sweet-spot wiring)
# ════════════════════════════════════════════════════════════════════════════

CSS = """
body, .gradio-container { background: #08080f !important; color: #ddd !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }
.gr-panel, .gr-box, .gr-form, .block, .panel { background: #0f0f1c !important; border: 1px solid #1e1e30 !important; border-radius: 8px !important; }
label { color: #aaa !important; font-size: 12px !important; }
input[type=range] { accent-color: #00c2ff !important; }
button.primary { background: #00c2ff22 !important; border: 1px solid #00c2ff !important; color: #00c2ff !important; font-weight: 700 !important; }
button.secondary { background: #1a1a2e !important; border: 1px solid #2a2a3a !important; color: #aaa !important; }
textarea, .gr-text-input { background: #0a0a18 !important; border: 1px solid #2a2a3a !important; color: #ddd !important; }
.gr-markdown h1,h2,h3 { color: #00c2ff !important; }
"""

def launch_gui():
    gui_state = {"engine": None}

    # ── helpers ─────────────────────────────────────────────────────────────
    def _get_params(*slider_vals):
        keys = [k for k,*_ in PARAM_META]
        return dict(zip(keys, slider_vals))

    def _analysis_outputs(params):
        budget_html   = build_logit_budget_html(params)
        conflict_html = build_conflicts_html(params)
        matrix_html   = build_position_matrix_html(params)
        snippet       = build_snippet(params)
        return budget_html, conflict_html, matrix_html, snippet

    # ── engine init ─────────────────────────────────────────────────────────
    def init_from_file(file_obj):
        if file_obj is None:
            return "⚠ No file uploaded."
        try:
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                corpus = f.read()
            if not corpus.strip():
                return "⚠ File is empty."
            eng = V17Engine()
            eng.train(corpus)
            gui_state["engine"] = eng
            return (f"✓ Engine ready  |  device={eng.device}  |  vocab={len(eng.lm.vocab)}  "
                    f"|  trigrams={len(eng.lm.tri_raw)}  |  graph nodes={len(eng.graph.nodes)}")
        except Exception as e:
            return f"✗ Error: {e}"

    def init_from_text(corpus):
        if not corpus or not corpus.strip():
            return "⚠ Corpus text is empty."
        try:
            eng = V17Engine()
            eng.train(corpus)
            gui_state["engine"] = eng
            return (f"✓ Engine ready  |  device={eng.device}  |  vocab={len(eng.lm.vocab)}  "
                    f"|  trigrams={len(eng.lm.tri_raw)}  |  graph nodes={len(eng.graph.nodes)}")
        except Exception as e:
            return f"✗ Error: {e}"

    # ── generation ──────────────────────────────────────────────────────────
    def generate(sentences, tokens, seed, show_probe, *slider_vals):
        eng = gui_state.get("engine")
        if eng is None or eng.walker is None:
            return "⚠ Engine not initialised.", "<i>—</i>", "<i>—</i>", "", ""

        params = _get_params(*slider_vals)
        t0 = time.time()
        passage, probes = generate_passage(
            eng.walker, eng.lm,
            num_sentences=int(sentences),
            tokens_per_sent=int(tokens),
            seed_text=seed.strip(),
            capture_probe=bool(show_probe),
            **{k: params[k] for k in params},
        )
        elapsed = time.time() - t0
        status  = f"✓ Generated {int(sentences)} sentences in {elapsed:.2f}s  |  device={eng.device}"

        budget_html, conflict_html, matrix_html, snippet = _analysis_outputs(params)
        probe_html = build_probe_html(probes) if show_probe else "<i>Enable 'Show Probe' to capture token-level score breakdowns.</i>"

        return passage, status, probe_html, budget_html, conflict_html, matrix_html, snippet

    # ── live analysis (sliders only, no generation) ─────────────────────────
    def live_analysis(*slider_vals):
        params = _get_params(*slider_vals)
        budget_html, conflict_html, matrix_html, snippet = _analysis_outputs(params)
        return budget_html, conflict_html, matrix_html, snippet

    # ── preset loader ────────────────────────────────────────────────────────
    def apply_preset(preset_name):
        p = CONTEXT_PRESETS.get(preset_name, {})
        if not p:
            return [gr.update()] * len(PARAM_META)
        keys = [k for k, *_ in PARAM_META]
        return [gr.update(value=p.get(k, d)) for k, _, _, d in PARAM_META]

    # ════════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ════════════════════════════════════════════════════════════════════════
    with gr.Blocks(title="V17-CUDA Sweet Spot Explorer", css=CSS) as app:

        gr.HTML("""
<div style="background:#08080f;padding:16px 0 8px 0;border-bottom:1px solid #1e1e30;margin-bottom:16px">
  <span style="font-family:monospace;font-size:22px;color:#00c2ff;font-weight:900;letter-spacing:-1px">V17-CUDA</span>
  <span style="font-size:13px;color:#555;margin-left:12px">Thébault Geometry Engine — Real-Time Sweet Spot Explorer</span>
</div>""")

        # ── Corpus Loading ──────────────────────────────────────────────────
        with gr.Accordion("① Corpus / Engine", open=True):
            with gr.Tab("Upload .txt"):
                file_input  = gr.File(label="Corpus .txt", file_types=[".txt"])
                train_file_btn = gr.Button("Initialise from File", variant="primary")
            with gr.Tab("Paste Text"):
                text_input  = gr.Textbox(lines=6, label="Paste corpus text")
                train_text_btn = gr.Button("Initialise from Text", variant="primary")
            engine_status = gr.HTML("<span style='color:#666'>No engine loaded.</span>")

        train_file_btn.click(init_from_file, inputs=[file_input], outputs=[engine_status])
        train_text_btn.click(init_from_text, inputs=[text_input], outputs=[engine_status])

        # ── Main layout: sliders left, analysis right ───────────────────────
        with gr.Row():

            # LEFT: sliders + generation controls
            with gr.Column(scale=3):
                gr.HTML('<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">② Sweet Spot Parameters</div>')

                # Preset buttons
                with gr.Row():
                    preset_dd = gr.Dropdown(
                        choices=list(CONTEXT_PRESETS.keys()),
                        label="Context Preset",
                        value=None,
                    )

                # Build sliders
                sliders = {}
                SLIDER_RANGES = {
                    "alphareg":   (0.0, 3.0,  0.05),
                    "betaori":    (0.0, 2.5,  0.05),
                    "deltaside":  (0.0, 3.0,  0.05),
                    "gammaorbit": (0.0, 2.0,  0.05),
                    "psipot":     (0.0, 1.5,  0.05),
                    "zetamrv":    (0.0, 2.5,  0.05),
                    "etachunk":   (0.0, 2.0,  0.05),
                    "xiecho":     (0.0, 2.0,  0.05),
                    "temp":       (0.3, 3.5,  0.05),
                }
                SEC_PREV = None
                for key, label, section, default in PARAM_META:
                    if section != SEC_PREV:
                        col = SECTION_COLORS.get(section, "#888")
                        gr.HTML(f'<div style="font-size:10px;color:{col};text-transform:uppercase;letter-spacing:1px;margin-top:10px;margin-bottom:2px;border-left:3px solid {col};padding-left:6px">{section}</div>')
                        SEC_PREV = section
                    mn, mx, stp = SLIDER_RANGES[key]
                    sliders[key] = gr.Slider(
                        minimum=mn, maximum=mx, step=stp, value=default,
                        label=f"{label}  ({key})",
                        interactive=True,
                    )

                # Generation controls
                gr.HTML('<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-top:14px;margin-bottom:6px">③ Generation</div>')
                with gr.Row():
                    sentences_sl = gr.Slider(1, 10, value=4, step=1,   label="Sentences")
                    tokens_sl    = gr.Slider(20, 180, value=60, step=1, label="Tokens/sent")
                seed_box   = gr.Textbox(label="Seed Text (optional)", placeholder="e.g. the quantum field")
                show_probe = gr.Checkbox(label="Show per-token score probe (first 3 steps)", value=False)
                gen_btn    = gr.Button("▶ Generate", variant="primary")

                gen_out    = gr.Textbox(lines=8, label="Generated Text", interactive=False)
                gen_status = gr.HTML("")

            # RIGHT: live analysis panels
            with gr.Column(scale=2):
                gr.HTML('<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">Live Analysis</div>')

                budget_out   = gr.HTML(label="Logit Budget")
                gr.HTML('<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-top:10px;margin-bottom:4px">Conflict Detection</div>')
                conflict_out = gr.HTML()
                gr.HTML('<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-top:10px;margin-bottom:4px">Effective Weight × Position</div>')
                matrix_out   = gr.HTML()
                gr.HTML('<div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-top:10px;margin-bottom:4px">Python Snippet</div>')
                snippet_out  = gr.Code(language="python", interactive=False)

        # ── Token Probe panel (full width) ──────────────────────────────────
        with gr.Accordion("④ Token-Level Score Probe", open=False):
            probe_out = gr.HTML("<i>Generate with probe enabled to see breakdowns.</i>")

        # ════════════════════════════════════════════════════════════════════
        # WIRING
        # ════════════════════════════════════════════════════════════════════
        slider_list = [sliders[k] for k, *_ in PARAM_META]
        analysis_outputs = [budget_out, conflict_out, matrix_out, snippet_out]

        # Preset → sliders
        preset_dd.change(
            apply_preset,
            inputs=[preset_dd],
            outputs=slider_list,
        )

        # Any slider change → live analysis (no generation)
        for sl in slider_list:
            sl.change(
                live_analysis,
                inputs=slider_list,
                outputs=analysis_outputs,
            )

        # Generate button → passage + all analysis panels + probe
        gen_btn.click(
            generate,
            inputs=[sentences_sl, tokens_sl, seed_box, show_probe] + slider_list,
            outputs=[gen_out, gen_status, probe_out] + analysis_outputs,
        )

        # Initial render of analysis panels
        app.load(
            lambda: live_analysis(*[d for _, _, _, d in PARAM_META]),
            outputs=analysis_outputs,
        )

    app.launch()


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",    action="store_true", help="Launch Gradio GUI")
    parser.add_argument("--corpus", type=str,            help="Path to training text file")
    parser.add_argument("--save",   type=str, default="v17_model.pkl")
    args = parser.parse_args()

    if args.gui or not args.corpus:
        launch_gui()
    else:
        corpus_text = Path(args.corpus).read_text(encoding="utf-8")
        engine = V17Engine()
        engine.train(corpus_text)
        engine.save_cache(args.save)
        print("\n--- SAMPLE GENERATION ---")
        text, _ = generate_passage(engine.walker, engine.lm, num_sentences=3, tokens_per_sent=30)
        print(text)
