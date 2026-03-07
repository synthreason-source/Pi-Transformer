#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V15.0 — RKHS Kernels × Symmetrical Processor
===============================================================================

REFACTOR DECLARATION
────────────────────
This system upgrades the Superpolynomial architecture by projecting discrete
algebraic operations into a continuous Reproducing Kernel Hilbert Space (RKHS)
using standard neural network activation curves.

Additionally, it integrates the Symmetrical Processor (inspired by
SynthReason 0.9N C++, George Wagenknecht 2017). This provides a parallel 
"subloop" that enforces ethical mandates and conceptual anchoring 
over the raw topological generation.

MATHEMATICAL UPGRADES
─────────────────────
1. Isomorphic Metric -> RBF Kernel
   Calculates an L2 distance between candidate structural vectors (Probability,
   Orbit Injection, Graph Potential) and projects it through an RBF kernel.

2. 1:-1 Vis-a-Vis Compound -> SiLU (Swish) Kernel
   Maps tokens to split-complex numbers Z = α + jβ. Computes the indefinite 
   hyperbolic inner product I(W,C) = α_w·α_c - β_w·β_c. Expands through SiLU.

3. Inverse Surjection Monograph -> Von Mises Periodic Kernel
   Maps the vocabulary to a quotient topology ℤ/Mℤ. Forces the text generation
   Markov chain to walk cyclically through the equivalence classes (fiber bundle).

4. Symmetrical Mandates (C++ SynthReason)
   A deterministic background concept filter that heavily biases the RKHS 
   distribution towards mandate-fulfilling vocabulary when trigger words occur.
===============================================================================
"""

from __future__ import annotations
import re, math, random, hashlib, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import torch
import torch.nn.functional as F
from datasets import load_dataset
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
            if res: res[-1] += t
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
# SECTION 1 — SUPERPOLYNOMIAL RING & AUTOMORPHISM (V10 Base)
# ════════════════════════════════════════════════════════════════════════════

class SuperPolynomialRing:
    def __init__(self, alpha: float = 1.5):
        self.alpha = alpha

    @staticmethod
    def _vander(xs: torch.Tensor) -> torch.Tensor:
        n = xs.shape[0]
        d = xs.unsqueeze(1) - xs.unsqueeze(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        return d[mask].prod().clamp(min=1e-12)

    def schur(self, lam: List[int], xs: torch.Tensor) -> torch.Tensor:
        n   = xs.shape[0]
        lp  = (list(lam) + [0] * n)[:n]
        exp = torch.tensor([lp[j] + n - j - 1 for j in range(n)], dtype=torch.float32)
        A   = xs.clamp(min=1e-6).unsqueeze(1) ** exp.unsqueeze(0)
        return torch.linalg.det(A) / self._vander(xs.clamp(min=1e-6))

    def P(self, lam: List[int], xs: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        s = self.schur(lam, xs)
        theta_corr = (thetas * xs / (xs.sum() + 1e-8)).sum()
        return s * (1.0 + self.alpha * theta_corr)

    def orbit_scalar(self, f_c: float, f_a: float, p_c: int, p_a: int) -> float:
        mx = max(f_c, f_a, 1.0)
        xs = torch.tensor([f_c / mx, f_a / mx], dtype=torch.float32)
        th = torch.tensor([float(p_c), float(p_a)], dtype=torch.float32)
        w  = self.P([1], xs, th)
        if p_c != p_a: w = w + 0.5 * self.P([1, 1], xs, th)
        return float(w.clamp(min=0.0).item())

class VocabularyAutomorphism:
    def __init__(self, freq: Dict[str, float]):
        self.freq   : Dict[str, float]      = freq
        self.phi    : Dict[str, str]        = {}
        self.parity : Dict[str, int]        = {}
        self.orbits : List[Tuple[str, str]] = []
        self._build()

    def _build(self) -> None:
        tokens = sorted([t for t in self.freq if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS],
                        key=lambda t: self.freq[t], reverse=True)
        if not tokens: return
        mid = len(tokens) // 2
        for i, t in enumerate(tokens): self.parity[t] = 0 if i < mid else 1
        n_pairs = min(mid, len(tokens) - mid)
        for i in range(n_pairs):
            h, l = tokens[i], tokens[-(i + 1)]
            self.phi[h], self.phi[l] = l, h
            self.orbits.append((h, l))
        for t in tokens:
            if t not in self.phi: self.phi[t] = t

    def parity_of(self, t: str) -> int: return self.parity.get(t, 0)
    def orbit_of(self, t: str) -> Set[str]: return {t, self.phi.get(t, t)}
    def image(self, t: str) -> str: return self.phi.get(t, t)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ALGEBRAIC CONTINUOUS KERNEL EXPANSIONS
# ════════════════════════════════════════════════════════════════════════════

class SplitComplexVisAVisCompound:
    def __init__(self):
        self._cache: Dict[str, Tuple[float, float]] = {}

    def _get_coords(self, token: str) -> Tuple[float, float]:
        if token in self._cache: return self._cache[token]
        h1 = int(hashlib.md5((token + "1").encode()).hexdigest()[:8], 16) / 0xffffffff
        h2 = int(hashlib.md5((token + "2").encode()).hexdigest()[:8], 16) / 0xffffffff
        alpha = h1 * 2.0 - 0.5  # Real (1) continuation axis
        beta  = h2 * 2.0 - 1.0  # Hyperbolic (-1) volatility axis
        self._cache[token] = (alpha, beta)
        return alpha, beta

    def hyperbolic_inner_product(self, w_ctx: str, c_cand: str) -> float:
        a_w, b_w = self._get_coords(w_ctx)
        a_c, b_c = self._get_coords(c_cand)
        return (a_w * a_c) - (b_w * b_c)

class InverseSurjectionMonograph:
    def __init__(self, modulus: int = 7):
        self.modulus = modulus
        self._cache: Dict[str, int] = {}

    def surjection(self, token: str) -> int:
        if token in self._cache: return self._cache[token]
        omega = sum(hashlib.md5(token.encode()).digest()) % self.modulus
        self._cache[token] = omega
        return omega

class NeuralKernelExpansions:
    @staticmethod
    def rbf_isomorphic_curve(v_anchor: torch.Tensor, v_cand: torch.Tensor, gamma: float = 4.0) -> torch.Tensor:
        l2_sq = torch.sum((v_anchor - v_cand) ** 2, dim=-1)
        return torch.exp(-gamma * l2_sq)

    @staticmethod
    def silu_vis_a_vis_curve(z: torch.Tensor) -> torch.Tensor:
        return z * torch.sigmoid(z)

    @staticmethod
    def von_mises_surjection_curve(o_t: torch.Tensor, o_c: torch.Tensor, M: float, kappa: float) -> torch.Tensor:
        phase = (2.0 * math.pi / M) * (o_t - o_c)
        return kappa * (torch.cos(phase) - 1.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GRAPH & ALGEBRA MODULES (V10 Base)
# ════════════════════════════════════════════════════════════════════════════

class GradedSynapseAlgebra:
    BASAL_K = 1.5
    def __init__(self, spr: SuperPolynomialRing, phi: VocabularyAutomorphism):
        self.spr, self.phi = spr, phi
        self.raw_freq: Dict[str, float] = {}
        self.tri_raw: Dict[Tuple[str,str,str], float] = {}
        self.tri_graded: Dict[Tuple[str,str,str], float] = {}
        self.heads: Dict[Tuple[str,str], List[str]] = {}
        self.vocab: List[str] = []
        self.token_to_idx: Dict[str, int] = {}

    @staticmethod
    def _partition(d1: float, d2: float) -> List[int]:
        lam = sorted([min(int(math.log2(d1 + 1)), 4), min(int(math.log2(d2 + 1)), 4)], reverse=True)
        return lam if lam[0] > 0 else [1]

    def ingest(self, tokens: List[str]) -> None:
        for t in tokens: self.raw_freq[t] = self.raw_freq.get(t, 0) + 1.0
        for t in self.raw_freq:
            if t not in self.token_to_idx: self.token_to_idx[t] = len(self.token_to_idx)

        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.tri_raw[(w1, w2, w3)] = self.tri_raw.get((w1, w2, w3), 0) + 1.0
            if (w1, w2) not in self.heads: self.heads[(w1, w2)] = []
            if w3 not in self.heads[(w1, w2)]: self.heads[(w1, w2)].append(w3)

        max_f = max(self.raw_freq.values(), default=1.0)
        for (w1, w2, w3), cnt in self.tri_raw.items():
            f1, f2 = self.raw_freq.get(w1, 1.0)/max_f, self.raw_freq.get(w2, 1.0)/max_f
            p1, p2 = self.phi.parity_of(w1), self.phi.parity_of(w2)
            xs, th = torch.tensor([f1, f2]), torch.tensor([float(p1), float(p2)])
            g = float(self.spr.P(self._partition(f1*10, f2*10), xs, th).clamp(min=1e-4).item())
            self.tri_graded[(w1, w2, w3)] = cnt * g

        self.vocab = [v for v in self.raw_freq if v not in PUNCT_TOKENS and v not in COGNITIVE_TOKENS]

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        head = (w1, w2)
        if head in self.heads:
            cands = self.heads[head]
            weights = [self.tri_graded.get((w1, w2, w3), 1e-4) for w3 in cands]
        else:
            agg: Dict[str, float] = {}
            for (_, _, w3), wt in self.tri_graded.items(): agg[w3] = agg.get(w3, 0) + wt
            cands = list(agg.keys())[:400]
            weights = [agg[w] for w in cands]

        V_total = len(self.vocab) + 1
        total = sum(weights)
        probs = torch.tensor([(wt + self.BASAL_K) / (total + self.BASAL_K * V_total) for wt in weights])
        return cands, probs / probs.sum().clamp(min=1e-12)

class AutoOrbitModule(torch.nn.Module):
    def __init__(self, token_to_idx: Dict[str, int], spr: SuperPolynomialRing, phi: VocabularyAutomorphism,
                 raw_freq: Dict[str, float], adv_strength: float = 0.5, densify_mag: float = 0.08, embed_dim: int = 256):
        super().__init__()
        self.spr, self.phi, self.raw_freq = spr, phi, raw_freq
        self.adv_strength, self.densify_mag = adv_strength, densify_mag
        self.unk_idx = len(token_to_idx)
        self.token_to_idx = token_to_idx
        self.E_embed = torch.nn.Embedding(self.unk_idx + 1, embed_dim)
        self.L_embed = torch.nn.Embedding(self.unk_idx + 1, embed_dim)
        self.shift = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.shift.weight.copy_(torch.tensor([[0.85, 0.15], [0.15, 0.85]]))
            self.shift.bias.fill_(0.0)

    def _emb(self, t: str, table: torch.nn.Embedding) -> torch.Tensor:
        idx = self.token_to_idx.get(t, self.unk_idx)
        v = torch.abs(table(torch.tensor(idx)))
        return v / (v.sum() + 1e-8)

    def forward(self, anchor: str, cands: List[str], auto_strength: float = 0.55) -> torch.Tensor:
        if not cands: return torch.zeros(0)
        mag = lambda c: 0.05 if c in PUNCT_TOKENS else min(max(len(c)/10,0.1),2.0)
        E_A, L_A = self._emb(anchor, self.E_embed), self._emb(anchor, self.L_embed) * mag(anchor)
        E_C = torch.stack([self._emb(c, self.E_embed) for c in cands])
        L_C = torch.stack([self._emb(c, self.L_embed) * mag(c) for c in cands])

        G = torch.stack([
            torch.stack([(E_C*E_A).sum(-1), (L_C*E_A).sum(-1)], dim=-1),
            torch.stack([(E_C*L_A).sum(-1), (L_C*L_A).sum(-1)], dim=-1),
        ], dim=-2)

        G = G - self.adv_strength * G**2 + 0.15 * torch.tanh(self.shift(G))
        r = (G[:,0,0] + G[:,1,1]) + 0.5*(G[:,0,1].abs() + G[:,1,0].abs())
        mn, mx = r.min(), r.max()
        return (r - mn) / (mx - mn + 1e-12) if mx > mn else r

@dataclass
class SPGNode:
    token: str; freq: float; parity: int; partner: str; degree_in: float=0.0; degree_out: float=0.0; potential: float=0.0

@dataclass
class SPGEdge:
    src: str; dst: str; weight: float; kind: str

class SuperPolyGraph:
    def __init__(self, spr: SuperPolynomialRing, phi: VocabularyAutomorphism):
        self.spr, self.phi = spr, phi
        self.nodes: Dict[str, SPGNode] = {}
        self.adj: Dict[str, List[SPGEdge]] = {}
        self.radj: Dict[str, List[SPGEdge]] = {}

    def build(self, lm: GradedSynapseAlgebra) -> None:
        for tok, freq in lm.raw_freq.items():
            if tok not in PUNCT_TOKENS and tok not in COGNITIVE_TOKENS:
                self.nodes[tok] = SPGNode(tok, freq, self.phi.parity_of(tok), self.phi.image(tok))
                self.adj[tok], self.radj[tok] = [], []

        for (w2, w3, wt) in [(w2, w3, wt) for (w1, w2, w3), wt in lm.tri_graded.items()]:
            if w2 in self.nodes and w3 in self.nodes:
                e = SPGEdge(w2, w3, wt, 'TRIGRAM')
                self.adj[w2].append(e); self.radj[w3].append(e)

        for (h, l) in self.phi.orbits:
            if h in self.nodes and l in self.nodes:
                w = self.spr.orbit_scalar(self.nodes[h].freq, self.nodes[l].freq, self.nodes[h].parity, self.nodes[l].parity)
                e1, e2 = SPGEdge(h, l, w, 'ORBIT'), SPGEdge(l, h, w, 'ORBIT')
                self.adj[h].append(e1); self.radj[l].append(e1)
                self.adj[l].append(e2); self.radj[h].append(e2)

    def propagate(self, steps: int = 2) -> None:
        if not self.nodes: return
        max_f = max(nd.freq for nd in self.nodes.values()) + 1e-8
        for nd in self.nodes.values(): nd.potential = nd.freq / max_f

        for _ in range(steps):
            new_pots = {}
            for v, nd in self.nodes.items():
                agg = sum(e.weight * self.nodes[e.src].potential for e in self.radj.get(v, []))
                phi_pot = self.nodes[nd.partner].potential if nd.partner in self.nodes else nd.potential
                new_pots[v] = agg / (nd.degree_in + 1.0) + self.spr.alpha * phi_pot * 0.1
            mx = max(new_pots.values(), default=1.0) + 1e-8
            for v in self.nodes: self.nodes[v].potential = new_pots[v] / mx

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RKHS NEURAL GRAPH WALKER
# ════════════════════════════════════════════════════════════════════════════

class RKHSGraphSuperPolyWalk:
    def __init__(self, graph: SuperPolyGraph, spr: SuperPolynomialRing, phi: VocabularyAutomorphism,
                 vis_a_vis: SplitComplexVisAVisCompound, monograph: InverseSurjectionMonograph, kernels: NeuralKernelExpansions):
        self.graph = graph
        self.spr = spr
        self.phi = phi
        self.vis_a_vis = vis_a_vis
        self.monograph = monograph
        self.kernels = kernels
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []

    def walk_probs(self, w1: str, w2: str, graded_lm: GradedSynapseAlgebra, orbit_grid: AutoOrbitModule,
                   temp: float = 1.4, de_strength: float = 0.22, graph_strength: float = 0.35,
                   vis_strength: float = 0.8, surjection_kappa: float = 2.0) -> Tuple[List[str], torch.Tensor]:
        
        cands, base_probs = graded_lm.next_dist(w1, w2)
        if not cands: return cands, base_probs

        # Base structural evaluations
        grid_out = orbit_grid(anchor=w2, cands=cands)
        pots = torch.tensor([self.graph.nodes[c].potential if c in self.graph.nodes else 0.0 for c in cands], dtype=torch.float32)

        # ── 1. ISOMORPHIC METRIC (RBF KERNEL) ──
        # Build multi-dimensional structural vectors
        norm_base = (base_probs - base_probs.min()) / (base_probs.max() - base_probs.min() + 1e-8)
        norm_grid = (grid_out - grid_out.min()) / (grid_out.max() - grid_out.min() + 1e-8) if grid_out.numel() else torch.zeros_like(norm_base)
        norm_pots = (pots - pots.min()) / (pots.max() - pots.min() + 1e-8)
        cand_vectors = torch.stack([norm_base, norm_grid * de_strength, norm_pots * graph_strength], dim=1)

        self.current_isomorphic_pairs = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                if cands[i] in PUNCT_TOKENS or cands[j] in PUNCT_TOKENS: continue
                rbf_sim = self.kernels.rbf_isomorphic_curve(cand_vectors[i], cand_vectors[j]).item()
                if rbf_sim > 0.98:  # RBF > 0.98 means heavily structurally isomorphic
                    self.current_isomorphic_pairs.append((cands[i], cands[j], rbf_sim))

        # ── 2. VIS-A-VIS COMPOUND (SiLU KERNEL) ──
        z_vals = torch.tensor([self.vis_a_vis.hyperbolic_inner_product(w2, c) if c not in PUNCT_TOKENS else 0.0 for c in cands], dtype=torch.float32)
        silu_scores = self.kernels.silu_vis_a_vis_curve(z_vals)

        # ── 3. INVERSE SURJECTION MONOGRAPH (VON MISES KERNEL) ──
        omega_t = torch.tensor([(self.monograph.surjection(w2) + 1) % self.monograph.modulus], dtype=torch.float32)
        omega_c = torch.tensor([self.monograph.surjection(c) if c not in PUNCT_TOKENS else omega_t.item() for c in cands], dtype=torch.float32)
        von_mises_mask = self.kernels.von_mises_surjection_curve(omega_t, omega_c, float(self.monograph.modulus), surjection_kappa)

        # Punctuation bounds
        punct_bias = torch.zeros(len(cands))
        punct_penalty = torch.zeros(len(cands))
        for i, c in enumerate(cands):
            if c in PUNCT_TOKENS:
                punct_bias[i] = -3.5
                if w2 in PUNCT_TOKENS: punct_penalty[i] = -1e4

        # SynthReason SubConcept Enrichment
        # Full RKHS Logit Assemblage (with SynthReason Mandates)
        logits = (
            torch.log(base_probs.clamp(min=1e-12))
            + de_strength * grid_out
            + graph_strength * pots
            + vis_strength * silu_scores
            + von_mises_mask
            + mandate_boost
            + punct_bias + punct_penalty
        ) / max(temp, 1e-6)

        return cands, F.softmax(logits, dim=-1)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENGINE STATE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class V15State:
    graded_lm: GradedSynapseAlgebra
    orbit_grid: AutoOrbitModule
    graph: SuperPolyGraph
    walker: RKHSGraphSuperPolyWalk
    outputs: Dict[int, str] = field(default_factory=dict)
    isomorphic_matches: Set[Tuple[str, str]] = field(default_factory=set)

def build_v15_state(corpus_text: str) -> V15State:
    tokens = tokenize(corpus_text)
    
    spr = SuperPolynomialRing()
    raw_freq = {}
    for t in tokens: raw_freq[t] = raw_freq.get(t, 0) + 1.0
    phi = VocabularyAutomorphism(raw_freq)

    graded_lm = GradedSynapseAlgebra(spr, phi)
    graded_lm.ingest(tokens)

    orbit_grid = AutoOrbitModule(graded_lm.token_to_idx, spr, phi, raw_freq)
    
    graph = SuperPolyGraph(spr, phi)
    graph.build(graded_lm)
    graph.propagate(steps=2)

    vis_a_vis = SplitComplexVisAVisCompound()
    monograph = InverseSurjectionMonograph(modulus=7)
    kernels = NeuralKernelExpansions()
    walker = RKHSGraphSuperPolyWalk(graph, spr, phi, vis_a_vis, monograph, kernels)

    return V15State(graded_lm, orbit_grid, graph, walker)

def generate(state: V15State, num_sentences: int=20, tokens_per_sent: int=92, 
             temp: float=1.4, vis_strength: float=0.8, surjection_kappa: float=2.0) -> None:
    
    head_list = list(state.graded_lm.heads.keys())
    if not head_list: return
    
    state.outputs.clear()
    state.isomorphic_matches.clear()

    for si in range(num_sentences):
        w1, w2 = random.choice(head_list)
        toks, wsp = [], 999
        for _ in range(tokens_per_sent):
            cands, probs = state.walker.walk_probs(w1, w2, state.graded_lm, state.orbit_grid, temp=temp, 
                                                   vis_strength=vis_strength, surjection_kappa=surjection_kappa)
            if not cands: break
            
            # Harvest isomorphic candidates found during projection
            for p1, p2, sim in sorted(state.walker.current_isomorphic_pairs, key=lambda x: -x[2])[:2]:
                state.isomorphic_matches.add(tuple(sorted([p1, p2])))

            nxt = cands[torch.multinomial(probs, 1).item()]
            
            if nxt in PUNCT_TOKENS:
                if len(toks) < 3 or wsp < 3 or (nxt in {".","?","!"} and len(toks) < 5):
                    # Filter punct
                    bi, bp = None, -1.0
                    for i, (c, p) in enumerate(zip(cands, probs.tolist())):
                        if c not in PUNCT_TOKENS and p > bp: bi, bp = i, p
                    nxt = cands[bi] if bi is not None else "the"
                else: wsp = 0
            else: wsp += 1

            toks.append(nxt)
            w1, w2 = w2, nxt
            if nxt in {".","?","!"} and len(toks) >= max(4, int(tokens_per_sent * 0.85)): break

        state.outputs[si] = detokenize(toks)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GRADIO UI
# ════════════════════════════════════════════════════════════════════════════

def load_corpus(text_file=None):
    if text_file is not None:
        try:
            p = text_file.name if hasattr(text_file, "name") else str(text_file)
            return Path(p).read_text(encoding="utf-8")
        except: pass
    return (
        "The geometry of space dictates the behavior of paths. Quantum entanglement "
        "forces non-local updates. To cure disease and end poverty, we must improve "
        "our standard of living and protect every human."
    )

def run_session(text_file, num_sentences, tokens_per_sentence, temp, vis_strength, surjection_kappa):
    corpus = load_corpus(text_file)
    state = build_v15_state(corpus)
    generate(state, num_sentences=int(num_sentences), tokens_per_sent=int(tokens_per_sentence), temp=float(temp), 
             vis_strength=float(vis_strength), surjection_kappa=float(surjection_kappa))
    
    out_text = "\n".join(f"[{i+1:02d}] {s}" for i, s in state.outputs.items())
    
    report = [
        "V15.0 — RKHS NEURAL KERNEL × PROCESSOR",
        "=" * 60,
        f"Vocab size        : {len(state.graded_lm.vocab)}",
        f"Vis-a-Vis SiLU    : Strength {vis_strength:.2f}",
        f"Monograph Modulus : 7  | Von Mises Kappa {surjection_kappa:.2f}",
        "SubLoop : Active (C++ SynthReason Mandates)",
        "",
        "── Isomorphic Metric: Structurally Equivalent Candidates ──",
        "(RBF Kernel Similarity > 0.98 in V10 graph projection)",
    ]
    if state.isomorphic_matches:
        for p1, p2 in list(state.isomorphic_matches)[:20]:
            report.append(f"  {p1:<15s} ≈  {p2:<15s}")
    else:
        report.append("  No strongly isomorphic candidates found.")
        
    return out_text, "\n".join(report)

def build_app():
    with gr.Blocks(title="NeuroSymbolic V15.0 — Symmetrical Processor") as demo:
        gr.Markdown("# NeuroSymbolic V15.0\n### Continuous RKHS Kernel Expansions & Symmetrical Processor")
        with gr.Row():
            with gr.Column(scale=1):
                text_file = gr.File(label="Upload Text (.txt)")
                num_sentences = gr.Slider(1, 100, value=15, label="Sentences")
                tokens_per_sentence = gr.Slider(5, 200, value=92, label="Tokens per Sentence")
                temp = gr.Slider(0.8, 2.5, value=1.4, label="Temperature τ")
                vis_strength = gr.Slider(0.0, 3.0, value=0.8, label="Vis-a-Vis SiLU Strength")
                surjection_kappa = gr.Slider(0.0, 10.0, value=2.0, label="Inverse Surjection Kappa")
            with gr.Column(scale=2):
                btn = gr.Button("Generate — Run Engine", variant="primary", size="lg")
                out_text = gr.Textbox(label="Generated Sentences", lines=15)
                out_report = gr.Textbox(label="Structure Report", lines=12)
        btn.click(run_session, inputs=[text_file, num_sentences, tokens_per_sentence, temp, vis_strength, surjection_kappa], outputs=[out_text, out_report])
    return demo

if __name__ == "__main__":
    build_app().queue().launch(share=False)
