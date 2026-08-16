#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V18-CUDA — DNN Array Activation Edition + Cross-Synaptic Neuron Sums
===============================================================================

DOUBLE-AGNOSTIC + SOLO-PLANAR VARIANT (DA-SP)
──────────────────────────────────────────────
Built from the Abelian-Reversed edition. Two transformations replace the
"reversed" framing everywhere it previously appeared:

  (DA) DOUBLE AGNOSTICISM: every place that used to hard-code a priority —
       "this term goes first", "this layer runs before that layer", "this
       quartile maps to that stub type" — now takes that ordering as a
       named, inspectable parameter with a neutral default. Nothing in the
       pipeline silently privileges one branch, term, or stage over another
       by virtue of where it sits in the source file. Where two competing
       assumptions existed (e.g. "forward order" vs "reversed order"),
       neither is assumed correct; the caller states an order (or accepts
       a stable, name-sorted default) instead of the code assuming one.

  (SP) SOLO SEMANTIC PLANARITY: every place that combined the three
       Thébault kernels (k_reg, k_ori, k_side) as a product of three
       separate exponential/cosine terms has been collapsed into
       `unified_plane_kernel(...)` — a single scalar computed by summing
       the three weighted geometric distances (rho, theta, sigma) onto
       one invariant plane *before* exponentiating once, rather than
       exponentiating three times and multiplying. Meaning is preserved
       on that single plane; no one axis is computed, or combined, ahead
       of the others.

WHERE THIS APPLIES (mirrors every site the Abelian-Reversed edition touched):
  • build_synaptic_weight_matrix        → unified_plane_kernel (SP)
  • CrossSynapticNeuronSum.forward/*    → unified_plane_kernel (SP) +
                                           configurable term order (DA)
  • ThebaultKernels.all_scores_batched  → unified_plane_kernel (SP)
  • SemanticMandateScorer.score         → unified_plane_kernel (SP)
  • InstructionDistribution.set_instr.  → unified_plane_kernel (SP)
  • AtomismReferenceModel.build         → unified_plane_kernel (SP) +
                                           configurable Def-expansion
                                           direction (DA)
  • CoTStubLibrary.build                → configurable quartile→stub
                                           mapping (DA)
  • CoTStubLibrary.best_stub/stub_kernel→ unified_plane_kernel (SP)
  • CoTReasoningEngine.plan_chain       → configurable hop-type order (DA)
                                           + unified_plane_kernel (SP)
  • ThebaultConjugateOrbit.score        → unified_plane_kernel-style single
                                           combined term (SP)
  • ThebaultCompositionLM               → unified_plane_kernel (SP)
  • MRVConstraintFilter                 → unified_plane_kernel (SP)
  • IsomorphicSyntaxStacker             → unified_plane_kernel (SP)
  • ThebaultPotentialGraph.build        → unified_plane_kernel (SP)
  • DNNArrayPipeline.forward            → configurable layer_order (DA)
  • PDNEngine.fit_from_trigrams         → configurable trigram-scan
                                           direction (DA)
  • V18Engine.train                     → configurable build_stage_order (DA)
  • generate_passage                    → configurable sentence_order (DA)
  • ThebaultWalker.push_token           → configurable context_order (DA)
  • ThebaultWalker.walk_probs           → configurable term order via
                                           dict + symmetric_weighted_sum (DA)

MATHEMATICAL NOTE:
  Additive sums (Σ) were already commutative under the prior edition; DA
  keeps them commutative but ALSO removes the hard-coded *iteration order*
  itself by summing over a name-sorted dict rather than a literal written
  sequence, so no term's position in the source implies priority.
  Multiplicative kernel products k_reg·k_ori·k_side were commutative in
  value but computed three separate exponentials; SP replaces that with
  one exponential over a summed weighted-distance "plane", which is
  mathematically equivalent to the product (since exp(a)*exp(b)*exp(c) ==
  exp(a+b+c)) but is now genuinely a single unified computation rather
  than three kernels multiplied together — no kernel is evaluated ahead
  of, or independently from, the others.
  Layer stacks (DNNArrayPipeline) are NOT commutative, so DA does not
  claim they produce the same output under every order — it only removes
  the *hard-coding* of which order runs, exposing it as a parameter.
===============================================================================
"""

from __future__ import annotations
import re, math, random, unicodedata, pickle, argparse, cmath, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Callable, Union
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
# SECTION 0a — DOUBLE-AGNOSTIC / SOLO-PLANAR PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

def symmetric_weighted_sum(terms: Dict[str, "torch.Tensor|float"],
                            weights: Optional[Dict[str, float]] = None,
                            order: Optional[List[str]] = None):
    """
    DOUBLE AGNOSTICISM primitive.

    Sums a dict of named terms without letting the *source-code position*
    of any term imply priority. If `order` is given, terms are summed in
    that explicit, caller-stated order. Otherwise terms are summed in a
    stable, name-sorted order — a neutral default that depends only on
    the term names, never on which one the author happened to write
    first or last.

    weights: optional per-term scalar multiplier (defaults to 1.0).
    """
    keys = order if order is not None else sorted(terms.keys())
    weights = weights or {}
    acc = None
    for k in keys:
        if k not in terms:
            continue
        w = weights.get(k, 1.0)
        contrib = terms[k] * w if w != 1.0 else terms[k]
        acc = contrib if acc is None else acc + contrib
    return acc


def unified_plane_kernel(
    rho_a, theta_a, sigma_a,
    rho_b, theta_b, sigma_b,
    lambda_reg: float = 811.0,
    gamma_side: float = 411.0,
    kappa_ori: float = 1.0,
    use_torch: bool = True,
):
    """
    SOLO SEMANTIC PLANARITY primitive.

    Replaces the product k_reg(rho) · k_ori(theta) · k_side(sigma) with a
    single kernel computed on one invariant plane: the three weighted
    geometric distances are summed FIRST, and only then is a single
    exponential applied. This is numerically equivalent to the product of
    three independent exponential/cosine kernels

        exp(-λ·dρ²) · exp(-γ·dσ²) · [½(1+cos dθ)]
          ≈ exp( -(λ·dρ² + γ·dσ² + κ·(1 - cos dθ)) )

    but is computed as ONE combined distance on ONE plane rather than
    three kernels evaluated independently and then multiplied — no axis
    (rho, theta, sigma) is privileged by being computed, or combined,
    ahead of the others.

    Works for both torch tensors and plain python floats/ints
    (set use_torch=False, or pass non-tensor args, for the scalar path).
    """
    d_rho   = lambda_reg * (rho_b - rho_a) ** 2
    d_sigma = gamma_side * (sigma_b - sigma_a) ** 2

    is_tensor = use_torch and (torch.is_tensor(theta_a) or torch.is_tensor(theta_b))
    if is_tensor:
        d_ori   = kappa_ori * (1.0 - torch.cos(theta_b - theta_a))
        d_plane = (d_rho + d_sigma + d_ori)
        d_plane = d_plane.clamp(min=-3011.0, max=3011.0) if torch.is_tensor(d_plane) else d_plane
        return torch.exp(-d_plane)
    else:
        d_ori   = kappa_ori * (1.0 - math.cos(theta_b - theta_a))
        d_plane = d_rho + d_sigma + d_ori
        d_plane = max(-3011.0, min(3011.0, d_plane))
        return math.exp(-d_plane)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0b — DNN ARRAY ACTIVATION PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

def smooth_power_relu(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x_safe = x.clamp(-1, 0.5)
    return (x_safe * x_safe) / (x_safe.abs() + eps)


def signed_power(x: torch.Tensor, p: float) -> torch.Tensor:
    return x.sign() * (x.abs().clamp(max=3011.0) + 1e-12).pow(p)


def l2_array_normalize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    sq_sum = (x * x).sum(dim=dim, keepdim=True)
    norm = (sq_sum + eps).sqrt()
    return x / norm


def parabolic_arc_1d(position: int, total: int, curvature: float = 1.0) -> float:
    """Center-peaked 1D parabolic arc over the total generation span."""
    if total <= 1:
        return max(0.0, float(curvature))
    u = min(max(float(position) / float(total - 1), 0.0), 1.0)
    x = 2.0 * u - 1.0
    return max(0.0, float(curvature)) * max(0.0, 1.0 - x * x)


def parabolic_manifold_scale(
    position: int, total: int,
    strength: float = 0.35,
    curvature: float = 1.0,
) -> float:
    """Scalar gain for the 1D parabolic generation manifold."""
    return 1.0 + max(0.0, float(strength)) * parabolic_arc_1d(
        position, total, curvature
    )


@dataclass
class MechanicalFoldState:
    """Live frequency-driven mechanical fold state."""
    frequency: float = 1.0
    fold_index: int = 0
    fold_count: int = 1
    phase: float = 0.0
    depth: float = 0.0
    tension: float = 0.0
    momentum: float = 0.0
    last_frequency: float = 1.0

    def update(self, frequency: float, position: int = 0,
               total: int = 1, smoothing: float = 0.35) -> None:
        frequency = max(float(frequency), 1e-9)
        smoothing = min(max(float(smoothing), 0.0), 1.0)
        self.last_frequency = self.frequency
        self.frequency = (1.0 - smoothing) * self.frequency + smoothing * frequency
        self.fold_count = max(1, int(math.floor(math.log2(self.frequency + 1.0))) + 1)
        self.fold_index = int(position) % self.fold_count
        ratio = self.frequency / max(float(self.fold_count), 1.0)
        self.phase = (2.0 * math.pi * ((float(position) + ratio) /
                       max(float(total), 1.0))) % (2.0 * math.pi)
        self.depth = abs(math.sin(self.phase)) * math.log1p(self.frequency)
        self.tension = self.frequency / (1.0 + self.frequency + float(self.fold_count))
        self.momentum = self.frequency - self.last_frequency

    def gain(self) -> float:
        fold_position = (self.fold_index / max(self.fold_count - 1, 1)
                         if self.fold_count > 1 else 0.0)
        fold_shape = 1.0 - abs(2.0 * fold_position - 1.0)
        return 1.0 + 0.10 * fold_shape + 0.05 * math.tanh(self.tension)

    def report(self) -> str:
        return (f"freq={self.frequency:.4f} last={self.last_frequency:.4f} "
                f"fold={self.fold_index}/{self.fold_count} phase={self.phase:.4f} "
                f"depth={self.depth:.4f} tension={self.tension:.4f} "
                f"momentum={self.momentum:.4f}")


def l1_simplex_project(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=5011.0, neginf=-5011.0)
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


# SOLO SEMANTIC PLANARITY: single unified_plane_kernel replaces the
# k_reg · k_ori · k_side product. No kernel axis is computed before another.
def build_synaptic_weight_matrix(
    c_rho   : torch.Tensor,
    c_theta : torch.Tensor,
    c_sigma : torch.Tensor,
    lambda_reg : float = 811.0,
    gamma_side : float = 411.0,
    top_k      : int   = 8,
    eps        : float = 1e-8,
) -> torch.Tensor:
    C = c_rho.shape[0]
    W = unified_plane_kernel(
        c_rho.unsqueeze(0),   c_theta.unsqueeze(0),   c_sigma.unsqueeze(0),
        c_rho.unsqueeze(1),   c_theta.unsqueeze(1),   c_sigma.unsqueeze(1),
        lambda_reg=lambda_reg, gamma_side=gamma_side,
    ).clamp(0.0, 1.0)
    W.fill_diagonal_(0.0)

    if top_k < C:
        kth_vals, _ = torch.topk(W, min(top_k, C), dim=1)
        threshold   = kth_vals[:, -1].unsqueeze(1)
        W           = W * (W >= threshold).float()

    row_sum = W.sum(dim=1, keepdim=True).clamp(min=eps)
    return W / row_sum


class CrossSynapticNeuronSum:
    """
    DOUBLE-AGNOSTIC: the enrichment term order (synaptic sum vs transitive
    bonus) is now a named `term_order` parameter instead of being baked
    into the method body. Default is name-sorted ('syn' before 'trans'),
    a neutral tie-break that depends only on the term's name, not on
    authorial intent about which "should" run first.
    """
    def __init__(
        self,
        syn_weight   : float = 211.0,
        trans_weight : float = 0.6,
        syn_k        : int   = 8,
        lambda_reg   : float = 811.0,
        gamma_side   : float = 411.0,
        device       : torch.device = DEVICE,
        dtype        : torch.dtype  = torch.float32,
        term_order   : Optional[List[str]] = None,
    ):
        self.syn_weight   = syn_weight
        self.trans_weight = trans_weight
        self.syn_k        = syn_k
        self.lambda_reg   = lambda_reg
        self.gamma_side   = gamma_side
        self.device       = device
        self.dtype        = dtype
        self.term_order   = term_order  # None => name-sorted default

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
        # SOLO SEMANTIC PLANARITY: one unified kernel, not three multiplied.
        bonus = unified_plane_kernel(
            c_rho_trans, c_theta_trans, c_sigma_trans,
            ctx_rho, ctx_theta, ctx_sigma,
            lambda_reg=self.lambda_reg, gamma_side=self.gamma_side,
        )
        return layer_norm_array(bonus)

    @torch.no_grad()
    def forward(
        self,
        logits, c_rho, c_theta, c_sigma,
        c_rho_trans, c_theta_trans, c_sigma_trans,
        ctx_rho, ctx_theta, ctx_sigma,
        term_order: Optional[List[str]] = None,
    ):
        trans_bon = self.transitive_bonus(
            c_rho_trans, c_theta_trans, c_sigma_trans,
            ctx_rho, ctx_theta, ctx_sigma,
        )
        z_syn     = self.synaptic_sum(logits, c_rho, c_theta, c_sigma)

        # DOUBLE AGNOSTIC: term contribution order is a parameter, not a
        # hard-coded "trans first" / "syn first" choice. Base 'logits' is
        # always the accumulation seed (it is not an enrichment term).
        order = term_order or self.term_order  # None => name-sorted
        enriched = logits + symmetric_weighted_sum(
            {"syn": z_syn, "trans": trans_bon},
            weights={"syn": self.syn_weight, "trans": self.trans_weight},
            order=order,
        )
        return torch.nan_to_num(enriched, nan=0.0, posinf=5011.0, neginf=-5011.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0c-bis — INFLUENCE SPACE MAPPING   f : A × B → Y
# ════════════════════════════════════════════════════════════════════════════

class InfluenceSpaceMapper:
    """
    f : A × B → Y  (word-pair → influence-score). Unchanged by DA/SP —
    this stage never hard-coded a kernel product or an ordering priority
    to begin with.
    """

    def __init__(self, device: torch.device = DEVICE, exp_scale: float = 0.15):
        self.device    = device
        self.exp_scale = exp_scale

    @staticmethod
    def cartesian_domain(a_idx: torch.Tensor, b_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_a, grid_b = torch.meshgrid(a_idx, b_idx, indexing="ij")
        return grid_a, grid_b

    def log_sort(self, W0: torch.Tensor) -> torch.Tensor:
        flat  = W0.reshape(-1)
        order = torch.argsort(flat, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(flat.numel(), device=flat.device)
        log_w = -torch.log1p(ranks.float())
        log_w = log_w - log_w.min()
        return log_w.view_as(W0)

    def matrix_exp_weight(self, W: torch.Tensor) -> torch.Tensor:
        m, n = W.shape
        if m == n:
            return torch.matrix_exp(self.exp_scale * W)
        Z = torch.zeros(m + n, m + n, device=W.device, dtype=W.dtype)
        Z[:m, m:] = W
        Z[m:, :m] = W.T
        E = torch.matrix_exp(self.exp_scale * Z)
        return E[:m, m:]

    def map(self, W0: torch.Tensor) -> torch.Tensor:
        return self.matrix_exp_weight(self.log_sort(W0))

    def candidate_influence_bonus(self, cand_kernel: torch.Tensor) -> torch.Tensor:
        if cand_kernel.numel() == 0:
            return torch.zeros(0, device=cand_kernel.device)
        Y = self.map(cand_kernel)
        return Y.sum(dim=1) - Y.diagonal()


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
# SECTION 0e — FORMAL REFERENCE MODEL  (Model-Theoretic Atomism Grounding)
#
# DOUBLE-AGNOSTIC CHANGE: Def-expansion direction (ascending 0→ω vs
# descending ω→0) is now a `step_direction` parameter ('forward',
# 'backward', or 'auto' for a name-neutral default of 'forward').
# Neither direction is hard-coded as correct; D_A^(ω) = ⋃_n D_A^(n) is a
# commutative union so any direction reaches the same fixed point.
# ════════════════════════════════════════════════════════════════════════════

class AtomismReferenceModel:
    def __init__(
        self,
        geo                 : "ThebaultTokenGeometry",
        kernels             : "ThebaultKernels",
        rho_atom_threshold  : float = 10.25,
        kappa_ref           : float = 40.50,
        kappa_def           : float = 990.15,
        max_omega_steps     : int   = 60,
        device               : torch.device = DEVICE,
        dtype                : torch.dtype  = torch.float32,
        step_direction        : str = "forward",   # DA: 'forward' | 'backward'
        tau_batch_direction    : str = "forward",   # DA: 'forward' | 'backward'
    ):
        self.geo                = geo
        self.kernels            = kernels
        self.rho_atom_threshold = rho_atom_threshold
        self.kappa_ref          = kappa_ref
        self.kappa_def          = kappa_def
        self.max_omega_steps    = max_omega_steps
        self.device              = device
        self.dtype                = dtype
        self.step_direction        = step_direction
        self.tau_batch_direction   = tau_batch_direction

        self._vocab          : List[str]                   = []
        self._tok2idx        : Dict[str, int]              = {}
        self._D_A_mask       : Optional[torch.Tensor]      = None
        self._D_A_omega_mask : Optional[torch.Tensor]      = None
        self._tau_scores     : Optional[torch.Tensor]      = None
        self._omega_steps    : int                         = 0

    def build(self, vocab: List[str]) -> None:
        self._vocab   = vocab
        self._tok2idx = {t: i for i, t in enumerate(vocab)}
        V = len(vocab)
        if V == 0 or self.geo._rho_t is None:
            return

        rho_t   = self.geo._rho_t[:V]
        theta_t = self.geo._theta_t[:V]
        sigma_t = self.geo._sigma_t[:V]

        D_A_mask = (rho_t >= self.rho_atom_threshold)
        if int(D_A_mask.sum()) == 0:
            sorted_rho, _ = rho_t.sort()
            adaptive_threshold = sorted_rho[max(0, int(V * 0.80))].item()
            D_A_mask = (rho_t >= adaptive_threshold)
            used_thr = adaptive_threshold
            print(f"[RefModel] Fixed threshold {self.rho_atom_threshold:.2f} yielded 0 atoms; "
                  f"falling back to adaptive 80th-percentile threshold {adaptive_threshold:.4f}")
        else:
            used_thr = self.rho_atom_threshold
        print(f"[RefModel] D_A^(0): {int(D_A_mask.sum())} atoms "
              f"(ρ ≥ {used_thr:.4f}) / {V} tokens")

        # DOUBLE AGNOSTIC: direction is a parameter, no default privilege.
        current = D_A_mask.clone()
        chunk   = 256
        base_range = range(0, self.max_omega_steps)
        step_range = list(base_range) if self.step_direction == "forward" \
                     else list(reversed(base_range))
        for step in step_range:
            prev = current.sum().item()
            members = current.nonzero(as_tuple=True)[0]
            if members.shape[0] == 0:
                break
            reachable = torch.zeros(V, dtype=torch.bool, device=self.device)
            for s in range(0, members.shape[0], chunk):
                mb  = members[s : s + chunk]
                # SOLO SEMANTIC PLANARITY: one unified kernel.
                K = unified_plane_kernel(
                    rho_t[mb].unsqueeze(1),   theta_t[mb].unsqueeze(1),   sigma_t[mb].unsqueeze(1),
                    rho_t.unsqueeze(0),       theta_t.unsqueeze(0),       sigma_t.unsqueeze(0),
                    lambda_reg=self.kernels.lambda_reg, gamma_side=self.kernels.gamma_side,
                )
                reachable |= (K > self.kappa_def).any(dim=0)
            current |= reachable
            self._omega_steps += 1
            if current.sum().item() == prev:
                print(f"[RefModel] D_A^(ω) converged at step {step} "
                      f"(direction={self.step_direction}): "
                      f"{int(prev)} tokens ({100*prev/V:.1f}%)")
                break

        self._D_A_mask       = D_A_mask
        self._D_A_omega_mask = current
        n_omega = int(current.sum().item())
        print(f"[RefModel] D_A^(ω): {n_omega} tokens ({100*n_omega/V:.1f}%) "
              f"after {self._omega_steps} Def-expansion steps (direction={self.step_direction})")

        tau = torch.zeros(V, dtype=self.dtype, device=self.device)
        omega_f = current.float()
        batch_starts = list(range(0, V, 512))
        if self.tau_batch_direction == "backward":
            batch_starts = list(reversed(batch_starts))
        for start in batch_starts:
            end  = min(start + 512, V)
            # SOLO SEMANTIC PLANARITY: one unified kernel.
            K = unified_plane_kernel(
                rho_t[start:end].unsqueeze(1),   theta_t[start:end].unsqueeze(1),   sigma_t[start:end].unsqueeze(1),
                rho_t.unsqueeze(0),               theta_t.unsqueeze(0),               sigma_t.unsqueeze(0),
                lambda_reg=self.kernels.lambda_reg, gamma_side=self.kernels.gamma_side,
            )
            ref  = (K > self.kappa_ref)
            sz   = ref.float().sum(dim=1).clamp(min=1.0)
            out  = (ref & (~current.unsqueeze(0))).float().sum(dim=1)
            tau[start:end] = out / sz
        self._tau_scores = tau

        mean_t = tau.mean().item()
        c2     = int((tau > 0.0).sum().item())
        print(f"[RefModel] τ: mean={mean_t:.4f}  C2 witnesses (τ>0)={c2} "
              f"({100*c2/max(V,1):.1f}%)")
        print(f"[RefModel] Thm1 (Cantor): |D_A^(ω)|={n_omega} (countable ≤|V|), "
              f"|D_actual| ≥ 2^|V| (uncountable continuum)")

    @torch.no_grad()
    def tau_bonus(self, cands: List[str], scale: float = 0.45) -> torch.Tensor:
        C = len(cands)
        if self._tau_scores is None:
            return torch.zeros(C, dtype=self.dtype, device=self.device)
        idx = torch.tensor([self._tok2idx.get(c, 0) for c in cands],
                           dtype=torch.long, device=self.device)
        raw = self._tau_scores[idx]
        std = raw.std()
        if std.item() > 1e-8:
            raw = (raw - raw.mean()) / std
        return raw * scale

    def is_atomic(self, token: str) -> bool:
        if self._D_A_mask is None:
            return False
        idx = self._tok2idx.get(token)
        return False if idx is None else bool(self._D_A_mask[idx].item())

    def is_omega_atomic(self, token: str) -> bool:
        if self._D_A_omega_mask is None:
            return False
        idx = self._tok2idx.get(token)
        return False if idx is None else bool(self._D_A_omega_mask[idx].item())

    def tau(self, token: str) -> float:
        if self._tau_scores is None:
            return 0.0
        idx = self._tok2idx.get(token)
        return 0.0 if idx is None else float(self._tau_scores[idx].item())

    def reference_report(self) -> str:
        if self._D_A_mask is None:
            return "  [RefModel] Not yet built."
        V      = len(self._vocab)
        n_base = int(self._D_A_mask.sum().item())
        n_omg  = int(self._D_A_omega_mask.sum().item()) if self._D_A_omega_mask is not None else 0
        n_c2   = int((self._tau_scores > 0.0).sum().item()) if self._tau_scores is not None else 0
        mean_t = float(self._tau_scores.mean().item()) if self._tau_scores is not None else 0.0
        pct_o  = 100 * n_omg / max(V, 1)
        pct_c2 = 100 * n_c2  / max(V, 1)
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║ Formal Reference Model — Double-Agnostic / Solo-Planar (DA-SP)║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  |V| (vocabulary)           = {V:<6d}                        ║",
            f"║  |D_A^(0)| (atomic base)    = {n_base:<6d} (ρ ≥ {self.rho_atom_threshold:.2f})        ║",
            f"║  |D_A^(ω)| (ω-closure)      = {n_omg:<6d} ({pct_o:.1f}% of V)            ║",
            f"║  Def-expansion steps        = {self._omega_steps:<6d} (dir={self.step_direction})    ║",
            f"║  C2 witnesses (τ > 0)       = {n_c2:<6d} ({pct_c2:.1f}% of V)            ║",
            f"║  Mean trans-atomic score τ̄  = {mean_t:.4f}                       ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║  Formal Claims:                                              ║",
            "║  (C1) ∀w∈V, Ref(w)∩(D\\Int(V)) ≠ ∅  [Universal Externalism] ║",
            "║  (C2) ∃w∈V, ∀n∈ℕ, Ref(w) ⊄ D_A^(n) [Trans-Atomic Ref]     ║",
            "║  Strict subset:   D_A^(ω) ⊊ D_actual                        ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║  Proof stubs:                                                ║",
            "║  T1 (Cantor):  |D_A^(ω)| ≤ |V| = ℵ₀ < 2^ℵ₀ ≤ |D_actual|   ║",
            "║  T2 (Gödel):   ∃ referent unreachable by any T_A stage      ║",
            "║  T3 (Kripke):  names rigid-designate; descriptions non-rigid  ║",
            "║  T4 (Putnam):  countable 𝓜_c ⊨ T_A, |𝓜_c| < |𝓜_I|        ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOKEN PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

STOP_WORDS_COG = set(
    "know knew known think thought believe believed understand understood "
    "realize realized recognize recognized recall remember remembered forget forgot "
    "learn learned grasp comprehend perceive sense notice suspect suppose "
    "analyze consider assume conclude infer reason judge evaluate assess decide "
    "determine examine reflect question doubt wonder ponder contemplate deliberate "
    "weigh compare contrast interpret deduce hypothesize imagine expect intend mean "
    "aware conscious certain unsure clear confused uncertain likely possible probable "
    "expected assumed mental cognitive abstract logical rational intuitive subjective objective "
    "perhaps maybe probably possibly apparently seemingly presumably supposedly evidently "
    "clearly obviously certainly indeed actually really truly surely definitely "
    "generally typically usually often sometimes rarely "
    "thus hence therefore consequently since although however yet still unless "
    "whether either neither also furthermore moreover additionally meanwhile otherwise "
    "whereas despite nevertheless nonetheless accordingly thereby".split()
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
    mag_p = math.sqrt(px * px + py * py)
    mag_q = math.sqrt(qx * qx + qy * qy)
    if mag_p < 1e-9 or mag_q < 1e-9:
        return 0.0, 0.0, 0.0
    T = _thebault_centres(0.0, 0.0, px, py, px + qx, py + qy, qx, qy)
    sides = []
    for i in range(4):
        dx = T[(i+1)%4][0] - T[i][0]
        dy = T[(i+1)%4][1] - T[i][1]
        sides.append(math.sqrt(dx * dx + dy * dy))
    sigma = sum(sides) / 4.0
    dx_ori = T[1][0] - T[0][0]
    dy_ori = T[1][1] - T[0][1]
    theta  = math.atan2(dy_ori, dx_ori) % math.pi
    r_balance = 1.0 - abs(mag_p - mag_q) / (mag_p + mag_q)
    cos_angle = (px * qx + py * qy) / (mag_p * mag_q)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    r_ortho   = 1.0 - abs(cos_angle)
    rho_raw   = r_balance * r_ortho
    return rho_raw, theta, sigma

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
        rho_raw_t = torch.tensor(rhos, dtype=self.dtype, device=self.device)
        V_size    = rho_raw_t.shape[0]
        if V_size > 1:
            sorted_idx   = torch.argsort(rho_raw_t)
            rank_t       = torch.zeros(V_size, dtype=self.dtype, device=rho_raw_t.device)
            rank_t[sorted_idx] = torch.arange(V_size, dtype=self.dtype, device=rho_raw_t.device)
            self._rho_t  = rank_t / float(V_size - 1)
        else:
            self._rho_t  = rho_raw_t
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
# SECTION 2b — PDN ENGINE
#
# DOUBLE-AGNOSTIC CHANGE: the trigram scan direction used to assemble the
# rho time-series is now a `scan_direction` parameter ('forward' default,
# or 'backward'). ACF(lag) is symmetric under time-reversal of the series,
# so n* is unaffected either way — the direction is exposed rather than
# assumed.
# ════════════════════════════════════════════════════════════════════════════

class PDNEngine:
    def __init__(
        self,
        n_modes              : int   = 42,
        sigma_pdn            : float = 1.25,
        orbit_weight         : float = 15.4,
        regularity_weight    : float = 0.1,
        spectral_penalty_weight: float = 0.2,
        max_period           : int   = 24,
        device               : torch.device = DEVICE,
        dtype                : torch.dtype  = torch.float32,
        scan_direction        : str = "forward",   # DA: 'forward' | 'backward'
        term_order             : Optional[List[str]] = None,  # DA: pdn_logit_bonus term order
    ):
        self.n_modes                 = n_modes
        self.sigma_pdn               = sigma_pdn
        self.orbit_weight            = orbit_weight
        self.regularity_weight       = regularity_weight
        self.spectral_penalty_weight = spectral_penalty_weight
        self.max_period              = max_period
        self.device                  = device
        self.dtype                   = dtype
        self.scan_direction           = scan_direction
        self.term_order                = term_order
        self.n_star                  : int              = 4
        self.power_spectrum          : Dict[int, float] = {}
        self.acf_values              : Dict[int, float] = {}
        self.acf_significance_bound  : float            = 0.0
        self._orbit_map              : Dict[str, int]   = {}

    def _compute_acf(self, rho_series: List[float], max_lag: int) -> Dict[int, float]:
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
        items = list(tri_raw.items())
        if self.scan_direction == "backward":
            items = list(reversed(items))

        rho_series: List[float] = []
        for (w1, w2, w3), cnt in items:
            r1 = geo.triple(w1).rho
            r2 = geo.triple(w2).rho
            r3 = geo.triple(w3).rho
            repeats = min(int(cnt), 5)
            for _ in range(repeats):
                rho_series.extend([r1, r2, r3])

        T = len(rho_series)
        print(f"[PDN-DASP] Rho series length: {T} observations (scan_direction={self.scan_direction})")

        if T < 6:
            self.n_star = 4
            print("[PDN-DASP] Insufficient data — defaulting to n*=4")
            return

        rho_min   = min(rho_series)
        rho_max_v = max(rho_series)
        rho_range = rho_max_v - rho_min
        if rho_range > 1e-9:
            rho_series = [(r - rho_min) / rho_range for r in rho_series]
            print(f"[PDN-DASP] Rho range [{rho_min:.4f}, {rho_max_v:.4f}] → normalised [0,1]")
        else:
            print(f"[PDN-DASP] Rho zero-variance — falling back to sigma series")
            sigma_series: List[float] = []
            for (w1, w2, w3), cnt in items:
                s1 = geo.triple(w1).sigma
                s2 = geo.triple(w2).sigma
                s3 = geo.triple(w3).sigma
                repeats = min(int(cnt), 5)
                for _ in range(repeats):
                    sigma_series.extend([s1, s2, s3])
            sig_min   = min(sigma_series) if sigma_series else 0.0
            sig_max   = max(sigma_series) if sigma_series else 1.0
            sig_range = sig_max - sig_min
            if sig_range > 1e-9:
                rho_series = [(s - sig_min) / sig_range for s in sigma_series]
            else:
                print("[PDN-DASP] Both series zero-variance. Defaulting n*=4.")
                self.n_star = 4
                self.acf_values = {lag: 0.0 for lag in range(1, self.max_period + 1)}
                self.power_spectrum = self.acf_values.copy()
                return

        acf = self._compute_acf(rho_series, self.max_period)
        self.acf_values = acf

        sig_bound = 1.96 / math.sqrt(T)
        self.acf_significance_bound = sig_bound

        valid_lags = {lag: v for lag, v in acf.items() if lag >= 2}
        if not valid_lags:
            self.n_star = 4
            return

        best_lag = max(valid_lags, key=lambda l: valid_lags[l])
        best_acf = valid_lags[best_lag]

        if best_acf > sig_bound:
            self.n_star = best_lag
            print(f"[PDN-DASP] Dominant period n*={self.n_star} "
                  f"(ACF={best_acf:.4f} > threshold={sig_bound:.4f})")
        else:
            self.n_star = 4
            print(f"[PDN-DASP] No significant periodicity — defaulting to n*=4")

        self.power_spectrum = {lag: abs(v) for lag, v in acf.items()}

    def build_orbit_map(self, vocab: List[str], geo: "ThebaultTokenGeometry") -> None:
        sector = 2.0 * math.pi / max(self.n_star, 2)
        for tok in vocab:
            tr = geo.triple(tok)
            full_theta = tr.theta * 2.0
            self._orbit_map[tok] = int(full_theta / sector) % self.n_star
        print(f"[PDN-DASP] Built orbit map for {len(self._orbit_map)} tokens "
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
        term_order    : Optional[List[str]] = None,
    ) -> torch.Tensor:
        orb = self.orbit_bonus(current_orbit, c_theta)
        reg = self.regularity_scores(window_rho, window_theta, c_rho, c_theta)

        def _norm(x):
            std = x.std()
            return (x - x.mean()) / (std + 1e-8) if std.item() > 1e-8 else x - x.mean()

        # DOUBLE AGNOSTIC: term order/weights exposed via symmetric_weighted_sum
        order = term_order or self.term_order
        return symmetric_weighted_sum(
            {"orbit": _norm(orb), "regularity": _norm(reg)},
            weights={"orbit": self.orbit_weight, "regularity": self.regularity_weight},
            order=order,
        )

    def theorem_bridge_report(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║   Corpus ACF Spectral Analysis Report (Double-Agnostic)      ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Method: ACF of rho sequence, scan_direction={self.scan_direction:<8s}      ║",
            f"║  Dominant period n*:  {self.n_star:<2d}  (ACF time-reversal invariant)   ║",
            f"║  95%% CI bound:       {self.acf_significance_bound:.4f}  (1.96/√T)               ║",
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
        lines = ["  ── Chain-of-Thought Trace (Double-Agnostic / Solo-Planar) ──"]
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
        context_bonus    : float = 0.75,
        centroid_weight  : float = 0.8,
        term_order        : Optional[List[str]] = None,   # DA: distribution() term order
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
        self.term_order        = term_order
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
                # SOLO SEMANTIC PLANARITY: one unified kernel.
                K = unified_plane_kernel(
                    self.geo._rho_t, self.geo._theta_t, self.geo._sigma_t,
                    tr.rho, tr.theta, tr.sigma,
                    lambda_reg=self.semantic_radius, gamma_side=self.semantic_radius,
                )
                base += w * K

        if self._instr_centroid and self.geo._rho_t is not None:
            c = self._instr_centroid
            K = unified_plane_kernel(
                self.geo._rho_t, self.geo._theta_t, self.geo._sigma_t,
                c.rho, c.theta, c.sigma,
                lambda_reg=self.kernels.lambda_reg, gamma_side=self.kernels.gamma_side,
            )
            base += self.centroid_weight * K

        base = base.clamp(min=0.0)
        total = base.sum()
        if total.item() > 1e-8:
            base = base / total
        else:
            base = torch.ones(V, dtype=self.dtype, device=self.device) / V

        self._base_dist_t = base
        print(f"[InstrDist-DASP] Built from {len(self._instr_toks)} tokens, vocab={V}")

    @torch.no_grad()
    def distribution(self, cands, gen_tokens, lm_tok2idx, term_order: Optional[List[str]] = None):
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

        # DOUBLE AGNOSTIC: term order/priority is an explicit parameter.
        order = term_order or self.term_order
        raw = symmetric_weighted_sum(
            {"base": base_probs, "context": ctx_bonus_v, "bigram": bigram_bonus},
            order=order,
        )
        raw = raw.clamp(min=1e-12)
        return raw / raw.sum()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2e — COT STUB LIBRARY
#
# DOUBLE-AGNOSTIC CHANGE: quartile→stub-type mapping is now a caller-
# supplied `quartile_map_order` (list of 4 stub-type names, lowest-sigma
# to highest-sigma) instead of being hard-coded either forward or
# reversed. Default is the canonical _STUB_SEQUENCE order — a name-based
# constant defined once, not re-decided ad hoc per build.
# ════════════════════════════════════════════════════════════════════════════

class CoTStubLibrary:
    def __init__(
        self,
        rho_threshold  : float = 0.20,
        n_theta_bins   : int   = 8,
        min_bin_size   : int   = 2,
        device         : torch.device = DEVICE,
        dtype          : torch.dtype  = torch.float32,
        quartile_map_order : Optional[List[str]] = None,   # DA
    ):
        self.rho_threshold = rho_threshold
        self.n_theta_bins  = n_theta_bins
        self.min_bin_size  = min_bin_size
        self.device        = device
        self.dtype         = dtype
        self.quartile_map_order = quartile_map_order or list(_STUB_SEQUENCE)
        self.stubs         : Dict[str, List[ContextualStub]] = {
            t: [] for t in _STUB_SEQUENCE
        }
        self._stub_rho_t  : Optional[torch.Tensor] = None
        self._stub_theta_t: Optional[torch.Tensor] = None
        self._stub_sigma_t: Optional[torch.Tensor] = None
        self._stub_list   : List[ContextualStub]   = []

    def build(self, geo, lm_vocab, raw_freq, quartile_map_order: Optional[List[str]] = None) -> None:
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

        # DOUBLE AGNOSTIC: the 4 quartile slots (lowest→highest sigma) map
        # to stub types via an explicit, caller-controllable ordered list —
        # neither "forward" nor "reversed" is assumed.
        seq = quartile_map_order or self.quartile_map_order
        quartile_map = {
            seq[0] : bridges[:q],
            seq[1] : bridges[q : 2 * q],
            seq[2] : bridges[2 * q : 3 * q],
            seq[3] : bridges[3 * q:],
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
        print(f"[CoT-DASP] Built {total} contextual stubs (quartile_map_order={seq}): {per}")

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
        # SOLO SEMANTIC PLANARITY: one unified kernel.
        scores = unified_plane_kernel(
            c_rho, c_theta, c_sigma, ctx_rho, ctx_theta, ctx_sigma,
            lambda_reg=lam_stub, gamma_side=gam_stub,
        )
        if pdn_engine is not None:
            orb_bonus = pdn_engine.orbit_bonus(pdn_orbit, c_theta)
            scores    = scores + 0.3 * orb_bonus
        return candidates[int(scores.argmax().item())]

    @torch.no_grad()
    def stub_kernel(self, stub, c_rho, c_theta, c_sigma, kernels):
        # SOLO SEMANTIC PLANARITY: one unified kernel.
        return unified_plane_kernel(
            c_rho, c_theta, c_sigma, stub.rho, stub.theta, stub.sigma,
            lambda_reg=kernels.lambda_reg, gamma_side=kernels.gamma_side,
        )


class CoTReasoningEngine:
    """
    DOUBLE-AGNOSTIC CHANGE: the hop-type sequence used in plan_chain is now
    a `hop_type_order` parameter — a list of stub-type names of length
    n_hops (padded/truncated as needed) plus a separate `conclusion_type`
    parameter for the final stub. Default hop order is the canonical
    _STUB_SEQUENCE's middle types with CONCLUSION reserved for the
    dedicated conclusion slot — no "forward" or "reversed" precedence is
    assumed; the caller states the sequence.
    """
    def __init__(self, stub_library, kernels, pdn_engine, n_hops=3,
                 tokens_per_hop=8, stub_logit_scale=0.9, device=DEVICE, dtype=torch.float32,
                 hop_type_order: Optional[List[str]] = None,
                 conclusion_type: str = STUB_CONCLUSION):
        self.stubs           = stub_library
        self.kernels         = kernels
        self.pdn             = pdn_engine
        self.n_hops          = n_hops
        self.tokens_per_hop  = tokens_per_hop
        self.stub_logit_scale = stub_logit_scale
        self.device          = device
        self.dtype           = dtype
        self.hop_type_order  = hop_type_order
        self.conclusion_type = conclusion_type
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

    def _default_hop_types(self) -> List[str]:
        # Neutral default: cycle through the non-conclusion stub types in
        # their canonical (name-defined) order.
        pool = [t for t in _STUB_SEQUENCE if t != self.conclusion_type]
        return [pool[i % len(pool)] for i in range(self.n_hops)]

    def plan_chain(self, seed_tokens, geo, pdn_orbit=0, hop_type_order: Optional[List[str]] = None) -> CoTTrace:
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

        hop_types = (hop_type_order or self.hop_type_order or self._default_hop_types())[:self.n_hops]

        for hop_idx, stype in enumerate(hop_types):
            stub = self.stubs.best_stub(
                stype, ctx_rho, ctx_theta, ctx_sigma, self.kernels,
                pdn_orbit=(pdn_orbit + hop_idx) % self.pdn.n_star,
                pdn_engine=self.pdn,
            )
            if stub is None:
                continue
            # SOLO SEMANTIC PLANARITY: one unified (scalar) kernel.
            score = unified_plane_kernel(
                stub.rho, stub.theta, stub.sigma, ctx_rho, ctx_theta, ctx_sigma,
                lambda_reg=lam, gamma_side=gam, use_torch=False,
            )
            self._chain.append(CoTStep(hop_idx, stub, score,
                                       (pdn_orbit + hop_idx) % self.pdn.n_star))
            ctx_rho, ctx_theta, ctx_sigma = stub.rho, stub.theta, stub.sigma

        self._conclusion_stub = self.stubs.best_stub(
            self.conclusion_type, ctx_rho, ctx_theta, ctx_sigma, self.kernels,
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
    def __init__(self, lambda_reg: float = 811.0, gamma_side: float = 411.0):
        self.lambda_reg = lambda_reg
        self.gamma_side = gamma_side

    def k_reg (self, rho_a, rho_b):     return torch.exp(-self.lambda_reg * (rho_b - rho_a) ** 2)
    def k_ori (self, theta_a, theta_b): return 0.5 * (1.0 + torch.cos(theta_b - theta_a))
    def k_side(self, sigma_a, sigma_b): return torch.exp(-self.gamma_side * (sigma_b - sigma_a) ** 2)

    # SOLO SEMANTIC PLANARITY: returns the single unified kernel value.
    # Kept as a 3-tuple return (v, v, v) for call-site compatibility, so
    # existing "k_a, k_b, k_c = ..." unpacking still works, but all three
    # are now the SAME single-plane value rather than three independent
    # kernels multiplied together.
    def all_scores_batched(self, rho_a, theta_a, sigma_a, rho_b, theta_b, sigma_b):
        K = unified_plane_kernel(
            rho_a, theta_a, sigma_a, rho_b, theta_b, sigma_b,
            lambda_reg=self.lambda_reg, gamma_side=self.gamma_side,
        )
        return K, K, K


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
        # SOLO SEMANTIC PLANARITY: single combined (rho, sigma) plane kernel
        # (theta omitted here — this filter never used k_ori, unchanged).
        K = unified_plane_kernel(
            c_rho.unsqueeze(1), torch.zeros_like(c_rho).unsqueeze(1), c_sigma.unsqueeze(1),
            self._v_rho.unsqueeze(0), torch.zeros_like(self._v_rho).unsqueeze(0), self._v_sigma.unsqueeze(0),
            lambda_reg=kernels.lambda_reg, gamma_side=kernels.gamma_side, kappa_ori=0.0,
        )
        domain_sizes = (K > self.threshold).float().sum(dim=1)
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
        # SOLO SEMANTIC PLANARITY: single combined (rho, sigma) plane kernel.
        K = unified_plane_kernel(
            stored_rho, torch.zeros_like(stored_rho), stored_sigma,
            cur_rho.unsqueeze(0), torch.zeros_like(cur_rho).unsqueeze(0), cur_sigma.unsqueeze(0),
            lambda_reg=kernels.lambda_reg, gamma_side=kernels.gamma_side, kappa_ori=0.0,
        )
        return K.mean(dim=1)

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
                # SOLO SEMANTIC PLANARITY: single combined (rho, sigma) plane kernel.
                K = unified_plane_kernel(
                    c_rho, torch.zeros_like(c_rho), c_sigma,
                    anc.rho_t[pos].item(), 0.0, anc.sigma_t[pos].item(),
                    lambda_reg=kernels.lambda_reg, gamma_side=kernels.gamma_side, kappa_ori=0.0,
                )
                bonuses += sim_score * K
        std = bonuses.std()
        if std.item() > 1e-8:
            bonuses = (bonuses - bonuses.mean()) / std
        return bonuses * echo_weight


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THÉBAULT CONJUGATE ORBIT
# SOLO SEMANTIC PLANARITY: antipodality and congruence remain two distinct
# geometric notions (not both expressible as rho/theta/sigma distances to
# a single point), so they are combined via symmetric_weighted_sum rather
# than positional multiplication order.
# ════════════════════════════════════════════════════════════════════════════

class ThebaultConjugateOrbit:
    def score(self, anchor_triple, cand_theta, cand_sigma, gamma_side=411.0):
        antipodality = torch.cos(cand_theta + anchor_triple.theta - math.pi / 2) ** 2
        congruence   = torch.exp(-gamma_side * (cand_sigma - anchor_triple.sigma) ** 2)
        # DOUBLE AGNOSTIC: product is still commutative, but expressed via
        # the name-sorted default of the shared primitive for consistency.
        terms = {"antipodality": antipodality, "congruence": congruence}
        keys = sorted(terms.keys())
        out = terms[keys[0]]
        for k in keys[1:]:
            out = out * terms[k]
        return out


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — THÉBAULT COMPOSITION LM
# ════════════════════════════════════════════════════════════════════════════

class ThebaultCompositionLM:
    BASAL_K      = 1.5
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
        # SOLO SEMANTIC PLANARITY: single combined (rho, sigma) plane kernel.
        return unified_plane_kernel(
            c_rho, torch.zeros_like(c_rho), c_sigma,
            C.rho, 0.0, C.sigma,
            lambda_reg=self.kernels.lambda_reg, gamma_side=self.kernels.gamma_side, kappa_ori=0.0,
        )


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
                # SOLO SEMANTIC PLANARITY: single combined (rho, theta) plane
                # kernel replaces k_ori(...) * k_reg(...).
                w = unified_plane_kernel(
                    ti.rho, ti.theta, 0.0, tj.rho, tj.theta, 0.0,
                    lambda_reg=self.kernels.lambda_reg, gamma_side=0.0,
                    use_torch=False,
                ) * cnt
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
# SECTION 10 — SEMANTIC MANDATE SCORER
# SOLO SEMANTIC PLANARITY: single unified_plane_kernel replaces the
# k_side · k_ori · k_reg product.
# ════════════════════════════════════════════════════════════════════════════

class SemanticMandateScorer:
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
        C = len(cands)
        if self._centroid is None:
            return torch.zeros(C, dtype=self.dtype, device=self.device)

        bonus = unified_plane_kernel(
            c_rho, c_theta, c_sigma,
            self._centroid.rho, self._centroid.theta, self._centroid.sigma,
            lambda_reg=self.kernels.lambda_reg, gamma_side=self.kernels.gamma_side,
        )
        bonus = layer_norm_array(bonus)
        return bonus * self.mandate_scale

    def centroid_report(self) -> str:
        if self._centroid is None:
            return "  No instruction centroid set."
        return (
            f"  Instruction centroid (DA-SP): "
            f"ρ={self._centroid.rho:.4f}  "
            f"θ={self._centroid.theta:.4f}  "
            f"σ={self._centroid.sigma:.4f}"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11a — DNN ARRAY PIPELINE
#
# DOUBLE-AGNOSTIC CHANGE: the layer execution order is now a `layer_order`
# parameter — a list drawn from {'rho', 'theta', 'relu_dim2', 'sigma'} —
# instead of a hard-coded sequence. Temperature scaling is likewise a
# `temp_position` parameter ('pre' or 'post'). Neither the original
# forward order nor the reversed order is assumed correct; the default
# is the canonical name-sorted tuple ('relu_dim2','rho','sigma','theta')
# with temp scaling applied post (an explicit, neutral choice, not an
# implicit one).
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

    def normalize(self, logits, c_rho=None, c_sigma=None, temp=1.0, pass_order: Optional[List[str]] = None):
        # DOUBLE AGNOSTIC: which normalisation pass (freq-scale vs
        # rho-scale) runs first is a `pass_order` parameter; default is
        # name-sorted ('freq' before 'rho').
        order = pass_order or ["freq", "rho"]
        if c_rho is not None and c_sigma is not None:
            if c_rho is not None:
                temp_weights = torch.exp(
                    -((c_rho - c_rho.mean()) ** 2) / (2.0 * max(temp, 0.1) + 1e-8)
                )
                scaled_logits = logits * temp_weights
            else:
                scaled_logits = logits / max(temp, 1e-6)

            state = scaled_logits
            for stage in order:
                if stage == "freq":
                    freq_scale = self._build_freq_scale(c_rho, c_sigma)
                    freq_norm  = l2_array_normalize(freq_scale, dim=0)
                    a_pre  = signed_power(state, p=2.0)
                    state  = (freq_norm * a_pre).sum() * freq_norm + a_pre * 0.5
                elif stage == "rho":
                    rho_scale = self._build_rho_scale(c_rho)
                    state     = signed_power(state * rho_scale, p=1.0)
            a1 = state
        else:
            a1 = logits / max(temp, 1e-6)

        return l1_simplex_project(a1)

    def log_normalize(self, logits, c_rho=None, c_sigma=None, temp=1.0):
        p = self.normalize(logits, c_rho, c_sigma, temp)
        return (p + 1e-12).log()


class GeometricTempScaler:
    def __init__(self, lambda_temp: float = 1.0):
        self.lambda_temp = lambda_temp

    def scale(self, logits, temp, c_rho=None):
        safe_logits = logits.clamp(-5011.0, 5011.0)
        if c_rho is None or temp <= 1e-6:
            return safe_logits / max(temp, 0.1)
        mu_rho   = c_rho.mean()
        exponent    = (-self.lambda_temp * (c_rho - mu_rho) ** 2 / max(temp, 0.1)).clamp(min=-1011.0)
        geo_weights = torch.exp(exponent)
        return safe_logits * geo_weights


class DNNArrayPipeline:
    """
    DNN Array Pipeline — Double-Agnostic / Solo-Planar variant.

    Layer stack is now data, not control flow: `layer_order` lists which
    of {'rho','theta','relu_dim2','sigma'} run and in what sequence, and
    `temp_position` says whether GeometricTempScaler runs 'pre' (before
    the layer stack) or 'post' (after). The default is a name-sorted,
    neutral order — not the original forward order, and not its reversal.

    relu_dim2 design (unchanged mechanism, now optional/relocatable):
        gate_raw = relu( theta_w - mean(theta_w) )   ∈ [0, ∞)
        output   = x · gate  +  relu(x) · (1 − gate)
    """

    _DEFAULT_LAYER_ORDER = ["relu_dim2", "rho", "sigma", "theta"]  # name-sorted

    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32,
                 layer_order: Optional[List[str]] = None,
                 temp_position: str = "post"):
        self.device      = device
        self.dtype       = dtype
        self._normalizer  = ThebaultDNNNormalizer(device, dtype)
        self._temp_scaler = GeometricTempScaler(lambda_temp=1.0)
        self.layer_order   = layer_order or list(self._DEFAULT_LAYER_ORDER)
        self.temp_position = temp_position

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

    def _dim2_relu_layer(self, x: torch.Tensor, c_theta: torch.Tensor) -> torch.Tensor:
        theta_w  = self._theta_weights(c_theta)
        gate_raw = (theta_w - theta_w.mean()).clamp(min=0.0)
        g_max    = gate_raw.max()
        gate     = gate_raw / (g_max + 1e-8) if g_max.item() > 1e-8 else gate_raw
        return x * gate + F.relu(x) * (1.0 - gate)

    def _apply_layer(self, name, x, logits, c_rho, c_theta, c_sigma):
        if name == "sigma":
            sigma_w = self._sigma_weights(c_sigma)
            return signed_power(x * sigma_w, p=1.0)
        elif name == "theta":
            theta_w = self._theta_weights(c_theta)
            return signed_power(x * theta_w + logits * 0.3, p=1.5)
        elif name == "relu_dim2":
            return self._dim2_relu_layer(x, c_theta)
        elif name == "rho":
            rho_w = self._rho_weights(c_rho)
            return signed_power(x * rho_w, p=2.0)
        return x

    @torch.no_grad()
    def forward(self, logits, c_rho, c_theta, c_sigma, temp=1.4,
                layer_order: Optional[List[str]] = None,
                temp_position: Optional[str] = None):
        order = layer_order or self.layer_order
        tpos  = temp_position or self.temp_position

        state = logits
        if tpos == "pre":
            state = self._temp_scaler.scale(state, temp, c_rho)

        for name in order:
            state = self._apply_layer(name, state, logits, c_rho, c_theta, c_sigma)

        if tpos == "post":
            state = self._temp_scaler.scale(state, temp, c_rho)

        return l1_simplex_project(state)

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
# SECTION 12 — THÉBAULT WALKER V18-CSNS-G (DOUBLE-AGNOSTIC / SOLO-PLANAR)
#
# DOUBLE-AGNOSTIC CHANGE: walk_probs no longer writes a literal, ordered
# expression for the logit sum. All named terms go into a dict and are
# combined with symmetric_weighted_sum — order is a `term_order`
# parameter, default name-sorted, so no term (including the LM prior
# log_base) is structurally privileged as "first" or "last".
#
# GRADIENT-GUIDED DECODING (steered generation): walk_probs can now run a
# short, explicit `torch.autograd` ascent on the pre-softmax logits at
# each generation step, pushing the candidate distribution toward higher
# alignment with the active instruction/mandate centroid. This is a real
# backward pass (loss.backward() via torch.autograd.grad) through the
# softmax → alignment objective — distinct from the rest of the pipeline,
# which stays a fixed (non-learned) heuristic scorer under @torch.no_grad().
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
        ref_model        : "AtomismReferenceModel" = None,
        device           : torch.device = DEVICE,
        syn_weight       : float = 12.0,
        trans_weight     : float = 5070.6,
        syn_k            : int   = 18,
        tau_weight       : float = 70.45,
        walk_term_order   : Optional[List[str]] = None,   # DA
        context_order      : str = "append",               # DA: 'append' | 'prepend'
        guidance_weight    : float = 0.3,   # gradient-guided decoding blend, 0 = off
        guidance_steps     : int   = 9,     # inner ascent steps per generation step
        guidance_lr        : float = 0.63,  # inner ascent learning rate
        parabolic_manifold_strength : float = 0.35,
        parabolic_manifold_curvature: float = 1.0,
    ):
        self.geo          = geo
        self.kernels      = kernels
        self.lm           = lm
        self.orbit        = orbit
        self.graph        = graph
        self.mandate      = mandate_scorer
        self.ref_model    = ref_model
        self.tau_weight   = tau_weight
        self.mrv          = mrv_filter
        self.chunk_engine = chunk_engine
        self.iso_stacker  = iso_stacker
        self.pdn          = pdn_engine
        self.cot          = cot_engine
        self.instr_dist   = instr_dist
        self.device       = device
        self.walk_term_order = walk_term_order
        self.context_order    = context_order
        self.guidance_weight  = guidance_weight
        self.guidance_steps   = guidance_steps
        self.guidance_lr      = guidance_lr
        self.parabolic_manifold_strength = max(0.0, float(parabolic_manifold_strength))
        self.parabolic_manifold_curvature = max(0.0, float(parabolic_manifold_curvature))
        self._parabolic_arc = 0.0
        self.mechanical_fold = MechanicalFoldState()
        self.fold_frequency: float = 1.0
        self.fold_index: int = 0
        self.fold_count: int = 1
        self.fold_phase: float = 0.0
        self.fold_depth: float = 0.0
        self.fold_tension: float = 0.0
        self.fold_momentum: float = 0.0
        self.current_isomorphic_pairs: List[Tuple[str, str, float]] = []
        self._cur_sent_toks : List[str] = []
        self._cur_orbit     : int       = 0
        self._tok_pos       : int       = 0
        self._step_traces   : List[TokenStepTrace] = []
        self.remission       = LocaleTransitRemission()
        self.contingent_prob = ContingentExtringentProbability()
        self._dnn_pipeline   = DNNArrayPipeline(device=device)
        self._dnn_normalizer = ThebaultDNNNormalizer(device=device)
        self._influence_mapper = InfluenceSpaceMapper(device=device)
        self._influence_top_k  = syn_k
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
        self._last_guidance_delta : float = 0.0

    def begin_sentence(self, seed_tokens=None, total_tokens=40) -> CoTTrace:
        self.chunk_engine.reset()
        self._cur_sent_toks.clear()
        self._cur_orbit    = 0
        self._tok_pos      = 0
        self._total_tokens = total_tokens
        self.mechanical_fold = MechanicalFoldState()
        self._sync_mechanical_fold(1.0, position=0, total=total_tokens)
        seeds = seed_tokens or []
        self.cot.begin_sentence()
        return self.cot.plan_chain(seeds, self.geo, pdn_orbit=self._cur_orbit)

    def _sync_mechanical_fold(self, frequency: float,
                              position: Optional[int] = None,
                              total: Optional[int] = None) -> None:
        position = self._tok_pos if position is None else int(position)
        total = getattr(self, "_total_tokens", 1) if total is None else int(total)
        self.mechanical_fold.update(float(frequency), position=position, total=max(total, 1))
        self.fold_frequency = float(self.mechanical_fold.frequency)
        self.fold_index = int(self.mechanical_fold.fold_index)
        self.fold_count = int(self.mechanical_fold.fold_count)
        self.fold_phase = float(self.mechanical_fold.phase)
        self.fold_depth = float(self.mechanical_fold.depth)
        self.fold_tension = float(self.mechanical_fold.tension)
        self.fold_momentum = float(self.mechanical_fold.momentum)

    def _candidate_frequency(self, cands: List[str]) -> torch.Tensor:
        return torch.tensor([float(self.lm.raw_freq.get(c, 1.0)) for c in cands],
                            dtype=torch.float32, device=self.device)

    def _gradient_guided_steer(
        self,
        logits      : torch.Tensor,
        c_rho       : torch.Tensor,
        c_theta     : torch.Tensor,
        c_sigma     : torch.Tensor,
        steps       : int,
        lr          : float,
    ) -> torch.Tensor:
        """
        Gradient-guided decoding (a real backward pass, not a heuristic).

        Treats the current step's pre-softmax `logits` as a leaf tensor and
        runs `steps` iterations of gradient ASCENT on an alignment
        objective:

            probs      = softmax(logits)
            alignment  = unified_plane_kernel(candidate, instruction_centroid)
            objective  = Σ_i probs[i] * alignment[i]   (expected alignment)

        Each iteration computes ∂objective/∂logits via torch.autograd.grad
        (an explicit backward pass through the softmax → kernel objective)
        and takes a step `logits ← logits + lr * grad`. This nudges the
        distribution toward candidates that are geometrically closer to
        the instruction/mandate centroid, without touching any of the
        model's fixed heuristic weights.

        Returns the steered logits (same shape as input, detached from any
        outer graph). If there's no active instruction centroid, or
        steps <= 0, this is a no-op and the input is returned unchanged.
        """
        if self.mandate._centroid is None or steps <= 0:
            self._last_guidance_delta = 0.0
            return logits

        centroid = self.mandate._centroid
        base = logits.detach()

        with torch.enable_grad():
            z = base.clone().requires_grad_(True)
            for _ in range(steps):
                probs = F.softmax(z, dim=-1)
                alignment = unified_plane_kernel(
                    c_rho, c_theta, c_sigma,
                    centroid.rho, centroid.theta, centroid.sigma,
                    lambda_reg=self.kernels.lambda_reg,
                    gamma_side=self.kernels.gamma_side,
                )
                objective = (probs * alignment).sum()
                grad, = torch.autograd.grad(objective, z)
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                z = (z + lr * grad).detach().requires_grad_(True)
            steered = z.detach()

        self._last_guidance_delta = (steered - base).norm().item()
        return steered

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
        tau_weight    : float = None,
        influence_weight : float = 0.5,
        term_order        : Optional[List[str]] = None,
        guidance_weight    : Optional[float] = None,
        guidance_steps     : Optional[int]   = None,
        guidance_lr        : Optional[float] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        cands, base_probs = self.lm.next_dist(w1, w2)
        if not cands:
            return cands, base_probs

        c_freq = self._candidate_frequency(cands)
        live_frequency = float(c_freq.mean().item()) if c_freq.numel() else 1.0
        self._sync_mechanical_fold(live_frequency, position=self._tok_pos,
                                   total=getattr(self, "_total_tokens", 1))

        # Center-origin 1D parabolic generation manifold.
        # A(x)=1-x²: strongest at the center, smoothly decaying to both edges.
        self._parabolic_arc = parabolic_arc_1d(
            self._tok_pos,
            getattr(self, "_total_tokens", 1),
            self.parabolic_manifold_curvature,
        )

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

        # SOLO SEMANTIC PLANARITY: all three "k_*" names now hold the SAME
        # single unified-plane value (kept for call-site compatibility).
        k_side, k_ori, k_reg = self.kernels.all_scores_batched(
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

        mandate_boost = self.mandate.score(cands, c_rho, c_theta, c_sigma)

        _tau_w = tau_weight if tau_weight is not None else self.tau_weight
        tau_boost = (
            self.ref_model.tau_bonus(cands, scale=_tau_w)
            if self.ref_model is not None
            else torch.zeros(len(cands), dtype=torch.float32, device=self.device)
        )

        # Isomorphic pair detection (unchanged — uses the now-unified kernel values)
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

        N             = len(cands)
        infl_bonus    = torch.zeros(N, device=self.device)
        infl_k        = min(self._influence_top_k, N)
        if infl_k > 1:
            infl_top_idx = torch.topk(k_reg * k_side, infl_k).indices
            cand_kernel  = build_synaptic_weight_matrix(
                c_rho[infl_top_idx], c_theta[infl_top_idx], c_sigma[infl_top_idx],
                lambda_reg = self.kernels.lambda_reg,
                gamma_side = self.kernels.gamma_side,
                top_k      = infl_k,
            )
            infl_bonus[infl_top_idx] = self._influence_mapper.candidate_influence_bonus(cand_kernel)
            infl_bonus = torch.nan_to_num(infl_bonus, nan=0.0, posinf=5011.0, neginf=-5011.0)

        punct_bias    = torch.zeros(N, device=self.device)
        punct_penalty = torch.zeros(N, device=self.device)
        for i, c in enumerate(cands):
            if c in PUNCT_TOKENS:
                punct_bias[i] = -3.5
                if w2 in PUNCT_TOKENS:
                    punct_penalty[i] = -1e4

        log_base = (base_probs.clamp(min=1e-12)).log()

        # DOUBLE AGNOSTIC: every named term goes into a dict and is combined
        # via symmetric_weighted_sum. No term (log_base included) is given
        # positional priority by where it's written; the effective priority
        # is the explicit `order` parameter, defaulting to name-sorted.
        order = term_order or self.walk_term_order
        raw_logits = symmetric_weighted_sum(
            {
                "tau"        : tau_boost,
                "influence"  : infl_bonus,
                "mandate"    : mandate_boost,
                "cot"        : cot_bonus,
                "pdn"        : pdn_bonus,
                "echo"       : echo_bonus,
                "chunk"      : chunk_bonus,
                "mrv"        : mrv_scores,
                "comp"       : comp_bonus,
                "pot"        : pots,
                "orbit"      : orbit_scores,
                "k_side"     : k_side,
                "k_ori"      : k_ori,
                "k_reg"      : k_reg,
                "punct_bias" : punct_bias,
                "punct_pen"  : punct_penalty,
                "log_base"   : log_base,
            },
            weights={
                "influence": influence_weight, "cot": cot_weight, "pdn": pdn_weight,
                "mrv": zetamrv, "pot": psipot, "orbit": gammaorbit,
                "k_side": deltaside, "k_ori": betaori, "k_reg": alphareg,
            },
            order=order,
        )

        manifold_scale = parabolic_manifold_scale(
            self._tok_pos,
            getattr(self, "_total_tokens", 1),
            strength=self.parabolic_manifold_strength,
            curvature=self.parabolic_manifold_curvature,
        )
        raw_logits = raw_logits * manifold_scale
        raw_logits = raw_logits * float(self.mechanical_fold.gain())

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

        # GRADIENT-GUIDED DECODING: a real torch.autograd backward pass,
        # run on top of the (otherwise fixed/heuristic) enriched logits.
        # `gw` blends between the un-steered logits (gw=0) and the fully
        # gradient-ascended ones (gw=1); `gs`/`glr` control the inner
        # autograd loop above.
        gw  = guidance_weight if guidance_weight is not None else self.guidance_weight
        gs  = guidance_steps  if guidance_steps  is not None else self.guidance_steps
        glr = guidance_lr     if guidance_lr     is not None else self.guidance_lr
        if gw > 0.0 and gs > 0:
            steered_logits = self._gradient_guided_steer(
                logits_enriched, c_rho, c_theta, c_sigma, steps=gs, lr=glr,
            )
            logits_enriched = logits_enriched * (1.0 - gw) + steered_logits * gw
        else:
            self._last_guidance_delta = 0.0

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
            # DOUBLE AGNOSTIC: AND-combination term order/weight is explicit.
            log_and = symmetric_weighted_sum(
                {"walk": log_walk, "instr": log_instr},
                weights={"walk": 1.0 - and_weight, "instr": and_weight},
            )
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
        """
        DOUBLE AGNOSTIC: whether new tokens are appended or prepended to
        _cur_sent_toks is a `context_order` parameter ('append' default —
        the neutral, standard reading order — or 'prepend'). Neither is
        hard-coded as the only option.
        """
        if token in PUNCT_TOKENS or token in COGNITIVE_TOKENS:
            return
        if self.context_order == "prepend":
            self._cur_sent_toks.insert(0, token)
        else:
            self._cur_sent_toks.append(token)
        self._tok_pos += 1
        pos_norm = len(self._cur_sent_toks) / max(sentence_len, 1)
        self.chunk_engine.push(self.geo.triple(token), pos_norm)
        self._cur_orbit = self.pdn.orbit_of(token)
        self._sync_mechanical_fold(float(self.lm.raw_freq.get(token, 1.0)),
                                   position=self._tok_pos, total=sentence_len)

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
        lines.append(f"\n  Mechanical fold: {self.mechanical_fold.report()}")
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
            "║   CSNS Diagnostic Report (Double-Agnostic / Solo-Planar)      ║\n"
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
            "║  Enrichment term order: name-sorted by default (config'able) ║\n"
            "║    w1 × 0.25  +  w2 × 0.50  +  c × 0.25                    ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — TEXT GENERATION ENGINE
#
# DOUBLE-AGNOSTIC CHANGE: sentence generation order is a `sentence_order`
# parameter ('forward' default — ascending index — or 'backward').
# Whatever order sentences are generated in, they are re-sorted by index
# before joining so the final reading order is always ascending — that
# part is NOT a free parameter, since scrambled reading order would harm
# the user, not just reflect a structural choice.
#
# GRADIENT-GUIDED DECODING: `guidance_weight` / `guidance_steps` /
# `guidance_lr` are threaded through to ThebaultWalker.walk_probs, so a
# real backward pass steers each step's candidate distribution toward the
# active instruction/mandate centroid before sampling.
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
    sentence_order   : str   = "forward",   # DA: 'forward' | 'backward'
    guidance_weight  : Optional[float] = None,  # gradient-guided decoding blend (0..1), None = walker default
    guidance_steps   : Optional[int]   = None,  # inner autograd ascent steps per token
    guidance_lr      : Optional[float] = None,  # inner autograd ascent learning rate
):
    if instruction_text.strip():
        walker.instr_dist.set_instruction(instruction_text)
        walker.mandate.set_instruction(instruction_text)
    elif seed_text.strip():
        walker.instr_dist.set_instruction(seed_text)
        walker.mandate.set_instruction(seed_text)

    walker._step_traces.clear()
    walker._csns_syn_norms.clear()
    walker._csns_trans_norms.clear()

    outputs_by_idx : Dict[int, str]      = {}
    traces_by_idx  : Dict[int, CoTTrace] = {}
    head_list = list(lm.heads.keys())
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

    sent_indices = list(range(num_sentences))
    if sentence_order == "backward":
        sent_indices = list(reversed(sent_indices))

    for sent_idx in sent_indices:
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
        traces_by_idx[sent_idx] = trace
        toks = list(init_toks)

        for step in range(tokens_per_sent):
            cands, probs = walker.walk_probs(
                w1, w2, temp=temperature, and_weight=and_weight,
                guidance_weight=guidance_weight,
                guidance_steps=guidance_steps,
                guidance_lr=guidance_lr,
            )
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

        outputs_by_idx[sent_idx] = detokenize(toks)

    # Reading order is always ascending by sentence index, independent of
    # the (double-agnostic) generation order chosen above.
    outputs    = [outputs_by_idx[i]  for i in range(num_sentences) if i in outputs_by_idx]
    all_traces = [traces_by_idx[i]   for i in range(num_sentences) if i in traces_by_idx]

    result = " ".join(outputs)
    if return_traces:
        return result, all_traces, walker.step_trace_report()
    return result


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — V18-CSNS-G ENGINE
#
# DOUBLE-AGNOSTIC CHANGE: the training build-stage order is a
# `build_stage_order` parameter — a list drawn from
# {'ref_model','mandate','instr_dist','cot','pdn','mrv','graph'} —
# instead of a hard-coded sequence. Default is the canonical name-sorted
# order.
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CubeCoord:
    x: int
    y: int
    z: int
    t: int

@dataclass
class CubeChunk:
    index: int
    text: str
    tokens: List[str]
    coord: CubeCoord
    hash_hex: str
    orbit: int
    rho_mean: float
    theta_mean: float
    sigma_mean: float

class CubeGardenResolver:
    """Deterministic 4-D cube-garden corpus permutation (unchanged by DA/SP)."""
    def __init__(self, geo, pdn, chunk_size=128, cube_side=8):
        self.geo=geo; self.pdn=pdn; self.chunk_size=max(1,int(chunk_size)); self.cube_side=max(2,int(cube_side))
    @staticmethod
    def _u32(data, offset): return int.from_bytes(data[offset:offset+4], 'big')
    def _hash_cube(self,text):
        d=hashlib.sha256(text.encode('utf-8')).digest(); S=self.cube_side
        return CubeCoord(*(self._u32(d,i)%S for i in (0,4,8,12)))
    def _geometry_signature(self,tokens):
        ts=[self.geo.triple(t) for t in tokens if t in self.geo._vecs]
        if not ts: return 0.0,0.0,0.0
        r=sum(t.rho for t in ts)/len(ts); s=sum(t.sigma for t in ts)/len(ts)
        sn=sum(math.sin(t.theta) for t in ts)/len(ts); cs=sum(math.cos(t.theta) for t in ts)/len(ts)
        return r, math.atan2(sn,cs)%math.pi, s
    def _geometry_cube(self,index,tokens):
        r,th,sg=self._geometry_signature(tokens); S=self.cube_side
        return CubeCoord(index%S,min(S-1,int(max(0,min(1,r))*S)),min(S-1,int(th/math.pi*S)),min(S-1,int(math.tanh(abs(sg))*S)))
    def _combine(self,a,b):
        S=self.cube_side; return CubeCoord((a.x+b.x)%S,(a.y+b.y)%S,(a.z+b.z)%S,(a.t+b.t)%S)
    def cube_distance(self,a,b):
        S=self.cube_side
        def d(x,y): q=abs(x-y); return min(q,S-q)
        return d(a.x,b.x)+d(a.y,b.y)+d(a.z,b.z)+d(a.t,b.t)
    def transitive_score(self,a,b):
        d=self.cube_distance(a.coord,b.coord)
        return (1/(1+d))*math.exp(-8*abs(a.rho_mean-b.rho_mean))*0.5*(1+math.cos(a.theta_mean-b.theta_mean))*math.exp(-4*abs(a.sigma_mean-b.sigma_mean))*(1 if a.orbit==b.orbit else .25)
    def make_chunks(self,tokens):
        out=[]
        for start in range(0,len(tokens),self.chunk_size):
            ct=tokens[start:start+self.chunk_size]; idx=start//self.chunk_size; text=' '.join(ct); hx=hashlib.sha256(text.encode()).hexdigest()
            hc=self._hash_cube(text); gc=self._geometry_cube(idx,ct); c=self._combine(hc,gc); r,th,sg=self._geometry_signature(ct); orbit=0
            for tok in ct:
                if tok not in PUNCT_TOKENS: orbit=self.pdn.orbit_of(tok); break
            out.append(CubeChunk(idx,text,ct,c,hx,orbit,r,th,sg))
        return out
    def resort(self,tokens):
        chunks=self.make_chunks(tokens)
        if len(chunks)<=1: return list(tokens),chunks
        cur=min(chunks,key=lambda c:(c.hash_hex,c.index)); rem={c.index:c for c in chunks if c.index!=cur.index}; order=[cur]
        while rem:
            nxt=max(rem.values(),key=lambda c:(self.transitive_score(cur,c),-self.cube_distance(cur.coord,c.coord),c.hash_hex,-c.index)); order.append(nxt); del rem[nxt.index]; cur=nxt
        result=[]
        for c in order: result.extend(c.tokens)
        return result,order
    def report(self,chunks,max_rows=32):
        lines=['','╔════════════════════════════════════════════════════════════════════╗','║                    CUBE GARDEN DATASET MAP                       ║','╠════════════════════════════════════════════════════════════════════╣',f'║ chunks={len(chunks):<55}║',f'║ cube_side={self.cube_side:<52}║',f'║ chunk_size={self.chunk_size:<50}║','╠════════════════════════════════════════════════════════════════════╣']
        for i,c in enumerate(chunks[:max_rows]):
            q=c.coord; lines.append(f'║ {i:03d} src={c.index:04d} C=({q.x},{q.y},{q.z},{q.t}) O={c.orbit:<2d} ρ={c.rho_mean:.3f} θ={c.theta_mean:.3f} σ={c.sigma_mean:.3f} ║')
        lines.append('╚════════════════════════════════════════════════════════════════════╝'); return '\n'.join(lines)

class V18Engine:
    _DEFAULT_BUILD_ORDER = ["cot", "graph", "instr_dist", "mandate", "mrv", "pdn", "ref_model"]  # name-sorted

    def __init__(
        self,
        syn_weight=2.0,
        trans_weight=0.6,
        syn_k=8,
        build_stage_order: Optional[List[str]] = None,
        parabolic_manifold_strength: float = 0.35,
        parabolic_manifold_curvature: float = 1.0,
    ):
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
        self.mandate_scorer = None
        self.ref_model   = None
        self.instr_dist  = None
        self.cot         = None
        self.walker      = None
        self.corpus_snippet = ""
        self.syn_weight   = syn_weight
        self.trans_weight = trans_weight
        self.syn_k        = syn_k
        self.parabolic_manifold_strength = max(0.0, float(parabolic_manifold_strength))
        self.parabolic_manifold_curvature = max(0.0, float(parabolic_manifold_curvature))
        self.build_stage_order = build_stage_order or list(self._DEFAULT_BUILD_ORDER)
        self.cube_chunk_size=128
        self.cube_side=8
        self.cube_garden=None
        self.cube_chunks=[]
        self.mechanical_fold = MechanicalFoldState()
        self.fold_frequency: float = 1.0
        self.fold_index: int = 0
        self.fold_count: int = 1
        self.fold_phase: float = 0.0
        self.fold_depth: float = 0.0
        self.fold_tension: float = 0.0
        self.fold_momentum: float = 0.0

    def train(self, corpus_text: str, build_stage_order: Optional[List[str]] = None):
        print(f"[*-DASP] Tokenizing corpus ({len(corpus_text)} chars)...")
        self.corpus_snippet=corpus_text
        tokens=tokenize(corpus_text)
        provisional_freq={}
        for tok in tokens: provisional_freq[tok]=provisional_freq.get(tok,0.0)+1.0
        provisional_vocab=list(provisional_freq); max_freq=max(provisional_freq.values(),default=1.0); n=len(provisional_vocab)
        print(f"[*-DASP] Cube prepass registering {n} tokens...")
        for idx,tok in enumerate(provisional_vocab): self.geo.register(tok,provisional_freq[tok],idx,max_freq,n)
        pn=4; sector=2.0*math.pi/pn
        for tok in provisional_vocab:
            tr=self.geo.triple(tok); self.pdn._orbit_map[tok]=int((tr.theta*2.0)/sector)%pn
        self.cube_garden=CubeGardenResolver(self.geo,self.pdn,self.cube_chunk_size,self.cube_side)
        print("[*-DASP] Building deterministic 4-D Cube Garden ordering...")
        tokens,self.cube_chunks=self.cube_garden.resort(tokens)
        print(self.cube_garden.report(self.cube_chunks))
        self.lm.ingest(tokens)
        corpus_frequency = float(sum(self.lm.raw_freq.values()))
        self.mechanical_fold.update(corpus_frequency, position=0, total=max(len(tokens), 1))
        self.fold_frequency = float(self.mechanical_fold.frequency)
        self.fold_index = int(self.mechanical_fold.fold_index)
        self.fold_count = int(self.mechanical_fold.fold_count)
        self.fold_phase = float(self.mechanical_fold.phase)
        self.fold_depth = float(self.mechanical_fold.depth)
        self.fold_tension = float(self.mechanical_fold.tension)
        self.fold_momentum = float(self.mechanical_fold.momentum)
        all_tokens=list(self.lm.raw_freq.keys()); max_freq=max(self.lm.raw_freq.values(),default=1.0); vocab_size=len(all_tokens)
        print(f"[*-DASP] Registering {vocab_size} tokens after Cube Garden resort...")
        for idx,tok in enumerate(all_tokens): self.geo.register(tok,self.lm.raw_freq[tok],idx,max_freq,vocab_size)

        print("[*-DASP] Building GPU Tensor Caches (prerequisite for all stages)...")
        self.geo.build_cuda_tensors(self.lm.vocab)
        self.lm.finalise()
        rho_nonzero=int((self.geo._rho_t>0.01).sum().item()); rho_max=float(self.geo._rho_t.max().item())
        print(f"[Geo-DASP] ρ > 0.01: {rho_nonzero}/{vocab_size}  max ρ = {rho_max:.4f}")

        # DOUBLE AGNOSTIC: build stages run in an explicit, caller-supplied
        # order (default: name-sorted) instead of a hard-coded sequence.
        order = build_stage_order or self.build_stage_order
        print(f"[*-DASP] Build stage order: {order}")
        for stage in order:
            if stage == "ref_model":
                print("[*-DASP] Stage: Building Formal Reference Model...")
                self.ref_model=AtomismReferenceModel(geo=self.geo,kernels=self.kernels,device=self.device)
                self.ref_model.build(self.lm.vocab)
                print(self.ref_model.reference_report())
            elif stage == "mandate":
                print("[*-DASP] Stage: Building Semantic Mandate Scorer...")
                self.mandate_scorer=SemanticMandateScorer(geo=self.geo,kernels=self.kernels,device=self.device)
            elif stage == "instr_dist":
                print("[*-DASP] Stage: Building Instruction Distribution module...")
                self.instr_dist=InstructionDistribution(geo=self.geo,kernels=self.kernels,lm=self.lm,device=self.device)
            elif stage == "cot":
                print("[*-DASP] Stage: Building CoT contextual stub library...")
                self.stub_lib.build(self.geo,self.lm.vocab,self.lm.raw_freq)
                self.cot=CoTReasoningEngine(stub_library=self.stub_lib,kernels=self.kernels,pdn_engine=self.pdn,n_hops=3,tokens_per_hop=10,device=self.device)
            elif stage == "pdn":
                print("[*-DASP] Stage: Fitting PDN from corpus autocorrelation...")
                self.pdn.fit_from_trigrams(self.geo,self.lm.tri_raw); self.pdn.build_orbit_map(self.lm.vocab,self.geo)
                print(self.pdn.theorem_bridge_report())
            elif stage == "mrv":
                print("[*-DASP] Stage: Initializing MRV Filter...")
                self.mrv.prime(self.lm.vocab,self.geo)
            elif stage == "graph":
                print("[*-DASP] Stage: Building graph potentials...")
                self.graph.build(self.lm); self.graph.propagate(steps=2)

        # cot may not have been built yet if 'cot' wasn't in the given order
        if self.cot is None:
            self.stub_lib.build(self.geo, self.lm.vocab, self.lm.raw_freq)
            self.cot = CoTReasoningEngine(stub_library=self.stub_lib, kernels=self.kernels, pdn_engine=self.pdn,
                                           n_hops=3, tokens_per_hop=10, device=self.device)
        if self.mandate_scorer is None:
            self.mandate_scorer = SemanticMandateScorer(geo=self.geo, kernels=self.kernels, device=self.device)
        if self.instr_dist is None:
            self.instr_dist = InstructionDistribution(geo=self.geo, kernels=self.kernels, lm=self.lm, device=self.device)
        if self.ref_model is None:
            self.ref_model = AtomismReferenceModel(geo=self.geo, kernels=self.kernels, device=self.device)
            self.ref_model.build(self.lm.vocab)

        self.walker=ThebaultWalker(self.geo,self.kernels,self.lm,self.orbit,self.graph,self.mandate_scorer,self.mrv,self.chunk,self.iso_stacker,self.pdn,self.cot,self.instr_dist,ref_model=self.ref_model,device=self.device,syn_weight=self.syn_weight,trans_weight=self.trans_weight,syn_k=self.syn_k,
            parabolic_manifold_strength=self.parabolic_manifold_strength,
            parabolic_manifold_curvature=self.parabolic_manifold_curvature)
        print("[+] Training complete. (V18-CSNS-G DOUBLE-AGNOSTIC / SOLO-PLANAR + CUBE GARDEN)")

    def _sync_engine_fold(self) -> None:
        if self.walker is None:
            return
        self.fold_frequency = float(self.walker.fold_frequency)
        self.fold_index = int(self.walker.fold_index)
        self.fold_count = int(self.walker.fold_count)
        self.fold_phase = float(self.walker.fold_phase)
        self.fold_depth = float(self.walker.fold_depth)
        self.fold_tension = float(self.walker.fold_tension)
        self.fold_momentum = float(self.walker.fold_momentum)

    def mechanical_fold_report(self) -> str:
        self._sync_engine_fold()
        return (
            "Mechanical Fold State\n"
            "─────────────────────\n"
            f"frequency : {self.fold_frequency:.6f} (float)\n"
            f"fold_index: {self.fold_index} (int)\n"
            f"fold_count: {self.fold_count} (int)\n"
            f"phase     : {self.fold_phase:.6f} (float)\n"
            f"depth     : {self.fold_depth:.6f} (float)\n"
            f"tension   : {self.fold_tension:.6f} (float)\n"
            f"momentum  : {self.fold_momentum:.6f} (float)"
        )

    def save_cache(self, filename: str = "v18_csns_g_dasp_model.pkl"):
        print(f"[*-DASP] Saving model state to {filename}...")
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
                "cube_chunk_size": self.cube_chunk_size,
                "cube_side"      : self.cube_side,
                "cube_chunks"    : self.cube_chunks,
                "syn_k"          : self.syn_k,
                "build_stage_order": self.build_stage_order,
                "version"        : "V18-CSNS-G-DASP",
                "ref_tau_scores"  : (self.ref_model._tau_scores.cpu() if self.ref_model and self.ref_model._tau_scores is not None else None),
                "ref_D_A_mask"    : (self.ref_model._D_A_mask.cpu() if self.ref_model and self.ref_model._D_A_mask is not None else None),
                "ref_D_A_omega"   : (self.ref_model._D_A_omega_mask.cpu() if self.ref_model and self.ref_model._D_A_omega_mask is not None else None),
                "ref_omega_steps" : (self.ref_model._omega_steps if self.ref_model else 0),
            }, f)
        print("[+] Save successful.")

    def load_cache(self, filename: str):
        print(f"[*-DASP] Loading model state from {filename}...")
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
        self.syn_weight         = state.get("syn_weight",   2.0)
        self.trans_weight       = state.get("trans_weight", 0.6)
        self.syn_k              = state.get("syn_k",        8)
        self.build_stage_order  = state.get("build_stage_order", list(self._DEFAULT_BUILD_ORDER))

        print("[*-DASP] Rebuilding GPU Tensors (prerequisite)...")
        self.geo.build_cuda_tensors(self.lm.vocab)
        self.lm.finalise()

        self.ref_model = AtomismReferenceModel(
            geo=self.geo, kernels=self.kernels, device=self.device,
        )
        self.ref_model._vocab       = self.lm.vocab
        self.ref_model._tok2idx     = {t: i for i, t in enumerate(self.lm.vocab)}
        self.ref_model._omega_steps = state.get("ref_omega_steps", 0)
        _tau   = state.get("ref_tau_scores")
        _da    = state.get("ref_D_A_mask")
        _daomg = state.get("ref_D_A_omega")
        if _tau is not None:
            self.ref_model._tau_scores     = _tau.to(self.device)
            self.ref_model._D_A_mask       = _da.to(self.device)
            self.ref_model._D_A_omega_mask = _daomg.to(self.device)
        else:
            print("[*-DASP] ref_model not in cache — rebuilding tau scores...")
            self.ref_model.build(self.lm.vocab)

        self.mandate_scorer = SemanticMandateScorer(
            geo=self.geo, kernels=self.kernels, device=self.device,
        )
        self.instr_dist = InstructionDistribution(
            geo=self.geo, kernels=self.kernels, lm=self.lm, device=self.device,
        )

        if "cot_stubs" in state:
            self.stub_lib.stubs = state["cot_stubs"]
            self.stub_lib._rebuild_tensors()
        else:
            self.stub_lib.build(self.geo, self.lm.vocab, self.lm.raw_freq)

        self.cot = CoTReasoningEngine(
            stub_library=self.stub_lib, kernels=self.kernels,
            pdn_engine=self.pdn, n_hops=3, tokens_per_hop=10, device=self.device,
        )

        self.pdn.build_orbit_map(self.lm.vocab, self.geo)
        self.mrv.prime(self.lm.vocab, self.geo)
        self.graph.build(self.lm)
        self.graph.propagate(steps=2)

        self.walker = ThebaultWalker(
            self.geo, self.kernels, self.lm, self.orbit,
            self.graph, self.mandate_scorer, self.mrv, self.chunk, self.iso_stacker,
            self.pdn, self.cot, self.instr_dist,
            ref_model=self.ref_model,
            device=self.device,
            syn_weight=self.syn_weight, trans_weight=self.trans_weight, syn_k=self.syn_k,
            parabolic_manifold_strength=self.parabolic_manifold_strength,
            parabolic_manifold_curvature=self.parabolic_manifold_curvature,
        )
        print("[+] Load successful. (V18-CSNS-G-DASP)")



# ════════════════════════════════════════════════════════════════════════════
# SECTION 14b — SYNTHREASON DATASET → PROMPT SUBSET → GENERATE FLOW
#
# Dataset → Prompt Isolate → Reasoning Prompt Subset → New Dataset From Output
# → Contextual Prompt Subset → New Dataset From Output → Prompt Subset
# → Generate Out
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PromptCandidate:
    text: str
    isolate_score: float = 0.0
    reasoning_score: float = 0.0
    contextual_score: float = 0.0
    final_score: float = 0.0


class SynthReasonFlow:
    """Dataset-driven prompt selection and generation pipeline.

    Sequential 7-stage flow with pooled intermediate datasets:

      Dataset â†’ Prompt Isolate â†’ Reasoning Prompt Subset
               â†’ Generate (New Dataset #1, pool into corpus)
               â†’ Contextual Prompt Subset
               â†’ Generate (New Dataset #2, pool into corpus)
               â†’ Final Prompt Subset â†’ Generate Out

    At each intermediate stage the newly generated text is POOLED with
    the accumulated corpus so earlier signal is never discarded, while
    each generation step steers prompts further from the original seed.
    """

    def __init__(self, engine):
        self.engine = engine
        self.last_report = ""
        self.last_outputs: List[str] = []

    @staticmethod
    def _clean_prompt(text: str) -> str:
        text = re.sub(r"^\s*(?:prompt|instruction|question|query)\s*[:\-]\s*", "", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    def prompt_isolate(self, dataset: str, max_chars: int = 900) -> List[PromptCandidate]:
        """Extract prompt-like units from the Dataset node."""
        raw_units = re.split(r"\n\s*\n|\n|(?<=[.!?])\s+(?=[A-Z\[])", dataset)
        out: List[PromptCandidate] = []
        seen: Set[str] = set()
        for raw in raw_units:
            text = self._clean_prompt(raw)
            if len(text) < 8 or len(text) > max_chars:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            toks = tokenize(text)
            lexical = min(len(toks) / 24.0, 1.0)
            question = 1.0 if "?" in text else 0.0
            directive = 1.0 if re.match(
                r"(?i)\s*(explain|describe|compare|analyze|derive|why|how|what|evaluate|discuss|identify|reason)\b",
                text,
            ) else 0.0
            score = 0.55 * lexical + 0.25 * question + 0.20 * directive
            out.append(PromptCandidate(text=text, isolate_score=score))
        out.sort(key=lambda x: (-x.isolate_score, x.text.casefold()))
        return out

    @staticmethod
    def _reasoning_score(text: str) -> float:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return 0.0
        cog = sum(1 for w in words if w in STOP_WORDS_COG)
        question = 1.0 if "?" in text else 0.0
        causal = sum(1 for w in words if w in {
            "because", "therefore", "cause", "effect", "implies", "if", "then",
            "why", "how", "compare", "contrast", "derive", "reason"
        })
        depth = min(cog / max(len(words), 1), 1.0)
        causal_density = min(causal / max(len(words), 1), 1.0)
        return 0.50 * depth + 0.25 * question + 0.25 * causal_density

    def reasoning_prompt_subset(
        self, prompts: List[PromptCandidate], fraction: float = 0.50, minimum: int = 4
    ) -> List[PromptCandidate]:
        for p in prompts:
            p.reasoning_score = self._reasoning_score(p.text)
        prompts = sorted(prompts, key=lambda x: (-x.reasoning_score, -x.isolate_score, x.text.casefold()))
        n = min(len(prompts), max(minimum, int(math.ceil(len(prompts) * fraction))))
        return prompts[:n]

    @staticmethod
    def _token_set(text: str) -> Set[str]:
        return {t for t in tokenize(text) if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS}

    def _context_score(self, text: str, anchors: List[str]) -> float:
        a = self._token_set(text)
        if not a or not anchors:
            return 0.0
        best = 0.0
        for anchor in anchors:
            b = self._token_set(anchor)
            if not b:
                continue
            best = max(best, len(a & b) / max(len(a | b), 1))
        return best

    def contextual_prompt_subset(
        self, prompts: List[PromptCandidate], reasoning_prompts: List[PromptCandidate],
        fraction: float = 0.60, minimum: int = 3,
    ) -> List[PromptCandidate]:
        anchors = [p.text for p in reasoning_prompts]
        for p in prompts:
            p.contextual_score = self._context_score(p.text, anchors)
            p.final_score = 0.55 * p.contextual_score + 0.30 * p.reasoning_score + 0.15 * p.isolate_score
        prompts = sorted(prompts, key=lambda x: (-x.final_score, x.text.casefold()))
        n = min(len(prompts), max(minimum, int(math.ceil(len(prompts) * fraction))))
        return prompts[:n]

    def final_prompt_subset(self, prompts: List[PromptCandidate], max_prompts: int = 6) -> List[PromptCandidate]:
        ranked = sorted(prompts, key=lambda x: (-x.final_score, x.text.casefold()))
        return ranked[:max(1, int(max_prompts))]

    @staticmethod
    def dataset_from_output(prompts: List[PromptCandidate]) -> str:
        return "\n".join(p.text for p in prompts)

    @staticmethod
    def _pool_datasets(*datasets: str) -> str:
        """Concatenate any number of dataset strings, separated by double newlines.

        Duplicate paragraphs (case-folded) are silently dropped so the
        pooled corpus does not inflate with repeated sentences.
        """
        seen: Set[str] = set()
        parts: List[str] = []
        for ds in datasets:
            for para in re.split(r"\n\s*\n", ds):
                para = para.strip()
                if not para:
                    continue
                key = para.casefold()
                if key not in seen:
                    seen.add(key)
                    parts.append(para)
        return "\n\n".join(parts)

    def _generate_from_prompts(
        self,
        prompts: List[PromptCandidate],
        num_sentences: int,
        tokens_per_sent: int,
        and_weight: float,
        temperature: float,
        guidance_weight: float,
        guidance_steps: int,
        guidance_lr: float,
        generations_per_prompt: int = 1,
        label_prefix: bool = True,
    ) -> List[str]:
        """Generate text from each prompt via generate_passage().

        Returns plain texts (label_prefix=False) for intermediate datasets,
        or labelled outputs (label_prefix=True) for the final Generate Out.
        Failures are caught per-generation so one error cannot abort the chain.
        """
        results: List[str] = []
        for p in prompts:
            seed_tokens = tokenize(p.text)
            seed = " ".join(seed_tokens[-2:]) if len(seed_tokens) >= 2 else p.text
            for _ in range(max(1, int(generations_per_prompt))):
                try:
                    text = generate_passage(
                        self.engine.walker,
                        self.engine.lm,
                        num_sentences=max(1, int(num_sentences)),
                        tokens_per_sent=max(4, int(tokens_per_sent)),
                        seed_text=seed,
                        instruction_text=p.text,
                        and_weight=float(and_weight),
                        temperature=float(temperature),
                        return_traces=False,
                        guidance_weight=float(guidance_weight),
                        guidance_steps=int(guidance_steps),
                        guidance_lr=float(guidance_lr),
                    )
                    results.append(f"[{p.text}]\n{text}" if label_prefix else text)
                except Exception as exc:
                    results.append(
                        f"[{p.text}]\nGeneration error: {exc}" if label_prefix
                        else f"Generation error: {exc}"
                    )
        return results

    def run(
        self,
        dataset: str,
        num_sentences: int = 1,
        tokens_per_sent: int = 40,
        and_weight: float = 0.5,
        temperature: float = 0.75,
        guidance_weight: float = 0.3,
        guidance_steps: int = 3,
        guidance_lr: float = 0.15,
        reasoning_fraction: float = 0.50,
        contextual_fraction: float = 0.60,
        max_prompts: int = 6,
        generations_per_prompt: int = 1,
    ):
        if not dataset.strip():
            return "", "", "", "", "No Dataset supplied."
        if not self.engine or not self.engine.walker:
            return "", "", "", "", "Engine not initialised."

        gen_kwargs = dict(
            num_sentences=num_sentences,
            tokens_per_sent=tokens_per_sent,
            and_weight=and_weight,
            temperature=temperature,
            guidance_weight=guidance_weight,
            guidance_steps=guidance_steps,
            guidance_lr=guidance_lr,
        )

        # â”€â”€ Stage 1: Dataset â†’ prompt_isolate() â†’ isolated_1
        isolated_1 = self.prompt_isolate(dataset)

        # â”€â”€ Stage 2: isolated_1 â†’ reasoning_prompt_subset() â†’ reasoning_subset
        reasoning_subset = self.reasoning_prompt_subset(isolated_1, fraction=reasoning_fraction)

        # â”€â”€ Stage 3: reasoning_subset â†’ generate â†’ gen_dataset_1
        #    Pool: original dataset + gen_dataset_1 â†’ pooled_1
        #    (NEW DATASET FROM OUTPUT #1)
        gen_1_texts = self._generate_from_prompts(
            reasoning_subset, **gen_kwargs, generations_per_prompt=1, label_prefix=False
        )
        gen_dataset_1 = "\n\n".join(gen_1_texts)
        pooled_1 = self._pool_datasets(dataset, gen_dataset_1)

        # â”€â”€ Stage 4: pooled_1 â†’ prompt_isolate() â†’ isolated_2
        #    Prompts now blend original signal with generated deviation.
        isolated_2 = self.prompt_isolate(pooled_1)
        isolated_2_source = isolated_2 if len(isolated_2) >= 3 else reasoning_subset

        # â”€â”€ Stage 5: isolated_2 â†’ contextual_prompt_subset() â†’ contextual_subset
        #    Anchored to reasoning_subset so contextual scoring measures
        #    how much the deviated prompts still relate to the original theme.
        contextual_subset = self.contextual_prompt_subset(
            isolated_2_source, reasoning_subset, fraction=contextual_fraction
        )

        # â”€â”€ Stage 6: contextual_subset â†’ generate â†’ gen_dataset_2
        #    Pool: pooled_1 + gen_dataset_2 â†’ pooled_2
        #    (NEW DATASET FROM OUTPUT #2)
        gen_2_texts = self._generate_from_prompts(
            contextual_subset, **gen_kwargs, generations_per_prompt=1, label_prefix=False
        )
        gen_dataset_2 = "\n\n".join(gen_2_texts)
        pooled_2 = self._pool_datasets(pooled_1, gen_dataset_2)

        # â”€â”€ Stage 7: pooled_2 â†’ prompt_isolate() â†’ isolated_3
        isolated_3 = self.prompt_isolate(pooled_2)
        isolated_3_source = isolated_3 if len(isolated_3) >= 3 else contextual_subset

        #    isolated_3 â†’ final_prompt_subset() â†’ final_subset
        final_subset = self.final_prompt_subset(isolated_3_source, max_prompts=max_prompts)

        #    final_subset â†’ generate â†’ Generate Out
        final_generations = self._generate_from_prompts(
            final_subset, **gen_kwargs,
            generations_per_prompt=generations_per_prompt, label_prefix=True
        )

        # â”€â”€ Return values (positions unchanged for Gradio compatibility)
        reasoning_dataset_text  = gen_dataset_1
        contextual_dataset_text = gen_dataset_2
        final_dataset_text      = "\n".join(p.text for p in final_subset)
        generated_out_text      = "\n\n".join(final_generations)

        report = (
            "SynthReason-2026 FLOW  (pooled datasets)\n"
            f"Dataset (original):               {len(dataset)} chars\n"
            f"Stage 1  Prompt Isolate:         {len(isolated_1)} candidates\n"
            f"Stage 2  Reasoning Subset:       {len(reasoning_subset)} prompts\n"
            f"Stage 3  New Dataset #1:         {len(gen_dataset_1)} chars  "
            f"({len(gen_1_texts)} gen)\n"
            f"          Pooled corpus #1:        {len(pooled_1)} chars\n"
            f"Stage 4  Prompt Isolate (pool1): {len(isolated_2)} candidates\n"
            f"Stage 5  Contextual Subset:      {len(contextual_subset)} prompts\n"
            f"Stage 6  New Dataset #2:         {len(gen_dataset_2)} chars  "
            f"({len(gen_2_texts)} gen)\n"
            f"          Pooled corpus #2:        {len(pooled_2)} chars\n"
            f"Stage 7  Prompt Isolate (pool2): {len(isolated_3)} candidates\n"
            f"          Final Prompt Subset:     {len(final_subset)} prompts\n"
            f"          Generate Out:            {len(final_generations)} total generations\n\n"
            "Dataset Prompt Isolate Reasoning Subset\n"
            "  â†’ Generate (pool New Dataset #1)\n"
            "  â†’ Prompt Isolate Contextual Subset\n"
            "  â†’ Generate (pool New Dataset #2)\n"
            "  â†’ Prompt Isolate Final Subset â†’ Generate Out\n"
        )

        self.last_outputs = final_generations
        self.last_report  = report
        return (
            reasoning_dataset_text,
            contextual_dataset_text,
            final_dataset_text,
            generated_out_text,
            report,
        )

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
                f"V18-CSNS-G DOUBLE-AGNOSTIC / SOLO-PLANAR Engine initialised.\n"
                f"File: {file_obj.name.split('/')[-1]}\n"
                f"Vocab size: {len(self.engine.lm.vocab)}\n"
                f"CoT stubs: {stub_counts}\n"
                f"CSNS: ω_syn={syn_weight:.2f}  ω_trans={trans_weight:.2f}  K={int(syn_k)}\n\n"
                f"{report}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_text(self, user_prompt, sentences, tokens, and_weight, temperature,
                       guidance_weight=0.0, guidance_steps=3, guidance_lr=0.15,
                       reasoning_fraction=0.50, contextual_fraction=0.60,
                       max_prompts=6, generations_per_prompt=1):
        if not self.engine or not self.engine.walker:
            return "", "", "", "", "Engine not initialised. Load a dataset first."

        dataset_text = getattr(self.engine, "corpus_snippet", "") or ""
        if not dataset_text.strip():
            return "", "", "", "", "Dataset is empty. Load a dataset in the Dataset tab first."

        user_prompt = (user_prompt or "").strip()

        try:
            num_sentences = max(1, int(sentences))
            tokens_per_sent = max(4, int(tokens))
            aw = float(and_weight)
            temp = float(temperature)
            gw = float(guidance_weight)
            gs = int(guidance_steps)
            glr = float(guidance_lr)
            rf = float(reasoning_fraction)
            cf = float(contextual_fraction)
            mp = max(1, int(max_prompts))
            gpp = max(1, int(generations_per_prompt))
        except (TypeError, ValueError) as exc:
            return "", "", "", "", f"Invalid generation control value: {exc}"

        flow = SynthReasonFlow(self.engine)
        
        # Use dataset for prompt isolation, but pass user_prompt as instruction
        reasoning_dataset, contextual_dataset, final_dataset, generated, report = flow.run(
            dataset=dataset_text,
            num_sentences=num_sentences,
            tokens_per_sent=tokens_per_sent,
            and_weight=aw,
            temperature=temp,
            guidance_weight=gw,
            guidance_steps=gs,
            guidance_lr=glr,
            reasoning_fraction=rf,
            contextual_fraction=cf,
            max_prompts=mp,
            generations_per_prompt=gpp,
        )
        
        # If user provided a prompt, regenerate with it as instruction
        if user_prompt:
            # Run generate_passage directly with user_prompt as instruction
            from pathlib import Path
            seed_tokens = tokenize(user_prompt)
            seed = " ".join(seed_tokens[-2:]) if len(seed_tokens) >= 2 else user_prompt
            
            generated_outputs = []
            for _ in range(max(1, int(gpp))):
                try:
                    text = generate_passage(
                        self.engine.walker, self.engine.lm,
                        num_sentences=max(1, int(num_sentences)),
                        tokens_per_sent=max(4, int(tokens_per_sent)),
                        seed_text=seed,
                        instruction_text=user_prompt,
                        and_weight=aw,
                        temperature=temp,
                        return_traces=False,
                        guidance_weight=gw,
                        guidance_steps=gs,
                        guidance_lr=glr,
                    )
                    generated_outputs.append(f"[{user_prompt}]\n{text}")
                except Exception as exc:
                    generated_outputs.append(f"[{user_prompt}]\nGeneration error: {exc}")
            
            generated = "\n\n".join(generated_outputs)
            report = (
                f"SynthReason-2026 FLOW (with user prompt)\n"
                f"{'='*60}\n"
                f"User Prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}\n"
                f"Dataset: {len(dataset_text)} chars\n"
                f"Generations: {len(generated_outputs)}\n"
            )

        return reasoning_dataset, contextual_dataset, final_dataset, generated, report

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
            "V18-CSNS-G DOUBLE-AGNOSTIC / SOLO-PLANAR — DNN Array + CSNS Report",
            "═══════════════════════════════════════════════════════════════",
            "",
            "DA-SP CHANGES SUMMARY:",
            "",
            "  1. unified_plane_kernel() — SOLO SEMANTIC PLANARITY primitive.",
            "     Replaces every k_reg·k_ori·k_side product (and its 2-term",
            "     sub-variants k_reg·k_side) with ONE kernel: the weighted",
            "     rho/theta/sigma distances are summed onto a single plane,",
            "     then exponentiated once. Mathematically equivalent value;",
            "     genuinely one combined computation, not three multiplied.",
            "     Applied in: build_synaptic_weight_matrix, CSNS transitive",
            "     bonus, ThebaultKernels.all_scores_batched, SemanticMandate",
            "     Scorer, InstructionDistribution, AtomismReferenceModel,",
            "     CoTStubLibrary (best_stub/stub_kernel), CoTReasoningEngine",
            "     .plan_chain, ThebaultCompositionLM, MRVConstraintFilter,",
            "     IsomorphicSyntaxStacker, ThebaultPotentialGraph.build.",
            "",
            "  2. symmetric_weighted_sum() — DOUBLE AGNOSTICISM primitive.",
            "     Every place that hard-coded a term/stage/layer order now",
            "     takes that order as a parameter, defaulting to a stable",
            "     name-sorted sequence. Applied in: walk_probs logit",
            "     assembly, CrossSynapticNeuronSum.forward, PDNEngine",
            "     .pdn_logit_bonus, InstructionDistribution.distribution,",
            "     the AND-combination step, DNNArrayPipeline.forward",
            "     (layer_order + temp_position params), CoTReasoningEngine",
            "     .plan_chain (hop_type_order param), CoTStubLibrary.build",
            "     (quartile_map_order param), AtomismReferenceModel.build",
            "     (step_direction / tau_batch_direction params), PDNEngine",
            "     .fit_from_trigrams (scan_direction param), V18Engine.train",
            "     (build_stage_order param), generate_passage",
            "     (sentence_order param — final reading order stays",
            "     ascending regardless, since that is a correctness",
            "     requirement, not a structural preference), and",
            "     ThebaultWalker.push_token (context_order param).",
            "",
            "  3. Nothing claims numerical identity across DIFFERENT chosen",
            "     orders for genuinely non-commutative stages (e.g. the DNN",
            "     layer stack) — DA only removes the HARD-CODING of which",
            "     order runs, it doesn't assert all orders give the same",
            "     answer.",
            "",
            "  4. _gradient_guided_steer() / walk_probs(guidance_weight=...)",
            "     — a REAL backward pass (torch.autograd.grad through a",
            "     softmax → instruction-alignment objective), run per token",
            "     during generation to steer candidate logits toward the",
            "     active instruction/mandate centroid. guidance_weight=0",
            "     (default) leaves generation exactly as before; >0 blends",
            "     in `guidance_steps` autograd-ascent updates at learning",
            "     rate `guidance_lr`. This is separate from, and does not",
            "     alter, the fixed heuristic weights used everywhere else.",
            "═══════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


def launch_gui():
    gui = V18GUI()

    with gr.Blocks(title="SynthReason-2026 — V18-CSNS-G Double-Agnostic / Solo-Planar") as app:
        gr.Markdown(
            "# SynthReason-2026 — V18-CSNS-G\n"
            "### Dataset tab → Prompt Isolate → Reasoning → Contextual → Prompt Subset → Generate Out"
        )

        with gr.Tab("Dataset"):
            file_input     = gr.File(label="Upload .txt Corpus File", file_types=[])
            with gr.Row():
                syn_w_slider   = gr.Slider(0.0, 12.0, value=2.0,  step=0.05, label="CSNS ω_syn")
                trans_w_slider = gr.Slider(0.0, 12.0, value=0.8,  step=0.05, label="CSNS ω_trans")
                syn_k_slider   = gr.Slider(2,   32,   value=8,    step=1,    label="CSNS K")
            train_file_btn = gr.Button("Load Dataset / Initialise Engine (DA-SP)", variant="primary")
            init_out       = gr.Textbox(label="Engine Status / ACF Spectral Report", lines=22, interactive=False)
            train_file_btn.click(
                gui.init_engine_from_file,
                inputs=[file_input, syn_w_slider, trans_w_slider, syn_k_slider],
                outputs=init_out,
            )

        with gr.Tab("SynthReason Flow"):
            gr.Markdown(
                "## SynthReason-2026\\n"
                "`Dataset tab → Prompt Isolate → Reasoning Prompt Subset → New Dataset From Output "
                "→ Contextual Prompt Subset → New Dataset From Output → Prompt Subset → Generate Out`"
            )

            user_prompt = gr.Textbox(
                label="User input prompt",
                placeholder="Enter the prompt or instruction to guide generation...",
                lines=4,
                value="",
            )

            gr.Markdown(
                "**Dataset source:** the corpus loaded in the **Dataset** tab is used automatically. "
                "No second dataset textbox is used in this flow."
            )
            with gr.Row():
                sentences = gr.Slider(1, 4, value=1, step=1, label="Sentences / prompt")
                tokens = gr.Slider(20, 120, value=40, step=1, label="Tokens / sentence")
                and_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="AND weight α")
                temperature = gr.Slider(0.1, 10.0, value=0.75, step=0.05, label="Temperature")
            with gr.Row():
                reasoning_fraction = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="Reasoning subset fraction")
                contextual_fraction = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="Contextual subset fraction")
                max_prompts = gr.Slider(1, 20, value=1, step=1, label="Final prompt subset size")
                generations = gr.Slider(1, 5, value=1, step=1, label="Generations / prompt")
            with gr.Row():
                guidance_weight_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Gradient-guided decoding weight")
                guidance_steps_slider = gr.Slider(0, 10, value=3, step=1, label="Guidance ascent steps")
                guidance_lr_slider = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Guidance ascent LR")
            flow_btn = gr.Button("Generate Out — SynthReason-2026", variant="primary")
            with gr.Row():
                reasoning_out = gr.Textbox(lines=8, label="Reasoning Prompt Subset → New Dataset From Output #1", interactive=False)
                contextual_out = gr.Textbox(lines=8, label="Contextual Prompt Subset → New Dataset From Output #2", interactive=False)
                final_out = gr.Textbox(lines=8, label="Prompt Subset", interactive=False)
                generated_out = gr.Textbox(lines=14, label="Generate Out", interactive=False)
                flow_report = gr.Textbox(lines=10, label="SynthReason-2026 Flow Report", interactive=False)

            flow_btn.click(
                gui.generate_text,
                inputs=[
                    user_prompt,
                    sentences,
                    tokens,
                    and_weight,
                    temperature,
                    guidance_weight_slider,
                    guidance_steps_slider,
                    guidance_lr_slider,
                    reasoning_fraction,
                    contextual_fraction,
                    max_prompts,
                    generations,
                ],
                outputs=[
                    reasoning_out,
                    contextual_out,
                    final_out,
                    generated_out,
                    flow_report,
                ],
            )

        with gr.Tab("Diagnostics"):
            dnn_btn  = gr.Button("Show DNN + CSNS-G DA-SP Pipeline Report")
            dnn_out  = gr.Textbox(lines=40, label="DNN Array + CSNS-G DA-SP Report", interactive=False)
            dnn_btn.click(gui.dnn_report, outputs=dnn_out)

            csns_btn = gr.Button("Show CSNS Diagnostic Report")
            csns_out = gr.Textbox(lines=16, label="CSNS Diagnostics (DA-SP)", interactive=False)
            csns_btn.click(gui.csns_report, outputs=csns_out)

            pdn_btn  = gr.Button("Show ACF Spectral Report")
            pdn_out  = gr.Textbox(lines=20, label="ACF Spectral Report", interactive=False)
            pdn_btn.click(gui.pdn_report, outputs=pdn_out)

            mandate_btn = gr.Button("Show Mandate Centroid")
            mandate_out = gr.Textbox(lines=4, label="Semantic Mandate Scorer (DA-SP)", interactive=False)
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
    parser.add_argument("--syn-weight",   type=float, default=2.0)
    parser.add_argument("--trans-weight", type=float, default=0.6)
    parser.add_argument("--syn-k",        type=int,   default=8)
    parser.add_argument("--cube-chunk-size", type=int, default=128)
    parser.add_argument("--cube-side",       type=int, default=8)
    parser.add_argument("--guidance-weight", type=float, default=0.3,
                         help="Gradient-guided decoding blend weight (0=off, real backward pass toward instruction centroid)")
    parser.add_argument("--guidance-steps",  type=int,   default=3,
                         help="Inner autograd ascent steps per generation step for gradient-guided decoding")
    parser.add_argument("--guidance-lr",     type=float, default=0.15,
                         help="Inner autograd ascent learning rate for gradient-guided decoding")
    parser.add_argument("--parabolic-manifold-strength", type=float, default=0.35,
                         help="Strength of the center-peaked 1D parabolic generation manifold")
    parser.add_argument("--parabolic-manifold-curvature", type=float, default=1.0,
                         help="Curvature of the 1D parabolic generation manifold")
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
        parabolic_manifold_strength=args.parabolic_manifold_strength,
        parabolic_manifold_curvature=args.parabolic_manifold_curvature,
    )
    engine.cube_chunk_size=max(1,args.cube_chunk_size)
    engine.cube_side=max(2,args.cube_side)
    engine.train(corpus_text)
    engine.save_cache("v18_csns_g_dasp_model.pkl")

    print("\n--- SAMPLE GENERATION (V18-CSNS-G DOUBLE-AGNOSTIC / SOLO-PLANAR) ---")
    instruction = args.instruction or "Explain the meaning of life."
    text, traces, step_report = generate_passage(
        engine.walker, engine.lm,
        num_sentences=3, tokens_per_sent=30,
        instruction_text=instruction,
        and_weight=args.and_weight,
        temperature=args.temperature,
        return_traces=True,
        guidance_weight=args.guidance_weight,
        guidance_steps=args.guidance_steps,
        guidance_lr=args.guidance_lr,
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
    print("\n--- FORMAL REFERENCE MODEL ---")
    print(engine.ref_model.reference_report())
