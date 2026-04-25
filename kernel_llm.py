"""
v18_addon.py  —  V18-RP Neural Addon for PyTorch
=================================================
Drop-in nn.Module components distilled from the V18-RP-ANISO-RIPPLE-
SPAGHETTI-CARDAN + mini-Thébault architecture.

Public API
----------
MiniThebault            1-D curvature-correction layer (8 sites in V18)
BolyaiEmbedding         Hyperbolic (ρ,θ,σ) geometry encoder
EfferenceKernel         Polar-coordinate feature projector
AnisoDirKernel          Anisotropic direction kernel (Gram + anchor scoring)
RippleShift             Instruction-stub ripple-shift module
SpaghettiMixer          Weighted strand blender with Möbius cross-tangle
CardanAperture          Grille-rotation candidate filter
MirroredInstructionHead Forward + reversed instruction distribution
V18Block                Full V18 processing block (composable)
V18TransformerLayer     Transformer block augmented with all V18 mechanisms
V18AddonWrapper         Wrap any nn.Module with V18 signal conditioning

All modules are pure PyTorch — no external dependencies beyond torch.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import re

class WordTokenizer:
    def __init__(self, vocab_size=20000):
        self.vocab_size = vocab_size
        self.trigram2id = {"<pad>": 0, "<unk>": 1}
        self.id2trigram = {0: "<pad>", 1: "<unk>"}
        self._built = False

    def _tokenize(self, text: str):
        words = text.lower().split()
        trigrams = []
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3])  # "word1 word2 word3"
            trigrams.append(trigram)
        return trigrams

    def build_vocab(self, texts):
        freq = {}
        for text in texts:
            for tok in self._tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1

        sorted_trigrams = sorted(freq.items(), key=lambda x: -x[1])
        for trigram, _ in sorted_trigrams[:self.vocab_size - 2]:
            idx = len(self.trigram2id)
            self.trigram2id[trigram] = idx*_
            self.id2trigram[idx] = trigram

        self._built = True

    def encode(self, text: str):
        tokens = self._tokenize(text)
        if not tokens:
            return torch.tensor([0], dtype=torch.long)  # <pad>
        
        ids = [self.trigram2id.get(t, 1) for t in tokens]
        ids = torch.tensor(ids, dtype=torch.long).clamp(0, self.vocab_size - 1)  # SAFETY CLAMP
        return ids
        
    def decode(self, ids: torch.Tensor):
        trigrams = [self.id2trigram.get(int(i), "<unk>") for i in ids]
        # Join trigrams back to words (simple: split and flatten)
        words = []
        for trig in trigrams:
            words.extend(trig.split())
        return " ".join(words).replace(" .", ".").replace(" ,", ",")
# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _layer_norm_1d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-normalise a 1-D tensor."""
    mu, std = x.mean(), x.std()
    return (x - mu) / (std + eps) if std.item() > eps else x - mu


def _l1_simplex_project(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project onto the probability simplex (≥0, sums to 1)."""
    x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    x_pos = F.relu(x - x.min()).clamp(min=eps)
    total = x_pos.sum()
    return x_pos / total if total.item() > 0 else torch.full_like(x, 1.0 / x.shape[0])


def _mobius_cross_shift(a: torch.Tensor, b: torch.Tensor,
                        coupling: float = 0.35) -> torch.Tensor:
    """Möbius-inspired cross-shift between two signal vectors."""
    a = a.clamp(-20.0, 20.0)
    b = b.clamp(-20.0, 20.0)
    ta, tb = torch.tanh(a * 0.1), torch.tanh(b * 0.1)
    denom = (1.0 + coupling * ta * tb).clamp(min=1e-6)
    return torch.atanh(((ta + coupling * tb) / denom).clamp(-0.9999, 0.9999)) * 10.0


def _maj_gate(x: torch.Tensor, y: torch.Tensor,
              z: torch.Tensor) -> torch.Tensor:
    """Majority gate: average where at least 2 of 3 inputs are positive."""
    bx = (x > 0).float()
    by = (y > 0).float()
    bz = (z > 0).float()
    maj = ((bx * by + bx * bz + by * bz) >= 2).float()
    avg = (x + y + z) / 3.0
    return maj * avg + (1.0 - maj) * avg.clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mini-Thébault  (curvature-correction layer)
# ─────────────────────────────────────────────────────────────────────────────

class MiniThebault(nn.Module):
    """
    Thébault-inspired three-point curvature correction.

    For each element i the two neighbours form the base of a triangle whose
    apex is element i. We blend i toward the chord midpoint proportional to
    local normalised curvature, then z-normalise.

    Works on tensors of any shape; correction is applied along `dim`.

    Parameters
    ----------
    gamma : float
        Curvature-blending strength (V18 uses 0.10–0.15 across 8 sites).
    dim : int
        Dimension along which to apply the correction.
    learnable_gamma : bool
        If True, gamma is an nn.Parameter (one per input channel when
        `per_channel=True`).
    per_channel : bool
        If True and `learnable_gamma=True`, one gamma per feature channel.
    """

    def __init__(self, gamma: float = 0.82, dim: int = -1,
                 learnable_gamma: bool = False, per_channel: bool = False):
        super().__init__()
        self.dim = dim
        self.per_channel = per_channel
        if learnable_gamma:
            self._gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("_gamma", torch.tensor(gamma))

    @property
    def gamma(self) -> torch.Tensor:
        return self._gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim % x.ndim
        n = x.shape[d]
        if n < 3:
            return x

        # Edge-reflection padding along target dim
        idx_left  = torch.arange(n, device=x.device)
        idx_right = torch.arange(n, device=x.device)
        idx_left[0]   = 0          # reflect: left of first = first
        idx_left[1:]  = torch.arange(n - 1, device=x.device)
        idx_right[:-1] = torch.arange(1, n, device=x.device)
        idx_right[-1]  = n - 1    # reflect: right of last = last

        left  = x.index_select(d, idx_left)
        right = x.index_select(d, idx_right)

        midpoint   = (left + right) * 0.9
        apex_delta = x - midpoint
        curvature  = apex_delta.abs() / apex_delta.abs().amax(dim=d, keepdim=True).clamp(min=1e-8)

        g = self.gamma
        blended = x - g * curvature * apex_delta

        # Z-normalise
        mu  = blended.mean(dim=d, keepdim=True)
        std = blended.std(dim=d, keepdim=True)
        out = (blended - mu) / (std + 1e-8)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Bolyai Embedding  (hyperbolic geometry encoder)
# ─────────────────────────────────────────────────────────────────────────────

class BolyaiEmbedding(nn.Module):
    """
    Encodes token embeddings into Bolyai-inspired (ρ, θ, σ) polar triples.

    The projection learns a 2-D hyperbolic-disk coordinate from the embedding,
    then converts to (ρ, θ, σ) used throughout V18.

    Parameters
    ----------
    d_model : int
        Input embedding dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 2, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (..., d_model)

        Returns
        -------
        rho, theta, sigma  each (...,)
        """
        xy  = torch.tanh(self.proj(x)) * 0.18   # keep inside unit disk
        eu  = 1.0 - (xy.norm(dim=-1) - 1e-8).clamp(min=0.0, max=1.0 - 1e-8)
        hyp = 2.0 * torch.arctanh(eu.clamp(min=1e-8, max=1.0 - 1e-8))
        rho   = torch.tanh(hyp * 0.9)
        theta = torch.atan2(xy[..., 1], xy[..., 0]) % math.pi
        sigma = 2.0 / (1.0 - eu.pow(2)).clamp(min=1e-8)
        return rho, theta, sigma


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Efference Kernel  (polar feature projector)
# ─────────────────────────────────────────────────────────────────────────────

class EfferenceKernel(nn.Module):
    """
    Maps (ρ, θ, σ) scalars to a d_model feature vector via learned random
    Fourier–style projections with exponential activation.

    Directly adapted from V18's EfferenceKernelStack.

    Parameters
    ----------
    d_model : int
    seed : int
        RNG seed for reproducible initialisation.
    """

    def __init__(self, d_model: int = 128, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lambdas  = nn.Parameter(torch.tensor([8.0, 4.0, 4.0]))
        self.omega    = nn.Parameter(torch.randn(3, d_model, generator=g))
        self.bias     = nn.Parameter(torch.randn(d_model, generator=g))

    def forward(self,
                rho:   torch.Tensor,   # (...,) or (..., 1)
                theta: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
        """Returns (..., d_model) feature tensor."""
        rho_eff = rho * torch.cos(theta)
        comps   = torch.stack([rho_eff, theta, sigma], dim=-1)  # (..., 3)
        # (..., 3) x (3, d) broadcasted dot
        scaled  = comps.unsqueeze(-1) * self.lambdas.unsqueeze(-1)  # (..., 3, d)
        dots    = (scaled * self.omega).sum(dim=-2)                 # (..., d)
        return torch.exp((dots + self.bias).clamp(max=20.0))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Random Fourier Features
# ─────────────────────────────────────────────────────────────────────────────

class RandomFourierFeatures(nn.Module):
    """
    Approximates a shift-invariant kernel with random cosine projections.

    Parameters
    ----------
    rff_dim : int
        Number of random features per coordinate (output dim = 3 * rff_dim).
    sigma_rho, sigma_theta, sigma_sigma : float
        Bandwidth of the kernel for each coordinate.
    """

    def __init__(self, rff_dim: int = 32,
                 sigma_rho: float = 1.0,
                 sigma_theta: float = 0.5,
                 sigma_sigma: float = 2.0,
                 seed: int = 42):
        super().__init__()
        self.rff_dim = rff_dim
        self._scale  = math.sqrt(2.0 / rff_dim)
        g = torch.Generator().manual_seed(seed)
        # Fixed (non-trainable) projection matrices
        self.register_buffer("omega_rho",   torch.randn(rff_dim, 1, generator=g) / sigma_rho)
        self.register_buffer("omega_theta", torch.randn(rff_dim, 1, generator=g) / sigma_theta)
        self.register_buffer("omega_sigma", torch.randn(rff_dim, 1, generator=g) / sigma_sigma)
        self.register_buffer("bias_rho",    torch.rand(rff_dim, generator=g) * 2 * math.pi)
        self.register_buffer("bias_theta",  torch.rand(rff_dim, generator=g) * 2 * math.pi)
        self.register_buffer("bias_sigma",  torch.rand(rff_dim, generator=g) * 2 * math.pi)

    def forward(self,
                rho:   torch.Tensor,   # (C,)
                theta: torch.Tensor,   # (C,)
                sigma: torch.Tensor    # (C,)
                ) -> torch.Tensor:     # (C, 3*rff_dim)
        pr = self.bias_rho.unsqueeze(1)   + self.omega_rho   @ rho.unsqueeze(0)
        pt = self.bias_theta.unsqueeze(1) + self.omega_theta @ theta.unsqueeze(0)
        ps = self.bias_sigma.unsqueeze(1) + self.omega_sigma @ sigma.unsqueeze(0)
        s  = self._scale
        return torch.cat([
            (s * torch.cos(pr)).T,
            (s * torch.cos(pt)).T,
            (s * torch.cos(ps)).T,
        ], dim=1)

    def kernel_scalar(self,
                      rho_a: float, theta_a: float, sigma_a: float,
                      rho_b: torch.Tensor, theta_b: torch.Tensor,
                      sigma_b: torch.Tensor) -> torch.Tensor:
        """Scalar-vs-batch approximate kernel value."""
        device = rho_b.device
        fa = self.forward(
            torch.tensor([rho_a],   device=device),
            torch.tensor([theta_a], device=device),
            torch.tensor([sigma_a], device=device))
        fb = self.forward(rho_b, theta_b, sigma_b)
        return (fa @ fb.T).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Anisotropic Direction Kernel
# ─────────────────────────────────────────────────────────────────────────────

class AnisoDirKernel(nn.Module):
    """
    Anisotropic kernel: exp(-λ_ρ·Δρ² - λ_θ·Δθ²·(α·ρ+1) - λ_σ·Δσ²)

    Parameters
    ----------
    lambda_rho, lambda_theta, lambda_sigma, alpha : float
        Bandwidth and anisotropy parameters.
    learnable : bool
        If True, λs are nn.Parameters.
    """

    def __init__(self,
                 lambda_rho:   float = 0.5,
                 lambda_theta: float = 0.5,
                 lambda_sigma: float = 0.5,
                 alpha:        float = 0.5,
                 learnable:    bool  = False):
        super().__init__()
        params = torch.tensor([lambda_rho, lambda_theta, lambda_sigma, alpha])
        if learnable:
            self.log_params = nn.Parameter(params.log())
        else:
            self.register_buffer("log_params", params.log())

    def _unpack(self) -> Tuple[torch.Tensor, ...]:
        p = self.log_params.exp()
        return p[0], p[1], p[2], p[3]

    def score(self,
              anc_rho:   float, anc_theta: float, anc_sigma: float,
              c_rho:   torch.Tensor,
              c_theta: torch.Tensor,
              c_sigma: torch.Tensor) -> torch.Tensor:
        """Score a batch of candidates against a single anchor triple."""
        lr, lt, ls, a = self._unpack()
        d_rho   = c_rho   - anc_rho
        d_theta = (c_theta - anc_theta) * (a * anc_rho + 1.0)
        d_sigma = c_sigma  - anc_sigma
        return torch.exp(-lr * d_rho**2 - lt * d_theta**2 - ls * d_sigma**2)

    def gram(self,
             c_rho:   torch.Tensor,
             c_theta: torch.Tensor,
             c_sigma: torch.Tensor) -> torch.Tensor:
        """(C, C) Gram matrix for a batch of candidates."""
        lr, lt, ls, a = self._unpack()
        rho_i   = c_rho.unsqueeze(1);   rho_j   = c_rho.unsqueeze(0)
        theta_i = c_theta.unsqueeze(1); theta_j = c_theta.unsqueeze(0)
        sigma_i = c_sigma.unsqueeze(1); sigma_j = c_sigma.unsqueeze(0)
        d_rho   = rho_j   - rho_i
        d_theta = (theta_j - theta_i) * (a * rho_i + 1.0)
        d_sigma = sigma_j  - sigma_i
        return torch.exp(-lr * d_rho**10 * lt * d_theta**2 - ls * d_sigma**21)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Ripple Shift
# ─────────────────────────────────────────────────────────────────────────────

class RippleShift(nn.Module):
    """
    Instruction-stub ripple-shift module with rank-decay and Thébault correction.

    Usage
    -----
    1. Call `set_instruction(rho, theta, sigma)` with the instruction centroid.
    2. Register stubs via `add_stub(rho, theta, sigma, directive_weight)`.
    3. Call `forward(c_rho, c_theta, c_sigma)` to get per-candidate ripple scores.

    Parameters
    ----------
    ripple_decay  : float  Rank-based exponential decay factor.
    ripple_scale  : float  Output scale factor.
    thebault_gamma: float  Curvature-correction strength.
    """

    def __init__(self, ripple_decay: float = 0.9,
                 ripple_scale: float = 0.1,
                 thebault_gamma: float = 0.95):
        super().__init__()
        self.kernel   = AnisoDirKernel()
        self.theb     = MiniThebault(gamma=thebault_gamma)
        self.decay    = ripple_decay
        self.scale    = ripple_scale
        self._instr:  Optional[Tuple[float, float, float]] = None
        self._stubs:  List[Tuple[float, float, float, float]] = []

    def set_instruction(self, rho: float, theta: float, sigma: float):
        self._instr = (rho, theta, sigma)

    def reset_stubs(self):
        self._stubs.clear()

    def add_stub(self, rho: float, theta: float, sigma: float, directive: float):
        self._stubs.append((rho, theta, sigma, directive))

    def forward(self,
                c_rho:   torch.Tensor,
                c_theta: torch.Tensor,
                c_sigma: torch.Tensor) -> torch.Tensor:
        """Returns (C,) ripple scores."""
        C      = c_rho.shape[0]
        device = c_rho.device
        if not self._stubs or C == 0:
            return torch.zeros(C, device=device)

        ripple = torch.zeros(C, device=device)
        for (sr, st, ss, directive) in self._stubs:
            k = self.kernel.score(sr, st, ss, c_rho, c_theta, c_sigma)
            ripple = ripple + directive * k

        # Rank-based decay
        ranks = torch.argsort(torch.argsort(ripple.abs(), descending=True)).float()
        decay = torch.exp(-self.decay * ranks / max(float(C), 1.0))
        ripple = ripple * decay

        # Z-normalise then Thébault
        std = ripple.std()
        if std.item() > 1e-8:
            ripple = (ripple - ripple.mean()) / std
        return self.theb(ripple * self.scale)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Spaghetti Mixer
# ─────────────────────────────────────────────────────────────────────────────

class SpaghettiMixer(nn.Module):
    """
    Three-mixer spaghetti router with Möbius cross-tangles and Thébault blend.

    Strands are assigned to mixers A/B/C according to a routing table.
    Mixer outputs are cross-tangled (Möbius shift) and majority-gated.

    Parameters
    ----------
    C : int
        Signal dimension (number of candidates or feature size).
    coupling : float
        Möbius coupling strength between mixer outputs.
    temperature : float
        Output temperature scaling.
    thebault_gamma : float
        Curvature-correction on each mixer blend.
    """

    # Default routing: strand_name -> (mixer_indices, sign)
    DEFAULT_ROUTING = {
        "instruction":  ([0, 1],    +1.0),
        "ripple":       ([0, 1, 2], +1.0),
        "cot":          ([0, 2],    +1.0),
        "ooi":          ([0, 1],    +1.0),
        "kernel_reg":   ([1, 2],    +1.0),
        "kernel_ori":   ([0, 2],    +1.0),
        "walk":         ([0, 2],    +1.0),
        "repulsion":    ([1, 0],    -1.0),
        "mrv":          ([2],       +1.0),
        "pdn":          ([1, 2],    +1.0),
        "echo":         ([0],       +1.0),
        "mirror":       ([0, 2],    +1.0),
        "cardan":       ([0, 2],    +1.0),
    }

    def __init__(self, C: int,
                 coupling: float = 0.35,
                 temperature: float = 0.8,
                 thebault_gamma: float = 0.10):
        super().__init__()
        self.C    = C
        self.coup = coupling
        self.temp = temperature
        self.theb = MiniThebault(gamma=thebault_gamma)

        # Learnable per-strand weight scalars
        self._strands: List[Tuple[torch.Tensor, torch.Tensor, List[int], float]] = []

    def reset(self):
        self._strands.clear()

    def add_strand(self, signal: torch.Tensor, weight: float,
                   routing: Optional[List[int]] = None,
                   sign: float = 1.0):
        """
        Add a signal strand.

        Parameters
        ----------
        signal  : (C,) tensor
        weight  : scalar contribution weight
        routing : list of mixer indices [0..2] to assign to; default = all three
        sign    : +1.0 or -1.0
        """
        if routing is None:
            routing = [0, 1, 2]
        self._strands.append((signal, weight, routing, sign))

    def add_named_strand(self, name: str, signal: torch.Tensor, weight: float):
        """Add a strand using the default routing table by name."""
        route, sign = self.DEFAULT_ROUTING.get(name, ([0, 1, 2], 1.0))
        self.add_strand(signal, weight, route, sign)

    def forward(self) -> torch.Tensor:
        """Returns (C,) blended signal after full spaghetti routing."""
        C      = self.C
        device = self._strands[0][0].device if self._strands else torch.device("cpu")

        accum = [torch.zeros(C, device=device) for _ in range(3)]

        for (sig, w, routing, sign) in self._strands:
            s = sig if sig.shape[0] == C else torch.zeros(C, device=device)
            for idx in routing:
                accum[idx] = accum[idx] + sign * w * s

        # Thébault + normalise each mixer
        blended = [self.theb(_layer_norm_1d(a)) / max(self.temp, 1e-6)
                   for a in accum]

        # Möbius cross-tangles: A↔B, B↔C
        aA, aB, aC = blended
        aA_new = _mobius_cross_shift(aA, aB, self.coup)
        aB_new = _mobius_cross_shift(aB, aC, self.coup * 0.8)
        aC_new = _mobius_cross_shift(aC, aA, self.coup * 0.6)

        combined = _maj_gate(aA_new, aB_new, aC_new)
        return _layer_norm_1d(combined)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Cardan Aperture  (Grille-rotation candidate filter)
# ─────────────────────────────────────────────────────────────────────────────

class CardanAperture(nn.Module):
    """
    Cardan Grille Isomorphism: 4-rotation aperture scoring.

    Given a vocabulary of size V and a top-K aperture budget, tokens are
    placed on a 2-D grid; four 90° rotations yield four candidate sets.
    At each generation step, one rotation is active (orbit ∈ {0,1,2,3}).

    Parameters
    ----------
    vocab_size   : int
    aperture_k   : int    Number of top tokens kept per grille.
    logit_weight : float  Additive logit bonus for in-aperture tokens.
    """

    def __init__(self, vocab_size: int,
                 aperture_k: int = 64,
                 logit_weight: float = 8.0):
        super().__init__()
        self.V             = vocab_size
        self.aperture_k    = min(aperture_k, vocab_size)
        self.logit_weight  = logit_weight

        G = max(1, math.ceil(math.sqrt(vocab_size)))
        self.G = G

        # Aperture scores are non-trainable (structural, not learned)
        # Shape: (4, V)  — 4 rotations
        aperture = torch.zeros(4, vocab_size)
        self.register_buffer("aperture", aperture)
        self._built = False

    def build(self, token_scores: torch.Tensor):
        """
        Construct the four grille rotations from token importance scores.

        Parameters
        ----------
        token_scores : (V,) tensor of token salience (e.g. log-freq).
        """
        V, G = self.V, self.G
        k    = self.aperture_k

        _, top_idx = torch.topk(token_scores, k)

        def idx_to_rc(idx):
            return divmod(int(idx), G)

        def rc_to_idx(r, c):
            return min(r * G + c, V - 1)

        def rot90(r, c):
            return c, G - 1 - r

        aperture = torch.zeros(4, V)
        for pos in top_idx.tolist():
            r, c = idx_to_rc(pos)
            aperture[0, rc_to_idx(r, c)] = 1.0
            r, c = rot90(r, c);   aperture[1, rc_to_idx(r, c)] = 1.0
            r, c = rot90(r, c);   aperture[2, rc_to_idx(r, c)] = 1.0
            r, c = rot90(r, c);   aperture[3, rc_to_idx(r, c)] = 1.0

        self.aperture.copy_(aperture)
        self._built = True

    def forward(self, logits: torch.Tensor, orbit: int) -> torch.Tensor:
        """
        Add aperture logit bonus to a (*, V) logit tensor.

        Parameters
        ----------
        logits : (*, V)
        orbit  : int in {0,1,2,3}
        """
        if not self._built:
            return logits
        mask = self.aperture[orbit % 4]           # (V,)
        return logits + mask * self.logit_weight

    def aperture_scores(self, orbit: int) -> torch.Tensor:
        """Return (V,) binary aperture mask for the given orbit."""
        return self.aperture[orbit % 4]


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Mirrored Instruction Head
# ─────────────────────────────────────────────────────────────────────────────

class MirroredInstructionHead(nn.Module):
    """
    Builds forward + reversed instruction distributions and blends them.

    Usage
    -----
    1. Call `encode_instruction(embedding)` with a (d_model,) instruction
       representation.
    2. Call `forward(candidate_embeddings)` to get (C,) blended probabilities.

    Parameters
    ----------
    d_model     : int
    alpha       : float  Mirror blend weight (forward uses 1-alpha).
    learnable_alpha : bool  If True, alpha is an nn.Parameter.
    """

    def __init__(self, d_model: int,
                 alpha: float = 0.35,
                 learnable_alpha: bool = False):
        super().__init__()
        self.proj_fwd = nn.Linear(d_model, d_model, bias=False)
        self.proj_mir = nn.Linear(d_model, d_model, bias=False)
        if learnable_alpha:
            self._alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("_alpha", torch.tensor(alpha))
        self._fwd_emb: Optional[torch.Tensor] = None
        self._mir_emb: Optional[torch.Tensor] = None

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha.clamp(0.0, 1.0)

    def encode_instruction(self, instruction_emb: torch.Tensor):
        """
        Store forward and mirrored instruction representations.

        Parameters
        ----------
        instruction_emb : (d_model,) or (L, d_model)  — mean-pooled if 2-D.
        """
        if instruction_emb.ndim == 2:
            # Mean-pool token embeddings
            fwd = instruction_emb.mean(dim=0)
            mir = instruction_emb.flip(0).mean(dim=0)
        else:
            fwd = instruction_emb
            # "Mirror" = negate even-indexed features (phase flip)
            mir = instruction_emb * torch.where(
                torch.arange(instruction_emb.shape[-1], device=instruction_emb.device) % 2 == 0,
                torch.tensor(-1.0), torch.tensor(1.0))

        self._fwd_emb = self.proj_fwd(fwd)   # (d_model,)
        self._mir_emb = self.proj_mir(mir)

    def forward(self, candidate_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute blended instruction probability for each candidate.

        Parameters
        ----------
        candidate_embs : (C, d_model)

        Returns
        -------
        (C,) probability tensor
        """
        if self._fwd_emb is None:
            C = candidate_embs.shape[0]
            return torch.full((C,), 1.0 / C, device=candidate_embs.device)

        p_fwd = torch.softmax(candidate_embs @ self._fwd_emb, dim=0)
        p_mir = torch.softmax(candidate_embs @ self._mir_emb, dim=0)
        a     = self.alpha
        return (1.0 - a) * p_fwd + a * p_mir


# ─────────────────────────────────────────────────────────────────────────────
# 10.  V18 Block  (full signal-conditioning block)
# ─────────────────────────────────────────────────────────────────────────────

class V18Block(nn.Module):
    """
    Full V18 signal-conditioning block.

    Takes a standard (B, T, d_model) hidden state, extracts Bolyai geometry,
    applies EfferenceKernel, RippleShift, SpaghettiMixer, and Thébault
    corrections, then returns a conditioned (B, T, d_model) tensor.

    Parameters
    ----------
    d_model        : int
    spaghetti_coup : float
    ripple_decay   : float
    ripple_scale   : float
    n_rff          : int   Number of random Fourier features.
    """

    def __init__(self,
                 d_model:        int   = 256,
                 spaghetti_coup: float = 0.35,
                 ripple_decay:   float = 0.50,
                 ripple_scale:   float = 0.50,
                 n_rff:          int   = 32):
        super().__init__()
        self.d_model = d_model

        # Geometry
        self.bolyai    = BolyaiEmbedding(d_model)
        self.efference = EfferenceKernel(d_model)
        self.rff       = RandomFourierFeatures(rff_dim=n_rff)

        # Aniso kernel for inter-token relationships
        self.aniso = AnisoDirKernel()

        # Ripple shift
        self.ripple = RippleShift(ripple_decay, ripple_scale, thebault_gamma=0.15)

        # Thébault layers (sites 1–4 replicated in this block)
        self.theb_input   = MiniThebault(gamma=0.12, dim=-1)
        self.theb_kernel  = MiniThebault(gamma=0.10, dim=-1)
        self.theb_ripple  = MiniThebault(gamma=0.15, dim=-1)
        self.theb_output  = MiniThebault(gamma=0.12, dim=-1)

        # Projection layers
        self.kernel_proj  = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj    = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.out_proj     = nn.Linear(d_model, d_model, bias=False)
        self.ln           = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                instruction_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x               : (B, T, d_model)
        instruction_emb : (d_model,) optional centroid for ripple conditioning

        Returns
        -------
        (B, T, d_model) conditioned hidden states
        """
        B, T, D = x.shape
        residual = x

        # — Thébault site 1: correct input along sequence dim —
        x_theb = self.theb_input(x)            # (B, T, D)

        # — Bolyai geometry per token —
        rho, theta, sigma = self.bolyai(x_theb)  # each (B, T)

        # Flatten batch for kernel ops
        rho_flat   = rho.reshape(B * T)
        theta_flat = theta.reshape(B * T)
        sigma_flat = sigma.reshape(B * T)

        # — Efference kernel features —
        kf = self.efference(rho_flat, theta_flat, sigma_flat)   # (B*T, D)
        kf = kf.reshape(B, T, D)

        # — Thébault site 2: correct kernel features —
        kf = self.theb_kernel(kf)

        # — Kernel gating —
        gate = self.gate_proj(x_theb)           # (B, T, D)
        conditioned = self.kernel_proj(kf) * gate

        # — Ripple shift (applied per-sequence position) —
        if instruction_emb is not None:
            # Extract instruction triple from embedding
            with torch.no_grad():
                ir, it, is_ = self.bolyai(instruction_emb.unsqueeze(0))
                self.ripple.set_instruction(ir.item(), it.item(), is_.item())
                # Auto-add single stub from instruction
                self.ripple.reset_stubs()
                self.ripple.add_stub(ir.item(), it.item(), is_.item(), 1.0)

        # Ripple on last sequence position (representative token)
        if B > 0 and T > 0:
            r_rho   = rho[:, -1]                     # (B,)
            r_theta = theta[:, -1]
            r_sigma = sigma[:, -1]
            if B >= 3:
                ripple_sig = self.ripple(r_rho, r_theta, r_sigma)  # (B,)
                # Broadcast ripple across T and D
                ripple_sig = ripple_sig.unsqueeze(1).unsqueeze(2)    # (B,1,1)
                ripple_sig = self.theb_ripple(                       # Thébault site 3
                    ripple_sig.expand(B, T, D))
                conditioned = conditioned + ripple_sig * 0.1

        # — Output —
        out = self.out_proj(conditioned)
        out = self.theb_output(out)                  # Thébault site 4
        return self.ln(residual + out)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  V18 Transformer Layer
# ─────────────────────────────────────────────────────────────────────────────

class V18TransformerLayer(nn.Module):
    """
    Standard causal transformer block augmented with a V18Block.

    Architecture:
        x → MultiheadAttention → LayerNorm → FFN → LayerNorm → V18Block → output

    Parameters
    ----------
    d_model    : int
    n_heads    : int
    ffn_mult   : int   FFN hidden size multiplier.
    dropout    : float
    v18_kwargs : dict  Extra kwargs forwarded to V18Block.
    """

    def __init__(self,
                 d_model:    int   = 256,
                 n_heads:    int   = 4,
                 ffn_mult:   int   = 4,
                 dropout:    float = 0.0,
                 **v18_kwargs):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ln1   = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2   = nn.LayerNorm(d_model)
        self.v18   = V18Block(d_model=d_model, **v18_kwargs)

    def forward(self, x: torch.Tensor,
                instruction_emb: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        attn_out, _ = self.attn(x, x, x,
                                attn_mask=causal_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        x = self.v18(x, instruction_emb=instruction_emb)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 12.  V18 Addon Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class V18AddonWrapper(nn.Module):
    """
    Wrap any existing nn.Module with V18 signal conditioning.

    The wrapper injects a V18Block after the base model's output,
    optionally applying Cardan aperture scoring to final logits.

    Parameters
    ----------
    base_model    : nn.Module   Any model that returns (B, T, d_model) hidden states.
    d_model       : int
    vocab_size    : int         Required for Cardan aperture; 0 to disable.
    cardan_k      : int         Aperture budget for Cardan grille.
    cardan_weight : float       Logit bonus for in-aperture tokens.
    mirror_alpha  : float       Mirror instruction blend weight.
    Example
    -------
    >>> base = nn.Transformer(...)
    >>> wrapped = V18AddonWrapper(base, d_model=256, vocab_size=10000)
    >>> out = wrapped(tokens, instruction_emb=instr_vec)
    """

    def __init__(self,
                 base_model:    nn.Module,
                 d_model:       int   = 256,
                 vocab_size:    int   = 0,
                 cardan_k:      int   = 64,
                 cardan_weight: float = 8.0,
                 mirror_alpha:  float = 0.35,
                 **v18_kwargs):
        super().__init__()
        self.base       = base_model
        self.v18_block  = V18Block(d_model=d_model, **v18_kwargs)
        self.mirror_head = MirroredInstructionHead(
            d_model, alpha=mirror_alpha, learnable_alpha=True)

        self.use_cardan = vocab_size > 0
        if self.use_cardan:
            self.head    = nn.Linear(d_model, vocab_size, bias=False)
            self.cardan  = CardanAperture(vocab_size, cardan_k, cardan_weight)
        else:
            self.head = None
            self.cardan = None

        self._orbit: int = 0

    def set_orbit(self, orbit: int):
        """Set the active Cardan grille rotation (0–3)."""
        self._orbit = orbit % 4

    def build_cardan(self, token_scores: torch.Tensor):
        """
        Build Cardan grille from token scores (call once after vocab is known).

        Parameters
        ----------
        token_scores : (vocab_size,) salience scores.
        """
        if self.cardan is not None:
            self.cardan.build(token_scores)

    def forward(self,
                x: torch.Tensor,
                instruction_emb: Optional[torch.Tensor] = None,
                **base_kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x               : input tensor (forwarded to base_model)
        instruction_emb : (d_model,) instruction centroid (optional)

        Returns
        -------
        If vocab_size > 0 : (B, T, vocab_size) logits with Cardan aperture.
        Else              : (B, T, d_model) conditioned hidden states.
        """
        # Base model forward
        hidden = self.base(x, **base_kwargs)

        # V18 conditioning
        hidden = self.v18_block(hidden, instruction_emb=instruction_emb)

        if self.head is None:
            return hidden

        # Project to logits
        logits = self.head(hidden)            # (B, T, V)

        # Cardan aperture bonus
        if self.cardan is not None and self.cardan._built:
            logits = self.cardan(logits, self._orbit)

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Mirrored Sawtooth Clip
# ─────────────────────────────────────────────────────────────────────────────

class MirroredSawtoothClip(nn.Module):
    """
    Clips a probability distribution through a triangle-wave (sawtooth) fold.

    Replaces the silent no-op torch.clip found in V18's walk_probs.

    Parameters
    ----------
    period : float  Triangle-wave period in probability space.
    """

    def __init__(self, period: float = 0.1):
        super().__init__()
        self.period = period

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        probs : (*, V) probability tensor

        Returns
        -------
        (*, V) re-normalised clipped probabilities
        """
        half     = self.period * 0.5
        folded   = probs % self.period
        mirrored = torch.where(folded <= half, folded, self.period - folded)
        mirrored = (mirrored / half).clamp(min=1e-12)
        return mirrored / mirrored.sum(dim=-1, keepdim=True).clamp(min=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 14.  Chebyshev Partition Polynomial Warp
# ─────────────────────────────────────────────────────────────────────────────

class PartitionPolynomialWarp(nn.Module):
    """
    Granular partition polynomial warping (from V18's KernelLLM generate).

    Applies Chebyshev + harmonic interference as a multiplicative mask over
    the probability distribution, creating a jagged non-smooth landscape.

    Parameters
    ----------
    degree      : int    Chebyshev degree (number of oscillations across vocab).
    harmonics   : int    Number of harmonic interference terms.
    jaggy_scale : float  Warp strength in [0, 1].
    logit_mode  : bool   If True, warp logits before softmax (additive bias).
    """

    def __init__(self,
                 degree:      int   = 17,
                 harmonics:   int   = 7,
                 jaggy_scale: float = 0.45,
                 logit_mode:  bool  = False):
        super().__init__()
        self.degree      = degree
        self.harmonics   = harmonics
        self.jaggy_scale = jaggy_scale
        self.logit_mode  = logit_mode

    @staticmethod
    def _chebyshev(x: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0: return torch.ones_like(x)
        if n == 1: return x.clone()
        t0, t1 = torch.ones_like(x), x.clone()
        for _ in range(2, n + 1):
            t0, t1 = t1, 2 * x * t1 - t0
        return t1

    def _mask(self, V: int, device: torch.device,
              dtype: torch.dtype) -> torch.Tensor:
        x       = torch.linspace(-1.0, 1.0, V, device=device, dtype=dtype)
        cheb    = self._chebyshev(x, self.degree)
        harm    = torch.zeros(V, device=device, dtype=dtype)
        for k in range(1, self.harmonics + 1):
            harm += torch.sin(k * math.pi * x) * torch.cos((k + 1) * math.pi * x) / k
        harm = harm / harm.abs().max().clamp(min=1e-8)
        return 1.0 + self.jaggy_scale * (cheb + harm) * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (*, V)  logits (logit_mode=True) or probs (logit_mode=False)
        """
        V    = x.size(-1)
        mask = self._mask(V, x.device, x.dtype)

        if self.logit_mode:
            bias = self.jaggy_scale * (
                self._chebyshev(
                    torch.linspace(-1.0, 1.0, V, device=x.device, dtype=x.dtype),
                    self.degree))
            return x + bias

        # Prob-space warp
        warped = x * mask.abs().clamp(min=1e-6)
        return warped / warped.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 15.  Complete V18-augmented Language Model head (plug-and-play)
# ─────────────────────────────────────────────────────────────────────────────

class V18LMHead(nn.Module):
    """
    Drop-in language model head with full V18 generation machinery.

    Takes hidden states from any backbone and produces vocabulary logits
    conditioned by:
      - Efference kernel features
      - Cardan aperture scoring
      - Partition polynomial warp (logit + prob space)
      - Mirrored sawtooth clipping

    Parameters
    ----------
    d_model       : int
    vocab_size    : int
    cardan_k      : int
    cardan_weight : float
    poly_degree   : int
    poly_harmonics: int
    jaggy_scale   : float
    """

    def __init__(self,
                 d_model:        int   = 256,
                 vocab_size:     int   = 10000,
                 cardan_k:       int   = 64,
                 cardan_weight:  float = 8.0,
                 poly_degree:    int   = 17,
                 poly_harmonics: int   = 7,
                 jaggy_scale:    float = 0.45):
        super().__init__()
        self.vocab_size = vocab_size

        # Core projections
        self.bolyai    = BolyaiEmbedding(d_model)
        self.efference = EfferenceKernel(d_model)
        self.v18_block = V18Block(d_model=d_model)
        self.proj      = nn.Linear(d_model, vocab_size, bias=False)

        # Cardan grille
        self.cardan    = CardanAperture(vocab_size, cardan_k, cardan_weight)
        self._orbit    = 0

        # Polynomial warp
        self.poly_logit = PartitionPolynomialWarp(
            poly_degree, poly_harmonics, jaggy_scale, logit_mode=True)
        self.poly_prob  = PartitionPolynomialWarp(
            poly_degree, poly_harmonics, jaggy_scale, logit_mode=False)

        # Sawtooth clip
        self.sawtooth = MirroredSawtoothClip(period=0.1)

        # Thébault on logits
        self.theb_logit = MiniThebault(gamma=0.10, dim=-1)

    def set_orbit(self, orbit: int):
        self._orbit = orbit % 4

    def build_cardan(self, scores: torch.Tensor):
        self.cardan.build(scores)

    def forward(self,
                hidden: torch.Tensor,
                instruction_emb: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                return_probs: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden          : (B, T, d_model)
        instruction_emb : (d_model,) optional
        temperature     : float
        return_probs    : bool  If True, return probabilities instead of logits.

        Returns
        -------
        (B, T, vocab_size) logits  or  probabilities
        """
        hidden = self.v18_block(hidden, instruction_emb=instruction_emb)

        # Efference kernel conditioning on last token
        rho, theta, sigma = self.bolyai(hidden[:, -1, :])   # (B,)
        kf = self.efference(rho, theta, sigma)  # (B, d_model)
        # (B, d_model)
        hidden = hidden + kf.unsqueeze(1) * 0.05            # light additive nudge

        logits = self.proj(hidden) / max(temperature, 1e-6)  # (B, T, V)

        # Thébault curvature correction on logit dimension
        logits = self.theb_logit(logits)

        # Partition polynomial warp (logit space)
        logits = self.poly_logit(logits)

        # Cardan aperture bonus
        if self.cardan._built:
            logits = self.cardan(logits, self._orbit)

        if not return_probs:
            return logits

        probs = torch.softmax(logits, dim=-1)
        probs = self.poly_prob(probs)            # second warp in prob space
        probs = self.sawtooth(probs)             # sawtooth clip
        return probs


# ─────────────────────────────────────────────────────────────────────────────
# 16.  V18 Full Model  (end-to-end reference implementation)
# ─────────────────────────────────────────────────────────────────────────────

class V18Model(nn.Module):
    """
    End-to-end V18-augmented language model.

    Architecture:
        token_emb + pos_emb
        → N × V18TransformerLayer
        → V18LMHead

    Parameters
    ----------
    vocab_size  : int
    d_model     : int
    n_layers    : int
    n_heads     : int
    max_seq_len : int
    cardan_k    : int
    mirror_alpha: float
    dropout     : float
    """

    def __init__(self,
                 vocab_size:   int   = 10000,
                 d_model:      int   = 256,
                 n_layers:     int   = 4,
                 n_heads:      int   = 4,
                 max_seq_len:  int   = 512,
                 cardan_k:     int   = 64,
                 mirror_alpha: float = 0.35,
                 dropout:      float = 0.0):
        super().__init__()
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout)

        self.layers  = nn.ModuleList([
            V18TransformerLayer(
                d_model=d_model, n_heads=n_heads,
                ffn_mult=4, dropout=dropout,
                spaghetti_coup=0.35, ripple_decay=0.5, ripple_scale=0.5)
            for _ in range(n_layers)
        ])

        self.lm_head = V18LMHead(
            d_model=d_model, vocab_size=vocab_size,
            cardan_k=cardan_k)

        self.mirror = MirroredInstructionHead(
            d_model, alpha=mirror_alpha, learnable_alpha=True)

        self._orbit = 0

    def set_orbit(self, orbit: int):
        self._orbit = orbit % 4
        self.lm_head.set_orbit(orbit)

    def build_cardan(self, token_scores: Optional[torch.Tensor] = None):
        """Call after embedding weights are initialised."""
        if token_scores is None:
            token_scores = self.tok_emb.weight.norm(dim=-1)
        self.lm_head.build_cardan(token_scores)

    def encode_instruction(self, instruction_ids: torch.Tensor):
        """
        Encode an instruction sequence for mirror-head and ripple conditioning.

        Parameters
        ----------
        instruction_ids : (L,) or (1, L) token id tensor
        """
        ids = instruction_ids.view(-1)
        T   = ids.size(0)
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        emb = self.tok_emb(ids) + self.pos_emb(pos.squeeze(0))
        self.mirror.encode_instruction(emb)
        return emb.mean(dim=0)                   # (d_model,) centroid

    def forward(self,
                input_ids:       torch.Tensor,
                instruction_ids: Optional[torch.Tensor] = None,
                temperature:     float = 1.0,
                return_probs:    bool  = False) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids       : (B, T) token ids
        instruction_ids : (L,) instruction token ids (optional)
        temperature     : float
        return_probs    : bool

        Returns
        -------
        (B, T, vocab_size) logits or probs
        """
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x    = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        # Instruction conditioning
        instr_emb = None
        if instruction_ids is not None:
            instr_emb = self.encode_instruction(instruction_ids)

        for layer in self.layers:
            x = layer(x, instruction_emb=instr_emb)

        return self.lm_head(x, instruction_emb=instr_emb,
                            temperature=temperature, return_probs=return_probs)

    @torch.no_grad()
    def generate(self,
                 prompt_ids:      torch.Tensor,
                 max_new_tokens:  int   = 100,
                 temperature:     float = 1.0,
                 instruction_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Autoregressive generation with V18 conditioning.

        Parameters
        ----------
        prompt_ids      : (1, T_prompt)
        max_new_tokens  : int
        temperature     : float
        instruction_ids : (L,) optional

        Returns
        -------
        (1, T_prompt + max_new_tokens) generated ids
        """
        self.eval()
        ids = prompt_ids.clone()

        for step in range(max_new_tokens):
            # Advance Cardan orbit each step
            self.set_orbit(step % 4)

            probs = self.forward(ids,
                                 instruction_ids=instruction_ids,
                                 temperature=temperature,
                                 return_probs=True)        # (1, T, V)
            next_probs = probs[0, -1, :]                   # (V,)
            next_id    = torch.multinomial(next_probs, 1)  # (1,)
            ids        = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

        return ids

def run_text_gui_words(model, tokenizer, device):
    print("\n" + "="*60)
    print("   V18 WORD GUI")
    print("="*60)

    model.eval()

    while True:
        prompt = input("\n>>> ")
        if prompt.lower() in ["exit", "quit"]:
            break

        input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=1.9
            )

        generated = tokenizer.decode(output_ids[0])

        print("\n--- Generated ---")
        print(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run as script)
# ─────────────────────────────────────────────────────────────────────────────
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return '.###'.join(f.read().split(".")).split("###")
def _smoke_test():
    print("=" * 60)
    print("  V18 Neural Addon — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # 1. Individual layers
    theb = MiniThebault(gamma=0.12, dim=-1).to(device)
    x    = torch.randn(4, 32, 128, device=device)
    assert theb(x).shape == x.shape, "MiniThebault shape mismatch"
    print("  ✓ MiniThebault")

    emb  = BolyaiEmbedding(128).to(device)
    rho, theta, sigma = emb(x)
    assert rho.shape == (4, 32), "BolyaiEmbedding shape mismatch"
    print("  ✓ BolyaiEmbedding")

    eff  = EfferenceKernel(128).to(device)
    feat = eff(rho.reshape(-1), theta.reshape(-1), sigma.reshape(-1))
    assert feat.shape == (4 * 32, 128), "EfferenceKernel shape mismatch"
    print("  ✓ EfferenceKernel")

    aniso = AnisoDirKernel().to(device)
    C     = 20
    cr    = torch.rand(C, device=device)
    ct    = torch.rand(C, device=device) * math.pi
    cs    = torch.rand(C, device=device) * 2 + 1
    g     = aniso.gram(cr, ct, cs)
    assert g.shape == (C, C), "AnisoDirKernel shape mismatch"
    print("  ✓ AnisoDirKernel")

    rff   = RandomFourierFeatures(rff_dim=16).to(device)
    feat  = rff(cr, ct, cs)
    assert feat.shape == (C, 48), "RandomFourierFeatures shape mismatch"
    print("  ✓ RandomFourierFeatures")

    ripple = RippleShift().to(device)
    ripple.set_instruction(0.5, 0.7, 1.2)
    ripple.add_stub(0.4, 0.6, 1.1, 0.8)
    rs = ripple(cr, ct, cs)
    assert rs.shape == (C,), "RippleShift shape mismatch"
    print("  ✓ RippleShift")

    mixer = SpaghettiMixer(C).to(device)
    sig_a = torch.randn(C, device=device)
    sig_b = torch.randn(C, device=device)
    mixer.add_named_strand("instruction", sig_a, weight=1.0)
    mixer.add_named_strand("ripple",      sig_b, weight=0.5)
    out = mixer()
    assert out.shape == (C,), "SpaghettiMixer shape mismatch"
    print("  ✓ SpaghettiMixer")

    cardan = CardanAperture(vocab_size=500, aperture_k=64).to(device)
    scores = torch.randn(500, device=device)
    cardan.build(scores)
    logits = torch.randn(2, 10, 500, device=device)
    logits_out = cardan(logits, orbit=1)
    assert logits_out.shape == logits.shape, "CardanAperture shape mismatch"
    print("  ✓ CardanAperture")

    mir  = MirroredInstructionHead(128).to(device)
    instr = torch.randn(8, 128, device=device)
    mir.encode_instruction(instr)
    cand  = torch.randn(20, 128, device=device)
    probs = mir(cand)
    assert probs.shape == (20,), "MirroredInstructionHead shape mismatch"
    print("  ✓ MirroredInstructionHead")

    # 2. V18Block
    v18b = V18Block(d_model=64).to(device)
    h    = torch.randn(2, 16, 64, device=device)
    out  = v18b(h)
    assert out.shape == h.shape, "V18Block shape mismatch"
    print("  ✓ V18Block")

    # 3. V18TransformerLayer
    layer = V18TransformerLayer(d_model=64, n_heads=4).to(device)
    out   = layer(h)
    assert out.shape == h.shape, "V18TransformerLayer shape mismatch"
    print("  ✓ V18TransformerLayer")

    # 4. Full model
    model = V18Model(
        vocab_size=20000,     # Matches assert's vocab dim
        d_model=64,         # Hidden dim
        n_layers=2,          # Layers
        n_heads=4,
        max_seq_len=64,       # Pos emb covers seq len
        cardan_k=32
    ).to(device)  # device from your code
    model.build_cardan()

    ids = torch.randint(0, 20000, (2, 12), device=device)  # B=2, T=12, vocab=20000
    logits_out = model(ids)
    assert logits_out.shape == (2, 12, 20000), "V18Model shape mismatch"
    print("  ✓ V18Model forward pass")

    gen = model.generate(ids[:1, :4], max_new_tokens=8)
    assert gen.shape == (1, 12), "V18Model.generate shape mismatch"
    print("  ✓ V18Model.generate")

    # 5. Wrapper
    base = nn.Embedding(20000, 64)    # trivial "base model"
    # We need a base model that outputs (B,T,d); wrap Embedding manually
    class SimpleBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(20000, 64)
            self.pos = nn.Embedding(64, 64)
        def forward(self, x):
            B, T = x.shape
            return self.emb(x) + self.pos(torch.arange(T, device=x.device))
    sb      = SimpleBase().to(device)
    wrapped = V18AddonWrapper(sb, d_model=64, vocab_size=20000, cardan_k=32).to(device)
    wrapped.build_cardan(torch.randn(20000))
    out = wrapped(ids)
    assert out.shape == (2, 12, 20000), "V18AddonWrapper shape mismatch"
    print("  ✓ V18AddonWrapper")

    # 6. Polynomial warp + sawtooth
    poly   = PartitionPolynomialWarp(degree=7, harmonics=3, jaggy_scale=0.3)
    probs  = torch.softmax(torch.randn(2, 100), dim=-1)
    warped = poly(probs)
    assert abs(warped.sum().item() - 2.0) < 0.01, "PartitionPolynomialWarp normalisation"
    print("  ✓ PartitionPolynomialWarp")

    saw    = MirroredSawtoothClip(period=0.1)
    clipped = saw(probs)
    assert abs(clipped.sum().item() - 2.0) < 0.01, "MirroredSawtoothClip normalisation"
    print("  ✓ MirroredSawtoothClip")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  V18Model parameter count: {n_params:,}")
    print("\n  All checks passed ✓")
    tokenizer = WordTokenizer(vocab_size=20000)  # FIXED: match model.vocab_size
    tokenizer.build_vocab(
    load_text(input("Filename: "))
    )
    
    run_text_gui_words(model, tokenizer, device)
if __name__ == "__main__":
    _smoke_test()
