"""
AI.py  —  V18-GEO Full: TransmutableGeoOp + Dataset + Training + Generation
=============================================================================
George Wagenknecht — April 2026

Fixes applied vs paste.txt:
  • max_seq_len consistent everywhere (128) → no pos_emb size mismatch
  • VOCAB raised to 10000 for meaningful trigram coverage
  • batch_size capped at 32 (was 2512)
  • num_workers=0 (safe on Windows)
  • Tokenizer saved/loaded alongside checkpoint (no dummy vocab)
  • generate_sample() reloads proper tokenizer
  • WordTokenizer build_vocab bug fixed (idx*_ → idx)
  • seq_len padded inputs clamped to max_seq_len in forward
"""

from __future__ import annotations

import math, os, json
from typing import List, Optional, Tuple, Callable, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import requests

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG — change these in one place only
# ─────────────────────────────────────────────────────────────────────────────
VOCAB       = 10_000
D_MODEL     = 64
N_LAYERS    = 4
N_HEADS     = 4
MAX_SEQ_LEN = 128      # ← single source of truth for pos_emb size
SEQ_LEN     = 64       # training window (must be < MAX_SEQ_LEN)
CARDAN_K    = 64
DROPOUT     = 0.1
EPOCHS      = 10
LR          = 6e-4
BATCH_GPU   = 32
BATCH_CPU   = 8
CHECKPOINT  = "v18_geo_tinyshakespeare.pt"
TOKENIZER_F = "v18_tokenizer.json"

# ─────────────────────────────────────────────────────────────────────────────
# TRANSMUTABLE GEOMETRIC OPERATOR
# ─────────────────────────────────────────────────────────────────────────────

class TransmutableGeoOp(nn.Module):
    """
    Wraps any callable with learnable Bolyai-space geometry (rho, theta, sigma).
    Pre-transform: scale(rho) → rotate(theta) → curvature-suppress(sigma).
    """
    def __init__(self, name: str, func: Callable,
                 rho: float = 1.0, theta: float = 0.0, sigma: float = 0.5,
                 learnable: bool = True,
                 weight_tensor: Optional[torch.Tensor] = None,
                 extra_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.op_name     = name
        self.func        = func
        self.extra_params = extra_params or {}
        _r = torch.tensor(float(rho))
        _t = torch.tensor(float(theta))
        _s = torch.tensor(float(sigma))
        if learnable:
            self.rho   = nn.Parameter(_r)
            self.theta = nn.Parameter(_t)
            self.sigma = nn.Parameter(_s)
        else:
            self.register_buffer("rho",   _r)
            self.register_buffer("theta", _t)
            self.register_buffer("sigma", _s)
        if weight_tensor is not None:
            self.weight_tensor = (weight_tensor if isinstance(weight_tensor, nn.Parameter)
                                  else nn.Parameter(weight_tensor.float()))
        else:
            self.weight_tensor = None

    def geo_transform(self, x: torch.Tensor) -> torch.Tensor:
        rho = self.rho.clamp(1e-3, 20.0)
        x   = x * rho
        if x.dim() >= 2 and x.shape[-1] > 1:
            x = torch.cos(self.theta) * x + torch.sin(self.theta) * torch.roll(x, 1, dims=-1)
        sig  = self.sigma.clamp(0.0, 1.0)
        apex = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return x * (1.0 - sig * apex / (apex + 1.0))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.geo_transform(x)
        if self.weight_tensor is not None:
            w = self.weight_tensor
            if w.shape == x.shape[-len(w.shape):]:
                x = x * w.clamp(-20.0, 20.0)
        return self.func(x, *args, **{**self.extra_params, **kwargs})

    def __repr__(self):
        return (f"TransmutableGeoOp({self.op_name!r}, "
                f"ρ={self.rho.item():.3f}, θ={self.theta.item():.3f}, "
                f"σ={self.sigma.item():.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def _build_registry() -> Dict[str, TransmutableGeoOp]:
    reg = {}
    # Math ops
    reg["tanh"]         = TransmutableGeoOp("tanh",      torch.tanh,             rho=0.1,  sigma=0.5)
    reg["tanh_full"]    = TransmutableGeoOp("tanh_full",  torch.tanh,             rho=1.0,  sigma=0.0)
    reg["exp"]          = TransmutableGeoOp("exp",        torch.exp,              rho=1.0,  sigma=0.0)
    reg["cos"]          = TransmutableGeoOp("cos",        torch.cos,              theta=math.pi/4, sigma=0.0)
    reg["sin"]          = TransmutableGeoOp("sin",        torch.sin,              theta=math.pi/4, sigma=0.0)
    reg["arctan"]       = TransmutableGeoOp("arctan",     torch.arctan,           rho=1.0,  sigma=0.3)
    reg["atan2"]        = TransmutableGeoOp("atan2",      lambda x,y: torch.atan2(x,y), rho=1.0, sigma=0.0, learnable=False)
    reg["sqrt"]         = TransmutableGeoOp("sqrt",       torch.sqrt,             rho=1.0,  sigma=0.0)
    reg["log"]          = TransmutableGeoOp("log",        lambda x: torch.log(x.clamp(min=1e-8)), rho=1.0, sigma=0.0, learnable=False)
    reg["abs"]          = TransmutableGeoOp("abs",        torch.abs,              rho=1.0,  sigma=0.0)
    reg["relu"]         = TransmutableGeoOp("relu",       F.relu,                 rho=1.0,  sigma=0.0)
    reg["gelu"]         = TransmutableGeoOp("gelu",       F.gelu,                 rho=1.0,  sigma=0.2)
    reg["softmax"]      = TransmutableGeoOp("softmax",    lambda x: torch.softmax(x, dim=-1), rho=1.0, sigma=0.35, learnable=False)
    reg["mean"]         = TransmutableGeoOp("mean",       lambda x: x.mean(),    rho=1.0,  sigma=0.0, learnable=False)
    reg["std"]          = TransmutableGeoOp("std",        lambda x: x.std(),     rho=1.0,  sigma=0.0, learnable=False)
    reg["norm_1d"]      = TransmutableGeoOp("norm_1d",    lambda x: x.norm(dim=-1, keepdim=True), rho=1.0, sigma=0.0, learnable=False)
    reg["clamp_signal"] = TransmutableGeoOp("clamp_sig",  lambda x: x.clamp(-20., 20.), rho=1.0, sigma=0.0, learnable=False)
    reg["clamp_unit"]   = TransmutableGeoOp("clamp_unit", lambda x: x.clamp(-0.9999, 0.9999), rho=1.0, sigma=0.0, learnable=False)
    reg["triu"]         = TransmutableGeoOp("triu",       lambda x: torch.triu(x, diagonal=1), rho=1.0, sigma=0.0, learnable=False)
    reg["topk"]         = TransmutableGeoOp("topk",       lambda x,k: torch.topk(x,k), rho=1.0, sigma=0.0, learnable=False)
    reg["argsort"]      = TransmutableGeoOp("argsort",    lambda x: torch.argsort(x, descending=True), rho=1.0, sigma=0.0, learnable=False)
    reg["multinomial"]  = TransmutableGeoOp("multinomial",lambda x: torch.multinomial(x, 1), rho=1.0, sigma=0.0, learnable=False)
    # Weight ops
    reg["gamma_blend"]    = TransmutableGeoOp("gamma_blend",  lambda x: x, rho=0.82,  sigma=0.15, weight_tensor=nn.Parameter(torch.tensor(0.82)))
    reg["lambda_scale"]   = TransmutableGeoOp("lambda_scale", lambda x: x, rho=8.0,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor([8.0,4.0,4.0])))
    reg["coupling_mob"]   = TransmutableGeoOp("coupling_mob", lambda x: x, rho=0.35,  sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.35)))
    reg["alpha_mirror"]   = TransmutableGeoOp("alpha_mirror", lambda x: x, rho=0.35,  sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.35)))
    reg["ripple_decay"]   = TransmutableGeoOp("ripple_decay", torch.exp,   rho=-0.9,  sigma=0.0)
    reg["logit_bonus"]    = TransmutableGeoOp("logit_bonus",  torch.add,   rho=8.0,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(8.0)))
    reg["ripple_scale"]   = TransmutableGeoOp("ripple_scale", lambda x: x, rho=0.1,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.1)))
    reg["theb_gamma_in"]  = TransmutableGeoOp("theb_in",      lambda x: x, rho=0.12,  sigma=0.12, weight_tensor=nn.Parameter(torch.tensor(0.12)))
    reg["theb_gamma_ker"] = TransmutableGeoOp("theb_ker",     lambda x: x, rho=0.10,  sigma=0.10, weight_tensor=nn.Parameter(torch.tensor(0.10)))
    reg["theb_gamma_rip"] = TransmutableGeoOp("theb_rip",     lambda x: x, rho=0.15,  sigma=0.15, weight_tensor=nn.Parameter(torch.tensor(0.15)))
    reg["theb_gamma_out"] = TransmutableGeoOp("theb_out",     lambda x: x, rho=0.12,  sigma=0.12, weight_tensor=nn.Parameter(torch.tensor(0.12)))
    reg["rff_scale"]      = TransmutableGeoOp("rff_scale",    lambda x: x, rho=1.0,   sigma=0.0)
    reg["poly_jaggy"]     = TransmutableGeoOp("jaggy",        lambda x: x, rho=0.45,  sigma=0.2,  weight_tensor=nn.Parameter(torch.tensor(0.45)))
    reg["sawtooth_period"]= TransmutableGeoOp("saw_period",   lambda x: x, rho=0.1,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.1)))
    reg["midpoint_scale"] = TransmutableGeoOp("midpoint",     lambda x: x, rho=0.9,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.9)))
    reg["hyp_scale"]      = TransmutableGeoOp("hyp_scale",    lambda x: x, rho=0.9,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.9)))
    reg["bolyai_disk"]    = TransmutableGeoOp("disk_scale",   lambda x: x, rho=0.18,  sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.18)))
    reg["sigma_curvature"]= TransmutableGeoOp("sigma_curv",   lambda x: x, rho=2.0,   sigma=0.5,  weight_tensor=nn.Parameter(torch.tensor(2.0)))
    reg["ripple_nudge"]   = TransmutableGeoOp("rip_nudge",    lambda x: x, rho=0.1,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.1)))
    reg["efference_nudge"]= TransmutableGeoOp("eff_nudge",    lambda x: x, rho=0.05,  sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.05)))
    reg["cardan_weight"]  = TransmutableGeoOp("cardan_w",     lambda x: x, rho=8.0,   sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(8.0)))
    reg["mob_coup_ab"]    = TransmutableGeoOp("mob_ab",       lambda x: x, rho=0.35,  sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.35)))
    reg["mob_coup_bc"]    = TransmutableGeoOp("mob_bc",       lambda x: x, rho=0.28,  sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.28)))
    reg["mob_coup_ca"]    = TransmutableGeoOp("mob_ca",       lambda x: x, rho=0.21,  sigma=0.0,  weight_tensor=nn.Parameter(torch.tensor(0.21)))
    return reg

GEO = _build_registry()


# ─────────────────────────────────────────────────────────────────────────────
# GEO UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _layer_norm_1d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu  = x.mean()
    std = x.std()
    return (x - mu) / (std.item() + eps)

def _l1_simplex_project(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x     = torch.nan_to_num(x, nan=0., posinf=50., neginf=-50.)
    x_pos = GEO["relu"](x - x.min()).clamp(min=eps)
    total = x_pos.sum()
    return x_pos / total if total.item() > 0 else torch.full_like(x, 1./x.shape[0])

def _mobius_cross_shift(a: torch.Tensor, b: torch.Tensor, coupling: float = 0.35) -> torch.Tensor:
    a   = GEO["clamp_signal"](a)
    b   = GEO["clamp_signal"](b)
    ta  = GEO["tanh"](a)
    tb  = GEO["tanh"](b)
    w   = GEO["coupling_mob"].weight_tensor.item() if GEO["coupling_mob"].weight_tensor is not None else coupling
    den = (1.0 + w * ta * tb).clamp(min=1e-6)
    return GEO["clamp_unit"]((ta + w * tb) / den) * 10.0

def _maj_gate(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    bx  = (x > 0).float(); by = (y > 0).float(); bz = (z > 0).float()
    maj = ((bx*by + bx*bz + by*bz) >= 2).float()
    avg = (x + y + z) / 3.0
    return maj * avg + (1.0 - maj) * GEO["relu"](avg)


# ─────────────────────────────────────────────────────────────────────────────
# WORD TOKENIZER  (trigram-based, fixed build_vocab)
# ─────────────────────────────────────────────────────────────────────────────

class WordTokenizer:
    def __init__(self, vocab_size: int = VOCAB):
        self.vocab_size  = vocab_size
        self.trigram2id  = {"<pad>": 0, "<unk>": 1}
        self.id2trigram  = {0: "<pad>", 1: "<unk>"}
        self._built      = False

    def _tokenize(self, text: str) -> List[str]:
        words = text.lower().split()
        return [" ".join(words[i:i+3]) for i in range(len(words) - 2)]

    def build_vocab(self, texts: List[str]):
        freq: Dict[str, int] = {}
        for text in texts:
            for tok in self._tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1
        for trigram, _ in sorted(freq.items(), key=lambda x: -x[1])[:self.vocab_size - 2]:
            idx = len(self.trigram2id)
            self.trigram2id[trigram] = idx      # ← BUG FIX: was idx*_ (freq)
            self.id2trigram[idx]     = trigram
        self._built = True

    def encode(self, text: str) -> torch.Tensor:
        tokens = self._tokenize(text)
        if not tokens:
            return torch.tensor([0], dtype=torch.long)
        ids = [self.trigram2id.get(t, 1) for t in tokens]
        return torch.tensor(ids, dtype=torch.long).clamp(0, self.vocab_size - 1)

    def decode(self, ids: torch.Tensor) -> str:
        trigrams = [self.id2trigram.get(int(i), "<unk>") for i in ids]
        words: List[str] = []
        for trig in trigrams:
            words.extend(trig.split())
        return " ".join(words).replace(" .", ".").replace(" ,", ",")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "trigram2id": self.trigram2id}, f)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path) as f:
            d = json.load(f)
        tok = cls(d["vocab_size"])
        tok.trigram2id  = {k: int(v) for k, v in d["trigram2id"].items()}
        tok.id2trigram  = {int(v): k for k, v in d["trigram2id"].items()}
        tok._built      = True
        return tok


# ─────────────────────────────────────────────────────────────────────────────
# 1. MiniThebault
# ─────────────────────────────────────────────────────────────────────────────

class MiniThebault(nn.Module):
    def __init__(self, gamma_op: TransmutableGeoOp = None, dim: int = -1):
        super().__init__()
        self.dim      = dim
        self.gamma_op = gamma_op if gamma_op is not None else GEO["theb_gamma_in"]
        self._gamma_buf = (self.gamma_op.weight_tensor
                          if self.gamma_op.weight_tensor is not None
                          else self.gamma_op.rho)

    @property
    def gamma(self) -> torch.Tensor:
        return self._gamma_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim % x.ndim
        n = x.shape[d]
        if n < 3:
            return x
        idx_l = torch.arange(n, device=x.device)
        idx_r = torch.arange(n, device=x.device)
        idx_l[0]  = 0;  idx_l[1:]  = torch.arange(n-1, device=x.device)
        idx_r[-1] = n-1; idx_r[:-1] = torch.arange(1, n, device=x.device)
        left  = x.index_select(d, idx_l)
        right = x.index_select(d, idx_r)
        mid   = (left + right) * GEO["midpoint_scale"].weight_tensor.item()
        delta = x - mid
        curv  = GEO["abs"](delta) / GEO["abs"](delta).amax(dim=d, keepdim=True).clamp(min=1e-8)
        g     = (self.gamma_op.weight_tensor.clamp(0.,1.)
                 if self.gamma_op.weight_tensor is not None
                 else self.gamma_op.rho)
        bl    = x - g * curv * delta
        mu    = bl.mean(dim=d, keepdim=True)
        std   = bl.std(dim=d, keepdim=True)
        return (bl - mu) / (std + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BolyaiEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class BolyaiEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 2, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disk  = GEO["bolyai_disk"].weight_tensor.item()
        hyp_s = GEO["hyp_scale"].weight_tensor.item()
        xy    = GEO["tanh_full"](self.proj(x)) * disk
        eu    = 1.0 - (xy.norm(dim=-1) - 1e-8).clamp(0., 1.-1e-8)
        hyp   = 2.0 * GEO["arctan"](eu.clamp(1e-8, 1.-1e-8))
        rho   = GEO["tanh_full"](hyp * hyp_s)
        theta = GEO["atan2"](xy[..., 1], xy[..., 0]) % math.pi
        sigma = GEO["sigma_curvature"].weight_tensor.item() / (1.0 - eu.pow(2)).clamp(min=1e-8)
        return rho, theta, sigma


# ─────────────────────────────────────────────────────────────────────────────
# 3. EfferenceKernel
# ─────────────────────────────────────────────────────────────────────────────

class EfferenceKernel(nn.Module):
    def __init__(self, d_model: int = 128, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lambdas = GEO["lambda_scale"]
        self.omega   = nn.Parameter(torch.randn(3, d_model, generator=g))
        self.bias    = nn.Parameter(torch.randn(d_model, generator=g))

    def forward(self, rho, theta, sigma) -> torch.Tensor:
        rho_eff = rho * GEO["cos"](theta)
        comps   = torch.stack([rho_eff, theta, sigma], dim=-1)
        lam     = GEO["lambda_scale"].weight_tensor.clamp(0.1, 20.0)
        scaled  = comps.unsqueeze(-1) * lam.unsqueeze(-1)
        dots    = (scaled * self.omega).sum(dim=-2)
        return GEO["exp"]((dots + self.bias).clamp(max=20.0))


# ─────────────────────────────────────────────────────────────────────────────
# 4. RandomFourierFeatures
# ─────────────────────────────────────────────────────────────────────────────

class RandomFourierFeatures(nn.Module):
    def __init__(self, rff_dim: int = 32, sigma_rho: float = 1.0,
                 sigma_theta: float = 0.5, sigma_sigma: float = 2.0, seed: int = 42):
        super().__init__()
        self.rff_dim = rff_dim
        self._scale  = math.sqrt(2.0 / rff_dim)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("omega_rho",   torch.randn(rff_dim,1,generator=g)/sigma_rho)
        self.register_buffer("omega_theta", torch.randn(rff_dim,1,generator=g)/sigma_theta)
        self.register_buffer("omega_sigma", torch.randn(rff_dim,1,generator=g)/sigma_sigma)
        self.register_buffer("bias_rho",    torch.rand(rff_dim,generator=g)*2*math.pi)
        self.register_buffer("bias_theta",  torch.rand(rff_dim,generator=g)*2*math.pi)
        self.register_buffer("bias_sigma",  torch.rand(rff_dim,generator=g)*2*math.pi)

    def forward(self, rho, theta, sigma) -> torch.Tensor:
        pr = self.bias_rho.unsqueeze(1)   + self.omega_rho   @ rho.unsqueeze(0)
        pt = self.bias_theta.unsqueeze(1) + self.omega_theta @ theta.unsqueeze(0)
        ps = self.bias_sigma.unsqueeze(1) + self.omega_sigma @ sigma.unsqueeze(0)
        s  = self._scale
        return torch.cat([(s*GEO["cos"](pr.T)), (s*GEO["cos"](pt.T)), (s*GEO["cos"](ps.T))], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AnisoDirKernel
# ─────────────────────────────────────────────────────────────────────────────

class AnisoDirKernel(nn.Module):
    def __init__(self, lambda_rho=0.5, lambda_theta=0.5,
                 lambda_sigma=0.5, alpha=0.5, learnable=False):
        super().__init__()
        params = torch.tensor([lambda_rho, lambda_theta, lambda_sigma, alpha])
        if learnable:
            self.log_params = nn.Parameter(params.log())
        else:
            self.register_buffer("log_params", params.log())

    def _unpack(self):
        p = GEO["exp"](self.log_params)
        return p[0], p[1], p[2], p[3]

    def score(self, anc_rho, anc_theta, anc_sigma, c_rho, c_theta, c_sigma):
        lr, lt, ls, a = self._unpack()
        d_rho   = c_rho   - anc_rho
        d_theta = (c_theta - anc_theta) * (a * anc_rho + 1.0)
        d_sigma = c_sigma  - anc_sigma
        return GEO["exp"](-(lr*d_rho**2 + lt*d_theta**2 + ls*d_sigma**2))

    def gram(self, c_rho, c_theta, c_sigma):
        lr, lt, ls, a = self._unpack()
        ri = c_rho.unsqueeze(1);   rj = c_rho.unsqueeze(0)
        ti = c_theta.unsqueeze(1); tj = c_theta.unsqueeze(0)
        si = c_sigma.unsqueeze(1); sj = c_sigma.unsqueeze(0)
        return GEO["exp"](-(lr*(rj-ri)**2 + lt*((tj-ti)*(a*ri+1.))**2 + ls*(sj-si)**2))


# ─────────────────────────────────────────────────────────────────────────────
# 6. RippleShift
# ─────────────────────────────────────────────────────────────────────────────

class RippleShift(nn.Module):
    def __init__(self, ripple_decay: float = 0.9, ripple_scale: float = 0.1,
                 thebault_gamma: float = 0.95):
        super().__init__()
        self.kernel    = AnisoDirKernel()
        self.theb      = MiniThebault(gamma_op=TransmutableGeoOp(
            "theb_rip_local", lambda x: x, rho=thebault_gamma, sigma=thebault_gamma,
            weight_tensor=nn.Parameter(torch.tensor(thebault_gamma))))
        self.decay_geo = TransmutableGeoOp("rip_decay_local", torch.exp,
            rho=ripple_decay, sigma=0., weight_tensor=nn.Parameter(torch.tensor(ripple_decay)))
        self.scale_geo = TransmutableGeoOp("rip_scale_local", lambda x: x,
            rho=ripple_scale, sigma=0., weight_tensor=nn.Parameter(torch.tensor(ripple_scale)))
        self._instr: Optional[Tuple[float,float,float]] = None
        self._stubs:  List[Tuple[float,float,float,float]] = []

    def set_instruction(self, rho, theta, sigma):  self._instr = (rho, theta, sigma)
    def reset_stubs(self):                          self._stubs.clear()
    def add_stub(self, rho, theta, sigma, d):       self._stubs.append((rho, theta, sigma, d))

    def forward(self, c_rho, c_theta, c_sigma) -> torch.Tensor:
        C      = c_rho.shape[0]
        device = c_rho.device
        if not self._stubs or C == 0:
            return torch.zeros(C, device=device)
        ripple = torch.zeros(C, device=device)
        for (sr, st, ss, d) in self._stubs:
            ripple = ripple + d * self.kernel.score(sr, st, ss, c_rho, c_theta, c_sigma)
        ranks  = GEO["argsort"](GEO["argsort"](GEO["abs"](ripple))).float()
        decay  = GEO["exp"](-(self.decay_geo.weight_tensor.item()) * ranks / max(float(C), 1.))
        ripple = ripple * decay
        std    = ripple.std()
        if std.item() > 1e-8:
            ripple = (ripple - ripple.mean()) / std
        return self.theb(ripple * self.scale_geo.weight_tensor.item())


# ─────────────────────────────────────────────────────────────────────────────
# 7. SpaghettiMixer
# ─────────────────────────────────────────────────────────────────────────────

class SpaghettiMixer(nn.Module):
    DEFAULT_ROUTING = {
        "instruction": ([0,1], +1.), "ripple":     ([0,1,2], +1.),
        "cot":         ([0,2], +1.), "ooi":        ([0,1],   +1.),
        "kernel_reg":  ([1,2], +1.), "kernel_ori": ([0,2],   +1.),
        "walk":        ([0,2], +1.), "repulsion":  ([1,0],   -1.),
        "mrv":         ([2],   +1.), "pdn":        ([1,2],   +1.),
        "echo":        ([0],   +1.), "mirror":     ([0,2],   +1.),
        "cardan":      ([0,2], +1.),
    }
    def __init__(self, C: int, coupling=0.35, temperature=0.8, thebault_gamma=0.10):
        super().__init__()
        self.C        = C
        self.temp_geo = TransmutableGeoOp("spag_temp", lambda x: x, rho=temperature, sigma=0.,
                                          weight_tensor=nn.Parameter(torch.tensor(temperature)))
        self.theb     = MiniThebault(gamma_op=TransmutableGeoOp(
            "theb_spag", lambda x: x, rho=thebault_gamma, sigma=thebault_gamma,
            weight_tensor=nn.Parameter(torch.tensor(thebault_gamma))))
        self._strands: List[Tuple] = []

    def reset(self):  self._strands.clear()

    def add_strand(self, signal, weight, routing=None, sign=1.):
        self._strands.append((signal, weight, routing or [0,1,2], sign))

    def add_named_strand(self, name, signal, weight):
        route, sign = self.DEFAULT_ROUTING.get(name, ([0,1,2], 1.))
        self.add_strand(signal, weight, route, sign)

    def forward(self) -> torch.Tensor:
        C      = self.C
        device = self._strands[0][0].device if self._strands else torch.device("cpu")
        accum  = [torch.zeros(C, device=device) for _ in range(3)]
        for (sig, w, routing, sign) in self._strands:
            s = sig if sig.shape[0] == C else torch.zeros(C, device=device)
            for idx in routing:
                accum[idx] = accum[idx] + sign * w * s
        temp    = max(self.temp_geo.weight_tensor.item(), 1e-6)
        blended = [self.theb(_layer_norm_1d(a / temp)) for a in accum]
        aA, aB, aC = blended
        aAnew = _mobius_cross_shift(aA, aB, GEO["mob_coup_ab"].weight_tensor.item())
        aBnew = _mobius_cross_shift(aB, aC, GEO["mob_coup_bc"].weight_tensor.item())
        aCnew = _mobius_cross_shift(aC, aA, GEO["mob_coup_ca"].weight_tensor.item())
        return _layer_norm_1d(_maj_gate(aAnew, aBnew, aCnew))


# ─────────────────────────────────────────────────────────────────────────────
# 8. CardanAperture
# ─────────────────────────────────────────────────────────────────────────────

class CardanAperture(nn.Module):
    def __init__(self, vocab_size: int, aperture_k: int = 64, logit_weight: float = 8.0):
        super().__init__()
        self.V         = vocab_size
        self.aperture_k = min(aperture_k, vocab_size)
        self.logit_geo  = TransmutableGeoOp("cardan_local", lambda x: x, rho=logit_weight, sigma=0.,
                                             weight_tensor=nn.Parameter(torch.tensor(logit_weight)))
        G = max(1, math.ceil(math.sqrt(vocab_size)))
        self.G = G
        self.register_buffer("aperture", torch.zeros(4, vocab_size))
        self._built = False

    def build(self, token_scores: torch.Tensor):
        V, G, k = self.V, self.G, self.aperture_k
        _, top_idx = GEO["topk"](token_scores, k)
        def idx2rc(i):    return divmod(int(i), G)
        def rc2idx(r, c): return min(r*G+c, V-1)
        def rot90(r, c):  return c, G-1-r
        aperture = torch.zeros(4, V)
        for pos in top_idx.tolist():
            r, c = idx2rc(pos)
            aperture[0, rc2idx(r,c)] = 1.
            r, c = rot90(r,c); aperture[1, rc2idx(r,c)] = 1.
            r, c = rot90(r,c); aperture[2, rc2idx(r,c)] = 1.
            r, c = rot90(r,c); aperture[3, rc2idx(r,c)] = 1.
        self.aperture.copy_(aperture)
        self._built = True

    def forward(self, logits, orbit: int):
        if not self._built: return logits
        return logits + self.aperture[orbit % 4] * self.logit_geo.weight_tensor.item()

    def aperture_scores(self, orbit: int): return self.aperture[orbit % 4]


# ─────────────────────────────────────────────────────────────────────────────
# 9. MirroredInstructionHead
# ─────────────────────────────────────────────────────────────────────────────

class MirroredInstructionHead(nn.Module):
    def __init__(self, d_model: int, alpha: float = 0.35, learnable_alpha: bool = False):
        super().__init__()
        self.proj_fwd = nn.Linear(d_model, d_model, bias=False)
        self.proj_mir = nn.Linear(d_model, d_model, bias=False)
        self.alpha_geo = TransmutableGeoOp("alpha_local", lambda x: x, rho=alpha, sigma=0.,
                                           learnable=learnable_alpha,
                                           weight_tensor=nn.Parameter(torch.tensor(alpha)))
        self._fwd_emb: Optional[torch.Tensor] = None
        self._mir_emb: Optional[torch.Tensor] = None

    @property
    def alpha(self): return self.alpha_geo.weight_tensor.clamp(0., 1.)

    def encode_instruction(self, emb: torch.Tensor):
        if emb.ndim == 2:
            fwd = emb.mean(dim=0)
            mir = emb.flip(0).mean(dim=0)
        else:
            fwd = emb
            idx = torch.arange(emb.shape[-1], device=emb.device)
            mir = emb * torch.where(idx % 2 == 0, torch.tensor(-1.), torch.tensor(1.))
        self._fwd_emb = self.proj_fwd(fwd)
        self._mir_emb = self.proj_mir(mir)

    def forward(self, cands: torch.Tensor) -> torch.Tensor:
        if self._fwd_emb is None:
            C = cands.shape[0]
            return torch.full((C,), 1./C, device=cands.device)
        p_fwd = GEO["softmax"](cands @ self._fwd_emb)
        p_mir = GEO["softmax"](cands @ self._mir_emb)
        a     = self.alpha
        return (1.-a)*p_fwd + a*p_mir


# ─────────────────────────────────────────────────────────────────────────────
# 10. MirroredSawtoothClip
# ─────────────────────────────────────────────────────────────────────────────

class MirroredSawtoothClip(nn.Module):
    def __init__(self, period: float = 0.1):
        super().__init__()
        self.period_geo = TransmutableGeoOp("saw_local", lambda x: x, rho=period, sigma=0.,
                                            weight_tensor=nn.Parameter(torch.tensor(period)))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        p    = self.period_geo.weight_tensor.item()
        half = p * 0.5
        fold = probs % p
        mir  = torch.where(fold <= half, fold, p - fold)
        mir  = (mir / half).clamp(min=1e-12)
        return mir / mir.sum(dim=-1, keepdim=True).clamp(min=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 11. PartitionPolynomialWarp
# ─────────────────────────────────────────────────────────────────────────────

class PartitionPolynomialWarp(nn.Module):
    def __init__(self, degree=17, harmonics=7, jaggy_scale=0.45, logit_mode=False):
        super().__init__()
        self.degree    = degree
        self.harmonics = harmonics
        self.jaggy_geo = TransmutableGeoOp("jaggy_local", lambda x: x, rho=jaggy_scale, sigma=0.,
                                           weight_tensor=nn.Parameter(torch.tensor(jaggy_scale)))
        self.logit_mode = logit_mode

    @staticmethod
    def _cheb(x: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0: return torch.ones_like(x)
        if n == 1: return x.clone()
        t0, t1 = torch.ones_like(x), x.clone()
        for _ in range(2, n+1):
            t0, t1 = t1, 2*x*t1 - t0
        return t1

    def _mask(self, V, device, dtype):
        x    = torch.linspace(-1., 1., V, device=device, dtype=dtype)
        cheb = self._cheb(x, self.degree)
        harm = torch.zeros(V, device=device, dtype=dtype)
        for k in range(1, self.harmonics+1):
            harm = harm + GEO["sin"](torch.tensor(k*math.pi)*x) * GEO["cos"](torch.tensor((k+1)*math.pi)*x) / k
        harm = harm / harm.abs().max().clamp(min=1e-8)
        js   = self.jaggy_geo.weight_tensor.item()
        return 1. + js * (cheb + harm) * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V    = x.size(-1)
        mask = self._mask(V, x.device, x.dtype)
        if self.logit_mode:
            js   = self.jaggy_geo.weight_tensor.item()
            cheb = self._cheb(torch.linspace(-1.,1.,V,device=x.device,dtype=x.dtype), self.degree)
            return x + js * cheb
        warped = GEO["abs"](x) * mask.clamp(min=1e-6)
        return warped / warped.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 12. V18Block
# ─────────────────────────────────────────────────────────────────────────────

class V18Block(nn.Module):
    def __init__(self, d_model=D_MODEL, spaghetti_coup=0.35,
                 ripple_decay=0.50, ripple_scale=0.50, n_rff=32):
        super().__init__()
        self.d_model    = d_model
        self.bolyai     = BolyaiEmbedding(d_model)
        self.efference  = EfferenceKernel(d_model)
        self.rff        = RandomFourierFeatures(rff_dim=n_rff)
        self.aniso      = AnisoDirKernel()
        self.ripple     = RippleShift(ripple_decay, ripple_scale, thebault_gamma=0.15)
        self.theb_input  = MiniThebault(gamma_op=GEO["theb_gamma_in"],  dim=-1)
        self.theb_kernel = MiniThebault(gamma_op=GEO["theb_gamma_ker"], dim=-1)
        self.theb_ripple = MiniThebault(gamma_op=GEO["theb_gamma_rip"], dim=-1)
        self.theb_output = MiniThebault(gamma_op=GEO["theb_gamma_out"], dim=-1)
        self.kernel_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj   = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.ln          = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, instruction_emb=None) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x_theb   = self.theb_input(x)
        rho, theta, sigma = self.bolyai(x_theb)
        rho_f   = rho.reshape(B*T)
        theta_f = theta.reshape(B*T)
        sigma_f = sigma.reshape(B*T)
        kf          = self.efference(rho_f, theta_f, sigma_f).reshape(B, T, D)
        kf          = self.theb_kernel(kf)
        gate        = self.gate_proj(x_theb)
        conditioned = self.kernel_proj(kf) * gate
        if instruction_emb is not None:
            with torch.no_grad():
                ir, it, is_ = self.bolyai(instruction_emb.unsqueeze(0))
                self.ripple.set_instruction(ir.item(), it.item(), is_.item())
                self.ripple.reset_stubs()
                self.ripple.add_stub(ir.item(), it.item(), is_.item(), 1.)
        conditioned = conditioned + kf * GEO["efference_nudge"].weight_tensor.item()
        nudge = GEO["ripple_nudge"].weight_tensor.item()
        if B >= 3 and T > 0:
            rip = self.ripple(rho[:,-1], theta[:,-1], sigma[:,-1])
            rip = self.theb_ripple(rip.unsqueeze(1).unsqueeze(2).expand(B, T, D))
            conditioned = conditioned + rip * nudge
        out = self.out_proj(conditioned)
        out = self.theb_output(out)
        return self.ln(residual + out)


# ─────────────────────────────────────────────────────────────────────────────
# 13. V18LMHead
# ─────────────────────────────────────────────────────────────────────────────

class V18LMHead(nn.Module):
    def __init__(self, d_model=D_MODEL, vocab_size=VOCAB, cardan_k=CARDAN_K,
                 cardan_weight=8.0, poly_degree=17, poly_harmonics=7, jaggy_scale=0.45):
        super().__init__()
        self.vocab_size = vocab_size
        self.bolyai     = BolyaiEmbedding(d_model)
        self.efference  = EfferenceKernel(d_model)
        self.v18_block  = V18Block(d_model=d_model)
        self.proj       = nn.Linear(d_model, vocab_size, bias=False)
        self.cardan     = CardanAperture(vocab_size, cardan_k, cardan_weight)
        self.poly_logit = PartitionPolynomialWarp(poly_degree, poly_harmonics, jaggy_scale, logit_mode=True)
        self.poly_prob  = PartitionPolynomialWarp(poly_degree, poly_harmonics, jaggy_scale, logit_mode=False)
        self.sawtooth   = MirroredSawtoothClip()
        self.theb_logit = MiniThebault(gamma_op=GEO["theb_gamma_ker"], dim=-1)
        self._orbit     = 0

    def set_orbit(self, orbit: int):  self._orbit = orbit % 4
    def build_cardan(self, scores):   self.cardan.build(scores)

    def forward(self, hidden, instruction_emb=None,
                temperature=1., return_probs=False):
        hidden = self.v18_block(hidden, instruction_emb=instruction_emb)
        logits = self.proj(hidden) / max(temperature, 1e-6)
        logits = self.theb_logit(logits)
        logits = self.poly_logit(logits)
        if self.cardan._built:
            logits = self.cardan(logits, self._orbit)
        if not return_probs:
            return logits
        probs = GEO["softmax"](logits)
        probs = self.poly_prob(probs)
        probs = self.sawtooth(probs)
        return probs


# ─────────────────────────────────────────────────────────────────────────────
# 14. V18TransformerLayer
# ─────────────────────────────────────────────────────────────────────────────

class V18TransformerLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, ffn_mult=4,
                 dropout=DROPOUT, **v18_kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_mult*d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult*d_model, d_model), nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.v18 = V18Block(d_model=d_model, **v18_kwargs)

    def forward(self, x, instruction_emb=None, key_padding_mask=None):
        T           = x.size(1)
        causal_mask = GEO["triu"](torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask,
                                key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return self.v18(x, instruction_emb=instruction_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 15. V18AddonWrapper
# ─────────────────────────────────────────────────────────────────────────────

class V18AddonWrapper(nn.Module):
    def __init__(self, base_model, d_model=D_MODEL, vocab_size=0,
                 cardan_k=CARDAN_K, cardan_weight=8., mirror_alpha=0.35, **v18_kwargs):
        super().__init__()
        self.base        = base_model
        self.v18_block   = V18Block(d_model=d_model, **v18_kwargs)
        self.mirror_head = MirroredInstructionHead(d_model, alpha=mirror_alpha, learnable_alpha=True)
        self.use_cardan  = vocab_size > 0
        if self.use_cardan:
            self.head   = nn.Linear(d_model, vocab_size, bias=False)
            self.cardan = CardanAperture(vocab_size, cardan_k, cardan_weight)
        else:
            self.head = self.cardan = None
        self._orbit = 0

    def set_orbit(self, orbit: int):        self._orbit = orbit % 4
    def build_cardan(self, scores):
        if self.cardan is not None: self.cardan.build(scores)

    def forward(self, x, instruction_emb=None, **base_kwargs):
        hidden = self.base(x, **base_kwargs)
        hidden = self.v18_block(hidden, instruction_emb=instruction_emb)
        if self.head is None: return hidden
        logits = self.head(hidden)
        if self.cardan is not None and self.cardan._built:
            logits = self.cardan(logits, self._orbit)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 16. V18Model  —  end-to-end LM
# ─────────────────────────────────────────────────────────────────────────────

class V18Model(nn.Module):
    """
    Full V18-GEO language model.
    All math/weight ops routed through TransmutableGeoOp (GEO registry).
    """
    def __init__(self, vocab_size: int = VOCAB, d_model: int = D_MODEL,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 max_seq_len: int = MAX_SEQ_LEN, cardan_k: int = CARDAN_K,
                 mirror_alpha: float = 0.35, dropout: float = DROPOUT):
        super().__init__()
        self.d_model     = d_model
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.tok_emb     = nn.Embedding(vocab_size, d_model)
        self.pos_emb     = nn.Embedding(max_seq_len, d_model)   # ← shape [max_seq_len, d_model]
        self.drop        = nn.Dropout(dropout)
        self.layers      = nn.ModuleList([
            V18TransformerLayer(d_model=d_model, n_heads=n_heads, ffn_mult=4,
                                dropout=dropout, spaghetti_coup=0.35,
                                ripple_decay=0.5, ripple_scale=0.5)
            for _ in range(n_layers)
        ])
        self.lm_head = V18LMHead(d_model=d_model, vocab_size=vocab_size, cardan_k=cardan_k)
        self.mirror  = MirroredInstructionHead(d_model, alpha=mirror_alpha, learnable_alpha=True)
        self._orbit  = 0

    def set_orbit(self, orbit: int):
        self._orbit = orbit % 4
        self.lm_head.set_orbit(orbit)

    def build_cardan(self, token_scores=None):
        if token_scores is None:
            token_scores = self.tok_emb.weight.norm(dim=-1)
        self.lm_head.build_cardan(token_scores)

    def encode_instruction(self, instruction_ids: torch.Tensor) -> torch.Tensor:
        ids = instruction_ids.view(-1)
        T   = min(ids.size(0), self.max_seq_len)
        ids = ids[:T]
        pos = torch.arange(T, device=ids.device)
        emb = self.tok_emb(ids) + self.pos_emb(pos)
        self.mirror.encode_instruction(emb)
        return emb.mean(dim=0)

    def forward(self, input_ids: torch.Tensor,
                instruction_ids=None, temperature=1., return_probs=False):
        B, T = input_ids.shape
        # Clamp T to max_seq_len (prevents pos_emb out-of-range)
        T   = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x   = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        instr_emb = None
        if instruction_ids is not None:
            instr_emb = self.encode_instruction(instruction_ids)
        for layer in self.layers:
            x = layer(x, instruction_emb=instr_emb)
        return self.lm_head(x, instruction_emb=instr_emb,
                            temperature=temperature, return_probs=return_probs)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens=100,
                 temperature=1., instruction_ids=None) -> torch.Tensor:
        self.eval()
        ids = prompt_ids.clone()
        for step in range(max_new_tokens):
            # Keep only last max_seq_len tokens
            ctx       = ids[:, -self.max_seq_len:]
            probs     = self.forward(ctx, instruction_ids=instruction_ids,
                                     temperature=temperature, return_probs=True)
            next_p    = probs[0, -1]
            next_p    = GEO["relu"](next_p).clamp(min=1e-12)
            next_p    = next_p / next_p.sum()
            next_id   = GEO["multinomial"](next_p)        # (1,)
            ids       = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
            self.set_orbit(step)
        return ids


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _get_data() -> str:
    with open(input("Filename: "), encoding="utf-8") as f:
        return f.read()


class Dataset(Dataset):
    """Sliding-window token dataset with shared tokenizer."""
    def __init__(self, tokens: torch.Tensor, seq_len: int = SEQ_LEN):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk  = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_datasets(seq_len: int = SEQ_LEN):
    """Download, tokenize, split 90/10 train/val. Returns (train_ds, val_ds, tokenizer)."""
    raw   = _get_data()
    tok   = WordTokenizer(vocab_size=VOCAB)
    tok.build_vocab([raw])
    tok.save(TOKENIZER_F)
    print(f"Vocabulary built: {len(tok.trigram2id):,} trigrams saved to {TOKENIZER_F}")

    tokens = tok.encode(raw)
    print(f"Dataset: {len(tokens):,} tokens")

    split     = int(0.9 * len(tokens))
    train_ds  = Dataset(tokens[:split],  seq_len=seq_len)
    val_ds    = Dataset(tokens[split:],  seq_len=seq_len)
    print(f"Train: {len(train_ds):,} windows  |  Val: {len(val_ds):,} windows")
    return train_ds, val_ds, tok


def collate_fn(batch):
    xs = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=0)
    ys = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=0)
    return xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = BATCH_GPU if device == "cuda" else BATCH_CPU
    print(f"Device: {device}  |  Batch: {batch_size}")

    train_ds, val_ds, tok = build_datasets(seq_len=SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    # Model — ALL dims use global constants (no mismatch possible)
    model = V18Model(
        vocab_size   = VOCAB,
        d_model      = D_MODEL,
        n_layers     = N_LAYERS,
        n_heads      = N_HEADS,
        max_seq_len  = MAX_SEQ_LEN,  # ← checkpoint will have pos_emb [128, D_MODEL]
        cardan_k     = CARDAN_K,
        dropout      = DROPOUT,
    ).to(device)
    model.build_cardan()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dl)*EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val = float("inf")
    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        t_loss, t_steps = 0., 0
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)                          # (B, T, V)
            loss   = criterion(logits.transpose(1, 2), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_loss  += loss.item()
            t_steps += 1
            if i % 50 == 0:
                print(f"  Epoch {epoch+1} step {i}/{len(train_dl)}  loss={loss.item():.4f}")
                break
        # ── val ──
        model.eval()
        v_loss, v_steps = 0., 0
        with torch.no_grad():
            for inputs, targets in val_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                v_loss  += criterion(logits.transpose(1, 2), targets).item()
                v_steps += 1
                if i % 50 == 0:
                    print(f"  Epoch {epoch+1} step {i}/{len(train_dl)}  loss={loss.item():.4f}")
                    break
        avg_t = t_loss/max(t_steps,1)
        avg_v = v_loss/max(v_steps,1)
        print(f"Epoch {epoch+1}/{EPOCHS}  train={avg_t:.4f}  val={avg_v:.4f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Saved best model → {CHECKPOINT}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(prompt: str = "To be or not to be",
                    max_new_tokens: int = 80, temperature: float = 1.2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(TOKENIZER_F):
        print(f"Tokenizer not found at {TOKENIZER_F}. Train first.")
        return
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found at {CHECKPOINT}. Train first.")
        return

    tok = WordTokenizer.load(TOKENIZER_F)
    print(f"Loaded tokenizer: {len(tok.trigram2id):,} trigrams")

    # Auto-detect checkpoint dims so load always matches
    state  = torch.load(CHECKPOINT, map_location=device)
    v_size = state["tok_emb.weight"].shape[0]
    d_size = state["tok_emb.weight"].shape[1]
    s_len  = state["pos_emb.weight"].shape[0]
    print(f"Checkpoint: vocab={v_size}, d_model={d_size}, max_seq_len={s_len}")

    model = V18Model(vocab_size=v_size, d_model=d_size,
                     max_seq_len=s_len).to(device)
    model.load_state_dict(state)
    model.eval()
    model.build_cardan()

    prompt_ids = tok.encode(prompt).unsqueeze(0).to(device)
    print(f"Prompt ids shape: {prompt_ids.shape}")

    with torch.no_grad():
        out_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                                 temperature=temperature)
    print("\n=== Generated ===")
    print(tok.decode(out_ids[0]))


# ─────────────────────────────────────────────────────────────────────────────
# TEXT GUI
# ─────────────────────────────────────────────────────────────────────────────

def run_gui():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(TOKENIZER_F) or not os.path.exists(CHECKPOINT):
        print("Run training first (choice 't').")
        return
    tok    = WordTokenizer.load(TOKENIZER_F)
    state  = torch.load(CHECKPOINT, map_location=device)
    v_size = state["tok_emb.weight"].shape[0]
    d_size = state["tok_emb.weight"].shape[1]
    s_len  = state["pos_emb.weight"].shape[0]
    model  = V18Model(vocab_size=v_size, d_model=d_size, max_seq_len=s_len).to(device)
    model.load_state_dict(state)
    model.eval()
    model.build_cardan()
    print("=" * 50 + "\n V18-GEO INTERACTIVE GUI\n" + "=" * 50)
    while True:
        prompt = input(">>> ").strip()
        if prompt.lower() in ("exit", "quit", "q"):
            break
        ids = tok.encode(prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=60, temperature=1.5)
        print(tok.decode(out[0]), "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    while True:
        choice = input("(t) Train   (g) Generate sample   (i) Interactive GUI\n>>> ").strip().lower()
        if choice == "t":
            train()
        elif choice == "g":
            generate_sample()
        elif choice == "i":
            run_gui()
        else:
            print("Unknown choice. Run again and enter t / g / i")
