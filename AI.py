import gc
import re
import json
import os
import queue
import shutil
import tempfile
import threading
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gradio as gr
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import HfApi, whoami, hf_hub_download

# ── Constants ─────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 50_000
SEQ_LEN     = 16
EMBED_DIM   = 128
HIDDEN_DIM  = 256
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3

BOW_VOCAB_CAP = 8_000

MODEL_FILE     = "simple_lm.pt"
TOKENIZER_FILE = "simple_tokenizer.json"

_TMP_DIR = tempfile.mkdtemp(prefix="simplelm_")

ACCEPTED_EXTENSIONS = [".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".rst", ".text"]

KERNEL_N_FREQS       = 8        # must match EfferenceKernel default; stored in checkpoint
CURVE_FOLD_HARMONICS = 4        # Fourier harmonics per fold-curve sheet; stored in checkpoint
CURVE_FOLD_SHEETS    = 3        # number of manifold sheets; stored in checkpoint
CURVE_FOLD_RBF       = 8        # angular RBF anchors for diffusion gate; stored in checkpoint

TRIGRAM_VOCAB_CAP    = 16_000   # max trigrams kept by frequency; rare → index 0
TRIGRAM_FILE         = "simple_trigrams.json"
GEOMETRY_GRID_SIZE   = 24
GEOMETRY_DECAY       = 0.96
TRIGRAM_BIAS_SCALE   = 0.35
GEOMETRY_BIAS_SCALE  = 0.25

# ── Text helpers ───────────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def _sentence_score(sentence: str) -> float:
    words   = sentence.lower().split()
    wc      = len(words)
    density = len(set(words)) / max(wc, 1)
    return wc + density


def paragraph_sort(paragraphs: List[List[str]]) -> List[List[str]]:
    return [sorted(para, key=_sentence_score) for para in paragraphs]


def split_paragraphs(text: str) -> List[List[str]]:
    raw_blocks = re.split(r"\n{2,}", text.strip())
    paragraphs: List[List[str]] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        sentences = split_sentences(block)
        if len(sentences) >= 2:
            paragraphs.append(sentences)
    return paragraphs


def prepare_training_text(text: str) -> str:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        sentences = split_sentences(text)
        if len(sentences) >= 2:
            paragraphs = [sentences]
        else:
            return text
    sorted_paragraphs = paragraph_sort(paragraphs)
    return "\n\n".join(" ".join(para) for para in sorted_paragraphs)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.t2i = {"<pad>": 0, "<unk>": 1}
        self.i2t = {0: "<pad>", 1: "<unk>"}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build(self, texts: List[str]):
        freq: dict = {}
        for text in texts:
            for tok in self.tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1
        items = sorted(freq.items(), key=lambda x: -x[1])[: self.vocab_size - 2]
        for tok, _ in items:
            idx = len(self.t2i)
            self.t2i[tok] = idx
            self.i2t[idx] = tok

    def encode(self, text: str) -> List[int]:
        return [self.t2i.get(tok, 1) for tok in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.i2t.get(int(i), "<unk>") for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.t2i, f)

    @staticmethod
    def load(path: str):
        tok = Tokenizer()
        with open(path, "r", encoding="utf-8") as f:
            tok.t2i = json.load(f)
        tok.i2t = {v: k for k, v in tok.t2i.items()}
        return tok


# ── Decoupled trigram memory ───────────────────────────────────────────────────

class TrigramMemory:
    def __init__(self, cap: int = TRIGRAM_VOCAB_CAP):
        self.cap = cap
        self.next_counts: Dict[Tuple[int, int], Dict[int, int]] = {}

    def build(self, token_ids: np.ndarray):
        ids = np.asarray(token_ids, dtype=np.int32)
        freq: Dict[Tuple[int, int, int], int] = {}
        for i in range(len(ids) - 2):
            tri = (int(ids[i]), int(ids[i + 1]), int(ids[i + 2]))
            freq[tri] = freq.get(tri, 0) + 1
        for (a, b, c), n in sorted(freq.items(), key=lambda kv: -kv[1])[: self.cap]:
            bucket = self.next_counts.setdefault((a, b), {})
            bucket[c] = bucket.get(c, 0) + n

    def bias_logits(self, prev2: int, prev1: int, vocab_size: int, device: str | torch.device) -> torch.Tensor:
        bias = torch.zeros(vocab_size, device=device)
        counts = self.next_counts.get((int(prev2), int(prev1)))
        if not counts:
            return bias
        idx = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
        vals = torch.tensor(list(counts.values()), dtype=torch.float32, device=device)
        vals = torch.log1p(vals)
        bias.scatter_(0, idx, vals)
        return bias

    def save(self, path: str):
        payload = {f"{a},{b}": bucket for (a, b), bucket in self.next_counts.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)

    @staticmethod
    def load(path: str) -> "TrigramMemory":
        tm = TrigramMemory()
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for k, bucket in raw.items():
            a, b = k.split(',')
            tm.next_counts[(int(a), int(b))] = {int(t): int(c) for t, c in bucket.items()}
        return tm


# ── Trigram vocabulary ─────────────────────────────────────────────────────────

class TrigramVocab:
    UNK = "<unk_tri>"

    def __init__(self, cap: int = TRIGRAM_VOCAB_CAP):
        self.cap   = cap
        self.tri2i: dict = {self.UNK: 0}
        self.size  = 1

    def build(self, token_ids: np.ndarray):
        freq: dict = {}
        ids = np.asarray(token_ids, dtype=np.int32)
        for i in range(1, len(ids) - 1):
            tri = (int(ids[i - 1]), int(ids[i]), int(ids[i + 1]))
            freq[tri] = freq.get(tri, 0) + 1
        for tri, _ in sorted(freq.items(), key=lambda kv: -kv[1])[: self.cap - 1]:
            if tri not in self.tri2i:
                self.tri2i[tri] = self.size
                self.size += 1

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int32)
        N   = len(ids)
        out = np.zeros(N, dtype=np.int32)
        for i in range(1, N - 1):
            tri    = (int(ids[i - 1]), int(ids[i]), int(ids[i + 1]))
            out[i] = self.tri2i.get(tri, 0)
        if N >= 3:
            out[0]     = out[1]
            out[N - 1] = out[N - 2]
        return out

    def save(self, path: str):
        payload = {self.UNK: 0}
        for k, v in self.tri2i.items():
            if isinstance(k, tuple):
                payload[f"{k[0]},{k[1]},{k[2]}"] = v
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load(path: str) -> "TrigramVocab":
        tv = TrigramVocab()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            if k == TrigramVocab.UNK:
                tv.tri2i[k] = v
            else:
                a, b, c = k.split(",")
                tv.tri2i[(int(a), int(b), int(c))] = v
        tv.size = max(tv.tri2i.values()) + 1
        return tv


UNK_ID = 1   # <unk> token index — suppressed at generated positions j ≥ l


class TextDataset(Dataset):
    """
    Dual-file-isomorphic dataset: token IDs and trigram IDs each live in their
    own memory-mapped binary file.

    Returns (x, y, contrast, diff_tok, trigrams, diff_tri) — a 6-tuple.
    """

    def __init__(
        self,
        token_ids:          np.ndarray,
        trigram_ids:        np.ndarray,
        seq_len:            int   = SEQ_LEN,
        vocab_size:         int   = VOCAB_SIZE,
        trigram_vocab_size: int   = TRIGRAM_VOCAB_CAP,
        noise_std:          float = 0.12,
        aug_prob:           float = 0.5,
    ):
        N = len(token_ids)
        assert len(trigram_ids) == N, "token and trigram arrays must have equal length"

        def _make_mmap(arr: np.ndarray, suffix: str) -> tuple[np.memmap, str]:
            fd, path = tempfile.mkstemp(dir=_TMP_DIR, suffix=suffix)
            os.close(fd)
            mm = np.memmap(path, dtype=np.int32, mode="w+", shape=(N,))
            mm[:] = arr.astype(np.int32)
            mm.flush()
            del mm
            return np.memmap(path, dtype=np.int32, mode="r", shape=(N,)), path

        self._tok_mmap, self._tok_path = _make_mmap(token_ids,   ".tok.dat")
        self._tri_mmap, self._tri_path = _make_mmap(trigram_ids, ".tri.dat")

        self.seq_len            = seq_len
        self.vocab_size         = vocab_size
        self.trigram_vocab_size = trigram_vocab_size
        self.noise_std          = noise_std
        self.aug_prob           = aug_prob

    def _spectral_perturb(self, seq: np.ndarray, vocab_cap: int) -> np.ndarray:
        X  = np.fft.rfft(seq)
        K  = len(X)
        lo = K // 2
        noise = (
            np.random.randn(K - lo) + 1j * np.random.randn(K - lo)
        ) * self.noise_std
        X[lo:] += noise
        out = np.fft.irfft(X, n=len(seq))
        return np.clip(np.round(out), 0, vocab_cap - 1).astype(np.int64)

    def __len__(self) -> int:
        return max(0, len(self._tok_mmap) - self.seq_len)

    def __getitem__(self, idx: int):
        L = self.seq_len

        x_raw  = self._tok_mmap[idx     : idx + L    ].astype(np.float32)
        y_raw  = self._tok_mmap[idx + 1 : idx + L + 1].astype(np.int64)
        tg_raw = self._tri_mmap[idx     : idx + L    ].astype(np.float32)

        x_int = (
            self._spectral_perturb(x_raw.copy(), self.vocab_size)
            if np.random.random() < self.aug_prob
            else x_raw.astype(np.int64)
        )

        mu       = x_raw.mean()
        sigma_lc = x_raw.std() + 1.0
        contrast = ((x_raw - mu) / sigma_lc).astype(np.float32)

        x_ext    = self._tok_mmap[idx : idx + L + 1].astype(np.float32)
        diff_tok = np.diff(x_ext).astype(np.float32)

        tg_int = (
            self._spectral_perturb(tg_raw.copy(), self.trigram_vocab_size)
            if np.random.random() < self.aug_prob
            else tg_raw.astype(np.int64)
        )

        tg_ext   = self._tri_mmap[idx : idx + L + 1].astype(np.float32)
        diff_tri = np.diff(tg_ext).astype(np.float32)

        return (
            torch.from_numpy(x_int),
            torch.from_numpy(y_raw),
            torch.from_numpy(contrast),
            torch.from_numpy(diff_tok),
            torch.from_numpy(tg_int),
            torch.from_numpy(diff_tri),
        )

    def close(self):
        for attr in ("_tok_mmap", "_tri_mmap"):
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        for path_attr in ("_tok_path", "_tri_path"):
            try:
                os.remove(getattr(self, path_attr))
            except (OSError, AttributeError):
                pass

    def __del__(self):
        self.close()


class BolyaiProjection(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.rho_proj   = nn.Linear(hidden_dim, 1)
        self.theta_proj = nn.Linear(hidden_dim, 1)
        self.sigma_proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor):
        rho   = F.softplus(self.rho_proj(h)).squeeze(-1)
        theta = torch.tanh(self.theta_proj(h)).squeeze(-1) * 3.141593
        sigma = F.softplus(self.sigma_proj(h)).squeeze(-1)
        return rho, theta, sigma


class CurveFold(nn.Module):
    def __init__(
        self,
        n_harmonics: int = CURVE_FOLD_HARMONICS,
        n_sheets:    int = CURVE_FOLD_SHEETS,
        n_rbf:       int = CURVE_FOLD_RBF,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_sheets    = n_sheets
        self.n_rbf       = n_rbf
        PI = 3.141593

        self.sheet_base   = nn.Parameter(torch.zeros(n_sheets))
        self.sheet_cosine = nn.Parameter(torch.zeros(n_sheets, n_harmonics))
        self.sheet_sine   = nn.Parameter(torch.zeros(n_sheets, n_harmonics))

        self.field_amp   = nn.Parameter(torch.ones(n_harmonics))
        self.field_phase = nn.Parameter(torch.zeros(n_harmonics))
        self.field_decay = nn.Parameter(torch.ones(n_harmonics))

        self.register_buffer(
            "sheet_phase",
            torch.linspace(0, 2 * PI, n_sheets + 1)[:-1],
        )

        self.log_sigma_att = nn.Parameter(torch.zeros(1))

        self.register_buffer(
            "rbf_centers",
            torch.linspace(-PI, PI, n_rbf),
        )
        self.rbf_log_tau  = nn.Parameter(torch.zeros(n_rbf))
        self.rbf_values   = nn.Parameter(torch.zeros(n_rbf))

        self.phase_coeff  = nn.Parameter(torch.zeros(n_harmonics))

    def _harmonics(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.arange(
            1, self.n_harmonics + 1, dtype=theta.dtype, device=theta.device
        )
        angles = theta.unsqueeze(-1) * k
        return torch.cos(angles), torch.sin(angles)

    def fold_radii(self, theta: torch.Tensor) -> torch.Tensor:
        cos_b, sin_b = self._harmonics(theta)
        harm = (
            torch.einsum("btk,sk->bts", cos_b, self.sheet_cosine)
            + torch.einsum("btk,sk->bts", sin_b, self.sheet_sine)
        )
        return F.softplus(self.sheet_base) + harm

    def radial_field(self, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        k      = torch.arange(
            1, self.n_harmonics + 1, dtype=theta.dtype, device=theta.device
        )
        angles = theta.unsqueeze(-1) * k + self.field_phase
        decay  = F.softplus(self.field_decay)
        radial = torch.exp(-rho.unsqueeze(-1) * decay)
        return (self.field_amp * torch.cos(angles) * radial).sum(-1)

    def diffusion_gate(self, theta: torch.Tensor) -> torch.Tensor:
        tau2   = torch.exp(2.0 * self.rbf_log_tau) + 1e-6
        diff   = theta.unsqueeze(-1) - self.rbf_centers
        kernel = torch.exp(-diff.pow(2) / (2.0 * tau2))
        values = torch.sigmoid(self.rbf_values)
        return (kernel * values).sum(-1) / (kernel.sum(-1) + 1e-6)

    def forward(
        self,
        rho:   torch.Tensor,
        theta: torch.Tensor,
        sigma: torch.Tensor,
    ):
        r_s   = self.fold_radii(theta)
        rho_e = rho.unsqueeze(-1)
        rho_p = (2.0 * r_s - rho_e).clamp(min=0.0)

        sigma_att = F.softplus(self.log_sigma_att) + 1e-4
        att_s     = torch.exp(-(rho_e - r_s).pow(2) / (2.0 * sigma_att ** 2))

        field  = self.radial_field(rho, theta)
        align  = field.unsqueeze(-1) * torch.cos(self.sheet_phase)
        w_s    = align + att_s
        alpha  = F.softmax(w_s, dim=-1)

        rho_reflected = (alpha * rho_p).sum(-1)

        g       = self.diffusion_gate(theta)
        rho_out = g * rho_reflected + (1.0 - g) * rho

        cos_b, sin_b = self._harmonics(theta)
        delta_theta  = (self.phase_coeff * sin_b).sum(-1)
        theta_out    = theta + delta_theta

        att_mean  = att_s.mean(-1)
        sigma_out = sigma * (1.0 + att_mean)

        return rho_out, theta_out, sigma_out


class EfferenceKernel(nn.Module):
    def __init__(self, out_dim: int, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 3 + 2 * 3 * n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        rho:   torch.Tensor,
        theta: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        coords = torch.stack([rho, theta, sigma], dim=-1)
        freqs  = 2.0 ** torch.arange(
            self.n_freqs, dtype=coords.dtype, device=coords.device
        )
        angles = coords.unsqueeze(-1) * freqs
        sins   = torch.sin(angles).reshape(coords.shape[0], -1)
        coses  = torch.cos(angles).reshape(coords.shape[0], -1)
        feats  = torch.cat([coords, sins, coses], dim=-1)
        return self.mlp(feats)


class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM, grid_size: int = GEOMETRY_GRID_SIZE):
        super().__init__()
        self.emb           = nn.Embedding(vocab_size, embed_dim)
        self.rnn           = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.bolyai_proj   = BolyaiProjection(hidden_dim)
        self.curve_fold    = CurveFold()
        self.kernel        = EfferenceKernel(out_dim=hidden_dim)
        self.gate          = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc            = nn.Linear(hidden_dim, vocab_size)
        self.geo_score_head = nn.Linear(3, 1)
        self.grid_size       = grid_size
        self.geo_readout     = nn.Sequential(
            nn.Linear(grid_size * grid_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def bolyai(self, x_theb: torch.Tensor):
        return self.bolyai_proj(x_theb)

    def _encode(self, x: torch.Tensor):
        B, T   = x.shape
        emb    = self.emb(x)
        out, _ = self.rnn(emb)

        rho, theta, sigma = self.bolyai(out)
        rho, theta, sigma = self.curve_fold(rho, theta, sigma)

        rho_flat   = rho.reshape(B * T)
        theta_flat = theta.reshape(B * T)
        sigma_flat = sigma.reshape(B * T)

        kfeat    = self.kernel(rho_flat, theta_flat, sigma_flat)

        out_flat = out.reshape(B * T, -1)
        fused    = self.gate(torch.cat([out_flat, kfeat], dim=-1))
        logits   = self.fc(fused).reshape(B, T, -1)
        geo_bias = self._sequential_geometry_bias(rho, theta, sigma)
        logits   = logits + GEOMETRY_BIAS_SCALE * geo_bias
        return logits, rho, theta, sigma

    def _coords_to_grid(self, rho: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        B, T = rho.shape
        G = self.grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, G, device=rho.device, dtype=rho.dtype),
            torch.linspace(-1.0, 1.0, G, device=rho.device, dtype=rho.dtype),
            indexing='ij'
        )
        xx = xx.view(1, 1, G, G)
        yy = yy.view(1, 1, G, G)
        rho_n = rho / (rho.detach().amax(dim=1, keepdim=True) + 1e-6)
        x = rho_n * torch.cos(theta)
        y = rho_n * torch.sin(theta)
        spread = sigma / (sigma.detach().amax(dim=1, keepdim=True) + 1e-6)
        spread = spread.clamp_min(0.05)
        dx2 = (xx - x.unsqueeze(-1).unsqueeze(-1)).pow(2)
        dy2 = (yy - y.unsqueeze(-1).unsqueeze(-1)).pow(2)
        den = 2.0 * spread.unsqueeze(-1).unsqueeze(-1).pow(2)
        return torch.exp(-(dx2 + dy2) / den)

    def _sequential_geometry_bias(self, rho: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor, decay: float = GEOMETRY_DECAY) -> torch.Tensor:
        fields = self._coords_to_grid(rho, theta, sigma)
        B, T, G, _ = fields.shape
        canvas = torch.zeros(B, G, G, device=fields.device, dtype=fields.dtype)
        biases = []
        for t in range(T):
            canvas = decay * canvas + fields[:, t]
            biases.append(self.geo_readout(canvas.reshape(B, -1)))
        return torch.stack(biases, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _, _ = self._encode(x)
        return logits

    def forward_full(self, x: torch.Tensor):
        return self._encode(x)

    def _pack_state(self, logits, c_rho, c_theta, c_sigma):
        logp = (
            torch.log(logits.clamp(min=1e-12))
            if logits.ndim == 1
            else logits
        )
        C = logp.shape[-1]
        if c_rho   is None: c_rho   = torch.zeros_like(logp)
        if c_theta is None: c_theta = torch.zeros_like(logp)
        if c_sigma is None: c_sigma = torch.zeros_like(logp)
        return torch.cat([logp, c_rho, c_theta, c_sigma], dim=-1), C

    def _unpack_state(self, state: torch.Tensor, C: int):
        return (
            state[..., :C],
            state[..., C:2 * C],
            state[..., 2 * C:3 * C],
            state[..., 3 * C:4 * C],
        )

    def loss(self, features: torch.Tensor, gold_indices: torch.Tensor) -> torch.Tensor:
        B, C, D = features.shape
        deltas  = self.geo_score_head(features.view(B * C, D)).view(B, C)
        probs   = F.softmax(deltas, dim=-1)
        targets = F.one_hot(gold_indices, C).float()
        return F.binary_cross_entropy(probs, targets)

    @torch.no_grad()
    def generate(
        self,
        start_ids: List[int],
        max_new_tokens: int = 30,
        device: str = "cpu",
        radius: float = 1.0,
        n_fold: int = 4,
        trigram_memory: TrigramMemory | None = None,
        trigram_bias_scale: float = TRIGRAM_BIAS_SCALE,
    ) -> List[int]:
        """
        Autoregressive generation with three constraints at every generated
        position j (where j ≥ l = len(start_ids)):

        1. <unk> suppression  (j ≥ l)
           Token UNK_ID is masked to −∞.

        2. Nucleus sampling with radius r
           Truncate to the smallest set of tokens whose cumulative probability
           ≤ r.  radius=1.0 → no truncation.

        3. Iterative abs-difference folding  (n_fold rounds)
           After nucleus truncation the sorted distribution is folded k times:

             ┌─────────────────────────────────────────────────────────┐
             │  for k in range(n_fold):                                │
             │      half  = V // 2                                     │
             │      p_new = abs( p[:half]  −  p[half:2*half] )        │
             │      p_new = p_new / p_new.sum()          # renormalise │
             │      keep only the top-half vocabulary indices          │
             └─────────────────────────────────────────────────────────┘

           Each round halves the candidate set and concentrates mass on
           tokens where the sorted-probability gradient is steepest:

             round 0  →  V       candidates (full nucleus)
             round 1  →  V/2     candidates
             round 2  →  V/4     candidates
             …
             round k  →  V/2^k  candidates

           n_fold=0 disables folding (standard nucleus sampling).
           n_fold=4 with V=50 000 leaves ~3 125 candidates before sampling.
        """
        self.eval()
        l = len(start_ids)
        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            logits = self(x)
            last   = logits[:, -1, :].clone()   # (1, V)

            if trigram_memory is not None and x.shape[1] >= 2:
                tri_bias = trigram_memory.bias_logits(
                    x[0, -2].item(), x[0, -1].item(), last.shape[-1], x.device
                )
                last = last + trigram_bias_scale * tri_bias.unsqueeze(0)

            # ── 1. Suppress <unk> at all generated positions ──────────────
            last[:, UNK_ID] = float("-inf")

            # ── 2. Nucleus / radius-r truncation ──────────────────────────
            if radius < 1.0:
                probs_sorted, sorted_idx = torch.sort(
                    F.softmax(last, dim=-1), descending=True
                )
                cumulative = torch.cumsum(probs_sorted, dim=-1)
                remove = cumulative - probs_sorted > radius
                probs_sorted[remove] = 0.0
                nucleus = torch.zeros_like(last)
                nucleus.scatter_(1, sorted_idx, probs_sorted)
                probs = nucleus / nucleus.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(last, dim=-1)

            # ── 3. Iterative abs-difference folding ───────────────────────
            #
            # Sort the (possibly nucleus-truncated) distribution descending.
            # At each of the n_fold rounds:
            #
            #   p_s   : current sorted probabilities  shape (1, W)
            #   idx_s : corresponding vocab indices   shape (1, W)
            #
            #   half       = W // 2
            #   p_top      = p_s[:, :half]            top half by mass
            #   p_bot      = p_s[:, half:2*half]      bottom half by mass
            #   folded     = |p_top − p_bot|           probability gradient
            #   p_s        = folded / folded.sum()     renormalise
            #   idx_s      = idx_s[:, :half]           discard bottom indices
            #
            # After all rounds the folded distribution is scattered back
            # to the full vocabulary tensor for multinomial sampling.
            if n_fold > 0:
                p_s, idx_s = torch.sort(probs, descending=True)  # (1, V)
                for _ in range(n_fold):
                    half = p_s.shape[-1] // 2
                    if half == 0:
                        break
                    # abs( top_half  −  bottom_half )
                    folded = torch.abs(p_s[..., :half] - p_s[..., half : 2 * half])
                    denom  = folded.sum(dim=-1, keepdim=True)
                    if denom.item() < 1e-12:
                        break
                    p_s   = folded / denom      # renormalise → valid distribution
                    idx_s = idx_s[..., :half]   # keep only top-half indices

                # Scatter folded probabilities back to full vocab positions
                probs = torch.zeros_like(last)
                probs.scatter_(1, idx_s, p_s)
                total = probs.sum(dim=-1, keepdim=True)
                if total.item() > 1e-12:
                    probs = probs / total

            next_id = torch.multinomial(probs, num_samples=1)
            x       = torch.cat([x, next_id], dim=1)

        return x[0].tolist()


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_all(model: SimpleLM, tokenizer: Tokenizer, tri_memory: TrigramMemory | None = None):
    torch.save(
        {
            "model_state":    model.state_dict(),
            "vocab_size":     len(tokenizer.t2i),
            "embed_dim":      EMBED_DIM,
            "hidden_dim":     HIDDEN_DIM,
            "seq_len":        SEQ_LEN,
            "n_freqs":        KERNEL_N_FREQS,
            "n_harmonics":    CURVE_FOLD_HARMONICS,
            "n_sheets":       CURVE_FOLD_SHEETS,
            "n_rbf":          CURVE_FOLD_RBF,
            "trigram_size":   sum(len(v) for v in tri_memory.next_counts.values()) if tri_memory is not None else 0,
        },
        MODEL_FILE,
    )
    tokenizer.save(TOKENIZER_FILE)
    if tri_memory is not None:
        tri_memory.save(TRIGRAM_FILE)


def load_all(device: str = "cpu"):
    tokenizer = Tokenizer.load(TOKENIZER_FILE)
    ckpt      = torch.load(MODEL_FILE, map_location=device)
    model     = SimpleLM(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tri_vocab = (
        TrigramMemory.load(TRIGRAM_FILE)
        if os.path.exists(TRIGRAM_FILE)
        else None
    )
    return model, tokenizer, tri_vocab


# ── Generation ────────────────────────────────────────────────────────────────

def ranked_generate(
    model: SimpleLM,
    tokenizer: Tokenizer,
    trigram_memory: TrigramMemory | None,
    prompt: str,
    length_weight: float = 0.35,
    n_samples: int = 5,
    max_new_tokens: int = 30,
    radius: float = 1.0,
    n_fold: int = 4,
    device: str = "cpu",
) -> str:
    start_ids  = tokenizer.encode(prompt) or [tokenizer.t2i["<unk>"]]
    prompt_len = len(prompt.split())

    completions = []
    for _ in range(n_samples):
        out_ids    = model.generate(
            start_ids,
            max_new_tokens=max_new_tokens,
            device=device,
            radius=radius,
            n_fold=n_fold,
            trigram_memory=trigram_memory,
        )
        completion = tokenizer.decode(out_ids)
        completions.append(completion)

    def score(c: str) -> float:
        c_len = len(c.split())
        if prompt_len == 0 and c_len == 0:
            return 1.0
        return 1.0 - abs(c_len - prompt_len) / max(c_len, prompt_len, 1)

    scored = [(score(c), c) for c in completions]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


# ── Training ──────────────────────────────────────────────────────────────────

def train_on_text(text: str, epochs: int, lr: float, batch_size: int) -> str:
    sorted_text = prepare_training_text(text)
    if not sorted_text or not sorted_text.strip():
        return "❌ No valid paragraphs found (need at least one paragraph with ≥ 2 sentences)."

    tokenizer = Tokenizer()
    tokenizer.build([sorted_text])
    token_ids = tokenizer.encode(sorted_text)

    if len(token_ids) <= SEQ_LEN:
        return f"❌ Text too short — need more than {SEQ_LEN} tokens after filtering."

    tok_arr   = np.asarray(token_ids, dtype=np.int32)
    tri_memory = TrigramMemory()
    tri_memory.build(tok_arr)
    tri_vocab = TrigramVocab()
    tri_vocab.build(tok_arr)
    trigram_ids = tri_vocab.encode(tok_arr)

    dataset = TextDataset(
        token_ids          = tok_arr,
        trigram_ids        = trigram_ids,
        seq_len            = SEQ_LEN,
        vocab_size         = len(tokenizer.t2i),
        trigram_vocab_size = tri_vocab.size,
    )
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = SimpleLM(vocab_size=len(tokenizer.t2i)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    log_lines = [
        f"Training on {device} | vocab={len(tokenizer.t2i):,} "
        f"| tokens={len(token_ids):,} | trigram-links={sum(len(v) for v in tri_memory.next_counts.values()):,}"
    ]

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device)

            logits, rho, theta, sigma = model.forward_full(xb)

            ce_loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            logp_pos = torch.log(
                F.softmax(logits, dim=-1).max(dim=-1).values.clamp(min=1e-12)
            )
            packed, C = model._pack_state(logp_pos, rho, theta, sigma)

            logp_u, c_rho_u, c_theta_u, c_sigma_u = model._unpack_state(packed, C)
            geo_feats = torch.stack([c_rho_u, c_theta_u, c_sigma_u], dim=-1)
            gold_pos  = logp_u.argmax(dim=-1)

            geo_loss = model.loss(geo_feats, gold_pos)

            loss = ce_loss + 0.1 * geo_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        log_lines.append(f"Epoch {epoch+1}/{epochs} — loss: {avg:.6f}")
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg:.6f}")

    save_all(model, tokenizer, tri_memory)
    log_lines.append(f"✅ Saved → {MODEL_FILE}, {TOKENIZER_FILE}, {TRIGRAM_FILE}")

    dataset.close()
    del model, optimizer, dataset, loader, token_ids, tok_arr, trigram_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "\n".join(log_lines)


# ── File upload helper ────────────────────────────────────────────────────────

def _extract_text_from_jsonl(raw: str) -> str:
    lines_out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                parts = [str(v) for v in obj.values() if isinstance(v, str) and v.strip()]
                lines_out.append(" ".join(parts))
            elif isinstance(obj, str):
                lines_out.append(obj)
        except json.JSONDecodeError:
            lines_out.append(line)
    return "\n".join(lines_out)


def _extract_text_from_json(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    parts: List[str] = []

    def collect(node):
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(data)
    return "\n".join(p for p in parts if p.strip())


def load_uploaded_file(file_obj) -> tuple[str, str, str]:
    if file_obj is None:
        return "", "", "No file uploaded."

    path = file_obj if isinstance(file_obj, str) else file_obj.name
    ext  = os.path.splitext(path)[1].lower()

    if ext not in ACCEPTED_EXTENSIONS:
        return (
            "", "",
            f"❌ Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(ACCEPTED_EXTENSIONS)}",
        )

    try:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    raw = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            return "", "", "❌ Could not decode file — please use a UTF-8 encoded text file."

        if ext == ".jsonl":
            text = _extract_text_from_jsonl(raw)
        elif ext == ".json":
            text = _extract_text_from_json(raw)
        elif ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            rows = []
            for line in raw.splitlines():
                cells = [c.strip().strip('"') for c in line.split(sep)]
                row_text = " ".join(c for c in cells if c)
                if row_text:
                    rows.append(row_text)
            text = "\n".join(rows)
        else:
            text = raw

        n_chars = len(text)
        n_words = len(text.split())
        fname   = os.path.basename(path)
        tmp_path = _write_tmp(text)
        del text

        preview = (
            _read_tmp(tmp_path)[:3000]
            + (f"\n\n…(preview only — {n_chars:,} chars total · full text used for training)"
               if n_chars > 3000 else "")
        )
        return preview, tmp_path, f"✅ Loaded '{fname}'  —  {n_chars:,} chars · {n_words:,} words"

    except Exception as e:
        return "", "", f"❌ Error reading file: {e}"


# ── Large-text temp-file helpers ──────────────────────────────────────────────

def _write_tmp(text: str) -> str:
    fd, path = tempfile.mkstemp(dir=_TMP_DIR, suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _read_tmp(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _delete_tmp(path: str):
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


# ── HuggingFace Dataset helpers ───────────────────────────────────────────────

def fetch_hf_configs(dataset_name: str):
    dataset_name = dataset_name.strip()
    if not dataset_name:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), "Enter a dataset name first."
    try:
        configs = get_dataset_config_names(dataset_name) or ["default"]
        return (
            gr.update(choices=configs, value=configs[0]),
            gr.update(choices=[], value=None),
            f"Found {len(configs)} config(s).",
        )
    except Exception as e:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), f"❌ {e}"


def fetch_hf_fields(dataset_name: str, config: str, split: str = "train"):
    dataset_name = dataset_name.strip()
    if not dataset_name or not config:
        return gr.update(choices=[], value=None), "Provide dataset name and config."
    try:
        cfg        = None if config in ("default", "") else config
        ds         = load_dataset(dataset_name, cfg, split=split, streaming=True, trust_remote_code=False)
        example    = next(iter(ds))
        str_fields = [k for k, v in example.items() if isinstance(v, str)] or list(example.keys())
        return gr.update(choices=str_fields, value=str_fields[0] if str_fields else None), f"Fields: {str_fields}"
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ {e}"


# ── Multithreaded HF dataset loading ─────────────────────────────────────────

_STOP = object()
_cancel_event = threading.Event()


def _hf_loader_worker(
    dataset_name, config, split, text_field, max_samples, result_queue, cancel_event
):
    PROGRESS_EVERY = 50
    try:
        cfg = None if config in ("default", "") else config
        ds  = load_dataset(dataset_name, cfg, split=split, streaming=True, trust_remote_code=False)

        texts: List[str] = []
        for i, example in enumerate(ds):
            if cancel_event.is_set():
                result_queue.put(("error", "⚠️ Loading cancelled by user."))
                return
            if i >= max_samples:
                break
            val = example.get(text_field, "")
            if isinstance(val, str) and val.strip():
                texts.append(val.strip())
            if i % PROGRESS_EVERY == 0:
                result_queue.put(("progress", i, len(texts)))

        full_text = "\n\n".join(texts)
        del texts
        tmp_path  = _write_tmp(full_text)
        n_chars, n_words = len(full_text), len(full_text.split())
        del full_text
        result_queue.put(("done", tmp_path, n_chars, n_words))

    except Exception as exc:
        result_queue.put(("error", f"❌ Error loading dataset: {exc}"))


def ui_load_hf_and_preview(dataset_name, config, split, text_field, max_samples):
    global _cancel_event
    dataset_name = dataset_name.strip()
    if not dataset_name or not text_field:
        yield "", "⚠️ Provide a dataset name and select a text field first."
        return

    _cancel_event = threading.Event()
    result_queue: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=_hf_loader_worker,
        args=(dataset_name, config, split, text_field, int(max_samples), result_queue, _cancel_event),
        daemon=True, name="hf-dataset-loader",
    )
    worker.start()

    while True:
        try:
            msg = result_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        kind = msg[0]
        if kind == "progress":
            _, scanned, collected = msg
            yield "", f"⏳ Loading…  {scanned:,} rows scanned · {collected:,} texts collected"
        elif kind == "done":
            tmp_path, n_chars, n_words = msg[1], msg[2], msg[3]
            preview = (
                _read_tmp(tmp_path)[:2000]
                + (f"\n…(truncated)" if n_chars > 2000 else "")
                + f"\n\n✅ Done — {n_chars:,} chars · {n_words:,} words"
            )
            yield tmp_path, preview
            break
        elif kind == "error":
            yield "", msg[1]
            break
    worker.join(timeout=1)


def ui_cancel_hf_load():
    _cancel_event.set()
    return "⚠️ Cancel requested — stopping after current row…"


# ── HuggingFace Hub push / pull ───────────────────────────────────────────────

def validate_token(hf_token: str) -> str:
    hf_token = hf_token.strip()
    if not hf_token:
        return "—"
    try:
        info = whoami(token=hf_token)
        return f"✅ Logged in as: {info['name']}"
    except Exception as e:
        return f"❌ Invalid token: {e}"


def push_to_hub(repo_id: str, hf_token: str, commit_message: str, private: bool) -> str:
    repo_id        = repo_id.strip()
    hf_token       = hf_token.strip()
    commit_message = commit_message.strip() or "Upload SimpleLM checkpoint"
    if not repo_id:
        return "❌ Provide a repo ID, e.g.  your-username/my-simple-lm"
    if not hf_token:
        return "❌ Provide a HuggingFace token with write access."
    if not os.path.exists(MODEL_FILE) or not os.path.exists(TOKENIZER_FILE):
        return "❌ No saved model found locally — train first."
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        card = f"""---
language: en
tags:
  - text-generation
  - gru
  - pytorch
---
# SimpleLM — {repo_id}

## Architecture
| Param | Value |
|-------|-------|
| Embedding dim | {EMBED_DIM} |
| Hidden dim | {HIDDEN_DIM} |
| Sequence length | {SEQ_LEN} |
| Curve fold harmonics | {CURVE_FOLD_HARMONICS} |

## Usage
```python
from app import load_all, ranked_generate
model, tokenizer = load_all()
print(ranked_generate(model, tokenizer, "The quick brown"))
```
"""
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(card); card_path = f.name
        for local, remote in [(MODEL_FILE, MODEL_FILE), (TOKENIZER_FILE, TOKENIZER_FILE), (card_path, "README.md")]:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=repo_id, repo_type="model", commit_message=commit_message)
        os.unlink(card_path)
        visibility = "private" if private else "public"
        return (f"✅ Pushed to https://huggingface.co/{repo_id}  [{visibility}]\n\n"
                f"Uploaded:\n  • {MODEL_FILE}\n  • {TOKENIZER_FILE}\n  • README.md")
    except Exception as e:
        return f"❌ Upload failed: {e}"


def pull_from_hub(repo_id: str, hf_token: str) -> str:
    global _model, _tokenizer, _tri_memory
    repo_id  = repo_id.strip()
    hf_token = hf_token.strip() or None
    if not repo_id:
        return "❌ Provide a repo ID."
    try:
        kwargs     = dict(repo_id=repo_id, repo_type="model", token=hf_token)
        model_path = hf_hub_download(filename=MODEL_FILE,     **kwargs)
        tok_path   = hf_hub_download(filename=TOKENIZER_FILE, **kwargs)
        shutil.copy(model_path, MODEL_FILE)
        shutil.copy(tok_path,   TOKENIZER_FILE)
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
        n_params = sum(p.numel() for p in _model.parameters())
        return f"✅ Loaded from {repo_id}\nVocab: {len(_tokenizer.t2i):,} tokens | Params: {n_params:,}"
    except Exception as e:
        return f"❌ Download failed: {e}"


# ── App state ─────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model: SimpleLM | None      = None
_tokenizer: Tokenizer | None = None
_tri_memory: TrigramMemory | None = None

AUTO_MODEL_REPO = "trainman999/Thinking-lite"

def _try_load():
    global _model, _tokenizer, _tri_memory
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
        return "✅ Existing local model loaded."
    print(f"No local model found. Downloading from {AUTO_MODEL_REPO} …")
    try:
        model_path = hf_hub_download(filename=MODEL_FILE,     repo_id=AUTO_MODEL_REPO, repo_type="model")
        tok_path   = hf_hub_download(filename=TOKENIZER_FILE, repo_id=AUTO_MODEL_REPO, repo_type="model")
        shutil.copy(model_path, MODEL_FILE)
        shutil.copy(tok_path,   TOKENIZER_FILE)
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
        n_params = sum(p.numel() for p in _model.parameters())
        return f"✅ Auto-loaded {AUTO_MODEL_REPO}  ({n_params:,} params)"
    except Exception as e:
        return f"⚠️ Could not download default model: {e}"


_startup_msg = _try_load()


def ui_train(text: str, hf_path: str, file_path: str, epochs: int, lr: float, batch_size: int):
    global _model, _tokenizer, _tri_memory
    training_source = _read_tmp(file_path) or _read_tmp(hf_path) or text
    if not training_source or not training_source.strip():
        return "❌ No text provided."
    result = train_on_text(training_source, int(epochs), float(lr), int(batch_size))
    del training_source
    gc.collect()
    if "✅" in result:
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
    return result


def ui_generate(prompt: str, max_new_tokens: int, n_samples: int, radius: float, n_fold: int):
    if _model is None or _tokenizer is None:
        return "❌ No model loaded. Train one first."
    if not prompt.strip():
        return "❌ Enter a prompt."
    return ranked_generate(
        _model, _tokenizer, _tri_memory, prompt,
        n_samples=int(n_samples),
        max_new_tokens=int(max_new_tokens),
        radius=float(radius),
        n_fold=int(n_fold),
        device=DEVICE,
    )


def model_info() -> str:
    lines = [f"Device: {DEVICE}"]
    if _model is not None and _tokenizer is not None:
        n_params = sum(p.numel() for p in _model.parameters())
        fold     = _model.curve_fold
        lines += [
            f"Vocab size:       {len(_tokenizer.t2i):,}",
            f"Parameters:       {n_params:,}",
            f"Embed dim:        {EMBED_DIM}",
            f"Hidden dim:       {HIDDEN_DIM}",
            f"Seq length:       {SEQ_LEN}",
            f"Fold harmonics:   {fold.n_harmonics}",
            f"Fold sheets:      {fold.n_sheets}",
            f"Fold RBF anchors: {fold.n_rbf}",
        ]
    else:
        lines.append("No model loaded yet.")
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="SimpleLM — Train & Generate",
    theme=gr.themes.Base(
        primary_hue="slate",
        secondary_hue="zinc",
        neutral_hue="zinc",
        font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ),
    css="""
    .startup-msg  { font-size: 0.8rem; color: #71717a; }
    .upload-status{ font-size: 0.8rem; }
    """,
) as demo:

    gr.Markdown(
        """
# 🧠 SimpleLM — GRU Language Model
Train a compact GRU language model on your own text or a Hugging Face dataset, then generate completions.
        """
    )
    gr.Markdown(f"**Runtime:** `{DEVICE.upper()}` | {_startup_msg}", elem_classes="startup-msg")

    with gr.Tabs():

        # ── Tab 1: HuggingFace Dataset ─────────────────────────────────────
        with gr.Tab("📦 HF Dataset"):
            gr.Markdown("### Load text from a Hugging Face dataset")
            with gr.Row():
                hf_name  = gr.Textbox(label="Dataset name", placeholder="wikitext", scale=3)
                hf_split = gr.Textbox(label="Split", value="train", scale=1)
            with gr.Row():
                btn_configs = gr.Button("1 — Fetch configs", variant="secondary")
                hf_status   = gr.Textbox(label="Status", interactive=False, scale=3)
            with gr.Row():
                hf_config  = gr.Dropdown(label="Config", choices=[], interactive=True, scale=2)
                btn_fields = gr.Button("2 — Fetch fields", variant="secondary", scale=1)
            hf_field = gr.Dropdown(label="Text field", choices=[], interactive=True)
            hf_max   = gr.Slider(label="Max samples", minimum=10, maximum=500000000, value=500, step=10)
            with gr.Row():
                btn_load_hf   = gr.Button("3 — Load text into editor ⬇", variant="primary", scale=4)
                btn_cancel_hf = gr.Button("✖ Cancel", variant="stop", scale=1)
            hf_preview = gr.Textbox(label="Preview / progress", lines=6, interactive=False)
            hf_full    = gr.State("")

            btn_configs.click(fetch_hf_configs, [hf_name], [hf_config, hf_field, hf_status])
            btn_fields.click(fetch_hf_fields, [hf_name, hf_config, hf_split], [hf_field, hf_status])
            btn_load_hf.click(ui_load_hf_and_preview,
                              inputs=[hf_name, hf_config, hf_split, hf_field, hf_max],
                              outputs=[hf_full, hf_preview])
            btn_cancel_hf.click(ui_cancel_hf_load, outputs=[hf_preview])

        # ── Tab 2: Train ────────────────────────────────────────────────────
        with gr.Tab("🏋 Train"):
            gr.Markdown(
                "### Train on text\n"
                "_Paragraphs sorted so **a\\[i\\] ≤ a\\[i+1\\]** (word-count + density). "
                "Paragraphs with k < 2 sentences are dropped._"
            )
            with gr.Group():
                gr.Markdown(f"#### 📂 Upload a training file\n_Accepted: {', '.join(ACCEPTED_EXTENSIONS)}_")
                with gr.Row():
                    upload_file   = gr.File(label="Drop or click to upload",
                                            file_types=ACCEPTED_EXTENSIONS,
                                            file_count="single", scale=4)
                    btn_load_file = gr.Button("Load file into editor ⬇", variant="primary", scale=1)
                upload_status = gr.Textbox(label="File status", interactive=False,
                                           value="—", elem_classes="upload-status")

            gr.Markdown("---")
            file_full  = gr.State("")
            train_text = gr.Textbox(label="Training text", placeholder="Paste text here…", lines=12)

            with gr.Row():
                btn_use_hf = gr.Button("⬆  Use HuggingFace text loaded above", variant="secondary")
                btn_clear  = gr.Button("🗑  Clear editor", variant="secondary")

            btn_use_hf.click(
                lambda p: (lambda t: t[:3000] + f"\n\n…({len(t):,} chars total)"
                           if len(t) > 3000 else t)(_read_tmp(p)),
                [hf_full], [train_text],
            )
            btn_clear.click(
                lambda hp, fp: (_delete_tmp(hp), _delete_tmp(fp), "", "", "", "—")[2:],
                inputs=[hf_full, file_full],
                outputs=[train_text, hf_full, file_full, upload_status],
            )
            btn_load_file.click(load_uploaded_file, [upload_file], [train_text, file_full, upload_status])
            upload_file.change(load_uploaded_file,  [upload_file], [train_text, file_full, upload_status])

            gr.Markdown("---")
            with gr.Row():
                t_epochs = gr.Slider(label="Epochs",     minimum=1,    maximum=50,   value=EPOCHS,     step=1)
                t_lr     = gr.Slider(label="LR",         minimum=1e-5, maximum=1e-1, value=LR,         step=1e-5)
                t_bs     = gr.Slider(label="Batch size", minimum=8,    maximum=256,  value=BATCH_SIZE,  step=8)
            btn_train = gr.Button("Train 🚀", variant="primary")
            train_log = gr.Textbox(label="Training log", lines=14, interactive=False)
            btn_train.click(ui_train, [train_text, hf_full, file_full, t_epochs, t_lr, t_bs], [train_log])

        # ── Tab 3: Generate ─────────────────────────────────────────────────
        with gr.Tab("✨ Generate"):
            gr.Markdown(
                "### Generate text completions\n"
                "_`<unk>` suppressed at all generated positions. "
                "**Radius r** = nucleus mass (1.0 = off). "
                "**Fold iterations** = rounds of abs-difference sharpening "
                "(each halves the candidate set; 0 = off)._"
            )
            gen_prompt = gr.Textbox(label="Prompt", placeholder="The quick brown fox…", lines=2)
            with gr.Row():
                gen_tokens  = gr.Slider(label="Max new tokens",              minimum=10,  maximum=1300, value=300,  step=5)
                gen_samples = gr.Slider(label="Samples (ranked)",            minimum=1,   maximum=20,   value=5,    step=1)
                gen_radius  = gr.Slider(label="Radius r (nucleus)",          minimum=0.1, maximum=1.0,  value=0.6,  step=0.05)
                gen_n_fold  = gr.Slider(label="Fold iterations (abs-diff)",  minimum=0,   maximum=10,   value=4,    step=1)
            btn_gen    = gr.Button("Generate ✨", variant="primary")
            gen_output = gr.Textbox(label="Completion", lines=6, interactive=False)
            btn_gen.click(
                ui_generate,
                [gen_prompt, gen_tokens, gen_samples, gen_radius, gen_n_fold],
                [gen_output],
            )

        # ── Tab 4: HuggingFace Hub ──────────────────────────────────────────
        with gr.Tab("🤗 HF Hub"):
            gr.Markdown("### Push your trained model to the Hugging Face Hub")
            with gr.Group():
                gr.Markdown("#### 🔑 Authentication")
                with gr.Row():
                    hub_token    = gr.Textbox(label="HF Token (write access)", placeholder="hf_…",
                                              type="password", scale=4)
                    btn_validate = gr.Button("Validate token", variant="secondary", scale=1)
                token_status = gr.Textbox(label="Token status", interactive=False, value="—")
                btn_validate.click(validate_token, [hub_token], [token_status])
                gr.Markdown("_[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)_")
            gr.Markdown("---")
            with gr.Group():
                gr.Markdown("#### ⬆  Push model")
                with gr.Row():
                    push_repo    = gr.Textbox(label="Repo ID", placeholder="your-username/my-simple-lm", scale=4)
                    push_private = gr.Checkbox(label="Private repo", value=False, scale=1)
                push_commit = gr.Textbox(label="Commit message", value="Upload SimpleLM checkpoint")
                btn_push    = gr.Button("Push to Hub ⬆", variant="primary")
                push_status = gr.Textbox(label="Push status", lines=5, interactive=False)
                btn_push.click(push_to_hub, [push_repo, hub_token, push_commit, push_private], [push_status])
            gr.Markdown("---")
            with gr.Group():
                gr.Markdown("#### ⬇  Load model from Hub")
                pull_repo   = gr.Textbox(label="Repo ID", placeholder="your-username/my-simple-lm")
                btn_pull    = gr.Button("Load from Hub ⬇", variant="secondary")
                pull_status = gr.Textbox(label="Load status", lines=4, interactive=False)
                btn_pull.click(pull_from_hub, [pull_repo, hub_token], [pull_status])

        # ── Tab 5: Model info ───────────────────────────────────────────────
        with gr.Tab("ℹ Model info"):
            info_box = gr.Textbox(label="Model stats", lines=8, interactive=False, value=model_info())
            gr.Button("Refresh").click(model_info, outputs=[info_box])


if __name__ == "__main__":
    demo.launch(share=False)
