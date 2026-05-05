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

KERNEL_N_FREQS       = 8
CURVE_FOLD_HARMONICS = 4
CURVE_FOLD_SHEETS    = 3
CURVE_FOLD_RBF       = 8

TRIGRAM_VOCAB_CAP   = 16_000
TRIGRAM_FILE        = "simple_trigrams.json"
GEOMETRY_GRID_SIZE  = 24
GEOMETRY_DECAY      = 0.96
TRIGRAM_BIAS_SCALE  = 0.35
GEOMETRY_BIAS_SCALE = 0.25

# ── NNVC constants ─────────────────────────────────────────────────────────────
NNVC_QUANT_STEP = 0.1    # quantisation step size Δ
NNVC_LAMBDA     = 0.01   # Lagrangian rate-distortion weight λ
NNVC_HYPER_DIM  = 64     # hyperprior latent dimension


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


# ── Trigram memory ─────────────────────────────────────────────────────────────

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

    def bias_logits(self, prev2: int, prev1: int, vocab_size: int, device) -> torch.Tensor:
        bias = torch.zeros(vocab_size, device=device)
        counts = self.next_counts.get((int(prev2), int(prev1)))
        if not counts:
            return bias
        idx  = torch.tensor(list(counts.keys()),   dtype=torch.long,    device=device)
        vals = torch.tensor(list(counts.values()), dtype=torch.float32, device=device)
        bias.scatter_(0, idx, torch.log1p(vals))
        return bias

    def save(self, path: str):
        payload = {f"{a},{b}": bucket for (a, b), bucket in self.next_counts.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load(path: str) -> "TrigramMemory":
        tm = TrigramMemory()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, bucket in raw.items():
            a, b = k.split(",")
            tm.next_counts[(int(a), int(b))] = {int(t): int(c) for t, c in bucket.items()}
        return tm


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


UNK_ID = 1


class TextDataset(Dataset):
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
        assert len(trigram_ids) == N

        def _make_mmap(arr: np.ndarray, suffix: str):
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
        noise = (np.random.randn(K - lo) + 1j * np.random.randn(K - lo)) * self.noise_std
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
            if np.random.random() < self.aug_prob else x_raw.astype(np.int64)
        )

        mu       = x_raw.mean()
        sigma_lc = x_raw.std() + 1.0
        contrast = ((x_raw - mu) / sigma_lc).astype(np.float32)
        x_ext    = self._tok_mmap[idx : idx + L + 1].astype(np.float32)
        diff_tok = np.diff(x_ext).astype(np.float32)

        tg_int = (
            self._spectral_perturb(tg_raw.copy(), self.trigram_vocab_size)
            if np.random.random() < self.aug_prob else tg_raw.astype(np.int64)
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
            try: delattr(self, attr)
            except AttributeError: pass
        for path_attr in ("_tok_path", "_tri_path"):
            try: os.remove(getattr(self, path_attr))
            except (OSError, AttributeError): pass

    def __del__(self):
        self.close()


# ════════════════════════════════════════════════════════════════════════════════
#  NNVC — Neural Network-based Video Coding algorithm stack
#
#  Implements the four core NNVC components as a unified codec module:
#
#  1. NNVCIntraPredictor   — Causal intra prediction; encode only the residual
#  2. NNVCHyperprior       — Scale hyperprior entropy model (Ballé et al. 2018)
#  3. NNVCContextModel     — Autoregressive causal context model
#  4. NNVCInLoopFilter     — Depth-wise separable residual post-filter (ALF-style)
#
#  Encoding path:
#    x → IntraPredict → residual → g_a → y → Q(y) → ŷ
#      → HyperpriorEntropy(y,ŷ) → σ(ẑ)
#      → ContextModel(ŷ)        → (μ_ctx, σ_ctx)
#      → GaussianRate(ŷ, μ, σ)  → R_main
#
#  Decoding path:
#    ŷ → g_s → x̂ → InLoopFilter → x̂_filt + intra_pred → x_out
#
#  Loss:  D + λ·(R_main + R_hyper)
# ════════════════════════════════════════════════════════════════════════════════

def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator: round() forward, identity backward."""
    return x + (x.round() - x).detach()


def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian CDF  Φ(x) via erf."""
    return 0.5 * (1.0 + torch.erf(x * (2.0 ** -0.5)))


def _gaussian_rate(
    y_hat: torch.Tensor,
    mu:    torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate bits per element under a Gaussian entropy model.

    Uses the probability mass of quantisation bin [n-½, n+½]:
        p(n) = Φ((n - μ + 0.5) / σ) − Φ((n - μ − 0.5) / σ)
        R    = −log₂ p(ŷ)          averaged over all elements

    This is the same rate term used in learned image/video compression
    (Ballé et al. 2017 → NNVC standardisation).
    """
    sigma  = sigma.clamp(min=1e-6)
    upper  = _gaussian_cdf((y_hat - mu + 0.5) / sigma)
    lower  = _gaussian_cdf((y_hat - mu - 0.5) / sigma)
    return -torch.log2((upper - lower).clamp(min=1e-12)).mean()


# ── 1. NNVC Intra Prediction ──────────────────────────────────────────────────

class NNVCIntraPredictor(nn.Module):
    """
    NNVC Intra Prediction adapted for 1-D embedding sequences.

    In video coding, intra prediction estimates the current block from
    spatially adjacent, already-decoded blocks so only the RESIDUAL
    (actual − predicted) must be entropy-coded.

    Here the "spatial neighbours" are the causally preceding embeddings
    in the token sequence.  A strictly causal convolution (left-padded,
    kernel size 3) predicts embedding[t] from embedding[t−2..t−1].
    The residual is what flows into the transform encoder (g_a).

    On the decoder side the same prediction is added back after g_s,
    exactly mirroring the NNVC intra loop.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.causal_conv = nn.Sequential(
            # kernel_size=3 + left-pad 2 → strictly causal (no future leakage)
            nn.Conv1d(dim, dim, kernel_size=3, padding=0),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            residual   : x − prediction  (fed to analysis transform g_a)
            prediction : causal estimate (added back after synthesis g_s)
        """
        h          = F.pad(x.transpose(1, 2), (2, 0))   # (B, dim, T+2) left-causal pad
        prediction = self.causal_conv(h).transpose(1, 2) # (B, T, dim)
        prediction = self.norm(prediction)
        return x - prediction, prediction


# ── 2. NNVC Scale Hyperprior ──────────────────────────────────────────────────

class NNVCHyperprior(nn.Module):
    """
    Scale Hyperprior entropy model — the core of NNVC rate estimation.

    Reference: Ballé et al. "Variational image compression with a scale
    hyperprior" (ICLR 2018); adopted into NNVC by JVET.

    Two-level hierarchy:
      • Main latent  y  (output of analysis transform g_a)
      • Hyper-latent z  (side-channel carrying per-element scale info)

    Encoder:
        z    = h_a(|y|)          — hyper-analysis (encodes scale structure)
        ẑ    = Q(z)              — quantise z (STE)

    Decoder:
        σ    = softplus(h_s(ẑ)) — predicted Gaussian scale for each y element
        R    ≈ −log₂ p(ŷ | σ)   — bits for the main latent under N(0, σ²)

    R_hyper is estimated with an L1 proxy (ẑ would itself need another
    entropy model; L1 is the standard differentiable stand-in).
    """
    def __init__(self, latent_dim: int, hyper_dim: int = NNVC_HYPER_DIM):
        super().__init__()
        # h_a : y → z  (hyper-analysis)
        self.h_a = nn.Sequential(
            nn.Linear(latent_dim, hyper_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hyper_dim, hyper_dim),
        )
        # h_s : ẑ → σ  (hyper-synthesis; produces per-element scale params)
        self.h_s = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hyper_dim, latent_dim),
        )

    def forward(
        self,
        y:     torch.Tensor,
        y_hat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y     : (B, T, latent_dim) — continuous latent (pre-quantise)
            y_hat : (B, T, latent_dim) — quantised latent
        Returns:
            sigma      : (B, T, latent_dim) — Gaussian scale params for entropy model
            R_hyper    : scalar             — differentiable rate proxy for z
        """
        z         = self.h_a(y.abs())                       # (B, T, hyper_dim)
        z_hat     = _ste_round(z)                           # Q(z) with STE
        sigma     = F.softplus(self.h_s(z_hat)) + 1e-6     # σ > 0
        R_hyper   = z_hat.abs().mean()                      # L1 proxy for H(ẑ)
        return sigma, R_hyper


# ── 3. NNVC Autoregressive Context Model ─────────────────────────────────────

class NNVCContextModel(nn.Module):
    """
    Autoregressive Context Model for entropy coding.

    In NNVC (and its image-compression precursors), a context model
    conditions the entropy distribution on already-decoded neighbours,
    reducing the rate beyond what the hyperprior alone achieves.

    Implementation: a masked (causal) 1-D convolution over the quantised
    latent sequence ŷ.  Outputs a (μ_ctx, σ_ctx) correction that is
    combined with the hyperprior's σ to form the final entropy model.

    The convolution is strictly causal (left-pad 2, kernel 3) so no
    future latent values are used — matching the NNVC coding order.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.masked_conv = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv1d(latent_dim * 2, latent_dim * 2, kernel_size=1),
        )

    def forward(self, y_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y_hat : (B, T, latent_dim) — quantised latent
        Returns:
            mu_ctx    : (B, T, latent_dim) — context mean
            sigma_ctx : (B, T, latent_dim) — context scale (> 0)
        """
        h         = F.pad(y_hat.transpose(1, 2), (2, 0))   # causal left-pad
        out       = self.masked_conv(h)                      # (B, 2·dim, T)
        mu_c, lsig = out.chunk(2, dim=1)
        return mu_c.transpose(1, 2), F.softplus(lsig.transpose(1, 2)) + 1e-6


# ── 4. NNVC In-Loop Filter (ALF-style) ───────────────────────────────────────

class NNVCInLoopFilter(nn.Module):
    """
    NNVC In-Loop Filter — learned post-reconstruction artifact removal.

    In the NNVC/VVC standard the Adaptive Loop Filter (ALF) applies a
    Wiener-filter-inspired CNN after reconstruction to suppress the
    ringing and blocking artifacts introduced by quantisation.

    This implementation mirrors ALF's depth-wise + point-wise (separable)
    convolution structure:
      DW-conv(5) → PW-conv(1) → GELU → DW-conv(3) → PW-conv(1)

    Applied as a residual over the synthesis output; a LayerNorm
    stabilises training (equivalent to ALF's gain normalisation).
    """
    def __init__(self, dim: int):
        super().__init__()
        # ALF-style depth-wise separable conv blocks
        self.dw1  = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)  # DW
        self.pw1  = nn.Conv1d(dim, dim, kernel_size=1)                          # PW
        self.dw2  = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # DW
        self.pw2  = nn.Conv1d(dim, dim, kernel_size=1)                          # PW
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, dim) → filtered (B, T, dim)"""
        h = x.transpose(1, 2)                  # (B, dim, T)
        h = F.gelu(self.pw1(self.dw1(h)))      # block 1
        h = self.pw2(self.dw2(h))              # block 2
        return self.norm(x + h.transpose(1, 2))  # residual + norm


# ── NNVC Codec: full pipeline ─────────────────────────────────────────────────

class NNVCCodec(nn.Module):
    """
    Full NNVC algorithm stack applied to embedding tensors.

    Encoding path
    ─────────────
    x  ──► NNVCIntraPredictor  ──► residual
        ──► g_a (analysis transform)  ──► y
        ──► Q(y) with STE  ──► ŷ
        ──► NNVCHyperprior(y, ŷ)  ──► σ(ẑ),  R_hyper
        ──► NNVCContextModel(ŷ)   ──► μ_ctx, σ_ctx
        ──► combined entropy model  ──► R_main

    Decoding path
    ─────────────
    ŷ  ──► g_s (synthesis transform)  ──► x̂
        ──► NNVCInLoopFilter  ──► x̂_filt
        ──► x̂_filt + intra_prediction  ──► x_out (normalised)

    R-D objective
    ─────────────
    L = D(x, x_out)  +  λ · (R_main + R_hyper)

    where D is MSE distortion and R is estimated via Gaussian CDF
    (the standard NNVC rate proxy).
    """

    def __init__(
        self,
        dim:        int,
        hyper_dim:  int   = NNVC_HYPER_DIM,
        quant_step: float = NNVC_QUANT_STEP,
    ):
        super().__init__()
        self.quant_step = quant_step

        # ── NNVC components ──────────────────────────────────────────────────
        self.intra_pred    = NNVCIntraPredictor(dim)       # 1. Intra prediction
        self.g_a           = nn.Sequential(                # 2a. Analysis transform g_a
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.g_s           = nn.Sequential(                # 2b. Synthesis transform g_s
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.hyperprior    = NNVCHyperprior(dim, hyper_dim)   # 3. Scale hyperprior
        self.context_model = NNVCContextModel(dim)             # 4. Context model
        self.in_loop       = NNVCInLoopFilter(dim)             # 5. In-loop filter
        self.norm          = nn.LayerNorm(dim)

    def _quantise(self, y: torch.Tensor) -> torch.Tensor:
        """Uniform scalar quantiser with STE gradient."""
        scaled = y / self.quant_step
        q      = scaled.round() * self.quant_step
        return y + (q - y).detach()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x       : (B, T, dim) — raw embedding tensor from nn.Embedding
        Returns:
            x_out   : (B, T, dim) — coded + filtered embedding
            rd_loss : scalar      — Lagrangian D + λ(R_main + R_hyper)
        """
        # ── Encoding path ────────────────────────────────────────────────────

        # 1. Intra prediction: only the residual is transformed
        residual, intra_pred = self.intra_pred(x)

        # 2. Analysis transform g_a  →  latent y
        y = self.g_a(residual)

        # 3. Uniform scalar quantisation  →  ŷ
        y_hat = self._quantise(y)

        # 4. Scale hyperprior: h_a(|y|) → z → ẑ → σ
        #    Returns per-element Gaussian scales and hyperlatent rate proxy
        sigma_hyper, R_hyper = self.hyperprior(y, y_hat)

        # 5. Autoregressive context model: masked-conv over ŷ → (μ_ctx, σ_ctx)
        mu_ctx, sigma_ctx = self.context_model(y_hat)

        # 6. Fuse hyperprior + context into combined entropy parameters
        #    Harmonic-mean fusion of the two scale estimates (standard in NNVC
        #    joint hyperprior + context implementations)
        sigma_combined = (
            (sigma_hyper * sigma_ctx) / (sigma_hyper + sigma_ctx + 1e-6) * 2.0
        )

        # 7. Gaussian CDF rate estimate  R_main = −log₂ p(ŷ | μ_ctx, σ_combined)
        R_main = _gaussian_rate(y_hat, mu_ctx, sigma_combined)

        # ── Decoding path ────────────────────────────────────────────────────

        # 8. Synthesis transform g_s
        x_hat = self.g_s(y_hat)

        # 9. In-loop filter (ALF-style depth-wise separable residual CNN)
        x_hat = self.in_loop(x_hat)

        # 10. Add intra prediction back (decoder loop-closure)
        x_out = self.norm(intra_pred + x_hat)

        # ── R-D objective ─────────────────────────────────────────────────────
        distortion = F.mse_loss(x_out, x)
        rd_loss    = distortion + NNVC_LAMBDA * (R_main + R_hyper)

        return x_out, rd_loss


# ── Geometry modules ──────────────────────────────────────────────────────────

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
    def __init__(self, n_harmonics=CURVE_FOLD_HARMONICS, n_sheets=CURVE_FOLD_SHEETS, n_rbf=CURVE_FOLD_RBF):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_sheets    = n_sheets
        self.n_rbf       = n_rbf
        PI = 3.141593

        self.sheet_base   = nn.Parameter(torch.zeros(n_sheets))
        self.sheet_cosine = nn.Parameter(torch.zeros(n_sheets, n_harmonics))
        self.sheet_sine   = nn.Parameter(torch.zeros(n_sheets, n_harmonics))
        self.field_amp    = nn.Parameter(torch.ones(n_harmonics))
        self.field_phase  = nn.Parameter(torch.zeros(n_harmonics))
        self.field_decay  = nn.Parameter(torch.ones(n_harmonics))
        self.register_buffer("sheet_phase", torch.linspace(0, 2 * PI, n_sheets + 1)[:-1])
        self.log_sigma_att = nn.Parameter(torch.zeros(1))
        self.register_buffer("rbf_centers", torch.linspace(-PI, PI, n_rbf))
        self.rbf_log_tau   = nn.Parameter(torch.zeros(n_rbf))
        self.rbf_values    = nn.Parameter(torch.zeros(n_rbf))
        self.phase_coeff   = nn.Parameter(torch.zeros(n_harmonics))

    def _harmonics(self, theta):
        k      = torch.arange(1, self.n_harmonics + 1, dtype=theta.dtype, device=theta.device)
        angles = theta.unsqueeze(-1) * k
        return torch.cos(angles), torch.sin(angles)

    def fold_radii(self, theta):
        cos_b, sin_b = self._harmonics(theta)
        harm = (torch.einsum("btk,sk->bts", cos_b, self.sheet_cosine)
                + torch.einsum("btk,sk->bts", sin_b, self.sheet_sine))
        return F.softplus(self.sheet_base) + harm

    def radial_field(self, rho, theta):
        k      = torch.arange(1, self.n_harmonics + 1, dtype=theta.dtype, device=theta.device)
        angles = theta.unsqueeze(-1) * k + self.field_phase
        decay  = F.softplus(self.field_decay)
        radial = torch.exp(-rho.unsqueeze(-1) * decay)
        return (self.field_amp * torch.cos(angles) * radial).sum(-1)

    def diffusion_gate(self, theta):
        tau2   = torch.exp(2.0 * self.rbf_log_tau) + 1e-6
        diff   = theta.unsqueeze(-1) - self.rbf_centers
        kernel = torch.exp(-diff.pow(2) / (2.0 * tau2))
        values = torch.sigmoid(self.rbf_values)
        return (kernel * values).sum(-1) / (kernel.sum(-1) + 1e-6)

    def forward(self, rho, theta, sigma):
        r_s        = self.fold_radii(theta)
        rho_e      = rho.unsqueeze(-1)
        rho_p      = (2.0 * r_s - rho_e).clamp(min=0.0)
        sigma_att  = F.softplus(self.log_sigma_att) + 1e-4
        att_s      = torch.exp(-(rho_e - r_s).pow(2) / (2.0 * sigma_att ** 2))
        field      = self.radial_field(rho, theta)
        align      = field.unsqueeze(-1) * torch.cos(self.sheet_phase)
        alpha      = F.softmax(align + att_s, dim=-1)
        rho_reflected = (alpha * rho_p).sum(-1)
        g          = self.diffusion_gate(theta)
        rho_out    = g * rho_reflected + (1.0 - g) * rho
        cos_b, sin_b = self._harmonics(theta)
        theta_out  = theta + (self.phase_coeff * sin_b).sum(-1)
        sigma_out  = sigma * (1.0 + att_s.mean(-1))
        return rho_out, theta_out, sigma_out


class EfferenceKernel(nn.Module):
    def __init__(self, out_dim: int, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(3 + 2 * 3 * n_freqs, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, rho, theta, sigma):
        coords = torch.stack([rho, theta, sigma], dim=-1)
        freqs  = 2.0 ** torch.arange(self.n_freqs, dtype=coords.dtype, device=coords.device)
        angles = coords.unsqueeze(-1) * freqs
        feats  = torch.cat([coords,
                            torch.sin(angles).reshape(coords.shape[0], -1),
                            torch.cos(angles).reshape(coords.shape[0], -1)], dim=-1)
        return self.mlp(feats)


# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM, grid_size: int = GEOMETRY_GRID_SIZE):
        super().__init__()
        self.emb            = nn.Embedding(vocab_size, embed_dim)
        self.nnvc           = NNVCCodec(embed_dim)              # ← full NNVC stack
        self.rnn            = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.bolyai_proj    = BolyaiProjection(hidden_dim)
        self.curve_fold     = CurveFold()
        self.kernel         = EfferenceKernel(out_dim=hidden_dim)
        self.gate           = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc             = nn.Linear(hidden_dim, vocab_size)
        self.geo_score_head = nn.Linear(3, 1)
        self.grid_size      = grid_size
        self.geo_readout    = nn.Sequential(
            nn.Linear(grid_size * grid_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def bolyai(self, x):
        return self.bolyai_proj(x)

    def _encode(self, x: torch.Tensor):
        B, T = x.shape

        # Embed → full NNVC codec pipeline → coded embedding
        emb, rd_loss = self.nnvc(self.emb(x))        # (B,T,dim), scalar

        out, _       = self.rnn(emb)
        rho, theta, sigma = self.bolyai(out)
        rho, theta, sigma = self.curve_fold(rho, theta, sigma)

        rho_f, theta_f, sigma_f = rho.reshape(B*T), theta.reshape(B*T), sigma.reshape(B*T)
        kfeat    = self.kernel(rho_f, theta_f, sigma_f)
        fused    = self.gate(torch.cat([out.reshape(B*T, -1), kfeat], dim=-1))
        logits   = self.fc(fused).reshape(B, T, -1)
        geo_bias = self._sequential_geometry_bias(rho, theta, sigma)
        logits   = logits + GEOMETRY_BIAS_SCALE * geo_bias
        return logits, rho, theta, sigma, rd_loss

    def _coords_to_grid(self, rho, theta, sigma):
        B, T = rho.shape
        G = self.grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, G, device=rho.device, dtype=rho.dtype),
            torch.linspace(-1.0, 1.0, G, device=rho.device, dtype=rho.dtype),
            indexing="ij",
        )
        xx = xx.view(1, 1, G, G); yy = yy.view(1, 1, G, G)
        rho_n  = rho / (rho.detach().amax(dim=1, keepdim=True) + 1e-6)
        x_c    = rho_n * torch.cos(theta)
        y_c    = rho_n * torch.sin(theta)
        spread = (sigma / (sigma.detach().amax(dim=1, keepdim=True) + 1e-6)).clamp_min(0.05)
        dx2 = (xx - x_c.unsqueeze(-1).unsqueeze(-1)).pow(2)
        dy2 = (yy - y_c.unsqueeze(-1).unsqueeze(-1)).pow(2)
        return torch.exp(-(dx2 + dy2) / (2.0 * spread.unsqueeze(-1).unsqueeze(-1).pow(2)))

    def _sequential_geometry_bias(self, rho, theta, sigma, decay=GEOMETRY_DECAY):
        fields = self._coords_to_grid(rho, theta, sigma)
        B, T, G, _ = fields.shape
        canvas = torch.zeros(B, G, G, device=fields.device, dtype=fields.dtype)
        biases = []
        for t in range(T):
            canvas = decay * canvas + fields[:, t]
            biases.append(self.geo_readout(canvas.reshape(B, -1)))
        return torch.stack(biases, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _, _, _ = self._encode(x)
        return logits

    def forward_full(self, x: torch.Tensor):
        """Returns (logits, rho, theta, sigma, rd_loss)."""
        return self._encode(x)

    def _pack_state(self, logits, c_rho, c_theta, c_sigma):
        logp = torch.log(logits.clamp(min=1e-12)) if logits.ndim == 1 else logits
        C    = logp.shape[-1]
        z    = torch.zeros_like(logp)
        c_rho   = c_rho   if c_rho   is not None else z
        c_theta = c_theta if c_theta is not None else z
        c_sigma = c_sigma if c_sigma is not None else z
        return torch.cat([logp, c_rho, c_theta, c_sigma], dim=-1), C

    def _unpack_state(self, state, C):
        return state[..., :C], state[..., C:2*C], state[..., 2*C:3*C], state[..., 3*C:4*C]

    def loss(self, features: torch.Tensor, gold_indices: torch.Tensor) -> torch.Tensor:
        B, C, D = features.shape
        deltas  = self.geo_score_head(features.view(B*C, D)).view(B, C)
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
        self.eval()
        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new_tokens):
            logits = self(x)
            last   = logits[:, -1, :].clone()

            if trigram_memory is not None and x.shape[1] >= 2:
                tri_bias = trigram_memory.bias_logits(
                    x[0, -2].item(), x[0, -1].item(), last.shape[-1], x.device
                )
                last = last + trigram_bias_scale * tri_bias.unsqueeze(0)

            last[:, UNK_ID] = float("-inf")

            if radius < 1.0:
                probs_s, idx_s = torch.sort(F.softmax(last, dim=-1), descending=True)
                cum = torch.cumsum(probs_s, dim=-1)
                probs_s[cum - probs_s > radius] = 0.0
                nucleus = torch.zeros_like(last)
                nucleus.scatter_(1, idx_s, probs_s)
                probs = nucleus / nucleus.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(last, dim=-1)

            if n_fold > 0:
                p_s, idx_s = torch.sort(probs, descending=True)
                for _ in range(n_fold):
                    half = p_s.shape[-1] // 2
                    if half == 0:
                        break
                    folded = torch.abs(p_s[..., :half] - p_s[..., half:2*half])
                    denom  = folded.sum(dim=-1, keepdim=True)
                    if denom.item() < 1e-12:
                        break
                    p_s   = folded / denom
                    idx_s = idx_s[..., :half]
                probs = torch.zeros_like(last)
                probs.scatter_(1, idx_s, p_s)
                total = probs.sum(dim=-1, keepdim=True)
                if total.item() > 1e-12:
                    probs = probs / total

            x = torch.cat([x, torch.multinomial(probs, num_samples=1)], dim=1)

        return x[0].tolist()


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_all(model: SimpleLM, tokenizer: Tokenizer, tri_memory: TrigramMemory | None = None):
    torch.save(
        {
            "model_state":     model.state_dict(),
            "vocab_size":      len(tokenizer.t2i),
            "embed_dim":       EMBED_DIM,
            "hidden_dim":      HIDDEN_DIM,
            "seq_len":         SEQ_LEN,
            "n_freqs":         KERNEL_N_FREQS,
            "n_harmonics":     CURVE_FOLD_HARMONICS,
            "n_sheets":        CURVE_FOLD_SHEETS,
            "n_rbf":           CURVE_FOLD_RBF,
            "nnvc_quant_step": NNVC_QUANT_STEP,
            "nnvc_lambda":     NNVC_LAMBDA,
            "nnvc_hyper_dim":  NNVC_HYPER_DIM,
            "trigram_size":    sum(len(v) for v in tri_memory.next_counts.values()) if tri_memory else 0,
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
    tri_mem = TrigramMemory.load(TRIGRAM_FILE) if os.path.exists(TRIGRAM_FILE) else None
    return model, tokenizer, tri_mem


# ── Generation ────────────────────────────────────────────────────────────────

def ranked_generate(
    model, tokenizer, trigram_memory, prompt,
    length_weight=0.35, n_samples=5, max_new_tokens=30,
    radius=1.0, n_fold=4, device="cpu",
) -> str:
    start_ids  = tokenizer.encode(prompt) or [tokenizer.t2i["<unk>"]]
    prompt_len = len(prompt.split())
    completions = [
        tokenizer.decode(
            model.generate(start_ids, max_new_tokens=max_new_tokens, device=device,
                           radius=radius, n_fold=n_fold, trigram_memory=trigram_memory)
        )
        for _ in range(n_samples)
    ]
    def score(c):
        c_len = len(c.split())
        return 1.0 if (prompt_len == 0 and c_len == 0) else \
               1.0 - abs(c_len - prompt_len) / max(c_len, prompt_len, 1)
    return sorted(completions, key=score)[0]


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

    tok_arr    = np.asarray(token_ids, dtype=np.int32)
    tri_memory = TrigramMemory(); tri_memory.build(tok_arr)
    tri_vocab  = TrigramVocab();  tri_vocab.build(tok_arr)
    trigram_ids = tri_vocab.encode(tok_arr)

    dataset = TextDataset(
        token_ids=tok_arr, trigram_ids=trigram_ids,
        seq_len=SEQ_LEN, vocab_size=len(tokenizer.t2i), trigram_vocab_size=tri_vocab.size,
    )
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = SimpleLM(vocab_size=len(tokenizer.t2i)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    trigram_links = sum(len(v) for v in tri_memory.next_counts.values())
    log_lines = [
        f"Training on {device} | vocab={len(tokenizer.t2i):,} | tokens={len(token_ids):,} | trigram-links={trigram_links:,}",
        f"NNVC stack: IntraPredictor + ScaleHyperprior + ContextModel + InLoopFilter",
        f"  quant_step={NNVC_QUANT_STEP} | λ={NNVC_LAMBDA} | hyper_dim={NNVC_HYPER_DIM}",
    ]

    model.train()
    for epoch in range(epochs):
        total_ce = 0.0; total_rd = 0.0

        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device)

            logits, rho, theta, sigma, rd_loss = model.forward_full(xb)

            ce_loss  = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            logp_pos = torch.log(F.softmax(logits, dim=-1).max(dim=-1).values.clamp(min=1e-12))
            packed, C = model._pack_state(logp_pos, rho, theta, sigma)
            logp_u, c_rho_u, c_theta_u, c_sigma_u = model._unpack_state(packed, C)
            geo_feats = torch.stack([c_rho_u, c_theta_u, c_sigma_u], dim=-1)
            geo_loss  = model.loss(geo_feats, logp_u.argmax(dim=-1))

            # Combined: task CE + geometry + NNVC Lagrangian D+λR
            loss = ce_loss + 0.1 * geo_loss + rd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_ce += ce_loss.item()
            total_rd += rd_loss.item()

        avg_ce = total_ce / len(loader)
        avg_rd = total_rd / len(loader)
        msg = f"Epoch {epoch+1}/{epochs} — ce: {avg_ce:.6f} | nnvc_rd (D+λR): {avg_rd:.6f}"
        log_lines.append(msg)
        print(msg)

    save_all(model, tokenizer, tri_memory)
    log_lines.append(f"✅ Saved → {MODEL_FILE}, {TOKENIZER_FILE}, {TRIGRAM_FILE}")

    dataset.close()
    del model, optimizer, dataset, loader, token_ids, tok_arr, trigram_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "\n".join(log_lines)


# ── File helpers ──────────────────────────────────────────────────────────────

def _extract_text_from_jsonl(raw: str) -> str:
    lines_out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                lines_out.append(" ".join(str(v) for v in obj.values() if isinstance(v, str) and v.strip()))
            elif isinstance(obj, str):
                lines_out.append(obj)
        except json.JSONDecodeError:
            lines_out.append(line)
    return "\n".join(lines_out)


def _extract_text_from_json(raw: str) -> str:
    try: data = json.loads(raw)
    except json.JSONDecodeError: return raw
    parts: List[str] = []
    def collect(node):
        if isinstance(node, str): parts.append(node)
        elif isinstance(node, dict):
            for v in node.values(): collect(v)
        elif isinstance(node, list):
            for item in node: collect(item)
    collect(data)
    return "\n".join(p for p in parts if p.strip())


def load_uploaded_file(file_obj):
    if file_obj is None:
        return "", "", "No file uploaded."
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    ext  = os.path.splitext(path)[1].lower()
    if ext not in ACCEPTED_EXTENSIONS:
        return "", "", f"❌ Unsupported file type '{ext}'. Accepted: {', '.join(ACCEPTED_EXTENSIONS)}"
    try:
        raw = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    raw = f.read()
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            return "", "", "❌ Could not decode file."
        if ext == ".jsonl": text = _extract_text_from_jsonl(raw)
        elif ext == ".json": text = _extract_text_from_json(raw)
        elif ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            text = "\n".join(" ".join(c.strip().strip('"') for c in l.split(sep) if c.strip()) for l in raw.splitlines())
        else:
            text = raw
        n_chars, n_words = len(text), len(text.split())
        fname    = os.path.basename(path)
        tmp_path = _write_tmp(text); del text
        preview  = _read_tmp(tmp_path)[:3000] + (f"\n\n…({n_chars:,} chars total)" if n_chars > 3000 else "")
        return preview, tmp_path, f"✅ Loaded '{fname}'  —  {n_chars:,} chars · {n_words:,} words"
    except Exception as e:
        return "", "", f"❌ Error reading file: {e}"


def _write_tmp(text: str) -> str:
    fd, path = tempfile.mkstemp(dir=_TMP_DIR, suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(text)
    return path


def _read_tmp(path: str) -> str:
    if not path: return ""
    try:
        with open(path, "r", encoding="utf-8") as f: return f.read()
    except OSError: return ""


def _delete_tmp(path: str):
    if path:
        try: os.remove(path)
        except OSError: pass


# ── HuggingFace Dataset helpers ───────────────────────────────────────────────

def fetch_hf_configs(dataset_name: str):
    dataset_name = dataset_name.strip()
    if not dataset_name:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), "Enter a dataset name first."
    try:
        configs = get_dataset_config_names(dataset_name) or ["default"]
        return gr.update(choices=configs, value=configs[0]), gr.update(choices=[], value=None), f"Found {len(configs)} config(s)."
    except Exception as e:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), f"❌ {e}"


def fetch_hf_fields(dataset_name: str, config: str, split: str = "train"):
    dataset_name = dataset_name.strip()
    if not dataset_name or not config:
        return gr.update(choices=[], value=None), "Provide dataset name and config."
    try:
        cfg = None if config in ("default", "") else config
        ds  = load_dataset(dataset_name, cfg, split=split, streaming=True, trust_remote_code=False)
        ex  = next(iter(ds))
        sf  = [k for k, v in ex.items() if isinstance(v, str)] or list(ex.keys())
        return gr.update(choices=sf, value=sf[0] if sf else None), f"Fields: {sf}"
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ {e}"


_cancel_event = threading.Event()


def _hf_loader_worker(dataset_name, config, split, text_field, max_samples, result_queue, cancel_event):
    try:
        cfg   = None if config in ("default", "") else config
        ds    = load_dataset(dataset_name, cfg, split=split, streaming=True, trust_remote_code=False)
        texts = []
        for i, ex in enumerate(ds):
            if cancel_event.is_set():
                result_queue.put(("error", "⚠️ Loading cancelled.")); return
            if i >= max_samples: break
            val = ex.get(text_field, "")
            if isinstance(val, str) and val.strip(): texts.append(val.strip())
            if i % 50 == 0: result_queue.put(("progress", i, len(texts)))
        full_text = "\n\n".join(texts); del texts
        tmp_path  = _write_tmp(full_text)
        result_queue.put(("done", tmp_path, len(full_text), len(full_text.split())))
        del full_text
    except Exception as exc:
        result_queue.put(("error", f"❌ {exc}"))


def ui_load_hf_and_preview(dataset_name, config, split, text_field, max_samples):
    global _cancel_event
    dataset_name = dataset_name.strip()
    if not dataset_name or not text_field:
        yield "", "⚠️ Provide a dataset name and select a text field first."; return
    _cancel_event = threading.Event()
    rq: queue.Queue = queue.Queue()
    threading.Thread(target=_hf_loader_worker,
                     args=(dataset_name, config, split, text_field, int(max_samples), rq, _cancel_event),
                     daemon=True).start()
    while True:
        try: msg = rq.get(timeout=0.25)
        except queue.Empty: continue
        if msg[0] == "progress":
            yield "", f"⏳ Loading… {msg[1]:,} rows · {msg[2]:,} texts"
        elif msg[0] == "done":
            _, tmp, nc, nw = msg
            preview = _read_tmp(tmp)[:2000] + ("\n…(truncated)" if nc > 2000 else "") + f"\n\n✅ Done — {nc:,} chars · {nw:,} words"
            yield tmp, preview; break
        elif msg[0] == "error":
            yield "", msg[1]; break


def ui_cancel_hf_load():
    _cancel_event.set()
    return "⚠️ Cancel requested…"


# ── HuggingFace Hub ───────────────────────────────────────────────────────────

def validate_token(hf_token: str) -> str:
    hf_token = hf_token.strip()
    if not hf_token: return "—"
    try:
        return f"✅ Logged in as: {whoami(token=hf_token)['name']}"
    except Exception as e:
        return f"❌ {e}"


def push_to_hub(repo_id: str, hf_token: str, commit_message: str, private: bool) -> str:
    repo_id = repo_id.strip(); hf_token = hf_token.strip()
    commit_message = commit_message.strip() or "Upload SimpleLM checkpoint"
    if not repo_id: return "❌ Provide a repo ID."
    if not hf_token: return "❌ Provide a HuggingFace write token."
    if not os.path.exists(MODEL_FILE): return "❌ No saved model — train first."
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        card = f"""---
language: en
tags: [text-generation, gru, pytorch, nnvc]
---
# SimpleLM — {repo_id}
## NNVC Stack
| Component | Role |
|-----------|------|
| NNVCIntraPredictor | Causal intra prediction; encode residual only |
| NNVCHyperprior | Scale hyperprior (Ballé 2018); Gaussian CDF rate |
| NNVCContextModel | Autoregressive masked-conv entropy refinement |
| NNVCInLoopFilter | ALF-style depth-wise separable post-filter |
## Hyperparams
| | |
|---|---|
| Quant step Δ | {NNVC_QUANT_STEP} |
| λ (R-D weight) | {NNVC_LAMBDA} |
| Hyper dim | {NNVC_HYPER_DIM} |
"""
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(card); card_path = f.name
        for local, remote in [(MODEL_FILE, MODEL_FILE), (TOKENIZER_FILE, TOKENIZER_FILE), (card_path, "README.md")]:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=repo_id, repo_type="model", commit_message=commit_message)
        os.unlink(card_path)
        return f"✅ Pushed to https://huggingface.co/{repo_id}  [{'private' if private else 'public'}]"
    except Exception as e:
        return f"❌ Upload failed: {e}"


def pull_from_hub(repo_id: str, hf_token: str) -> str:
    global _model, _tokenizer, _tri_memory
    repo_id = repo_id.strip(); hf_token = hf_token.strip() or None
    if not repo_id: return "❌ Provide a repo ID."
    try:
        kw = dict(repo_id=repo_id, repo_type="model", token=hf_token)
        shutil.copy(hf_hub_download(filename=MODEL_FILE,     **kw), MODEL_FILE)
        shutil.copy(hf_hub_download(filename=TOKENIZER_FILE, **kw), TOKENIZER_FILE)
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
        return f"✅ Loaded from {repo_id}\nVocab: {len(_tokenizer.t2i):,} | Params: {sum(p.numel() for p in _model.parameters()):,}"
    except Exception as e:
        return f"❌ {e}"


# ── App state ─────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model: SimpleLM | None           = None
_tokenizer: Tokenizer | None      = None
_tri_memory: TrigramMemory | None = None
AUTO_MODEL_REPO = "trainman999/Thinking-lite"


def _try_load():
    global _model, _tokenizer, _tri_memory
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
        return "✅ Existing local model loaded."
    try:
        shutil.copy(hf_hub_download(filename=MODEL_FILE,     repo_id=AUTO_MODEL_REPO, repo_type="model"), MODEL_FILE)
        shutil.copy(hf_hub_download(filename=TOKENIZER_FILE, repo_id=AUTO_MODEL_REPO, repo_type="model"), TOKENIZER_FILE)
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
        return f"✅ Auto-loaded {AUTO_MODEL_REPO} ({sum(p.numel() for p in _model.parameters()):,} params)"
    except Exception as e:
        return f"⚠️ Could not download default model: {e}"


_startup_msg = _try_load()


def ui_train(text, hf_path, file_path, epochs, lr, batch_size):
    global _model, _tokenizer, _tri_memory
    src = _read_tmp(file_path) or _read_tmp(hf_path) or text
    if not src or not src.strip(): return "❌ No text provided."
    result = train_on_text(src, int(epochs), float(lr), int(batch_size))
    del src; gc.collect()
    if "✅" in result:
        _model, _tokenizer, _tri_memory = load_all(device=DEVICE)
    return result


def ui_generate(prompt, max_new_tokens, n_samples, radius, n_fold):
    if _model is None or _tokenizer is None: return "❌ No model loaded."
    if not prompt.strip(): return "❌ Enter a prompt."
    return ranked_generate(_model, _tokenizer, _tri_memory, prompt,
                           n_samples=int(n_samples), max_new_tokens=int(max_new_tokens),
                           radius=float(radius), n_fold=int(n_fold), device=DEVICE)


def model_info() -> str:
    lines = [f"Device: {DEVICE}"]
    if _model is not None and _tokenizer is not None:
        fold = _model.curve_fold
        lines += [
            f"Vocab size:           {len(_tokenizer.t2i):,}",
            f"Parameters:           {sum(p.numel() for p in _model.parameters()):,}",
            f"Embed / Hidden dim:   {EMBED_DIM} / {HIDDEN_DIM}",
            f"Seq length:           {SEQ_LEN}",
            f"Fold harmonics/sheets:{fold.n_harmonics} / {fold.n_sheets}",
            f"─── NNVC stack ───────────────────────",
            f"  IntraPredictor:     causal conv (k=3, left-pad 2)",
            f"  ScaleHyperprior:    h_a + h_s, hyper_dim={NNVC_HYPER_DIM}",
            f"  ContextModel:       masked autoregressive conv",
            f"  InLoopFilter:       DW-sep residual CNN (ALF-style)",
            f"  Quant step Δ:       {NNVC_QUANT_STEP}",
            f"  λ (R-D weight):     {NNVC_LAMBDA}",
        ]
    else:
        lines.append("No model loaded yet.")
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="SimpleLM — Train & Generate",
    theme=gr.themes.Base(
        primary_hue="slate", secondary_hue="zinc", neutral_hue="zinc",
        font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ),
    css=".startup-msg{font-size:0.8rem;color:#71717a}.upload-status{font-size:0.8rem}",
) as demo:

    gr.Markdown("# 🧠 SimpleLM — GRU Language Model\n"
                "Train a compact GRU language model with NNVC-based embedding coding.")
    gr.Markdown(f"**Runtime:** `{DEVICE.upper()}` | {_startup_msg}", elem_classes="startup-msg")

    with gr.Tabs():

        with gr.Tab("📦 HF Dataset"):
            gr.Markdown("### Load text from a Hugging Face dataset")
            with gr.Row():
                hf_name  = gr.Textbox(label="Dataset name", placeholder="wikitext", scale=3)
                hf_split = gr.Textbox(label="Split", value="train", scale=1)
            with gr.Row():
                btn_configs = gr.Button("1 — Fetch configs", variant="secondary")
                hf_status   = gr.Textbox(label="Status", interactive=False, scale=3)
            with gr.Row():
                hf_config  = gr.Dropdown(label="Config",     choices=[], interactive=True, scale=2)
                btn_fields = gr.Button("2 — Fetch fields", variant="secondary", scale=1)
            hf_field = gr.Dropdown(label="Text field", choices=[], interactive=True)
            hf_max   = gr.Slider(label="Max samples", minimum=10, maximum=500_000_000, value=500, step=10)
            with gr.Row():
                btn_load_hf   = gr.Button("3 — Load text ⬇", variant="primary", scale=4)
                btn_cancel_hf = gr.Button("✖ Cancel", variant="stop", scale=1)
            hf_preview = gr.Textbox(label="Preview / progress", lines=6, interactive=False)
            hf_full    = gr.State("")
            btn_configs.click(fetch_hf_configs, [hf_name], [hf_config, hf_field, hf_status])
            btn_fields.click(fetch_hf_fields,   [hf_name, hf_config, hf_split], [hf_field, hf_status])
            btn_load_hf.click(ui_load_hf_and_preview, [hf_name, hf_config, hf_split, hf_field, hf_max], [hf_full, hf_preview])
            btn_cancel_hf.click(ui_cancel_hf_load, outputs=[hf_preview])

        with gr.Tab("🏋 Train"):
            gr.Markdown(
                "### Train on text\n"
                "_NNVC pipeline applied each step: IntraPredict → g_a → Hyperprior → ContextModel → Quantise → g_s → InLoopFilter._\n"
                "_Training log shows `ce` (task) and `nnvc_rd` (D + λR) separately._"
            )
            with gr.Group():
                gr.Markdown(f"#### 📂 Upload file\n_Accepted: {', '.join(ACCEPTED_EXTENSIONS)}_")
                with gr.Row():
                    upload_file   = gr.File(label="Drop or click to upload",
                                            file_types=ACCEPTED_EXTENSIONS, file_count="single", scale=4)
                    btn_load_file = gr.Button("Load into editor ⬇", variant="primary", scale=1)
                upload_status = gr.Textbox(label="File status", interactive=False, value="—", elem_classes="upload-status")
            gr.Markdown("---")
            file_full  = gr.State("")
            train_text = gr.Textbox(label="Training text", placeholder="Paste text here…", lines=12)
            with gr.Row():
                btn_use_hf = gr.Button("⬆ Use HuggingFace text", variant="secondary")
                btn_clear  = gr.Button("🗑 Clear editor",         variant="secondary")
            btn_use_hf.click(
                lambda p: (lambda t: t[:3000] + f"\n…({len(t):,} chars)" if len(t) > 3000 else t)(_read_tmp(p)),
                [hf_full], [train_text],
            )
            btn_clear.click(
                lambda hp, fp: (_delete_tmp(hp), _delete_tmp(fp), "", "", "", "—")[2:],
                [hf_full, file_full], [train_text, hf_full, file_full, upload_status],
            )
            btn_load_file.click(load_uploaded_file, [upload_file], [train_text, file_full, upload_status])
            upload_file.change(load_uploaded_file,  [upload_file], [train_text, file_full, upload_status])
            gr.Markdown("---")
            with gr.Row():
                t_epochs = gr.Slider(label="Epochs",     minimum=1,    maximum=50,   value=EPOCHS,    step=1)
                t_lr     = gr.Slider(label="LR",         minimum=1e-5, maximum=1e-1, value=LR,        step=1e-5)
                t_bs     = gr.Slider(label="Batch size", minimum=8,    maximum=256,  value=BATCH_SIZE, step=8)
            btn_train = gr.Button("Train 🚀", variant="primary")
            train_log = gr.Textbox(label="Training log", lines=14, interactive=False)
            btn_train.click(ui_train, [train_text, hf_full, file_full, t_epochs, t_lr, t_bs], [train_log])

        with gr.Tab("✨ Generate"):
            gr.Markdown("### Generate text completions")
            gen_prompt = gr.Textbox(label="Prompt", placeholder="The quick brown fox…", lines=2)
            with gr.Row():
                gen_tokens  = gr.Slider(label="Max new tokens",             minimum=10,  maximum=1300, value=300,  step=5)
                gen_samples = gr.Slider(label="Samples (ranked)",           minimum=1,   maximum=20,   value=5,    step=1)
                gen_radius  = gr.Slider(label="Radius r (nucleus)",         minimum=0.1, maximum=1.0,  value=0.6,  step=0.05)
                gen_n_fold  = gr.Slider(label="Fold iterations (abs-diff)", minimum=0,   maximum=10,   value=4,    step=1)
            btn_gen    = gr.Button("Generate ✨", variant="primary")
            gen_output = gr.Textbox(label="Completion", lines=6, interactive=False)
            btn_gen.click(ui_generate, [gen_prompt, gen_tokens, gen_samples, gen_radius, gen_n_fold], [gen_output])

        with gr.Tab("🤗 HF Hub"):
            gr.Markdown("### Push / pull model")
            with gr.Group():
                gr.Markdown("#### 🔑 Auth")
                with gr.Row():
                    hub_token    = gr.Textbox(label="HF Token", placeholder="hf_…", type="password", scale=4)
                    btn_validate = gr.Button("Validate", variant="secondary", scale=1)
                token_status = gr.Textbox(label="Token status", interactive=False, value="—")
                btn_validate.click(validate_token, [hub_token], [token_status])
            gr.Markdown("---")
            with gr.Group():
                gr.Markdown("#### ⬆ Push")
                with gr.Row():
                    push_repo    = gr.Textbox(label="Repo ID", placeholder="user/repo", scale=4)
                    push_private = gr.Checkbox(label="Private", value=False, scale=1)
                push_commit = gr.Textbox(label="Commit message", value="Upload SimpleLM checkpoint")
                btn_push    = gr.Button("Push ⬆", variant="primary")
                push_status = gr.Textbox(label="Status", lines=5, interactive=False)
                btn_push.click(push_to_hub, [push_repo, hub_token, push_commit, push_private], [push_status])
            gr.Markdown("---")
            with gr.Group():
                gr.Markdown("#### ⬇ Load from Hub")
                pull_repo   = gr.Textbox(label="Repo ID", placeholder="user/repo")
                btn_pull    = gr.Button("Load ⬇", variant="secondary")
                pull_status = gr.Textbox(label="Status", lines=4, interactive=False)
                btn_pull.click(pull_from_hub, [pull_repo, hub_token], [pull_status])

        with gr.Tab("ℹ Model info"):
            info_box = gr.Textbox(label="Model stats", lines=16, interactive=False, value=model_info())
            gr.Button("Refresh").click(model_info, outputs=[info_box])


if __name__ == "__main__":
    demo.launch(share=False)
