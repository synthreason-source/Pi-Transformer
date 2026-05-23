# =============================================================================
#  layer_isomorphism_torch.py
#  All 14 isomorphism layers (L0–L13) as torch.nn.Module subclasses.
#
#  Design principles
#  -----------------
#  • Every layer is a self-contained nn.Module with a custom __init__ that
#    registers its hyper-parameters as nn.Parameter (learnable) or as named
#    buffers (non-gradient scalars that still move with .to(device)).
#  • forward() accepts and returns plain Python / numpy inputs where the
#    upstream code expects them, but all heavy maths runs on torch tensors.
#  • IsomorphismPipeline wires all layers together and mirrors the API of
#    the original IsomorphismGenerator so app.py needs only a one-line swap.
#  • No external dependencies beyond torch, numpy, math, collections.
# =============================================================================

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_tensor(probs, dtype=torch.float64) -> torch.Tensor:
    if isinstance(probs, torch.Tensor):
        return probs.to(dtype)
    if isinstance(probs, np.ndarray):
        return torch.from_numpy(probs.astype(np.float64)).to(dtype)
    return torch.tensor(probs, dtype=dtype)


def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Safe L1 normalisation."""
    s = t.sum()
    return t / s.clamp(min=1e-30)


def _char_trigrams(word: str):
    return {word[i:i + 3] for i in range(len(word) - 2)} if len(word) >= 3 else {word}


# ---------------------------------------------------------------------------
# L0 – Raw distribution with repetition penalty
# ---------------------------------------------------------------------------

class L0_RawDist(nn.Module):
    """
    Extracts the CPD posterior for a context, applies per-token repetition
    penalty, and returns a normalised probability distribution.

    Custom init
    -----------
    rep_penalty : learnable scalar (clamped ≥ 1.0 in forward)
                  Registered as nn.Parameter so it participates in gradient
                  flow when the pipeline is fine-tuned end-to-end.
    """

    def __init__(self, rep_penalty: float = 1.13):
        super().__init__()
        # Learnable repetition-penalty exponent base
        self.rep_penalty = nn.Parameter(torch.tensor(rep_penalty, dtype=torch.float64))

    def forward(
        self,
        dist,                   # nltk ConditionalProbDist slice (cpd[ctx])
        history: Counter,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        pen = self.rep_penalty.clamp(min=1.0)

        raw: List[Tuple[str, float]] = []
        for s in dist.samples():
            if not s:
                continue
            p   = max(1e-12, float(dist.prob(s)))
            cnt = history[s]
            if cnt > 0:
                p /= pen.detach().item() ** cnt
            raw.append((s, p))

        if not raw:
            return raw, {}

        raw.sort(key=lambda x: x[1], reverse=True)

        probs_t = _to_tensor([p for _, p in raw])
        probs_t = _normalise(probs_t)

        words   = [w for w, _ in raw]
        pairs   = list(zip(words, probs_t.tolist()))

        layer = {
            "name":   "L0_RAW_DIST",
            "words":  words,
            "probs":  probs_t.detach().numpy(),
            "source": f"CPD posterior + rep_penalty={pen.detach().item():.4f}",
        }
        return pairs, layer


# ---------------------------------------------------------------------------
# L1 – Temperature scaling
# ---------------------------------------------------------------------------

class L1_TempScaled(nn.Module):
    """
    Applies temperature scaling: p_i ∝ p_i^(1/T).

    Custom init
    -----------
    temperature : learnable scalar (clamped ≥ 1e-3).
                  Initialised from the constructor argument.
    """

    def __init__(self, temperature: float = 4.3):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float64))

    def forward(
        self, pairs: List[Tuple[str, float]]
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        T = self.temperature.clamp(min=1e-3)

        probs_t  = _to_tensor([p for _, p in pairs])
        scaled_t = probs_t.pow(1.0 / T)
        scaled_t = _normalise(scaled_t)

        words = [w for w, _ in pairs]
        out   = list(zip(words, scaled_t.tolist()))

        layer = {
            "name":   "L1_TEMP_SCALED",
            "words":  words,
            "probs":  scaled_t.detach().numpy(),
            "source": f"temperature={T.detach().item():.4f}",
        }
        return out, layer


# ---------------------------------------------------------------------------
# L2 – Insight penalty
# ---------------------------------------------------------------------------

class L2_InsightPenalty(nn.Module):
    """
    Penalises tokens whose probability exceeds the mean by a factor
    proportional to `insight_penalty`.  Acts as a soft entropy booster.

    Custom init
    -----------
    insight_penalty : learnable, clamped ≥ 0.
    """

    def __init__(self, insight_penalty: float = 3.95):
        super().__init__()
        self.insight_penalty = nn.Parameter(
            torch.tensor(insight_penalty, dtype=torch.float64)
        )

    def forward(
        self, pairs: List[Tuple[str, float]]
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        strength = self.insight_penalty.clamp(min=0.0)

        probs_t  = _to_tensor([p for _, p in pairs])
        mean_p   = probs_t.mean().clamp(min=1e-30)
        excess   = (probs_t - mean_p).clamp(min=0.0)
        penalised = probs_t / (1.0 + strength * excess / mean_p)
        penalised = _normalise(penalised.clamp(min=1e-12))

        words = [w for w, _ in pairs]
        out   = list(zip(words, penalised.tolist()))

        layer = {
            "name":   "L2_INSIGHT",
            "words":  words,
            "probs":  penalised.detach().numpy(),
            "source": f"insight_penalty={strength.detach().item():.4f}",
        }
        return out, layer


# ---------------------------------------------------------------------------
# L3 – Top-K / Top-P truncation
# ---------------------------------------------------------------------------

class L3_TopKTopP(nn.Module):
    """
    Truncates the candidate set to the top-K tokens and then applies
    nucleus (top-P) filtering.

    Custom init
    -----------
    top_k   : integer buffer (not learnable; registered so it moves with device).
    top_p   : learnable scalar in (0, 1].
    """

    def __init__(self, top_k: int = 100, top_p: float = 1.0):
        super().__init__()
        # top_k is discrete — keep as buffer
        self.register_buffer("top_k_buf", torch.tensor(top_k, dtype=torch.int64))
        # top_p is continuous — learnable
        self.top_p = nn.Parameter(torch.tensor(top_p, dtype=torch.float64))

    def forward(
        self, pairs: List[Tuple[str, float]]
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        k    = int(self.top_k_buf.item())
        p_th = float(self.top_p.clamp(1e-3, 1.0).item())

        truncated = pairs[:k]
        kept, cumulative = [], 0.0
        for w, p in truncated:
            kept.append((w, p))
            cumulative += p
            if cumulative >= p_th:
                break

        probs_t = _to_tensor([p for _, p in kept])
        probs_t = _normalise(probs_t)

        words = [w for w, _ in kept]
        out   = list(zip(words, probs_t.tolist()))

        layer = {
            "name":   "L3_TOPK_TOPP",
            "words":  words,
            "probs":  probs_t.detach().numpy(),
            "source": f"top_k={k} top_p={p_th:.4f}",
        }
        return out, layer


# ---------------------------------------------------------------------------
# Shared Gaussian-gradient zone layer base
# ---------------------------------------------------------------------------

class _ZoneGradientBase(nn.Module):
    """
    Base class for zone layers L4–L9.

    Every zone layer applies a Gaussian gradient over the ranked candidate
    list, centring the peak at the mean rank of tokens in the active zone.

    Custom init (shared)
    --------------------
    sigma : learnable width of the Gaussian (clamped > 0).
    floor : learnable minimum weight (clamped to [0, 1)).
    Both are per-layer learnable parameters.
    """

    def __init__(self, name: str, sigma: float, floor: float, source_hint: str = ""):
        super().__init__()
        self._layer_name  = name
        self._source_hint = source_hint
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float64))
        self.floor = nn.Parameter(torch.tensor(floor, dtype=torch.float64))

    def _gradient(
        self,
        zone_set: set,
        candidates: List[Tuple[str, float]],
    ) -> torch.Tensor:
        n = len(candidates)
        if n == 0:
            return torch.zeros(0, dtype=torch.float64)

        sigma = self.sigma.clamp(min=1e-6)
        floor = self.floor.clamp(min=0.0, max=1.0 - 1e-6)

        indices   = torch.arange(n, dtype=torch.float64)
        norm_idx  = indices / max(1, n - 1)

        zone_ranks = [
            i / max(1, n - 1)
            for i, (w, _) in enumerate(candidates)
            if w in zone_set
        ]
        centre = float(torch.tensor(zone_ranks).mean()) if zone_ranks else 0.0

        gauss   = torch.exp(-0.5 * ((norm_idx - centre) / sigma) ** 2)
        weights = floor + (1.0 - floor) * gauss
        return _normalise(weights)

    def _make_layer(self, weights: torch.Tensor, candidates, source_extra: str = "") -> Dict:
        return {
            "name":   self._layer_name,
            "words":  [w for w, _ in candidates],
            "probs":  weights.detach().numpy(),
            "source": f"{self._source_hint} {source_extra}".strip(),
        }


# ---------------------------------------------------------------------------
# L4 – Frequency zone gradient
# ---------------------------------------------------------------------------

class L4_ZoneFreq(_ZoneGradientBase):
    """
    Gaussian gradient over the frequency-zone bucket (high / mid / low)
    appropriate for the current prompt words.

    Custom init
    -----------
    sigma, floor : learnable Gaussian shape (inherited from base).
    freq_high_thresh, freq_mid_thresh : integer buffers.
    """

    def __init__(self, sigma: float = 0.50, floor: float = 0.05,
                 freq_high_thresh: int = 10, freq_mid_thresh: int = 3):
        super().__init__("L4_ZONE_FREQ", sigma, floor, "freq_zone")
        self.register_buffer("freq_high_thresh",
                             torch.tensor(freq_high_thresh, dtype=torch.int64))
        self.register_buffer("freq_mid_thresh",
                             torch.tensor(freq_mid_thresh, dtype=torch.int64))

    def forward(
        self,
        candidates:   List[Tuple[str, float]],
        prompt_words: List[str],
        freq_zones:   Dict[str, List[str]],
        token_freq:   Dict[str, int],
    ) -> Dict:
        hi_th = int(self.freq_high_thresh.item())
        mi_th = int(self.freq_mid_thresh.item())

        high_set = set(freq_zones.get("high", []))
        if any(w in high_set for w in prompt_words):
            key = "high"
        elif all(token_freq.get(w, 0) < mi_th for w in prompt_words):
            key = "low"
        else:
            key = "mid"

        zone_set = set(freq_zones.get(key, []))
        weights  = self._gradient(zone_set, candidates)
        return self._make_layer(weights, candidates, f"key={key}")


# ---------------------------------------------------------------------------
# L5 – Alpha-zone gradient
# ---------------------------------------------------------------------------

class L5_ZoneAlpha(_ZoneGradientBase):
    """
    Gaussian gradient favouring tokens that share an initial letter with
    any prompt word.

    Custom init
    -----------
    sigma, floor : learnable (inherited).
    """

    def __init__(self, sigma: float = 0.40, floor: float = 0.05):
        super().__init__("L5_ZONE_ALPHA", sigma, floor, "alpha_zone")

    def forward(
        self,
        candidates:   List[Tuple[str, float]],
        prompt_words: List[str],
        alpha_zones:  Dict[str, List[str]],
    ) -> Dict:
        alpha_words: set = set()
        for w in prompt_words:
            if w and w[0].isalpha():
                alpha_words.update(alpha_zones.get(w[0], []))

        weights = self._gradient(alpha_words, candidates)
        keys    = [w[0] for w in prompt_words if w]
        return self._make_layer(weights, candidates, f"keys={keys}")


# ---------------------------------------------------------------------------
# L6 – Bigram n-gram zone gradient
# ---------------------------------------------------------------------------

class L6_ZoneBigram(_ZoneGradientBase):
    """
    Gaussian gradient using successors predicted by prompt bigrams.

    Custom init
    -----------
    sigma, floor : learnable (inherited).
    """

    def __init__(self, sigma: float = 0.25, floor: float = 0.04):
        super().__init__("L6_ZONE_BIGRAM", sigma, floor, "ngram bigram context")

    def forward(
        self,
        candidates:   List[Tuple[str, float]],
        prompt_words: List[str],
        ngram_zones:  Dict,
    ) -> Dict:
        bigram_words: set = set()
        for i in range(len(prompt_words) - 1):
            bigram_words.update(
                ngram_zones.get((prompt_words[i], prompt_words[i + 1]), [])
            )

        weights = self._gradient(bigram_words, candidates)
        return self._make_layer(weights, candidates)


# ---------------------------------------------------------------------------
# L7 – Live trigram context zone gradient
# ---------------------------------------------------------------------------

class L7_ZoneTrigram(_ZoneGradientBase):
    """
    Gaussian gradient using the two most-recent generated tokens as the
    live n-gram context key.

    Custom init
    -----------
    sigma, floor : learnable (inherited, tighter defaults for live ctx).
    """

    def __init__(self, sigma: float = 0.20, floor: float = 0.03):
        super().__init__("L7_ZONE_TRIGRAM", sigma, floor, "live_ctx")

    def forward(
        self,
        candidates:    List[Tuple[str, float]],
        context_deque: deque,
        ngram_zones:   Dict,
    ) -> Dict:
        ctx_list = list(context_deque)
        if len(ctx_list) >= 2:
            key      = tuple(ctx_list[-2:])
            zone_set = set(ngram_zones.get(key, []))
            src      = f"live_ctx={key}"
        elif len(ctx_list) == 1:
            key      = (ctx_list[-1],)
            zone_set = set(ngram_zones.get(key, []))
            src      = f"live_ctx=({ctx_list[-1]},)"
        else:
            zone_set = set()
            src      = "no live context"

        weights = self._gradient(zone_set, candidates)
        return self._make_layer(weights, candidates, src)


# ---------------------------------------------------------------------------
# L8 – Character-trigram neighbour gradient
# ---------------------------------------------------------------------------

class L8_ZoneCharTrig(_ZoneGradientBase):
    """
    Gaussian gradient over tokens that share character-trigrams with the
    prompt words (surface-form similarity).

    Custom init
    -----------
    sigma, floor : learnable (inherited).
    """

    def __init__(self, sigma: float = 0.35, floor: float = 0.04):
        super().__init__("L8_ZONE_CHAR_TRIG", sigma, floor, "char-trigram neighbours")

    def forward(
        self,
        candidates:    List[Tuple[str, float]],
        prompt_words:  List[str],
        char_trig_idx: Dict[str, set],
    ) -> Dict:
        prompt_tgs: set = set()
        for w in prompt_words:
            prompt_tgs |= _char_trigrams(w)

        char_neighbours: set = set()
        for tg in prompt_tgs:
            char_neighbours |= char_trig_idx.get(tg, set())

        weights = self._gradient(char_neighbours, candidates)
        return self._make_layer(weights, candidates)


# ---------------------------------------------------------------------------
# L9 – Latent BOS quartile gradient
# ---------------------------------------------------------------------------

class L9_ZoneLatent(_ZoneGradientBase):
    """
    Gaussian gradient anchored by the latent-BOS cosine-similarity quartile
    that best matches the current prompt.

    Custom init
    -----------
    sigma, floor : learnable (inherited).
    """

    def __init__(self, sigma: float = 0.30, floor: float = 0.04):
        super().__init__("L9_ZONE_LATENT", sigma, floor, "latent_bos_quartile")

    def forward(
        self,
        candidates:         List[Tuple[str, float]],
        prompt_words:       List[str],
        latent_sorted_keys: List,
        latent_bos_data:    Dict[str, List[str]],
    ) -> Dict:
        n_keys = len(latent_sorted_keys)
        q_key  = "q0"
        for ctx in latent_sorted_keys:
            if any(w in ctx for w in prompt_words):
                rank  = latent_sorted_keys.index(ctx)
                q_key = f"q{min(3, rank * 4 // max(1, n_keys))}"
                break

        zone_set = set(latent_bos_data.get(q_key, []))
        weights  = self._gradient(zone_set, candidates)
        return self._make_layer(weights, candidates, f"quartile={q_key}")


# ---------------------------------------------------------------------------
# L10 – History repetition column
# ---------------------------------------------------------------------------

class L10_History(nn.Module):
    """
    Produces a 1/(1+count) column for each candidate token based on how
    many times it has already appeared in the generated sequence.

    Custom init
    -----------
    smoothing : learnable additive offset in the denominator (clamped ≥ 0).
                Default 1.0 recovers the original formula.
    """

    def __init__(self, smoothing: float = 1.0):
        super().__init__()
        self.smoothing = nn.Parameter(torch.tensor(smoothing, dtype=torch.float64))

    def forward(
        self,
        candidates: List[Tuple[str, float]],
        history:    Counter,
    ) -> Dict:
        smooth = self.smoothing.clamp(min=0.0)
        counts = torch.tensor(
            [history[w] for w, _ in candidates], dtype=torch.float64
        )
        hist_vec = 1.0 / (smooth + counts)
        hist_vec = _normalise(hist_vec)

        words = [w for w, _ in candidates]
        return {
            "name":   "L10_HISTORY",
            "words":  words,
            "probs":  hist_vec.detach().numpy(),
            "source": f"repetition history smoothing={smooth.detach().item():.4f}",
        }


# ---------------------------------------------------------------------------
# L11 – Tensor blend of zone layers
# ---------------------------------------------------------------------------

class L11_TensorBlend(nn.Module):
    """
    Row-weighted vstack of L4–L10.  Each row's weight is its mean probability
    over the candidate set; the resulting column vector is the blended
    distribution.

    Custom init
    -----------
    zone_weights : learnable 7-vector of per-row mixing weights
                   (L4, L5, L6, L7, L8, L9, L10).  Initialised to all-ones
                   (uniform) and normalised in forward via softmax so they
                   always sum to 1.
    eps          : small constant buffer for numerical safety.
    """

    N_ROWS = 7  # L4 through L10

    def __init__(self, init_weights: Optional[List[float]] = None):
        super().__init__()
        if init_weights is None:
            init_weights = [1.0] * self.N_ROWS
        assert len(init_weights) == self.N_ROWS
        self.zone_weights = nn.Parameter(
            torch.tensor(init_weights, dtype=torch.float64)
        )
        self.register_buffer("eps", torch.tensor(1e-12, dtype=torch.float64))

    def forward(
        self,
        zone_layers: List[Dict],   # L4..L10 dicts
        candidates:  List[Tuple[str, float]],
    ) -> Dict:
        n = len(candidates)
        rows = []
        for layer in zone_layers:
            p = _to_tensor(layer["probs"])
            if p.shape[0] != n:
                p = F.pad(p, (0, n - p.shape[0]))[:n]
            rows.append(p.unsqueeze(0))                 # (1, n)

        zone_matrix = torch.cat(rows, dim=0)            # (N_ROWS, n)
        row_weights = F.softmax(self.zone_weights, dim=0)  # (N_ROWS,) summing to 1

        blended = (zone_matrix * row_weights.unsqueeze(1)).sum(dim=0)   # (n,)
        blended = _normalise(blended.clamp(min=float(self.eps)))

        words = [w for w, _ in candidates]
        return {
            "name":   "L11_TENSOR_BLEND",
            "words":  words,
            "probs":  blended.detach().numpy(),
            "source": (
                f"row-weighted blend of L4..L10 ({len(zone_layers)} rows) "
                f"softmax_weights={row_weights.tolist()}"
            ),
        }


# ---------------------------------------------------------------------------
# L12 – Final distribution (geometric mean of L3 and L11)
# ---------------------------------------------------------------------------

class L12_Final(nn.Module):
    """
    Geometric mean of the top-K/P distribution (L3) and the tensor-blend
    distribution (L11).

    Custom init
    -----------
    blend_alpha : learnable exponent for L3's contribution in the geometric
                  mean: p_final ∝ p_L3^α · p_L11^(1-α).
                  Initialised to 0.5 (equal weight); clamped to (0, 1).
    """

    def __init__(self, blend_alpha: float = 0.5):
        super().__init__()
        self.blend_alpha = nn.Parameter(torch.tensor(blend_alpha, dtype=torch.float64))

    def forward(
        self,
        L3_pairs: List[Tuple[str, float]],
        L11:      Dict,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        alpha = self.blend_alpha.clamp(min=1e-6, max=1.0 - 1e-6)
        beta  = 1.0 - alpha

        p3  = _to_tensor([p for _, p in L3_pairs]).clamp(min=1e-24)
        p11 = _to_tensor(L11["probs"]).clamp(min=1e-24)

        blended = p3.pow(alpha) * p11.pow(beta)
        blended = _normalise(blended)

        # Sort descending
        sorted_idx = torch.argsort(blended, descending=True)
        words_arr  = [L3_pairs[i][0] for i in sorted_idx.tolist()]
        probs_arr  = blended[sorted_idx]

        out = list(zip(words_arr, probs_arr.tolist()))
        layer = {
            "name":   "L12_FINAL",
            "words":  words_arr,
            "probs":  probs_arr.detach().numpy(),
            "source": f"geo_mean(L3^{alpha.detach().item():.3f}, L11^{beta.detach().item():.3f})",
        }
        return out, layer


# ---------------------------------------------------------------------------
# L13 – Contextual requestor position
# ---------------------------------------------------------------------------

class L13_CtxReqPos(nn.Module):
    """
    Encodes the pi-stream cursor position at the moment of each draw as a
    Gaussian gradient over the candidate set.  The normalised position
    [0, 1] drives the Gaussian centre so tokens at rank ≈ pos·(n-1) get
    the highest weight.

    Custom init
    -----------
    sigma : learnable Gaussian width (clamped > 0).
    floor : learnable minimum weight (clamped to [0, 1)).
    Both default to values used in the original code.
    """

    def __init__(self, sigma: float = 0.30, floor: float = 0.04):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float64))
        self.floor = nn.Parameter(torch.tensor(floor, dtype=torch.float64))

    def forward(
        self,
        candidates:  List[Tuple[str, float]],
        draw_pos:    int,
        stream_len:  int,
    ) -> Dict:
        n         = len(candidates)
        sigma     = self.sigma.clamp(min=1e-6)
        floor     = self.floor.clamp(min=0.0, max=1.0 - 1e-6)
        norm_pos  = (draw_pos % max(1, stream_len)) / max(1, stream_len - 1)

        indices  = torch.arange(n, dtype=torch.float64) / max(1, n - 1)
        gauss    = torch.exp(-0.5 * ((indices - norm_pos) / sigma) ** 2)
        weights  = floor + (1.0 - floor) * gauss
        weights  = _normalise(weights)

        words = [w for w, _ in candidates]
        return {
            "name":   "L13_CTX_REQ_POS",
            "words":  words,
            "probs":  weights.detach().numpy(),
            "source": (
                f"ctx_req_pos={draw_pos} "
                f"norm={norm_pos:.4f} "
                f"stream_len={stream_len}"
            ),
        }


# ---------------------------------------------------------------------------
# LayerFrame (unchanged data class)
# ---------------------------------------------------------------------------

class LayerFrame:
    """Container for one generation step's full layer stack."""
    __slots__ = (
        "step", "layers", "chosen", "context_window",
        "zone_name", "draw_pos", "next_draw_pos",
    )

    def __init__(
        self,
        step:           int,
        layers:         List[Dict],
        chosen:         str         = "",
        context_window: Tuple       = (),
        zone_name:      str         = "",
        draw_pos:       int         = 0,
        next_draw_pos:  int         = 0,
    ):
        self.step           = step
        self.layers         = layers
        self.chosen         = chosen
        self.context_window = context_window
        self.zone_name      = zone_name
        self.draw_pos       = draw_pos
        self.next_draw_pos  = next_draw_pos

    def get(self, name: str) -> Optional[Dict]:
        for layer in self.layers:
            if layer["name"] == name:
                return layer
        return None

    def tensor(self) -> torch.Tensor:
        rows = [_to_tensor(l["probs"]) for l in self.layers]
        if not rows:
            return torch.zeros(0, dtype=torch.float64)
        max_len = max(r.shape[0] for r in rows)
        padded  = [F.pad(r, (0, max_len - r.shape[0])) for r in rows]
        return torch.stack(padded)   # (n_layers, max_vocab)


# ---------------------------------------------------------------------------
# IsomorphismPipeline  – drop-in replacement for IsomorphismGenerator
# ---------------------------------------------------------------------------

class IsomorphismPipeline(nn.Module):
    """
    Full 14-layer isomorphic probability pipeline as a single nn.Module.

    Every layer is a child module discoverable by standard PyTorch tooling
    (parameters(), state_dict(), etc.).  The pipeline can be fine-tuned
    end-to-end if differentiable loss signals are available.

    Custom init
    -----------
    All hyper-parameters are forwarded to the corresponding layer modules.
    The layer modules register them as nn.Parameter (learnable) or as named
    buffers (discrete / non-differentiable values).

    Drop-in API
    -----------
    Identical to the original IsomorphismGenerator:
        gen = IsomorphismPipeline(cpd, context_index, vocab, ...)
        gen.seed_stream(stream)
        for frame in gen.generate(prompt, n_words, draw_fn): ...
    """

    LAYER_NAMES = [
        "L0_RAW_DIST",      "L1_TEMP_SCALED",   "L2_INSIGHT",
        "L3_TOPK_TOPP",     "L4_ZONE_FREQ",      "L5_ZONE_ALPHA",
        "L6_ZONE_BIGRAM",   "L7_ZONE_TRIGRAM",   "L8_ZONE_CHAR_TRIG",
        "L9_ZONE_LATENT",   "L10_HISTORY",        "L11_TENSOR_BLEND",
        "L12_FINAL",        "L13_CTX_REQ_POS",
    ]

    def __init__(
        self,
        cpd,
        context_index,
        vocab,
        ngram_n:         int   = 2,
        temperature:     float = 4.3,
        top_k:           int   = 100,
        top_p:           float = 1.0,
        rep_penalty:     float = 1.13,
        insight_penalty: float = 3.95,
        history:         Optional[Counter] = None,
        # ── per-layer custom init overrides ──────────────────────────
        l4_sigma: float = 0.50,   l4_floor: float = 0.05,
        l5_sigma: float = 0.40,   l5_floor: float = 0.05,
        l6_sigma: float = 0.25,   l6_floor: float = 0.04,
        l7_sigma: float = 0.20,   l7_floor: float = 0.03,
        l8_sigma: float = 0.35,   l8_floor: float = 0.04,
        l9_sigma: float = 0.30,   l9_floor: float = 0.04,
        l10_smoothing:    float = 1.0,
        l11_init_weights: Optional[List[float]] = None,
        l12_blend_alpha:  float = 0.5,
        l13_sigma: float = 0.30,  l13_floor: float = 0.04,
    ):
        super().__init__()

        # ── non-module state ──────────────────────────────────────────
        self.cpd            = cpd
        self.ctx_idx        = context_index
        self.vocab          = set(vocab)
        self.ngram_n        = max(2, int(ngram_n))
        self.context_window = self.ngram_n - 1
        self.history        = Counter(history) if history else Counter()
        self._step          = 0
        self._pos           = 0
        self._stream: List[int] = []
        self._char_trig_index: Dict[str, set] = (
            getattr(context_index, "_trig_index", {}) if context_index else {}
        )

        # ── child layer modules ───────────────────────────────────────
        self.l0  = L0_RawDist(rep_penalty=rep_penalty)
        self.l1  = L1_TempScaled(temperature=temperature)
        self.l2  = L2_InsightPenalty(insight_penalty=insight_penalty)
        self.l3  = L3_TopKTopP(top_k=top_k, top_p=top_p)
        self.l4  = L4_ZoneFreq(sigma=l4_sigma, floor=l4_floor)
        self.l5  = L5_ZoneAlpha(sigma=l5_sigma, floor=l5_floor)
        self.l6  = L6_ZoneBigram(sigma=l6_sigma, floor=l6_floor)
        self.l7  = L7_ZoneTrigram(sigma=l7_sigma, floor=l7_floor)
        self.l8  = L8_ZoneCharTrig(sigma=l8_sigma, floor=l8_floor)
        self.l9  = L9_ZoneLatent(sigma=l9_sigma, floor=l9_floor)
        self.l10 = L10_History(smoothing=l10_smoothing)
        self.l11 = L11_TensorBlend(init_weights=l11_init_weights)
        self.l12 = L12_Final(blend_alpha=l12_blend_alpha)
        self.l13 = L13_CtxReqPos(sigma=l13_sigma, floor=l13_floor)

    # ── internal helpers ──────────────────────────────────────────────

    def _dist_for_ctx(self, ctx_tuple):
        for cut in range(len(ctx_tuple), 0, -1):
            trial = ("",) * (self.context_window - cut) + ctx_tuple[-cut:]
            try:
                d = self.cpd[trial]
                if list(d.samples()):
                    return d
            except Exception:
                continue
        try:
            d = self.cpd[("",) * self.context_window]
            if list(d.samples()):
                return d
        except Exception:
            pass
        return None

    # ── public API ────────────────────────────────────────────────────

    def seed_stream(self, stream: list):
        """Attach the raw pi-stream so L13 can read its length & cursor."""
        self._stream = list(stream)
        self._pos    = 0

    def step(
        self,
        context_deque: deque,
        prompt_words:  List[str],
        draw:          float,
        zone_name:     str = "",
    ) -> Optional[LayerFrame]:
        dist = self._dist_for_ctx(tuple(context_deque))
        if dist is None:
            return None

        # ── forward pass through the 14 layers ───────────────────────
        L0_pairs, L0 = self.l0(dist, self.history)
        if not L0_pairs:
            return None

        L1_pairs, L1 = self.l1(L0_pairs)
        L2_pairs, L2 = self.l2(L1_pairs)
        L3_pairs, L3 = self.l3(L2_pairs)
        if not L3_pairs:
            return None

        ci = self.ctx_idx

        if ci is None:
            # Fallback: uniform zone layers
            flat = _normalise(torch.ones(len(L3_pairs), dtype=torch.float64))
            flat_np = flat.detach().numpy()
            words = [w for w, _ in L3_pairs]
            zone_layers = [
                {"name": n, "words": words, "probs": flat_np.copy(),
                 "source": "no context_index"}
                for n in ["L4_ZONE_FREQ", "L5_ZONE_ALPHA", "L6_ZONE_BIGRAM",
                           "L7_ZONE_TRIGRAM", "L8_ZONE_CHAR_TRIG", "L9_ZONE_LATENT"]
            ]
        else:
            L4 = self.l4(L3_pairs, prompt_words, ci.freq_zones, ci.token_freq)
            L5 = self.l5(L3_pairs, prompt_words, ci.alpha_zones)
            L6 = self.l6(L3_pairs, prompt_words, ci.ngram_zones)
            L7 = self.l7(L3_pairs, context_deque, ci.ngram_zones)
            L8 = self.l8(L3_pairs, prompt_words, self._char_trig_index)
            L9 = self.l9(
                L3_pairs, prompt_words,
                ci.latent_sorted_keys, ci.latent_bos_data,
            )
            zone_layers = [L4, L5, L6, L7, L8, L9]

        L10 = self.l10(L3_pairs, self.history)
        L11 = self.l11(zone_layers + [L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)

        # ── L13: contextual requestor position ────────────────────────
        draw_pos   = self._pos
        stream_len = max(1, len(self._stream))
        L13 = self.l13(L3_pairs, draw_pos, stream_len)

        # Geometric blend of L12 and L13
        l12_map = dict(L12_pairs)
        l13_map = dict(zip(L13["words"], L13["probs"].tolist()))
        all_words = list(l12_map.keys())
        floor_val = 1e-12

        blended = [
            (w, math.sqrt(
                max(floor_val, l12_map.get(w, floor_val)) *
                max(floor_val, l13_map.get(w, floor_val))
            ))
            for w in all_words
        ]
        bt      = sum(p for _, p in blended)
        blended = [(w, p / bt) for w, p in blended] if bt > 0 else blended

        # Prefer unseen tokens
        unseen = [(w, p) for w, p in blended if self.history[w] == 0]
        pool   = unseen if unseen else blended
        t      = sum(p for _, p in pool)
        pool   = [(w, p / t) for w, p in pool] if t > 0 else pool

        # Sample
        chosen, cumulative = pool[-1][0], 0.0
        for w, p in pool:
            cumulative += p
            if draw < cumulative:
                chosen = w
                break

        self.history[chosen] += 1

        # Advance cursor
        next_draw_pos = (draw_pos + (draw_pos % max(1, stream_len))) % stream_len
        self._pos     = next_draw_pos
        self._step   += 1

        return LayerFrame(
            step           = self._step - 1,
            layers         = [L0, L1, L2, L3] + zone_layers + [L10, L11, L12, L13],
            chosen         = chosen,
            context_window = tuple(context_deque),
            zone_name      = zone_name,
            draw_pos       = draw_pos,
            next_draw_pos  = next_draw_pos,
        )

    def generate(self, prompt: str, n_words: int, draw_fn, zone_fn=None):
        """
        Identical signature to the original IsomorphismGenerator.generate().
        Yields LayerFrame objects.
        """
        tokens       = [w.lower() for w in prompt.split() if w.isalpha()]
        vocab_tokens = [w for w in tokens if w in self.vocab]

        if len(vocab_tokens) >= self.context_window:
            init = vocab_tokens[-self.context_window:]
        else:
            init = [""] * (self.context_window - len(vocab_tokens)) + vocab_tokens

        ctx = deque(init, maxlen=self.context_window)

        for _ in range(n_words):
            zone_name = zone_fn(draw_fn()) if zone_fn is not None else ""
            draw      = draw_fn()
            frame     = self.step(ctx, tokens, draw, zone_name=zone_name)
            if frame is None:
                ctx.clear()
                ctx.extend([""] * self.context_window)
                continue
            ctx.append(frame.chosen)
            yield frame

    # ── PyTorch convenience ───────────────────────────────────────────

    def param_summary(self) -> str:
        """
        Returns a human-readable table of all learnable parameters with
        their current values — handy for debugging or logging.
        """
        lines = [f"{'Parameter':<45} {'Value':>14}"]
        lines.append("-" * 61)
        for name, param in self.named_parameters():
            v = param.data
            if v.numel() == 1:
                lines.append(f"  {name:<43} {v.item():>14.6f}")
            else:
                lines.append(
                    f"  {name:<43} shape={list(v.shape)}  "
                    f"mean={v.mean().item():.4f}"
                )
        return "\n".join(lines)

    # ── last-run frame store ──────────────────────────────────────────
    #    Populated by generate_text(); accessible as pipeline.frames
    frames: List[LayerFrame] = []

    @staticmethod
    def _frames_to_text(
        frames:        List[LayerFrame],
        prompt:        str  = "",
        capitalise:    bool = True,
        include_prompt: bool = True,
    ) -> str:
        """
        Derive a plain string from a list of LayerFrame objects.

        This is the canonical text-assembly routine used internally by
        ``generate_text()``.  It is also exposed as a static method so
        callers can re-render text from a saved frame list without
        re-running the pipeline:

            text = IsomorphismPipeline._frames_to_text(
                pipeline.frames, prompt="alice rabbit"
            )

        Parameters
        ----------
        frames : list[LayerFrame]
            Ordered frames from a ``generate()`` run.  The text is built
            by joining ``frame.chosen`` for every frame that has a
            non-empty ``.chosen`` value.
        prompt : str
            Original prompt.  Prepended when ``include_prompt=True``.
        capitalise : bool
            Capitalise sentence starts (first word and words after . ! ?).
        include_prompt : bool
            Whether to include the prompt words before the generated ones.

        Returns
        -------
        str
        """
        gen_words = [f.chosen for f in frames if f.chosen]

        prompt_words: List[str] = (
            prompt.strip().split()
            if include_prompt and prompt.strip()
            else []
        )
        all_words = prompt_words + gen_words

        if not all_words:
            return ""

        if not capitalise:
            return " ".join(all_words)

        result: List[str] = []
        cap_next = True
        for w in all_words:
            result.append(w.capitalize() if cap_next else w)
            cap_next = bool(w.rstrip("\"'")[-1:] in {".", "!", "?"})
        return " ".join(result)

    def _make_draw_fn(
        self,
        stream:            Optional[List[int]],
        digits_per_sample: int,
        seed:              Optional[int],
    ):
        """
        Build and return a ``draw_fn`` (``() -> float in [0,1)``) from
        whichever source is available: an explicit stream argument, a
        previously attached stream, or a PRNG fallback.
        """
        if stream is not None:
            self.seed_stream(stream)

        active_stream = getattr(self, "_stream", [])

        if active_stream:
            pos   = [self._pos]
            dps   = max(1, int(digits_per_sample))
            s_len = len(active_stream)

            def _draw_pi() -> float:
                val  = 0
                base = 26 ** dps
                for _ in range(dps):
                    val    = val * 26 + active_stream[pos[0] % s_len]
                    pos[0] = (pos[0] + 1) % s_len
                self._pos = pos[0]
                return val / base

            return _draw_pi

        import random as _random
        rng = _random.Random(seed)
        return rng.random

    def generate_text(
        self,
        prompt:     str,
        n_words:    int,
        stream:     Optional[List[int]] = None,
        *,
        digits_per_sample: int          = 3,
        seed:              Optional[int] = None,
        capitalise:        bool          = True,
        include_prompt:    bool          = True,
        zone_fn                          = None,
    ) -> str:
        """
        Generate text and return it as a plain string.

        Internally collects every ``LayerFrame`` yielded by ``generate()``
        into ``self.frames`` before assembling the final string via
        ``_frames_to_text()``.  The frames remain available after the call:

            text = pipeline.generate_text("alice rabbit", n_words=40)
            # inspect the tensor for the third generated token:
            print(pipeline.frames[2].tensor())
            # re-render text from stored frames at any time:
            text2 = IsomorphismPipeline._frames_to_text(
                pipeline.frames, prompt="alice rabbit"
            )

        Parameters
        ----------
        prompt : str
            Seed text.
        n_words : int
            Number of tokens to generate.
        stream : list[int] | None
            Raw pi-stream.  If omitted, uses the stream attached via
            ``seed_stream()``.  Falls back to a PRNG seeded by ``seed``.
        digits_per_sample : int
            Stream positions consumed per draw (default 3).
        seed : int | None
            PRNG seed used when no stream is available.
        capitalise : bool
            Auto-capitalise sentence starts.
        include_prompt : bool
            Prepend prompt words to the returned string.
        zone_fn : callable | None
            Optional ``float -> str`` zone mapper passed to ``generate()``.

        Returns
        -------
        str
            The assembled text derived from ``self.frames``.
        """
        draw_fn = self._make_draw_fn(stream, digits_per_sample, seed)

        # ── collect every frame — this is the source of truth ─────────
        self.frames = list(
            self.generate(
                prompt  = prompt,
                n_words = n_words,
                draw_fn = draw_fn,
                zone_fn = zone_fn,
            )
        )

        # ── derive text purely from the frames ─────────────────────────
        return self._frames_to_text(
            self.frames,
            prompt         = prompt,
            capitalise     = capitalise,
            include_prompt = include_prompt,
        )


# ---------------------------------------------------------------------------
# Real corpus builder  (used by __main__ and importable for testing)
# ---------------------------------------------------------------------------

def build_real_cpd(
    corpus: str,
    ngram_n: int = 2,
    lidstone_gamma: float = 0.1,
):
    """
    Build a genuine NLTK ConditionalProbDist + vocab from raw text.

    Returns (cpd, vocab, tokens) — identical types to what app.py produces.
    All heavy lifting is done with the same NLTK primitives so the
    IsomorphismPipeline sees exactly the same distribution objects it
    would encounter in production.
    """
    import os, sys
    import nltk
    from nltk.util import ngrams as nltk_ngrams
    from nltk.probability import (
        ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist,
    )

    NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)
    for pkg, path in [("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)

    tokens = corpus.lower().split()
    if not tokens:
        raise ValueError("Corpus produced zero tokens.")

    ngram_n = max(2, int(ngram_n))
    padded  = [""] * (ngram_n - 1) + tokens + [""]
    all_ng  = list(nltk_ngrams(padded, ngram_n))

    cfd   = ConditionalFreqDist((tuple(ng[:-1]), ng[-1]) for ng in all_ng)
    vocab = set(tokens) | {""}

    class _LidFactory:
        def __init__(self, gamma, bins):
            self.gamma = gamma; self.bins = bins
        def __call__(self, fd):
            return LidstoneProbDist(fd, gamma=self.gamma, bins=self.bins)

    cpd = ConditionalProbDist(
        cfd,
        _LidFactory(gamma=float(lidstone_gamma), bins=max(1, len(vocab))),
    )
    return cpd, vocab, tokens


def build_real_context_index(vocab, cpd, tokens):
    """
    Build a ContextZoneIndex from real corpus tokens.

    Requires app.py to be importable (or the ContextZoneIndex class to be
    defined in this file).  Falls back gracefully to None if unavailable.
    """
    try:
        import sys, os
        # Try importing from app.py sitting beside this file
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from app import ContextZoneIndex
        from collections import Counter
        return ContextZoneIndex(vocab, cpd, Counter(tokens))
    except Exception as e:
        print(f"  [context_index] not available ({e}); zone layers will use uniform weights.")
        return None


# ---------------------------------------------------------------------------
# Smoke-test / CLI entry point
# ---------------------------------------------------------------------------

_EMBEDDED_CORPUS = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it.
So she was considering in her own mind whether the pleasure of making a
daisy chain would be worth the trouble of getting up and picking the daisies.
Suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so very remarkable in that, nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear!
I shall be late! When the Rabbit actually took a watch out of its waistcoat
pocket and looked at it and hurried on, Alice started to her feet.
The rabbit hole went straight on like a tunnel for some way and then dipped
suddenly down. Either the well was very deep or she fell very slowly,
for she had plenty of time as she went down to look about her and wonder
what was going to happen next. She tried to look down and make out what she
was coming to, but it was too dark to see anything. Then she looked at the
sides of the well and noticed that they were filled with cupboards and
bookshelves. Here and there she saw maps and pictures hung upon pegs.
She took down a jar from one of the shelves as she passed. The jar was
labelled Orange Marmalade but to her great disappointment it was empty.
"""


# =============================================================================
#  gradio_pipeline.py
#  Gradio UI for IsomorphismPipeline (layer_isomorphism_torch.py)
#
#  Tabs
#  ────
#  Generate      — corpus, prompt, params → text output + per-step summary
#  Layer Inspector — pick any generated step, see all 14 layers side-by-side
#  Parameters    — live param table; adjust & re-apply without rebuilding CPD
#  Model I/O     — save / load pipeline state_dict
# =============================================================================


import os
import sys
import json
import random
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import gradio as gr
import torch

# ---------------------------------------------------------------------------
# Make layer_isomorphism_torch importable whether this file lives beside it
# or is run from a different cwd.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Global state  (single-user demo; replace with session state for multi-user)
# ---------------------------------------------------------------------------
STATE: dict = {
    "pipeline":   None,   # IsomorphismPipeline
    "cpd":        None,
    "vocab":      None,
    "tokens":     None,
    "ctx_idx":    None,
    "frames":     [],     # List[LayerFrame] from last generate_text() call
    "prompt":     "",
    "corpus_src": "",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_corpus_and_pipeline(
    corpus_text: str,
    ngram_n: int,
    gamma: float,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    insight_penalty: float,
    l12_blend_alpha: float,
) -> str:
    """Build CPD + context index + pipeline. Returns status log string."""
    log = []
    try:
        log.append(f"Building {ngram_n}-gram CPD (γ={gamma}) …")
        cpd, vocab, tokens = build_real_cpd(corpus_text, ngram_n, gamma)
        log.append(f"  vocab={len(vocab)}  tokens={len(tokens)}")

        log.append("Building ContextZoneIndex …")
        ctx_idx = build_real_context_index(vocab, cpd, tokens)
        log.append(
            "  ContextZoneIndex ready."
            if ctx_idx is not None
            else "  app.py not found — zone layers will be uniform."
        )

        pipeline = IsomorphismPipeline(
            cpd             = cpd,
            context_index   = ctx_idx,
            vocab           = vocab,
            ngram_n         = ngram_n,
            temperature     = temperature,
            top_k           = top_k,
            top_p           = top_p,
            rep_penalty     = rep_penalty,
            insight_penalty = insight_penalty,
            l12_blend_alpha = l12_blend_alpha,
        )

        STATE.update(
            pipeline   = pipeline,
            cpd        = cpd,
            vocab      = vocab,
            tokens     = tokens,
            ctx_idx    = ctx_idx,
            corpus_src = f"{len(corpus_text)} chars, {len(tokens)} tokens",
        )
        log.append("Pipeline ready ✓")
    except Exception as e:
        log.append(f"ERROR: {e}")
        log.append(traceback.format_exc())
    return "\n".join(log)


def _require_pipeline() -> tuple[Optional[IsomorphismPipeline], str]:
    p = STATE.get("pipeline")
    if p is None:
        return None, "⚠ Build the model first (Model tab)."
    return p, ""


# ---------------------------------------------------------------------------
# Tab: Model / Corpus
# ---------------------------------------------------------------------------

def tab_model_build(
    corpus_file,
    pasted_corpus: str,
    ngram_n: int,
    gamma: float,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    insight_penalty: float,
    l12_blend_alpha: float,
):
    if corpus_file is not None:
        path = corpus_file if isinstance(corpus_file, str) else corpus_file.name
        try:
            corpus_text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"Could not read file: {e}", "", ""
    elif pasted_corpus and pasted_corpus.strip():
        corpus_text = pasted_corpus.strip()
    else:
        corpus_text = _EMBEDDED_CORPUS.strip()

    log = _build_corpus_and_pipeline(
        corpus_text, ngram_n, gamma, temperature, top_k, top_p,
        rep_penalty, insight_penalty, l12_blend_alpha,
    )
    stats = STATE["corpus_src"]
    return log, stats, ""


# ---------------------------------------------------------------------------
# Tab: Generate
# ---------------------------------------------------------------------------

def tab_generate(
    prompt: str,
    n_words: int,
    seed: int,
    capitalise: bool,
    include_prompt: bool,
):
    pipeline, err = _require_pipeline()
    if pipeline is None:
        return err, "", "", []

    # Reset history so each run is fresh
    pipeline.history.clear()
    pipeline._step = 0
    pipeline._pos  = 0

    rng = random.Random(seed)
    try:
        text = pipeline.generate_text(
            prompt,
            n_words        = n_words,
            seed           = seed,
            capitalise     = capitalise,
            include_prompt = include_prompt,
            zone_fn        = None,
        )
    except Exception as e:
        return f"Generation error: {e}\n{traceback.format_exc()}", "", "", []

    STATE["frames"] = pipeline.frames
    STATE["prompt"] = prompt

    # ── per-step summary table ────────────────────────────────────────
    rows = []
    for i, f in enumerate(pipeline.frames):
        l12 = f.get("L12_FINAL")
        top2 = ""
        if l12 and l12["words"]:
            idx = int(l12["probs"].argmax())
            top2 = f"{l12['words'][idx]} ({l12['probs'][idx]:.3f})"
        rows.append([
            i,
            f.chosen,
            f.zone_name or "—",
            f.draw_pos,
            top2,
            ",".join(w for w in f.context_window if w) or "BOS",
        ])

    headers = ["Step", "Chosen", "Zone", "Draw pos", "L12 top", "Context"]
    step_choices = [str(i) for i in range(len(pipeline.frames))]

    return (
        text,
        f"{len(pipeline.frames)} frames stored.",
        _dataframe_md(headers, rows),
        gr.update(choices=step_choices, value=step_choices[0] if step_choices else None),
    )


def _dataframe_md(headers, rows) -> str:
    if not rows:
        return "_No data_"
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    lines = [fmt(headers), sep] + [fmt(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab: Layer Inspector
# ---------------------------------------------------------------------------

def tab_inspect_step(step_str: str):
    frames = STATE.get("frames", [])
    if not frames:
        return "_No frames — run Generate first._", ""

    try:
        idx = int(step_str)
    except (ValueError, TypeError):
        idx = 0
    idx = max(0, min(idx, len(frames) - 1))

    f = frames[idx]
    meta = (
        f"**Step {f.step}** · chosen=`{f.chosen}` · "
        f"zone=`{f.zone_name or '—'}` · draw_pos={f.draw_pos} · "
        f"ctx={f.context_window}"
    )

    # Per-layer table: top-5 tokens for each layer
    rows = []
    for layer in f.layers:
        words = layer["words"]
        probs = layer["probs"]
        if len(words) == 0:
            rows.append([layer["name"], "—", "—", "—", "—", "—", layer["source"][:60]])
            continue
        top_n   = min(5, len(words))
        top_idx = np.argsort(probs)[::-1][:top_n]
        tops    = "  ".join(f"{words[i]}:{probs[i]:.3f}" for i in top_idx)
        chosen_p = probs[words.index(f.chosen)] if f.chosen in words else 0.0
        rows.append([
            layer["name"],
            tops,
            f"{chosen_p:.4f}",
            f"{probs.max():.4f}",
            f"{probs.min():.4f}",
            f"{float(np.std(probs)):.4f}",
            layer["source"][:60],
        ])

    headers = ["Layer", "Top-5 tokens", "chosen_p", "max_p", "min_p", "std_p", "Source"]
    table_md = _dataframe_md(headers, rows)

    # Tensor shape info
    t      = f.tensor()
    tensor_info = f"Tensor shape: `{list(t.shape)}` (layers × vocab)"

    return meta + "\n\n" + tensor_info, table_md


# ---------------------------------------------------------------------------
# Tab: Parameters
# ---------------------------------------------------------------------------

def tab_params_show():
    pipeline, err = _require_pipeline()
    if pipeline is None:
        return err
    return pipeline.param_summary()


def tab_params_update(
    rep_penalty:     float,
    temperature:     float,
    insight_penalty: float,
    top_p:           float,
    l10_smoothing:   float,
    l12_blend_alpha: float,
    l4_sigma: float, l4_floor: float,
    l5_sigma: float, l5_floor: float,
    l6_sigma: float, l6_floor: float,
    l7_sigma: float, l7_floor: float,
    l8_sigma: float, l8_floor: float,
    l9_sigma: float, l9_floor: float,
    l13_sigma: float, l13_floor: float,
):
    pipeline, err = _require_pipeline()
    if pipeline is None:
        return err

    with torch.no_grad():
        pipeline.l0.rep_penalty.copy_(torch.tensor(rep_penalty,     dtype=torch.float64))
        pipeline.l1.temperature.copy_(torch.tensor(temperature,     dtype=torch.float64))
        pipeline.l2.insight_penalty.copy_(torch.tensor(insight_penalty, dtype=torch.float64))
        pipeline.l3.top_p.copy_(torch.tensor(top_p,               dtype=torch.float64))
        pipeline.l10.smoothing.copy_(torch.tensor(l10_smoothing,   dtype=torch.float64))
        pipeline.l12.blend_alpha.copy_(torch.tensor(l12_blend_alpha, dtype=torch.float64))
        pipeline.l4.sigma.copy_(torch.tensor(l4_sigma, dtype=torch.float64))
        pipeline.l4.floor.copy_(torch.tensor(l4_floor, dtype=torch.float64))
        pipeline.l5.sigma.copy_(torch.tensor(l5_sigma, dtype=torch.float64))
        pipeline.l5.floor.copy_(torch.tensor(l5_floor, dtype=torch.float64))
        pipeline.l6.sigma.copy_(torch.tensor(l6_sigma, dtype=torch.float64))
        pipeline.l6.floor.copy_(torch.tensor(l6_floor, dtype=torch.float64))
        pipeline.l7.sigma.copy_(torch.tensor(l7_sigma, dtype=torch.float64))
        pipeline.l7.floor.copy_(torch.tensor(l7_floor, dtype=torch.float64))
        pipeline.l8.sigma.copy_(torch.tensor(l8_sigma, dtype=torch.float64))
        pipeline.l8.floor.copy_(torch.tensor(l8_floor, dtype=torch.float64))
        pipeline.l9.sigma.copy_(torch.tensor(l9_sigma, dtype=torch.float64))
        pipeline.l9.floor.copy_(torch.tensor(l9_floor, dtype=torch.float64))
        pipeline.l13.sigma.copy_(torch.tensor(l13_sigma, dtype=torch.float64))
        pipeline.l13.floor.copy_(torch.tensor(l13_floor, dtype=torch.float64))

    return pipeline.param_summary()


# ---------------------------------------------------------------------------
# Tab: Model I/O
# ---------------------------------------------------------------------------

def tab_save_model():
    pipeline, err = _require_pipeline()
    if pipeline is None:
        return None, err

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pt", prefix="isomorphism_pipeline_"
    )
    tmp.close()
    torch.save(pipeline.state_dict(), tmp.name)
    return tmp.name, f"Saved state_dict to {Path(tmp.name).name}"


def tab_load_model(model_file):
    pipeline, err = _require_pipeline()
    if pipeline is None:
        return f"Build the model first, then load weights.\n{err}"
    if model_file is None:
        return "No file uploaded."
    path = model_file if isinstance(model_file, str) else model_file.name
    try:
        sd = torch.load(path, map_location="cpu")
        pipeline.load_state_dict(sd, strict=False)
        return f"Loaded weights from {Path(path).name}\n\n" + pipeline.param_summary()
    except Exception as e:
        return f"Load failed: {e}"


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'JetBrains Mono', 'Fira Mono', monospace; }
.output-text textarea { font-family: Georgia, serif; font-size: 1.05em; }
.layer-table { font-size: 0.78em; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="IsomorphismPipeline", css=_CSS, theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# IsomorphismPipeline\n"
            "14-layer isomorphic probability tensor · PyTorch n-gram generator"
        )

        # ════════════════════════════════════════════════════════════
        with gr.Tabs():

            # ── Tab 1: Model ─────────────────────────────────────────
            with gr.TabItem("⚙ Model"):
                gr.Markdown("### Corpus & Model Configuration")
                gr.Markdown(
                    "Upload a `.txt` file **or** paste text below. "
                    "Leave both empty to use the built-in Alice excerpt."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        m_file   = gr.File(label="Upload corpus (.txt)", file_types=[".txt", ".md"], type="filepath")
                        m_pasted = gr.Textbox(label="Paste corpus", lines=6,
                                              placeholder="Paste plain text here …")
                    with gr.Column(scale=1):
                        m_ngram   = gr.Slider(2, 5,   value=2,   step=1,     label="N-gram order")
                        m_gamma   = gr.Slider(0.001, 1.0, value=0.1, step=0.001, label="Lidstone γ")
                        m_temp    = gr.Slider(0.1, 15.0, value=4.3, step=0.05,  label="Temperature")
                        m_topk    = gr.Slider(1, 200,  value=40,  step=1,     label="Top-K")
                        m_topp    = gr.Slider(0.01, 1.0, value=0.95, step=0.01, label="Top-P")
                        m_rep     = gr.Slider(1.0, 10.0, value=1.13, step=0.01, label="Rep penalty")
                        m_insight = gr.Slider(0.0, 15.0, value=3.95, step=0.05, label="Insight penalty")
                        m_alpha   = gr.Slider(0.01, 0.99, value=0.5, step=0.01, label="L12 blend α")

                m_build_btn = gr.Button("Build Model", variant="primary")
                m_log       = gr.Textbox(label="Build log", lines=8, interactive=False)
                m_stats     = gr.Textbox(label="Corpus stats", interactive=False)
                m_err       = gr.Textbox(visible=False)

                m_build_btn.click(
                    tab_model_build,
                    inputs=[m_file, m_pasted, m_ngram, m_gamma,
                            m_temp, m_topk, m_topp, m_rep, m_insight, m_alpha],
                    outputs=[m_log, m_stats, m_err],
                )

            # ── Tab 2: Generate ──────────────────────────────────────
            with gr.TabItem("✍ Generate"):
                gr.Markdown("### Text Generation")
                with gr.Row():
                    with gr.Column(scale=1):
                        g_prompt   = gr.Textbox(label="Prompt", value="alice rabbit hole", lines=2)
                        g_words    = gr.Slider(1, 2500, value=60, step=1, label="Words to generate")
                        g_seed     = gr.Number(label="PRNG seed", value=42, precision=0)
                        g_caps     = gr.Checkbox(label="Capitalise sentence starts", value=True)
                        g_inc_pr   = gr.Checkbox(label="Include prompt in output", value=True)
                        g_run_btn  = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        g_out_text  = gr.Textbox(label="Generated text", lines=10,
                                                  elem_classes=["output-text"])
                        g_frame_info = gr.Textbox(label="Frame store", interactive=False, lines=1)

                g_step_dd   = gr.Dropdown(
                    label="Jump to Layer Inspector step",
                    choices=[], interactive=True,
                )
                g_step_table = gr.Markdown(label="Per-step summary", elem_classes=["layer-table"])

                g_run_btn.click(
                    tab_generate,
                    inputs=[g_prompt, g_words, g_seed, g_caps, g_inc_pr],
                    outputs=[g_out_text, g_frame_info, g_step_table, g_step_dd],
                )

            # ── Tab 3: Layer Inspector ───────────────────────────────
            with gr.TabItem("🔬 Layer Inspector"):
                gr.Markdown(
                    "### Per-step Layer Stack\n"
                    "Select a generation step to inspect all 14 layers side-by-side."
                )
                with gr.Row():
                    li_step_num = gr.Number(label="Step index", value=0, precision=0, minimum=0)
                    li_go_btn   = gr.Button("Inspect", variant="secondary")

                li_meta  = gr.Markdown()
                li_table = gr.Markdown(elem_classes=["layer-table"])

                li_go_btn.click(
                    tab_inspect_step,
                    inputs=[li_step_num],
                    outputs=[li_meta, li_table],
                )

                # Clicking dropdown in Generate tab populates the step number here
                g_step_dd.change(
                    fn=lambda s: int(s) if s else 0,
                    inputs=[g_step_dd],
                    outputs=[li_step_num],
                )

            # ── Tab 4: Parameters ────────────────────────────────────
            with gr.TabItem("🎛 Parameters"):
                gr.Markdown(
                    "### Live Parameter Adjustment\n"
                    "Modify learnable parameters in-place without rebuilding the model. "
                    "Click **Apply** then re-generate to hear the effect."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Core**")
                        p_rep     = gr.Slider(1.0, 10.0, value=1.13, step=0.01, label="rep_penalty (L0)")
                        p_temp    = gr.Slider(0.1, 15.0, value=4.3,  step=0.05, label="temperature (L1)")
                        p_insight = gr.Slider(0.0, 15.0, value=3.95, step=0.05, label="insight_penalty (L2)")
                        p_topp    = gr.Slider(0.01, 1.0, value=0.95, step=0.01, label="top_p (L3)")
                        p_smooth  = gr.Slider(0.0, 5.0,  value=1.0,  step=0.05, label="smoothing (L10)")
                        p_balpha  = gr.Slider(0.01, 0.99,value=0.5,  step=0.01, label="blend_alpha (L12)")
                    with gr.Column():
                        gr.Markdown("**Zone σ / floor**")
                        p_l4s = gr.Slider(0.01, 2.0, value=0.50, step=0.01, label="L4 σ")
                        p_l4f = gr.Slider(0.00, 0.5, value=0.05, step=0.01, label="L4 floor")
                        p_l5s = gr.Slider(0.01, 2.0, value=0.40, step=0.01, label="L5 σ")
                        p_l5f = gr.Slider(0.00, 0.5, value=0.05, step=0.01, label="L5 floor")
                        p_l6s = gr.Slider(0.01, 2.0, value=0.25, step=0.01, label="L6 σ")
                        p_l6f = gr.Slider(0.00, 0.5, value=0.04, step=0.01, label="L6 floor")
                        p_l7s = gr.Slider(0.01, 2.0, value=0.20, step=0.01, label="L7 σ")
                        p_l7f = gr.Slider(0.00, 0.5, value=0.03, step=0.01, label="L7 floor")
                    with gr.Column():
                        gr.Markdown("**Zone σ / floor (cont.)**")
                        p_l8s  = gr.Slider(0.01, 2.0, value=0.35, step=0.01, label="L8 σ")
                        p_l8f  = gr.Slider(0.00, 0.5, value=0.04, step=0.01, label="L8 floor")
                        p_l9s  = gr.Slider(0.01, 2.0, value=0.30, step=0.01, label="L9 σ")
                        p_l9f  = gr.Slider(0.00, 0.5, value=0.04, step=0.01, label="L9 floor")
                        p_l13s = gr.Slider(0.01, 2.0, value=0.30, step=0.01, label="L13 σ")
                        p_l13f = gr.Slider(0.00, 0.5, value=0.04, step=0.01, label="L13 floor")

                p_apply_btn = gr.Button("Apply Parameters", variant="primary")
                p_show_btn  = gr.Button("Refresh Summary", variant="secondary")
                p_summary   = gr.Textbox(label="Parameter summary", lines=28, interactive=False)

                p_inputs = [
                    p_rep, p_temp, p_insight, p_topp, p_smooth, p_balpha,
                    p_l4s, p_l4f, p_l5s, p_l5f, p_l6s, p_l6f,
                    p_l7s, p_l7f, p_l8s, p_l8f, p_l9s, p_l9f,
                    p_l13s, p_l13f,
                ]

                p_apply_btn.click(tab_params_update, inputs=p_inputs, outputs=[p_summary])
                p_show_btn.click(tab_params_show,   inputs=[],        outputs=[p_summary])

            # ── Tab 5: Model I/O ─────────────────────────────────────
            with gr.TabItem("💾 Model I/O"):
                gr.Markdown(
                    "### Save / Load\n"
                    "Saves and loads the PyTorch `state_dict` (all 27 learnable parameters). "
                    "The corpus and CPD are **not** saved — rebuild the model first, then load weights."
                )
                with gr.Row():
                    with gr.Column():
                        io_save_btn  = gr.Button("Save state_dict (.pt)", variant="primary")
                        io_save_file = gr.File(label="Download", interactive=False)
                        io_save_log  = gr.Textbox(label="Save log", lines=2, interactive=False)

                    with gr.Column():
                        io_load_file = gr.File(
                            label="Upload .pt file",
                            file_types=[".pt", ".pth"],
                            type="filepath",
                        )
                        io_load_btn = gr.Button("Load weights", variant="secondary")
                        io_load_log = gr.Textbox(label="Load log", lines=20, interactive=False)

                io_save_btn.click(tab_save_model, inputs=[], outputs=[io_save_file, io_save_log])
                io_load_btn.click(tab_load_model, inputs=[io_load_file], outputs=[io_load_log])

        gr.Markdown(
            "_Build the model first (⚙ Model tab), then generate (✍ Generate), "
            "then inspect individual steps (🔬 Layer Inspector)._"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gradio UI for IsomorphismPipeline")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_ui().queue(max_size=4).launch(
        server_name = args.host,
        server_port = args.port,
        share       = args.share,
        show_error  = True,
    )
