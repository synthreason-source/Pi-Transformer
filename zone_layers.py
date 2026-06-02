"""
zone_layers.py
==============
Zone layers L4–L9 and L14 refactored onto the vectorised granule geometry
kernels from granule_geometry.py.

Each layer is a thin nn.Module wrapper that
  1. Selects the relevant granule set from the context index
  2. Delegates all geometry to zone_dist_from_granules()
  3. Returns (pairs, layerdict) exactly as the original pipeline expects
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from granule_geometry import CandidateMatrix, zone_dist_from_granules, EPS


def _cand(pairs) -> CandidateMatrix:
    return CandidateMatrix.from_pairs(pairs)


def _char_trigrams(w: str):
    return {w[i:i+3] for i in range(len(w) - 2)} if len(w) >= 3 else {w}


# ─────────────────────────────────────────────────────────────────────────────
class ZoneBase(nn.Module):
    """Shared sigma / floor parameters for every geometry zone layer."""
    def __init__(self, name: str, sigma: float, floor: float):
        super().__init__()
        self.layer_name = name
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float64))
        self.floor = nn.Parameter(torch.tensor(floor, dtype=torch.float64))

    def _run(self, pairs, granules):
        return zone_dist_from_granules(
            self.layer_name, _cand(pairs), granules, self.sigma, self.floor
        )


# ─────────────────────────────────────────────────────────────────────────────
class L4_ZoneFreq(ZoneBase):
    """Frequency-band zone: selects high / mid / low frequency granule."""
    def __init__(self, sigma=0.50, floor=0.05):
        super().__init__("L4_ZONE_FREQ", sigma, floor)

    def forward(self, pairs, prompt_words, freq_zones, token_freq):
        high = set(freq_zones.get("high", []))
        mid_thresh = 3
        if any(w in high for w in prompt_words):
            key = "high"
        elif all(token_freq.get(w, 0) < mid_thresh for w in prompt_words):
            key = "low"
        else:
            key = "mid"
        return self._run(pairs, {key: freq_zones.get(key, [])})


class L5_ZoneAlpha(ZoneBase):
    """Alphabetic-neighbourhood zone."""
    def __init__(self, sigma=0.40, floor=0.05):
        super().__init__("L5_ZONE_ALPHA", sigma, floor)

    def forward(self, pairs, prompt_words, alpha_zones):
        zone = set().union(*(alpha_zones.get(w, []) for w in prompt_words if w in alpha_zones))
        return self._run(pairs, {"alpha": list(zone)})


class L6_ZoneBigram(ZoneBase):
    """Bigram-context zone."""
    def __init__(self, sigma=0.25, floor=0.04):
        super().__init__("L6_ZONE_BIGRAM", sigma, floor)

    def forward(self, pairs, prompt_words, ngram_zones):
        zone = set().union(*(
            ngram_zones.get((prompt_words[i], prompt_words[i+1]), [])
            for i in range(len(prompt_words) - 1)
        ))
        return self._run(pairs, {"bigram": list(zone)})


class L7_ZoneTrigram(ZoneBase):
    """Trigram-context zone."""
    def __init__(self, sigma=0.20, floor=0.03):
        super().__init__("L7_ZONE_TRIGRAM", sigma, floor)

    def forward(self, pairs, ctx, ngram_zones):
        cl  = list(ctx)
        key = tuple(cl[-2:]) if len(cl) >= 2 else (tuple(cl[-1:]) if cl else ())
        return self._run(pairs, {"trigram": list(ngram_zones.get(key, []))})


class L8_ZoneCharTrig(ZoneBase):
    """Character-trigram zone."""
    def __init__(self, sigma=0.35, floor=0.04):
        super().__init__("L8_ZONE_CHAR_TRIG", sigma, floor)

    def forward(self, pairs, prompt_words, char_trig_index):
        tgs  = set().union(*(_char_trigrams(w) for w in prompt_words))
        zone = set().union(*(char_trig_index.get(t, set()) for t in tgs))
        return self._run(pairs, {"chartrig": list(zone)})


class L9_ZoneLatent(ZoneBase):
    """Latent-space zone."""
    def __init__(self, sigma=0.30, floor=0.04):
        super().__init__("L9_ZONE_LATENT", sigma, floor)

    def forward(self, pairs, prompt_words, latent_sorted_keys, latent_bos_data):
        qkey = latent_sorted_keys[0] if latent_sorted_keys and prompt_words else ""
        for w in prompt_words:
            for k in latent_sorted_keys:
                if w in latent_bos_data.get(k, set()):
                    qkey = k; break
        return self._run(pairs, {"latent": list(latent_bos_data.get(qkey, []))})


# ─────────────────────────────────────────────────────────────────────────────
class L14_LockedStateIndex(nn.Module):
    """
    Locked-state zone with memory.
    Uses granule geometry when no lock is held, hard-locks to committed token
    when the context key has been seen before.
    """
    LAYER_NAME = "L14_LOCKED_STATE_INDEX"

    def __init__(self, sigma=0.25, floor=0.03, lock_strength=1.0):
        super().__init__()
        self.sigma         = nn.Parameter(torch.tensor(sigma,         dtype=torch.float64))
        self.floor         = nn.Parameter(torch.tensor(floor,         dtype=torch.float64))
        self.lock_strength = nn.Parameter(torch.tensor(lock_strength, dtype=torch.float64))
        self.locked:   Dict[Tuple, str] = {}
        self.observed: set              = set()

    def reset_state(self): self.locked.clear(); self.observed.clear()

    def commit(self, key, token):
        if key and key not in self.locked:
            self.locked[key] = token

    @staticmethod
    def key_from_ctx(ctx):
        return tuple(w for w in ctx if w)

    def forward(self, pairs, ctx, draw_pos: int, stream_len: int):
        cand = _cand(pairs)
        words = cand.words
        key   = self.key_from_ctx(ctx)
        fl    = self.floor.clamp(0.0, 1.0 - 1e-6)
        lw    = self.lock_strength.clamp(0.0, 1.0)

        if key and key in self.locked:
            tok  = self.locked[key]
            base = torch.full((cand.W,), float(fl), dtype=torch.float64)
            if tok in words:
                base[words.index(tok)] = float(fl + lw * (1.0 - fl))
            wts = base / base.sum().clamp_min(EPS) * lw + lw.detach()
            wts = wts / wts.sum().clamp_min(EPS)
        else:
            self.observed.add(key)
            norm_pos = draw_pos / max(1, stream_len - 1)
            info = word_granule_area_positional(cand, norm_pos, self.sigma, fl)
            wts  = info["word_area"]
            wts  = wts / wts.sum().clamp_min(EPS)

        ldict = {
            "name":    self.LAYER_NAME,
            "words":   words,
            "probs":   wts.detach().cpu().numpy(),
            "probs_t": wts,
        }
        return list(zip(words, wts.tolist())), ldict


def word_granule_area_positional(
    cand:     CandidateMatrix,
    norm_pos: float,
    sigma:    torch.Tensor,
    floor:    torch.Tensor,
) -> Dict[str, Any]:
    """Positional Gaussian area used when no lock is held in L14."""
    from granule_geometry import granule_areas, gaussian_granule_masks
    import torch
    W   = cand.W
    sig = sigma.clamp_min(1e-6)
    fl  = floor.clamp(0.0, 1.0 - 1e-6)
    idx = cand.idx
    diff   = (idx / max(1, W - 1)) - norm_pos
    mask1d = fl + (1.0 - fl) * torch.exp(-0.5 * (diff / sig) ** 2)
    return {"word_area": mask1d}
