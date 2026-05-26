# =============================================================================
#  layer_isomorphism_torch.py
#  All 15 isomorphism layers (L0–L14) as torch.nn.Module subclasses.
#
#  L0–L13 are the original token / zone / blend / cursor stack.
#  L14 (NEW) is a previous-state-dependent index dimension with monotonic
#  write-once semantics keyed by trigram prefix.  Its modulus indication
#  is offset by the count of "missing states" (observed-but-uncommitted
#  trigram keys), so unresolved context literally pushes the cursor
#  forward through the modulus.
#
#  Design principles
#  -----------------
#  • Every layer is a self-contained nn.Module with a custom __init__ that
#    registers its hyper-parameters as nn.Parameter (learnable) or as named
#    buffers (non-gradient scalars that still move with .to(device)).
#  • forward() accepts and returns plain Python / numpy inputs where the
#    upstream code expects them, but all heavy maths runs on torch tensors.
#  • IsomorphismPipeline wires L0..L13 together and mirrors the original
#    API.  LockedIsomorphismPipeline subclasses it and inserts L14 into
#    the final sampling stage.
#  • No external dependencies beyond torch, numpy, math, collections.
# =============================================================================

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from datasets import load_dataset, Dataset, DatasetDict
import re
from dataclasses import dataclass, asdict
@dataclass
class HFSquadRecord:
    split: str
    original_index: int
    id: str
    title: str
    context: str
    question: str
    answer_text: str
    answer_start: int | None
    tokens: List[str]
    first_token: str
    last_token: str
    kept: bool
    drop_reason: str


class HFSquadSentenceDatasetPreprocessor:
    """
    Hugging Face SQuAD-backed analogue of SentenceDatasetPreprocessor.

    Each dataset entity becomes one token sequence. We then enforce the same
    quota-balanced boundary invariant already used by SentenceDatasetPreprocessor:
      - a first token may appear at most boundaryquota times
      - a last token may appear at most boundaryquota times
      - acceptance is greedy in dataset order
      - boundaryquota=1 gives strict globally-unique beginnings and endings

    Public attributes intentionally mirror SentenceDatasetPreprocessor enough
    for SentenceAwareGenerator / buildsentencepipeline-style code to use them.
    """

    def __init__(
        self,
        dataset_name: str = "squad",
        config_name: str | None = None,
        split_names: Sequence[str] = ("train", "validation"),
        lowercase: bool = True,
        minsentencelen: int = 3,
        uniquemiddlepool: bool = True,
        strict: bool = True,
        boundaryquota: int = 1,
        include_question: bool = True,
        include_context: bool = True,
        include_answer: bool = True,
        qca_mode: str = "question_context_answer",
        sep_qc: str | None = None,
        sep_ca: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split_names = tuple(split_names)
        self.lowercase = bool(lowercase)
        self.minsentencelen = max(2, int(minsentencelen))
        self.uniquemiddlepool = bool(uniquemiddlepool)
        self.strict = bool(strict)
        self.boundaryquota = max(1, int(boundaryquota))

        self.include_question = bool(include_question)
        self.include_context = bool(include_context)
        self.include_answer = bool(include_answer)
        self.qca_mode = str(qca_mode)
        self.sep_qc = sep_qc
        self.sep_ca = sep_ca

        self.sentences: List[List[str]] = []
        self.beginnings: List[str] = []
        self.endings: List[str] = []
        self.middlepool: List[str] = []
        self.tokens: List[str] = []

        self.records: List[HFSquadRecord] = []
        self.keptrecords: List[HFSquadRecord] = []
        self.droppedrecords: List[HFSquadRecord] = []

        self.dropped = 0
        self.skipped = 0
        self.begincounts: Counter = Counter()
        self.endcounts: Counter = Counter()

        self.beginningsset: Set[str] = set()
        self.endingsset: Set[str] = set()
        self.middleset: Set[str] = set()

        self.process()

    @staticmethod
    def _word_tokenize(text: str, lowercase: bool = True) -> List[str]:
        if not text:
            return []
        if lowercase:
            text = text.lower()
        return text.split()

    def _load_dataset(self) -> DatasetDict:
        if self.config_name is None:
            return load_dataset(self.dataset_name)
        return load_dataset(self.dataset_name, self.config_name)

    @staticmethod
    def _pick_answer(example: Dict) -> Tuple[str, int | None]:
        ans = example.get("answers", {}) or {}
        texts = ans.get("text", []) if isinstance(ans, dict) else []
        starts = ans.get("answer_start", []) if isinstance(ans, dict) else []
        text0 = texts[0] if texts else ""
        start0 = starts[0] if starts else None
        return text0, start0

    def _build_entity_tokens(self, example: Dict) -> List[str]:
        q = self._word_tokenize(example.get("question", ""), self.lowercase)
        c = self._word_tokenize(example.get("context", ""), self.lowercase)
        a_text, _ = self._pick_answer(example)
        a = self._word_tokenize(a_text, self.lowercase)

        mode = self.qca_mode.lower()

        if mode == "question_only":
            parts = [q] if self.include_question else []
        elif mode == "question_answer":
            parts = []
            if self.include_question:
                parts.append(q)
            if self.include_answer:
                if self.sep_qc:
                    parts.append([self.sep_qc])
                parts.append(a)
        elif mode == "question_context":
            parts = []
            if self.include_question:
                parts.append(q)
            if self.include_context:
                if self.sep_qc:
                    parts.append([self.sep_qc])
                parts.append(c)
        else:
            parts = []
            if self.include_question:
                parts.append(q)
            if self.include_context:
                if parts and self.sep_qc:
                    parts.append([self.sep_qc])
                parts.append(c)
            if self.include_answer:
                if parts and self.sep_ca:
                    parts.append([self.sep_ca])
                parts.append(a)

        out: List[str] = []
        for block in parts:
            out.extend(block)
        return out

    def _iter_entities(self):
        ds = self._load_dataset()
        for split in self.split_names:
            if split not in ds:
                continue
            split_ds: Dataset = ds[split]
            for idx, ex in enumerate(split_ds):
                yield split, idx, ex

    def _record_from_example(self, split: str, idx: int, ex: Dict) -> HFSquadRecord:
        answer_text, answer_start = self._pick_answer(ex)
        toks = self._build_entity_tokens(ex)

        if len(toks) < self.minsentencelen:
            self.skipped += 1
            return HFSquadRecord(
                split=split,
                original_index=idx,
                id=str(ex.get("id", f"{split}-{idx}")),
                title=str(ex.get("title", "")),
                context=str(ex.get("context", "")),
                question=str(ex.get("question", "")),
                answer_text=answer_text,
                answer_start=answer_start,
                tokens=toks,
                first_token=toks[0] if toks else "",
                last_token=toks[-1] if toks else "",
                kept=False,
                drop_reason=f"too_short_lt_{self.minsentencelen}",
            )

        first = toks[0]
        last = toks[-1]

        if self.strict:
            if self.begincounts[first] >= self.boundaryquota:
                self.dropped += 1
                return HFSquadRecord(
                    split=split,
                    original_index=idx,
                    id=str(ex.get("id", f"{split}-{idx}")),
                    title=str(ex.get("title", "")),
                    context=str(ex.get("context", "")),
                    question=str(ex.get("question", "")),
                    answer_text=answer_text,
                    answer_start=answer_start,
                    tokens=toks,
                    first_token=first,
                    last_token=last,
                    kept=False,
                    drop_reason=f"begin_quota_full:{first}",
                )
            if self.endcounts[last] >= self.boundaryquota:
                self.dropped += 1
                return HFSquadRecord(
                    split=split,
                    original_index=idx,
                    id=str(ex.get("id", f"{split}-{idx}")),
                    title=str(ex.get("title", "")),
                    context=str(ex.get("context", "")),
                    question=str(ex.get("question", "")),
                    answer_text=answer_text,
                    answer_start=answer_start,
                    tokens=toks,
                    first_token=first,
                    last_token=last,
                    kept=False,
                    drop_reason=f"end_quota_full:{last}",
                )

            self.begincounts[first] += 1
            self.endcounts[last] += 1

        return HFSquadRecord(
            split=split,
            original_index=idx,
            id=str(ex.get("id", f"{split}-{idx}")),
            title=str(ex.get("title", "")),
            context=str(ex.get("context", "")),
            question=str(ex.get("question", "")),
            answer_text=answer_text,
            answer_start=answer_start,
            tokens=toks,
            first_token=first,
            last_token=last,
            kept=True,
            drop_reason="",
        )

    def process(self) -> None:
        orderedpool: List[str] = []

        for split, idx, ex in self._iter_entities():
            rec = self._record_from_example(split, idx, ex)
            self.records.append(rec)

            if rec.kept:
                self.keptrecords.append(rec)
                s = rec.tokens
                self.sentences.append(s)
                self.beginnings.append(s[0])
                self.endings.append(s[-1])
                orderedpool.extend(s[1:-1])
                self.tokens.extend(s)
            else:
                self.droppedrecords.append(rec)

        if self.uniquemiddlepool:
            seen = set()
            self.middlepool = []
            for w in orderedpool:
                if w not in seen:
                    seen.add(w)
                    self.middlepool.append(w)
        else:
            self.middlepool = orderedpool

        self.beginningsset = set(self.beginnings)
        self.endingsset = set(self.endings)
        self.middleset = set(self.middlepool)

        if not self.sentences:
            raise ValueError(
                f"No SQuAD entities survived the quota-boundary invariant "
                f"(quota={self.boundaryquota}, dropped={self.dropped}, skipped={self.skipped})."
            )

    def tocorpus(self) -> str:
        return " ".join(self.tokens)

    def vocab(self) -> set:
        return set(self.tokens)

    def isbeginning(self, token: str) -> bool:
        return token in self.beginningsset

    def isnaturalending(self, token: str) -> bool:
        return token in self.endingsset

    def samplearbitrary(
        self,
        rngvalue: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        if not self.middlepool:
            return ""
        n = len(self.middlepool)
        if rngvalue is not None:
            i = int(rngvalue * n) % n
        else:
            rng = rng or random
            i = rng.randrange(n)
        return self.middlepool[i]

    def boundarybalancereport(self) -> str:
        def stats(c: Counter, label: str) -> str:
            if not c:
                return f"{label}: empty"
            vals = list(c.values())
            mn, mx = min(vals), max(vals)
            avg = sum(vals) / len(vals)
            perfectly = all(v == vals[0] for v in vals)
            return (
                f"{label}: {len(c)} words, "
                f"min={mn} max={mx} avg={avg:.2f} "
                f"{'perfectly balanced' if perfectly else 'imbalanced'}"
            )

        lines = [
            f"Boundary quota {self.boundaryquota}",
            stats(self.begincounts, "beginnings"),
            stats(self.endcounts, "endings"),
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        nsent = len(self.sentences)
        avg = sum(len(s) for s in self.sentences) / max(1, nsent)
        return "\n".join(
            [
                "HFSquadSentenceDatasetPreprocessor",
                f"dataset {self.dataset_name}",
                f"mode {self.qca_mode}",
                f"boundaryquota {self.boundaryquota}",
                f"entities kept {nsent}",
                f"dropped quota {self.dropped}",
                f"skipped too short {self.skipped}",
                f"total tokens {len(self.tokens)}",
                f"vocab size {len(self.vocab())}",
                f"unique beginnings {len(self.beginningsset)}",
                f"unique endings {len(self.endingsset)}",
                f"middle-pool size {len(self.middlepool)}",
                f"avg entity len {avg:.2f}",
                f"beginning example {self.beginnings[0] if self.beginnings else ''}",
                f"ending example {self.endings[0] if self.endings else ''}",
                f"middle example {self.middlepool[0] if self.middlepool else ''}",
                self.boundarybalancereport(),
            ]
        )

    def auditrows(self) -> List[Dict]:
        return [asdict(r) for r in self.records]

    def keptrows(self) -> List[Dict]:
        return [asdict(r) for r in self.keptrecords]

    def droppedrows(self) -> List[Dict]:
        return [asdict(r) for r in self.droppedrecords]


# ═══════════════════════════════════════════════════════════════════════════
#  SentenceAwareGenerator
# ═══════════════════════════════════════════════════════════════════════════

class SentenceAwareGenerator:
    """
    Wraps a pipeline so that whenever the live context's most-recent
    token is a NATURAL ENDING (one of the globally-unique sentence-final
    words), the next step:

        1. samples a word ARBITRARILY from the preprocessor's middle pool
        2. pushes it into the context as the new "beginning"
        3. the pipeline then PREDICTS UPON that word in the following step

    The arbitrary draw uses the same pi-stream-backed draw function the
    pipeline uses for L13, so the whole loop stays a single deterministic
    isomorphism over the input pi-stream.
    """

    def __init__(
        self,
        pipeline:    IsomorphismPipeline,
        preprocessor: Optional[SentenceDatasetPreprocessor] = None,
        *,
        emit_seed:   bool = True,
    ):
        if preprocessor is None:
            preprocessor = getattr(pipeline, "preprocessor", None)
        if preprocessor is None:
            raise ValueError(
                "SentenceAwareGenerator needs a preprocessor — either pass "
                "one in or use build_sentence_pipeline() which attaches it."
            )
        self.pipeline  = pipeline
        self.pre       = preprocessor
        self.emit_seed = bool(emit_seed)

    # ── helpers ──────────────────────────────────────────────────────

    def _seed_context(self, prompt: str) -> deque:
        tokens       = [w.lower() for w in prompt.split() if w.isalpha()]
        vocab_tokens = [w for w in tokens if w in self.pipeline.vocab]
        cw           = self.pipeline.context_window
        if len(vocab_tokens) >= cw:
            init = vocab_tokens[-cw:]
        else:
            init = [""] * (cw - len(vocab_tokens)) + vocab_tokens
        return deque(init, maxlen=cw)

    def _format(self, words: List[str], prompt: str, capitalise: bool) -> str:
        """
        Insert periods after natural endings, capitalise sentence starts.
        Beginnings and endings are themselves real corpus words, so we
        rely on the preprocessor's `is_natural_ending` to detect breaks.
        """
        prompt_words = prompt.strip().split() if prompt.strip() else []
        all_words    = prompt_words + words
        if not all_words:
            return ""

        out:      List[str] = []
        cap_next: bool      = True
        pre                 = self.pre
        for w in all_words:
            tok = w
            out.append(tok)
        return " ".join(out)

    # ── main API ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt:            str,
        n_words:           int,
        *,
        stream:            Optional[List[int]] = None,
        digits_per_sample: int  = 3,
        seed:              Optional[int] = None,
    ) -> List[str]:
        """Return the raw list of emitted tokens (including arbitrary seeds)."""
        pipe = self.pipeline
        pre  = self.pre

        # FIX 2 + 3: reset all mutable per-run state before building
        # draw_fn so that cursor, history, step counter, and L14 lock
        # table all start from zero on every call.

        draw_fn = pipe._make_draw_fn(stream, digits_per_sample, seed)

        prompt_tokens = [w.lower() for w in prompt.split() if w.isalpha()]
        ctx           = self._seed_context(prompt)

        words:  List[str]        = []
        frames: List[LayerFrame] = []
        produced                 = 0

        # safety cap: don't loop forever if every prediction is an ending
        max_iters = n_words * 8
        iters     = 0

        while produced < n_words and iters < max_iters:
            iters += 1
            last  = ctx[-1] if (len(ctx) and ctx[-1]) else ""

          
            # Normal pipeline step
            frame = pipe.step(ctx, prompt_tokens, draw_fn())
            if frame is None:
                # dead context — arbitrary restart from the middle pool
                ctx.clear()
                ctx.extend([""] * pipe.context_window)
                seed_word = pre.sample_arbitrary(rng_value=draw_fn())
                if not seed_word:
                    break
                ctx.append(seed_word)
                if self.emit_seed:
                    words.append(seed_word)
                    produced += 1
                continue

            frames.append(frame)
            ctx.append(frame.chosen)
            words.append(frame.chosen)
            produced += 1

        pipe.frames = frames
        return words

    def generate_text(
        self,
        prompt:            str,
        n_words:           int,
        *,
        stream:            Optional[List[int]] = None,
        digits_per_sample: int  = 3,
        seed:              Optional[int] = None,
        capitalise:        bool = True,
    ) -> str:
        words = self.generate(
            prompt            = prompt,
            n_words           = n_words,
            stream            = stream,
            digits_per_sample = digits_per_sample,
            seed              = seed,
        )
        return self._format(words, prompt, capitalise)


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
    """

    def __init__(self, rep_penalty: float = 1.13):
        super().__init__()
        self.rep_penalty = nn.Parameter(torch.tensor(rep_penalty, dtype=torch.float64))

    def forward(self, dist, history: Counter) -> Tuple[List[Tuple[str, float]], Dict]:
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
    """Applies temperature scaling: p_i ∝ p_i^(1/T)."""

    def __init__(self, temperature: float = 4.3):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float64))

    def forward(self, pairs: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], Dict]:
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
    """Penalises tokens whose probability exceeds the mean."""

    def __init__(self, insight_penalty: float = 3.95):
        super().__init__()
        self.insight_penalty = nn.Parameter(
            torch.tensor(insight_penalty, dtype=torch.float64)
        )

    def forward(self, pairs: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], Dict]:
        strength = self.insight_penalty.clamp(min=0.0)

        probs_t   = _to_tensor([p for _, p in pairs])
        mean_p    = probs_t.mean().clamp(min=1e-30)
        excess    = (probs_t - mean_p).clamp(min=0.0)
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
    """Truncates the candidate set to top-K then applies nucleus (top-P)."""

    def __init__(self, top_k: int = 100, top_p: float = 1.0):
        super().__init__()
        self.register_buffer("top_k_buf", torch.tensor(top_k, dtype=torch.int64))
        self.top_p = nn.Parameter(torch.tensor(top_p, dtype=torch.float64))

    def forward(self, pairs: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], Dict]:
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
# Shared Gaussian-gradient zone layer base (L4–L9, L13)
# ---------------------------------------------------------------------------

class _ZoneGradientBase(nn.Module):
    """
    Base class for zone layers L4–L9.  Each applies a Gaussian gradient
    over the ranked candidate list, centring the peak at the mean rank
    of tokens in the active zone.
    """

    def __init__(self, name: str, sigma: float, floor: float, source_hint: str = ""):
        super().__init__()
        self._layer_name  = name
        self._source_hint = source_hint
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float64))
        self.floor = nn.Parameter(torch.tensor(floor, dtype=torch.float64))

    def _gradient(self, zone_set: set, candidates: List[Tuple[str, float]]) -> torch.Tensor:
        n = len(candidates)
        if n == 0:
            return torch.zeros(0, dtype=torch.float64)

        sigma = self.sigma.clamp(min=1e-6)
        floor = self.floor.clamp(min=0.0, max=1.0 - 1e-6)

        indices  = torch.arange(n, dtype=torch.float64)
        norm_idx = indices / max(1, n - 1)

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
    def __init__(self, sigma: float = 0.50, floor: float = 0.05,
                 freq_high_thresh: int = 10, freq_mid_thresh: int = 3):
        super().__init__("L4_ZONE_FREQ", sigma, floor, "freq_zone")
        self.register_buffer("freq_high_thresh",
                             torch.tensor(freq_high_thresh, dtype=torch.int64))
        self.register_buffer("freq_mid_thresh",
                             torch.tensor(freq_mid_thresh, dtype=torch.int64))

    def forward(self, candidates, prompt_words, freq_zones, token_freq):
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
    def __init__(self, sigma: float = 0.40, floor: float = 0.05):
        super().__init__("L5_ZONE_ALPHA", sigma, floor, "alpha_zone")

    def forward(self, candidates, prompt_words, alpha_zones):
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
    def __init__(self, sigma: float = 0.25, floor: float = 0.04):
        super().__init__("L6_ZONE_BIGRAM", sigma, floor, "ngram bigram context")

    def forward(self, candidates, prompt_words, ngram_zones):
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
    def __init__(self, sigma: float = 0.20, floor: float = 0.03):
        super().__init__("L7_ZONE_TRIGRAM", sigma, floor, "live_ctx")

    def forward(self, candidates, context_deque, ngram_zones):
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
    def __init__(self, sigma: float = 0.35, floor: float = 0.04):
        super().__init__("L8_ZONE_CHAR_TRIG", sigma, floor, "char-trigram neighbours")

    def forward(self, candidates, prompt_words, char_trig_idx):
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
    def __init__(self, sigma: float = 0.30, floor: float = 0.04):
        super().__init__("L9_ZONE_LATENT", sigma, floor, "latent_bos_quartile")

    def forward(self, candidates, prompt_words, latent_sorted_keys, latent_bos_data):
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
    """1/(smoothing+count) column. Default smoothing=1.0 reproduces the original."""

    def __init__(self, smoothing: float = 1.0):
        super().__init__()
        self.smoothing = nn.Parameter(torch.tensor(smoothing, dtype=torch.float64))

    def forward(self, candidates, history):
        smooth   = self.smoothing.clamp(min=0.0)
        counts   = torch.tensor([history[w] for w, _ in candidates], dtype=torch.float64)
        hist_vec = _normalise(1.0 / (smooth + counts))

        words = [w for w, _ in candidates]
        return {
            "name":   "L10_HISTORY",
            "words":  words,
            "probs":  hist_vec.detach().numpy(),
            "source": f"repetition history smoothing={smooth.detach().item():.4f}",
        }


# ---------------------------------------------------------------------------
# L11 – Tensor blend of zone layers (softmax row weights)
# ---------------------------------------------------------------------------

class L11_TensorBlend(nn.Module):
    N_ROWS = 7  # L4..L10

    def __init__(self, init_weights: Optional[List[float]] = None):
        super().__init__()
        if init_weights is None:
            init_weights = [1.0] * self.N_ROWS
        assert len(init_weights) == self.N_ROWS
        self.zone_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float64))
        self.register_buffer("eps", torch.tensor(1e-12, dtype=torch.float64))

    def forward(self, zone_layers, candidates):
        n    = len(candidates)
        rows = []
        for layer in zone_layers:
            p = _to_tensor(layer["probs"])
            if p.shape[0] != n:
                p = F.pad(p, (0, n - p.shape[0]))[:n]
            rows.append(p.unsqueeze(0))

        zone_matrix = torch.cat(rows, dim=0)
        row_weights = F.softmax(self.zone_weights, dim=0)

        blended = (zone_matrix * row_weights.unsqueeze(1)).sum(dim=0)
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
    def __init__(self, blend_alpha: float = 0.5):
        super().__init__()
        self.blend_alpha = nn.Parameter(torch.tensor(blend_alpha, dtype=torch.float64))

    def forward(self, L3_pairs, L11):
        alpha = self.blend_alpha.clamp(min=1e-6, max=1.0 - 1e-6)
        beta  = 1.0 - alpha

        p3  = _to_tensor([p for _, p in L3_pairs]).clamp(min=1e-24)
        p11 = _to_tensor(L11["probs"]).clamp(min=1e-24)

        blended = p3.pow(alpha) * p11.pow(beta)
        blended = _normalise(blended)

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
# L13 – Contextual requestor position (Gaussian over pi-cursor)
# ---------------------------------------------------------------------------

class L13_CtxReqPos(nn.Module):
    def __init__(self, sigma: float = 0.30, floor: float = 0.04):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float64))
        self.floor = nn.Parameter(torch.tensor(floor, dtype=torch.float64))

    def forward(self, candidates, draw_pos, stream_len):
        n        = len(candidates)
        sigma    = self.sigma.clamp(min=1e-6)
        floor    = self.floor.clamp(min=0.0, max=1.0 - 1e-6)
        norm_pos = (draw_pos % max(1, stream_len)) / max(1, stream_len - 1)

        indices = torch.arange(n, dtype=torch.float64) / max(1, n - 1)
        gauss   = torch.exp(-0.5 * ((indices - norm_pos) / sigma) ** 2)
        weights = _normalise(floor + (1.0 - floor) * gauss)

        words = [w for w, _ in candidates]
        return {
            "name":   "L13_CTX_REQ_POS",
            "words":  words,
            "probs":  weights.detach().numpy(),
            "source": (
                f"ctx_req_pos={draw_pos} norm={norm_pos:.4f} stream_len={stream_len}"
            ),
        }


# ---------------------------------------------------------------------------
# L14 – Locked state index (NEW)
# ---------------------------------------------------------------------------

class L14_LockedStateIndex(nn.Module):
    """
    Previous-state-dependent index dimension with monotonic write-once
    semantics — the "forbid altering" rule.

    Keyed by the trigram prefix (last two non-empty tokens of the live
    context, matching L7's key space).  Maintains:

        _locked   : Dict[key -> first committed token]
        _observed : Set[keys seen but not yet committed]

    Forward branches
    ----------------
    LOCKED   — key already in _locked.  Returns a near one-hot
               distribution on the locked token (weighted by
               `lock_strength`), with the floor used as the residual
               mass on everything else.  Higher transient indexes
               therefore CANNOT alter the previously locked state.

    UNLOCKED — key not yet locked.  Records it in _observed and emits
               a Gaussian gradient.  The Gaussian centre is the live
               cursor position offset by the number of "missing states":

                   offset_pos = (draw_pos + n_missing) mod stream_len

               so unresolved context literally pushes the cursor
               forward through the modulus.

    Custom init
    -----------
    sigma         : Gaussian width (clamped > 0)
    floor         : minimum weight  (clamped to [0, 1))
    lock_strength : hardness of the lock peak (clamped to [0, 1])
                    1.0 -> hard lock (~one-hot)
                    0.0 -> recovers a flat floor (no locking effect)
    """

    LAYER_NAME = "L14_LOCKED_STATE_INDEX"

    def __init__(
        self,
        sigma:         float = 0.25,
        floor:         float = 0.03,
        lock_strength: float = 1.0,
    ):
        super().__init__()
        self.sigma         = nn.Parameter(torch.tensor(sigma,         dtype=torch.float64))
        self.floor         = nn.Parameter(torch.tensor(floor,         dtype=torch.float64))
        self.lock_strength = nn.Parameter(torch.tensor(lock_strength, dtype=torch.float64))

        # Non-parameter state (not in state_dict; reset between runs).
        self._locked:   Dict[Tuple[str, ...], str] = {}
        self._observed: Set[Tuple[str, ...]]       = set()

    # ── state control ────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Forget every lock and observation.  Called at run start."""
        self._locked.clear()
        self._observed.clear()

    def commit(self, key: Tuple[str, ...], token: str) -> bool:
        """
        Lock ``token`` under ``key`` iff key is not already locked.
        Returns True on a successful new lock, False otherwise.  This
        enforces the write-once / monotonic rule: higher-transient
        indexes cannot overwrite a previous commitment.
        """
        if not key or not token:
            return False
        if key in self._locked:
            return False
        self._locked[key] = token
        self._observed.discard(key)
        return True

    @property
    def n_locked(self) -> int:
        return len(self._locked)

    @property
    def n_missing(self) -> int:
        return len(self._observed)

    # ── trigram key extraction ───────────────────────────────────────

    @staticmethod
    def key_from_ctx(context_deque: deque) -> Tuple[str, ...]:
        """Trigram prefix: last two non-empty tokens of the context."""
        ctx_list = [w for w in context_deque if w]
        if len(ctx_list) >= 2:
            return tuple(ctx_list[-2:])
        if len(ctx_list) == 1:
            return (ctx_list[-1],)
        return ()

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        candidates:    List[Tuple[str, float]],
        context_deque: deque,
        draw_pos:      int,
        stream_len:    int,
    ) -> Dict:
        n     = len(candidates)
        words = [w for w, _ in candidates]

        if n == 0:
            return {
                "name":   self.LAYER_NAME,
                "words":  [],
                "probs":  np.zeros(0, dtype=np.float64),
                "source": "empty candidates",
                "key":    (),
                "locked": False,
            }

        sigma = self.sigma.clamp(min=1e-6)
        floor = self.floor.clamp(min=0.0, max=1.0 - 1e-6)
        lockw = self.lock_strength.clamp(min=0.0, max=1.0)

        key = self.key_from_ctx(context_deque)

        # ─── LOCKED branch ───────────────────────────────────────────
        if key and key in self._locked:
            locked_token = self._locked[key]
            f = float(floor)
            w = float(lockw)

            weights = torch.full((n,), f, dtype=torch.float64)
            if locked_token in words:
                idx          = words.index(locked_token)
                weights[idx] = f + w * (1.0 - f)
            # else: locked token isn't in the candidate set; we degrade
            # gracefully to a uniform floor — the parent's L12·L13
            # contribution then dominates the blend for this step.
            weights = _normalise(weights)

            source = (
                f"LOCKED key={key} -> '{locked_token}' "
                f"(lock_strength={w:.3f}, n_locked={self.n_locked})"
            )
            locked_flag = True

        # ─── UNLOCKED branch ─────────────────────────────────────────
        else:
            if key:
                self._observed.add(key)

            missing    = self.n_missing
            sl         = max(1, int(stream_len))
            offset_pos = (int(draw_pos) + missing) % sl
            norm_pos   = offset_pos / max(1, sl - 1)

            indices = torch.arange(n, dtype=torch.float64) / max(1, n - 1)
            gauss   = torch.exp(-0.5 * ((indices - norm_pos) / sigma) ** 2)
            weights = _normalise(floor + (1.0 - floor) * gauss)

            source = (
                f"UNLOCKED key={key or '∅'} "
                f"draw_pos={draw_pos} missing={missing} "
                f"offset={offset_pos} norm={norm_pos:.4f}"
            )
            locked_flag = False

        return {
            "name":   self.LAYER_NAME,
            "words":  words,
            "probs":  weights.detach().numpy(),
            "source": source,
            "key":    key,
            "locked": locked_flag,
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
        return torch.stack(padded)


# ---------------------------------------------------------------------------
# IsomorphismPipeline  – drop-in replacement for IsomorphismGenerator
# ---------------------------------------------------------------------------

class IsomorphismPipeline(nn.Module):
    """
    Full 14-layer isomorphic probability pipeline (L0..L13) as a single
    nn.Module.  See LockedIsomorphismPipeline below for the 15-layer
    variant that adds L14.
    """

    SAVE_FORMAT_VERSION = 2

    LAYER_NAMES = [
        "L0_RAW_DIST",      "L1_TEMP_SCALED",   "L2_INSIGHT",
        "L3_TOPK_TOPP",     "L4_ZONE_FREQ",     "L5_ZONE_ALPHA",
        "L6_ZONE_BIGRAM",   "L7_ZONE_TRIGRAM",  "L8_ZONE_CHAR_TRIG",
        "L9_ZONE_LATENT",   "L10_HISTORY",      "L11_TENSOR_BLEND",
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

        self._init_hparams: Dict = dict(
            ngram_n         = self.ngram_n,
            temperature     = float(temperature),
            top_k           = int(top_k),
            top_p           = float(top_p),
            rep_penalty     = float(rep_penalty),
            insight_penalty = float(insight_penalty),
            l4_sigma=float(l4_sigma), l4_floor=float(l4_floor),
            l5_sigma=float(l5_sigma), l5_floor=float(l5_floor),
            l6_sigma=float(l6_sigma), l6_floor=float(l6_floor),
            l7_sigma=float(l7_sigma), l7_floor=float(l7_floor),
            l8_sigma=float(l8_sigma), l8_floor=float(l8_floor),
            l9_sigma=float(l9_sigma), l9_floor=float(l9_floor),
            l10_smoothing   = float(l10_smoothing),
            l11_init_weights= list(l11_init_weights) if l11_init_weights is not None else None,
            l12_blend_alpha = float(l12_blend_alpha),
            l13_sigma=float(l13_sigma), l13_floor=float(l13_floor),
        )

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

        unseen = [(w, p) for w, p in blended if self.history[w] == 0]
        pool   = unseen if unseen else blended
        t      = sum(p for _, p in pool)
        pool   = [(w, p / t) for w, p in pool] if t > 0 else pool

        chosen, cumulative = pool[-1][0], 0.0
        for w, p in pool:
            cumulative += p
            if draw < cumulative:
                chosen = w
                break

        self.history[chosen] += 1

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

    # ─────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────

    def _extract_cfd_counts(self) -> Dict[Tuple[str, ...], Dict[str, int]]:
        counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
        for ctx in self.cpd.conditions():
            dist = self.cpd[ctx]
            fd   = getattr(dist, "freqdist", lambda: None)()
            if fd is None:
                continue
            ctx_counts = {str(w): int(c) for w, c in fd.items()}
            if ctx_counts:
                counts[tuple(ctx)] = ctx_counts
        return counts

    def _build_save_payload(
        self,
        kind:           str,
        corpus_text:    Optional[str] = None,
        lidstone_gamma: Optional[float] = None,
        tokens:         Optional[List[str]] = None,
        include_history: bool = True,
    ) -> Dict:
        payload: Dict = {
            "format_version": self.SAVE_FORMAT_VERSION,
            "kind":           kind,
            "state_dict":     self.state_dict(),
            "hparams":        dict(self._init_hparams),
            "class_name":     type(self).__name__,
        }
        if include_history:
            payload["history"] = dict(self.history)

        if kind == "full":
            if corpus_text is None:
                raise ValueError("full save requires corpus_text")
            payload["corpus_text"]    = corpus_text
            payload["lidstone_gamma"] = float(lidstone_gamma) if lidstone_gamma is not None else 0.1
            payload["tokens"]         = list(tokens) if tokens else []
            payload["vocab"]          = sorted(self.vocab)

        elif kind == "midstate":
            payload["lidstone_gamma"] = (
                float(lidstone_gamma) if lidstone_gamma is not None else 0.1
            )
            payload["vocab"]          = sorted(self.vocab)
            payload["tokens"]         = list(tokens) if tokens else []
            payload["cfd_counts"]     = self._extract_cfd_counts()
        return payload

    def save(
        self,
        path:           str,
        *,
        kind:           str             = "full",
        corpus_text:    Optional[str]   = None,
        lidstone_gamma: Optional[float] = None,
        tokens:         Optional[List[str]] = None,
        include_history: bool           = True,
    ) -> str:
        if kind not in ("full", "weights", "midstate"):
            raise ValueError(
                f"kind must be 'full', 'midstate' or 'weights', got {kind!r}"
            )

        payload = self._build_save_payload(
            kind            = kind,
            corpus_text     = corpus_text,
            lidstone_gamma  = lidstone_gamma,
            tokens          = tokens,
            include_history = include_history,
        )
        torch.save(payload, path)
        return path

    @classmethod
    def _construct_from_payload(cls, payload, cpd, ctx_idx, vocab, ngram_n):
        """Build a pipeline of the appropriate class from a save payload."""
        hparams = dict(payload.get("hparams", {}))
        init_kwargs = dict(hparams)
        init_kwargs.pop("ngram_n", None)

        # If the saved class is LockedIsomorphismPipeline, dispatch to it.
        saved_class = payload.get("class_name", cls.__name__)
        target_cls = cls
        if saved_class == "LockedIsomorphismPipeline" and cls is IsomorphismPipeline:
            target_cls = LockedIsomorphismPipeline

        return target_cls(
            cpd           = cpd,
            context_index = ctx_idx,
            vocab         = vocab,
            ngram_n       = ngram_n,
            **init_kwargs,
        )

    @classmethod
    def load(
        cls,
        path:           str,
        *,
        rebuild_context_index: bool = True,
    ) -> "IsomorphismPipeline":
        payload = torch.load(path, map_location="cpu", weights_only=False)

        fmt = payload.get("format_version", 0)
        if fmt > cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Snapshot format version {fmt} is newer than supported "
                f"({cls.SAVE_FORMAT_VERSION}). Upgrade the code."
            )

        if payload.get("kind") != "full":
            raise ValueError(
                "load() requires a 'full' snapshot.  For weights-only files "
                "use IsomorphismPipeline.load_into(existing_pipeline, path)."
            )

        corpus_text = payload["corpus_text"]
        gamma       = float(payload.get("lidstone_gamma", 0.1))
        ngram_n     = int(payload.get("hparams", {}).get("ngram_n", 2))

        cpd, vocab, tokens = build_real_cpd(corpus_text, ngram_n, gamma)
        ctx_idx = build_real_context_index(vocab, cpd, tokens) if rebuild_context_index else None

        pipeline = cls._construct_from_payload(payload, cpd, ctx_idx, vocab, ngram_n)

        missing, unexpected = pipeline.load_state_dict(payload["state_dict"], strict=False)
        if missing or unexpected:
            print(f"  [load] missing keys: {list(missing)}")
            print(f"  [load] unexpected keys: {list(unexpected)}")

        if "history" in payload and isinstance(payload["history"], dict):
            pipeline.history = Counter(payload["history"])

        return pipeline

    @classmethod
    def load_midstate(
        cls,
        path:                  str,
        *,
        rebuild_context_index: bool = True,
    ) -> "IsomorphismPipeline":
        payload = torch.load(path, map_location="cpu", weights_only=False)

        fmt = payload.get("format_version", 0)
        if fmt > cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Snapshot format version {fmt} is newer than supported "
                f"({cls.SAVE_FORMAT_VERSION}). Upgrade the code."
            )
        if payload.get("kind") != "midstate":
            raise ValueError(
                f"load_midstate() requires a 'midstate' snapshot; got "
                f"kind={payload.get('kind')!r}."
            )

        cfd_counts = payload.get("cfd_counts")
        if cfd_counts is None:
            raise ValueError(
                "Midstate file is missing 'cfd_counts' — was it written "
                "by an older version of the code?"
            )

        ngram_n = int(payload.get("hparams", {}).get("ngram_n", 2))
        gamma   = float(payload.get("lidstone_gamma", 0.1))
        vocab   = set(payload.get("vocab") or [])
        tokens  = list(payload.get("tokens") or [])

        cpd = _cpd_from_counts(cfd_counts, vocab, gamma)

        ctx_idx = (
            build_real_context_index(vocab, cpd, tokens)
            if rebuild_context_index and tokens
            else None
        )

        pipeline = cls._construct_from_payload(payload, cpd, ctx_idx, vocab, ngram_n)

        missing, unexpected = pipeline.load_state_dict(payload["state_dict"], strict=False)
        if missing or unexpected:
            print(f"  [load_midstate] missing keys: {list(missing)}")
            print(f"  [load_midstate] unexpected keys: {list(unexpected)}")

        if "history" in payload and isinstance(payload["history"], dict):
            pipeline.history = Counter(payload["history"])

        return pipeline

    def load_into(self, path: str, *, strict: bool = False) -> Dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)

        fmt = payload.get("format_version", 0)
        if fmt > self.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Snapshot format version {fmt} is newer than supported "
                f"({self.SAVE_FORMAT_VERSION})."
            )

        sd = payload.get("state_dict", payload)
        result = self.load_state_dict(sd, strict=strict)

        missing    = list(getattr(result, "missing_keys",    []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

        if "history" in payload and isinstance(payload["history"], dict):
            self.history = Counter(payload["history"])

        return {
            "missing":        missing,
            "unexpected":     unexpected,
            "kind":           payload.get("kind", "weights"),
            "format_version": fmt,
        }

    # ── last-run frame store ──────────────────────────────────────────
    frames: List[LayerFrame] = []

    @staticmethod
    def _frames_to_text(
        frames:        List[LayerFrame],
        prompt:        str  = "",
        capitalise:    bool = True,
        include_prompt: bool = True,
    ) -> str:
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

    def _make_draw_fn(self, stream, digits_per_sample, seed):
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
        draw_fn = self._make_draw_fn(stream, digits_per_sample, seed)

        self.frames = list(
            self.generate(
                prompt  = prompt,
                n_words = n_words,
                draw_fn = draw_fn,
                zone_fn = zone_fn,
            )
        )

        return self._frames_to_text(
            self.frames,
            prompt         = prompt,
            capitalise     = capitalise,
            include_prompt = include_prompt,
        )


# ---------------------------------------------------------------------------
# LockedIsomorphismPipeline – 15-layer variant including L14
# ---------------------------------------------------------------------------

class LockedIsomorphismPipeline(IsomorphismPipeline):
    """
    IsomorphismPipeline + L14_LockedStateIndex.

    Adds a previous-state-dependent index dimension keyed by trigram
    prefix.  Once a key is committed during generation it is locked —
    higher transient indexes (later steps) cannot alter the state.
    The unlocked branch uses a Gaussian whose centre is offset by the
    count of observed-but-uncommitted keys ("missing states").

    Adds the following hyper-parameters
    -----------------------------------
        l14_sigma          : Gaussian width (unlocked branch)
        l14_floor          : floor weight
        l14_lock_strength  : peak hardness on the locked token   [0..1]
        l14_blend_alpha    : exponent of L14 in the final blend  (0..1)
                             0 = ignore L14; 1 = pure L14
    """

    LAYER_NAMES = IsomorphismPipeline.LAYER_NAMES + ["L14_LOCKED_STATE_INDEX"]

    def __init__(
        self,
        *args,
        l14_sigma:         float = 0.25,
        l14_floor:         float = 0.03,
        l14_lock_strength: float = 1.0,
        l14_blend_alpha:   float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.l14 = L14_LockedStateIndex(
            sigma         = l14_sigma,
            floor         = l14_floor,
            lock_strength = l14_lock_strength,
        )
        self.l14_blend_alpha = nn.Parameter(
            torch.tensor(l14_blend_alpha, dtype=torch.float64)
        )

        self._init_hparams.update(
            l14_sigma         = float(l14_sigma),
            l14_floor         = float(l14_floor),
            l14_lock_strength = float(l14_lock_strength),
            l14_blend_alpha   = float(l14_blend_alpha),
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def seed_stream(self, stream):
        super().seed_stream(stream)
        self.l14.reset_state()

    def reset_locked_state(self) -> None:
        """Wipe L14's lock table without disturbing the stream."""
        self.l14.reset_state()

    def lock_table_summary(self, limit: int = 50) -> str:
        items = list(self.l14._locked.items())
        if not items:
            return "(lock table empty)"
        lines = [
            f"Locked: {len(items)}   Missing/observed: {self.l14.n_missing}",
            "-" * 60,
        ]
        for k, v in items[:limit]:
            lines.append(f"  {str(k):<40} -> {v}")
        if len(items) > limit:
            lines.append(f"  ... and {len(items) - limit} more")
        return "\n".join(lines)

    # ── step (overrides parent) ──────────────────────────────────────

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
            flat    = _normalise(torch.ones(len(L3_pairs), dtype=torch.float64))
            flat_np = flat.detach().numpy()
            words   = [w for w, _ in L3_pairs]
            zone_layers = [
                {"name": n, "words": words, "probs": flat_np.copy(),
                 "source": "no context_index"}
                for n in (
                    "L4_ZONE_FREQ", "L5_ZONE_ALPHA", "L6_ZONE_BIGRAM",
                    "L7_ZONE_TRIGRAM", "L8_ZONE_CHAR_TRIG", "L9_ZONE_LATENT",
                )
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

        L10            = self.l10(L3_pairs, self.history)
        L11            = self.l11(zone_layers + [L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)

        draw_pos   = self._pos
        stream_len = max(1, len(self._stream))
        L13        = self.l13(L3_pairs, draw_pos, stream_len)

        # ── NEW: L14 ─────────────────────────────────────────────────
        L14 = self.l14(L3_pairs, context_deque, draw_pos, stream_len)

        # Geometric blend L12 · L13 · L14
        alpha14   = float(self.l14_blend_alpha.clamp(min=1e-6, max=1.0 - 1e-6))
        floor_val = 1e-12

        l12_map = dict(L12_pairs)
        l13_map = dict(zip(L13["words"], L13["probs"].tolist()))
        l14_map = dict(zip(L14["words"], L14["probs"].tolist()))

        blended = []
        for w in l12_map.keys():
            p12 = min(floor_val, l12_map.get(w, floor_val))
            p13 = torch.argmax(torch.tensor(L13["probs"]))
            p14 = torch.argmax(torch.tensor(L14["probs"]))
            base   = math.sqrt(p12 * p13)
            merged = (base ** (1.0 - alpha14)) * (p14 ** alpha14)
            blended.append((w, merged))

        total = sum(p for _, p in blended)
        if total > 0:
            blended = [(w, p / total) for w, p in blended]

        unseen = [(w, p) for w, p in blended if self.history[w] == 0]
        pool   = unseen if unseen else blended
        t      = sum(p for _, p in pool)
        pool   = [(w, p / t) for w, p in pool] if t > 0 else pool

        chosen, cumulative = pool[-1][0] if pool else "", 0.0
        for w, p in pool:
            cumulative += p
            if draw < cumulative:
                chosen = w
                break

        # COMMIT — the write-once rule in action
        key = L14_LockedStateIndex.key_from_ctx(context_deque)
        if key and chosen:
            self.l14.commit(key, chosen)

        self.history[chosen] += 1

        next_draw_pos = (draw_pos + (draw_pos % max(1, stream_len))) % stream_len
        self._pos     = next_draw_pos
        self._step   += 1

        return LayerFrame(
            step           = self._step - 1,
            layers         = [L0, L1, L2, L3] + zone_layers
                              + [L10, L11, L12, L13, L14],
            chosen         = chosen,
            context_window = tuple(context_deque),
            zone_name      = zone_name,
            draw_pos       = draw_pos,
            next_draw_pos  = next_draw_pos,
        )


# ---------------------------------------------------------------------------
# Real corpus builder
# ---------------------------------------------------------------------------

def _cpd_from_counts(
    cfd_counts:     Dict[Tuple[str, ...], Dict[str, int]],
    vocab:          set,
    lidstone_gamma: float = 0.1,
):
    from nltk.probability import (
        ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist, FreqDist,
    )

    cfd = ConditionalFreqDist()
    for ctx, counts in cfd_counts.items():
        ctx_key = tuple(ctx)
        fd = FreqDist()
        for word, c in counts.items():
            fd[word] = int(c)
        cfd[ctx_key] = fd

    class _LidFactory:
        def __init__(self, gamma, bins):
            self.gamma = gamma; self.bins = bins
        def __call__(self, fd):
            return LidstoneProbDist(fd, gamma=self.gamma, bins=self.bins)

    return ConditionalProbDist(
        cfd,
        _LidFactory(gamma=float(lidstone_gamma), bins=max(1, len(vocab))),
    )


def build_real_cpd(corpus: str, ngram_n: int = 2, lidstone_gamma: float = 0.1):
    import os
    import nltk
    from nltk.util import ngrams as nltk_ngrams

    NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)
    for pkg, path in [("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
            except Exception:
                pass

    tokens = corpus.lower().split()
    if not tokens:
        raise ValueError("Corpus produced zero tokens.")

    ngram_n = max(2, int(ngram_n))
    padded  = [""] * (ngram_n - 1) + tokens + [""]
    all_ng  = list(nltk_ngrams(padded, ngram_n))

    cfd_counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
    for ng in all_ng:
        ctx, word = tuple(ng[:-1]), ng[-1]
        cfd_counts.setdefault(ctx, {})
        cfd_counts[ctx][word] = cfd_counts[ctx].get(word, 0) + 1

    vocab = set(tokens) | {""}
    cpd   = _cpd_from_counts(cfd_counts, vocab, lidstone_gamma)
    return cpd, vocab, tokens


def build_real_context_index(vocab, cpd, tokens):
    """Build a ContextZoneIndex from real corpus tokens (if app.py is available)."""
    try:
        import sys, os
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from app import ContextZoneIndex
        return ContextZoneIndex(vocab, cpd, Counter(tokens))
    except Exception as e:
        print(f"  [context_index] not available ({e}); zone layers will use uniform weights.")
        return None



# =============================================================================
#  Gradio helpers
# =============================================================================

def _make_heatmap(frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frames:
        return None

    layer_names = [layer["name"] for layer in frames[0].layers]
    n_layers = len(layer_names)
    n_steps  = len(frames)

    mat = np.zeros((n_layers, n_steps))
    for s, frame in enumerate(frames):
        for r, layer in enumerate(frame.layers):
            words = layer.get("words", [])
            probs = layer.get("probs", np.array([]))
            if frame.chosen in words and len(probs):
                idx = words.index(frame.chosen)
                if idx < len(probs):
                    mat[r, s] = float(probs[idx])

    fig, ax = plt.subplots(figsize=(max(8, n_steps * 0.30), max(4, n_layers * 0.45)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                   cmap="viridis", origin="upper")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Layer")
    ax.set_title("P(chosen token) — per layer x step", fontsize=9)
    fig.colorbar(im, ax=ax, label="probability", shrink=0.8)
    fig.tight_layout()
    return fig


def _make_step_log(frames, limit=40):
    if not frames:
        return ""
    header = "{:>4}  {:<18} {:>9}  {:>9}".format("Step", "Chosen", "draw_pos", "next_pos")
    lines  = [header, "-" * len(header)]
    for i, f in enumerate(frames[:limit]):
        lines.append("{:>4}  {:<18} {:>9}  {:>9}".format(
            i, f.chosen, f.draw_pos, f.next_draw_pos))
    if len(frames) > limit:
        lines.append("... ({} more steps not shown)".format(len(frames) - limit))
    return "\\n".join(lines)


def run_generation(
    corpus_file, prompt, n_words, use_locked, seed,
    ngram_n, lidstone_gamma,
    temperature, top_k, top_p,
    rep_penalty, insight_penalty, l12_blend_alpha,
    l13_sigma, l13_floor,
    l14_sigma, l14_floor, l14_lock_strength, l14_blend_alpha,
    progress=gr.Progress(track_tqdm=True),
):
    # read corpus
    if corpus_file is None:
        return "Please upload a corpus .txt file.", "", None, ""
    try:
        path = corpus_file.name if hasattr(corpus_file, "name") else str(corpus_file)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            corpus_text = fh.read()
    except Exception as exc:
        return "Could not read corpus file:\\n{}".format(exc), "", None, ""

    if not corpus_text.strip():
        return "The uploaded corpus file is empty.", "", None, ""
    if not str(prompt).strip():
        return "Please enter a prompt.", "", None, ""

    try:
        progress(0.10, desc="Building CPD & context index ...")
        cpd, vocab, tokens = build_real_cpd(
            corpus_text, ngram_n=int(ngram_n), lidstone_gamma=float(lidstone_gamma)
        )
        ctx_idx = build_real_context_index(vocab, cpd, tokens)

        progress(0.40, desc="Constructing pipeline ...")
        cls = LockedIsomorphismPipeline if bool(use_locked) else IsomorphismPipeline
        kwargs = dict(
            cpd=cpd, context_index=ctx_idx, vocab=vocab,
            ngram_n=int(ngram_n),
            temperature=float(temperature),
            top_k=int(top_k), top_p=float(top_p),
            rep_penalty=float(rep_penalty),
            insight_penalty=float(insight_penalty),
            l12_blend_alpha=float(l12_blend_alpha),
            l13_sigma=float(l13_sigma), l13_floor=float(l13_floor),
        )
        if bool(use_locked):
            kwargs.update(
                l14_sigma=float(l14_sigma), l14_floor=float(l14_floor),
                l14_lock_strength=float(l14_lock_strength),
                l14_blend_alpha=float(l14_blend_alpha),
            )
        pipe = cls(**kwargs)

        progress(0.60, desc="Generating text ...")
        seed_val = int(seed) if seed is not None else None
        text = pipe.generate_text(
            prompt=str(prompt).strip(),
            n_words=int(n_words),
            seed=seed_val,
        )

        progress(0.85, desc="Building heatmap ...")
        frames   = getattr(pipe, "frames", [])
        heatmap  = _make_heatmap(frames)
        step_log = _make_step_log(frames)

        param_summary = pipe.param_summary()
        if bool(use_locked) and hasattr(pipe, "lock_table_summary"):
            param_summary += "\\n\\n" + pipe.lock_table_summary()

        progress(1.0, desc="Done!")
        return text, param_summary, heatmap, step_log

    except Exception:
        return "Error:\\n\\n{}".format(traceback.format_exc()), "", None, ""
def buildhfsquadpipeline(
    dataset_name: str = "squad",
    config_name: str | None = None,
    split_names: Sequence[str] = ("train", "validation"),
    *,
    locked: bool = True,
    ngram_n: int = 3,
    lidstone_gamma: float = 0.1,
    minsentencelen: int = 3,
    pipelinekwargs: Optional[Dict] = None,
    preprocessorkwargs: Optional[Dict] = None,
) -> Tuple[IsomorphismPipeline, HFSquadSentenceDatasetPreprocessor]:
    """
    Hugging Face SQuAD -> HFSquadSentenceDatasetPreprocessor -> CPD/context index -> pipeline

    Mirrors buildsentencepipeline() but sources corpus entities from a HF SQuAD dataset
    instead of raw text sentence splitting.
    """
    pre = HFSquadSentenceDatasetPreprocessor(
        dataset_name=dataset_name,
        config_name=config_name,
        split_names=split_names,
        minsentencelen=minsentencelen,
        **(preprocessorkwargs or {}),
    )
    corpustext = pre.tocorpus()
    cpd, vocab, tokens = build_real_cpd(
        corpustext,
        ngram_n=int(ngram_n),
        lidstone_gamma=float(lidstone_gamma),
    )
    ctxidx = build_real_context_index(vocab, cpd, tokens)

    cls = LockedIsomorphismPipeline if locked else IsomorphismPipeline
    kwargs = dict(
        cpd=cpd,
        context_index=ctxidx,
        vocab=vocab,
        ngram_n=int(ngram_n),
    )
    if pipelinekwargs:
        kwargs.update(pipelinekwargs)

    pipe = cls(**kwargs)
    pipe.preprocessor = pre
    return pipe, pre
if __name__ == "__main__":
    pipe, pre = buildhfsquadpipeline(
        dataset_name="squad",
        locked=True,
        ngram_n=2,
        lidstone_gamma=0.1,
        minsentencelen=3,
        pipelinekwargs=dict(
            temperature=4.3,
            top_k=100,
            top_p=1.0,
            rep_penalty=1.13,
            insight_penalty=3.95,
            l12_blend_alpha=0.5,
            l13_sigma=0.30,
            l13_floor=0.04,
            l14_sigma=0.25,
            l14_floor=0.03,
            l14_lock_strength=1.0,
            l14_blend_alpha=0.5,
        ),
        preprocessorkwargs=dict(
            boundaryquota=10000000,
            strict=True,
            lowercase=True,
            uniquemiddlepool=True,
            qca_mode="question_context_answer",
            sep_qc=None,
            sep_ca=None,
        ),
    )

    print(pre.summary())
    while True:
        gen = SentenceAwareGenerator(pipe, pre)
        text = gen.generate_text(
            prompt=input("USER: "),
            n_words=800,
            seed=42,
            capitalise=True,
        )
        print(text)
        print()
