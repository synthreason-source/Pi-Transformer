# -*- coding: utf-8 -*-

import os
import sys
import re
import math
import json
import gzip
import pickle
import hashlib
import tempfile
import threading
from collections import defaultdict, deque, Counter
from difflib import SequenceMatcher

import numpy as np
import gradio as gr
from mpmath import mp, pi as mpi

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import (
    ConditionalFreqDist,
    ConditionalProbDist,
    LidstoneProbDist,
)
from nltk.collocations import (
    BigramCollocationFinder,
    TrigramCollocationFinder,
    BigramAssocMeasures,
    TrigramAssocMeasures,
)

NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

for pkg, path in [
    ("punkt",     "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("words",     "corpora/words"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass

from nltk.corpus import words as nltk_words

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300000)

DEFAULTS = dict(
    PI_PREC=15000,
    PI_STREAM_LEN=12000,
    DIGITS_PER_SAMPLE=3,
    NGRAM_N=2,
    LIDSTONE_GAMMA=0.1,
    GEN_WORDS=400,
    WORD_FIND_MIN=2,
    TEMPERATURE=4.3,
    TOP_K=100,
    TOP_P=1.0,
    REP_PENALTY=1.13,
    SEASHELL_ENABLE=True,
    SEASHELL_STRENGTH=14.35,
    SEASHELL_DECAY=0.185,
    SEASHELL_PEAKS=14,
    SEASHELL_WIDTH=0.96,
    SEASHELL_FLOOR=0.35,
    SEMICIRCLE_ENABLE=True,
    SEMICIRCLE_STRENGTH=0.6,
    SEMICIRCLE_ARCHES=5,
    SEMICIRCLE_RADIUS=1.0,
    SEMICIRCLE_SPEED=0.05,
    SEMICIRCLE_FLOOR=0.05,
    BEND_DEGREES=13.0,
    OFFSET=0,
    VERTEX="A",
    FUZZY_THRESHOLD=0.92,
    MAX_SOLUTIONS=15,
    BEND_STEP=0.5,
    OFFSET_STEP=50,
    BEND_MAX=45.0,
    INSIGHT_PENALTY=3.95,
    CYCLES_N=32,
    SPAN_WORDS=8,          # NEW: words per PSPACE zone span
)

EMBEDDED_CORPUS = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it.

So she was considering in her own mind whether the pleasure of making a
daisy chain would be worth the trouble of getting up and picking the daisies.

Suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that, nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear!
I shall be late!"

When the Rabbit actually took a watch out of its waistcoat pocket and looked
at it and hurried on, Alice started to her feet.

The rabbit hole went straight on like a tunnel for some way and then dipped
suddenly down.

Either the well was very deep or she fell very slowly, for she had plenty of
time as she went down to look about her and wonder what was going to happen next.
"""

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

HF_REPO_ID    = "trainman999/Thinking-lite"
LOCAL_CACHE_DIR = os.path.join(NLTK_DATA_DIR, "hf_model_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
HF_CACHE = {"loaded": False, "tokenizer": None, "model": None, "status": "Idle"}
CACHE    = dict(key=None, cpd=None, vocab=None, stream=None, context_index=None)
UI_STATE = {'version': 0}


# ---------------------------------------------------------------------------
# Insight penalty
# ---------------------------------------------------------------------------

def apply_insight_penalty(scored, strength):
    if not scored or strength <= 0.0:
        return scored
    mean_p = sum(p for _, p in scored) / len(scored)
    if mean_p <= 0:
        return scored
    penalised = []
    for word, p in scored:
        excess = max(0.0, p - mean_p)
        p_adj  = p / (1.0 + strength * excess / mean_p)
        penalised.append((word, max(1e-12, p_adj)))
    total = sum(p for _, p in penalised)
    if total > 0:
        penalised = [(w, p / total) for w, p in penalised]
    return penalised


def tokenise_alpha(text):
    if text is None:
        return []
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    elif not isinstance(text, str):
        text = str(text)
    return text.lower().split()


def extract_word_pairs(prompt):
    try:
        words = [w.lower() for w in word_tokenize(prompt) if w.isalpha()]
    except Exception:
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", prompt)]
    return list(ngrams(words, 2))


def capitalise_text(words):
    return " ".join(words) if words else ""


class _LidstoneFactory:
    __slots__ = ("gamma", "bins")

    def __init__(self, gamma, bins):
        self.gamma = float(gamma)
        self.bins  = max(1, int(bins))

    def __call__(self, fd):
        return LidstoneProbDist(fd, gamma=self.gamma, bins=self.bins)


def build_model(corpus, ngram_n, lidstone_gamma):
    if isinstance(corpus, bytes):
        corpus = corpus.decode("utf-8", errors="ignore")
    elif not isinstance(corpus, str):
        corpus = str(corpus) if corpus is not None else ""
    ngram_n        = max(2, int(ngram_n))
    lidstone_gamma = float(lidstone_gamma)
    tokens = tokenise_alpha(corpus)
    if not tokens:
        raise ValueError(
            "Corpus produced zero tokens after tokenisation. "
            "Upload a non-empty text corpus or paste some text."
        )
    padded    = [""] * (ngram_n - 1) + tokens + [""]
    trigrams_ = list(ngrams(padded, ngram_n))
    cfd   = ConditionalFreqDist((tuple(tg[:-1]), tg[-1]) for tg in trigrams_)
    vocab = set(tokens) | {""}
    for ctx in list(cfd.conditions()):
        if len(cfd[ctx]) == 0:
            cfd[ctx][""] += 1
    cpd = ConditionalProbDist(
        cfd,
        _LidstoneFactory(gamma=lidstone_gamma, bins=max(1, len(vocab))),
    )
    return cpd, vocab


def build_pi_stream(decimals, length):
    decimals = int(decimals)
    length   = int(length)
    mp.dps   = decimals + 50
    D    = 10 ** decimals
    frac = int(mp.floor(mpi * D)) - 3 * D
    stream = []
    for _ in range(length):
        frac *= 26
        stream.append(frac // D)
        frac %= D
    return stream


# ---------------------------------------------------------------------------
# ContextZoneIndex — sorted zones + trigram latent-space BOS sort
# ---------------------------------------------------------------------------

class ContextZoneIndex:
    """
    Pre-built attention zones over corpus vocabulary.

    Zones
    -----
    freq_zones      : high / mid / low frequency buckets
    alpha_zones     : first-letter buckets
    ngram_zones     : bigram-context → ranked successors
    latent_bos_data : trigram contexts sorted by cosine similarity to the
                      sentence-beginning (BOS) anchor vector, bucketed
                      into quartiles q0 (most BOS-like) … q3 (least).

    All zones share the identical ZoneRetriever protocol:
        key_fn(prompt_words) → [keys]
        data[key]            → [word, …]
    """

    FREQ_HIGH_THRESH = 10
    FREQ_MID_THRESH  = 3

    _ZONE_SPEC_TEMPLATES = [
        ("ngram_bigram",  0.25, 0.04),
        ("ngram_unigram", 0.30, 0.04),
        ("alpha",         0.40, 0.05),
        ("freq",          0.50, 0.05),
        ("trigram_char",  0.35, 0.04),
        ("latent_bos",    0.30, 0.04),   # NEW
    ]

    def __init__(self, vocab, cpd, token_freq):
        self.vocab      = set(vocab)
        self.token_freq = dict(token_freq)
        self._build_freq_zones()
        self._build_alpha_zones()
        self._build_ngram_zones(cpd)
        self._build_trigram_latent_index(cpd)   # NEW
        self._make_zone_specs()
        self.prompt_zone = {}

    # ── zone builders ──────────────────────────────────────────────────────

    def _build_freq_zones(self):
        pairs = sorted(
            [(w, self.token_freq.get(w, 0)) for w in self.vocab if w],
            key=lambda x: x[1], reverse=True,
        )
        high, mid, low = [], [], []
        for w, c in pairs:
            if   c >= self.FREQ_HIGH_THRESH: high.append(w)
            elif c >= self.FREQ_MID_THRESH:  mid.append(w)
            else:                            low.append(w)
        self.freq_zones = {"high": high, "mid": mid, "low": low}

    def _build_alpha_zones(self):
        zones = defaultdict(list)
        for w in sorted(self.vocab):
            if w and w[0].isalpha():
                zones[w[0]].append(w)
        self.alpha_zones = dict(zones)

    def _build_ngram_zones(self, cpd):
        zones = {}
        try:
            for ctx in cpd.conditions():
                d = cpd[ctx]
                ranked = sorted(
                    [s for s in d.samples() if s],
                    key=lambda s: d.prob(s), reverse=True,
                )
                if ranked:
                    zones[ctx] = ranked
        except Exception:
            pass
        self.ngram_zones = zones

    # ── NEW: trigram latent-space index, sorted by BOS-anchor proximity ───

    def _build_trigram_latent_index(self, cpd):
        """
        1. Build a dense probability vector over vocab_list for every
           trigram context key in ngram_zones.
        2. Compute a BOS anchor = mean of all ("", w) context vectors.
        3. Sort all contexts by cosine similarity to that anchor (desc).
        4. Bucket into quartiles q0..q3; each bucket holds the union of
           successor words for its contexts — this is the latent-sorted
           dataset the latent_bos zone spec reads from.
        """
        self.vocab_list = sorted(w for w in self.vocab if w)
        V = len(self.vocab_list)
        if V == 0:
            self.latent_vectors     = {}
            self.latent_sorted_keys = []
            self.latent_sim_scores  = {}
            self.latent_bos_data    = {}
            return

        word_to_idx = {w: i for i, w in enumerate(self.vocab_list)}

        def ctx_to_vector(ctx):
            vec = np.zeros(V, dtype=np.float64)
            try:
                d = cpd[ctx]
                for s in d.samples():
                    if s and s in word_to_idx:
                        vec[word_to_idx[s]] = max(0.0, float(d.prob(s)))
            except Exception:
                pass
            norm = vec.sum()
            if norm > 0:
                vec /= norm
            return vec

        self.latent_vectors = {ctx: ctx_to_vector(ctx) for ctx in self.ngram_zones}

        # BOS anchor: mean of all ("", w) context vectors
        bos_vecs = [
            vec for ctx, vec in self.latent_vectors.items()
            if len(ctx) >= 1 and ctx[0] == ""
        ]
        if bos_vecs:
            anchor = np.mean(np.vstack(bos_vecs), axis=0)
            a_norm = np.linalg.norm(anchor)
            if a_norm > 0:
                anchor /= a_norm
        else:
            anchor = np.ones(V, dtype=np.float64) / max(1, V)

        def cosine_sim(vec):
            n = np.linalg.norm(vec)
            return float(np.dot(vec / n, anchor)) if n >= 1e-12 else 0.0

        self.latent_sim_scores  = {ctx: cosine_sim(v) for ctx, v in self.latent_vectors.items()}
        self.latent_sorted_keys = sorted(
            self.latent_vectors.keys(),
            key=lambda ctx: self.latent_sim_scores[ctx],
            reverse=True,
        )

        # Build quartile buckets over the sorted dataset
        n_keys          = len(self.latent_sorted_keys)
        latent_bos_data = {f"q{q}": [] for q in range(4)}
        for rank, ctx in enumerate(self.latent_sorted_keys):
            q = min(3, rank * 4 // max(1, n_keys))
            latent_bos_data[f"q{q}"].extend(self.ngram_zones.get(ctx, []))

        for q_key in latent_bos_data:
            seen_l, deduped = set(), []
            for w in latent_bos_data[q_key]:
                if w not in seen_l:
                    seen_l.add(w)
                    deduped.append(w)
            latent_bos_data[q_key] = deduped

        self.latent_bos_data = latent_bos_data

    def print_latent_sorted_dataset(self, top_n=20):
        """Print trigram contexts sorted by BOS-latent cosine similarity."""
        print(f"{'Rank':<5} {'Context':<28} {'CosSim':>8}  Top successors")
        print("-" * 78)
        for rank, ctx in enumerate(self.latent_sorted_keys[:top_n]):
            sim   = self.latent_sim_scores[ctx]
            succs = self.ngram_zones.get(ctx, [])[:5]
            ctx_s = " | ".join(f"'{w}'" for w in ctx)
            print(f"{rank:<5} ({ctx_s:<26}) {sim:>8.4f}  {succs}")

    # ── zone specs ────────────────────────────────────────────────────────

    def _make_zone_specs(self):
        def _char_trigrams(word):
            return {word[i:i+3] for i in range(len(word) - 2)} if len(word) >= 3 else {word}

        def key_fn_bigram(prompt_words):
            return [(prompt_words[i], prompt_words[i+1]) for i in range(len(prompt_words)-1)]

        def key_fn_unigram(prompt_words):
            return [(w,) for w in prompt_words if w]

        def key_fn_alpha(prompt_words):
            return [w[0] for w in prompt_words if w and w[0].isalpha()]

        high_set = set(self.freq_zones["high"])

        def key_fn_freq(prompt_words):
            if any(w in high_set for w in prompt_words):
                return ["high"]
            if all(self.token_freq.get(w, 0) < self.FREQ_MID_THRESH for w in prompt_words):
                return ["low"]
            return ["mid"]

        trig_index = defaultdict(set)
        for v in self.vocab:
            if v:
                for tg in _char_trigrams(v):
                    trig_index[tg].add(v)
        self._trig_index = trig_index
        self._trig_data  = {}

        def key_fn_trigram(prompt_words):
            prompt_tgs = set()
            for w in prompt_words:
                prompt_tgs |= _char_trigrams(w)
            neighbours = set()
            for tg in prompt_tgs:
                neighbours |= trig_index.get(tg, set())
            self._trig_data["__trig__"] = sorted(neighbours)
            return ["__trig__"]

        # NEW: latent_bos key_fn — map prompt words to BOS-quartile bucket
        n_keys = len(self.latent_sorted_keys)

        def key_fn_latent(prompt_words):
            for ctx in self.latent_sorted_keys:
                if any(w in ctx for w in prompt_words):
                    rank = self.latent_sorted_keys.index(ctx)
                    q    = min(3, rank * 4 // max(1, n_keys))
                    return [f"q{q}"]
            return ["q0"]

        templates = {t[0]: (t[1], t[2]) for t in self._ZONE_SPEC_TEMPLATES}
        self.zone_specs = [
            dict(name="ngram_bigram",  key_fn=key_fn_bigram,   data=self.ngram_zones,     sigma=templates["ngram_bigram"][0],  floor=templates["ngram_bigram"][1]),
            dict(name="ngram_unigram", key_fn=key_fn_unigram,  data=self.ngram_zones,     sigma=templates["ngram_unigram"][0], floor=templates["ngram_unigram"][1]),
            dict(name="alpha",         key_fn=key_fn_alpha,    data=self.alpha_zones,     sigma=templates["alpha"][0],         floor=templates["alpha"][1]),
            dict(name="freq",          key_fn=key_fn_freq,     data=self.freq_zones,      sigma=templates["freq"][0],          floor=templates["freq"][1]),
            dict(name="trigram_char",  key_fn=key_fn_trigram,  data=self._trig_data,      sigma=templates["trigram_char"][0],  floor=templates["trigram_char"][1]),
            dict(name="latent_bos",    key_fn=key_fn_latent,   data=self.latent_bos_data, sigma=templates["latent_bos"][0],    floor=templates["latent_bos"][1]),
        ]

    def select_zones_for_prompt(self, prompt_words):
        selected, seen = [], set()

        def _add(words):
            for w in words:
                if w and w not in seen:
                    seen.add(w)
                    selected.append(w)

        for spec in self.zone_specs:
            for key in spec["key_fn"](prompt_words):
                _add(spec["data"].get(key, []))
        return selected

    @staticmethod
    def zone_gradient(zone_set, candidates, sigma=0.35, floor=0.05):
        n = len(candidates)
        if n == 0:
            return np.array([], dtype=np.float64)
        indices  = np.arange(n, dtype=np.float64)
        norm_idx = indices / max(1, n - 1)
        zone_ranks = [
            i / max(1, n - 1)
            for i, (w, _) in enumerate(candidates)
            if w in zone_set
        ]
        centre  = float(np.mean(zone_ranks)) if zone_ranks else 0.0
        gauss   = np.exp(-0.5 * ((norm_idx - centre) / max(1e-6, sigma)) ** 2)
        weights = floor + (1.0 - floor) * gauss
        weights /= weights.sum()
        return weights


def build_context_index(vocab, cpd, corpus_tokens):
    return ContextZoneIndex(vocab, cpd, Counter(corpus_tokens))


# ---------------------------------------------------------------------------
# vstack + np.roll column-match loop
# ---------------------------------------------------------------------------

def stream_to_matrix(stream, n_cols=26):
    arr  = np.array(stream, dtype=np.int32)
    trim = (len(arr) // n_cols) * n_cols
    return arr[:trim].reshape(-1, n_cols)


def roll_until_column_match(arr, max_iters=None):
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_cols    = arr.shape[1]
    max_iters = max_iters or n_cols
    original  = arr.copy()
    stacked   = arr.copy()
    for shift in range(1, max_iters + 1):
        rolled  = np.roll(original, shift, axis=1)
        stacked = np.vstack([stacked, rolled])
        if np.any(np.all(rolled == original, axis=0)):
            return stacked, shift
    return stacked, -1


# ---------------------------------------------------------------------------
# SeashellResonator
# ---------------------------------------------------------------------------

class SeashellResonator:
    def __init__(self, sampler, strength, decay, peaks, width, floor):
        self.sampler       = sampler
        self.base_strength = max(0.0, float(strength))
        self.decay         = min(0.9999, max(0.0, float(decay)))
        self.peaks         = max(1, int(peaks))
        self.width         = max(0.02, float(width))
        self.floor         = max(1e-6, float(floor))
        self.energy        = 1.0
        self.step_index    = 0
        self.centers = []
        self.phases  = []
        self.spreads = []
        self._seed_from_stream()

    def _seed_from_stream(self):
        self.centers, self.phases, self.spreads = [], [], []
        for _ in range(self.peaks):
            self.centers.append(self.sampler.next_unit())
            self.phases.append(2.0 * math.pi * self.sampler.next_unit())
            self.spreads.append(self.width * (0.65 + 0.7 * self.sampler.next_unit()))

    def reset(self):
        self.energy     = 1.0
        self.step_index = 0
        self._seed_from_stream()

    def _wrapped_distance(self, a, b):
        d = abs(a - b)
        return min(d, 1.0 - d)

    def gains(self, n_items):
        if n_items <= 0:
            return []
        t       = self.step_index
        ls      = self.base_strength * self.energy
        drift   = 0.017 * math.sin(0.11 * t)
        shimmer = 0.09  * math.sin(0.19 * t + 1.7)
        gains   = []
        for rank in range(n_items):
            idx      = rank / max(1, n_items - 1)
            response = 0.0
            for center, phase, spread in zip(self.centers, self.phases, self.spreads):
                mc     = (center + drift * math.sin(phase + 0.07 * t)) % 1.0
                d      = self._wrapped_distance(idx, mc)
                gauss  = math.exp(-(d * d) / max(1e-9, 2.0 * spread * spread))
                ripple = 0.5 + 0.5 * math.cos(
                    (d / max(1e-9, spread)) * math.pi * (1.5 + shimmer) + phase + 0.13 * t
                )
                response += gauss * (0.55 + 0.45 * ripple)
            response /= max(1, self.peaks)
            gains.append(max(self.floor, self.floor + (1.0 - self.floor) + ls * response))
        total = sum(gains)
        if total > 0:
            gains = [g / total for g in gains]
        return gains

    def apply(self, scored):
        if not scored:
            return scored
        g        = self.gains(len(scored))
        weighted = [(w, p * gain) for (w, p), gain in zip(scored, g)]
        total    = sum(p for _, p in weighted)
        if total > 0:
            weighted = [(w, p / total) for w, p in weighted]
        self.energy *= self.decay
        self.step_index += 1
        if self.energy < 0.08:
            self.energy = 1.0
            self._seed_from_stream()
        return weighted


# ---------------------------------------------------------------------------
# SemicircleWaveMask  — BUG FIX: sign inversion (-p*gain) corrected to p*gain
# ---------------------------------------------------------------------------

class SemicircleWaveMask:
    def __init__(self, sampler, strength, arches, radius, speed, floor):
        self.sampler    = sampler
        self.strength   = max(0.0, float(strength))
        self.arches     = max(1, int(arches))
        self.radius     = max(0.02, float(radius))
        self.speed      = float(speed)
        self.floor      = max(1e-6, float(floor))
        self.step_index = 0
        self.phase0     = self.sampler.next_unit()

    def reset(self):
        self.step_index = 0
        self.phase0     = self.sampler.next_unit()

    def _arch(self, x):
        d = (x - 0.5) / self.radius
        v = 1.0 - d * d
        return math.sqrt(v) if v > 0.0 else 0.0

    def gains(self, n_items):
        if n_items <= 0:
            return []
        phase = self.phase0 + self.speed * self.step_index
        gains = []
        for rank in range(n_items):
            idx      = rank / max(1, n_items - 1)
            slot_pos = (idx * self.arches + phase) % 1.0
            arch     = self._arch(slot_pos)
            gains.append(max(self.floor, (1.0 - self.strength) + self.strength * arch))
        total = sum(gains)
        if total > 0:
            gains = [g / total for g in gains]
        return gains

    def apply(self, scored):
        if not scored:
            return scored
        g        = self.gains(len(scored))
        weighted = [(w, p * gain) for (w, p), gain in zip(scored, g)]  # FIXED: was -p*gain
        total    = sum(p for _, p in weighted)
        if total > 0:
            weighted = [(w, p / total) for w, p in weighted]
        self.step_index += 1
        return weighted


# ---------------------------------------------------------------------------
# PiSampler
# ---------------------------------------------------------------------------

class PiSampler:
    def __init__(
        self,
        stream,
        digits_per_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seashell_enable,
        seashell_strength,
        seashell_decay,
        seashell_peaks,
        seashell_width,
        seashell_floor,
        semicircle_enable=False,
        semicircle_strength=0.09,
        semicircle_arches=5,
        semicircle_radius=1.0,
        semicircle_speed=0.05,
        semicircle_floor=0.05,
        insight_penalty=1.5,
        instruction_context=None,
    ):
        self.stream              = stream
        self.digits_per_sample   = digits_per_sample
        self.pos                 = 0
        self.temperature         = max(1e-3, float(temperature))
        self.top_k               = max(1, int(top_k))
        self.top_p               = max(1e-3, min(1.0, float(top_p)))
        self.repetition_penalty  = max(1.0, float(repetition_penalty))
        self.insight_penalty     = max(0.0, float(insight_penalty))
        self.history             = Counter()
        self.instruction_context = list(instruction_context) if instruction_context else []
        self.seashell   = None
        self.semicircle = None
        if seashell_enable:
            self.seashell = SeashellResonator(
                self, seashell_strength, seashell_decay,
                seashell_peaks, seashell_width, seashell_floor,
            )
        if semicircle_enable:
            self.semicircle = SemicircleWaveMask(
                self, semicircle_strength, semicircle_arches,
                semicircle_radius, semicircle_speed, semicircle_floor,
            )

    def seek(self, pos):
        self.pos = pos % len(self.stream)
        self.history.clear()
        if self.seashell   is not None: self.seashell.reset()
        if self.semicircle is not None: self.semicircle.reset()

    def next_unit(self):
        val  = 0
        base = 26 ** self.digits_per_sample
        for _ in range(self.digits_per_sample):
            val  = val * 26 + self.stream[self.pos % max(1, val + 1)]
            self.pos += 1
        return val / base

    def set_context_index(self, context_index):
        self._context_index = context_index

    def _zone_attention_blend(self, scored):
        n = len(scored)
        if n == 0:
            return scored

        ctx    = getattr(self, "_context_index", None)
        prompt = self.instruction_context

        if ctx is not None and prompt:
            zone_selection = ctx.select_zones_for_prompt(prompt)
        else:
            zone_selection = [w for w, _ in scored]

        gradient_rows = []
        if ctx is not None and prompt:
            for spec in ctx.zone_specs:
                for key in spec["key_fn"](prompt):
                    words = spec["data"].get(key, [])
                    if words:
                        g = ContextZoneIndex.zone_gradient(
                            set(words), scored,
                            sigma=spec["sigma"], floor=spec["floor"],
                        )
                        gradient_rows.append(g)

        if not gradient_rows:
            gradient_rows.append(np.ones(n, dtype=np.float64) / n)

        zone_matrix = np.vstack(gradient_rows)
        history_col = np.array(
            [1.0 / (1.0 + self.history[w]) for w, _ in scored],
            dtype=np.float64,
        ).reshape(1, -1)
        full_matrix = np.vstack([zone_matrix, history_col])
        row_norms   = full_matrix.sum(axis=1, keepdims=True).clip(1e-12)
        augmented   = np.hstack([full_matrix, row_norms])

        row_weights = augmented[:, :n].mean(axis=1)
        row_weights = np.clip(row_weights, 1e-12, None)
        row_weights /= row_weights.sum()
        weights = (augmented[:, :n] * row_weights[:, None]).sum(axis=0)
        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()

        blended = [
            (word, math.sqrt(max(1e-24, p) * float(w)))
            for (word, p), w in zip(scored, weights)
        ]
        total = sum(p for _, p in blended)
        if total > 0:
            blended = [(ww, pp / total) for ww, pp in blended]
        return blended

    def sample(self, dist):
        samples = list(dist.samples())
        if not samples:
            return ""

        base_scored = []
        for s in samples:
            p     = max(1e-12, float(dist.prob(s)))
            count = self.history[s]
            if count > 0:
                p /= self.repetition_penalty ** count
            base_scored.append((s, p))

        base_scored = apply_insight_penalty(base_scored, self.insight_penalty)
        scored      = [(s, p ** (1.0 / self.temperature)) for s, p in base_scored]
        total       = sum(p for _, p in scored)
        scored      = [(s, p / total) for s, p in scored]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[: self.top_k]

        kept, accum = [], 0.0
        for s, p in scored:
            kept.append((s, p))
            accum += p
            if accum >= self.top_p:
                break
        scored = kept

        if self.seashell   is not None: scored = self.seashell.apply(scored)
        if self.semicircle is not None: scored = self.semicircle.apply(scored)
        scored = self._zone_attention_blend(scored)

        unseen     = [(w, p) for w, p in scored if self.history[w] == 0]
        pool       = unseen if unseen else scored
        pool_total = sum(p for _, p in pool)
        if pool_total > 0:
            pool = [(w, p / pool_total) for w, p in pool]

        draw = self.next_unit()
        cumulative, chosen = 0.0, pool[-1][0]
        for word, p in pool:
            cumulative += p
            if draw < cumulative:
                chosen = word
                break

        self.history[chosen] += 1
        return chosen


# ---------------------------------------------------------------------------
# Triangle
# ---------------------------------------------------------------------------

class Triangle:
    def __init__(self, stream_len, offset_extra=0, bend_degrees=13.0):
        base       = offset_extra % stream_len
        bend_shift = int(round((bend_degrees / 360.0) * stream_len))
        self.A     = base % stream_len
        self.B     = (base + stream_len // 3 + bend_shift) % stream_len
        self.C     = (base + 2 * stream_len // 3 + bend_shift) % stream_len
        self.vertices = {"A": self.A, "B": self.B, "C": self.C}


# ---------------------------------------------------------------------------
# PSPACE Semantic Area Definitions
# ---------------------------------------------------------------------------

PSPACE_ZONES = [
    {"name": "NARRATIVE",    "desc": "High-frequency corpus words",        "pi_min": 0.00, "pi_max": 0.20, "zone_key": "freq",         "freq_tier": "high"},
    {"name": "DESCRIPTIVE",  "desc": "Alphabetic neighbourhood of prompt", "pi_min": 0.20, "pi_max": 0.40, "zone_key": "alpha",        "freq_tier": None},
    {"name": "RELATIONAL",   "desc": "Bigram-conditioned successors",      "pi_min": 0.40, "pi_max": 0.60, "zone_key": "ngram_bigram", "freq_tier": None},
    {"name": "EXISTENTIAL",  "desc": "Rare / low-frequency vocab",         "pi_min": 0.60, "pi_max": 0.80, "zone_key": "freq",         "freq_tier": "low"},
    {"name": "TRANSITIONAL", "desc": "Char-trigram neighbours",            "pi_min": 0.80, "pi_max": 1.01, "zone_key": "trigram_char", "freq_tier": None},
]


def _pi_activate_zone(pi_val):
    for z in PSPACE_ZONES:
        if z["pi_min"] <= pi_val < z["pi_max"]:
            return z["name"]
    return PSPACE_ZONES[-1]["name"]


def _restrict_specs_to_zone(context_index, zone_name):
    target   = next((z for z in PSPACE_ZONES if z["name"] == zone_name), None)
    if target is None or context_index is None:
        return getattr(context_index, "zone_specs", [])
    key      = target["zone_key"]
    matching = [s for s in context_index.zone_specs if s["name"] == key]
    return matching if matching else context_index.zone_specs


# ---------------------------------------------------------------------------
# _score_only  — PiSampler.sample up to but NOT including the final draw
# ---------------------------------------------------------------------------

def _score_only(sampler, dist):
    samples = list(dist.samples())
    if not samples:
        return []

    base_scored = []
    for s in samples:
        p     = max(1e-12, float(dist.prob(s)))
        count = sampler.history[s]
        if count > 0:
            p /= sampler.repetition_penalty ** count
        base_scored.append((s, p))

    base_scored = apply_insight_penalty(base_scored, sampler.insight_penalty)
    scored      = [(s, p ** (1.0 / sampler.temperature)) for s, p in base_scored]
    total       = sum(p for _, p in scored)
    if total > 0:
        scored = [(s, p / total) for s, p in scored]

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[: sampler.top_k]
    kept, accum = [], 0.0
    for s, p in scored:
        kept.append((s, p))
        accum += p
        if accum >= sampler.top_p:
            break
    scored = kept

    if sampler.seashell   is not None: scored = sampler.seashell.apply(scored)
    if sampler.semicircle is not None: scored = sampler.semicircle.apply(scored)
    return sampler._zone_attention_blend(scored)


def _combine_scored_geometric(scored_lists, floor=1e-12):
    if not scored_lists:
        return []
    cycle_maps = [dict(sl) for sl in scored_lists]
    union = set()
    for d in cycle_maps: union.update(d.keys())
    if not union:
        return []
    K        = len(cycle_maps)
    combined = []
    for w in union:
        log_sum = sum(math.log(max(floor, d.get(w, floor))) for d in cycle_maps)
        combined.append((w, math.exp(log_sum / K)))
    total = sum(p for _, p in combined)
    if total > 0:
        combined = [(w, p / total) for w, p in combined]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


# ---------------------------------------------------------------------------
# generate_text_pspace
# Pi stream activates PSPACE semantic areas; each area prints its span.
# Latent-BOS-sorted dataset informs the latent_bos zone spec throughout.
# Concatenation of all spans == full generated text.
# ---------------------------------------------------------------------------

def generate_text_pspace(
    cpd, sampler, prompt, n_words, ngram_n,
    vocab=None, context_index=None, n_cycles=3,
    span_words=8,
    verbose_zones=True,
):
    """
    Per span of `span_words` tokens:
      1. Draw pi-stream unit → activate PSPACE semantic zone.
      2. Restrict zone_specs to that zone for the span.
      3. Run combinatorial cycles over (zone_spec, prompt_bigram, pi_offset).
      4. Print and collect the span.
    Concatenation of spans == full generated text.
    """
    n_spans        = max(1, (n_words + span_words - 1) // span_words)
    context_window = ngram_n - 1
    seed_words     = tokenise_alpha(prompt)

    if vocab is not None:
        seed_in_vocab = [w for w in seed_words if w in vocab]
    else:
        seed_in_vocab = list(seed_words)

    if len(seed_in_vocab) >= context_window:
        init = seed_in_vocab[-context_window:]
    else:
        init = [""] * (context_window - len(seed_in_vocab)) + seed_in_vocab

    context   = deque(init, maxlen=context_window)
    out_words = list(seed_words)
    zone_log  = []

    if context_index is not None:
        sampler.set_context_index(context_index)

    original_specs = list(context_index.zone_specs) if context_index is not None else []
    original_ctx   = list(getattr(sampler, "instruction_context", []) or [])

    prompt_alpha = [w for w in seed_words if w]
    if len(prompt_alpha) >= 2:
        prompt_bigrams = [(prompt_alpha[i], prompt_alpha[i+1]) for i in range(len(prompt_alpha)-1)]
    elif prompt_alpha:
        prompt_bigrams = [(prompt_alpha[-1],)]
    else:
        prompt_bigrams = [tuple()]

    stream_len = max(1, len(sampler.stream))
    pi_offsets = [0, max(1, stream_len // 97), max(1, stream_len // 53)][:n_cycles]

    def diagonal_triples(K):
        snap = getattr(context_index, "zone_specs", [None]) if context_index else [None]
        nz, nb, no = len(snap), len(prompt_bigrams), len(pi_offsets)
        for k in range(K):
            yield (
                snap[k % nz] if snap else None,
                prompt_bigrams[k % nb],
                pi_offsets[k % no],
            )

    def dist_for_ctx(ctxtuple):
        for cut in range(len(ctxtuple), 0, -1):
            trial = ("",) * (context_window - cut) + ctxtuple[-cut:]
            try:
                d = cpd[trial]
                if list(d.samples()):
                    return d
            except Exception:
                continue
        try:
            d = cpd[tuple([""] * context_window)]
            if list(d.samples()):
                return d
        except Exception:
            pass
        return None

    words_remaining = n_words

    for _span_i in range(n_spans):
        if words_remaining <= 0:
            break

        # ── 1. Pi activates PSPACE zone ───────────────────────────────────
        pi_val    = sampler.next_unit()
        zone_name = _pi_activate_zone(pi_val)

        # ── 2. Restrict specs to active zone ──────────────────────────────
        if context_index is not None:
            context_index.zone_specs = _restrict_specs_to_zone(context_index, zone_name)

        span_len       = min(span_words, words_remaining)
        span_collected = []

        # ── 3. Generate span tokens ───────────────────────────────────────
        for _ in range(span_len):
            dist = dist_for_ctx(tuple(context))
            if dist is None:
                context.clear()
                context.extend([""] * context_window)
                continue

            cycle_scored = []
            saved_pos    = sampler.pos

            for zs, bg, off in diagonal_triples(n_cycles):
                if context_index is not None and zs is not None:
                    context_index.zone_specs = [zs]
                sampler.instruction_context = list(bg) if bg else []
                sampler.pos = (saved_pos + off) % stream_len

                scored = _score_only(sampler, dist)
                if scored:
                    cycle_scored.append(scored)

            sampler.pos = saved_pos
            sampler.instruction_context = original_ctx
            if context_index is not None:
                context_index.zone_specs = _restrict_specs_to_zone(context_index, zone_name)

            if not cycle_scored:
                context.clear(); context.extend([""] * context_window); continue

            combined = _combine_scored_geometric(cycle_scored)
            if not combined:
                context.clear(); context.extend([""] * context_window); continue

            unseen     = [(w, p) for w, p in combined if sampler.history[w] == 0]
            pool       = unseen if unseen else combined
            pool_total = sum(p for _, p in pool)
            if pool_total > 0:
                pool = [(w, p / pool_total) for w, p in pool]

            draw = sampler.next_unit()
            cumulative, chosen = 0.0, pool[-1][0]
            for word, p in pool:
                cumulative += p
                if draw < cumulative:
                    chosen = word
                    break

            sampler.history[chosen] += 1
            if chosen == "":
                context.clear(); context.extend([""] * context_window); continue

            span_collected.append(chosen)
            out_words.append(chosen)
            context.append(chosen)

        words_remaining -= span_len

        # ── 4. Print and record activated zone span ───────────────────────
        span_text = " ".join(span_collected)
        zone_log.append((zone_name, span_text))
        if verbose_zones:
            zdef = next((z for z in PSPACE_ZONES if z["name"] == zone_name), {})
            print(f"[PSPACE:{zone_name}] (pi={pi_val:.4f}, {zdef.get('desc','')}) → {span_text}")

    # Restore
    if context_index is not None:
        context_index.zone_specs = original_specs
    sampler.instruction_context = original_ctx

    return capitalise_text(out_words), zone_log


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def corpus_fingerprint(corpus):
    if isinstance(corpus, bytes):
        return hashlib.md5(corpus).hexdigest()
    return hashlib.md5((corpus or "").encode("utf-8", errors="ignore")).hexdigest()


def resolve_corpus(file_obj, pasted_corpus):
    if file_obj is not None:
        try:
            with open(file_obj.name, "rb") as fh:
                raw = fh.read()
            try:
                text = gzip.decompress(raw).decode("utf-8", errors="ignore")
                return text, "gzip file"
            except Exception:
                return raw.decode("utf-8", errors="ignore"), "uploaded file"
        except Exception:
            pass
    if pasted_corpus and pasted_corpus.strip():
        return pasted_corpus.strip(), "pasted text"
    return EMBEDDED_CORPUS.strip(), "embedded corpus"


# ---------------------------------------------------------------------------
# get_or_build — build / cache model, stream, context index
# ---------------------------------------------------------------------------

def get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log,
                 progress=None):
    def _prog(frac, desc):
        if progress is not None:
            progress(frac, desc=desc)
        log.append(desc)

    key = (corpus_fingerprint(corpus), int(ngram_n), float(lidstone_gamma),
           int(pi_prec), int(pi_stream_len))
    if CACHE.get('key') == key and CACHE.get('cpd') is not None:
        _prog(1.0, 'Using cached model and stream.')
        return CACHE['cpd'], CACHE['vocab'], CACHE['stream']
    CACHE['_corpus_text'] = corpus

    _prog(0.05, 'Building ngram model…')
    cpd, vocab = build_model(corpus, ngram_n, lidstone_gamma)

    _prog(0.35, f'Building pi stream (prec={pi_prec}, len={pi_stream_len})…')
    raw_stream = build_pi_stream(pi_prec, pi_stream_len)

    _prog(0.55, 'Applying roll_until_column_match…')
    matrix  = stream_to_matrix(raw_stream, n_cols=26)
    stacked, match_shift = roll_until_column_match(matrix)
    log.append(
        f'Column match shift={match_shift}; stacked shape={stacked.shape}.'
        if match_shift >= 0
        else f'No exact column match; stacked shape={stacked.shape}.'
    )
    stream = stacked.flatten().tolist()

    _prog(0.72, 'Building context zone index + trigram latent space…')
    corpus_tokens = tokenise_alpha(CACHE.get('_corpus_text', ''))
    context_index = build_context_index(vocab, cpd, corpus_tokens)

    log.append(
        f'Zone index: high={len(context_index.freq_zones["high"])} '
        f'mid={len(context_index.freq_zones["mid"])} '
        f'low={len(context_index.freq_zones["low"])} '
        f'alpha_keys={len(context_index.alpha_zones)} '
        f'ngram_keys={len(context_index.ngram_zones)}.'
    )

    # Latent BOS sort summary
    if context_index.latent_sorted_keys:
        top_ctx = context_index.latent_sorted_keys[0]
        top_sim = context_index.latent_sim_scores[top_ctx]
        log.append(
            f'Latent BOS sort: {len(context_index.latent_sorted_keys)} contexts sorted. '
            f'Top context={top_ctx}, cosine_sim={top_sim:.4f}. '
            + 'Quartile sizes: '
            + ', '.join(
                f'q{i}={len(context_index.latent_bos_data.get(f"q{i}", []))}'
                for i in range(4)
            )
        )

    CACHE.update(key=key, cpd=cpd, vocab=vocab, stream=stream, context_index=context_index)
    _prog(1.0, 'Model ready — cached.')
    return cpd, vocab, stream


# ---------------------------------------------------------------------------
# _make_sampler
# ---------------------------------------------------------------------------

def _make_sampler(stream, temperature, top_k, top_p, rep_penalty,
                  seashell_enable, seashell_strength, seashell_decay,
                  seashell_peaks, seashell_width, seashell_floor,
                  insight_penalty,
                  semicircle_enable=False, semicircle_strength=0.6,
                  semicircle_arches=5, semicircle_radius=1.0,
                  semicircle_speed=0.05, semicircle_floor=0.05,
                  instruction_context=None,
                  context_index=None):
    sampler = PiSampler(
        stream,
        digits_per_sample=DEFAULTS['DIGITS_PER_SAMPLE'],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        seashell_enable=seashell_enable,
        seashell_strength=seashell_strength,
        seashell_decay=seashell_decay,
        seashell_peaks=seashell_peaks,
        seashell_width=seashell_width,
        seashell_floor=seashell_floor,
        semicircle_enable=semicircle_enable,
        semicircle_strength=semicircle_strength,
        semicircle_arches=semicircle_arches,
        semicircle_radius=semicircle_radius,
        semicircle_speed=semicircle_speed,
        semicircle_floor=semicircle_floor,
        insight_penalty=insight_penalty,
        instruction_context=instruction_context,
    )
    if context_index is not None:
        sampler.set_context_index(context_index)
    return sampler


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------

def run_single(
    file_obj, pasted_corpus, prompt,
    pi_prec, pi_stream_len, ngram_n, lidstone_gamma,
    gen_words, temperature, top_k, top_p, rep_penalty,
    seashell_enable, seashell_strength, seashell_decay,
    seashell_peaks, seashell_width, seashell_floor,
    bend_degrees, offset, vertex, insight_penalty,
    semicircle_enable=False, semicircle_strength=0.6,
    semicircle_arches=5, semicircle_radius=1.0,
    semicircle_speed=0.05, semicircle_floor=0.05,
    progress=gr.Progress(track_tqdm=False),
):
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f'Corpus source: {source} ({len(corpus)} chars).')
    cpd, vocab, stream = get_or_build(
        corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log, progress=progress
    )
    triangle = Triangle(int(pi_stream_len), offset_extra=int(offset), bend_degrees=float(bend_degrees))
    start    = triangle.vertices[vertex]
    log.append(f'Triangle A={triangle.A} B={triangle.B} C={triangle.C} vertex={vertex} start={start}')

    ctx_words = tokenise_alpha(prompt or '')
    sampler   = _make_sampler(
        stream, temperature, top_k, top_p, rep_penalty,
        seashell_enable, seashell_strength, seashell_decay,
        seashell_peaks, seashell_width, seashell_floor,
        insight_penalty,
        semicircle_enable, semicircle_strength, semicircle_arches,
        semicircle_radius, semicircle_speed, semicircle_floor,
        instruction_context=ctx_words,
        context_index=CACHE.get('context_index'),
    )
    sampler.seek(start)

    text, zone_log = generate_text_pspace(
        cpd, sampler,
        prompt=prompt or '', n_words=int(gen_words), ngram_n=int(ngram_n),
        vocab=vocab,
        context_index=CACHE.get('context_index'),
        n_cycles=DEFAULTS['CYCLES_N'],
        span_words=DEFAULTS['SPAN_WORDS'],
        verbose_zones=False,
    )
    for zname, ztext in zone_log:
        log.append(f'[PSPACE:{zname}] {ztext}')

    oov = [w for w in tokenise_alpha(prompt or '') if w not in vocab]
    if oov:
        log.append(f'{len(oov)} prompt tokens not in corpus vocab: {oov}')
    log.append(f'Done. {len(text.split())} output tokens.')

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix='.txt', prefix='pigenerate_', mode='w', encoding='utf-8'
    )
    tmp.write(text)
    tmp.close()
    return text, '\n'.join(log), tmp.name


# ---------------------------------------------------------------------------
# run_generate  (lightweight tab)
# ---------------------------------------------------------------------------

def run_generate(file_obj, pasted_corpus, prompt,
                 pi_prec, pi_stream_len, ngram_n, lidstone_gamma,
                 temperature, text_length, rep_penalty, insight_penalty,
                 semicircle_enable, semicircle_strength, semicircle_arches,
                 semicircle_radius, semicircle_speed, semicircle_floor,
                 progress=gr.Progress(track_tqdm=False)):
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f'Corpus source: {source} ({len(corpus)} chars).')
    cpd, vocab, stream = get_or_build(
        corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log, progress=progress
    )
    ctx_words = tokenise_alpha(prompt or '')
    sampler   = _make_sampler(
        stream, temperature,
        DEFAULTS['TOP_K'], DEFAULTS['TOP_P'], rep_penalty,
        DEFAULTS['SEASHELL_ENABLE'], DEFAULTS['SEASHELL_STRENGTH'],
        DEFAULTS['SEASHELL_DECAY'],  DEFAULTS['SEASHELL_PEAKS'],
        DEFAULTS['SEASHELL_WIDTH'],  DEFAULTS['SEASHELL_FLOOR'],
        insight_penalty,
        bool(semicircle_enable), semicircle_strength, semicircle_arches,
        semicircle_radius, semicircle_speed, semicircle_floor,
        instruction_context=ctx_words,
        context_index=CACHE.get('context_index'),
    )
    sampler.seek(0)

    text, zone_log = generate_text_pspace(
        cpd, sampler,
        prompt or '', int(text_length), int(ngram_n),
        vocab=vocab,
        context_index=CACHE.get('context_index'),
        n_cycles=DEFAULTS['CYCLES_N'],
        span_words=DEFAULTS['SPAN_WORDS'],
        verbose_zones=False,
    )
    for zname, ztext in zone_log:
        log.append(f'[PSPACE:{zname}] {ztext}')

    oov = [w for w in tokenise_alpha(prompt or '') if w not in vocab]
    if oov:
        log.append(f'{len(oov)} prompt tokens not in corpus vocab: {oov}')
    if semicircle_enable:
        log.append(
            f'Semicircle wave mask ON '
            f'(strength={semicircle_strength}, arches={int(semicircle_arches)}, '
            f'radius={semicircle_radius}, speed={semicircle_speed}, floor={semicircle_floor}).'
        )
    log.append(f'Generated {len(text.split())} tokens.')
    return text, '\n'.join(log)



def save_model_ui(file_obj, pasted_corpus, pi_prec, pi_stream_len, ngram_n, lidstone_gamma):
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f'Corpus source: {source} ({len(corpus)} chars).')
    cpd, vocab, stream = get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl.gz', prefix='pi_model_')
    tmp.close()
    save_model_to_path(tmp.name, cpd, vocab, stream, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, corpus)
    log.append(f'Saved model to {os.path.basename(tmp.name)}')
    return tmp.name, '\n'.join(log)


def load_model_ui(modelfile):
    if modelfile is None:
        return 'No file uploaded.', gr.update(), gr.update(), gr.update(), gr.update()
    path = modelfile.name if hasattr(modelfile, 'name') else modelfile
    cpd, vocab, stream, config, errors = load_model_from_path(path)
    if cpd is None:
        return 'Failed to load model: ' + ' | '.join(errors), gr.update(), gr.update(), gr.update(), gr.update()
    CACHE.update(key=('LOADED', path), cpd=cpd, vocab=vocab, stream=stream)
    log = [f'Loaded model from {os.path.basename(path)}.']
    if errors:
        log.extend([f'! {e}' for e in errors])
    return (
        '\n'.join(log),
        gr.update(value=config.get('pi_prec', DEFAULTS['PI_PREC'])),
        gr.update(value=config.get('pi_stream_len', DEFAULTS['PI_STREAM_LEN'])),
        gr.update(value=config.get('ngram_n', DEFAULTS['NGRAM_N'])),
        gr.update(value=config.get('lidstone_gamma', DEFAULTS['LIDSTONE_GAMMA'])),
    )

def save_model_to_path(path, cpd, vocab, stream, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, corpustext):
    payload = dict(
        magic='PI_TRIGRAM_MODEL_V1',
        version=1,
        cpd=cpd,
        vocab=set(vocab),
        stream=list(stream),
        config=dict(
            ngram_n=int(ngram_n),
            lidstone_gamma=float(lidstone_gamma),
            pi_prec=int(pi_prec),
            pi_stream_len=int(pi_stream_len),
            digits_per_sample=int(DEFAULTS['DIGITS_PER_SAMPLE']),
        ),
        corpus_sha256=corpus_fingerprint(corpustext),
        corpus_chars=len(corpustext) if corpustext else 0,
        vocab_size=len(vocab),
    )
    with gzip.open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model_from_path(path):
    errors = []
    try:
        with gzip.open(path, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
        except Exception as e2:
            return None, None, None, None, [
                f'Could not read file: {e}',
                f'Uncompressed fallback also failed: {e2}',
            ]
    if not isinstance(payload, dict):
        return None, None, None, None, ['File does not contain a model dict.']
    cpd = payload.get('cpd')
    vocab = payload.get('vocab')
    stream = payload.get('stream')
    config = payload.get('config', {})
    if cpd is None or vocab is None or stream is None:
        return None, None, None, None, errors + ['Missing required fields: cpd/vocab/stream.']
    return cpd, set(vocab), list(stream), dict(config), errors


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title='Full features app') as demo:
        status_md = gr.Markdown("### Ready\n\nNo model loaded. Use **Load Latest from HF** or load a local file.")
        with gr.Tabs():

            # ----------------------------------------------------------------
            # Tab: Generate
            # ----------------------------------------------------------------
            with gr.TabItem('Generate'):
                with gr.Row():
                    with gr.Column(scale=1):
                        filein = gr.File(label='Upload corpus (.txt)', file_types=['.txt', '.md'], type='filepath')
                        pasted = gr.Textbox(label='or paste corpus here', lines=6)
                        promptin = gr.Textbox(label='Prompt', lines=3, value='alice rabbit hole')
                    with gr.Column(scale=1):
                        pi_prec = gr.Slider(500, 30000, value=DEFAULTS['PI_PREC'], step=500, label='precision')
                        pi_stream_len = gr.Slider(500, 30000, value=DEFAULTS['PI_STREAM_LEN'], step=500, label='stream length')
                        ngram_n = gr.Slider(2, 6, value=DEFAULTS['NGRAM_N'], step=1, label='n-gram order')
                        lidstone_gamma = gr.Slider(0.001, 1.0, value=DEFAULTS['LIDSTONE_GAMMA'], step=0.001, label='Lidstone gamma')
                        temperature = gr.Slider(0.1, 5.0, value=DEFAULTS['TEMPERATURE'], step=0.05, label='Temperature')
                        text_length = gr.Slider(1, 2000, value=DEFAULTS['GEN_WORDS'], step=1, label='Text length')
                        rep_penalty = gr.Slider(1.0, 2.0, value=DEFAULTS['REP_PENALTY'], step=0.01, label='Repetition penalty')
                        insight_penalty_gen = gr.Slider(
                            0.0, 5.0,
                            value=DEFAULTS['INSIGHT_PENALTY'],
                            step=0.05,
                            label='Insight penalty â€” push away from conclusion-encoding labels',
                        )

                with gr.Accordion('Semicircle wave mask', open=False):
                    gr.Markdown(
                        'Reweights candidate probabilities with a travelling '
                        'train of semi-circular arches across the rank axis.'
                    )
                    semicircle_enable = gr.Checkbox(
                        value=DEFAULTS['SEMICIRCLE_ENABLE'],
                        label='Enable semicircle wave mask',
                    )
                    semicircle_strength = gr.Slider(
                        0.0, 1.0, value=DEFAULTS['SEMICIRCLE_STRENGTH'], step=0.01,
                        label='Strength â€” blend amount of the mask (0 = pass-through)',
                    )
                    semicircle_arches = gr.Slider(
                        1, 30, value=DEFAULTS['SEMICIRCLE_ARCHES'], step=1,
                        label='Arches â€” number of semi-circles tiled across candidates',
                    )
                    semicircle_radius = gr.Slider(
                        0.05, 2.0, value=DEFAULTS['SEMICIRCLE_RADIUS'], step=0.05,
                        label='Radius â€” arch width (>=1 fills the slot, smaller = narrower)',
                    )
                    semicircle_speed = gr.Slider(
                        0.0, 1.0, value=DEFAULTS['SEMICIRCLE_SPEED'], step=0.005,
                        label='Speed â€” phase advance per generation step (wave travel)',
                    )
                    semicircle_floor = gr.Slider(
                        1e-3, 1.0, value=DEFAULTS['SEMICIRCLE_FLOOR'], step=0.005,
                        label='Floor â€” minimum gain so no candidate is fully zeroed',
                    )

                btn_gen = gr.Button('Generate', variant='primary')
                outtext = gr.Textbox(label='Generated text', lines=18)
                outlog = gr.Textbox(label='Log', lines=8)
                btn_gen.click(
                    run_generate,
                    inputs=[
                        filein, pasted, promptin,
                        pi_prec, pi_stream_len, ngram_n, lidstone_gamma,
                        temperature, text_length, rep_penalty, insight_penalty_gen,
                        semicircle_enable, semicircle_strength, semicircle_arches,
                        semicircle_radius, semicircle_speed, semicircle_floor,
                    ],
                    outputs=[outtext, outlog],
                )

            # ----------------------------------------------------------------
            # Tab: Model I/O
            # ----------------------------------------------------------------
            with gr.TabItem('Model I/O'):
                gr.Markdown('Save/load compiled trigram model.')

                # â”€â”€ Load latest from HF (universal, always force-downloads) â”€â”€
                gr.Markdown('#### Load latest model from Hugging Face')
                model_hf_repo = gr.Textbox(label='HF repo ID', value=HF_REPO_ID)
                model_hf_token = gr.Textbox(label='HF token (optional)', type='password')
                load_latest_btn = gr.Button('ðŸ”„  Load Latest from HF', variant='primary')
                load_latest_log = gr.Textbox(label='Load log', lines=3, interactive=False)

                gr.Markdown('---')
                gr.Markdown('#### Save / load local .pkl.gz')
                save_btn = gr.Button('Save model', variant='secondary')
                save_file = gr.File(label='Saved model file', interactive=False)
                model_log = gr.Textbox(label='Model I/O log', lines=8)
                load_file = gr.File(label='Load saved model .pkl.gz', file_types=['.gz', '.pkl'], type='filepath')
                load_btn = gr.Button('Load local model', variant='secondary')

                def _load_latest_and_update_status(repo, tok):
                    msg = load_hf_model_on_demand(repo, tok)
                    status = HF_CACHE.get('status', 'Unknown')
                    return msg, f"### Model status\n\n`{status}`"

                load_latest_btn.click(
                    _load_latest_and_update_status,
                    inputs=[model_hf_repo, model_hf_token],
                    outputs=[load_latest_log, status_md],
                )
                save_btn.click(
                    save_model_ui,
                    inputs=[filein, pasted, pi_prec, pi_stream_len, ngram_n, lidstone_gamma],
                    outputs=[save_file, model_log],
                )
                load_btn.click(
                    load_model_ui,
                    inputs=[load_file],
                    outputs=[model_log, pi_prec, pi_stream_len, ngram_n, lidstone_gamma],
                )

        gr.Markdown('Tip: model caches are reused until corpus or configuration changes.')

        # auto-load on startup removed â€” use "Load Latest from HF" button

    return demo


if __name__ == '__main__':
    build_ui().queue(max_size=8).launch(
        server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'),
        show_error=True,
    )
