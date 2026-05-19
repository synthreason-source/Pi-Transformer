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
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("words", "corpora/words"),
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

HF_REPO_ID = "trainman999/Thinking-lite"
LOCAL_CACHE_DIR = os.path.join(NLTK_DATA_DIR, "hf_model_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
HF_CACHE = {"loaded": False, "tokenizer": None, "model": None, "status": "Idle"}
CACHE = dict(key=None, cpd=None, vocab=None, stream=None, context_index=None)
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
        p_adj = p / (1.0 + strength * excess / mean_p)
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
        self.bins = max(1, int(bins))

    def __call__(self, fd):
        return LidstoneProbDist(fd, gamma=self.gamma, bins=self.bins)


def build_model(corpus, ngram_n, lidstone_gamma):
    if isinstance(corpus, bytes):
        corpus = corpus.decode("utf-8", errors="ignore")
    elif not isinstance(corpus, str):
        corpus = str(corpus) if corpus is not None else ""
    ngram_n = int(ngram_n)
    if ngram_n < 2:
        ngram_n = 2
    lidstone_gamma = float(lidstone_gamma)
    tokens = tokenise_alpha(corpus)
    if not tokens:
        raise ValueError(
            "Corpus produced zero tokens after tokenisation. "
            "Upload a non-empty text corpus or paste some text."
        )
    padded = [""] * (ngram_n - 1) + tokens + [""]
    trigrams_ = list(ngrams(padded, ngram_n))
    cfd = ConditionalFreqDist((tuple(tg[:-1]), tg[-1]) for tg in trigrams_)
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
    length = int(length)
    mp.dps = decimals + 50
    D = 10 ** decimals
    frac = int(mp.floor(mpi * D)) - 3 * D
    stream = []
    for _ in range(length):
        frac *= 26
        stream.append(frac // D)
        frac %= D
    return stream


# ---------------------------------------------------------------------------
# ContextZoneIndex — sorted/categorised data structures for attention
# ---------------------------------------------------------------------------

class ContextZoneIndex:
    """
    Pre-built, sorted data structures that the prompt can attend to.

    Zones
    -----
    freq_zones   : dict[str, list[str]]
        Words bucketed by corpus frequency tier:
        "high"   >= FREQ_HIGH_THRESH  occurrences
        "mid"    >= FREQ_MID_THRESH   occurrences
        "low"    < FREQ_MID_THRESH    occurrences
        Each list is sorted descending by frequency.

    alpha_zones  : dict[str, list[str]]
        Words bucketed by first letter (a–z), sorted alphabetically within
        each bucket.

    ngram_zones  : dict[tuple, list[str]]
        For every bigram context tuple in the CPD, the sorted list of
        successor words ordered by descending conditional probability.

    prompt_zone  : dict[str, list[str]]
        Dynamically populated per generation call: for each prompt word,
        stores the words in the corpus vocab that share at least one
        alphabetic trigram with it (loose semantic neighbourhood).

    Attention gradient
    ------------------
    `zone_gradient(zone_words, candidates, sigma)` returns a weight vector
    over `candidates` shaped as a Gaussian curve centred on the rank of the
    highest-scored zone word inside `candidates`.  Words outside the attended
    zone are masked toward a floor value, not zeroed — this is a soft mask.
    """

    FREQ_HIGH_THRESH = 10
    FREQ_MID_THRESH  = 3

    def __init__(self, vocab, cpd, token_freq):
        self.vocab      = set(vocab)
        self.token_freq = dict(token_freq)   # word → int count
        self._build_freq_zones()
        self._build_alpha_zones()
        self._build_ngram_zones(cpd)
        self._make_zone_specs()
        self.prompt_zone = {}

    # ──────────────────────────────────────────────────────────────────────
    # ZoneRetriever protocol — isomorphic retrieval syntax across all zones
    #
    # Every zone type is structurally identical:
    #
    #   key_fn(prompt_words)  →  [key₁, key₂, …]   (prompt → lookup keys)
    #   data[key]             →  [word₁, word₂, …]  (sorted candidate list)
    #   gradient params       →  (sigma, floor)      (attention curve shape)
    #
    # _ZONE_SPECS encodes this triple for each zone so that zone builders,
    # zone selection, and gradient computation all go through one code path.
    # ──────────────────────────────────────────────────────────────────────

    # Each spec: (name, sigma, floor)
    # key_fn and data are added dynamically in __init__ after the structures
    # are built.  See _make_zone_specs().
    _ZONE_SPEC_TEMPLATES = [
        ("ngram_bigram",  0.25, 0.04),
        ("ngram_unigram", 0.30, 0.04),
        ("alpha",         0.40, 0.05),
        ("freq",          0.50, 0.05),
        ("trigram_char",  0.35, 0.04),
    ]

    # ── zone builders ─────────────────────────────────────────────────────

    def _build_freq_zones(self):
        pairs = sorted(
            [(w, self.token_freq.get(w, 0)) for w in self.vocab if w],
            key=lambda x: x[1], reverse=True,
        )
        high, mid, low = [], [], []
        for w, c in pairs:
            if c >= self.FREQ_HIGH_THRESH:
                high.append(w)
            elif c >= self.FREQ_MID_THRESH:
                mid.append(w)
            else:
                low.append(w)
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

    # ── ZoneRetriever specs ───────────────────────────────────────────────

    def _make_zone_specs(self):
        """
        Build self.zone_specs: a list of dicts, one per zone type.
        Every spec has the same interface:
            spec['name']    : str
            spec['key_fn']  : prompt_words -> list of lookup keys
            spec['data']    : dict mapping key -> sorted word list
            spec['sigma']   : float  (Gaussian attention width)
            spec['floor']   : float  (minimum attention weight)

        This is the isomorphism: every zone is retrieved via the identical
        pattern  data.get(key_fn(prompt)[i], [])  regardless of zone type.
        The only things that differ are key_fn and data — not the retrieval
        syntax.
        """

        def _char_trigrams(word):
            return {word[i:i+3] for i in range(len(word) - 2)} if len(word) >= 3 else {word}

        # --- ngram_bigram: key = (w_i, w_{i+1}) ---
        def key_fn_bigram(prompt_words):
            return [
                (prompt_words[i], prompt_words[i + 1])
                for i in range(len(prompt_words) - 1)
            ]

        # --- ngram_unigram: key = (w_i,) ---
        def key_fn_unigram(prompt_words):
            return [(w,) for w in prompt_words if w]

        # --- alpha: key = first letter of each prompt word ---
        def key_fn_alpha(prompt_words):
            return [w[0] for w in prompt_words if w and w[0].isalpha()]

        # --- freq: key = "high" | "mid" | "low" selected by prompt ---
        high_set = set(self.freq_zones["high"])

        def key_fn_freq(prompt_words):
            if any(w in high_set for w in prompt_words):
                return ["high"]
            if all(self.token_freq.get(w, 0) < self.FREQ_MID_THRESH
                   for w in prompt_words):
                return ["low"]
            return ["mid"]

        # --- trigram_char: key = frozenset of char-trigrams of prompt ---
        # Data is built on-the-fly as a single-key dict keyed by sentinel.
        trig_index = defaultdict(set)
        for v in self.vocab:
            if v:
                for tg in _char_trigrams(v):
                    trig_index[tg].add(v)
        self._trig_index = trig_index

        def key_fn_trigram(prompt_words):
            prompt_tgs = set()
            for w in prompt_words:
                prompt_tgs |= _char_trigrams(w)
            # Use a single sentinel key "__trig__" that maps to neighbours
            neighbours = set()
            for tg in prompt_tgs:
                neighbours |= trig_index.get(tg, set())
            self._trig_data["__trig__"] = sorted(neighbours)
            return ["__trig__"]

        self._trig_data = {}

        # Assemble specs — all five use the same retrieval interface
        templates = {t[0]: (t[1], t[2]) for t in self._ZONE_SPEC_TEMPLATES}
        self.zone_specs = [
            dict(name="ngram_bigram",  key_fn=key_fn_bigram,   data=self.ngram_zones,  sigma=templates["ngram_bigram"][0],  floor=templates["ngram_bigram"][1]),
            dict(name="ngram_unigram", key_fn=key_fn_unigram,  data=self.ngram_zones,  sigma=templates["ngram_unigram"][0], floor=templates["ngram_unigram"][1]),
            dict(name="alpha",         key_fn=key_fn_alpha,    data=self.alpha_zones,  sigma=templates["alpha"][0],         floor=templates["alpha"][1]),
            dict(name="freq",          key_fn=key_fn_freq,     data=self.freq_zones,   sigma=templates["freq"][0],          floor=templates["freq"][1]),
            dict(name="trigram_char",  key_fn=key_fn_trigram,  data=self._trig_data,   sigma=templates["trigram_char"][0],  floor=templates["trigram_char"][1]),
        ]

    # ── prompt-driven zone selection (isomorphic across all zone types) ───

    def select_zones_for_prompt(self, prompt_words):
        """
        Unified retrieval: for every zone spec, call key_fn(prompt_words)
        to get the lookup keys, then retrieve data[key] for each key.
        All five zone types use the identical retrieval syntax.

        Returns a priority-ordered, deduped list of candidate words.
        """
        selected = []
        seen = set()

        def _add(words):
            for w in words:
                if w and w not in seen:
                    seen.add(w)
                    selected.append(w)

        # Isomorphic retrieval loop — same syntax for every zone type
        for spec in self.zone_specs:
            keys = spec["key_fn"](prompt_words)   # prompt → lookup keys
            for key in keys:
                _add(spec["data"].get(key, []))    # data[key] → word list

        return selected

    # ── attention gradient ────────────────────────────────────────────────

    @staticmethod
    def zone_gradient(zone_set, candidates, sigma=0.35, floor=0.05):
        """
        Soft Gaussian attention mask over `candidates`.

        Parameters
        ----------
        zone_set   : set of words that belong to the attended zone
        candidates : list of (word, prob) in rank order (index 0 = highest prob)
        sigma      : std-dev of the Gaussian window in normalised rank units
        floor      : minimum weight so no candidate is fully zeroed

        Returns a numpy weight vector of shape (len(candidates),) summing to 1.
        """
        n = len(candidates)
        if n == 0:
            return np.array([], dtype=np.float64)

        indices = np.arange(n, dtype=np.float64)
        norm_idx = indices / max(1, n - 1)   # [0, 1]

        # Find the normalised rank centroid of zone members within candidates
        zone_ranks = [
            i / max(1, n - 1)
            for i, (w, _) in enumerate(candidates)
            if w in zone_set
        ]
        if zone_ranks:
            centre = float(np.mean(zone_ranks))
        else:
            centre = 0.0   # default: attend toward high-probability end

        gauss = np.exp(-0.5 * ((norm_idx - centre) / max(1e-6, sigma)) ** 2)
        weights = floor + (1.0 - floor) * gauss
        weights /= weights.sum()
        return weights


def build_context_index(vocab, cpd, corpus_tokens):
    """Build a ContextZoneIndex from already-tokenised corpus data."""
    freq = Counter(corpus_tokens)
    return ContextZoneIndex(vocab, cpd, freq)


# ---------------------------------------------------------------------------
# vstack + np.roll column-match loop
# ---------------------------------------------------------------------------

def stream_to_matrix(stream, n_cols=26):
    """Reshape flat stream into a 2D numpy array of shape (rows, n_cols)."""
    arr = np.array(stream, dtype=np.int32)
    trim = (len(arr) // n_cols) * n_cols
    return arr[:trim].reshape(-1, n_cols)


def roll_until_column_match(arr, max_iters=None):
    """
    vstack arr with np.roll(arr, shift, axis=1) for shift=1,2,... until
    at least one column index i satisfies rolled[:, i] == original[:, i]
    element-wise. Returns (stacked_array, winning_shift).
    shift=-1 means no column match found within max_iters.
    """
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_cols = arr.shape[1]
    if max_iters is None:
        max_iters = n_cols

    original = arr.copy()
    stacked = arr.copy()

    for shift in range(1, max_iters + 1):
        rolled = np.roll(original, shift, axis=1)
        stacked = np.vstack([stacked, rolled])
        if np.any(np.all(rolled == original, axis=0)):
            return stacked, shift

    return stacked, -1


# ---------------------------------------------------------------------------
# SeashellResonator
# ---------------------------------------------------------------------------

class SeashellResonator:
    def __init__(self, sampler, strength, decay, peaks, width, floor):
        self.sampler = sampler
        self.base_strength = max(0.0, float(strength))
        self.decay = min(0.9999, max(0.0, float(decay)))
        self.peaks = max(1, int(peaks))
        self.width = max(0.02, float(width))
        self.floor = max(1e-6, float(floor))
        self.energy = 1.0
        self.step_index = 0
        self.centers = []
        self.phases = []
        self.spreads = []
        self._seed_from_stream()

    def _seed_from_stream(self):
        self.centers = []
        self.phases = []
        self.spreads = []
        for _ in range(self.peaks):
            c = self.sampler.next_unit()
            ph = 2.0 * math.pi * self.sampler.next_unit()
            spread = self.width * (0.65 + 0.7 * self.sampler.next_unit())
            self.centers.append(c)
            self.phases.append(ph)
            self.spreads.append(spread)

    def reset(self):
        self.energy = 1.0
        self.step_index = 0
        self._seed_from_stream()

    def _wrapped_distance(self, a, b):
        d = abs(a - b)
        return min(d, 1.0 - d)

    def gains(self, n_items):
        if n_items <= 0:
            return []
        gains = []
        t = self.step_index
        live_strength = self.base_strength * self.energy
        drift = 0.017 * math.sin(0.11 * t)
        shimmer = 0.09 * math.sin(0.19 * t + 1.7)
        for rank in range(n_items):
            idx = rank / max(1, n_items - 1)
            response = 0.0
            for center, phase, spread in zip(self.centers, self.phases, self.spreads):
                moving_center = (center + drift * math.sin(phase + 0.07 * t)) % 1.0
                d = self._wrapped_distance(idx, moving_center)
                gauss = math.exp(-(d * d) / max(1e-9, 2.0 * spread * spread))
                ripple = 0.5 + 0.5 * math.cos(
                    (d / max(1e-9, spread)) * math.pi * (1.5 + shimmer)
                    + phase
                    + 0.13 * t
                )
                response += gauss * (0.55 + 0.45 * ripple)
            response /= max(1, self.peaks)
            gain = self.floor + (1.0 - self.floor) + live_strength * response
            gains.append(max(self.floor, gain))
        total = sum(gains)
        if total > 0:
            gains = [g / total for g in gains]
        return gains

    def apply(self, scored):
        if not scored:
            return scored
        g = self.gains(len(scored))
        weighted = [(w, p * gain) for (w, p), gain in zip(scored, g)]
        total = sum(p for _, p in weighted)
        if total > 0:
            weighted = [(w, p / total) for w, p in weighted]
        self.energy *= self.decay
        self.step_index += 1
        if self.energy < 0.08:
            self.energy = 1.0
            self._seed_from_stream()
        return weighted


# ---------------------------------------------------------------------------
# SemicircleWaveMask
# ---------------------------------------------------------------------------

class SemicircleWaveMask:
    def __init__(self, sampler, strength, arches, radius, speed, floor):
        self.sampler = sampler
        self.strength = max(0.0, float(strength))
        self.arches = max(1, int(arches))
        self.radius = max(0.02, float(radius))
        self.speed = float(speed)
        self.floor = max(1e-6, float(floor))
        self.step_index = 0
        self.phase0 = self.sampler.next_unit()

    def reset(self):
        self.step_index = 0
        self.phase0 = self.sampler.next_unit()

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
            idx = rank / max(1, n_items - 1)
            slot_pos = (idx * self.arches + phase) % 1.0
            arch = self._arch(slot_pos)
            gain = (1.0 - self.strength) + self.strength * arch
            gains.append(max(self.floor, gain))
        total = sum(gains)
        if total > 0:
            gains = [g / total for g in gains]
        return gains

    def apply(self, scored):
        if not scored:
            return scored
        g = self.gains(len(scored))
        weighted = [(w, -p * gain) for (w, p), gain in zip(scored, g)]
        total = sum(p for _, p in weighted)
        if total > 0:
            weighted = [(w, p / total) for w, p in weighted]
        self.step_index += 1
        return weighted


# ---------------------------------------------------------------------------
# PiSampler  — XOR fusion replaced with vstack/hstack instruction-context blend
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
        self.stream = stream
        self.digits_per_sample = digits_per_sample
        self.pos = 0
        self.temperature = max(1e-3, float(temperature))
        self.top_k = max(1, int(top_k))
        self.top_p = max(1e-3, min(1.0, float(top_p)))
        self.repetition_penalty = max(1.0, float(repetition_penalty))
        self.insight_penalty = max(0.0, float(insight_penalty))
        self.history = Counter()
        self.instruction_context = list(instruction_context) if instruction_context else []
        self.seashell = None
        if seashell_enable:
            self.seashell = SeashellResonator(
                self,
                seashell_strength,
                seashell_decay,
                seashell_peaks,
                seashell_width,
                seashell_floor,
            )
        self.semicircle = None
        if semicircle_enable:
            self.semicircle = SemicircleWaveMask(
                self,
                semicircle_strength,
                semicircle_arches,
                semicircle_radius,
                semicircle_speed,
                semicircle_floor,
            )

    def seek(self, pos):
        self.pos = pos % len(self.stream)
        self.history.clear()
        if self.seashell is not None:
            self.seashell.reset()
        if self.semicircle is not None:
            self.semicircle.reset()

    def next_unit(self):
        val = 0
        base = 26 ** self.digits_per_sample
        for _ in range(self.digits_per_sample):
            val = val * 26 + self.stream[self.pos % len(self.stream)]
            self.pos += 1
        return val / base

    # ------------------------------------------------------------------
    # Zone-attention blend
    # The prompt selects which sorted/categorised data structure zones to
    # attend to.  Within each attended zone, a Gaussian probability gradient
    # masks away the non-attended parts of the candidate set.
    # ------------------------------------------------------------------

    def set_context_index(self, context_index):
        """Attach the pre-built ContextZoneIndex for this generation run."""
        self._context_index = context_index

    def _zone_attention_blend(self, scored):
        """
        Attention over sorted/categorised data structures.

        Pipeline
        --------
        1. Use self.instruction_context (prompt words) to SELECT which zones
           of the ContextZoneIndex to attend to — this returns an ordered
           priority list of candidate words (zone_selection).

        2. Build the zone membership set from zone_selection.

        3. Compute a per-zone Gaussian attention gradient over the scored
           candidates.  The gradient is a soft probability mask: candidates
           whose words sit inside the attended zone get a high weight, those
           outside get a floor weight.  Multiple zones are combined by
           vstack-ing their gradient rows and taking a weighted column mean.

        4. hstack a history-penalty column alongside the zone-gradient matrix
           so recency is factored into the final weight at no extra cost.

        5. Blend the resulting weight vector with the base probabilities via
           geometric mean and renormalise.
        """
        n = len(scored)
        if n == 0:
            return scored

        ctx = getattr(self, '_context_index', None)
        prompt = self.instruction_context

        # ── step 1: prompt selects zones ──────────────────────────────────
        if ctx is not None and prompt:
            zone_selection = ctx.select_zones_for_prompt(prompt)
        else:
            zone_selection = [w for w, _ in scored]

        zone_set = set(zone_selection)

        # ── step 2: per-zone gradient rows ────────────────────────────────
        # We compute one gradient row per distinct attended zone type so that
        # the zones' signals can be independently weighted before combining.
        gradient_rows = []

        if ctx is not None and prompt:
            # Isomorphic gradient loop — same retrieval syntax for all zone types.
            # For each zone spec: retrieve words via data[key_fn(prompt)[i]],
            # then compute a Gaussian attention gradient using the spec's sigma/floor.
            for spec in ctx.zone_specs:
                keys = spec["key_fn"](prompt)          # prompt → lookup keys
                for key in keys:
                    words = spec["data"].get(key, [])  # data[key] → word list
                    if words:
                        g = ContextZoneIndex.zone_gradient(
                            set(words), scored,
                            sigma=spec["sigma"],
                            floor=spec["floor"],
                        )
                        gradient_rows.append(g)

        # Fallback: uniform gradient if nothing was selected
        if not gradient_rows:
            gradient_rows.append(np.ones(n, dtype=np.float64) / n)

        # ── step 3: vstack zone gradient rows ─────────────────────────────
        zone_matrix = np.vstack(gradient_rows)        # (n_zones, n)

        # ── step 4: hstack history-penalty column ─────────────────────────
        history_col = np.array(
            [1.0 / (1.0 + self.history[w]) for w, _ in scored],
            dtype=np.float64,
        ).reshape(1, -1)                              # (1, n)
        full_matrix = np.vstack([zone_matrix, history_col])  # (n_zones+1, n)

        # hstack a normalisation sentinel: each row's own L1 norm as a
        # (n_rows, 1) column — this anchors the per-row scale before combining
        row_norms = full_matrix.sum(axis=1, keepdims=True).clip(1e-12)
        augmented = np.hstack([full_matrix, row_norms])       # (n_zones+1, n+1)

        # ── step 5: weighted column mean over candidate axis ──────────────
        # Weight each zone row by its mean gradient value (zones that are
        # more focused/informative get higher weight)
        row_weights = augmented[:, :n].mean(axis=1)           # (n_zones+1,)
        row_weights = np.clip(row_weights, 1e-12, None)
        row_weights /= row_weights.sum()
        weights = (augmented[:, :n] * row_weights[:, None]).sum(axis=0)  # (n,)
        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()

        # ── step 6: geometric-mean blend with base probabilities ──────────
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
            p = max(1e-12, float(dist.prob(s)))
            count = self.history[s]
            if count > 0:
                p /= self.repetition_penalty ** count
            base_scored.append((s, p))

        base_scored = apply_insight_penalty(base_scored, self.insight_penalty)

        scored = [(s, p ** (1.0 / self.temperature)) for s, p in base_scored]
        total = sum(p for _, p in scored)
        scored = [(s, p / total) for s, p in scored]

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[: self.top_k]
        kept = []
        accum = 0.0
        for s, p in scored:
            kept.append((s, p))
            accum += p
            if accum >= self.top_p:
                break
        scored = kept

        if self.seashell is not None:
            scored = self.seashell.apply(scored)

        if self.semicircle is not None:
            scored = self.semicircle.apply(scored)

        # zone-attention blend: prompt selects data structures, gradient masks non-attended
        scored = self._zone_attention_blend(scored)

        # ── novelty filter: prefer words not yet emitted ────────────────────
        unseen = [(w, p) for w, p in scored if self.history[w] == 0]
        pool = unseen if unseen else scored   # fall back if all candidates seen
        pool_total = sum(p for _, p in pool)
        if pool_total > 0:
            pool = [(w, p / pool_total) for w, p in pool]
        # ─────────────────────────────────────────────────────────────────────

        # single pi-stream draw for final selection
        draw = self.next_unit()
        cumulative = 0.0
        chosen = pool[-1][0]
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
        base = offset_extra % stream_len
        bend_shift = int(round((bend_degrees / 360.0) * stream_len))
        self.A = base % stream_len
        self.B = (base + stream_len // 3 + bend_shift) % stream_len
        self.C = (base + 2 * stream_len // 3 + bend_shift) % stream_len
        self.vertices = {"A": self.A, "B": self.B, "C": self.C}


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(cpd, sampler, prompt, n_words, ngram_n, vocab=None):
    context_window = ngram_n - 1
    seed_words = tokenise_alpha(prompt)
    if vocab is not None:
        seed_in_vocab = [w for w in seed_words if w in vocab]
    else:
        seed_in_vocab = list(seed_words)
    if len(seed_in_vocab) >= context_window:
        init = seed_in_vocab[-context_window:]
    else:
        init = [""] * (context_window - len(seed_in_vocab)) + seed_in_vocab
    context = deque(init, maxlen=context_window)
    words = list(seed_words)

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

    for _ in range(n_words):
        dist = dist_for_ctx(tuple(context))
        if dist is None:
            context.clear()
            context.extend([""] * context_window)
            continue
        word = sampler.sample(dist)
        if word == "":
            context.clear()
            context.extend([""] * context_window)
            continue
        words.append(word)
        context.append(word)
    return capitalise_text(words)


# ---------------------------------------------------------------------------
# Combinatorial cycles around generate_text
#
# Per-token, K cycles (K in [1,3]) walk a diagonal slice of the cartesian
# product (zone_spec, prompt_bigram, pi_stream_offset). Each cycle produces
# a full scored candidate list via _score_only (which mirrors PiSampler.sample
# up to but not including the final draw). The K lists are merged by geometric
# mean over the union vocab and a single pi-stream draw selects the winner.
#
# Scope: cheap. Diagonal walk keeps total cost O(K * sample-work) rather than
# O(|zones| * |bigrams| * |offsets|).
# ---------------------------------------------------------------------------

def _score_only(sampler, dist):
    """
    Mirror PiSampler.sample() through every blend step but stop before the
    final draw. Returns the scored list [(word, prob), ...].

    Does not mutate sampler.history. Does not call sampler.next_unit().
    Reads sampler.instruction_context and sampler._context_index.zone_specs
    which the cycle wrapper has already temporarily mutated.
    """
    samples = list(dist.samples())
    if not samples:
        return []

    base_scored = []
    for s in samples:
        p = max(1e-12, float(dist.prob(s)))
        count = sampler.history[s]
        if count > 0:
            p /= sampler.repetition_penalty ** count
        base_scored.append((s, p))

    base_scored = apply_insight_penalty(base_scored, sampler.insight_penalty)

    scored = [(s, p ** (1.0 / sampler.temperature)) for s, p in base_scored]
    total = sum(p for _, p in scored)
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

    if sampler.seashell is not None:
        scored = sampler.seashell.apply(scored)
    if sampler.semicircle is not None:
        scored = sampler.semicircle.apply(scored)

    scored = sampler._zone_attention_blend(scored)
    return scored


def _combine_scored_geometric(scored_lists, floor=1e-12):
    """Geometric-mean combine of K [(word, prob)] lists over the union vocab."""
    if not scored_lists:
        return []

    cycle_maps = [dict(sl) for sl in scored_lists]
    union = set()
    for d in cycle_maps:
        union.update(d.keys())
    if not union:
        return []

    K = len(cycle_maps)
    combined = []
    for w in union:
        log_sum = 0.0
        for d in cycle_maps:
            log_sum += math.log(max(floor, d.get(w, floor)))
        combined.append((w, math.exp(log_sum / K)))

    total = sum(p for _, p in combined)
    if total > 0:
        combined = [(w, p / total) for w, p in combined]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


def generate_text_combo_cycles(
    cpd, sampler, prompt, n_words, ngram_n,
    vocab=None, context_index=None, n_cycles=3,
):
    """
    generate_text + per-token combinatorial cycles over
    (zone_spec, prompt_bigram, pi_stream_offset).

    Pass n_cycles=1 and context_index=None to degenerate to the original
    generate_text behaviour.
    """
    n_cycles = max(1, min(3, int(n_cycles)))

    context_window = ngram_n - 1
    seed_words = tokenise_alpha(prompt)

    if vocab is not None:
        seed_in_vocab = [w for w in seed_words if w in vocab]
    else:
        seed_in_vocab = list(seed_words)
    if len(seed_in_vocab) >= context_window:
        init = seed_in_vocab[-context_window:]
    else:
        init = [""] * (context_window - len(seed_in_vocab)) + seed_in_vocab
    context = deque(init, maxlen=context_window)
    out_words = list(seed_words)

    # ── product axes ────────────────────────────────────────────────────
    prompt_alpha = [w for w in seed_words if w]
    if len(prompt_alpha) >= 2:
        prompt_bigrams = [
            (prompt_alpha[i], prompt_alpha[i + 1])
            for i in range(len(prompt_alpha) - 1)
        ]
    elif prompt_alpha:
        prompt_bigrams = [(prompt_alpha[-1],)]
    else:
        prompt_bigrams = [tuple()]

    if context_index is not None and getattr(context_index, "zone_specs", None):
        zone_specs = list(context_index.zone_specs)
    else:
        zone_specs = [None]

    stream_len = max(1, len(sampler.stream))
    pi_offsets = [
        0,
        max(1, stream_len // 97),
        max(1, stream_len // 53),
    ][:n_cycles]

    def diagonal_triples(K):
        nz, nb, no = len(zone_specs), len(prompt_bigrams), len(pi_offsets)
        for k in range(K):
            yield (
                zone_specs[k % nz],
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

    # ── attach context index, snapshot mutable state ────────────────────
    if context_index is not None:
        sampler.set_context_index(context_index)

    original_ctx = list(getattr(sampler, "instruction_context", []) or [])
    original_specs = (
        list(context_index.zone_specs) if context_index is not None else []
    )

    # ── per-token loop ──────────────────────────────────────────────────
    for _ in range(n_words):
        dist = dist_for_ctx(tuple(context))
        if dist is None:
            context.clear()
            context.extend([""] * context_window)
            continue

        cycle_scored = []
        saved_pos = sampler.pos

        for zs, bg, off in diagonal_triples(n_cycles):
            if context_index is not None and zs is not None:
                context_index.zone_specs = [zs]
            sampler.instruction_context = list(bg) if bg else []
            sampler.pos = (saved_pos + off) % stream_len

            scored = _score_only(sampler, dist)
            if scored:
                cycle_scored.append(scored)

        # restore mutated state before the canonical draw
        sampler.pos = saved_pos
        sampler.instruction_context = original_ctx
        if context_index is not None:
            context_index.zone_specs = original_specs

        if not cycle_scored:
            context.clear()
            context.extend([""] * context_window)
            continue

        combined = _combine_scored_geometric(cycle_scored)
        if not combined:
            context.clear()
            context.extend([""] * context_window)
            continue

        # ── novelty filter: prefer words not yet emitted ────────────────────
        unseen_combo = [(w, p) for w, p in combined if sampler.history[w] == 0]
        pool_combo = unseen_combo if unseen_combo else combined
        pool_combo_total = sum(p for _, p in pool_combo)
        if pool_combo_total > 0:
            pool_combo = [(w, p / pool_combo_total) for w, p in pool_combo]
        # ─────────────────────────────────────────────────────────────────────

        # single canonical pi-stream draw per token
        draw = sampler.next_unit()
        cumulative = 0.0
        chosen = pool_combo[-1][0]
        for word, p in pool_combo:
            cumulative += p
            if draw < cumulative:
                chosen = word
                break

        sampler.history[chosen] += 1

        if chosen == "":
            context.clear()
            context.extend([""] * context_window)
            continue

        out_words.append(chosen)
        context.append(chosen)

    return capitalise_text(out_words)


def all_pairs_match(pairs, text, fuzzy_threshold):
    lower_text = text.lower()
    for pair in pairs:
        pair_str = " ".join(pair)
        if pair_str in lower_text:
            continue
        if SequenceMatcher(None, pair_str, lower_text).quick_ratio() >= fuzzy_threshold:
            continue
        return False, pair
    return True, None


def collocation_association_score(text, prompt, min_freq=1, measure="pmi"):
    text_tokens = tokenise_alpha(text)
    prompt_tokens = tokenise_alpha(prompt)
    pairs = list(ngrams(prompt_tokens, 2))
    trigrams = list(ngrams(prompt_tokens, 3))
    bigram_finder = BigramCollocationFinder.from_words(text_tokens)
    trigram_finder = TrigramCollocationFinder.from_words(text_tokens)
    if min_freq > 1:
        bigram_finder.apply_freq_filter(min_freq)
        trigram_finder.apply_freq_filter(min_freq)
    bm = BigramAssocMeasures()
    tm = TrigramAssocMeasures()
    if measure == "likelihood_ratio":
        bigram_scores = dict(bigram_finder.score_ngrams(bm.likelihood_ratio))
        trigram_scores = dict(trigram_finder.score_ngrams(tm.likelihood_ratio))
    else:
        bigram_scores = dict(bigram_finder.score_ngrams(bm.pmi))
        trigram_scores = dict(trigram_finder.score_ngrams(tm.pmi))
    score = 0.0
    matched_pairs = []
    for pair in pairs:
        if pair in bigram_scores:
            score += bigram_scores[pair]
            matched_pairs.append(pair)
    for tri in trigrams:
        if tri in trigram_scores:
            score += trigram_scores[tri]
    return score, matched_pairs


def find_words(stream, dictionary, word_find_min):
    prefixes = set()
    for w in dictionary:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])
    buf = deque(maxlen=35)
    all_chars = []
    found = defaultdict(list)
    for pos, digit in enumerate(stream):
        ch = chr(ord('a') + digit)
        buf.append(ch)
        all_chars.append(ch)
        s = ''.join(buf)
        for length in range(word_find_min, min(16, len(s)) + 1):
            cand = s[-length:]
            if cand not in prefixes:
                continue
            if cand in dictionary:
                found[cand].append(pos - length + 1)
    return ''.join(all_chars), found


# ---------------------------------------------------------------------------
# Corpus / model utilities
# ---------------------------------------------------------------------------

def resolve_corpus(file_obj, pasted_corpus):
    if file_obj is not None:
        path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            if txt.strip():
                return txt, f'file:{os.path.basename(path)}'
        except Exception as e:
            return EMBEDDED_CORPUS, f'embedded fallback (file read failed: {e})'
    if pasted_corpus and pasted_corpus.strip():
        return pasted_corpus, 'pasted text'
    return EMBEDDED_CORPUS, 'embedded fallback'


def corpus_fingerprint(corpus):
    if isinstance(corpus, str):
        b = corpus.encode('utf-8', errors='ignore')
    elif isinstance(corpus, bytes):
        b = corpus
    else:
        b = str(corpus).encode('utf-8', errors='ignore')
    return hashlib.sha256(b).hexdigest()


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


def get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log,
                 progress=None):
    """Build (or retrieve cached) model+stream.  If `progress` is a gr.Progress
    instance it will emit labelled steps so the UI shows a load bar."""

    def _prog(frac, desc):
        if progress is not None:
            progress(frac, desc=desc)
        log.append(desc)

    key = (corpus_fingerprint(corpus), int(ngram_n), float(lidstone_gamma), int(pi_prec), int(pi_stream_len))
    if CACHE.get('key') == key and CACHE.get('cpd') is not None:
        _prog(1.0, 'Using cached model and stream.')
        return CACHE['cpd'], CACHE['vocab'], CACHE['stream']
    CACHE['_corpus_text'] = corpus

    _prog(0.05, 'Building trigram model…')
    cpd, vocab = build_model(corpus, ngram_n, lidstone_gamma)

    _prog(0.35, f'Building pi stream  (prec={pi_prec}, len={pi_stream_len})…')
    raw_stream = build_pi_stream(pi_prec, pi_stream_len)

    # --- vstack + np.roll column-match loop ---
    _prog(0.55, 'Applying roll_until_column_match to stream…')
    matrix = stream_to_matrix(raw_stream, n_cols=26)
    stacked, match_shift = roll_until_column_match(matrix)
    if match_shift >= 0:
        log.append(f'Column match found at shift={match_shift}; stacked shape={stacked.shape}.')
    else:
        log.append(f'No exact column match within {matrix.shape[1]} shifts; using full stacked shape={stacked.shape}.')
    stream = stacked.flatten().tolist()
    # ------------------------------------------

    _prog(0.75, 'Building context zone index…')
    corpus_tokens = tokenise_alpha(
        CACHE.get('_corpus_text', '')
    )
    context_index = build_context_index(vocab, cpd, corpus_tokens)
    log.append(
        f'Zone index: high={len(context_index.freq_zones["high"])} '
        f'mid={len(context_index.freq_zones["mid"])} '
        f'low={len(context_index.freq_zones["low"])} '
        f'alpha_keys={len(context_index.alpha_zones)} '
        f'ngram_keys={len(context_index.ngram_zones)}.'
    )
    CACHE.update(key=key, cpd=cpd, vocab=vocab, stream=stream, context_index=context_index)
    _prog(1.0, 'Model ready — cached.')
    return cpd, vocab, stream


def _make_sampler(stream, temperature, top_k, top_p, rep_penalty,
                  seashell_enable, seashell_strength, seashell_decay,
                  seashell_peaks, seashell_width, seashell_floor,
                  insight_penalty,
                  semicircle_enable=False, semicircle_strength=0.6,
                  semicircle_arches=5, semicircle_radius=1.0,
                  semicircle_speed=0.05, semicircle_floor=0.05,
                  instruction_context=None,
                  context_index=None):
    return PiSampler(
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


# ---------------------------------------------------------------------------
# run_single / run_search / run_generate
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
    cpd, vocab, stream = get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log, progress=progress)
    triangle = Triangle(int(pi_stream_len), offset_extra=int(offset), bend_degrees=float(bend_degrees))
    start = triangle.vertices[vertex]
    log.append(f'Triangle A={triangle.A} B={triangle.B} C={triangle.C} vertex={vertex} start={start}')
    ctx_words = tokenise_alpha(prompt or '')
    sampler = _make_sampler(
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
    text = generate_text_combo_cycles(
        cpd, sampler,
        prompt=prompt or '', n_words=int(gen_words), ngram_n=int(ngram_n),
        vocab=vocab,
        context_index=CACHE.get('context_index'),
        n_cycles=DEFAULTS['CYCLES_N'],
    )
    oov = [w for w in tokenise_alpha(prompt or '') if w not in vocab]
    if oov:
        log.append(f'{len(oov)} prompt tokens not in corpus vocab: {oov}')
    log.append(f'Done. {len(text.split())} output tokens.')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', prefix='pigenerate_', mode='w', encoding='utf-8')
    tmp.write(text)
    tmp.close()
    return text, '\n'.join(log), tmp.name


def run_search(
    file_obj, pasted_corpus, prompt,
    pi_prec, pi_stream_len, ngram_n, lidstone_gamma,
    gen_words, temperature, rep_penalty,
    seashell_enable, seashell_strength, seashell_decay,
    seashell_peaks, seashell_width, seashell_floor,
    vertex, bend_max, bend_step, offset_step,
    fuzzy_threshold, max_solutions, insight_penalty,
    semicircle_enable=False, semicircle_strength=0.6,
    semicircle_arches=5, semicircle_radius=1.0,
    semicircle_speed=0.05, semicircle_floor=0.05,
    progress=gr.Progress(track_tqdm=False),
):
    log = []
    if not prompt or not prompt.strip():
        return '', 'Prompt is empty — search needs word pairs.', None
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f'Corpus source: {source} ({len(corpus)} chars).')
    cpd, vocab, stream = get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log, progress=progress)
    pairs = extract_word_pairs(prompt)
    if not pairs:
        return '', 'No valid word pairs extracted from prompt.', None
    log.append(f'Prompt yields {len(pairs)} word pairs.')
    ctx_words = tokenise_alpha(prompt)
    bend_values = []
    b = 0.0
    while b <= float(bend_max) + 1e-9:
        bend_values.append(round(b, 4))
        b += float(bend_step)
    offset_values = list(range(0, int(pi_stream_len), max(1, int(offset_step))))
    total = len(bend_values) * len(offset_values)
    log.append(f'Search grid: {len(bend_values)} bends × {len(offset_values)} offsets = {total} candidates.')
    scored_results = []
    counter = 0
    for bend in bend_values:
        for offset in offset_values:
            counter += 1
            progress(counter / max(1, total), desc=f'bend={bend} offset={offset}')
            triangle = Triangle(int(pi_stream_len), offset_extra=offset, bend_degrees=bend)
            start = triangle.vertices[vertex]
            sampler = _make_sampler(
                stream, temperature, DEFAULTS['TOP_K'], DEFAULTS['TOP_P'], rep_penalty,
                seashell_enable, seashell_strength, seashell_decay,
                seashell_peaks, seashell_width, seashell_floor,
                insight_penalty,
                semicircle_enable, semicircle_strength, semicircle_arches,
                semicircle_radius, semicircle_speed, semicircle_floor,
                instruction_context=ctx_words,
                context_index=CACHE.get('context_index'),
            )
            sampler.seek(start)
            text = generate_text_combo_cycles(
                cpd, sampler,
                prompt=prompt, n_words=int(gen_words), ngram_n=int(ngram_n),
                vocab=vocab,
                context_index=CACHE.get('context_index'),
                n_cycles=DEFAULTS['CYCLES_N'],
            )
            exact_ok, _failed = all_pairs_match(pairs, text, fuzzy_threshold=float(fuzzy_threshold))
            assoc_score, matched_pairs = collocation_association_score(text, prompt, min_freq=1, measure='pmi')
            if exact_ok or assoc_score > 0:
                scored_results.append({
                    'prompt': prompt,
                    'bend': bend,
                    'offset': offset,
                    'vertex': vertex,
                    'text': text,
                    'assoc_score': assoc_score,
                    'matched_pairs': matched_pairs,
                    'exact_ok': exact_ok,
                })
                log.append(f'✓ candidate bend={bend} offset={offset} score={assoc_score:.4f}')
    if not scored_results:
        log.append('No matches found in the searched grid.')
        return '', '\n'.join(log), None
    scored_results.sort(key=lambda r: (r['exact_ok'], r['assoc_score'], len(r['matched_pairs'])), reverse=True)
    top_results = scored_results[: int(max_solutions)]
    parts = []
    for i, r in enumerate(top_results, 1):
        parts.append(
            f"=== MATCH {i} ===\n"
            f"bend = {r['bend']} offset = {r['offset']} vertex = {r['vertex']}\n"
            f"assoc_score = {r['assoc_score']:.4f} exact_ok = {r['exact_ok']}\n"
            f"matched_pairs = {r['matched_pairs']}\n\n"
            f"{r['text']}\n"
        )
    rendered = '\n'.join(parts)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', prefix='pi_search_', mode='w', encoding='utf-8')
    for r in top_results:
        tmp.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.close()
    return rendered, '\n'.join(log), tmp.name


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


def save_hf_model(repo_id, token=None):
    try:
        from huggingface_hub import HfApi, upload_file
        if CACHE.get('cpd') is None or CACHE.get('stream') is None or CACHE.get('vocab') is None:
            corpus, _ = resolve_corpus(None, None)
            cpd, vocab, stream = get_or_build(
                corpus, DEFAULTS['NGRAM_N'], DEFAULTS['LIDSTONE_GAMMA'],
                DEFAULTS['PI_PREC'], DEFAULTS['PI_STREAM_LEN'], [],
            )
        else:
            cpd, vocab, stream = CACHE['cpd'], CACHE['vocab'], CACHE['stream']
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl.gz', prefix='pi_model_')
        tmp.close()
        save_model_to_path(
            tmp.name, cpd, vocab, stream,
            DEFAULTS['NGRAM_N'], DEFAULTS['LIDSTONE_GAMMA'],
            DEFAULTS['PI_PREC'], DEFAULTS['PI_STREAM_LEN'], EMBEDDED_CORPUS,
        )
        kwargs = dict(
            path_or_fileobj=tmp.name,
            path_in_repo='pi_model.pkl.gz',
            repo_id=repo_id,
            repo_type='model',
        )
        if token and token.strip():
            kwargs['token'] = token.strip()
        upload_file(**kwargs)
        return f'Saved full model to {repo_id}/pi_model.pkl.gz'
    except Exception as e:
        return f'Save failed: {e}'


def _hf_latest_commit_sha(repo_id, token=None):
    """Return the latest commit sha for repo_id, or None on failure."""
    try:
        from huggingface_hub import list_repo_commits
        kwargs = dict(repo_id=repo_id, repo_type='model')
        if token and token.strip():
            kwargs['token'] = token.strip()
        commits = list(list_repo_commits(**kwargs))
        return commits[0].commit_id if commits else None
    except Exception:
        return None


def load_hf_model_on_demand(repo_id, token=None,
                            progress=gr.Progress(track_tqdm=False)):
    """
    Resolves the latest commit on the HF repo before every download.
    Uses force_download=True so huggingface_hub never serves a stale
    cached blob when the remote file has been updated.
    Skips the download only when the in-memory model already matches
    the latest remote commit sha.
    """
    try:
        from huggingface_hub import hf_hub_download

        progress(0.05, desc='Resolving latest HF commit…')
        latest_sha = _hf_latest_commit_sha(repo_id, token)
        already_loaded_sha = HF_CACHE.get('loaded_sha')
        if (
            HF_CACHE.get('loaded')
            and latest_sha is not None
            and latest_sha == already_loaded_sha
            and CACHE.get('cpd') is not None
        ):
            progress(1.0, desc='Already up-to-date.')
            return f'Model already up-to-date (sha={latest_sha[:7]})'

        progress(0.20, desc=f'Downloading pi_model.pkl.gz from {repo_id}…')
        kwargs = dict(
            repo_id=repo_id,
            filename='pi_model.pkl.gz',
            repo_type='model',
            cache_dir=LOCAL_CACHE_DIR,
            force_download=True,
        )
        if token and token.strip():
            kwargs['token'] = token.strip()
        path = hf_hub_download(**kwargs)

        progress(0.65, desc='Deserialising model from disk…')
        cpd, vocab, stream, config, errors = load_model_from_path(path)
        if cpd is None:
            progress(1.0, desc='Load failed.')
            return 'Load failed: ' + ' | '.join(errors)

        progress(0.90, desc='Updating in-memory cache…')
        CACHE.update(key=('HF', repo_id), cpd=cpd, vocab=vocab, stream=stream)
        corpus_tokens = tokenise_alpha(CACHE.get('_corpus_text', ''))
        if corpus_tokens:
            CACHE['context_index'] = build_context_index(vocab, cpd, corpus_tokens)
        UI_STATE['version'] += 1
        sha_tag = f' sha={latest_sha[:7]}' if latest_sha else ''
        HF_CACHE.update(
            loaded=True,
            loaded_sha=latest_sha,
            tokenizer=None,
            model=None,
            status=f'Loaded latest model from {repo_id}{sha_tag}',
        )
        progress(1.0, desc='Done.')
        return f'Loaded latest model from {repo_id}{sha_tag}.'
    except Exception as e:
        return f'Load failed: {e}'


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
    sampler = _make_sampler(
        stream, temperature,
        DEFAULTS['TOP_K'], DEFAULTS['TOP_P'], rep_penalty,
        DEFAULTS['SEASHELL_ENABLE'], DEFAULTS['SEASHELL_STRENGTH'],
        DEFAULTS['SEASHELL_DECAY'], DEFAULTS['SEASHELL_PEAKS'],
        DEFAULTS['SEASHELL_WIDTH'], DEFAULTS['SEASHELL_FLOOR'],
        insight_penalty,
        bool(semicircle_enable), semicircle_strength, semicircle_arches,
        semicircle_radius, semicircle_speed, semicircle_floor,
        instruction_context=ctx_words,
        context_index=CACHE.get('context_index'),
    )
    sampler.seek(0)
    text = generate_text_combo_cycles(
        cpd, sampler,
        prompt or '', int(text_length), int(ngram_n),
        vocab=vocab,
        context_index=CACHE.get('context_index'),
        n_cycles=DEFAULTS['CYCLES_N'],
    )
    oov = [w for w in tokenise_alpha(prompt or '') if w not in vocab]
    if oov:
        log.append(f'{len(oov)} prompt tokens not in corpus vocab: {oov}')
    if semicircle_enable:
        log.append(
            f'Semicircle wave mask ON '
            f'(strength={semicircle_strength}, arches={int(semicircle_arches)}, '
            f'radius={semicircle_radius}, speed={semicircle_speed}, '
            f'floor={semicircle_floor}).'
        )
    log.append(f'Generated {len(text.split())} tokens.')
    return text, '\n'.join(log)


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
                            label='Insight penalty — push away from conclusion-encoding labels',
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
                        label='Strength — blend amount of the mask (0 = pass-through)',
                    )
                    semicircle_arches = gr.Slider(
                        1, 30, value=DEFAULTS['SEMICIRCLE_ARCHES'], step=1,
                        label='Arches — number of semi-circles tiled across candidates',
                    )
                    semicircle_radius = gr.Slider(
                        0.05, 2.0, value=DEFAULTS['SEMICIRCLE_RADIUS'], step=0.05,
                        label='Radius — arch width (>=1 fills the slot, smaller = narrower)',
                    )
                    semicircle_speed = gr.Slider(
                        0.0, 1.0, value=DEFAULTS['SEMICIRCLE_SPEED'], step=0.005,
                        label='Speed — phase advance per generation step (wave travel)',
                    )
                    semicircle_floor = gr.Slider(
                        1e-3, 1.0, value=DEFAULTS['SEMICIRCLE_FLOOR'], step=0.005,
                        label='Floor — minimum gain so no candidate is fully zeroed',
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
            # Tab: Search
            # ----------------------------------------------------------------
            with gr.TabItem('Search'):
                gr.Markdown('Prompt-aligned search with collocations.')
                search_prompt = gr.Textbox(label='Prompt', lines=3, value='alice rabbit hole')
                search_bend_max = gr.Slider(1.0, 90.0, value=DEFAULTS['BEND_MAX'], step=0.5, label='Bend max')
                search_bend_step = gr.Slider(0.1, 5.0, value=DEFAULTS['BEND_STEP'], step=0.1, label='Bend step')
                search_offset_step = gr.Slider(1, 1000, value=DEFAULTS['OFFSET_STEP'], step=1, label='Offset step')
                search_fuzzy = gr.Slider(0.0, 1.0, value=DEFAULTS['FUZZY_THRESHOLD'], step=0.01, label='Fuzzy threshold')
                search_max = gr.Slider(1, 25, value=DEFAULTS['MAX_SOLUTIONS'], step=1, label='Max solutions')
                insight_penalty_search = gr.Slider(
                    0.0, 5.0,
                    value=DEFAULTS['INSIGHT_PENALTY'],
                    step=0.05,
                    label='Insight penalty — push away from conclusion-encoding labels',
                )

                with gr.Accordion('Semicircle wave mask', open=False):
                    search_semicircle_enable = gr.Checkbox(
                        value=DEFAULTS['SEMICIRCLE_ENABLE'],
                        label='Enable semicircle wave mask',
                    )
                    search_semicircle_strength = gr.Slider(
                        0.0, 1.0, value=DEFAULTS['SEMICIRCLE_STRENGTH'], step=0.01,
                        label='Strength',
                    )
                    search_semicircle_arches = gr.Slider(
                        1, 30, value=DEFAULTS['SEMICIRCLE_ARCHES'], step=1,
                        label='Arches',
                    )
                    search_semicircle_radius = gr.Slider(
                        0.05, 2.0, value=DEFAULTS['SEMICIRCLE_RADIUS'], step=0.05,
                        label='Radius',
                    )
                    search_semicircle_speed = gr.Slider(
                        0.0, 1.0, value=DEFAULTS['SEMICIRCLE_SPEED'], step=0.005,
                        label='Speed',
                    )
                    search_semicircle_floor = gr.Slider(
                        1e-3, 1.0, value=DEFAULTS['SEMICIRCLE_FLOOR'], step=0.005,
                        label='Floor',
                    )

                search_btn = gr.Button('Run search', variant='primary')
                search_out = gr.Textbox(label='Search result', lines=16)
                search_log = gr.Textbox(label='Search log', lines=8)
                search_file = gr.File(label='Search output')
                search_btn.click(
                    run_search,
                    inputs=[
                        filein, pasted, search_prompt,
                        pi_prec, pi_stream_len, ngram_n, lidstone_gamma,
                        text_length, temperature, rep_penalty,
                        gr.Checkbox(value=DEFAULTS['SEASHELL_ENABLE'], visible=False),
                        gr.Slider(visible=False, value=DEFAULTS['SEASHELL_STRENGTH']),
                        gr.Slider(visible=False, value=DEFAULTS['SEASHELL_DECAY']),
                        gr.Slider(visible=False, value=DEFAULTS['SEASHELL_PEAKS']),
                        gr.Slider(visible=False, value=DEFAULTS['SEASHELL_WIDTH']),
                        gr.Slider(visible=False, value=DEFAULTS['SEASHELL_FLOOR']),
                        gr.Radio(choices=['A', 'B', 'C'], value=DEFAULTS['VERTEX'], visible=False),
                        search_bend_max, search_bend_step, search_offset_step,
                        search_fuzzy, search_max, insight_penalty_search,
                        search_semicircle_enable, search_semicircle_strength,
                        search_semicircle_arches, search_semicircle_radius,
                        search_semicircle_speed, search_semicircle_floor,
                    ],
                    outputs=[search_out, search_log, search_file],
                )

            # ----------------------------------------------------------------
            # Tab: Model I/O
            # ----------------------------------------------------------------
            with gr.TabItem('Model I/O'):
                gr.Markdown('Save/load compiled trigram model.')

                # ── Load latest from HF (universal, always force-downloads) ──
                gr.Markdown('#### Load latest model from Hugging Face')
                model_hf_repo = gr.Textbox(label='HF repo ID', value=HF_REPO_ID)
                model_hf_token = gr.Textbox(label='HF token (optional)', type='password')
                load_latest_btn = gr.Button('🔄  Load Latest from HF', variant='primary')
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

            # ----------------------------------------------------------------
            # Tab: Thinking-lite
            # ----------------------------------------------------------------
            with gr.TabItem('Thinking-lite'):
                gr.Markdown('Use the buttons below to load or save.')
                hfstatus = gr.Textbox(label='Status', value='Idle', lines=2, interactive=False)
                hf_repo = gr.Textbox(label='Hugging Face repo', value=HF_REPO_ID)
                hf_token = gr.Textbox(label='HF token', type='password')
                with gr.Row():
                    hf_open_btn = gr.Button('Open Thinking-lite', variant='secondary')
                    hf_save_btn = gr.Button('Save to Hugging Face', variant='primary')
                    hf_load_btn = gr.Button('Load from Hugging Face', variant='secondary')
                hf_log = gr.Textbox(label='Log', lines=4, interactive=False)
                hf_open_btn.click(lambda: 'Thinking-lite ready', inputs=None, outputs=[hfstatus])
                hf_save_btn.click(save_hf_model, inputs=[hf_repo, hf_token], outputs=[hf_log])
                def _hf_load_and_status(repo, tok):
                    msg = load_hf_model_on_demand(repo, tok)
                    status = HF_CACHE.get('status', 'Unknown')
                    return msg, f"### Model status\n\n`{status}`"

                hf_load_btn.click(_hf_load_and_status, inputs=[hf_repo, hf_token], outputs=[hf_log, status_md])

        gr.Markdown('Tip: model caches are reused until corpus or configuration changes.')

        # auto-load on startup removed — use "Load Latest from HF" button

    return demo


if __name__ == '__main__':
    build_ui().queue(max_size=8).launch(
        server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'),
        show_error=True,
    )
