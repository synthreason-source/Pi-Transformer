#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
π → BASE-26 → NLTK TRIGRAM GENERATOR
════════════════════════════════════

GRADIO VERSION (Hugging Face Spaces ready)

Features
--------
- File upload OR pasted corpus OR embedded fallback
- All knobs exposed: precision, stream length, n-gram order,
  Lidstone gamma, generation length, sampling temperature,
  top-k / top-p, repetition penalty, seashell resonator,
  triangle bend / offset / vertex, fuzzy threshold.
- Two modes:
    1. Single generate (one bend/offset/vertex)
    2. Brute-force prompt-aligned search
- Live status log
- Downloadable dataset export
"""

import os
import sys
import io
import re
import math
import json
import gzip
import pickle
import hashlib
import tempfile
from collections import defaultdict, deque, Counter
from difflib import SequenceMatcher

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

# Make NLTK download silently into a writable dir (HF Spaces friendly).
NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

def _ensure_nltk():
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

_ensure_nltk()

from nltk.corpus import words as nltk_words  # noqa: E402


# ============================================================
# CONFIG DEFAULTS
# ============================================================

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300_000)

DEFAULTS = dict(
    PI_PREC=15000,
    PI_STREAM_LEN=12000,
    DIGITS_PER_SAMPLE=3,
    NGRAM_N=2,
    LIDSTONE_GAMMA=0.1,
    GEN_WORDS=400,
    WORD_FIND_MIN=2,
    TEMPERATURE=2.5,
    TOP_K=100,
    TOP_P=1.0,
    REP_PENALTY=1.08,
    SEASHELL_ENABLE=True,
    SEASHELL_STRENGTH=4.35,
    SEASHELL_DECAY=0.985,
    SEASHELL_PEAKS=4,
    SEASHELL_WIDTH=0.16,
    SEASHELL_FLOOR=0.35,
    BEND_DEGREES=13.0,
    OFFSET=0,
    VERTEX="A",
    FUZZY_THRESHOLD=0.72,
    MAX_SOLUTIONS=5,
    BEND_STEP=0.5,
    OFFSET_STEP=50,
    BEND_MAX=45.0,
)


# ============================================================
# EMBEDDED CORPUS
# ============================================================

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


# ============================================================
# TOKEN HELPERS
# ============================================================

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenise_alpha(text):
    """Lowercase, strip punctuation, keep only alphabetic word tokens.
    Used for BOTH corpus and prompt so they share a vocabulary."""
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
    if not words:
        return ""
    return " ".join(words)


# ============================================================
# MODEL
# ============================================================

class _LidstoneFactory:
    """Picklable replacement for `lambda fd: LidstoneProbDist(fd, gamma, bins)`.
    Used by ConditionalProbDist so the whole CPD can be saved with pickle."""
    __slots__ = ("gamma", "bins")

    def __init__(self, gamma, bins):
        self.gamma = float(gamma)
        self.bins = max(1, int(bins))

    def __call__(self, fd):
        return LidstoneProbDist(fd, gamma=self.gamma, bins=self.bins)


def build_model(corpus, ngram_n, lidstone_gamma):
    # Defensive coercion — Gradio sliders can return floats; corpus could in
    # theory arrive as bytes or as something with __str__ if upstream caching
    # mishandles it. Make this function robust to all of those.
    if isinstance(corpus, bytes):
        corpus = corpus.decode("utf-8", errors="ignore")
    elif not isinstance(corpus, str):
        corpus = str(corpus) if corpus is not None else ""

    ngram_n = int(ngram_n)
    if ngram_n < 2:
        ngram_n = 2
    lidstone_gamma = float(lidstone_gamma)

    tokens = tokenise_alpha(corpus)
    if not isinstance(tokens, list):
        tokens = list(tokens)

    if not tokens:
        raise ValueError(
            "Corpus produced zero tokens after tokenisation. "
            "Upload a non-empty text corpus or paste some text."
        )

    padded = ["<s>"] * (ngram_n - 1) + tokens + ["</s>"]

    trigrams_ = list(ngrams(padded, ngram_n))

    cfd = ConditionalFreqDist(
        (tuple(tg[:-1]), tg[-1]) for tg in trigrams_
    )

    vocab = set(tokens) | {"</s>"}

    for ctx in list(cfd.conditions()):
        if len(cfd[ctx]) == 0:
            cfd[ctx]["</s>"] += 1

    cpd = ConditionalProbDist(
        cfd,
        _LidstoneFactory(gamma=lidstone_gamma, bins=max(1, len(vocab))),
    )

    return cpd, vocab


# ============================================================
# WORD LIST
# ============================================================

def load_dictionary(vocab):
    try:
        words = {w.lower() for w in nltk_words.words() if w.isalpha()}
        return words | set(vocab)
    except Exception:
        return set(vocab)


# ============================================================
# PI STREAM
# ============================================================

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


# ============================================================
# SEASHELL RESONATOR
# ============================================================

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


# ============================================================
# PI SAMPLER
# ============================================================

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
    ):
        self.stream = stream
        self.digits_per_sample = digits_per_sample
        self.pos = 0
        self.temperature = max(1e-3, float(temperature))
        self.top_k = max(1, int(top_k))
        self.top_p = max(1e-3, min(1.0, float(top_p)))
        self.repetition_penalty = max(1.0, float(repetition_penalty))
        self.history = Counter()
        self.seashell = None

        if seashell_enable:
            self.seashell = SeashellResonator(
                sampler=self,
                strength=seashell_strength,
                decay=seashell_decay,
                peaks=seashell_peaks,
                width=seashell_width,
                floor=seashell_floor,
            )

    def seek(self, pos):
        self.pos = pos % len(self.stream)
        self.history.clear()
        if self.seashell is not None:
            self.seashell.reset()

    def next_unit(self):
        val = 0
        base = 26 ** self.digits_per_sample
        for _ in range(self.digits_per_sample):
            val = val * 26 + self.stream[self.pos % len(self.stream)]
            self.pos += 1
        return val / base

    def _xor_probability_fusion(self, scored, u_a, u_b, u_c):
        xor_scores = []
        for rank, (word, base_p) in enumerate(scored):
            idx = rank / max(1, len(scored) - 1)
            region_a = (1.0 - abs(idx - u_a)) * (1.0 - u_b) * (1.0 - u_c)
            region_b = u_b * (1.0 - abs(idx - u_a)) * (1.0 - u_c)
            region_c = u_c * (1.0 - u_a) * (1.0 - u_b)
            xor_blend = max(region_a, region_b, region_c)
            orthogonality = 1.0 - abs(u_a - u_b) * abs(u_b - u_c)
            final_p = base_p * xor_blend * (1.0 + 0.8 * orthogonality)
            xor_scores.append((word, final_p))
        return xor_scores

    def sample(self, dist):
        samples = list(dist.samples())
        if not samples:
            return "</s>"

        base_scored = []
        for s in samples:
            p = max(1e-12, float(dist.prob(s)))
            count = self.history[s]
            if count > 0:
                p /= self.repetition_penalty ** count
            base_scored.append((s, p))

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

        u_a = self.next_unit()
        u_b = self.next_unit()
        u_c = self.next_unit()

        xor_scored = self._xor_probability_fusion(scored, u_a, u_b, u_c)
        xor_total = sum(p for _, p in xor_scored)

        if xor_total <= 0:
            chosen = scored[-1][0] if scored else "</s>"
        else:
            xor_scored = [(w, p / xor_total) for w, p in xor_scored]
            xor_draw = (
                u_a * (1 - u_b) * (1 - u_c)
                + u_b * (1 - u_a) * (1 - u_c)
                + u_c * (1 - u_a) * (1 - u_b)
            ) / 1.5

            cumulative = 0.0
            chosen = xor_scored[-1][0]
            for word, p in xor_scored:
                cumulative += p
                if xor_draw < cumulative:
                    chosen = word
                    break

        self.history[chosen] += 1
        return chosen


# ============================================================
# TRIANGLE
# ============================================================

class Triangle:
    def __init__(self, stream_len, offset_extra=0, bend_degrees=13.0):
        base = offset_extra % stream_len
        bend_shift = int(round((bend_degrees / 360.0) * stream_len))

        self.A = base % stream_len
        self.B = (base + stream_len // 3 + bend_shift) % stream_len
        self.C = (base + 2 * stream_len // 3 + bend_shift) % stream_len

        self.vertices = {"A": self.A, "B": self.B, "C": self.C}


# ============================================================
# GENERATION
# ============================================================

def generate_text(cpd, sampler, prompt, n_words, ngram_n, vocab=None):
    """Generate n_words tokens, seeded by `prompt`.

    Prompt handling:
      * Tokenised the same way as the corpus.
      * Tokens are echoed in the output verbatim (so the user sees their prompt
        at the start), even if they're out-of-vocab.
      * Only in-vocab tokens are used to seed the trigram context. Out-of-vocab
        tokens are skipped from the context so the model still has a chance to
        continue.
      * When the full (n-1)-gram context has no continuations, we back off to
        shorter suffixes, then finally to '<s>' padding.
    """
    context_window = ngram_n - 1
    seed_words = tokenise_alpha(prompt)

    # In-vocab subset for trigram seeding.
    if vocab is not None:
        seed_in_vocab = [w for w in seed_words if w in vocab]
    else:
        seed_in_vocab = list(seed_words)

    if len(seed_in_vocab) >= context_window:
        init = seed_in_vocab[-context_window:]
    else:
        init = ["<s>"] * (context_window - len(seed_in_vocab)) + seed_in_vocab

    context = deque(init, maxlen=context_window)
    words = list(seed_words)  # echo the user's prompt verbatim in output

    def _dist_for(ctx_tuple):
        """Return a usable dist for `ctx_tuple`, backing off to shorter
        suffixes (padded with <s>) if the full context has no continuations."""
        # try full context, then progressively shorter suffixes
        for cut in range(len(ctx_tuple) + 1):
            trial = ("<s>",) * cut + ctx_tuple[cut:]
            try:
                d = cpd[trial]
                if list(d.samples()):
                    return d
            except Exception:
                continue
        # final fallback: all-<s> start-of-sentence
        try:
            d = cpd[("<s>",) * context_window]
            if list(d.samples()):
                return d
        except Exception:
            pass
        return None

    for _ in range(n_words):
        ctx = tuple(context)
        dist = _dist_for(ctx)

        if dist is None:
            # nothing in the model can continue — reset and try again next step
            context.clear()
            context.extend(["<s>"] * context_window)
            continue

        word = sampler.sample(dist)

        if word in ("</s>", ""):
            context.clear()
            context.extend(["<s>"] * context_window)
            continue

        words.append(word)
        context.append(word)

    return capitalise_text(words)


# ============================================================
# FIND WORDS
# ============================================================

def find_words(stream, dictionary, word_find_min):
    prefixes = set()
    for w in dictionary:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])

    buf = deque(maxlen=35)
    all_chars = []
    found = defaultdict(list)

    for pos, digit in enumerate(stream):
        ch = chr(ord("a") + digit)
        buf.append(ch)
        all_chars.append(ch)
        s = "".join(buf)
        for length in range(word_find_min, min(16, len(s)) + 1):
            cand = s[-length:]
            if cand not in prefixes:
                continue
            if cand in dictionary:
                found[cand].append(pos - length + 1)

    return "".join(all_chars), found


# ============================================================
# FUZZY / PAIR MATCHING
# ============================================================

def fuzzy_score(target, text):
    return SequenceMatcher(None, target.lower(), text.lower()).quick_ratio()


def all_pairs_match(pairs, text, fuzzy_threshold):
    lower_text = text.lower()
    for pair in pairs:
        pair_str = " ".join(pair)
        if pair_str in lower_text:
            continue
        if fuzzy_score(pair_str, text) < fuzzy_threshold:
            return False, pair
    return True, None


# ============================================================
# GRADIO STATE / PIPELINE
# ============================================================

# Cache for compiled corpus + pi stream so we don't rebuild every call.
_CACHE = {"key": None, "cpd": None, "vocab": None, "stream": None}


def _cache_key(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len):
    return (
        hash(corpus),
        int(ngram_n),
        float(lidstone_gamma),
        int(pi_prec),
        int(pi_stream_len),
    )


def get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log):
    key = _cache_key(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len)

    cached_key = _CACHE["key"]
    if cached_key == key and _CACHE["cpd"] is not None:
        log.append("✓ Using cached model + π stream.")
        return _CACHE["cpd"], _CACHE["vocab"], _CACHE["stream"]

    # Special-case: a loaded model is in the cache. Reuse it if the requested
    # config (ngram_n, lidstone_gamma, pi_prec, pi_stream_len) matches what
    # was loaded. We ignore the corpus content in this check — the user
    # already supplied the model directly. Note: the actual stream length
    # used for triangle math comes from the cached stream itself.
    if (
        isinstance(cached_key, tuple)
        and len(cached_key) >= 6
        and cached_key[0] == "LOADED"
        and _CACHE["cpd"] is not None
        and cached_key[2] == int(ngram_n)
        and abs(cached_key[3] - float(lidstone_gamma)) < 1e-9
        and cached_key[4] == int(pi_prec)
        and cached_key[5] == int(pi_stream_len)
    ):
        log.append("✓ Using loaded model from file (no rebuild).")
        return _CACHE["cpd"], _CACHE["vocab"], _CACHE["stream"]

    log.append("Building trigram model...")
    cpd, vocab = build_model(corpus, ngram_n, lidstone_gamma)

    log.append(f"Building π stream ({pi_prec} dps → {pi_stream_len} symbols)...")
    stream = build_pi_stream(pi_prec, pi_stream_len)

    _CACHE.update(key=key, cpd=cpd, vocab=vocab, stream=stream)
    log.append("✓ Cached.")
    return cpd, vocab, stream


def resolve_corpus(file_obj, pasted_corpus):
    """Priority: uploaded file > pasted text > embedded fallback."""
    if file_obj is not None:
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if txt.strip():
                return txt, f"file: {os.path.basename(path)}"
        except Exception as e:
            return EMBEDDED_CORPUS, f"file read failed ({e}); using embedded"

    if pasted_corpus and pasted_corpus.strip():
        return pasted_corpus, "pasted text"

    return EMBEDDED_CORPUS, "embedded fallback"


# ============================================================
# SAVE / LOAD MODEL
# ============================================================
#
# Format (gzipped pickle):
#   {
#     "magic":   "PI_TRIGRAM_MODEL_V1",
#     "version": 1,
#     "cpd":     ConditionalProbDist,
#     "vocab":   set[str],
#     "stream":  list[int],   # base-26 π digits
#     "config":  {
#         "ngram_n": int,
#         "lidstone_gamma": float,
#         "pi_prec": int,
#         "pi_stream_len": int,
#         "digits_per_sample": int,
#         "corpus_sha256": str,
#         "corpus_chars":  int,
#         "vocab_size":    int,
#     },
#   }
#
# Pickle is used because nltk.ConditionalProbDist isn't naturally JSON-serializable.
# Only load files from sources you trust — pickle can execute code on load.

MODEL_MAGIC = "PI_TRIGRAM_MODEL_V1"
MODEL_VERSION = 1


def _corpus_fingerprint(corpus):
    if isinstance(corpus, str):
        b = corpus.encode("utf-8", errors="ignore")
    elif isinstance(corpus, bytes):
        b = corpus
    else:
        b = str(corpus).encode("utf-8", errors="ignore")
    return hashlib.sha256(b).hexdigest()


def save_model_to_path(
    path,
    cpd, vocab, stream,
    ngram_n, lidstone_gamma, pi_prec, pi_stream_len,
    corpus_text,
):
    payload = {
        "magic": MODEL_MAGIC,
        "version": MODEL_VERSION,
        "cpd": cpd,
        "vocab": set(vocab),
        "stream": list(stream),
        "config": {
            "ngram_n": int(ngram_n),
            "lidstone_gamma": float(lidstone_gamma),
            "pi_prec": int(pi_prec),
            "pi_stream_len": int(pi_stream_len),
            "digits_per_sample": int(DEFAULTS["DIGITS_PER_SAMPLE"]),
            "corpus_sha256": _corpus_fingerprint(corpus_text),
            "corpus_chars": len(corpus_text) if corpus_text else 0,
            "vocab_size": len(vocab),
        },
    }
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model_from_path(path):
    """Return (cpd, vocab, stream, config, errors_list).
    On failure, cpd is None and errors_list is populated."""
    errors = []
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        # Try uncompressed pickle as a fallback.
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e2:
            return None, None, None, None, [
                f"Could not read file: {e}",
                f"Uncompressed fallback also failed: {e2}",
            ]

    if not isinstance(payload, dict):
        return None, None, None, None, [
            "File does not contain a model dict."
        ]
    if payload.get("magic") != MODEL_MAGIC:
        errors.append(
            f"Magic header mismatch (got {payload.get('magic')!r}, "
            f"expected {MODEL_MAGIC!r}). Proceeding cautiously."
        )
    if payload.get("version", 0) > MODEL_VERSION:
        errors.append(
            f"File version {payload.get('version')} is newer than "
            f"supported version {MODEL_VERSION}."
        )

    cpd = payload.get("cpd")
    vocab = payload.get("vocab")
    stream = payload.get("stream")
    config = payload.get("config", {})

    if cpd is None or vocab is None or stream is None:
        return None, None, None, None, errors + [
            "Missing required fields (cpd/vocab/stream)."
        ]

    return cpd, set(vocab), list(stream), dict(config), errors


def action_save_model(
    file_obj, pasted_corpus,
    pi_prec, pi_stream_len, ngram_n, lidstone_gamma,
):
    """Build (or reuse cached) model with current corpus+config and write it
    to a gzipped pickle that the user can download."""
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f"Corpus source: {source} ({len(corpus)} chars).")

    cpd, vocab, stream = get_or_build(
        corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log
    )

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pkl.gz", prefix="pi_trigram_model_"
    )
    tmp.close()
    save_model_to_path(
        tmp.name,
        cpd, vocab, stream,
        ngram_n, lidstone_gamma, pi_prec, pi_stream_len,
        corpus,
    )

    size_kb = os.path.getsize(tmp.name) / 1024.0
    log.append(
        f"✓ Saved model to {os.path.basename(tmp.name)} ({size_kb:.1f} KB)."
    )
    log.append(
        f"  config: ngram_n={int(ngram_n)} γ={float(lidstone_gamma)} "
        f"π_prec={int(pi_prec)} stream_len={int(pi_stream_len)} "
        f"vocab={len(vocab)}"
    )
    return tmp.name, "\n".join(log)


def action_load_model(model_file):
    """Load a saved model into the in-process cache and return updated
    slider/log values so the UI reflects the loaded config."""
    if model_file is None:
        return (
            "No file uploaded. Pick a .pkl.gz model file first.",
            # leave slider values untouched
            gr.update(), gr.update(), gr.update(), gr.update(),
        )

    path = model_file.name if hasattr(model_file, "name") else model_file
    cpd, vocab, stream, config, errors = load_model_from_path(path)

    if cpd is None:
        return (
            "Failed to load model:\n" + "\n".join(errors),
            gr.update(), gr.update(), gr.update(), gr.update(),
        )

    ngram_n = int(config.get("ngram_n", DEFAULTS["NGRAM_N"]))
    lidstone_gamma = float(config.get("lidstone_gamma", DEFAULTS["LIDSTONE_GAMMA"]))
    pi_prec = int(config.get("pi_prec", DEFAULTS["PI_PREC"]))
    pi_stream_len = int(config.get("pi_stream_len", DEFAULTS["PI_STREAM_LEN"]))

    # Install into the cache so generation uses it directly without rebuilding.
    # Use a sentinel key that no recomputation will match, so any later
    # corpus/config tweak by the user will rebuild cleanly.
    _CACHE.update(
        key=("LOADED", path, ngram_n, lidstone_gamma, pi_prec, pi_stream_len),
        cpd=cpd, vocab=vocab, stream=stream,
    )

    lines = [
        f"✓ Loaded model from {os.path.basename(path)}.",
        f"  ngram_n={ngram_n}  γ={lidstone_gamma}",
        f"  π_prec={pi_prec}  stream_len={pi_stream_len}",
        f"  vocab={len(vocab)}  stream_len_actual={len(stream)}",
    ]
    if config.get("corpus_sha256"):
        lines.append(
            f"  corpus_sha256={config['corpus_sha256'][:16]}…  "
            f"corpus_chars={config.get('corpus_chars', '?')}"
        )
    if errors:
        lines.append("Warnings:")
        for e in errors:
            lines.append(f"  ! {e}")
    lines.append(
        "Slider values have been updated to match. You can now Generate/Search "
        "without re-uploading the corpus."
    )

    return (
        "\n".join(lines),
        gr.update(value=pi_prec),
        gr.update(value=pi_stream_len),
        gr.update(value=ngram_n),
        gr.update(value=lidstone_gamma),
    )


def run_single(
    file_obj,
    pasted_corpus,
    prompt,
    pi_prec,
    pi_stream_len,
    ngram_n,
    lidstone_gamma,
    gen_words,
    temperature,
    top_k,
    top_p,
    rep_penalty,
    seashell_enable,
    seashell_strength,
    seashell_decay,
    seashell_peaks,
    seashell_width,
    seashell_floor,
    bend_degrees,
    offset,
    vertex,
):
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f"Corpus source: {source} ({len(corpus)} chars).")

    cpd, vocab, stream = get_or_build(
        corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log
    )

    triangle = Triangle(
        int(pi_stream_len),
        offset_extra=int(offset),
        bend_degrees=float(bend_degrees),
    )
    start = triangle.vertices[vertex]
    log.append(
        f"Triangle: A={triangle.A} B={triangle.B} C={triangle.C} "
        f"→ vertex {vertex} = {start}"
    )

    sampler = PiSampler(
        stream,
        digits_per_sample=DEFAULTS["DIGITS_PER_SAMPLE"],
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
    )
    sampler.seek(start)

    log.append(f"Generating {int(gen_words)} words...")
    text = generate_text(
        cpd, sampler, prompt=prompt or "", n_words=int(gen_words),
        ngram_n=int(ngram_n), vocab=vocab,
    )

    seed_tokens = tokenise_alpha(prompt or "")
    oov = [w for w in seed_tokens if w not in vocab]
    if oov:
        log.append(f"⚠ {len(oov)} prompt token(s) not in corpus vocab: {oov}")
    log.append(f"✓ Done. {len(text.split())} tokens output.")

    # Write a downloadable copy.
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", prefix="pi_gen_", mode="w", encoding="utf-8"
    )
    tmp.write(text)
    tmp.close()

    return text, "\n".join(log), tmp.name


def run_search(
    file_obj,
    pasted_corpus,
    prompt,
    pi_prec,
    pi_stream_len,
    ngram_n,
    lidstone_gamma,
    gen_words,
    temperature,
    top_k,
    top_p,
    rep_penalty,
    seashell_enable,
    seashell_strength,
    seashell_decay,
    seashell_peaks,
    seashell_width,
    seashell_floor,
    vertex,
    bend_max,
    bend_step,
    offset_step,
    fuzzy_threshold,
    max_solutions,
    progress=gr.Progress(track_tqdm=False),
):
    log = []
    if not prompt or not prompt.strip():
        return "", "Prompt is empty — search needs word pairs.", None

    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f"Corpus source: {source} ({len(corpus)} chars).")

    cpd, vocab, stream = get_or_build(
        corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log
    )

    pairs = extract_word_pairs(prompt)
    if not pairs:
        return "", "No valid word pairs extracted from prompt.", None
    log.append(f"Prompt yields {len(pairs)} word pairs.")

    # Enumerate (bend, offset) grid.
    bend_values = []
    b = 0.0
    while b <= float(bend_max) + 1e-9:
        bend_values.append(round(b, 4))
        b += float(bend_step)

    offset_values = list(range(0, int(pi_stream_len), max(1, int(offset_step))))
    total = len(bend_values) * len(offset_values)
    log.append(
        f"Search grid: {len(bend_values)} bends × "
        f"{len(offset_values)} offsets = {total} candidates."
    )

    found = []
    counter = 0
    for bend in bend_values:
        for offset in offset_values:
            counter += 1
            progress(counter / max(1, total), desc=f"bend={bend} offset={offset}")

            triangle = Triangle(
                int(pi_stream_len),
                offset_extra=offset,
                bend_degrees=bend,
            )
            start = triangle.vertices[vertex]

            sampler = PiSampler(
                stream,
                digits_per_sample=DEFAULTS["DIGITS_PER_SAMPLE"],
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
            )
            sampler.seek(start)

            text = generate_text(
                cpd, sampler,
                prompt=prompt, n_words=int(gen_words),
                ngram_n=int(ngram_n), vocab=vocab,
            )

            matches, _failed = all_pairs_match(
                pairs, text, fuzzy_threshold=float(fuzzy_threshold)
            )

            if matches:
                found.append({
                    "prompt": prompt,
                    "bend": bend,
                    "offset": offset,
                    "vertex": vertex,
                    "text": text,
                })
                log.append(f"✓ MATCH bend={bend} offset={offset}")
                if len(found) >= int(max_solutions):
                    break
        if len(found) >= int(max_solutions):
            break

    if not found:
        log.append("No matches found in the searched grid.")
        return "", "\n".join(log), None

    # Render results
    parts = []
    for i, r in enumerate(found, 1):
        parts.append(
            f"=== MATCH {i} ===\n"
            f"bend = {r['bend']}   offset = {r['offset']}   vertex = {r['vertex']}\n\n"
            f"{r['text']}\n"
        )
    rendered = "\n".join(parts)

    # JSONL export
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".jsonl", prefix="pi_search_",
        mode="w", encoding="utf-8"
    )
    for r in found:
        tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.close()

    return rendered, "\n".join(log), tmp.name


# ============================================================
# UI
# ============================================================

DESCRIPTION = """
# π → base-26 → NLTK trigram generator

A deterministic text generator seeded by the base-26 expansion of π, sampled via
a bent-triangle vertex map, scored through an NLTK trigram language model with
optional **seashell cavity resonance** coloration and **XOR probability fusion**.

- **Single Generate** — pick one bend/offset/vertex and produce text.
- **Prompt-Aligned Search** — brute-force the (bend × offset) grid until the
  generated text contains all word pairs from your prompt (exact or fuzzy).

Upload your own corpus, paste one in, or use the embedded *Alice* fragment.
"""


def build_ui():
    with gr.Blocks(title="π → base-26 → trigram", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Corpus")
                file_in = gr.File(
                    label="Upload corpus (.txt)",
                    file_types=[".txt", ".md"],
                    type="filepath",
                )
                pasted = gr.Textbox(
                    label="…or paste corpus here",
                    lines=6,
                    placeholder="(optional) paste corpus text — used if no file uploaded",
                )

                gr.Markdown("### Prompt")
                prompt_in = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="alice rabbit hole",
                    placeholder="Words that must appear as ordered pairs in output",
                )

            with gr.Column(scale=1):
                gr.Markdown("### π stream & model")
                with gr.Row():
                    pi_prec = gr.Slider(500, 30000, value=DEFAULTS["PI_PREC"],
                                        step=500, label="π precision (dps)")
                    pi_stream_len = gr.Slider(500, 30000, value=DEFAULTS["PI_STREAM_LEN"],
                                              step=500, label="π stream length")
                with gr.Row():
                    ngram_n = gr.Slider(2, 6, value=DEFAULTS["NGRAM_N"], step=1,
                                        label="n-gram order (n)")
                    lidstone_gamma = gr.Slider(0.001, 1.0,
                                               value=DEFAULTS["LIDSTONE_GAMMA"],
                                               step=0.001, label="Lidstone γ")
                gen_words = gr.Slider(50, 5000, value=DEFAULTS["GEN_WORDS"],
                                      step=50, label="Generated words")

                gr.Markdown("### Sampling")
                with gr.Row():
                    temperature = gr.Slider(0.1, 5.0, value=DEFAULTS["TEMPERATURE"],
                                            step=0.05, label="Temperature")
                    rep_penalty = gr.Slider(1.0, 2.0, value=DEFAULTS["REP_PENALTY"],
                                            step=0.01, label="Repetition penalty")
                with gr.Row():
                    top_k = gr.Slider(1, 500, value=DEFAULTS["TOP_K"],
                                      step=1, label="top-k")
                    top_p = gr.Slider(0.05, 1.0, value=DEFAULTS["TOP_P"],
                                      step=0.01, label="top-p")

        with gr.Accordion("Seashell resonator", open=False):
            seashell_enable = gr.Checkbox(value=DEFAULTS["SEASHELL_ENABLE"],
                                          label="Enable seashell coloration")
            with gr.Row():
                seashell_strength = gr.Slider(0.0, 10.0,
                                              value=DEFAULTS["SEASHELL_STRENGTH"],
                                              step=0.05, label="Strength")
                seashell_decay = gr.Slider(0.5, 0.9999,
                                           value=DEFAULTS["SEASHELL_DECAY"],
                                           step=0.0005, label="Decay")
            with gr.Row():
                seashell_peaks = gr.Slider(1, 12, value=DEFAULTS["SEASHELL_PEAKS"],
                                           step=1, label="Peaks")
                seashell_width = gr.Slider(0.02, 0.6, value=DEFAULTS["SEASHELL_WIDTH"],
                                           step=0.01, label="Width")
                seashell_floor = gr.Slider(0.0, 1.0, value=DEFAULTS["SEASHELL_FLOOR"],
                                           step=0.01, label="Floor")

        with gr.Accordion("Triangle (single-generate)", open=True):
            with gr.Row():
                bend_degrees = gr.Slider(0.0, 90.0,
                                         value=DEFAULTS["BEND_DEGREES"],
                                         step=0.1, label="Bend (degrees)")
                offset = gr.Slider(0, 30000, value=DEFAULTS["OFFSET"],
                                   step=1, label="Offset")
                vertex = gr.Radio(["A", "B", "C"], value=DEFAULTS["VERTEX"],
                                  label="Vertex")

        with gr.Accordion("Search grid (prompt-aligned)", open=False):
            with gr.Row():
                bend_max = gr.Slider(1.0, 90.0, value=DEFAULTS["BEND_MAX"],
                                     step=0.5, label="Bend max (°)")
                bend_step = gr.Slider(0.1, 5.0, value=DEFAULTS["BEND_STEP"],
                                      step=0.1, label="Bend step (°)")
                offset_step = gr.Slider(1, 1000, value=DEFAULTS["OFFSET_STEP"],
                                        step=1, label="Offset step")
            with gr.Row():
                fuzzy_threshold = gr.Slider(0.0, 1.0,
                                            value=DEFAULTS["FUZZY_THRESHOLD"],
                                            step=0.01, label="Fuzzy threshold")
                max_solutions = gr.Slider(1, 25, value=DEFAULTS["MAX_SOLUTIONS"],
                                          step=1, label="Max solutions")

        with gr.Accordion("💾 Save / Load model", open=False):
            gr.Markdown(
                "Save the compiled trigram model + π stream + config as a "
                "single `.pkl.gz` file. Loading restores it into the cache so "
                "Generate/Search runs without rebuilding from the corpus.\n\n"
                "⚠️ **Only load model files from sources you trust** — the "
                "format is gzipped pickle, which can execute code on load."
            )
            with gr.Row():
                btn_save_model = gr.Button("💾 Save model (download)",
                                           variant="secondary")
                save_model_file = gr.File(label="Saved model file",
                                          interactive=False)
            with gr.Row():
                load_model_file = gr.File(
                    label="Upload saved model (.pkl.gz)",
                    file_types=[".gz", ".pkl"],
                    type="filepath",
                )
                btn_load_model = gr.Button("📂 Load model", variant="secondary")
            model_io_log = gr.Textbox(label="Model I/O log", lines=6)

        with gr.Row():
            btn_single = gr.Button("▶ Single Generate", variant="primary")
            btn_search = gr.Button("🔍 Prompt-Aligned Search", variant="secondary")

        with gr.Row():
            out_text = gr.Textbox(label="Generated text",
                                  lines=18)
        with gr.Row():
            out_log = gr.Textbox(label="Log", lines=8)
            out_file = gr.File(label="Download output")

        # Wire up save/load.
        btn_save_model.click(
            action_save_model,
            inputs=[file_in, pasted, pi_prec, pi_stream_len, ngram_n, lidstone_gamma],
            outputs=[save_model_file, model_io_log],
        )
        btn_load_model.click(
            action_load_model,
            inputs=[load_model_file],
            outputs=[model_io_log, pi_prec, pi_stream_len, ngram_n, lidstone_gamma],
        )

        # Wire up generation.
        single_inputs = [
            file_in, pasted, prompt_in,
            pi_prec, pi_stream_len, ngram_n, lidstone_gamma, gen_words,
            temperature, top_k, top_p, rep_penalty,
            seashell_enable, seashell_strength, seashell_decay,
            seashell_peaks, seashell_width, seashell_floor,
            bend_degrees, offset, vertex,
        ]
        btn_single.click(
            run_single,
            inputs=single_inputs,
            outputs=[out_text, out_log, out_file],
        )

        search_inputs = [
            file_in, pasted, prompt_in,
            pi_prec, pi_stream_len, ngram_n, lidstone_gamma, gen_words,
            temperature, top_k, top_p, rep_penalty,
            seashell_enable, seashell_strength, seashell_decay,
            seashell_peaks, seashell_width, seashell_floor,
            vertex,
            bend_max, bend_step, offset_step,
            fuzzy_threshold, max_solutions,
        ]
        btn_search.click(
            run_search,
            inputs=search_inputs,
            outputs=[out_text, out_log, out_file],
        )

        gr.Markdown(
            "**Tip:** the model+stream are cached, so re-running with the same "
            "corpus / π settings is fast. Changing precision, stream length, "
            "n-gram order, or Lidstone γ triggers a rebuild. Use **Save / Load "
            "model** above to skip the rebuild entirely on later sessions."
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        show_error=True,
    )
