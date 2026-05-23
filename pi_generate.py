# -*- coding: utf-8 -*-
# ===========================================================================
#  app.py  —  Pi-Trigram Generator  +  Layer Tensor tab
#  Self-contained single file. layer_isomorphism classes are embedded below.
# ===========================================================================

from __future__ import annotations

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
from typing import Dict, List, Optional, Tuple

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
    SPAN_WORDS=8,
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

HF_REPO_ID      = "trainman999/Thinking-lite"
LOCAL_CACHE_DIR  = os.path.join(NLTK_DATA_DIR, "hf_model_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
HF_CACHE = {"loaded": False, "tokenizer": None, "model": None, "status": "Idle"}
CACHE    = dict(key=None, cpd=None, vocab=None, stream=None, context_index=None)
UI_STATE = {"version": 0}



def tokenise_alpha(text):
    if text is None:
        return []
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    elif not isinstance(text, str):
        text = str(text)
    return text.lower().split()



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
# ContextZoneIndex
# ---------------------------------------------------------------------------

class ContextZoneIndex:
    FREQ_HIGH_THRESH = 10
    FREQ_MID_THRESH  = 3

    _ZONE_SPEC_TEMPLATES = [
        ("ngram_bigram",  0.25, 0.04),
        ("ngram_unigram", 0.30, 0.04),
        ("alpha",         0.40, 0.05),
        ("freq",          0.50, 0.05),
        ("trigram_char",  0.35, 0.04),
        ("latent_bos",    0.30, 0.04),
    ]

    def __init__(self, vocab, cpd, token_freq):
        self.vocab      = set(vocab)
        self.token_freq = dict(token_freq)
        self._build_freq_zones()
        self._build_alpha_zones()
        self._build_ngram_zones(cpd)
        self._build_trigram_latent_index(cpd)
        self._make_zone_specs()
        self.prompt_zone = {}

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

    def _build_trigram_latent_index(self, cpd):
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
        print(f"{'Rank':<5} {'Context':<28} {'CosSim':>8}  Top successors")
        print("-" * 78)
        for rank, ctx in enumerate(self.latent_sorted_keys[:top_n]):
            sim   = self.latent_sim_scores[ctx]
            succs = self.ngram_zones.get(ctx, [])[:5]
            ctx_s = " | ".join(f"'{w}'" for w in ctx)
            print(f"{rank:<5} ({ctx_s:<26}) {sim:>8.4f}  {succs}")

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
# PSPACE zones
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
            path = file_obj if isinstance(file_obj, str) else file_obj.name
            with open(path, "rb") as fh:
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
# get_or_build
# ---------------------------------------------------------------------------

def get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log,
                 progress=None):
    def _prog(frac, desc):
        if progress is not None:
            progress(frac, desc=desc)
        log.append(desc)

    key = (corpus_fingerprint(corpus), int(ngram_n), float(lidstone_gamma),
           int(pi_prec), int(pi_stream_len))
    if CACHE.get("key") == key and CACHE.get("cpd") is not None:
        _prog(1.0, "Using cached model and stream.")
        return CACHE["cpd"], CACHE["vocab"], CACHE["stream"]
    CACHE["_corpus_text"] = corpus

    _prog(0.05, "Building ngram model...")
    cpd, vocab = build_model(corpus, ngram_n, lidstone_gamma)

    _prog(0.35, f"Building pi stream (prec={pi_prec}, len={pi_stream_len})...")
    raw_stream = build_pi_stream(pi_prec, pi_stream_len)

    _prog(0.55, "Applying roll_until_column_match...")
    matrix  = stream_to_matrix(raw_stream, n_cols=26)
    stacked, match_shift = roll_until_column_match(matrix)
    log.append(
        f"Column match shift={match_shift}; stacked shape={stacked.shape}."
        if match_shift >= 0
        else f"No exact column match; stacked shape={stacked.shape}."
    )
    stream = stacked.flatten().tolist()

    _prog(0.72, "Building context zone index + trigram latent space...")
    corpus_tokens = tokenise_alpha(CACHE.get("_corpus_text", ""))
    context_index = build_context_index(vocab, cpd, corpus_tokens)

    log.append(
        f"Zone index: high={len(context_index.freq_zones['high'])} "
        f"mid={len(context_index.freq_zones['mid'])} "
        f"low={len(context_index.freq_zones['low'])} "
        f"alpha_keys={len(context_index.alpha_zones)} "
        f"ngram_keys={len(context_index.ngram_zones)}."
    )

    if context_index.latent_sorted_keys:
        top_ctx = context_index.latent_sorted_keys[0]
        top_sim = context_index.latent_sim_scores[top_ctx]
        log.append(
            f"Latent BOS sort: {len(context_index.latent_sorted_keys)} contexts sorted. "
            f"Top context={top_ctx}, cosine_sim={top_sim:.4f}. "
            + "Quartile sizes: "
            + ", ".join(
                f"q{i}={len(context_index.latent_bos_data.get(f'q{i}', []))}"
                for i in range(4)
            )
        )

    CACHE.update(key=key, cpd=cpd, vocab=vocab, stream=stream, context_index=context_index)
    _prog(1.0, "Model ready — cached.")
    return cpd, vocab, stream



# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def save_model_to_path(path, cpd, vocab, stream, ngram_n, lidstone_gamma,
                       pi_prec, pi_stream_len, corpustext):
    payload = dict(
        magic="PI_TRIGRAM_MODEL_V1",
        version=1,
        cpd=cpd,
        vocab=set(vocab),
        stream=list(stream),
        config=dict(
            ngram_n=int(ngram_n),
            lidstone_gamma=float(lidstone_gamma),
            pi_prec=int(pi_prec),
            pi_stream_len=int(pi_stream_len),
            digits_per_sample=int(DEFAULTS["DIGITS_PER_SAMPLE"]),
        ),
        corpus_sha256=corpus_fingerprint(corpustext),
        corpus_chars=len(corpustext) if corpustext else 0,
        vocab_size=len(vocab),
    )
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model_from_path(path):
    errors = []
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e2:
            return None, None, None, None, [
                f"Could not read file: {e}",
                f"Uncompressed fallback also failed: {e2}",
            ]
    if not isinstance(payload, dict):
        return None, None, None, None, ["File does not contain a model dict."]
    cpd    = payload.get("cpd")
    vocab  = payload.get("vocab")
    stream = payload.get("stream")
    config = payload.get("config", {})
    if cpd is None or vocab is None or stream is None:
        return None, None, None, None, errors + ["Missing required fields: cpd/vocab/stream."]
    return cpd, set(vocab), list(stream), dict(config), errors


def save_model_ui(file_obj, pasted_corpus, pi_prec, pi_stream_len, ngram_n, lidstone_gamma):
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f"Corpus source: {source} ({len(corpus)} chars).")
    cpd, vocab, stream = get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl.gz", prefix="pi_model_")
    tmp.close()
    save_model_to_path(tmp.name, cpd, vocab, stream, ngram_n, lidstone_gamma,
                       pi_prec, pi_stream_len, corpus)
    log.append(f"Saved model to {os.path.basename(tmp.name)}")
    return tmp.name, "\n".join(log)


def load_model_ui(modelfile):
    if modelfile is None:
        return "No file uploaded.", gr.update(), gr.update(), gr.update(), gr.update()
    path = modelfile if isinstance(modelfile, str) else modelfile.name
    cpd, vocab, stream, config, errors = load_model_from_path(path)
    if cpd is None:
        return "Failed to load model: " + " | ".join(errors), gr.update(), gr.update(), gr.update(), gr.update()
    CACHE.update(key=("LOADED", path), cpd=cpd, vocab=vocab, stream=stream)
    log = [f"Loaded model from {os.path.basename(path)}."]
    if errors:
        log.extend([f"! {e}" for e in errors])
    return (
        "\n".join(log),
        gr.update(value=config.get("pi_prec",        DEFAULTS["PI_PREC"])),
        gr.update(value=config.get("pi_stream_len",  DEFAULTS["PI_STREAM_LEN"])),
        gr.update(value=config.get("ngram_n",        DEFAULTS["NGRAM_N"])),
        gr.update(value=config.get("lidstone_gamma", DEFAULTS["LIDSTONE_GAMMA"])),
    )


# ---------------------------------------------------------------------------
# HF model loader (stub — kept for UI compatibility)
# ---------------------------------------------------------------------------

def load_hf_model_on_demand(repo_id, hf_token=None):
    HF_CACHE["status"] = f"HF loading not implemented in this build ({repo_id})."
    return HF_CACHE["status"]


# ===========================================================================
#  Layer Isomorphism  (embedded — no separate file needed)
# ===========================================================================

def _li_normalise(vec: np.ndarray) -> np.ndarray:
    t = vec.sum()
    return vec / t if t > 1e-30 else np.ones_like(vec) / max(1, len(vec))


def _li_insight_penalty(pairs, strength):
    if not pairs or strength <= 0:
        return pairs
    mean_p = sum(p for _, p in pairs) / len(pairs)
    if mean_p <= 0:
        return pairs
    out = []
    for w, p in pairs:
        excess = max(0.0, p - mean_p)
        p2 = p / (1.0 + strength * excess / mean_p)
        out.append((w, max(1e-12, p2)))
    t = sum(p for _, p in out)
    return [(w, p / t) for w, p in out] if t > 0 else out


def _li_zone_gradient(zone_set, candidates, sigma=0.35, floor=0.05) -> np.ndarray:
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
    return _li_normalise(weights)


def _li_char_trigrams(word: str):
    return {word[i:i+3] for i in range(len(word) - 2)} if len(word) >= 3 else {word}


class LayerFrame:
    __slots__ = ("step", "layers", "chosen", "context_window", "zone_name", "draw_pos", "next_draw_pos")

    def __init__(
        self,
        step: int,
        layers: List[Dict],
        chosen: str = "",
        context_window: Tuple[str, ...] = (),
        zone_name: str = "",
        draw_pos: int = 0,
        next_draw_pos: int = 0,
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

    def tensor(self) -> np.ndarray:
        rows = [l["probs"] for l in self.layers]
        if not rows:
            return np.array([])
        max_len = max(len(r) for r in rows)
        padded  = [np.pad(r, (0, max_len - len(r))) for r in rows]
        return np.vstack(padded)


class IsomorphismGenerator:
    """
    13-layer isomorphic probability tensor for every generated token.

    L0  RAW_DIST       CPD posterior + rep_penalty
    L1  TEMP_SCALED    temperature scaling
    L2  INSIGHT        insight penalty
    L3  TOPK_TOPP      top-k / top-p truncation
    L4  ZONE_FREQ      freq-zone Gaussian gradient
    L5  ZONE_ALPHA     alpha-zone Gaussian gradient
    L6  ZONE_BIGRAM    prompt bigram ngram_zones gradient
    L7  ZONE_TRIGRAM   live (w-2,w-1) context ngram_zones gradient
    L8  ZONE_CHAR_TRIG char-trigram neighbour gradient
    L9  ZONE_LATENT    latent-BOS quartile gradient
    L10 HISTORY        1/(1+count) repetition column
    L11 TENSOR_BLEND   row-weighted vstack of L4..L10
    L12 FINAL          geometric mean(L3, L11) -> chosen token
    """

    LAYER_NAMES = [
        "L0_RAW_DIST", "L1_TEMP_SCALED", "L2_INSIGHT", "L3_TOPK_TOPP",
        "L4_ZONE_FREQ", "L5_ZONE_ALPHA", "L6_ZONE_BIGRAM", "L7_ZONE_TRIGRAM",
        "L8_ZONE_CHAR_TRIG", "L9_ZONE_LATENT", "L10_HISTORY",
        "L11_TENSOR_BLEND", "L12_FINAL", "L13_CTX_REQ_POS",
    ]

    def __init__(
        self,
        cpd,
        context_index,
        vocab,
        ngram_n: int = 2,
        temperature: float = 4.3,
        top_k: int = 100,
        top_p: float = 1.0,
        rep_penalty: float = 1.13,
        insight_penalty: float = 3.95,
        history: Optional[Counter] = None,
    ):
        self.cpd             = cpd
        self.ctx_idx         = context_index
        self.vocab           = set(vocab)
        self.ngram_n         = max(2, int(ngram_n))
        self.context_window  = self.ngram_n - 1
        self.temperature     = max(1e-3, float(temperature))
        self.top_k           = max(1, int(top_k))
        self.top_p           = max(1e-3, min(1.0, float(top_p)))
        self.rep_penalty     = max(1.0, float(rep_penalty))
        self.insight_penalty = max(0.0, float(insight_penalty))
        self.history         = Counter(history) if history else Counter()
        self._step           = 0
        self._pos            = 0   # contextual requestor position (pi-stream cursor)
        self._char_trig_index: Dict[str, set] = (
            getattr(context_index, "_trig_index", {}) if context_index else {}
        )

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

    def _build_L0(self, dist):
        raw = []
        for s in dist.samples():
            if s:
                p   = max(1e-12, float(dist.prob(s)))
                cnt = self.history[s]
                if cnt > 0:
                    p /= self.rep_penalty ** cnt
                raw.append((s, p))
        raw.sort(key=lambda x: x[1], reverse=True)
        t   = sum(p for _, p in raw)
        raw = [(w, p / t) for w, p in raw] if t > 0 else raw
        words = [w for w, _ in raw]
        probs = np.array([p for _, p in raw], dtype=np.float64)
        return raw, {"name": "L0_RAW_DIST", "words": words, "probs": probs,
                     "source": "CPD posterior + rep_penalty"}

    def _build_L1(self, L0_pairs):
        scaled = [(w, p ** (1.0 / self.temperature)) for w, p in L0_pairs]
        t      = sum(p for _, p in scaled)
        scaled = [(w, p / t) for w, p in scaled] if t > 0 else scaled
        words  = [w for w, _ in scaled]
        probs  = np.array([p for _, p in scaled], dtype=np.float64)
        return scaled, {"name": "L1_TEMP_SCALED", "words": words, "probs": probs,
                        "source": f"temperature={self.temperature}"}

    def _build_L2(self, L1_pairs):
        penalised = _li_insight_penalty(L1_pairs, self.insight_penalty)
        words     = [w for w, _ in penalised]
        probs     = np.array([p for _, p in penalised], dtype=np.float64)
        return penalised, {"name": "L2_INSIGHT", "words": words, "probs": probs,
                           "source": f"insight_penalty={self.insight_penalty}"}

    def _build_L3(self, L2_pairs):
        trunc       = L2_pairs[:self.top_k]
        kept, accum = [], 0.0
        for w, p in trunc:
            kept.append((w, p))
            accum += p
            if accum >= self.top_p:
                break
        t    = sum(p for _, p in kept)
        kept = [(w, p / t) for w, p in kept] if t > 0 else kept
        words = [w for w, _ in kept]
        probs = np.array([p for _, p in kept], dtype=np.float64)
        return kept, {"name": "L3_TOPK_TOPP", "words": words, "probs": probs,
                      "source": f"top_k={self.top_k} top_p={self.top_p}"}

    def _zone_layer(self, name, zone_set, candidates, sigma, floor, source) -> Dict:
        g = _li_zone_gradient(zone_set, candidates, sigma=sigma, floor=floor)
        return {"name": name, "words": [w for w, _ in candidates],
                "probs": g, "source": source}

    def _build_zone_layers(self, L3_pairs, prompt_words, context_deque):
        ci     = self.ctx_idx
        layers = []

        if ci is None:
            flat = _li_normalise(np.ones(len(L3_pairs)))
            for name in ["L4_ZONE_FREQ", "L5_ZONE_ALPHA", "L6_ZONE_BIGRAM",
                         "L7_ZONE_TRIGRAM", "L8_ZONE_CHAR_TRIG", "L9_ZONE_LATENT"]:
                layers.append({"name": name, "words": [w for w, _ in L3_pairs],
                                "probs": flat.copy(), "source": "no context_index"})
            return layers

        high_set = set(ci.freq_zones.get("high", []))
        if any(w in high_set for w in prompt_words):
            freq_key = "high"
        elif all(ci.token_freq.get(w, 0) < ci.FREQ_MID_THRESH for w in prompt_words):
            freq_key = "low"
        else:
            freq_key = "mid"
        layers.append(self._zone_layer(
            "L4_ZONE_FREQ", set(ci.freq_zones.get(freq_key, [])),
            L3_pairs, 0.50, 0.05, f"freq_zone={freq_key}"))

        alpha_words: set = set()
        for w in prompt_words:
            if w and w[0].isalpha():
                alpha_words.update(ci.alpha_zones.get(w[0], []))
        layers.append(self._zone_layer(
            "L5_ZONE_ALPHA", alpha_words, L3_pairs, 0.40, 0.05,
            f"alpha_keys={[w[0] for w in prompt_words if w]}"))

        bigram_words: set = set()
        for i in range(len(prompt_words) - 1):
            bigram_words.update(ci.ngram_zones.get(
                (prompt_words[i], prompt_words[i + 1]), []))
        layers.append(self._zone_layer(
            "L6_ZONE_BIGRAM", bigram_words, L3_pairs, 0.25, 0.04,
            "ngram bigram context"))

        ctx_list   = list(context_deque)
        trig_words: set = set()
        if len(ctx_list) >= 2:
            trig_key   = tuple(ctx_list[-2:])
            trig_words = set(ci.ngram_zones.get(trig_key, []))
            trig_src   = f"live_ctx={trig_key}"
        elif len(ctx_list) == 1:
            trig_key   = (ctx_list[-1],)
            trig_words = set(ci.ngram_zones.get(trig_key, []))
            trig_src   = f"live_ctx=({ctx_list[-1]},)"
        else:
            trig_src = "no live context"
        layers.append(self._zone_layer(
            "L7_ZONE_TRIGRAM", trig_words, L3_pairs, 0.20, 0.03, trig_src))

        prompt_tgs: set = set()
        for w in prompt_words:
            prompt_tgs |= _li_char_trigrams(w)
        char_neighbours: set = set()
        for tg in prompt_tgs:
            char_neighbours |= self._char_trig_index.get(tg, set())
        layers.append(self._zone_layer(
            "L8_ZONE_CHAR_TRIG", char_neighbours, L3_pairs, 0.35, 0.04,
            "char-trigram neighbours"))

        n_keys = len(ci.latent_sorted_keys)
        q_key  = "q0"
        for ctx in ci.latent_sorted_keys:
            if any(w in ctx for w in prompt_words):
                rank  = ci.latent_sorted_keys.index(ctx)
                q_key = f"q{min(3, rank * 4 // max(1, n_keys))}"
                break
        layers.append(self._zone_layer(
            "L9_ZONE_LATENT", set(ci.latent_bos_data.get(q_key, [])),
            L3_pairs, 0.30, 0.04, f"latent_bos_quartile={q_key}"))

        return layers

    def _build_L10(self, L3_pairs) -> Dict:
        hist_vec = np.array(
            [1.0 / (1.0 + self.history[w]) for w, _ in L3_pairs],
            dtype=np.float64)
        return {"name": "L10_HISTORY", "words": [w for w, _ in L3_pairs],
                "probs": _li_normalise(hist_vec), "source": "repetition history"}

    def _build_L11(self, zone_layers, L10, L3_pairs) -> Dict:
        n           = len(L3_pairs)
        rows        = [l["probs"] for l in zone_layers] + [L10["probs"]]
        zone_matrix = np.vstack(rows)
        row_norms   = zone_matrix.sum(axis=1, keepdims=True).clip(1e-12)
        augmented   = np.hstack([zone_matrix, row_norms])
        row_weights = augmented[:, :n].mean(axis=1)
        row_weights = np.clip(row_weights, 1e-12, None)
        row_weights /= row_weights.sum()
        weights     = (augmented[:, :n] * row_weights[:, None]).sum(axis=0)
        return {"name": "L11_TENSOR_BLEND",
                "words": [w for w, _ in L3_pairs],
                "probs": _li_normalise(np.clip(weights, 1e-12, None)),
                "source": f"row-weighted blend of L4..L10 ({len(rows)} rows)"}

    def _build_L12(self, L3_pairs, L11):
        blended = [
            (w, math.sqrt(max(1e-24, p) * float(lw)))
            for (w, p), lw in zip(L3_pairs, L11["probs"])
        ]
        t       = sum(p for _, p in blended)
        blended = [(w, p / t) for w, p in blended] if t > 0 else blended
        blended.sort(key=lambda x: x[1], reverse=True)
        words = [w for w, _ in blended]
        probs = np.array([p for _, p in blended], dtype=np.float64)
        return blended, {"name": "L12_FINAL", "words": words, "probs": probs,
                         "source": "geometric mean(L3, L11)"}

    def _build_L13(self, L3_pairs: list, draw_pos: int, stream_len: int) -> Dict:
        """
        L13  CONTEXTUAL REQUESTOR POSITION
        ───────────────────────────────────
        Records the pi-stream cursor position at the moment this token was
        drawn.  The position is normalised to [0, 1] over the stream length
        and broadcast as a uniform-weighted column over the L3 candidate set,
        then added to the tensor.  After this layer is built, the internal
        cursor advances by draw_pos % stream_len so the *next* step's draw
        begins from a position that encodes where *this* step landed.
        """
        n        = len(L3_pairs)
        norm_pos = (draw_pos % max(1, stream_len)) / max(1, stream_len - 1)
        # Broadcast the scalar position as a rank-weighted gradient:
        # words near rank = norm_pos * (n-1) get the highest weight.
        centre   = norm_pos
        indices  = np.arange(n, dtype=np.float64) / max(1, n - 1)
        sigma    = 0.30
        floor    = 0.04
        gauss    = np.exp(-0.5 * ((indices - centre) / max(1e-6, sigma)) ** 2)
        weights  = floor + (1.0 - floor) * gauss
        weights  = _li_normalise(weights)
        return {
            "name":   "L13_CTX_REQ_POS",
            "words":  [w for w, _ in L3_pairs],
            "probs":  weights,
            "source": f"ctx_req_pos={draw_pos} norm={norm_pos:.4f} stream_len={stream_len}",
        }

    def step(
        self,
        context_deque: deque,
        prompt_words: List[str],
        draw: float,
        zone_name: str = "",
    ) -> Optional[LayerFrame]:
        dist = self._dist_for_ctx(tuple(context_deque))
        if dist is None:
            return None

        L0_pairs, L0 = self._build_L0(dist)
        if not L0_pairs:
            return None

        L1_pairs, L1 = self._build_L1(L0_pairs)
        L2_pairs, L2 = self._build_L2(L1_pairs)
        L3_pairs, L3 = self._build_L3(L2_pairs)
        if not L3_pairs:
            return None

        zone_layers    = self._build_zone_layers(L3_pairs, prompt_words, context_deque)
        L10            = self._build_L10(L3_pairs)
        L11            = self._build_L11(zone_layers, L10, L3_pairs)
        L12_pairs, L12 = self._build_L12(L3_pairs, L11)

        # ── L13: contextual requestor position ────────────────────────
        # Capture where the pi-stream cursor sat when we called step().
        draw_pos    = self._pos
        stream_len  = max(1, len(getattr(self, '_stream', [])) or 1)
        L13         = self._build_L13(L3_pairs, draw_pos, stream_len)

        # Geometric blend of L12 final probs with L13 positional weights
        l12_map = dict(zip([w for w, _ in L12_pairs], [p for _, p in L12_pairs]))
        l13_map = dict(zip(L13['words'], L13['probs'].tolist()))
        all_words = list(l12_map.keys())
        floor     = 1e-12
        blended   = [
            (w, math.sqrt(max(floor, l12_map.get(w, floor)) *
                          max(floor, l13_map.get(w, floor))))
            for w in all_words
        ]
        bt = sum(p for _, p in blended)
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

        # Advance internal cursor by draw_pos so the NEXT step starts
        # from a position that encodes where THIS step landed.
        next_draw_pos   = (draw_pos + (draw_pos % max(1, stream_len))) % stream_len
        self._pos       = next_draw_pos

        return LayerFrame(
            step           = self._step,
            layers         = [L0, L1, L2, L3] + zone_layers + [L10, L11, L12, L13],
            chosen         = chosen,
            context_window = tuple(context_deque),
            zone_name      = zone_name,
            draw_pos       = draw_pos,
            next_draw_pos  = next_draw_pos,
        )

    def seed_stream(self, stream: list):
        """Attach the raw pi-stream so L13 can read its length."""
        self._stream = list(stream)
        self._pos    = 0

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
            self._step += 1
            ctx.append(frame.chosen)
            yield frame


# ---------------------------------------------------------------------------
# Layer Tensor Gradio backend
# ---------------------------------------------------------------------------

def _layer_iso_run(
    file_obj, pasted_corpus, prompt,
    pi_prec, pi_stream_len, ngram_n, lidstone_gamma,
    temperature, gen_words, rep_penalty, insight_penalty,
    progress=gr.Progress(track_tqdm=False),
):
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f"Corpus source: {source} ({len(corpus)} chars).")
    cpd, vocab, stream = get_or_build(
        corpus, int(ngram_n), float(lidstone_gamma),
        int(pi_prec), int(pi_stream_len), log, progress=progress,
    )
    context_index = CACHE.get("context_index")

    gen = IsomorphismGenerator(
        cpd             = cpd,
        context_index   = context_index,
        vocab           = vocab,
        ngram_n         = int(ngram_n),
        temperature     = float(temperature),
        top_k           = DEFAULTS["TOP_K"],
        top_p           = DEFAULTS["TOP_P"],
        rep_penalty     = float(rep_penalty),
        insight_penalty = float(insight_penalty),
    )

    _pos    = [0]
    _stream = list(stream)
    _dps    = DEFAULTS["DIGITS_PER_SAMPLE"]
    gen.seed_stream(_stream)   # attach stream so L13 can read length & cursor

    def _draw():
        val  = 0
        base = 26 ** _dps
        for _ in range(_dps):
            val  = val * 26 + _stream[_pos[0] % max(1, val + 1)]
            _pos[0] += 1
        # Keep external cursor in sync with the IsomorphismGenerator internal cursor
        _pos[0] = gen._pos % max(1, len(_stream))
        return val / base

    def _zone_fn(_draw_val):
        return _pi_activate_zone(_draw_val)

    frames = []
    try:
        for frame in gen.generate(
            prompt  = prompt or "",
            n_words = int(gen_words),
            draw_fn = _draw,
            zone_fn = _zone_fn,
        ):
            frames.append(frame)
    except Exception as e:
        log.append(f"Generation error: {e}")

    if not frames:
        return "", "No frames generated.", "{}", "\n".join(log)

    words    = [f.chosen for f in frames if f.chosen]
    out_text = (" ".join(tokenise_alpha(prompt or "")) + " " + " ".join(words)).strip()

    header = "| Step | Chosen | Zone | Ctx | " + " | ".join(
        l.split("_", 1)[1] for l in IsomorphismGenerator.LAYER_NAMES
    ) + " |"
    sep  = "|" + "---|" * (4 + len(IsomorphismGenerator.LAYER_NAMES))
    rows = [header, sep]

    for f in frames[:60]:
        top_per_layer = []
        for layer in f.layers:
            if len(layer["words"]) > 0:
                idx = int(np.argmax(layer["probs"]))
                top_per_layer.append(f"{layer['words'][idx]}:{layer['probs'][idx]:.2f}")
            else:
                top_per_layer.append("-")
        ctx_str = ",".join(w for w in f.context_window if w) or "BOS"
        rows.append(
            f"| {f.step} | **{f.chosen}** | {f.zone_name or '-'} | {ctx_str} | "
            + " | ".join(top_per_layer) + " |"
        )
    table_md = "\n".join(rows)

    tensor_log = []
    for f in frames:
        tensor_log.append({
            "step":          f.step,
            "chosen":        f.chosen,
            "zone":          f.zone_name,
            "ctx":           list(f.context_window),
            "draw_pos":      f.draw_pos,
            "next_draw_pos": f.next_draw_pos,
            "shape":         list(f.tensor().shape),
            "layers": [
                {
                    "name": l["name"],
                    "top3": sorted(
                        zip(l["words"], l["probs"].tolist()),
                        key=lambda x: -x[1]
                    )[:3],
                }
                for l in f.layers
            ],
        })

    log.append(f"Generated {len(frames)} frames, {len(words)} tokens.")
    return out_text, table_md, json.dumps(tensor_log, indent=2), "\n".join(log)


# ===========================================================================
#  Gradio UI
# ===========================================================================

def build_ui():
    with gr.Blocks(title="Pi Trigram — Layer Tensor") as demo:
        gr.Markdown("## Pi Trigram · Layer Tensor Generator")

        with gr.Tabs():

            # ── Tab: Layer Tensor ─────────────────────────────────────────
            with gr.TabItem("Layer Tensor"):
                gr.Markdown(
                    "**Isomorphism generator** — runs the full 14-layer probability tensor "
                    "(`L0`–`L13`) for every token.  Layer **L13** records the pi-stream "
                    "contextual requestor position at each draw and feeds it forward to the "
                    "next step."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        lt_filein   = gr.File(label="Upload corpus (.txt)", file_types=[".txt", ".md"], type="filepath")
                        lt_pasted   = gr.Textbox(label="or paste corpus here", lines=5)
                        lt_promptin = gr.Textbox(label="Prompt", lines=2, value="alice rabbit hole")
                    with gr.Column(scale=1):
                        lt_pi_prec        = gr.Slider(500,   30000, value=DEFAULTS["PI_PREC"],        step=500,   label="Pi precision")
                        lt_pi_stream_len  = gr.Slider(500,   30000, value=DEFAULTS["PI_STREAM_LEN"],  step=500,   label="Stream length")
                        lt_ngram_n        = gr.Slider(2,     6,     value=DEFAULTS["NGRAM_N"],         step=1,     label="N-gram order")
                        lt_lidstone_gamma = gr.Slider(0.001, 1.0,   value=DEFAULTS["LIDSTONE_GAMMA"], step=0.001, label="Lidstone gamma")
                        lt_temperature    = gr.Slider(0.1,   5.0,   value=DEFAULTS["TEMPERATURE"],    step=0.05,  label="Temperature")
                        lt_gen_words      = gr.Slider(1,     500,   value=80,                          step=1,     label="Words to generate")
                        lt_rep_penalty    = gr.Slider(1.0,   2.0,   value=DEFAULTS["REP_PENALTY"],     step=0.01,  label="Repetition penalty")
                        lt_insight_pen    = gr.Slider(0.0,   5.0,   value=DEFAULTS["INSIGHT_PENALTY"], step=0.05,  label="Insight penalty")

                btn_lt       = gr.Button("Run Layer Tensor", variant="primary")
                lt_out_text  = gr.Textbox(label="Generated text", lines=6)
                lt_out_table = gr.Markdown(label="Layer-by-layer top token per step")
                lt_out_json  = gr.Code(label="Full tensor log (JSON)", language="json", lines=20)
                lt_out_log   = gr.Textbox(label="Build log", lines=5)

                btn_lt.click(
                    _layer_iso_run,
                    inputs=[
                        lt_filein, lt_pasted, lt_promptin,
                        lt_pi_prec, lt_pi_stream_len, lt_ngram_n, lt_lidstone_gamma,
                        lt_temperature, lt_gen_words, lt_rep_penalty, lt_insight_pen,
                    ],
                    outputs=[lt_out_text, lt_out_table, lt_out_json, lt_out_log],
                )

            # ── Tab: Model I/O ────────────────────────────────────────────
            with gr.TabItem("Model I/O"):
                gr.Markdown("Save/load compiled trigram model.")

                gr.Markdown("#### Load latest model from Hugging Face")
                model_hf_repo   = gr.Textbox(label="HF repo ID", value=HF_REPO_ID)
                model_hf_token  = gr.Textbox(label="HF token (optional)", type="password")
                load_latest_btn = gr.Button("🔄  Load Latest from HF", variant="primary")
                load_latest_log = gr.Textbox(label="Load log", lines=3, interactive=False)
                status_md       = gr.Markdown("### Ready")

                gr.Markdown("---")
                gr.Markdown("#### Save / load local .pkl.gz")
                save_btn  = gr.Button("Save model", variant="secondary")
                save_file = gr.File(label="Saved model file", interactive=False)
                model_log = gr.Textbox(label="Model I/O log", lines=8)
                load_file = gr.File(label="Load saved model .pkl.gz",
                                    file_types=[".gz", ".pkl"], type="filepath")
                load_btn  = gr.Button("Load local model", variant="secondary")

                def _load_latest_and_update_status(repo, tok):
                    msg    = load_hf_model_on_demand(repo, tok)
                    status = HF_CACHE.get("status", "Unknown")
                    return msg, f"### Model status\n\n`{status}`"

                load_latest_btn.click(
                    _load_latest_and_update_status,
                    inputs=[model_hf_repo, model_hf_token],
                    outputs=[load_latest_log, status_md],
                )
                save_btn.click(
                    save_model_ui,
                    inputs=[lt_filein, lt_pasted,
                            lt_pi_prec, lt_pi_stream_len, lt_ngram_n, lt_lidstone_gamma],
                    outputs=[save_file, model_log],
                )
                load_btn.click(
                    load_model_ui,
                    inputs=[load_file],
                    outputs=[model_log,
                             lt_pi_prec, lt_pi_stream_len, lt_ngram_n, lt_lidstone_gamma],
                )

        gr.Markdown("Tip: model cache is reused until corpus or configuration changes.")

    return demo


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == "__main__":
    build_ui().queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        show_error=True,
    )
