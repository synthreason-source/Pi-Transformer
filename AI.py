#!/usr/bin/env python3
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
CACHE = dict(key=None, cpd=None, vocab=None, stream=None)


def tokenise_alpha(text):
    if text is None:
        return []
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    elif not isinstance(text, str):
        text = str(text)
    return _WORD_RE.findall(text.lower())


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
        raise ValueError("Corpus produced zero tokens after tokenisation. Upload a non-empty text corpus or paste some text.")
    padded = ["<s>"] * (ngram_n - 1) + tokens + ["</s>"]
    trigrams_ = list(ngrams(padded, ngram_n))
    cfd = ConditionalFreqDist((tuple(tg[:-1]), tg[-1]) for tg in trigrams_)
    vocab = set(tokens) | {"</s>"}
    for ctx in list(cfd.conditions()):
        if len(cfd[ctx]) == 0:
            cfd[ctx]["</s>"] += 1
    cpd = ConditionalProbDist(cfd, _LidstoneFactory(gamma=lidstone_gamma, bins=max(1, len(vocab))))
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
                ripple = 0.5 + 0.5 * math.cos((d / max(1e-9, spread)) * math.pi * (1.5 + shimmer) + phase + 0.13 * t)
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


class PiSampler:
    def __init__(self, stream, digits_per_sample, temperature, top_k, top_p, repetition_penalty, seashell_enable, seashell_strength, seashell_decay, seashell_peaks, seashell_width, seashell_floor):
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
            self.seashell = SeashellResonator(self, seashell_strength, seashell_decay, seashell_peaks, seashell_width, seashell_floor)
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
            xor_draw = (u_a * (1 - u_b) * (1 - u_c) + u_b * (1 - u_a) * (1 - u_c) + u_c * (1 - u_a) * (1 - u_b)) / 1.5
            cumulative = 0.0
            chosen = xor_scored[-1][0]
            for word, p in xor_scored:
                cumulative += p
                if xor_draw < cumulative:
                    chosen = word
                    break
        self.history[chosen] += 1
        return chosen


class Triangle:
    def __init__(self, stream_len, offset_extra=0, bend_degrees=13.0):
        base = offset_extra % stream_len
        bend_shift = int(round((bend_degrees / 360.0) * stream_len))
        self.A = base % stream_len
        self.B = (base + stream_len // 3 + bend_shift) % stream_len
        self.C = (base + 2 * stream_len // 3 + bend_shift) % stream_len
        self.vertices = {"A": self.A, "B": self.B, "C": self.C}


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
        init = ["<s>"] * (context_window - len(seed_in_vocab)) + seed_in_vocab
    context = deque(init, maxlen=context_window)
    words = list(seed_words)
    def dist_for_ctx(ctxtuple):
        for cut in range(len(ctxtuple), 0, -1):
            trial = ("<s>",) * (context_window - cut) + ctxtuple[-cut:]
            try:
                d = cpd[trial]
                if list(d.samples()):
                    return d
            except Exception:
                continue
        try:
            d = cpd[tuple(["<s>"] * context_window)]
            if list(d.samples()):
                return d
        except Exception:
            pass
        return None
    for _ in range(n_words):
        dist = dist_for_ctx(tuple(context))
        if dist is None:
            context.clear()
            context.extend(["<s>"] * context_window)
            continue
        word = sampler.sample(dist)
        if word == "</s>":
            context.clear()
            context.extend(["<s>"] * context_window)
            continue
        words.append(word)
        context.append(word)
    return capitalise_text(words)


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
            return None, None, None, None, [f'Could not read file: {e}', f'Uncompressed fallback also failed: {e2}']
    if not isinstance(payload, dict):
        return None, None, None, None, ['File does not contain a model dict.']
    cpd = payload.get('cpd')
    vocab = payload.get('vocab')
    stream = payload.get('stream')
    config = payload.get('config', {})
    if cpd is None or vocab is None or stream is None:
        return None, None, None, None, errors + ['Missing required fields: cpd/vocab/stream.']
    return cpd, set(vocab), list(stream), dict(config), errors


def get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log):
    key = (corpus_fingerprint(corpus), int(ngram_n), float(lidstone_gamma), int(pi_prec), int(pi_stream_len))
    if CACHE.get('key') == key and CACHE.get('cpd') is not None:
        log.append('Using cached model and stream.')
        return CACHE['cpd'], CACHE['vocab'], CACHE['stream']
    log.append('Building trigram model...')
    cpd, vocab = build_model(corpus, ngram_n, lidstone_gamma)
    log.append(f'Building stream with pi_prec={pi_prec}, pi_stream_len={pi_stream_len}...')
    stream = build_pi_stream(pi_prec, pi_stream_len)
    CACHE.update(key=key, cpd=cpd, vocab=vocab, stream=stream)
    log.append('Cached.')
    return cpd, vocab, stream


def run_single(file_obj, pasted_corpus, prompt, pi_prec, pi_stream_len, ngram_n, lidstone_gamma, gen_words, temperature, top_k, top_p, rep_penalty, seashell_enable, seashell_strength, seashell_decay, seashell_peaks, seashell_width, seashell_floor, bend_degrees, offset, vertex):
    log = []
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f'Corpus source: {source} ({len(corpus)} chars).')
    cpd, vocab, stream = get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log)
    triangle = Triangle(int(pi_stream_len), offset_extra=int(offset), bend_degrees=float(bend_degrees))
    start = triangle.vertices[vertex]
    log.append(f'Triangle A={triangle.A} B={triangle.B} C={triangle.C} vertex={vertex} start={start}')
    sampler = PiSampler(stream, digits_per_sample=DEFAULTS['DIGITS_PER_SAMPLE'], temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=rep_penalty, seashell_enable=seashell_enable, seashell_strength=seashell_strength, seashell_decay=seashell_decay, seashell_peaks=seashell_peaks, seashell_width=seashell_width, seashell_floor=seashell_floor)
    sampler.seek(start)
    text = generate_text(cpd, sampler, prompt=prompt or '', n_words=int(gen_words), ngram_n=int(ngram_n), vocab=vocab)
    oov = [w for w in tokenise_alpha(prompt or '') if w not in vocab]
    if oov:
        log.append(f'{len(oov)} prompt tokens not in corpus vocab: {oov}')
    log.append(f'Done. {len(text.split())} output tokens.')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', prefix='pigenerate_', mode='w', encoding='utf-8')
    tmp.write(text)
    tmp.close()
    return text, '\n'.join(log), tmp.name


def run_search(file_obj, pasted_corpus, prompt, pi_prec, pi_stream_len, ngram_n, lidstone_gamma, gen_words, temperature, rep_penalty, seashell_enable, seashell_strength, seashell_decay, seashell_peaks, seashell_width, seashell_floor, vertex, bend_max, bend_step, offset_step, fuzzy_threshold, max_solutions, progress=gr.Progress(track_tqdm=False)):
    log = []
    if not prompt or not prompt.strip():
        return '', 'Prompt is empty — search needs word pairs.', None
    corpus, source = resolve_corpus(file_obj, pasted_corpus)
    log.append(f'Corpus source: {source} ({len(corpus)} chars).')
    cpd, vocab, stream = get_or_build(corpus, ngram_n, lidstone_gamma, pi_prec, pi_stream_len, log)
    pairs = extract_word_pairs(prompt)
    if not pairs:
        return '', 'No valid word pairs extracted from prompt.', None
    log.append(f'Prompt yields {len(pairs)} word pairs.')
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
            sampler = PiSampler(stream, digits_per_sample=DEFAULTS['DIGITS_PER_SAMPLE'], temperature=temperature, top_k=DEFAULTS['TOP_K'], top_p=DEFAULTS['TOP_P'], repetition_penalty=rep_penalty, seashell_enable=seashell_enable, seashell_strength=seashell_strength, seashell_decay=seashell_decay, seashell_peaks=seashell_peaks, seashell_width=seashell_width, seashell_floor=seashell_floor)
            sampler.seek(start)
            text = generate_text(cpd, sampler, prompt=prompt, n_words=int(gen_words), ngram_n=int(ngram_n), vocab=vocab)
            exact_ok, _failed = all_pairs_match(pairs, text, fuzzy_threshold=float(fuzzy_threshold))
            assoc_score, matched_pairs = collocation_association_score(text, prompt, min_freq=1, measure='pmi')
            if exact_ok or assoc_score > 0:
                scored_results.append({'prompt': prompt, 'bend': bend, 'offset': offset, 'vertex': vertex, 'text': text, 'assoc_score': assoc_score, 'matched_pairs': matched_pairs, 'exact_ok': exact_ok})
                log.append(f'✓ candidate bend={bend} offset={offset} score={assoc_score:.4f}')
    if not scored_results:
        log.append('No matches found in the searched grid.')
        return '', '\n'.join(log), None
    scored_results.sort(key=lambda r: (r['exact_ok'], r['assoc_score'], len(r['matched_pairs'])), reverse=True)
    top_results = scored_results[: int(max_solutions)]
    parts = []
    for i, r in enumerate(top_results, 1):
        parts.append(f"=== MATCH {i} ===\n" f"bend = {r['bend']}   offset = {r['offset']}   vertex = {r['vertex']}\n" f"assoc_score = {r['assoc_score']:.4f}   exact_ok = {r['exact_ok']}\n" f"matched_pairs = {r['matched_pairs']}\n\n" f"{r['text']}\n")
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
    return '\n'.join(log), gr.update(value=config.get('pi_prec', DEFAULTS['PI_PREC'])), gr.update(value=config.get('pi_stream_len', DEFAULTS['PI_STREAM_LEN'])), gr.update(value=config.get('ngram_n', DEFAULTS['NGRAM_N'])), gr.update(value=config.get('lidstone_gamma', DEFAULTS['LIDSTONE_GAMMA']))


def load_hf_model_on_demand(repo_id, token=None):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        HF_CACHE['status'] = f'Loading {repo_id}…'
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, cache_dir=LOCAL_CACHE_DIR, token=token)
        mdl = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True, cache_dir=LOCAL_CACHE_DIR, token=token)
        HF_CACHE.update(loaded=True, tokenizer=tok, model=mdl, status=f'Loaded {repo_id}')
        return HF_CACHE['status']
    except Exception as e:
        HF_CACHE.update(loaded=False, tokenizer=None, model=None, status=f'Load failed: {e}')
        return HF_CACHE['status']


def save_hf_model(repo_id, token=None):
    try:
        from huggingface_hub import HfApi, upload_file
        if CACHE.get('cpd') is None or CACHE.get('stream') is None or CACHE.get('vocab') is None:
            corpus, _ = resolve_corpus(None, None)
            cpd, vocab, stream = get_or_build(corpus, DEFAULTS['NGRAM_N'], DEFAULTS['LIDSTONE_GAMMA'], DEFAULTS['PI_PREC'], DEFAULTS['PI_STREAM_LEN'], [])
        else:
            cpd, vocab, stream = CACHE['cpd'], CACHE['vocab'], CACHE['stream']
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl.gz', prefix='pi_model_')
        tmp.close()
        save_model_to_path(tmp.name, cpd, vocab, stream, DEFAULTS['NGRAM_N'], DEFAULTS['LIDSTONE_GAMMA'], DEFAULTS['PI_PREC'], DEFAULTS['PI_STREAM_LEN'], EMBEDDED_CORPUS)
        upload_file(path_or_fileobj=tmp.name, path_in_repo='pi_model.pkl.gz', repo_id=repo_id, repo_type='model', token=token)
        return f'Saved full model to {repo_id}/pi_model.pkl.gz'
    except Exception as e:
        return f'Save failed: {e}'


def load_hf_model_on_demand(repo_id, token=None):
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename='pi_model.pkl.gz', repo_type='model', token=token, cache_dir=LOCAL_CACHE_DIR)
        cpd, vocab, stream, config, errors = load_model_from_path(path)
        if cpd is None:
            return 'Load failed: ' + ' | '.join(errors)
        CACHE.update(key=('HF', repo_id), cpd=cpd, vocab=vocab, stream=stream)
        HF_CACHE.update(loaded=True, tokenizer=None, model=None, status=f'Loaded full model from {repo_id}')
        if config:
            return f'Loaded full model from {repo_id} with config {config}'
        return f'Loaded full model from {repo_id}'
    except Exception as e:
        return f'Load failed: {e}'


def run_generate(prompt, temperature, text_length):
    corpus = EMBEDDED_CORPUS
    cpd, vocab = build_model(corpus, DEFAULTS['NGRAM_N'], DEFAULTS['LIDSTONE_GAMMA'])
    stream = build_pi_stream(DEFAULTS['PI_PREC'], DEFAULTS['PI_STREAM_LEN'])
    sampler = PiSampler(stream, DEFAULTS['DIGITS_PER_SAMPLE'], temperature, DEFAULTS['TOP_K'], DEFAULTS['TOP_P'], DEFAULTS['REP_PENALTY'], DEFAULTS['SEASHELL_ENABLE'], DEFAULTS['SEASHELL_STRENGTH'], DEFAULTS['SEASHELL_DECAY'], DEFAULTS['SEASHELL_PEAKS'], DEFAULTS['SEASHELL_WIDTH'], DEFAULTS['SEASHELL_FLOOR'])
    sampler.seek(0)
    text = generate_text(cpd, sampler, prompt or '', int(text_length), DEFAULTS['NGRAM_N'], vocab)
    return text, f'Generated {len(text.split())} tokens.'


def build_ui():
    with gr.Blocks(title='Full features app') as demo:
        startup_status = gr.Markdown(f"### Loading model…\n\nCurrent status: `{HF_CACHE['status']}`")
        with gr.Tabs():
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
                btn = gr.Button('Generate', variant='primary')
                outtext = gr.Textbox(label='Generated text', lines=18)
                outlog = gr.Textbox(label='Log', lines=8)
                outfile = gr.File(label='Download output')
                btn.click(run_generate, inputs=[promptin, temperature, text_length], outputs=[outtext, outlog])
            with gr.TabItem('Search'):
                gr.Markdown('Prompt-aligned search with collocations.')
                search_prompt = gr.Textbox(label='Prompt', lines=3, value='alice rabbit hole')
                search_bend_max = gr.Slider(1.0, 90.0, value=DEFAULTS['BEND_MAX'], step=0.5, label='Bend max')
                search_bend_step = gr.Slider(0.1, 5.0, value=DEFAULTS['BEND_STEP'], step=0.1, label='Bend step')
                search_offset_step = gr.Slider(1, 1000, value=DEFAULTS['OFFSET_STEP'], step=1, label='Offset step')
                search_fuzzy = gr.Slider(0.0, 1.0, value=DEFAULTS['FUZZY_THRESHOLD'], step=0.01, label='Fuzzy threshold')
                search_max = gr.Slider(1, 25, value=DEFAULTS['MAX_SOLUTIONS'], step=1, label='Max solutions')
                search_btn = gr.Button('Run search', variant='primary')
                search_out = gr.Textbox(label='Search result', lines=16)
                search_log = gr.Textbox(label='Search log', lines=8)
                search_file = gr.File(label='Search output')
                search_btn.click(run_search, inputs=[filein, pasted, search_prompt, pi_prec, pi_stream_len, ngram_n, lidstone_gamma, text_length, temperature, rep_penalty, gr.Checkbox(value=DEFAULTS['SEASHELL_ENABLE'], visible=False), gr.Slider(visible=False, value=DEFAULTS['SEASHELL_STRENGTH']), gr.Slider(visible=False, value=DEFAULTS['SEASHELL_DECAY']), gr.Slider(visible=False, value=DEFAULTS['SEASHELL_PEAKS']), gr.Slider(visible=False, value=DEFAULTS['SEASHELL_WIDTH']), gr.Slider(visible=False, value=DEFAULTS['SEASHELL_FLOOR']), gr.Radio(choices=['A', 'B', 'C'], value=DEFAULTS['VERTEX'], visible=False), search_bend_max, search_bend_step, search_offset_step, search_fuzzy, search_max], outputs=[search_out, search_log, search_file])
            with gr.TabItem('Model I/O'):
                gr.Markdown('Save/load compiled trigram model.')
                save_btn = gr.Button('Save model', variant='primary')
                save_file = gr.File(label='Saved model file', interactive=False)
                model_log = gr.Textbox(label='Model I/O log', lines=8)
                load_file = gr.File(label='Load saved model .pkl.gz', file_types=['.gz', '.pkl'], type='filepath')
                load_btn = gr.Button('Load model', variant='secondary')
                save_btn.click(save_model_ui, inputs=[filein, pasted, pi_prec, pi_stream_len, ngram_n, lidstone_gamma], outputs=[save_file, model_log])
                load_btn.click(load_model_ui, inputs=[load_file], outputs=[model_log, pi_prec, pi_stream_len, ngram_n, lidstone_gamma])
            with gr.TabItem('Thinking-lite'):
                gr.Markdown('This tab is lazy. Use the buttons below to load or save.')
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
                hf_load_btn.click(load_hf_model_on_demand, inputs=[hf_repo, hf_token], outputs=[hf_log])
        gr.Markdown('Tip: model caches are reused until corpus or configuration changes.')
    return demo


if __name__ == '__main__':
    
    build_ui().queue(max_size=8).launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'), show_error=True)
