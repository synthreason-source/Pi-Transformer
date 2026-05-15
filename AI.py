#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import math
from collections import deque, Counter
import tempfile
import gradio as gr
from mpmath import mp, pi as mpi

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist

NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300000)

DEFAULTS = dict(
    PI_PREC=15000,
    PI_STREAM_LEN=12000,
    DIGITS_PER_SAMPLE=3,
    NGRAM_N=2,
    LIDSTONE_GAMMA=0.1,
    REP_PENALTY=1.08,
    TEMPERATURE=2.5,
)

EMBEDDED_CORPUS = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it.
"""

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenise_alpha(text):
    if text is None:
        return []
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    elif not isinstance(text, str):
        text = str(text)
    return text.lower().split()

def build_model(corpus, ngram_n, lidstone_gamma):
    if isinstance(corpus, bytes):
        corpus = corpus.decode("utf-8", errors="ignore")
    elif not isinstance(corpus, str):
        corpus = str(corpus) if corpus is not None else ""
    ngram_n = max(2, int(ngram_n))
    tokens = tokenise_alpha(corpus)
    if not tokens:
        raise ValueError("Corpus produced zero tokens after tokenisation.")
    padded = ["<s>"] * (ngram_n - 1) + tokens + ["</s>"]
    trigs = list(ngrams(padded, ngram_n))
    cfd = ConditionalFreqDist((tuple(tg[:-1]), tg[-1]) for tg in trigs)
    vocab = set(tokens) | {"</s>"}
    class _LidstoneFactory:
        def __init__(self, gamma, bins):
            self.gamma = float(gamma)
            self.bins = max(1, int(bins))
        def __call__(self, fd):
            return LidstoneProbDist(fd, gamma=self.gamma, bins=self.bins)
    cpd = ConditionalProbDist(cfd, _LidstoneFactory(float(lidstone_gamma), max(1, len(vocab))))
    return cpd, vocab

def build_pi_stream(decimals, length):
    mp.dps = int(decimals) + 50
    D = 10 ** int(decimals)
    frac = int(mp.floor(mpi * D)) - 3 * D
    stream = []
    for _ in range(int(length)):
        frac *= 26
        stream.append(frac // D)
        frac %= D
    return stream

class PiSampler:
    def __init__(self, stream, digits_per_sample, temperature, repetition_penalty):
        self.stream = stream
        self.digits_per_sample = int(digits_per_sample)
        self.temperature = max(1e-3, float(temperature))
        self.repetition_penalty = max(1.0, float(repetition_penalty))
        self.pos = 0
        self.history = Counter()
    def seek(self, pos):
        self.pos = pos % len(self.stream)
        self.history.clear()
    def next_unit(self):
        val = 0
        base = 26 ** self.digits_per_sample
        for _ in range(self.digits_per_sample):
            val = val * 26 + self.stream[self.pos % len(self.stream)]
            self.pos += 1
        return val / base
    def sample(self, dist):
        samples = list(dist.samples())
        if not samples:
            return "</s>"
        scored = []
        for s in samples:
            p = max(1e-12, float(dist.prob(s)))
            if self.history[s] > 0:
                p /= self.repetition_penalty ** self.history[s]
            scored.append((s, p ** (1.0 / self.temperature)))
        total = sum(p for _, p in scored)
        scored = [(s, p / total) for s, p in scored]
        scored.sort(key=lambda x: x[1], reverse=True)
        x = self.next_unit()
        cum = 0.0
        for w, p in scored:
            cum += p
            if x < cum:
                self.history[w] += 1
                return w
        self.history[scored[-1][0]] += 1
        return scored[-1][0]

def generate_text(cpd, sampler, prompt, n_words, ngram_n, vocab=None):
    context_window = ngram_n - 1
    seed = tokenise_alpha(prompt)
    seed_in_vocab = [w for w in seed if vocab is None or w in vocab]
    init = seed_in_vocab[-context_window:] if len(seed_in_vocab) >= context_window else ["<s>"] * (context_window - len(seed_in_vocab)) + seed_in_vocab
    context = deque(init, maxlen=context_window)
    words = list(seed)
    def dist_for_ctx(ctxtuple):
        for cut in range(len(ctxtuple), 0, -1):
            trial = ("<s>",) * (context_window - cut) + ctxtuple[-cut:]
            try:
                d = cpd[trial]
                if list(d.samples()):
                    return d
            except Exception:
                pass
        try:
            d = cpd[tuple(["<s>"] * context_window)]
            if list(d.samples()):
                return d
        except Exception:
            pass
        return None
    for _ in range(int(n_words)):
        dist = dist_for_ctx(tuple(context))
        if dist is None:
            context.clear(); context.extend(["<s>"] * context_window); continue
        w = sampler.sample(dist)
        if w == "</s>":
            context.clear(); context.extend(["<s>"] * context_window); continue
        words.append(w)
        context.append(w)
    return " ".join(words)

def read_corpus(file_obj, pasted_corpus):
    if file_obj is not None:
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if txt.strip():
                return txt, f"file:{os.path.basename(path)}"
        except Exception as e:
            return EMBEDDED_CORPUS, f"embedded fallback (file read failed: {e})"
    if pasted_corpus and pasted_corpus.strip():
        return pasted_corpus, "pasted text"
    return EMBEDDED_CORPUS, "embedded fallback"

def run_generate(file_obj, pasted_corpus, prompt, temperature, text_length):
    corpus, source = read_corpus(file_obj, pasted_corpus)
    cpd, vocab = build_model(corpus, DEFAULTS["NGRAM_N"], DEFAULTS["LIDSTONE_GAMMA"])
    stream = build_pi_stream(DEFAULTS["PI_PREC"], DEFAULTS["PI_STREAM_LEN"])
    sampler = PiSampler(stream, DEFAULTS["DIGITS_PER_SAMPLE"], temperature, DEFAULTS["REP_PENALTY"])
    sampler.seek(0)
    text = generate_text(cpd, sampler, prompt or "", int(text_length), DEFAULTS["NGRAM_N"], vocab)
    log = f"Corpus source: {source}\nGenerated {len(text.split())} tokens."
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="generated_", mode="w", encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return text, log, tmp.name

def build_ui():
    with gr.Blocks(title="Minimal generator") as demo:
        gr.Markdown("# π → base-26 → NLTK trigram generator")
        filein = gr.File(label="Upload corpus (.txt)", file_types=[".txt", ".md"], type="filepath")
        pasted = gr.Textbox(label="or paste corpus here", lines=6)
        promptin = gr.Textbox(label="Prompt", lines=3, value="alice rabbit hole")
        temperature = gr.Slider(0.1, 5.0, value=2.5, step=0.05, label="Temperature")
        text_length = gr.Slider(1, 2000, value=400, step=1, label="Text length")
        btn = gr.Button("Generate", variant="primary")
        outtext = gr.Textbox(label="Generated text", lines=18)
        outlog = gr.Textbox(label="Log", lines=4)
        outfile = gr.File(label="Download output")
        btn.click(run_generate, inputs=[filein, pasted, promptin, temperature, text_length], outputs=[outtext, outlog, outfile])
    return demo

if __name__ == "__main__":
    build_ui().queue(max_size=8).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), show_error=True)
