#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
π → BASE-26 → NLTK TRIGRAM LLM → NATURAL TEXT
═══════════════════════════════════════════════
  Gradio UI — minimal, matching the CLI experience.
"""

import sys, os, re, time
from collections import defaultdict, deque
from mpmath import mp, pi as mpi

import nltk
from nltk.util import ngrams
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words as nltk_words

import gradio as gr

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300_000)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PI_PREC           = 15_000
PI_STREAM_LEN     = 12_000
DIGITS_PER_SAMPLE = 3
NGRAM_N           = 3
LIDSTONE_GAMMA    = 0.1
GEN_WORDS         = 120
WORD_FIND_MIN     = 4
DATASET_PATH      = "pi_dataset.txt"

# ── CORPUS ────────────────────────────────────────────────────────────────────
def _embedded_corpus():
    return (
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do: once or twice she had peeped into the book her "
        "sister was reading, but it had no pictures or conversations in it, \"and what "
        "is the use of a book,\" thought Alice \"without pictures or conversation?\"\n\n"
        "So she was considering in her own mind (as well as she could, for the hot "
        "day made her feel very sleepy and stupid), whether the pleasure of making "
        "a daisy-chain would be worth the trouble of getting up and picking the "
        "daisies, when suddenly a White Rabbit with pink eyes ran close by her.\n\n"
        "There was nothing so very remarkable in that; nor did Alice think it so "
        "very much out of the way to hear the Rabbit say to itself, \"Oh dear! Oh "
        "dear! I shall be late!\" (when she thought it over afterwards, it occurred "
        "to her that she ought to have wondered at this, but at the time it all "
        "seemed quite natural); but when the Rabbit actually took a watch out of "
        "its waistcoat-pocket, and looked at it, and then hurried on, Alice started "
        "to her feet, for it flashed across her mind that she had never before "
        "seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, "
        "and burning with curiosity, she ran across the field after it, and fortunately "
        "was just in time to see it pop down a large rabbit-hole under the hedge.\n\n"
        "In another moment down went Alice after it, never once considering how in "
        "the world she was to get out again.\n\n"
        "The rabbit-hole went straight on like a tunnel for some way, and then "
        "dipped suddenly down, so suddenly that Alice had not a moment to think "
        "about stopping herself before she found herself falling down a very deep well.\n\n"
        "Either the well was very deep, or she fell very slowly, for she had plenty "
        "of time as she went down to look about her and to wonder what was going "
        "to happen next. First, she tried to look down and make out what she was "
        "coming to, but it was too dark to see anything; then she looked at the "
        "sides of the well, and noticed that they were filled with cupboards and "
        "book-shelves; here and there she saw maps and pictures hung upon pegs. "
        "She took down a jar from one of the shelves as she passed; it was labelled "
        "\"ORANGE MARMALADE\", but to her great disappointment it was empty: she did "
        "not like to drop the jar for fear of killing somebody, so managed to put "
        "it into one of the cupboards as she fell past it."
    )

def _load_corpus(uploaded_file=None):
    if uploaded_file is not None:
        path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text.strip():
                return text
        except Exception as e:
            print(f"Upload read failed ({e}), falling back.")
    try:
        with open("xaa.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        pass
    return _embedded_corpus()

# ═══════════════════════════════════════════════════════════════════════════
# NLTK MODEL
# ═══════════════════════════════════════════════════════════════════════════

def build_nltk_model(corpus: str):
    tokenizer = RegexpTokenizer(r"[a-z]+")
    tokens    = tokenizer.tokenize(corpus.lower())
    pad       = ["", ""]
    padded    = pad + tokens + [""]
    tgrams    = list(ngrams(padded, NGRAM_N))
    cfd       = ConditionalFreqDist((tuple(tg[:-1]), tg[-1]) for tg in tgrams)
    vocab     = set(tokens)
    bins      = len(vocab) + 1
    cpd       = ConditionalProbDist(cfd, LidstoneProbDist, LIDSTONE_GAMMA, bins)
    return cpd, tokens, vocab

def load_nltk_words():
    try:
        return set(
            w.lower() for w in nltk_words.words()
            if w.isalpha() and WORD_FIND_MIN <= len(w) <= 15
        )
    except Exception:
        return set()

# ═══════════════════════════════════════════════════════════════════════════
# SEED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def seed_to_offset(seed: str, stream_len: int) -> int:
    h = 0
    for ch in seed.lower():
        if ch.isalpha():
            h = (h * 31 + (ord(ch) - ord('a') + 1)) % stream_len
    return h

def seed_context(seed: str, default=("", "")):
    toks = [w for w in re.findall(r"[a-z]+", seed.lower()) if w]
    if len(toks) >= 2: return (toks[0], toks[1])
    if len(toks) == 1: return (toks[0], default[1])
    return default

# ═══════════════════════════════════════════════════════════════════════════
# π STREAM
# ═══════════════════════════════════════════════════════════════════════════

def build_pi_stream(n_decimal=PI_PREC, length=PI_STREAM_LEN):
    # Extra guard digits ensure floor() sees the true value, not a rounded one.
    mp.dps = n_decimal + 60

    # Exact integer arithmetic from here on:
    #   D    = 10^n_decimal  (Python int — infinite precision)
    #   frac = floor(π × D) − 3×D  →  the fractional digits of π as an integer
    # No string conversion, no rounding, no floats in the loop.
    D    = 10 ** n_decimal
    frac = int(mp.floor(mpi * D)) - 3 * D

    stream = []
    for _ in range(length):
        frac *= 26
        stream.append(frac // D)   # pure Python integer division — exact
        frac  = frac % D
    return stream

# ═══════════════════════════════════════════════════════════════════════════
# SAMPLER
# ═══════════════════════════════════════════════════════════════════════════

class PiSampler:
    def __init__(self, stream):
        self.stream = stream
        self.pos    = 0

    def seek(self, pos):
        self.pos = pos % len(self.stream)

    def _next_unit(self):
        val  = 0
        base = 26 ** DIGITS_PER_SAMPLE
        for _ in range(DIGITS_PER_SAMPLE):
            val = val * 26 + self.stream[self.pos % len(self.stream)]
            self.pos += 1
        return val / base

    def sample(self, prob_dist):
        u = self._next_unit()
        cumulative = 0.0
        for outcome in sorted(prob_dist.samples()):
            cumulative += prob_dist.prob(outcome)
            if u < cumulative:
                return outcome
        return sorted(prob_dist.samples())[-1]

# ═══════════════════════════════════════════════════════════════════════════
# TRIANGLE
# ═══════════════════════════════════════════════════════════════════════════

class Triangle:
    def __init__(self, stream_len, seed=""):
        offset   = seed_to_offset(seed, stream_len) if seed else 0
        self.A   = offset
        self.B   = (offset + stream_len // 3) % stream_len
        self.C   = (offset + 2 * stream_len // 3) % stream_len
        self.vertices = {"A": self.A, "B": self.B, "C": self.C}

    def zone(self, pos):
        if pos < self.B: return "α"
        if pos < self.C: return "β"
        return "γ"

# ═══════════════════════════════════════════════════════════════════════════
# WORD FINDER
# ═══════════════════════════════════════════════════════════════════════════

def find_words_in_stream(stream, dictionary, triangle):
    prefixes = set()
    for w in dictionary:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])

    buf      = deque(maxlen=35)
    all_chars = []
    word_cat  = defaultdict(list)
    seen_at   = {}

    for pos, digit in enumerate(stream):
        ch = chr(ord('a') + digit)
        buf.append(ch)
        all_chars.append(ch)
        buf_str = "".join(buf)
        buf_len = len(buf_str)

        for length in range(WORD_FIND_MIN, min(15, buf_len) + 1):
            start_buf = buf_len - length
            candidate = buf_str[start_buf:]
            if candidate not in prefixes:
                continue
            if candidate not in dictionary:
                continue
            global_start = pos - length + 1
            if seen_at.get(global_start, 0) >= length:
                continue
            seen_at[global_start] = length
            word_cat[candidate].append(global_start)

    return "".join(all_chars), word_cat

# ═══════════════════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_text(cpd, sampler, n_words=GEN_WORDS, init_context=("", "")):
    context   = init_context
    words_out = []
    for _ in range(n_words):
        dist = cpd[context]
        if not dist.samples():
            context = ("", "")
            dist    = cpd[context]
        word = sampler.sample(dist)
        words_out.append(word)
        context = (context[1], word)
    text = " ".join(words_out)
    text = re.sub(r"\. (\w)", lambda m: ". " + m.group(1).upper(), text)
    return text[0].upper() + text[1:] if text else text

# ═══════════════════════════════════════════════════════════════════════════
# DATASET WRITER
# ═══════════════════════════════════════════════════════════════════════════

def write_dataset(stream_text, word_cat, triangle, generations, path=DATASET_PATH):
    lines = ["=== NLTK-GENERATED NATURAL TEXT ==="]
    for vertex, text in generations.items():
        lines.append(f"\n-- Vertex {vertex} --")
        for i in range(0, len(text), 80):
            lines.append(text[i:i+80])
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

# ═══════════════════════════════════════════════════════════════════════════
# GRADIO CALLBACK
# ═══════════════════════════════════════════════════════════════════════════

def gradio_run(seed_phrase, uploaded_corpus):
    try:
        corpus     = _load_corpus(uploaded_corpus)
        dictionary = load_nltk_words()
        cpd, _, _  = build_nltk_model(corpus)
        stream     = build_pi_stream()
        triangle   = Triangle(len(stream), seed=seed_phrase)

        stream_text, word_cat = find_words_in_stream(stream, dictionary, triangle)

        sampler     = PiSampler(stream)
        ctx         = seed_context(seed_phrase)
        generations = {}

        for vertex, start_pos in triangle.vertices.items():
            sampler.seek(start_pos)
            generations[vertex] = generate_text(
                cpd, sampler, n_words=GEN_WORDS, init_context=ctx
            )

        dataset_path = write_dataset(stream_text, word_cat, triangle, generations)

        # ── format output like the CLI ──
        out = []
        out.append(f'BASE-26 → NLTK TRIGRAM LLM  |  seed: "{seed_phrase}"\n')

        for vertex, text in generations.items():
            out.append(f"── Vertex {vertex} ──")
            words = text.split()
            line, lines = [], []
            for w in words:
                line.append(w)
                if sum(len(x) + 1 for x in line) > 72:
                    lines.append(" ".join(line))
                    line = []
            if line:
                lines.append(" ".join(line))
            out.extend(f"    {ln}" for ln in lines)
            out.append("")

        longest = max(word_cat, key=lambda w: len(w)) if word_cat else "—"
        tri_hits = [w for w, ps in word_cat.items()
                    if len({triangle.zone(p) for p in ps}) >= 2]

        out.append(f"Words found in stream : {len(word_cat):,}")
        out.append(f"Longest               : {longest}")
        out.append(f"Cross-zone hits       : {len(tri_hits)}")
        out.append(f"\nDataset written → {dataset_path}")

        return "\n".join(out), dataset_path

    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}", None


# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
#output-box textarea { font-family: monospace; font-size: 13px; }
"""

with gr.Blocks(title="π → BASE-26 → NLTK Trigram Generator", css=CSS) as demo:

    gr.Markdown("## π → BASE-26 → NLTK Trigram Generator")
    gr.Markdown(
        "Enter a seed phrase and hit **Generate**. "
        "Three texts are produced from triangle vertices in the π base-26 stream.\n\n"
        "Optionally upload a `.txt` corpus — otherwise falls back to `xaa.txt` "
        "or the embedded *Alice* excerpt."
    )

    with gr.Row():
        seed_input = gr.Textbox(
            label="Seed phrase",
            placeholder='e.g. "hello world", "quantum", ""',
            value="hello world",
            scale=3,
        )
        corpus_upload = gr.File(
            label="Corpus (optional)",
            file_types=[".txt", ".md", ".text"],
            type="filepath",
            scale=1,
        )

    run_btn = gr.Button("▶  Generate", variant="primary")

    output_text = gr.Textbox(
        label="Output",
        lines=28,
        elem_id="output-box",
        interactive=False,
    )
    dataset_file = gr.File(label="💾  Download pi_dataset.txt")

    run_btn.click(
        fn=gradio_run,
        inputs=[seed_input, corpus_upload],
        outputs=[output_text, dataset_file],
    )

    seed_input.submit(
        fn=gradio_run,
        inputs=[seed_input, corpus_upload],
        outputs=[output_text, dataset_file],
    )


if __name__ == "__main__":
    demo.launch()
