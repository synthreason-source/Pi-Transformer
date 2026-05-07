#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
π → BASE-26 → NLTK TRIGRAM LLM → NATURAL TEXT
═══════════════════════════════════════════════
  1. Corpus       — embedded public-domain prose, tokenised with NLTK
  2. Word list    — nltk.corpus.words
  3. LLM          — nltk ConditionalProbDist trigram model with Lidstone smoothing
  4. Pi entropy   — base-26 stream of π replaces all random sampling
  5. Triangle     — vertices A / B / C at 0 / ⅓ / ⅔ of the stream seed
                    three independently-reproducible texts
  6. Seed text    — user-provided phrase offsets the triangle and seeds the trigram context
  7. Dataset      — words found live in the stream + generated paragraphs
                    all written to pi_dataset.txt
"""

import sys, os, time
from collections import defaultdict, deque, Counter
from mpmath import mp, pi as mpi

import nltk
from nltk.util import ngrams
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words as nltk_words

import gradio as gr

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300_000)

R  = "\033[0m"
B  = "\033[1m"
DM = "\033[2m"
CY = "\033[96m"
GR = "\033[92m"
YL = "\033[93m"
RD = "\033[91m"
MG = "\033[95m"

def c(code, t): return f"{code}{t}{R}"

PI_PREC           = 15_000
PI_STREAM_LEN     = 12_000
DIGITS_PER_SAMPLE = 3
NGRAM_N           = 3
LIDSTONE_GAMMA    = 0.1
GEN_WORDS         = 120
WORD_FIND_MIN     = 4
DATASET_PATH      = "pi_dataset.txt"
CONTEXT_WINDOW    = 2
SEED_OFFSET       = 0

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
            print(f"  Upload read failed ({e}), falling back.")
    try:
        with open("xaa.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        pass
    return _embedded_corpus()

def _alpha_lower_chars(text: str):
    for ch in text:
        lc = ch.lower()
        if 'a' <= lc <= 'z':
            yield lc

def _tokenise_alpha(text: str):
    tokens = []
    current = []
    for ch in text:
        lc = ch.lower()
        if 'a' <= lc <= 'z':
            current.append(lc)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens

def _capitalise_after_period(words: list) -> str:
    text = " ".join(words)
    if not text:
        return text
    chars = list(text)
    i = 0
    while i < len(chars) - 2:
        if chars[i] == '.' and chars[i + 1] == ' ':
            j = i + 2
            if j < len(chars) and 'a' <= chars[j] <= 'z':
                chars[j] = chars[j].upper()
        i += 1
    result = "".join(chars)
    return result[0].upper() + result[1:] if result else result

def build_nltk_model(corpus: str):
    tokenizer = RegexpTokenizer(r"[a-z]+")
    tokens = tokenizer.tokenize(corpus.lower())
    padded = ["", ""] + tokens + [""]
    trigrams_ = list(ngrams(padded, NGRAM_N))
    cfd = ConditionalFreqDist((tuple(tg[:-1]), tg[-1]) for tg in trigrams_)
    cpd = ConditionalProbDist(cfd, LidstoneProbDist, LIDSTONE_GAMMA)
    vocab = set(tokens)
    return cpd, tokens, vocab

def load_nltk_words():
    try:
        word_set = set(
            w.lower() for w in nltk_words.words()
            if w.isalpha() and WORD_FIND_MIN <= len(w) <= 15
        )
        print(f"  {c(DM,'nltk.corpus.words')}  {len(word_set):,} words")
        return word_set
    except Exception as e:
        print(f"  nltk words failed ({e}), using corpus vocab")
        return set()

def seed_to_offset(seed: str, stream_len: int) -> int:
    h = 0
    for ch in _alpha_lower_chars(seed):
        h = (h * 31 + (ord(ch) - ord('a') + 1)) % stream_len
    return h

def seed_context(seed: str, default=("", ""), window=2):
    toks = _tokenise_alpha(seed)
    if window <= 0:
        return ()
    if len(toks) >= window:
        return tuple(toks[:window])
    if len(toks) > 0:
        pad = list(default[-(window - len(toks)):]) if default else [""] * (window - len(toks))
        return tuple(pad[: window - len(toks)] + toks)
    return tuple(default[:window]) if default else tuple("" for _ in range(window))

def build_pi_stream(n_decimal: int = PI_PREC, length: int = PI_STREAM_LEN):
    print(f"  Computing π to {n_decimal} decimal digits…", end=" ", flush=True)
    t0 = time.time()
    mp.dps = n_decimal + 60
    D = 10 ** n_decimal
    frac = int(mp.floor(mpi * D)) - 3 * D
    stream = []
    for _ in range(length):
        frac *= 26
        stream.append(frac // D)
        frac = frac % D
    print(f"done ({time.time()-t0:.1f}s)  {length:,} base-26 digits")
    return stream

class PiSampler:
    def __init__(self, stream: list, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
        self.stream = stream
        self.pos = 0
        self.temperature = max(1e-6, float(temperature))
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.min_p = float(min_p)

    def seek(self, pos: int):
        self.pos = pos % len(self.stream)

    def _next_unit(self) -> float:
        val = 0
        base = 26 ** DIGITS_PER_SAMPLE
        for _ in range(DIGITS_PER_SAMPLE):
            val = val * 26 + self.stream[self.pos % len(self.stream)]
            self.pos += 1
        return val / base

    def sample(self, prob_dist) -> str:
        samples = sorted(prob_dist.samples())
        if not samples:
            return ""
        probs = [max(0.0, float(prob_dist.prob(s))) for s in samples]
        if self.temperature != 1.0:
            probs = [p ** (1.0 / self.temperature) for p in probs]
        total = sum(probs) or 1.0
        probs = [p / total for p in probs]
        ranked = list(zip(samples, probs))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if self.min_p > 0.0 and ranked:
            best = ranked[0][1]
            ranked = [(s, p) for s, p in ranked if p >= best * self.min_p] or ranked[:1]
        if self.top_k > 0:
            ranked = ranked[:self.top_k]
        if 0.0 < self.top_p < 1.0:
            kept = []
            acc = 0.0
            for s, p in ranked:
                kept.append((s, p))
                acc += p
                if acc >= self.top_p:
                    break
            ranked = kept or ranked[:1]
        total = sum(p for _, p in ranked) or 1.0
        u = self._next_unit()
        cumulative = 0.0
        for s, p in ranked:
            cumulative += p / total
            if u < cumulative:
                return s
        return ranked[-1][0]

class Triangle:
    def __init__(self, stream_len: int, seed: str = "", offset_extra: int = 0):
        offset = (seed_to_offset(seed, stream_len) + int(offset_extra)) % stream_len if seed else int(offset_extra) % stream_len
        self.A = offset
        self.B = (offset + stream_len // 3) % stream_len
        self.C = (offset + 2 * stream_len // 3) % stream_len
        self.vertices = {"A": self.A, "B": self.B, "C": self.C}

    def zone(self, pos: int) -> str:
        if pos < self.B: return "α"
        if pos < self.C: return "β"
        return "γ"

    def energy(self, pos: int) -> float:
        return sum(1 / (abs(pos - v) + 1) for v in self.vertices.values())

def find_words_in_stream(stream: list, dictionary: set, triangle: Triangle):
    prefixes = set()
    for w in dictionary:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])
    buf = deque(maxlen=35)
    all_chars = []
    word_cat = defaultdict(list)
    seen_at = {}
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

def generate_text(cpd, sampler: PiSampler, n_words: int = GEN_WORDS, init_context: tuple = ("", ""), context_window: int = 2) -> str:
    context_window = max(1, int(context_window))
    init = list(init_context)[-context_window:]
    if len(init) < context_window:
        init = ([""] * (context_window - len(init))) + init
    context = deque(init, maxlen=context_window)
    words_out = []
    for _ in range(n_words):
        ctx = tuple(context)
        dist = cpd[ctx]
        if not dist.samples():
            context.clear()
            context.extend([""] * context_window)
            dist = cpd[tuple(context)]
        word = sampler.sample(dist)
        words_out.append(word)
        context.append(word)
    return _capitalise_after_period(words_out)

def write_dataset(stream_text: str, word_cat: dict, triangle: Triangle, generations: dict, path: str):
    lines = ["=== NLTK-GENERATED NATURAL TEXT ==="]
    for vertex, text in generations.items():
        lines.append(f"\n-- Vertex {vertex} --")
        for i in range(0, len(text), 80):
            lines.append(text[i:i+80])
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    kb = os.path.getsize(path) / 1024
    print(f"  {c(GR,'✓')}  {c(B, path)}  ({kb:.1f} KB,  {len(lines)} lines)")

def gradio_run(seed_phrase, uploaded_corpus, temperature, top_k, top_p, min_p,
               repetition_penalty, presence_penalty, frequency_penalty,
               digits_per_sample, gen_words, gamma, context_window, seed_offset):
    global DIGITS_PER_SAMPLE, GEN_WORDS, LIDSTONE_GAMMA
    DIGITS_PER_SAMPLE = int(digits_per_sample)
    GEN_WORDS = int(gen_words)
    LIDSTONE_GAMMA = float(gamma)
    try:
        corpus = _load_corpus(uploaded_corpus)
        dictionary = load_nltk_words()
        cpd, _, _ = build_nltk_model(corpus)
        stream = build_pi_stream()
        triangle = Triangle(len(stream), seed=seed_phrase, offset_extra=seed_offset)
        stream_text, word_cat = find_words_in_stream(stream, dictionary, triangle)
        sampler = PiSampler(stream, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
        ctx = seed_context(seed_phrase, window=int(context_window))
        generations = {}
        for vertex, start_pos in triangle.vertices.items():
            sampler.seek(start_pos)
            generations[vertex] = generate_text(
                cpd, sampler,
                n_words=GEN_WORDS,
                init_context=ctx,
                context_window=int(context_window),
            )
        write_dataset(stream_text, word_cat, triangle, generations, DATASET_PATH)
        out = [f'BASE-26 → NLTK TRIGRAM LLM  |  seed: "{seed_phrase}"\n']
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
        tri_hits = [w for w, ps in word_cat.items() if len({triangle.zone(p) for p in ps}) >= 2]
        out.append(f"Words found in stream : {len(word_cat):,}")
        out.append(f"Longest               : {longest}")
        out.append(f"Cross-zone hits       : {len(tri_hits)}")
        out.append(f"\nDataset written → {DATASET_PATH}")
        return "\n".join(out), DATASET_PATH
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}", None

CSS = "#output-box textarea { font-family: monospace; font-size: 13px; }"

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
            value="Is there inherent order in nature or is it all chaos and chance?",
            scale=3,
        )
        corpus_upload = gr.File(
            label="Corpus (optional)",
            file_types=[".txt", ".md", ".text"],
            type="filepath",
            scale=1,
        )

    with gr.Row():
        temperature = gr.Slider(0.1, 12.5, value=2.5, step=0.05, label="Temperature")
        top_k = gr.Slider(0, 1100, value=100, step=1, label="Top-k")
        top_p = gr.Slider(0.0, 11.0, value=1.0, step=0.01, label="Top-p")
        min_p = gr.Slider(0.0, 11.0, value=0.52, step=0.01, label="Min-p")

    with gr.Row():
        repetition_penalty = gr.Slider(1.0, 13.0, value=1.0, step=0.01, label="Repetition penalty")
        presence_penalty = gr.Slider(0.0, 12.0, value=1.21, step=0.01, label="Presence penalty")
        frequency_penalty = gr.Slider(0.0, 12.0, value=0.0, step=0.01, label="Frequency penalty")
        digits_per_sample = gr.Slider(1, 16, value=3, step=1, label="π digits per sample")

    with gr.Row():
        gen_words = gr.Slider(10, 1500, value=120, step=1, label="Generated words")
        gamma = gr.Slider(0.001, 11.0, value=0.994, step=0.1, label="Lidstone γ")
        context_window = gr.Slider(2, 18, value=2, step=1, label="Context window")
        seed_offset = gr.Slider(0, PI_STREAM_LEN - 1, value=11999, step=1, label="Seed offset")

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
        inputs=[
            seed_input, corpus_upload,
            temperature, top_k, top_p, min_p,
            repetition_penalty, presence_penalty, frequency_penalty,
            digits_per_sample, gen_words, gamma, context_window, seed_offset
        ],
        outputs=[output_text, dataset_file],
    )

    seed_input.submit(
        fn=gradio_run,
        inputs=[
            seed_input, corpus_upload,
            temperature, top_k, top_p, min_p,
            repetition_penalty, presence_penalty, frequency_penalty,
            digits_per_sample, gen_words, gamma, context_window, seed_offset
        ],
        outputs=[output_text, dataset_file],
    )

if __name__ == "__main__":
    try:
        demo.launch()
    except KeyboardInterrupt:
        print(f"\n{c(GR,'✅  Stopped.')}  Partial results saved to {DATASET_PATH}")
