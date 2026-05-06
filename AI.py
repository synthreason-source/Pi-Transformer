#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
π → BASE-26 → NLTK TRIGRAM LLM → NATURAL TEXT
═══════════════════════════════════════════════
  1. Corpus       — embedded public-domain prose, tokenised with NLTK
  2. Word list    — nltk.corpus.words  (75 k English words)
  3. LLM          — nltk ConditionalProbDist trigram model with Lidstone smoothing
  4. Pi entropy   — base-26 stream of π replaces all random sampling
  5. Triangle     — vertices A / B / C at 0 / ⅓ / ⅔ of the stream seed
                    three independently-reproducible texts
  6. Seed text    — user‑provided phrase offsets the triangle and seeds the trigram context
  7. Dataset      — words found live in the stream + generated paragraphs
                    all written to pi_dataset.txt
"""

import sys, os, re, math, json, time
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

# ── ANSI ──────────────────────────────────────────────────────────────────────
R = "\033[0m"   # reset
B = "\033[1m"   # bold
DM = "\033[2m"  # dim
CY = "\033[96m" # cyan
GR = "\033[92m" # green
YL = "\033[93m" # yellow
RD = "\033[91m" # red
MG = "\033[95m" # magenta

def c(code, t): return f"{code}{t}{R}"

# ── CONFIG ────────────────────────────────────────────────────────────────────
PI_PREC        = 15_000   # decimal digits of π
PI_STREAM_LEN  = 12_000   # base-26 chars extracted
DIGITS_PER_SAMPLE = 3     # base-26 digits consumed per word choice (26³=17576 bins)
NGRAM_N        = 3        # trigram model
LIDSTONE_GAMMA = 0.1      # smoothing
GEN_WORDS      = 120      # words per generated paragraph
WORD_FIND_MIN  = 4        # minimum word length to report from stream
DATASET_PATH   = "pi_dataset.txt"

# ── PUBLIC-DOMAIN CORPUS ──────────────────────────────────────────────────────
def _load_embedded_corpus():
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
    """
    Priority order:
      1. Gradio-uploaded file (filepath string or file-like)
      2. xaa.txt on disk
      3. Embedded Alice fallback
    """
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

    return _load_embedded_corpus()


# ═════════════════════════════════════════════════════════════════════════════
# 1. NLTK SETUP
# ═════════════════════════════════════════════════════════════════════════════

def build_nltk_model(corpus: str):
    """Tokenise corpus and build an NLTK trigram conditional probability model."""
    tokenizer = RegexpTokenizer(r"[a-z]+")
    tokens    = tokenizer.tokenize(corpus.lower())

    # Build trigram (context = prev 2 words → next word)
    pad = ["", ""]
    padded = pad + tokens + [""]
    trigrams = list(ngrams(padded, NGRAM_N))

    # ConditionalFreqDist: context (w1,w2) → FreqDist over w3
    cfd = ConditionalFreqDist(
        (tuple(tg[:-1]), tg[-1]) for tg in trigrams
    )

    vocab = set(tokens)
    bins  = len(vocab) + 1   # ensures Lidstone always has >=1 bin
    cpd   = ConditionalProbDist(cfd, LidstoneProbDist, LIDSTONE_GAMMA, bins)

    # Vocabulary from corpus + NLTK word list
    return cpd, tokens, vocab


def load_nltk_words():
    """Load word list from nltk.corpus.words (built from hunspell)."""
    try:
        word_set = set(w.lower() for w in nltk_words.words()
                       if w.isalpha() and WORD_FIND_MIN <= len(w) <= 15)
        print(f"  {c(DM,'nltk.corpus.words')}  {len(word_set):,} words")
        return word_set
    except Exception as e:
        print(f"  nltk words failed ({e}), using corpus vocab")
        return set()


# ── SEED HELPERS ─────────────────────────────────────────────────────────────
def seed_to_offset(seed: str, stream_len: int) -> int:
    """Deterministically map a seed string to an integer in [0, stream_len)."""
    h = 0
    for ch in seed.lower():
        if ch.isalpha():
            h = (h * 31 + (ord(ch) - ord('a') + 1)) % stream_len
    return h

def seed_context(seed: str, default=("", "")):
    """Return a two‑word tuple to prime the trigram model."""
    toks = [w for w in re.findall(r"[a-z]+", seed.lower()) if w]
    if len(toks) >= 2:
        return (toks[0], toks[1])
    if len(toks) == 1:
        return (toks[0], default[1])
    return default

# ═════════════════════════════════════════════════════════════════════════════
# 2. π BASE-26 STREAM
# ═════════════════════════════════════════════════════════════════════════════

def build_pi_stream(n_decimal: int = PI_PREC, length: int = PI_STREAM_LEN):
    """
    Compute π fractional digits via mpmath and convert to a list of
    base-26 integers (0-25).
    """
    print(f"  Computing π to {n_decimal} decimal digits…", end=" ", flush=True)
    t0 = time.time()
    mp.dps   = n_decimal + 50
    pi_str   = mp.nstr(mpi, n_decimal, strip_zeros=False).replace(".", "")
    D        = 10 ** (len(pi_str) - 1)
    frac     = int(pi_str[1:])
    stream   = []
    for _ in range(length):
        frac *= 26
        stream.append(frac // D)
        frac  = frac % D
    print(f"done ({time.time()-t0:.1f}s)  {length:,} base-26 digits")
    return stream


# ═════════════════════════════════════════════════════════════════════════════
# 3. π-ENTROPY SAMPLER
# ═════════════════════════════════════════════════════════════════════════════

class PiSampler:
    """
    Wraps the pre‑computed base-26 stream and exposes sample().
    Call seek(pos) to jump to a triangle vertex before generation.
    """
    def __init__(self, stream: list):
        self.stream = stream
        self.pos    = 0

    def seek(self, pos: int):
        self.pos = pos % len(self.stream)

    def _next_unit(self) -> float:
        """Consume DIGITS_PER_SAMPLE digits → float in [0, 1)."""
        val  = 0
        base = 26 ** DIGITS_PER_SAMPLE
        for _ in range(DIGITS_PER_SAMPLE):
            val = val * 26 + self.stream[self.pos % len(self.stream)]
            self.pos += 1
        return val / base

    def sample(self, prob_dist) -> str:
        """
        Sample one outcome from an NLTK ProbDist using π as entropy.
        """
        u = self._next_unit()
        cumulative = 0.0
        samples = prob_dist.samples()
        for outcome in sorted(samples):
            cumulative += prob_dist.prob(outcome)
            if u < cumulative:
                return outcome
        return sorted(samples)[-1]


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRIANGLE REFERENCE (with seed‑driven offset)
# ═════════════════════════════════════════════════════════════════════════════

class Triangle:
    """
    Three vertices evenly spaced across the π stream, optionally shifted
    by a seed‑derived offset so the whole triangle moves together.
    """
    def __init__(self, stream_len: int, seed: str = ""):
        offset = seed_to_offset(seed, stream_len) if seed else 0
        base_a = offset
        base_b = (offset + stream_len // 3) % stream_len
        base_c = (offset + 2 * stream_len // 3) % stream_len
        self.A = base_a
        self.B = base_b
        self.C = base_c
        self.vertices = {"A": self.A, "B": self.B, "C": self.C}

    def zone(self, pos: int) -> str:
        if pos < self.B: return "α"
        if pos < self.C: return "β"
        return "γ"

    def energy(self, pos: int) -> float:
        return sum(1 / (abs(pos - v) + 1) for v in self.vertices.values())


# ═════════════════════════════════════════════════════════════════════════════
# 5. REAL-TIME WORD FINDER (from raw π stream)
# ═════════════════════════════════════════════════════════════════════════════

def find_words_in_stream(stream: list, dictionary: set, triangle: Triangle):
    """
    Walk the pre‑computed base-26 stream, convert digits → letters,
    and detect English words in real time (ending-at-current-position).
    Returns the letter string and word catalogue.
    """
    prefixes = set()
    for w in dictionary:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])

    buf       = deque(maxlen=15 + 20)
    all_chars = []
    word_cat  = defaultdict(list)   # word → [positions]
    seen_at   = {}                  # start_pos → longest length found there
    row       = []

    for pos, digit in enumerate(stream):
        ch = chr(ord('a') + digit)
        buf.append(ch)
        all_chars.append(ch)
        row.append(ch.upper())

        buf_str = "".join(buf)
        buf_len = len(buf_str)

        for length in range(WORD_FIND_MIN, min(15, buf_len) + 1):
            start_buf    = buf_len - length
            candidate    = buf_str[start_buf:]
            if candidate not in prefixes:
                continue
            if candidate not in dictionary:
                continue
            global_start = pos - length + 1
            if seen_at.get(global_start, 0) >= length:
                continue
            seen_at[global_start] = length
            word_cat[candidate].append(global_start)

            zone   = triangle.zone(global_start)
            energy = triangle.energy(global_start)
            lo     = max(0, start_buf - 4)
            ctx    = buf_str[lo:buf_len].upper()
            rel    = start_buf - lo
            ctx_hi = ctx[:rel] + c(GR+B, f"[{ctx[rel:rel+length]}]") + ctx[rel+length:]
            global_start_str = c(CY, f'{global_start:>6}')
            candidate_str    = c(YL, f'{candidate:<16}')

    return "".join(all_chars), word_cat


# ═════════════════════════════════════════════════════════════════════════════
# 6. NATURAL TEXT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_text(cpd, sampler: PiSampler,
                  n_words: int = GEN_WORDS,
                  init_context: tuple = ("", "")) -> str:
    """
    Generate natural text using the NLTK trigram CPD.
    """
    context = init_context
    words_out = []

    for _ in range(n_words):
        dist = cpd[context]
        if not dist.samples():
            context = ("", "")
            dist    = cpd[context]

        word = sampler.sample(dist)

        words_out.append(word)
        context = (context[1], word)

    # ── tidy into sentences ──
    text = " ".join(words_out)
    text = re.sub(r"\. (\w)", lambda m: ". " + m.group(1).upper(), text)
    return text[0].upper() + text[1:] if text else text


# ═════════════════════════════════════════════════════════════════════════════
# 7. DATASET WRITER
# ═════════════════════════════════════════════════════════════════════════════

def write_dataset(stream_text: str, word_cat: dict,
                  triangle: Triangle, generations: dict, path: str):
    lines = []
    lines.append("=== NLTK-GENERATED NATURAL TEXT ===")
    for vertex, text in generations.items():
        lines.append(f"\n-- Vertex {vertex} --")
        for i in range(0, len(text), 80):
            lines.append(text[i:i+80])
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    kb = os.path.getsize(path) / 1024
    print(f"  {c(GR,'✓')}  {c(B, path)}  ({kb:.1f} KB,  {len(lines)} lines)")


# ═════════════════════════════════════════════════════════════════════════════
# 8. CORE RUNNER  (shared by CLI and Gradio)
# ═════════════════════════════════════════════════════════════════════════════

def run_core(SEED_PHRASE: str, uploaded_file=None,
             pi_prec: int = PI_PREC,
             stream_len: int = PI_STREAM_LEN,
             gen_words: int = GEN_WORDS,
             gamma: float = LIDSTONE_GAMMA,
             word_min: int = WORD_FIND_MIN,
             digits_per_sample: int = DIGITS_PER_SAMPLE):
    """
    All original logic from the while-True loop, unchanged.
    uploaded_file: filepath string from Gradio upload (or None).
    """
    global LIDSTONE_GAMMA, WORD_FIND_MIN, DIGITS_PER_SAMPLE
    LIDSTONE_GAMMA    = gamma
    WORD_FIND_MIN     = word_min
    DIGITS_PER_SAMPLE = digits_per_sample

    corpus = _load_corpus(uploaded_file)

    print(c(B, "📖  Loading NLTK word list…"), flush=True)
    dictionary = load_nltk_words()

    cpd, corpus_tokens, vocab = build_nltk_model(corpus)

    stream   = build_pi_stream(n_decimal=pi_prec, length=stream_len)
    triangle = Triangle(len(stream), seed=SEED_PHRASE)

    stream_text, word_cat = find_words_in_stream(stream, dictionary, triangle)

    print(c(B, "\n✨  Generating natural text from each triangle vertex:\n"))
    sampler     = PiSampler(stream)
    generations = {}

    ctx = seed_context(SEED_PHRASE)

    for vertex, start_pos in triangle.vertices.items():
        sampler.seek(start_pos)
        text = generate_text(cpd, sampler, n_words=gen_words, init_context=ctx)
        generations[vertex] = text

        zone = triangle.zone(start_pos)
        words_in_text = text.split()
        line, lines = [], []
        for w in words_in_text:
            line.append(w)
            if sum(len(x)+1 for x in line) > 72:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            print(f"    {ln}")
        print()

    print(c(B, "💾  Writing dataset…"), flush=True)
    write_dataset(stream_text, word_cat, triangle, generations, DATASET_PATH)

    by_len = defaultdict(list)
    for w in word_cat:
        by_len[len(w)].append(w)
    tri_hits = [w for w, ps in word_cat.items()
                if len({triangle.zone(p) for p in ps}) >= 2]
    longest  = max(word_cat, key=lambda w: len(w)) if word_cat else "—"

    return generations, triangle, word_cat, longest, tri_hits, corpus


# ═════════════════════════════════════════════════════════════════════════════
# 9. GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════

def gradio_run(seed_phrase, uploaded_corpus,
               pi_prec, stream_len, gen_words,
               gamma, word_min, digits_per_samp):
    """Gradio callback — calls run_core() and formats output."""
    try:
        generations, triangle, word_cat, longest, tri_hits, corpus = run_core(
            SEED_PHRASE     = seed_phrase,
            uploaded_file   = uploaded_corpus,
            pi_prec         = int(pi_prec),
            stream_len      = int(stream_len),
            gen_words       = int(gen_words),
            gamma           = float(gamma),
            word_min        = int(word_min),
            digits_per_sample = int(digits_per_samp),
        )
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}", None

    # ── build output text ──
    out = []
    out.append(f'BASE-26 → NLTK TRIGRAM LLM  |  seed: "{seed_phrase}"')
    corpus_src = "uploaded file" if uploaded_corpus else ("xaa.txt" if os.path.exists("xaa.txt") else "embedded Alice corpus")
    out.append(f"Corpus source: {corpus_src}  ({len(corpus):,} chars)\n")

    out.append("△  Triangle vertices")
    for name, pos in triangle.vertices.items():
        out.append(f"   {name} = position {pos:>6}  (zone {triangle.zone(pos)})")
    out.append("")

    out.append("✨  Generated text from each triangle vertex:\n")
    for vertex, text in generations.items():
        out.append(f"── Vertex {vertex} ──")
        words_in_text = text.split()
        line, lines = [], []
        for w in words_in_text:
            line.append(w)
            if sum(len(x)+1 for x in line) > 72:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        out.extend(lines)
        out.append("")

    out.append(f"📖  Words found in stream : {len(word_cat):,}")
    out.append(f"    Longest               : {longest}")
    out.append(f"    Cross-zone hits       : {len(tri_hits)}")
    out.append(f"\n💾  Dataset written → {DATASET_PATH}")

    return "\n".join(out), DATASET_PATH


with gr.Blocks(title="π → BASE-26 → NLTK Trigram Generator") as demo:
    gr.Markdown("# π → BASE-26 → NLTK Trigram Generator")
    gr.Markdown(
        "Converts π digits into a base-26 stream, then uses an NLTK trigram LLM "
        "to generate reproducible natural text from three triangle vertices.\n\n"
        "**Corpus priority:** uploaded file → `xaa.txt` on disk → embedded Alice excerpt."
    )

    with gr.Row():
        seed_phrase = gr.Textbox(label="Seed phrase", value="hello world",
                                 placeholder='try "alice rabbit", "quantum", …')
        uploaded_corpus = gr.File(
            label="Upload corpus (.txt / .md / .text) — optional",
            file_types=[".txt", ".md", ".text"],
            type="filepath",
        )

    with gr.Row():
        pi_prec    = gr.Slider(2_000,  30_000, value=PI_PREC,        step=500,  label="π decimal precision")
        stream_len = gr.Slider(1_000,  50_000, value=PI_STREAM_LEN,  step=1000, label="Base-26 stream length")

    with gr.Row():
        gen_words       = gr.Slider(20,   300,  value=GEN_WORDS,         step=10,   label="Words per vertex")
        gamma           = gr.Slider(0.01, 1.0,  value=LIDSTONE_GAMMA,    step=0.01, label="Lidstone gamma")

    with gr.Row():
        word_min        = gr.Slider(2,    12,   value=WORD_FIND_MIN,     step=1,    label="Min word length for stream search")
        digits_per_samp = gr.Slider(1,    5,    value=DIGITS_PER_SAMPLE, step=1,    label="π digits per sample")

    run_btn      = gr.Button("▶  Generate", variant="primary")
    output_text  = gr.Textbox(label="Output", lines=30)
    dataset_file = gr.File(label="💾 Download pi_dataset.txt")

    run_btn.click(
        fn=gradio_run,
        inputs=[seed_phrase, uploaded_corpus,
                pi_prec, stream_len, gen_words,
                gamma, word_min, digits_per_samp],
        outputs=[output_text, dataset_file],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═════════════════════════════════════════════════════════════════════════════



if __name__ == "__main__":
    demo.launch()
   
