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
# Try to read an external file; if it doesn't exist, fall back to an embedded
# public‑domain excerpt (the opening of Alice's Adventures in Wonderland).
def _load_corpus():
    try:
        with open("xaa.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Embedded fallback – public domain
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

CORPUS = _load_corpus()

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
    cpd = ConditionalProbDist(cfd, LidstoneProbDist, LIDSTONE_GAMMA)

    # Vocabulary from corpus + NLTK word list
    vocab = set(tokens)
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
    base-26 integers (0-25).  Stored as a list so we can seek to any
    triangle vertex without recomputing.
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
    return stream          # list of ints 0-25


# ═════════════════════════════════════════════════════════════════════════════
# 3. π-ENTROPY SAMPLER
#    Uses DIGITS_PER_SAMPLE consecutive base-26 digits to produce one
#    index into a probability distribution — no random module needed.
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
        Walks the cumulative distribution until the π-unit is consumed.
        """
        u = self._next_unit()
        cumulative = 0.0
        samples = prob_dist.samples()
        # Sort for determinism
        for outcome in sorted(samples):
            cumulative += prob_dist.prob(outcome)
            if u < cumulative:
                return outcome
        return sorted(samples)[-1]   # fallback: last sample


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
    COLS      = 60

    #print(c(B, "\n🔍  Streaming π base-26 — words surface in real time:\n"))
    #print(c(DM, f"  {'POS':>6}  {'WORD':<16} {'LEN':>3}  ZONE  ENERGY  CONTEXT"))
    #print(c(DM, "  " + "─" * 66))

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
            # Avoid f‑string nesting by building parts separately
            global_start_str = c(CY, f'{global_start:>6}')
            candidate_str    = c(YL, f'{candidate:<16}')
            #print(f"{global_start_str}  {candidate_str} {len(candidate):>3}  {zone}  {energy:5.2f}  {ctx_hi}")

    return "".join(all_chars), word_cat


# ═════════════════════════════════════════════════════════════════════════════
# 6. NATURAL TEXT GENERATION
#    NLTK trigram model + PiSampler → reproducible natural sentences
# ═════════════════════════════════════════════════════════════════════════════

def generate_text(cpd, sampler: PiSampler,
                  n_words: int = GEN_WORDS,
                  init_context: tuple = ("", "")) -> str:
    """
    Generate natural text using the NLTK trigram CPD.
    The first trigram state is taken from `init_context`; thereafter
    the model proceeds as usual.
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
    # Capitalise after full stop
    text = re.sub(r"\\. (\\w)", lambda m: ". " + m.group(1).upper(), text)
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
# 8. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # 👇  USER‑DEFINED SEED – change this to any phrase you like

    while True:

        SEED_PHRASE = input("USER: ")   # ← try "", "hello world", etc.


        print(f"BASE-26 → NLTK TRIGRAM LLM → NATURAL TEXT'")

        print(c(B, "📖  Loading NLTK word list…"), flush=True)
        dictionary = load_nltk_words()

        #print(c(B, "\n📚  Building NLTK trigram model from corpus…"), flush=True)
        cpd, corpus_tokens, vocab = build_nltk_model(CORPUS)
        

        #print(c(B, "\n🔢  Computing π stream…"), flush=True)
        stream   = build_pi_stream()
        triangle = Triangle(len(stream), seed=SEED_PHRASE)

        # Print triangle vertices info
        msg = f'△  Triangle vertices (seed="{SEED_PHRASE}")'
        #print(f"  {c(B, msg)}")
        #for name, pos in triangle.vertices.items():
            #print(f"     {c(CY,name)} = position {pos:>6}  (zone {triangle.zone(pos)})")

        stream_text, word_cat = find_words_in_stream(stream, dictionary, triangle)

        print(c(B, "\n✨  Generating natural text from each triangle vertex:\n"))
        sampler     = PiSampler(stream)
        generations = {}

        ctx = seed_context(SEED_PHRASE)

        for vertex, start_pos in triangle.vertices.items():
            sampler.seek(start_pos)
            text = generate_text(cpd, sampler, n_words=GEN_WORDS, init_context=ctx)
            generations[vertex] = text

            zone = triangle.zone(start_pos)
            #print(f"  {c(B+YL, f'Vertex {vertex}')}  "
                  #f"{c(DM, f'stream pos {start_pos}  zone {zone}')}\n")
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

        #print(f"\n{c(B,'═'*66)}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{c(GR,'✅  Stopped.')}  Partial results saved to {DATASET_PATH}")
