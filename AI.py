"""
π → BASE-26 → NLTK TRIGRAM LLM → NATURAL TEXT
═══════════════════════════════════════════════
  1. Corpus       — embedded public-domain prose, tokenised with NLTK
  2. Word list    — nltk.corpus.words  (75 k English words)
  3. LLM          — nltk ConditionalProbDist trigram model with Lidstone smoothing
  4. Pi entropy   — base-26 stream of π replaces all random sampling
  5. Triangle     — vertices A / B / C at 0 / ⅓ / ⅔ of the stream seed
                    three independently-reproducible texts
  6. Dataset      — words found live in the stream + generated paragraphs
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
R="\033[0m"; B="\033[1m"; DM="\033[2m"
CY="\033[96m"; GR="\033[92m"; YL="\033[93m"; RD="\033[91m"; MG="\033[95m"
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
# Multiple genres; all pre-1928 or clearly public domain.
file = open("xaa.txt", "r")

# Read the entire content of the file
CORPUS = file.read()

# Close the file
file.close()
# ═════════════════════════════════════════════════════════════════════════════
# 1. NLTK SETUP
# ═════════════════════════════════════════════════════════════════════════════

def build_nltk_model(corpus: str):
    """Tokenise corpus and build an NLTK trigram conditional probability model."""
    tokenizer = RegexpTokenizer(r"[a-z]+")
    tokens    = tokenizer.tokenize(corpus.lower())

    # Build trigram (context = prev 2 words → next word)
    pad = ["<s>", "<s>"]
    padded = pad + tokens + ["</s>"]
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
    Wraps the pre-computed base-26 stream and exposes sample().
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
# 4. TRIANGLE REFERENCE
# ═════════════════════════════════════════════════════════════════════════════

class Triangle:
    """
    Three vertices evenly spaced across the π stream.
    Each vertex is a starting position for PiSampler → different text.
    """
    def __init__(self, stream_len: int):
        self.A = 0
        self.B = stream_len // 3
        self.C = 2 * stream_len // 3
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
    Walk the pre-computed base-26 stream, convert digits → letters,
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

    print(c(B, "\n🔍  Streaming π base-26 — words surface in real time:\n"))
    print(c(DM, f"  {'POS':>6}  {'WORD':<16} {'LEN':>3}  ZONE  ENERGY  CONTEXT"))
    print(c(DM, "  " + "─" * 66))

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
    return "".join(all_chars), word_cat


# ═════════════════════════════════════════════════════════════════════════════
# 6. NATURAL TEXT GENERATION
#    NLTK trigram model + PiSampler → reproducible natural sentences
# ═════════════════════════════════════════════════════════════════════════════

def generate_text(cpd, sampler: PiSampler, n_words: int = GEN_WORDS) -> str:
    """
    Generate natural text using the NLTK trigram CPD.
    Word choices are driven entirely by PiSampler (no random).
    """
    context = ("<s>", "<s>")
    words_out = []

    for _ in range(n_words):
        dist = cpd[context]
        if not dist.samples():
            context = ("<s>", "<s>")
            dist    = cpd[context]

        word = sampler.sample(dist)

        if word in ("</s>", "<s>"):
            context = ("<s>", "<s>")
            words_out.append(".")
            continue

        words_out.append(word)
        context = (context[1], word)

    # ── tidy into sentences ──
    text = " ".join(words_out)
    # Capitalise after full stop
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
# 8. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{c(B,'═'*66)}")
    print(f"  {c(B+CY,'π → BASE-26 → NLTK TRIGRAM LLM → NATURAL TEXT')}")
    print(f"{c(B,'═'*66)}\n")

    # ── NLTK word list (dictionary for stream scanning) ──
    print(c(B, "📖  Loading NLTK word list…"), flush=True)
    dictionary = load_nltk_words()

    # ── NLTK language model from embedded corpus ──
    print(c(B, "\n📚  Building NLTK trigram model from corpus…"), flush=True)
    cpd, corpus_tokens, vocab = build_nltk_model(CORPUS)
    print(f"  {c(DM,'corpus')}  {len(corpus_tokens):,} tokens  "
          f"|  vocab {len(vocab):,}  "
          f"|  contexts {len(cpd.conditions()):,}")

    # ── π base-26 stream ──
    print(c(B, "\n🔢  Computing π stream…"), flush=True)
    stream   = build_pi_stream()
    triangle = Triangle(len(stream))

    print(f"  {c(B,'△  Triangle vertices')}")
    for name, pos in triangle.vertices.items():
        print(f"     {c(CY,name)} = position {pos:>6}  (zone {triangle.zone(pos)})")

    # ── real-time word finding ──
    stream_text, word_cat = find_words_in_stream(stream, dictionary, triangle)

    # ── generate natural text from each triangle vertex ──
    print(c(B, "\n✨  Generating natural text from each triangle vertex:\n"))
    sampler     = PiSampler(stream)
    generations = {}

    for vertex, start_pos in triangle.vertices.items():
        sampler.seek(start_pos)
        text = generate_text(cpd, sampler, n_words=GEN_WORDS)
        generations[vertex] = text

        zone = triangle.zone(start_pos)
        print(f"  {c(B+YL, f'Vertex {vertex}')}  "
              f"{c(DM, f'stream pos {start_pos}  zone {zone}')}\n")
        # wrap at 72 chars
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

    # ── write dataset ──
    print(c(B, "💾  Writing dataset…"), flush=True)
    write_dataset(stream_text, word_cat, triangle, generations, DATASET_PATH)

    # ── summary ──
    by_len = defaultdict(list)
    for w in word_cat:
        by_len[len(w)].append(w)
    tri_hits = [w for w, ps in word_cat.items()
                if len({triangle.zone(p) for p in ps}) >= 2]
    longest  = max(word_cat, key=lambda w: len(w)) if word_cat else "—"


    print(f"\n{c(B,'═'*66)}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{c(GR,'✅  Stopped.')}  Partial results saved to {DATASET_PATH}")
