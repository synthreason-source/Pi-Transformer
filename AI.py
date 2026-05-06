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
CORPUS = """
Alice was beginning to get very tired of sitting by her sister on the bank and of having nothing
to do once or twice she had peeped into the book her sister was reading but it had no pictures or
conversations in it and what is the use of a book without pictures or conversations so she was
considering in her own mind whether the pleasure of making a daisy chain would be worth the trouble
of getting up and picking the daisies when suddenly a white rabbit with pink eyes ran close by her
there was nothing so very remarkable in that nor did Alice think it so very much out of the way to
hear the rabbit say to itself oh dear oh dear I shall be too late but when the rabbit actually took
a watch out of its waistcoat pocket and looked at it and then hurried on Alice started to her feet
for it flashed across her mind that she had never before seen a rabbit with either a waistcoat pocket
or a watch to take out of it and burning with curiosity she ran across the field after it.
Call me Ishmael some years ago never mind how long precisely having little money in my purse and
nothing particular to interest me on shore I thought I would sail about a little and see the watery
part of the world it is a way I have of driving off the spleen and regulating the circulation
whenever I find myself growing grim about the mouth whenever it is a damp drizzly November in my
soul whenever I find myself involuntarily pausing before coffin warehouses and bringing up the rear
of every funeral I meet and especially whenever my hypos get such an upper hand of me that it
requires a strong moral principle to prevent me from deliberately stepping into the street and
methodically knocking peoples hats off then I account it high time to get to sea as soon as I can
this is my substitute for pistol and ball with a philosophical flourish Cato throws himself upon
his sword I quietly take to the ship there is nothing surprising in this if they only knew it
almost all men in their degree some time or other cherish very nearly the same feelings towards
the ocean with me.
It is a truth universally acknowledged that a single man in possession of a good fortune must be
in want of a wife however little known the feelings or views of such a man may be on his first
entering a neighbourhood this truth is so well fixed in the minds of the surrounding families
that he is considered as the rightful property of some one or other of their daughters my dear
Mr Bennet said his lady to him one day have you heard that Netherfield Park is let at last
Mr Bennet replied that he had not but Mrs Bennet was not so easily silenced and insisted on his
listening to what she had to say a single man of large fortune four or five thousand a year what
a fine thing for our girls do you not think so my dear replied Mr Bennet how so can it affect them
my dear you must know that I am thinking of his marrying one of them.
Marley was dead to begin with there is no doubt whatever about that the register of his burial was
signed by the clergyman the clerk the undertaker and the chief mourner Scrooge signed it and
Scrooges name was good upon Change for anything he chose to put his hand to old Marley was as dead
as a door nail mind I do not mean to say that I know of my own knowledge what there is particularly
dead about a door nail I might have been inclined myself to regard a coffin nail as the deadest
piece of ironmongery in the trade but the wisdom of our ancestors is in the simile and my unhallowed
hands shall not disturb it or the countrys done for you will therefore permit me to repeat
emphatically that Marley was as dead as a door nail.
The old man was thin and gaunt with deep wrinkles in the back of his neck the brown blotches of
the benevolent skin cancer the sun brings from its reflection on the tropic sea were on his cheeks
the blotches ran well down the sides of his face and his hands had the deep creased scars from
handling heavy fish on the cords but none of these scars were fresh they were as old as erosions
in a fishless desert everything about him was old except his eyes and they were the same color as
the sea and were cheerful and undefeated the old man was thin and gaunt with deep wrinkles in the
back of his neck he looked at the old man sleeping in the chair and at the brown sacks that held
his gear the wind was steady and the sea was calm and beautiful in the early morning light.
Whether I shall turn out to be the hero of my own life or whether that station will be held by
anybody else these pages must show to begin my life with the beginning of my life I record that
I was born as I have been informed and believe on a Friday at twelve o clock at night it was
remarked that the clock began to strike and I began to cry simultaneously both of which events
I am informed took place in the same moment of time whether I had any knowledge of it at the
time is a matter of no consequence since I was not there to observe it and indeed I am far from
sure that I should have been able to form any opinion upon the subject even if I had been present.
The sea was calm and the morning air was fresh and the light fell golden on the ancient stones
of the harbor and the boats moved gently on the quiet water and the birds called in the distance
and the sound of the waves was soft and regular like breathing and the world seemed at peace
with itself as if the long night of troubles had finally passed and a new day was beginning with
all the promise of clear skies and steady winds and good fortune for those who sailed the deep.
In the forests of the night the great creatures moved silently through the dark spaces between
the trees where no light fell and the only sound was the soft press of heavy feet on the ancient
earth and the breathing of large bodies and the occasional snap of a branch and the world was
old and dark and full of things that had no names and the night stretched out endlessly in all
directions without a single star to give hope or direction to the wandering traveler who might
find himself alone in such a place far from any road or house or fire.
The winter was long and cold and the snow lay deep across the fields and the road was lost under
the white and the trees were black against the pale sky and the wind moved through the bare
branches with a sound like distant voices and the house was warm inside with the fire burning
and the smell of bread and the sound of quiet conversation and the evening coming down outside
like a dark curtain drawn across the world to keep in the warmth and the light and the company
of those who had gathered there against the cold and the dark and the silence of the winter.
She had been told many times that the city was dangerous and that a young woman alone should not
walk through its streets at night but she had never believed it and she walked now quickly along
the wet pavement with her coat pulled close and her head down and the lights of the shops
reflecting in the puddles and the sound of traffic all around her and the feeling of being
utterly alone in the midst of all these people who passed her without looking up as if she were
not there at all or were perhaps a ghost of some former self walking the same streets she had
walked a hundred times before in a different life when things had been simpler and the world had
seemed a more reasonable and comprehensible place to live in.
The storm came in from the west with great clouds piling up on the horizon and the light turning
yellow and strange before the darkness arrived and the rain began to fall in heavy drops that
rattled on the leaves and the wind came suddenly and bent the long grass flat and the trees
swayed and creaked and the thunder rolled across the sky from one end to the other and the
lightning showed everything in a cold white light for an instant before the darkness returned
and the rain fell harder and the river began to rise and the fields were full of running water.
"""

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
