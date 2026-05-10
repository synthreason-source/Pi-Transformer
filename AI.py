#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
π → BASE-26 → NLTK TRIGRAM GENERATOR
════════════════════════════════════

PURE TERMINAL VERSION
NO GRADIO

PROMPT-SEEDED SEARCH MODE
-------------------------
1. User enters a prompt
2. Prompt becomes:
   - trigram seed context
   - search target
   - beginning of generated text
3. Prompt is converted into ordered word pairs
4. Brute-force search scans:
   - bend_degrees
   - stream offsets
5. Generated text must contain ALL prompt pairs
6. Exact + fuzzy matching supported

FEATURES
--------
1. Embedded / external corpus
2. NLTK trigram language model
3. π base-26 entropy stream
4. Deterministic sampling
5. Bent-triangle vertex mapping
6. Prompt-aligned generation
7. Exact + fuzzy matching
8. Dataset export
"""

import sys
import re
from collections import defaultdict, deque, Counter
from difflib import SequenceMatcher

from mpmath import mp, pi as mpi

import nltk
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.probability import (
    ConditionalFreqDist,
    ConditionalProbDist,
    LidstoneProbDist,
)
from nltk.corpus import words as nltk_words


# ============================================================
# CONFIG
# ============================================================

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300_000)

PI_PREC = 15000
PI_STREAM_LEN = 12000

DIGITS_PER_SAMPLE = 3

NGRAM_N = 3
CONTEXT_WINDOW = NGRAM_N - 1

LIDSTONE_GAMMA = 0.1

GEN_WORDS = 160

WORD_FIND_MIN = 2

DATASET_PATH = "pi_dataset.txt"


# ============================================================
# EMBEDDED CORPUS
# ============================================================

def embedded_corpus():
    return """
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

def tokenise_alpha(text):

    tokenizer = RegexpTokenizer(r"[a-z]+")

    return tokenizer.tokenize(text.lower())


def extract_word_pairs(prompt):

    words = [
        w.lower()
        for w in word_tokenize(prompt)
        if w.isalpha()
    ]

    return list(ngrams(words, 2))


def capitalise_text(words):

    if not words:
        return ""

    txt = " ".join(words)

    chars = list(txt)

    if chars:
        chars[0] = chars[0].upper()

    for i in range(len(chars) - 2):

        if chars[i] == "." and chars[i + 1] == " ":
            chars[i + 2] = chars[i + 2].upper()

    txt = "".join(chars)

    txt = re.sub(
        r'([.!?])\s*([A-Z])',
        r'\1\n\n\2',
        txt
    )

    return txt


# ============================================================
# MODEL
# ============================================================

def build_model(corpus):

    tokens = tokenise_alpha(corpus)

    padded = (
        ["<s>"] * (NGRAM_N - 1)
        + tokens
        + ["</s>"]
    )

    trigrams_ = list(ngrams(padded, NGRAM_N))

    cfd = ConditionalFreqDist(
        (tuple(tg[:-1]), tg[-1])
        for tg in trigrams_
    )

    vocab = set(tokens) | {"</s>"}

    for ctx in list(cfd.conditions()):

        if len(cfd[ctx]) == 0:
            cfd[ctx]["</s>"] += 1

    cpd = ConditionalProbDist(
        cfd,
        lambda fd: LidstoneProbDist(
            fd,
            gamma=LIDSTONE_GAMMA,
            bins=max(1, len(vocab))
        )
    )

    return cpd, vocab


# ============================================================
# WORD LIST
# ============================================================

def load_dictionary(vocab):

    try:

        words = {
            w.lower()
            for w in nltk_words.words()
            if w.isalpha()
        }

        return words | set(vocab)

    except Exception:

        return set(vocab)


# ============================================================
# PI STREAM
# ============================================================

def build_pi_stream(
    decimals=PI_PREC,
    length=PI_STREAM_LEN,
):

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
# PI SAMPLER
# ============================================================

class PiSampler:

    def __init__(
        self,
        stream,
        temperature=2.5,
        top_k=100,
        top_p=1.0,
        repetition_penalty=1.08,
    ):

        self.stream = stream

        self.pos = 0

        self.temperature = temperature

        self.top_k = top_k

        self.top_p = top_p

        self.repetition_penalty = repetition_penalty

        self.history = Counter()

    def seek(self, pos):

        self.pos = pos % len(self.stream)

        self.history.clear()

    def next_unit(self):

        val = 0

        base = 26 ** DIGITS_PER_SAMPLE

        for _ in range(DIGITS_PER_SAMPLE):

            val = (
                val * 26
                + self.stream[self.pos % len(self.stream)]
            )

            self.pos += 1

        return val / base

    def sample(self, dist):

        samples = list(dist.samples())

        if not samples:
            return "</s>"

        scored = []

        for s in samples:

            p = max(1e-12, float(dist.prob(s)))

            count = self.history[s]

            if count > 0:
                p /= (
                    self.repetition_penalty ** count
                )

            scored.append((s, p))

        scored = [
            (s, p ** (1.0 / self.temperature))
            for s, p in scored
        ]

        total = sum(p for _, p in scored)

        scored = [
            (s, p / total)
            for s, p in scored
        ]

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        scored = scored[:self.top_k]

        kept = []

        accum = 0.0

        for s, p in scored:

            kept.append((s, p))

            accum += p

            if accum >= self.top_p:
                break

        scored = kept

        total = sum(p for _, p in scored)

        u = self.next_unit()

        cumulative = 0.0

        for s, p in scored:

            cumulative += p / total

            if u < cumulative:

                self.history[s] += 1

                return s

        chosen = scored[-1][0]

        self.history[chosen] += 1

        return chosen


# ============================================================
# TRIANGLE
# ============================================================

class Triangle:

    def __init__(
        self,
        stream_len,
        offset_extra=0,
        bend_degrees=13.0,
    ):

        base = offset_extra % stream_len

        bend_shift = int(
            round(
                (bend_degrees / 360.0)
                * stream_len
            )
        )

        self.A = base % stream_len

        self.B = (
            base
            + stream_len // 3
            + bend_shift
        ) % stream_len

        self.C = (
            base
            + 2 * stream_len // 3
            + bend_shift
        ) % stream_len

        self.vertices = {
            "A": self.A,
            "B": self.B,
            "C": self.C,
        }


# ============================================================
# GENERATION
# ============================================================

def generate_text(
    cpd,
    sampler,
    prompt="",
    n_words=GEN_WORDS,
):

    seed_words = tokenise_alpha(prompt)

    if len(seed_words) >= CONTEXT_WINDOW:

        init = seed_words[-CONTEXT_WINDOW:]

    else:

        init = (
            ["<s>"] * (
                CONTEXT_WINDOW - len(seed_words)
            )
            + seed_words
        )

    context = deque(
        init,
        maxlen=CONTEXT_WINDOW,
    )

    words = list(seed_words)

    for _ in range(n_words):

        ctx = tuple(context)

        try:

            dist = cpd[ctx]

            samples = list(dist.samples())

        except Exception:

            samples = []

        if not samples:

            context.clear()

            context.extend(
                ["<s>"] * CONTEXT_WINDOW
            )

            continue

        word = sampler.sample(dist)

        if word in ("</s>", ""):

            context.clear()

            context.extend(
                ["<s>"] * CONTEXT_WINDOW
            )

            continue

        words.append(word)

        context.append(word)

    return capitalise_text(words)


# ============================================================
# FIND WORDS
# ============================================================

def find_words(stream, dictionary):

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

        for length in range(
            WORD_FIND_MIN,
            min(16, len(s)) + 1,
        ):

            cand = s[-length:]

            if cand not in prefixes:
                continue

            if cand in dictionary:

                found[cand].append(
                    pos - length + 1
                )

    return "".join(all_chars), found


# ============================================================
# FUZZY SCORE
# ============================================================

def fuzzy_score(target, text):

    return SequenceMatcher(
        None,
        target.lower(),
        text.lower()
    ).quick_ratio()


# ============================================================
# PAIR MATCHING
# ============================================================

def all_pairs_match(
    pairs,
    text,
    fuzzy_threshold=0.72,
):

    lower_text = text.lower()

    for pair in pairs:

        pair_str = " ".join(pair)

        exact = pair_str in lower_text

        if exact:
            continue

        score = fuzzy_score(
            pair_str,
            text
        )

        if score < fuzzy_threshold:
            return False, pair

    return True, None


# ============================================================
# PROMPT SEARCH
# ============================================================

def brute_force_prompt_search(
    prompt,
    cpd,
    stream,
    vertex="A",
    max_solutions=10,
):

    pairs = extract_word_pairs(prompt)

    if not pairs:

        print(
            "No valid word pairs extracted."
        )

        return []

    print(
        "\nSearching bend+offset space..."
    )

    print(f"\nPrompt:\n{prompt}\n")

    found = []

    for bend_x10 in range(0, 451, 5):

        bend = bend_x10 / 10.0

        print(f"bend = {bend:.1f}")

        for offset in range(
            0,
            PI_STREAM_LEN,
            5,
        ):

            triangle = Triangle(
                PI_STREAM_LEN,
                offset_extra=offset,
                bend_degrees=bend,
            )

            start = triangle.vertices[vertex]

            sampler = PiSampler(stream)

            sampler.seek(start)

            text = generate_text(
                cpd,
                sampler,
                prompt=prompt,
                n_words=GEN_WORDS,
            )

            matches_all, failed_pair = (
                all_pairs_match(
                    pairs,
                    text,
                )
            )

            if matches_all:

                found.append(
                    {
                        "prompt": prompt,
                        "bend": bend,
                        "offset": offset,
                        "vertex": vertex,
                        "text": text,
                    }
                )

                print("\nFOUND MATCH")

                print(
                    f"bend={bend:.1f} "
                    f"offset={offset}"
                )

                print("\nGenerated:\n")

                print(text)

                print()

                if (
                    len(found)
                    >= max_solutions
                ):
                    return found

    return found


# ============================================================
# MAIN
# ============================================================

def main():

    filename = input(
        "Corpus filename "
        "(ENTER for embedded corpus): "
    ).strip()

    if filename:

        with open(
            filename,
            "r",
            encoding="utf-8",
        ) as f:

            corpus = f.read()

    else:

        corpus = embedded_corpus()

    print("\nBuilding trigram model...")

    cpd, vocab = build_model(corpus)

    dictionary = load_dictionary(vocab)

    print("Building pi stream...")

    stream = build_pi_stream()

    print("Finding words in stream...")

    stream_text, found_words = find_words(
        stream,
        dictionary,
    )

    while True:

        print("\n==========================")
        print("PROMPT-ALIGNED SEARCH")
        print("==========================")

        prompt = input(
            "\nEnter prompt:\n> "
        ).strip()

        if not prompt:
            continue

        results = brute_force_prompt_search(
            prompt=prompt,
            cpd=cpd,
            stream=stream,
            vertex="A",
        )

        print("\n====================")
        print("SEARCH RESULTS")
        print("====================\n")

        if not results:

            print("No matches found.")

        else:

            for i, r in enumerate(results, 1):

                print(
                    f"[{i}] "
                    f"bend={r['bend']:.1f} "
                    f"offset={r['offset']} "
                    f"vertex={r['vertex']}"
                )

                print("\nGenerated response:\n")

                print(r["text"])

                print(
                    "\n"
                    + "=" * 50
                    + "\n"
                )

        again = input(
            "\nSearch again? (y/n): "
        ).strip().lower()

        if again != "y":
            break


if __name__ == "__main__":
    main()