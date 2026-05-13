#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
π → BASE-26 → NLTK TRIGRAM GENERATOR
════════════════════════════════════

PURE TERMINAL VERSION + GRADIO DEMO

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
9. Seashell-style resonant probability coloration
10. Gradio demo
11. File upload
12. Hugging Face dataset support with config
"""

import sys
import re
import math
import os
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

import gradio as gr
from datasets import load_dataset

nltk.download('punkt_tab')
# ============================================================
# CONFIG
# ============================================================

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300_000)

PI_PREC = 15000
PI_STREAM_LEN = 12000

DIGITS_PER_SAMPLE = 3

NGRAM_N = 4
CONTEXT_WINDOW = NGRAM_N - 1

LIDSTONE_GAMMA = 0.1

GEN_WORDS = 1600

WORD_FIND_MIN = 4

DATASET_PATH = "pi_dataset.txt"

# Seashell resonance config
SEASHELL_ENABLE = True
SEASHELL_STRENGTH = 4.35
SEASHELL_DECAY = 0.985
SEASHELL_PEAKS = 4
SEASHELL_WIDTH = 0.16
SEASHELL_FLOOR = 0.35


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
# SEASHELL RESONATOR
# ============================================================

class SeashellResonator:
    """
    Simulates seashell-like cavity coloration by boosting/attenuating
    ranked candidate probabilities with a few decaying resonant peaks.
    """

    def __init__(
        self,
        sampler,
        strength=SEASHELL_STRENGTH,
        decay=SEASHELL_DECAY,
        peaks=SEASHELL_PEAKS,
        width=SEASHELL_WIDTH,
        floor=SEASHELL_FLOOR,
    ):
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

                ripple = 0.5 + 0.5 * math.cos(
                    (d / max(1e-9, spread)) * math.pi * (1.5 + shimmer)
                    + phase
                    + 0.13 * t
                )

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

        weighted = []
        for (word, p), gain in zip(scored, g):
            weighted.append((word, p * gain))

        total = sum(p for _, p in weighted)
        if total > 0:
            weighted = [(w, p / total) for w, p in weighted]

        self.energy *= self.decay
        self.step_index += 1

        if self.energy < 0.08:
            self.energy = 1.0
            self._seed_from_stream()

        return weighted


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
        seashell_enable=SEASHELL_ENABLE,
        seashell_strength=SEASHELL_STRENGTH,
        seashell_decay=SEASHELL_DECAY,
        seashell_peaks=SEASHELL_PEAKS,
        seashell_width=SEASHELL_WIDTH,
        seashell_floor=SEASHELL_FLOOR,
    ):
        self.stream = stream
        self.pos = 0
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.history = Counter()
        self.seashell = None

        if seashell_enable:
            self.seashell = SeashellResonator(
                sampler=self,
                strength=seashell_strength,
                decay=seashell_decay,
                peaks=seashell_peaks,
                width=seashell_width,
                floor=seashell_floor,
            )

    def seek(self, pos):
        self.pos = pos % len(self.stream)
        self.history.clear()
        if self.seashell is not None:
            self.seashell.reset()

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

    def _xor_probability_fusion(self, scored, u_a, u_b, u_c):
        """XOR fusion: Creates mutually exclusive probability regions"""
        xor_scores = []

        u_a_norm = u_a
        u_b_norm = u_b
        u_c_norm = u_c

        for rank, (word, base_p) in enumerate(scored):
            idx = rank / max(1, len(scored) - 1)

            region_a = (1.0 - abs(idx - u_a_norm)) * (1.0 - u_b_norm) * (1.0 - u_c_norm)
            region_b = u_b_norm * (1.0 - abs(idx - u_a_norm)) * (1.0 - u_c_norm)
            region_c = u_c_norm * (1.0 - u_a_norm) * (1.0 - u_b_norm)

            xor_blend = max(region_a, region_b, region_c)

            orthogonality = 1.0 - abs(u_a_norm - u_b_norm) * abs(u_b_norm - u_c_norm)
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
                p /= (self.repetition_penalty ** count)
            base_scored.append((s, p))

        scored = [
            (s, p ** (1.0 / self.temperature))
            for s, p in base_scored
        ]

        total = sum(p for _, p in scored)
        scored = [
            (s, p / total)
            for s, p in scored
        ]

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:self.top_k]

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
            xor_scored = [
                (w, p / xor_total) for w, p in xor_scored
            ]

            xor_draw = (
                u_a * (1 - u_b) * (1 - u_c) +
                u_b * (1 - u_a) * (1 - u_c) +
                u_c * (1 - u_a) * (1 - u_b)
            ) / 1.5

            cumulative = 0.0
            chosen = xor_scored[-1][0]
            for word, p in xor_scored:
                cumulative += p
                if xor_draw < cumulative:
                    chosen = word
                    break

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
# DATASET HELPERS
# ============================================================

def dataset_rows_to_text(ds, max_rows=5000):
    texts = []
    count = 0

    for row in ds:
        parts = []
        for v in row.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, (int, float, bool)):
                parts.append(str(v))
            elif isinstance(v, list):
                parts.append(" ".join(str(x) for x in v))
            elif isinstance(v, dict):
                parts.append(" ".join(f"{a}:{b}" for a, b in v.items()))

        row_text = " ".join(parts).strip()
        if row_text:
            texts.append(row_text)

        count += 1
        if count >= max_rows:
            break

    return "\n".join(texts)


def load_corpus_from_source(
    source_mode,
    uploaded_file=None,
    hf_dataset_name="",
    hf_config_name="",
    hf_split="train",
    hf_text_field="",
):
    if source_mode == "embedded":
        return embedded_corpus(), "embedded"

    if source_mode == "file":
        if not uploaded_file:
            raise ValueError("No file uploaded.")

        file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".txt", ".md", ".py", ".log", ".text"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), f"file:{os.path.basename(file_path)}"

        if ext == ".json":
            ds = load_dataset("json", data_files=file_path, split="train")
        elif ext == ".csv":
            ds = load_dataset("csv", data_files=file_path, split="train")
        elif ext == ".parquet":
            ds = load_dataset("parquet", data_files=file_path, split="train")
        elif ext == ".txt":
            ds = load_dataset("text", data_files=file_path, split="train")
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), f"file:{os.path.basename(file_path)}"

        if hf_text_field and hf_text_field in ds.column_names:
            text = "\n".join(str(x) for x in ds[hf_text_field] if x is not None)
        else:
            text = dataset_rows_to_text(ds)

        return text, f"dataset-file:{os.path.basename(file_path)}"

    if source_mode == "huggingface":
        if not hf_dataset_name.strip():
            raise ValueError("Hugging Face dataset name is required.")

        if hf_config_name.strip():
            ds = load_dataset(
                hf_dataset_name.strip(),
                hf_config_name.strip(),
                split=hf_split.strip() or "train",
            )
        else:
            ds = load_dataset(
                hf_dataset_name.strip(),
                split=hf_split.strip() or "train",
            )

        if hf_text_field.strip():
            field = hf_text_field.strip()
            if field not in ds.column_names:
                raise ValueError(f"Field '{field}' not found. Available: {ds.column_names}")
            text = "\n".join(str(x) for x in ds[field] if x is not None)
        else:
            text = dataset_rows_to_text(ds)

        label = hf_dataset_name.strip()
        if hf_config_name.strip():
            label += f"/{hf_config_name.strip()}"
        label += f":{hf_split.strip() or 'train'}"

        return text, label

    raise ValueError(f"Unknown source mode: {source_mode}")


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
# UI RUNNER
# ============================================================

def run_search_ui(
    prompt,
    source_mode,
    uploaded_file,
    hf_dataset_name,
    hf_config_name,
    hf_split,
    hf_text_field,
    vertex,
    max_solutions,
):
    try:
        corpus, source_label = load_corpus_from_source(
            source_mode=source_mode,
            uploaded_file=uploaded_file,
            hf_dataset_name=hf_dataset_name,
            hf_config_name=hf_config_name,
            hf_split=hf_split,
            hf_text_field=hf_text_field,
        )

        cpd, vocab = build_model(corpus)
        dictionary = load_dictionary(vocab)
        stream = build_pi_stream()
        stream_text, found_words = find_words(stream, dictionary)

        results = brute_force_prompt_search(
            prompt=prompt,
            cpd=cpd,
            stream=stream,
            vertex=vertex,
            max_solutions=int(max_solutions),
        )

        if not results:
            return (
                f"Source: {source_label}\n"
                f"Corpus chars: {len(corpus)}\n"
                f"Vocab size: {len(vocab)}\n"
                f"Pi words found: {len(found_words)}\n"
                f"Matches: 0",
                []
            )

        formatted = []
        for r in results:
            formatted.append({
                "prompt": r["prompt"],
                "bend": r["bend"],
                "offset": r["offset"],
                "vertex": r["vertex"],
                "text": r["text"],
            })

        summary = (
            f"Source: {source_label}\n"
            f"Corpus chars: {len(corpus)}\n"
            f"Vocab size: {len(vocab)}\n"
            f"Pi words found: {len(found_words)}\n"
            f"Matches: {len(results)}"
        )

        return summary, formatted

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", []


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


# ============================================================
# GRADIO DEMO
# ============================================================

def build_demo():
    with gr.Blocks(title="Pi Base-26 NLTK Generator") as demo:
        gr.Markdown(
            """
# π → BASE-26 → NLTK Generator

Use an embedded corpus, upload a file, or load a Hugging Face dataset with optional config and split.
"""
        )

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter prompt",
                lines=3,
            )
            source_mode = gr.Radio(
                choices=["embedded", "file", "huggingface"],
                value="embedded",
                label="Corpus source",
            )

        with gr.Row():
            uploaded_file = gr.File(
                label="Upload corpus file",
                file_count="single",
                type="filepath"
            )

        with gr.Row():
            hf_dataset_name = gr.Textbox(
                label="HF dataset name",
                placeholder="e.g. ag_news"
            )
            hf_config_name = gr.Textbox(
                label="HF config name",
                placeholder="optional"
            )
            hf_split = gr.Textbox(
                label="HF split",
                value="train"
            )
            hf_text_field = gr.Textbox(
                label="HF text field",
                placeholder="optional"
            )

        with gr.Row():
            vertex = gr.Dropdown(
                choices=["A", "B", "C"],
                value="A",
                label="Triangle vertex",
            )
            max_solutions = gr.Slider(
                1, 20, value=5, step=1, label="Max solutions"
            )

        run_btn = gr.Button("Run search")
        summary = gr.Textbox(label="Summary", lines=6)
        results = gr.JSON(label="Matches")

        run_btn.click(
            fn=run_search_ui,
            inputs=[
                prompt,
                source_mode,
                uploaded_file,
                hf_dataset_name,
                hf_config_name,
                hf_split,
                hf_text_field,
                vertex,
                max_solutions,
            ],
            outputs=[summary, results],
        )

    return demo


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

        demo = build_demo()
        demo.launch()
   
