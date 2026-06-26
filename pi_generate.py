from pathlib import Path
import re
import random
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
import gradio as gr


class Vocab:
    def __init__(self, words):
        self.word2idx = {}
        self.idx2word = []
        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)

    def __len__(self):
        return len(self.idx2word)

    def encode(self, words):
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def decode(self, ids):
        return " ".join(self.idx2word[i] for i in ids)


class CorpusIndexer:
    def __init__(self, words, window_size=400, overlap=200):
        self.windows = []
        start = 0
        while start < len(words):
            chunk = words[start:start + window_size]
            self.windows.append({
                "words": chunk,
                "counter": Counter(w.lower() for w in chunk)
            })
            start += overlap

    def search(self, query, breadth=10):
        terms = re.findall(r"\w+", query.lower())
        scored = []
        for win in self.windows:
            score = sum(win["counter"].get(t, 0) for t in terms)
            scored.append((score, win))

        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for _, win in scored[:breadth]:
            result.extend(win["words"])

        return result


class TrigramMarkov:
    """
    Trigram/bigram/unigram Markov model with a dataset-wide baseline
    distribution blended into every sampling step ("guided walk").

    baseline_probs is computed once from the full token stream passed in.
    At sample time, the local context distribution (trigram or bigram
    counts) is mixed with the baseline distribution via a weighted sum:

        combined = (1 - baseline_weight) * context_probs + baseline_weight * baseline_probs

    This nudges generation back toward the corpus's natural word
    frequencies even when local context is sparse, noisy, or skewed
    (which happens a lot here since the corpus gets filtered down to
    small windows by CorpusIndexer.search).
    """

    def __init__(self, words):
        words = [w.lower() for w in words]

        self.words = words
        self.vocab = Vocab(words)

        ids = self.vocab.encode(words)

        self.trigram = defaultdict(Counter)
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()

        for i, tok in enumerate(ids):
            self.unigram[tok] += 1

            if i >= 1:
                self.bigram[(ids[i - 1],)][tok] += 1

            if i >= 2:
                self.trigram[(ids[i - 2], ids[i - 1])][tok] += 1

        # --- baseline probability distribution over the whole corpus ---
        total = sum(self.unigram.values())
        self.baseline_probs = {
            tok: count / total for tok, count in self.unigram.items()
        } if total > 0 else {}

    def sample_counter(self, counter, temperature=0.9, top_k=800, baseline_weight=0.3):
        """
        Sample a token id from a local context counter (trigram/bigram/unigram),
        blended with the global baseline distribution.

        baseline_weight in [0, 1]:
            0.0 -> pure local context (original behavior)
            1.0 -> pure baseline (ignores local context entirely)
        """
        if not counter:
            return None

        items = list(counter.items())

        tokens = torch.tensor([k for k, _ in items])
        counts = torch.tensor([v for _, v in items], dtype=torch.float32)

        # local context distribution (normalized counts)
        context_probs = counts / counts.sum()

        # baseline distribution restricted to these same candidate tokens
        baseline_vals = torch.tensor(
            [self.baseline_probs.get(int(t), 1e-9) for t in tokens],
            dtype=torch.float32,
        )
        baseline_sum = baseline_vals.sum()
        if baseline_sum > 0:
            baseline_vals = baseline_vals / baseline_sum
        else:
            baseline_vals = torch.full_like(context_probs, 1.0 / len(context_probs))

        # guided walk: weighted sum of context + baseline probabilities
        bw = min(max(baseline_weight, 0.0), 1.0)
        combined_probs = (1 - bw) * context_probs + bw * baseline_vals

        logits = torch.log(combined_probs + 1e-9) / max(temperature, 1e-6)

        if top_k and len(logits) > top_k:
            vals, idx = torch.topk(logits, top_k)
            logits = vals
            tokens = tokens[idx]

        probs = F.softmax(logits, dim=0)

        return tokens[torch.multinomial(probs, 1).item()].item()

    def sample_baseline(self):
        """Pure baseline sample, used as a last-resort fallback."""
        if not self.baseline_probs:
            return None
        toks = torch.tensor(list(self.baseline_probs.keys()))
        probs = torch.tensor(list(self.baseline_probs.values()), dtype=torch.float32)
        probs = probs / probs.sum()
        return toks[torch.multinomial(probs, 1).item()].item()

    def next_token(self, context, temperature=0.9, top_k=800, baseline_weight=0.3):
        if len(context) >= 2:
            key = (context[-2], context[-1])
            if key in self.trigram:
                tok = self.sample_counter(self.trigram[key], temperature, top_k, baseline_weight)
                if tok is not None:
                    return tok

        if len(context) >= 1:
            key = (context[-1],)
            if key in self.bigram:
                tok = self.sample_counter(self.bigram[key], temperature, top_k, baseline_weight)
                if tok is not None:
                    return tok

        # fallback: blend baseline with itself (i.e. pure baseline sample)
        tok = self.sample_baseline()
        if tok is not None:
            return tok

        return self.sample_counter(self.unigram, temperature, top_k, baseline_weight)


def load_words(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().split()


cognitive_words = [
    "sense",
    "detect",
    "observe",
    "notice",
    "perceive",
    "recognize",
    "identify",
    "attend",
    "focus",
    "awareness",
    "monitor",
    "track",
    "encode",
    "remember",
    "recall",
    "retrieve",
    "associate",
    "connect",
    "compare",
    "categorize",
    "generalize",
    "abstract",
    "conceptualize",
    "interpret",
    "understand",
    "comprehend",
    "reason",
    "infer",
    "deduce",
    "analyze",
    "evaluate",
    "estimate",
    "predict",
    "anticipate",
    "simulate",
    "imagine",
    "hypothesize",
    "explore",
    "plan",
    "strategize",
    "prioritize",
    "choose",
    "decide",
    "act",
    "verify",
    "reflect",
    "learn",
    "adapt",
    "optimize",
    "improve",
]


def generate_text(corpus_file, prompt, baseline_weight=0.3):
    corpus_path = corpus_file.name if corpus_file else "corpus.txt"

    corpus_words = load_words(corpus_path)

    indexer = CorpusIndexer(corpus_words)

    # global baseline model — built once on the FULL corpus, never re-fit.
    # this is the "ground truth" distribution that every guided step
    # gets pulled toward, regardless of how narrow the filtered window gets.
    global_model = TrigramMarkov(corpus_words)

    prompt = (prompt or "").strip().lower()
    if not prompt:
        prompt = "the"

    filtered = indexer.search(prompt, breadth=10)
    if len(filtered) < 100:
        filtered = corpus_words

    model = TrigramMarkov(filtered)
    # always walk with the dataset-wide baseline, even after re-fitting
    # the local model on refreshed/filtered windows below.
    model.baseline_probs = global_model.baseline_probs

    output_words = prompt.split()

    if len(output_words) < 2:
        output_words.insert(0, random.choice(corpus_words).lower())

    refresh_interval = 15
    trailing_tokens = 5
    target_length = 400

    while len(output_words) < target_length:

        context_ids = model.vocab.encode(output_words[-2:])

        if len(context_ids) < 2:
            next_word = random.choice(filtered).lower()
        else:
            nxt = model.next_token(context_ids, baseline_weight=baseline_weight)
            if nxt is None:
                next_word = random.choice(filtered).lower()
            else:
                next_word = model.vocab.idx2word[nxt]

        output_words.append(next_word)

        if len(output_words) % refresh_interval == 0:

            trailing = " ".join(output_words[-trailing_tokens:])
            query = f" {cognitive_words[len(output_words) % len(cognitive_words)]} {prompt}"

            refreshed = indexer.search(query, breadth=3)

            if len(refreshed) > 100:
                filtered = refreshed
                model = TrigramMarkov(filtered)
                # re-attach the global baseline so the guided walk keeps
                # pulling toward overall corpus statistics after refitting
                model.baseline_probs = global_model.baseline_probs

    text = " ".join(output_words)

    return text


demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.File(label="Upload corpus.txt", file_types=[".txt"]),
        gr.Textbox(label="Prompt"),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.3,
            step=0.05,
            label="Baseline weight (0 = pure local context, 1 = pure dataset baseline)",
        ),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Guided Walk Markov (Baseline-Blended)",
)

if __name__ == "__main__":
    demo.launch()
