from pathlib import Path
import re
import random
from collections import defaultdict, Counter
from dataclasses import dataclass

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


@dataclass
class ZigZagState:
    direction: int = 1
    step: int = 0


class TrigramZigzagMarkov:
    def __init__(self, words, lowercase=True):
        if lowercase:
            words = [w.lower() for w in words]
        self.vocab = Vocab(words)
        self.word_ids = self.vocab.encode(words)
        self.trigram = defaultdict(Counter)
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()
        self._build_tables()

    def _build_tables(self):
        ids = self.word_ids
        for i in range(len(ids)):
            self.unigram[ids[i]] += 1
            if i >= 1:
                self.bigram[(ids[i - 1],)][ids[i]] += 1
            if i >= 2:
                self.trigram[(ids[i - 2], ids[i - 1])][ids[i]] += 1

    def _sample_from_counter(self, counter, temperature=1.0, top_k=None):
        if not counter:
            return None
        items = list(counter.items())
        tokens = torch.tensor([k for k, _ in items], dtype=torch.long)
        counts = torch.tensor([v for _, v in items], dtype=torch.float32)
        logits = torch.log(counts + 1e-9) / max(temperature, 1e-6)
        if top_k is not None and top_k > 0 and len(logits) > top_k:
            vals, idx = torch.topk(logits, top_k)
            tokens = tokens[idx]
            logits = vals
        probs = F.softmax(logits, dim=0)
        choice = torch.multinomial(probs, 1).item()
        return tokens[choice].item()

    def next_token(self, context, temperature=1.0, top_k=None):
        if len(context) >= 2:
            key = (context[-2], context[-1])
            if key in self.trigram and len(self.trigram[key]) > 0:
                return self._sample_from_counter(self.trigram[key], temperature, top_k)
        if len(context) >= 1:
            key = (context[-1],)
            if key in self.bigram and len(self.bigram[key]) > 0:
                return self._sample_from_counter(self.bigram[key], temperature, top_k)
        return self._sample_from_counter(self.unigram, temperature, top_k)

    def generate_linear(self, seed_words, length=400, temperature=0.9, top_k=800):
        ids = self.vocab.encode([w.lower() for w in seed_words])
        if len(ids) < 2:
            while len(ids) < 2:
                ids = [random.randrange(len(self.vocab))] + ids
        out = ids[:]
        state = ZigZagState(direction=1, step=0)
        while len(out) < length:
            nxt = self.next_token(out[-2:], temperature=temperature, top_k=top_k)
            if nxt is None:
                break
            out.append(nxt)
            state.step += 1
            if state.step % 2 == 0:
                state.direction *= -1
        return self.vocab.decode(out)


def load_corpus_words(corpus_path):
    path = Path(corpus_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split()


def build_generator(corpus_path):
    words = load_corpus_words(corpus_path)
    return TrigramZigzagMarkov(words)


def generate_text(corpus_file, prompt):
    corpus_path = corpus_file.name if corpus_file is not None else "corpus.txt"
    gen = build_generator(corpus_path)
    seed = (prompt or "").strip()
    if not seed:
        seed = "the"
    out = gen.generate_linear(seed.split(), length=400, temperature=0.9, top_k=800)
    return out.split(".")[0] + "."


demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.File(label="Upload corpus.txt", file_types=[".txt"]),
        gr.Textbox(label="Prompt"),
    ],
    outputs=gr.Textbox(label="Generated text"),
)

if __name__ == "__main__":
    demo.launch()
