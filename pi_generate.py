import re
import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. VOCAB
# =========================
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


# =========================
# 2. TRIGRAM ZIGZAG MARKOV
# =========================
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
                self.bigram[(ids[i-1],)][ids[i]] += 1
            if i >= 2:
                self.trigram[(ids[i-2], ids[i-1])][ids[i]] += 1

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

    def zigzag_indices(self, length):
        left, right = 0, length - 1
        out = []
        while left <= right:
            out.append(left)
            left += 1
            if left <= right:
                out.append(right)
                right -= 1
        return out

    def generate_linear(self, seed_words, length=30, temperature=1.0, top_k=None):
        ids = self.vocab.encode([w.lower() for w in seed_words])
        if len(ids) < 2:
            while len(ids) < 2:
                ids = [random.randrange(len(self.vocab))] + ids

        state = ZigZagState(direction=1, step=0)
        out = ids[:]

        while len(out) < length:
            context = out[-2:]
            nxt = self.next_token(context, temperature=temperature, top_k=top_k)
            if nxt is None:
                break
            out.append(nxt)
            state.step += 1
            if state.step % 2 == 0:
                state.direction *= -1

        return self.vocab.decode(out)

    def generate_zigzag_walk(self, seed_words, length=30, temperature=1.0, top_k=None):
        ids = self.vocab.encode([w.lower() for w in seed_words])
        if len(ids) < 2:
            while len(ids) < 2:
                ids = [random.randrange(len(self.vocab))] + ids

        out = ids[:]
        plan = self.zigzag_indices(length)

        for i in range(len(out), length):
            context = out[-2:]
            nxt = self.next_token(context, temperature=temperature, top_k=top_k)
            if nxt is None:
                break
            out.append(nxt)

        return self.vocab.decode(out)


# =========================
# 3. GENERATION
# =========================
@torch.no_grad()
def generate(model, vocab, prompt, device, iso_processor, bench_processor, streak_tracker,
             super_gen=None, use_super_probs=False, max_len=300):
    model.eval()
    streak_tracker.reset()

    words = re.sub(r"[^\w\s]", "", prompt.lower()).split()
    ids = vocab.encode(words)
    if len(ids) < model.n:
        ids = [0] * (model.n - len(ids)) + ids

    for step in range(max_len):
        context = ids[-model.n:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        logits = iso_processor(logits)
        logits = bench_processor(logits)
        logits = streak_tracker(logits)
        probs = F.softmax(logits, dim=-1)[0]

        if use_super_probs and super_gen is not None:
            super_probs, _ = super_gen.get_super_probs(x, device)
            mix = 0.5 * probs + 0.5 * super_probs
            probs = mix / mix.sum()

        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)

    return vocab.decode(ids)


# =========================
# 9. MAIN
# =========================
def main(txt_path="corpus.txt", n=5):

    words = open(txt_path, "r", encoding="utf-8").read().split()
    zigzag = TrigramZigzagMarkov(words)

    print("\nEnter prompts. Ctrl+C to quit.\n")
    while True:
        user_input = input("USER: ")
        seed = user_input.strip()
        print(zigzag.generate_linear(seed.split(), length=400, temperature=0.9, top_k=800).split(".")[0] + ".")
        print()

if __name__ == "__main__":
    filepath = input("Filename: ")
    main(filepath, n=3)
