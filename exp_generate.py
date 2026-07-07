import numpy as np
import re
from collections import Counter


class InfluenceSpaceMarkov:
    def __init__(self, beta=1.5):
        self.beta = beta

    def fit(self, text):
        # ---------------------
        # SPLIT INTO SENTENCES
        # ---------------------
        # Each sentence becomes its own array of words. Transition
        # counts are accumulated within a sentence only, so the last
        # word of one sentence never gets counted as leading into the
        # first word of the next.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentence_words = [s.lower().split() for s in sentences if s.strip()]

        all_words = [w for sent in sentence_words for w in sent]
        vocab = sorted(set(all_words))
        self.vocab = vocab
        self.word_to_idx = {
            w: i for i, w in enumerate(vocab)
        }
        n = len(vocab)
        # ---------------------
        # DOMAIN A x B
        # ---------------------
        counts = np.zeros((n, n))
        for sent in sentence_words:
            for a, b in zip(sent[:-1], sent[1:]):
                i = self.word_to_idx[a]
                j = self.word_to_idx[b]
                counts[i, j] += 1
        # ---------------------
        # LOG SORTING
        # ---------------------
        log_counts = np.log1p(counts)
        # ---------------------
        # INFLUENCE MAP
        # f : A x B -> Y
        # ---------------------
        influence = np.exp(
            self.beta * log_counts
        )
        influence[counts == 0] = 0
        self.Y = influence
        # ---------------------
        # MARKOV NORMALIZATION
        # ---------------------
        row_sums = influence.sum(
            axis=1,
            keepdims=True
        )
        self.P = np.divide(
            influence,
            row_sums + 1,
            out=np.ones_like(influence),
            where=row_sums != 0
        )
        return self

    def influence(self, word):
        i = self.word_to_idx[word]
        row = self.Y[i]
        ranking = np.argsort(row)[::-1]
        return [
            (self.vocab[j], row[j])
            for j in ranking
            if row[j] > 1
        ]

    def next_word(self, current):
        i = self.word_to_idx[current]
        probs = self.P[i]
        if probs.sum() == 0:
            return np.random.choice(self.vocab)
        return np.random.choice(
            self.vocab,
            p=probs
        )

    def generate(self, start, chunk_size=20, length=1000, goal_strength=2.0):
        text = [start]
        current = start
        for step in range(length):
            # change checkpoint every chunk
            i = self.word_to_idx[current]
            probs = self.P[i].copy()

            probs /= probs.sum()
            current = np.random.choice(
                self.vocab,
                p=probs
            )
            text.append(current)

        return " ".join(text)


with open("singlekb.txt", "r", encoding="utf8") as f:
    corpus = f.read()

model = InfluenceSpaceMarkov(beta=2.0)
model.fit(corpus)

while True:
    prompt = input("USER: ")
    print(
        model.generate(
            start=prompt.split()[-1],
            chunk_size=630,
            length=1000,
            goal_strength=30.0
        )
    )
