import numpy as np
import re
from collections import Counter

class InfluenceSpaceMarkov:

    def __init__(self, beta=1.5):
        self.beta = beta

    def fit(self, text):

        words = text.lower().split()

        vocab = sorted(set(words))

        self.vocab = vocab

        self.word_to_idx = {
            w:i for i,w in enumerate(vocab)
        }

        n = len(vocab)

        # ---------------------
        # DOMAIN A × B
        # ---------------------

        counts = np.zeros((n,n))

        for a,b in zip(words[:-1], words[1:]):

            i = self.word_to_idx[a]
            j = self.word_to_idx[b]

            counts[i,j] += 1

        # ---------------------
        # LOG SORTING
        # ---------------------

        log_counts = np.log1p(counts)

        # ---------------------
        # INFLUENCE MAP
        # f : A × B → Y
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
            row_sums,
            out=np.zeros_like(influence),
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
            if row[j] > 0
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

    def generate(self, start, length=50):

        text = [start]

        current = start

        for _ in range(length):

            current = self.next_word(current)

            text.append(current)

        return " ".join(text)
with open("singlekb.txt","r",encoding="utf8") as f:
    corpus = f.read()

model = InfluenceSpaceMarkov(beta=2.0)

model.fit(corpus)

print(model.generate(
    start=input("seed word: "),
    length=1000
))
