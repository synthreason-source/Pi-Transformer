import numpy as np
import random
from collections import defaultdict


# =========================
# LOAD DATA
# =========================
text = open("singlekb.txt", "r", encoding="utf-8").read().lower()
words = text.split()

vocab = sorted(list(set(words)))
V = len(vocab)

word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}


# =========================
# BUILD TRIGRAM STREAM
# =========================
trigrams = []
for i in range(len(words) - 2):
    trigrams.append((words[i], words[i+1], words[i+2]))


# =========================
# PARTITION (SPARSE WORLD MODEL)
# =========================
class Partition:

    def __init__(self, V, alpha=0.9):
        self.V = V
        self.alpha = alpha

        # sparse trigram field:
        # (a,b) -> {c: prob}
        self.field = defaultdict(lambda: defaultdict(dict))
        # sparse decomposition memory:
        # (a,b,c) -> strength
        self.decomp = defaultdict(float)

        self.energy = 1.0

    def observe(self, a, b, c):

        pred = self.field[a][b]

        # initialize distribution if empty
        if len(pred) == 0:
            for i in range(self.V):
                pred[i] = 1.0 / self.V

        # prediction
        p = pred.get(c, 1e-6)

        error = 1 - p

        # local update (no global gradients)
        pred[c] = pred.get(c, 0.0) + self.alpha * error

        # renormalize
        total = sum(pred.values())
        for k in pred:
            pred[k] /= total

        # decomposition enforcement (ordered structure)
        key = (a, b, c)
        self.decomp[key] = (
            self.alpha * self.decomp[key] +
            (1 - self.alpha)
        )

        # energy = instability
        self.energy = abs(error)


# =========================
# PARTITION SYSTEM
# =========================
class PartitionSystem:

    def __init__(self, n_parts, V):
        self.parts = [Partition(V) for _ in range(n_parts)]

    def assign(self, a, b):

        best_id = 0
        best_score = float("inf")

        for i, p in enumerate(self.parts):

            decomp_score = 0.0

            # accumulate structural consistency
            for c in p.field[a][b]:
                decomp_score += p.decomp.get((a, b, c), 0.0)

            score = p.energy - decomp_score

            if score < best_score:
                best_score = score
                best_id = i

        return best_id


# =========================
# TRAINING
# =========================
def train(system, data, epochs=6):

    for ep in range(epochs):

        random.shuffle(data)

        for a, b, c in data:

            ai = word_to_ix[a]
            bi = word_to_ix[b]
            ci = word_to_ix[c]

            pid = system.assign(ai, bi)
            part = system.parts[pid]

            part.observe(ai, bi, ci)

        print(f"epoch {ep} done")


# =========================
# GENERATION
# =========================
def generate(system, start_a, start_b, length=50):

    a = word_to_ix[start_a]
    b = word_to_ix[start_b]

    pid = system.assign(a, b)
    part = system.parts[pid]

    out = [start_a, start_b]

    for _ in range(length):

        probs = part.field[a][b]

        if len(probs) == 0:
            next_ix = random.randint(0, V - 1)
        else:
            keys = list(probs.keys())
            vals = list(probs.values())
            vals = np.array(vals)
            vals = vals / vals.sum()

            next_ix = np.random.choice(keys, p=vals)

        word = ix_to_word[next_ix]
        out.append(word)

        # shift trigram window
        a, b = b, next_ix

        # switch partition dynamically
        pid = system.assign(a, b)
        part = system.parts[pid]

    return " ".join(out)


# =========================
# RUN
# =========================
if __name__ == "__main__":

    system = PartitionSystem(n_parts=6, V=V)

    print("training...")
    train(system, trigrams, epochs=8)

    while True:
        seed = input("Bigram: ").split()
        seed = random.choice(trigrams)
        print(generate(system, seed[-2], seed[-1], length=600))
