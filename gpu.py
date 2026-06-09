import torch
import numpy as np
import random

# =========================
# DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# LOAD DATA
# =========================
text = open("singlekb.txt", "r", encoding="utf-8").read().lower()[:9999]
words = text.split()

vocab = sorted(list(set(words)))
V = len(vocab)

word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}


# =========================
# BUILD TRIGRAMS
# =========================
trigrams = []
for i in range(len(words) - 2):
    trigrams.append((words[i], words[i+1], words[i+2]))


# =========================
# PARTITION (GPU FIELD MODEL)
# =========================
class Partition:

    def __init__(self, V, alpha=0.9):
        self.V = V
        self.alpha = alpha

        # trigram field on GPU
        self.field = torch.ones((V, V, V), device=device) / V

        # decomposition tensor
        self.decomp = torch.zeros((V, V, V), device=device)

        self.energy = 1.0

    def observe(self, a, b, c):

        pred = self.field[a, b]

        target = torch.zeros(self.V, device=device)
        target[c] = 1.0

        error = target - pred

        # local update rule (no backprop)
        self.field[a, b] += self.alpha * error

        # normalization (keep probability distribution valid)
        self.field[a, b] = torch.clamp(self.field[a, b], min=1e-8)
        self.field[a, b] /= torch.sum(self.field[a, b])

        # decomposition enforcement
        self.decomp[a, b, c] = (
            self.alpha * self.decomp[a, b, c] + (1 - self.alpha)
        )

        self.energy = torch.mean(torch.abs(error)).item()


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

            decomp_score = torch.mean(p.decomp[a, b]).item()

            score = p.energy - decomp_score

            if score < best_score:
                best_score = score
                best_id = i

        return best_id


# =========================
# TRAINING
# =========================
def train(system, trigrams, epochs=6):

    for ep in range(epochs):

        random.shuffle(trigrams)

        for a, b, c in trigrams:

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
def generate(system, start_a, start_b, length=60):

    a = word_to_ix[start_a]
    b = word_to_ix[start_b]

    pid = system.assign(a, b)
    part = system.parts[pid]

    out = [start_a, start_b]

    for _ in range(length):

        probs = part.field[a, b].detach().cpu().numpy()
        probs = probs / probs.sum()

        c = np.random.choice(range(V), p=probs)

        out.append(ix_to_word[c])

        a, b = b, c

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

    print("\n--- GENERATED TEXT ---\n")

    seed = random.choice(trigrams)
    print(generate(system, seed[0], seed[1], length=60))
