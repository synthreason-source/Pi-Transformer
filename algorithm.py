#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adiabatic-Dark-State LLM
========================

A small hybrid text generator whose "physics feature seeder" is not a
made-up black box -- it is a literal implementation of the round-trip
adiabatic dark-state transfer equations:

    H(t) = [[0,      JL(t), 0    ],
            [JL(t),  0,      JR(t)],
            [0,      JR(t),  0    ]]

    hamiltonian_round(t) = H(t)        for 0 <= t <= T   (forward leg)
                          = H(2T - t)   for T  <  t <= 2T (backward leg,
                                                            time-reversed
                                                            pulse order)

    d|psi>/dt = -i H(t) |psi>     (hbar = 1)

Every word that has been generated so far becomes ONE round-trip leg of
this chain: the word is hashed into its own (T, JL/JR peak times, J0,
sigma), the 3-site state |psi> is adiabatically driven q0 -> q2 -> q0
(or wherever it lands) under that word's pulses, and the OUTPUT state of
word i becomes the INPUT state to word i+1's leg. The final populations
|psi|^2 after propagating through the whole generated sequence are the
"adiabatic features" fed into the LSTM, exactly where the earlier
quantum-circuit version fed in qubit expectation values.

This keeps a single trainable scalar (`adiabatic_weight`) that rescales
every leg's coupling strength J0 by a learned, input-dependent factor --
the direct analogue of the previous `quantum_weight * scalar_input`
feedback path.
"""

from __future__ import annotations
import hashlib
import math
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim

KB_LEN = 30000
MAX_SEED_WORDS = 12          # how many recent words form the adiabatic chain
STEPS_PER_LEG = 10           # RK4 steps per half-leg (forward or backward)


# ════════════════════════════════════════════════════════════════════════
# 1. WORD -> PULSE PARAMETERS
# ════════════════════════════════════════════════════════════════════════

def word_to_pulse_params(word: str) -> tuple[float, float, float, float, float]:
    """
    Deterministically hash a word into one round-trip leg's parameters:
    (T, TR, TL, J0, sigma), with TR < TL always preserved so every leg
    keeps the paper's counter-intuitive pulse order (JR peaks first).
    """
    cleaned = word.lower().strip(".,!?;:\"'()[]{}") or "_"
    h = int(hashlib.md5(cleaned.encode("utf-8")).hexdigest(), 16)

    T      = 10.0 + (h % 1000) / 1000.0 * 10.0                       # T in [10, 20]
    frac_r = 0.20 + ((h // 1_000)       % 100) / 100.0 * 0.30         # TR/T in [0.20, 0.50]
    frac_l = 0.55 + ((h // 100_000)     % 100) / 100.0 * 0.30         # TL/T in [0.55, 0.85]
    J0     = 4.0  + ((h // 10_000_000)  % 100) / 100.0 * 4.0          # J0 in [4, 8]
    sig_f  = 0.12 + ((h // 1_000_000_000) % 100) / 100.0 * 0.08       # sigma/T in [0.12, 0.20]

    TR, TL, sigma = frac_r * T, frac_l * T, sig_f * T
    return T, TR, TL, J0, sigma


# ════════════════════════════════════════════════════════════════════════
# 2. ADIABATIC FEATURE SEEDER  (the paper's equations, as a torch module)
# ════════════════════════════════════════════════════════════════════════

class AdiabaticFeatureSeeder(nn.Module):
    """
    Runs the round-trip dark-state protocol as a chain: one leg per
    generated word, output state of leg i feeds leg i+1. Fully
    differentiable (complex-valued autograd) w.r.t. `adiabatic_weight`.
    """

    def __init__(self, seed_words: list[str], steps_per_leg: int = STEPS_PER_LEG):
        super().__init__()
        self.steps_per_leg = steps_per_leg
        # Trainable global coupling-strength modulation -- the analogue of
        # the previous quantum_weight, learned end-to-end with the LSTM.
        self.adiabatic_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.seed_words: list[str] = []
        self.update_words(seed_words)

    def update_words(self, seed_words: list[str]) -> None:
        """Feed the seeder off whatever text has been generated so far."""
        self.seed_words = list(seed_words)[-MAX_SEED_WORDS:]

    @staticmethod
    def _build_H(t: torch.Tensor, TR: float, TL: float, J0: torch.Tensor, sigma: float) -> torch.Tensor:
        jl = J0 * torch.exp(-((t - TL) ** 2) / (2.0 * sigma ** 2))
        jr = J0 * torch.exp(-((t - TR) ** 2) / (2.0 * sigma ** 2))
        zero = torch.zeros_like(jl)
        row0 = torch.stack([zero, jl, zero])
        row1 = torch.stack([jl, zero, jr])
        row2 = torch.stack([zero, jr, zero])
        H = torch.stack([row0, row1, row2]).to(torch.cfloat)
        return H

    def _H_round(self, t: torch.Tensor, T: float, TR: float, TL: float,
                 J0: torch.Tensor, sigma: float) -> torch.Tensor:
        # hamiltonian_round(t): forward H(t) for t<=T, backward H(2T-t) after.
        if t.item() <= T:
            return self._build_H(t, TR, TL, J0, sigma)
        return self._build_H(2.0 * T - t, TR, TL, J0, sigma)

    def evolve_leg(self, psi0: torch.Tensor, T: float, TR: float, TL: float,
                   J0: torch.Tensor, sigma: float) -> torch.Tensor:
        """RK4-integrate d|psi>/dt = -i H_round(t) |psi> over one full round trip [0, 2T]."""
        n_total = 2 * self.steps_per_leg
        dt = (2.0 * T) / n_total
        psi = psi0
        t = torch.tensor(0.0, dtype=torch.float32)

        def rhs(tt, y):
            H = self._H_round(tt, T, TR, TL, J0, sigma)
            return -1j * (H @ y)

        for _ in range(n_total):
            k1 = rhs(t, psi)
            k2 = rhs(t + dt / 2, psi + dt / 2 * k1)
            k3 = rhs(t + dt / 2, psi + dt / 2 * k2)
            k4 = rhs(t + dt, psi + dt * k3)
            psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            psi = psi / torch.linalg.norm(psi)
            t = t + dt

        return psi

    def forward(self, scalar_input: torch.Tensor) -> torch.Tensor:
        """
        Propagate |q0> through one round-trip leg per seed word, chaining
        the output state of each leg into the next. `scalar_input` (the
        LSTM's own running context) trainably rescales every leg's J0 via
        `adiabatic_weight`, exactly mirroring the earlier
        `quantum_weight * scalar_input -> theta` feedback path.
        """
        # Keep the coupling-strength multiplier positive and bounded, since
        # J0 has to stay physically meaningful (a real coupling strength).
        factor = 0.5 + 1.5 * torch.sigmoid(self.adiabatic_weight * scalar_input)

        psi = torch.tensor([1.0, 0.0, 0.0], dtype=torch.cfloat)  # |q0>
        if not self.seed_words:
            populations = (psi.abs() ** 2).real
            return populations.to(torch.float32)

        for word in self.seed_words:
            T, TR, TL, J0_base, sigma = word_to_pulse_params(word)
            J0 = torch.as_tensor(J0_base, dtype=torch.float32) * factor
            psi = self.evolve_leg(psi, T, TR, TL, J0, sigma)

        populations = (psi.abs() ** 2).real
        return populations.to(torch.float32)  # [P0, P1, P2], sums to ~1


# ════════════════════════════════════════════════════════════════════════
# 3. HYBRID LSTM + ADIABATIC MODEL
# ════════════════════════════════════════════════════════════════════════

class AdiabaticCorrelatedTextGenerator(nn.Module):
    def __init__(self, seed_words: list[str], vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.adiabatic_seeder = AdiabaticFeatureSeeder(seed_words)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 3, hidden_dim, batch_first=True)  # +3 = P0,P1,P2
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def refresh_seeder(self, words: list[str]) -> None:
        """Feed the adiabatic chain the words generated so far, without
        touching any trained weights (embedding / lstm / fc / adiabatic_weight)."""
        self.adiabatic_seeder.update_words(words)

    def forward(self, text_tensor: torch.Tensor, hidden=None):
        batch_size, seq_len = text_tensor.shape
        embedded = self.embedding(text_tensor)

        scalar_seed = text_tensor[:, 0].float().mean() / 1000.0
        adia_features = self.adiabatic_seeder(scalar_seed)  # [P0, P1, P2]

        adia_broadcast = adia_features.repeat(batch_size, seq_len, 1)
        lstm_input = torch.cat([embedded, adia_broadcast], dim=-1)

        out, hidden = self.lstm(lstm_input, hidden)
        logits = self.fc(out)
        return logits, hidden


# ════════════════════════════════════════════════════════════════════════
# 4. WORD CORRELATIONS (Trigrams/Bigrams)
# ════════════════════════════════════════════════════════════════════════

def build_correlation_matrix(words):
    trigrams = defaultdict(Counter)
    bigrams = defaultdict(Counter)
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i + 1], words[i + 2]
        trigrams[(w1, w2)][w3] += 1
        bigrams[w1][w2] += 1
    return dict(trigrams), dict(bigrams)


# ════════════════════════════════════════════════════════════════════════
# 5. HYBRID GENERATION (LSTM + Adiabatic + Correlation Fallback)
# ════════════════════════════════════════════════════════════════════════

def generate_correlated_text(model, prompt_str, trigrams, bigrams, vocab_to_int, int_to_vocab,
                              max_words=15, temperature=0.7):
    model.eval()
    words = prompt_str.split()

    with torch.no_grad():
        for _ in range(max_words):
            # The adiabatic chain always feeds off whatever has been
            # generated (or prompted) so far.
            model.refresh_seeder(words)

            next_word = None
            if len(words) >= 2:
                context = (words[-2], words[-1])
                if context in trigrams:
                    possible_next = trigrams[context]
                    candidates, counts = zip(*possible_next.items())
                    total = sum(counts)
                    probs = [c / total for c in counts]
                    next_word = torch.multinomial(torch.tensor(probs), num_samples=1).item()
                    next_word = candidates[next_word]

            if not next_word and len(words) >= 1:
                context = words[-1]
                if context in bigrams:
                    possible_next = bigrams[context]
                    candidates, counts = zip(*possible_next.items())
                    total = sum(counts)
                    probs = [c / total for c in counts]
                    next_word = torch.multinomial(torch.tensor(probs), num_samples=1).item()
                    next_word = candidates[next_word]

            if not next_word:
                prompt_indices = [vocab_to_int.get(w, 0) for w in words[-5:]]
                input_tensor = torch.tensor([prompt_indices], dtype=torch.long)
                logits, _ = model(input_tensor)
                next_word_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_word_logits, dim=-1)
                next_word_idx = torch.multinomial(probs, num_samples=1).item()
                next_word = int_to_vocab.get(next_word_idx, "the")

            words.append(next_word)

    return " ".join(words)


# ════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        with open(input("Filename: "), "r", encoding="utf-8") as file:
            dataset_corpus = file.read()[:KB_LEN]
    except FileNotFoundError:
        dataset_corpus = ("quantum computing and neural networks integration allows hybrid "
                           "workflow optimization cascades. adiabatic dark state transfer "
                           "moves population between sites through a dark eigenstate that "
                           "never touches the lossy middle site.")

    words = dataset_corpus.split()
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)

    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i: word for i, word in enumerate(vocab)}

    trigrams, bigrams = build_correlation_matrix(words)

    seq_length = 8
    inputs, targets = [], []
    for i in range(len(words) - seq_length):
        seq_in = words[i:i + seq_length]
        seq_out = words[i + 1:i + seq_length + 1]
        inputs.append([vocab_to_int[w] for w in seq_in])
        targets.append([vocab_to_int[w] for w in seq_out])

    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)

    initial_seed_words = words[:MAX_SEED_WORDS]

    model = AdiabaticCorrelatedTextGenerator(initial_seed_words, vocab_size, embed_dim=32, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("--- Training Adiabatic-Dark-State Hybrid Text Generator ---")
    for epoch in range(25):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/25 | Loss: {loss.item():.4f} | "
                  f"adiabatic_weight={model.adiabatic_seeder.adiabatic_weight.item():.4f}")

    while True:
        test_prompt = input("USER: ")
        result = generate_correlated_text(model, test_prompt, trigrams, bigrams, vocab_to_int, int_to_vocab,
                                           max_words=600, temperature=0.7)
        print("\n--- Adiabatic-Dark-State Correlation Result ---")
        print(f"Prompt: '{test_prompt}'")
        print(f"Generated Output: {result}")
