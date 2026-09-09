"""
markov_freq_logic.py
====================
Markov model using vocabulary frequencies for p_t and transition probabilities for self.P,
incorporating your exact custom logic step and an incremental approximation for generation.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\[\]{}\-]")
BOS, EOS = "<BOS>", "<EOS>"


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [" ".join(t) for t in zip(tokens, tokens[1:], tokens[2:])]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def detokenize(tokens: List[str]) -> str:
    text = " ".join(t for t in tokens if t not in (BOS, EOS))
    text = re.sub(r"\s+([.,!?;:)\]}])", r"\1", text)
    text = re.sub(r"([({\[<])\s+", r"\1", text)
    return text


@dataclass
class FreqMarkovModel:
    vocab: List[str]
    token_to_idx: Dict[str, int]
    P: np.ndarray  # shape [V, V], transition probabilities

    @classmethod
    def from_corpus(cls, text: str) -> "FreqMarkovModel":
        sentences = split_sentences(text)
        unigram: Counter = Counter()
        bigram: Dict[str, Counter] = defaultdict(Counter)

        for sent in sentences:
            tokens = tokenize(sent)
            if not tokens:
                continue
            seq = [BOS] + tokens + [EOS]
            for t in seq:
                unigram[t] += 1
            for left, right in zip(seq, seq[1:]):
                bigram[left][right] += 1

        vocab = sorted(list(unigram.keys()))
        token_to_idx = {t: i for i, t in enumerate(vocab)}
        V = len(vocab)

        P = np.zeros((V, V), dtype=np.float64)
        for left, right_counts in bigram.items():
            if left not in token_to_idx:
                continue
            i = token_to_idx[left]
            for right, count in right_counts.items():
                if right not in token_to_idx:
                    continue
                j = token_to_idx[right]
                P[i, j] += count
            
            row_total = P[i, :].sum()
            if row_total > 0:
                P[i, :] /= row_total

        # Handle zero rows
        for i in range(V):
            if P[i, :].sum() == 0:
                if EOS in token_to_idx:
                    P[i, token_to_idx[EOS]] = 1.0
                else:
                    P[i, :] = 1.0 / V

        return cls(vocab=vocab, token_to_idx=token_to_idx, P=P)

    def step(self, p_t: np.ndarray, temperature: float = 1.0, eps: float = 1e-12) -> np.ndarray:
        # --- Your Custom Logic ---
        raw = p_t @ self.P
        condition = np.exp(np.maximum(raw, eps)) >= np.cos(p_t)
        logits = np.where(condition, np.sort(np.exp(np.maximum(raw, eps))), raw / 2.0)
        
        # Scaling & Softmax
        scaled = logits / temperature
        shifted = scaled - scaled.max()
        exps = np.exp(shifted)
        p_next = exps / exps.sum()
        return p_next

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, decay: float = 0.9) -> str:
        prompt_tokens = tokenize(prompt) if prompt.strip() else []
        generated = list(prompt_tokens)

        if not generated:
            generated = [BOS]

        # Initialize p_t using an incremental exponential decay approximation
        p_t = np.zeros(len(self.vocab), dtype=np.float64)
        for token in generated:
            if token in self.token_to_idx:
                p_t *= decay
                p_t[self.token_to_idx[token]] += (1.0 - decay)

        total_count = p_t.sum()
        if total_count > 0:
            p_t /= total_count

        for _ in range(max_new_tokens):
            p_next = self.step(p_t, temperature=temperature)
            
            # Sample from distribution
            next_idx = int(np.random.choice(len(p_next), p=p_next))
            next_token = self.vocab[next_idx]

            if next_token == EOS:
                break

            generated.append(next_token)

            # Incrementally update p_t using exponential decay approximation (O(1) update)
            p_t *= decay
            if next_token in self.token_to_idx:
                p_t[self.token_to_idx[next_token]] += (1.0 - decay)
            
            total_count = p_t.sum()
            if total_count > 0:
                p_t /= total_count

        return detokenize(generated)


def main():
    parser = argparse.ArgumentParser(description="Frequency-based Custom Logic Markov Model with Approximation")
    parser.add_argument("--corpus", required=True, help="Path to text file.")
    parser.add_argument("--prompt", default="", help="Generation prompt.")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--decay", type=float, default=0.9, help="Exponential decay factor for history approximation.")
    args = parser.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    model = FreqMarkovModel.from_corpus(text)
    while True:
        output = model.generate(prompt=input("USER: "), max_new_tokens=args.max_tokens, temperature=args.temp, decay=args.decay)
        print("\n--- Output ---")
        print(output)


if __name__ == "__main__":
    main()
