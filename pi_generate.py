"""
Markov Influence Model (order-N / trigram-capable)
====================================================
Implements the pipeline from the diagram:

  1. Domain Construction : build A x B, the Cartesian product of
                            "current state" (A) and "next state" (B).
                            A state is a tuple of `order` consecutive
                            words. For order=1 that's a single word
                            (bigram model); for order=2 it's a pair of
                            words (trigram model), etc. Two states are
                            connected if a training sentence contains
                            state_i immediately followed by state_j
                            (i.e. they overlap by `order - 1` words:
                            (a, b) -> (b, c)). A = B = set of observed
                            states, so the domain is still a square
                            m x m matrix and the matrix exponential in
                            step 2 stays well-defined.
  2. Weighting            : log-sort state frequencies, then apply
                            matrix-exponential weighting to the raw
                            transition matrix to get multi-step
                            "influence scores" (not just 1-hop
                            probabilities).
  3. Mapping              : f(a_i, b_j) -> influence score, giving the
                            codomain Y.
  4. Result                : Y is used to generate text, sampling each
                            next state by its influence-weighted score
                            instead of the raw n-gram probability.

Why order matters (bigram vs. trigram)
----------------------------------------
A bigram (order=1) model's state is a single word, so a very common
word like "the" becomes a high-degree hub with lots of self-reinforcing
loops -> generated text degenerates into "the the the forest forest
forest ...". A trigram (order=2) model's state is a *pair* of words,
so the next word depends on two words of context instead of one. This
sharply reduces repetition because "the the" and "the fox" are now
different states with different, more specific outgoing transitions.

Why matrix-exponential weighting?
----------------------------------
A plain first-order chain over states only "sees" one step ahead:
P[i, j] is the probability of going straight from state i to state j.
Taking the matrix exponential expm(P) sums the contribution of ALL
path lengths between i and j (1-step, 2-step, 3-step, ... weighted by
1/k!), so a transition gets a high "influence score" if state i can
reach state j either directly OR through a short chain of intermediate
states. This is the same idea used in diffusion kernels / communica-
bility scores for graphs.
"""

import re
import math
from collections import Counter

import numpy as np
from scipy.linalg import expm
KB_LEN = 29999

# ---------------------------------------------------------------------
# 1. Domain Construction: build A x B (state pairs) as an m x m matrix
# ---------------------------------------------------------------------
class MarkovInfluenceModel:
    def __init__(self, text: str, order: int = 2, alpha: float = 1.0):
        """
        text  : training corpus
        order : number of words per state.
                order=1 -> bigram model (state = 1 word)
                order=2 -> trigram model (state = 2 words)   [default]
                order=3 -> 4-gram model (state = 3 words), etc.
        alpha : scales the transition matrix before exponentiating.
                Larger alpha -> influence scores decay faster with
                path length (more weight on direct transitions).
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.alpha = alpha
        self.tokens = self._tokenize(text)

        # states are overlapping windows of `order` consecutive words
        self.states = [
            tuple(self.tokens[i:i + order])
            for i in range(len(self.tokens) - order + 1)
        ]
        self.vocab = sorted(set(self.states))           # Set A = Set B
        self.index = {s: i for i, s in enumerate(self.vocab)}
        self.m = len(self.vocab)

        self.counts = self._build_count_matrix()          # A x B counts
        self.P = self._row_normalize(self.counts)          # 1-step probs
        self.log_ranked_vocab = self._log_sort_vocab()     # step 2a
        self.Y = self._compute_influence_scores()          # step 2b/3

    # -- tokenizing -----------------------------------------------------
    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"[a-zA-Z']+", text.lower())

    # -- 1. Cartesian product A x B, as an m x m count matrix over states
    def _build_count_matrix(self):
        counts = np.zeros((self.m, self.m), dtype=float)
        # state_i -> state_j is a valid transition when they overlap by
        # (order - 1) words, i.e. state_j drops state_i's first word
        # and appends one new word (a sliding window by 1).
        for s_i, s_j in zip(self.states, self.states[1:]):
            counts[self.index[s_i], self.index[s_j]] += 1.0
        return counts

    @staticmethod
    def _row_normalize(mat):
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div-by-zero for unseen rows
        return mat / row_sums

    # -- 2a. log-sorting: rank states by log-frequency --------------------
    def _log_sort_vocab(self):
        freq = Counter(self.states)
        # log-sorting compresses the frequency range so a few very
        # common states don't totally dominate the ranking
        ranked = sorted(freq.items(), key=lambda kv: -math.log(kv[1] + 1))
        return ranked

    # -- 2b/3. matrix exponential weighting -> influence scores (Y) -----
    def _compute_influence_scores(self):
        # expm(alpha * P) sums contributions from paths of every length
        # between state i and state j, giving each pair (a_i, b_j) its
        # influence score f(a_i, b_j).
        #
        # Caveat: expm(alpha*P) always contains the identity term
        # (P^0 = I), so the diagonal (self-influence) is guaranteed to
        # be the largest entry in every row -- an artifact of the math,
        # not a signal from the data. Left uncorrected, this makes a
        # state look "most influenced by itself," which drags text
        # generation into self-loops (e.g. "the fox -> the fox"). We
        # zero the diagonal before renormalizing so influence scores
        # reflect genuine multi-hop reachability to *other* states.
        M = expm(self.alpha * self.P)
        M = np.clip(M, 0.0, None)  # expm() can leave tiny negative
                                    # floating-point residuals (e.g.
                                    # -1e-17) even though the true
                                    # exponential of a nonnegative
                                    # matrix is nonnegative; clip them
                                    # so downstream sampling never sees
                                    # a negative "probability"
        np.fill_diagonal(M, 0.0)
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        Y = M / row_sums
        # renormalize once more to guard against residual floating-
        # point drift (rows summing to e.g. 0.9999999997 or 1.0000000002)
        Y = Y / Y.sum(axis=1, keepdims=True).clip(min=1e-15)
        return Y

    # -- helper: normalize a "current state" argument to a tuple ---------
    def _as_state(self, s):
        if isinstance(s, str):
            s = tuple(self._tokenize(s))
        return tuple(s)

    # -- 4. Result: use Y to generate text -------------------------------
    def influence(self, a, b) -> float:
        """f(a, b): the influence score for a specific state pair.
        a, b can be strings (e.g. "the fox") or word tuples."""
        a, b = self._as_state(a), self._as_state(b)
        if a not in self.index or b not in self.index:
            return 0.0
        return float(self.Y[self.index[a], self.index[b]])

    def top_influenced(self, a, k: int = 5):
        """Top-k states most influenced by current state `a`."""
        a = self._as_state(a)
        if a not in self.index:
            return []
        row = self.Y[self.index[a]]
        top_idx = np.argsort(-row)[:k]
        return [(self.vocab[i], float(row[i])) for i in top_idx]

    def generate(self, start=None, length: int = 20, seed: int = None):
        """start: a seed state, e.g. "the fox" for order=2, or a tuple.
        If omitted, or not found in the trained vocabulary (wrong
        length, unseen words, etc.), a random valid start state is
        chosen instead so generation never hard-fails."""
        rng = np.random.default_rng(seed)

        current = None
        if start is not None:
            candidate = self._as_state(start)
            if len(candidate) == self.order and candidate in self.index:
                current = candidate
            else:
                # doesn't match a trained state (wrong length, unseen
                # words, etc.) -- fall back to random rather than error
                print(f"Warning: {candidate} not in vocabulary of "
                      f"states, picking a random start instead.")

        if current is None:
            current = self.vocab[rng.integers(len(self.vocab))]

        out = list(current)
        for _ in range(length - self.order):
            row = self.Y[self.index[current]]
            if row.sum() == 0:
                break
            row = np.clip(row, 0.0, None)
            row = row / row.sum()
            next_state_idx = rng.choice(len(self.vocab), p=row)
            current = self.vocab[next_state_idx]
            out.append(current[-1])  # each step advances by exactly 1 new word
        return " ".join(out)


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    with open("singlekb.txt","r",encoding="utf8") as f:
        corpus = f.read()[:KB_LEN]

    print("\n" + "=" * 60)
    print("TRIGRAM MODEL (order=2) -- state = 2 words")
    print("=" * 60)
    model = MarkovInfluenceModel(corpus, order=2, alpha=1.0)
    print("\nGenerated text (unseen seed 'what are you' -> falls back to random start):")
    print(" ", model.generate(input("USER: "), length=250, seed=42))
