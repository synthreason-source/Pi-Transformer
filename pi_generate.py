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
        self._sentence_words = sentence_words  # kept for build_ngram_choice()
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

    # ---------------------------------------------------------------
    # AXIOM OF CHOICE OVER N-GRAM CONTEXTS
    # ---------------------------------------------------------------
    #
    # For an n-gram order n, every context (a tuple of n-1 consecutive
    # words seen in the dataset) has an associated set of candidate
    # next-words — every word ever observed to follow that context.
    # That set is non-empty by construction (it only exists because at
    # least one continuation was observed), which is exactly the
    # Axiom-of-Choice precondition: given a collection of non-empty
    # sets {S_context}, AC guarantees a choice function picking one
    # element from each. AC itself only asserts a selector *exists* —
    # it doesn't say which element. Since the collection here is
    # finite, we make the choice function concrete and reproducible
    # instead of leaving it as a mere existence claim: for each
    # context, pick the highest-count candidate, tie-broken
    # alphabetically. That's a genuine function (same input always
    # gives the same output), which "the correct n-gram sequence" for
    # a context needs to mean anything.
    def build_ngram_choice(self, n=2):
        context_counts = {}
        for sent in self._sentence_words:
            for k in range(len(sent) - n + 1):
                context = tuple(sent[k:k + n - 1])
                nxt = sent[k + n - 1]
                context_counts.setdefault(context, Counter())[nxt] += 1

        choice = {}
        for context, candidates in context_counts.items():
            best_count = max(candidates.values())
            # deterministic tie-break: alphabetically first among the
            # highest-count candidates, so the "choice" is well-defined
            # rather than arbitrary
            winners = sorted(w for w, c in candidates.items() if c == best_count)
            choice[context] = winners[0]

        self.ngram_n = n
        self.ngram_choice = choice          # the choice function itself
        self.ngram_candidates = context_counts  # the underlying non-empty sets
        return choice

    def trace_sequence(self, start_context, max_len=50):
        """Follow the choice function deterministically from
        start_context, producing the single 'correct' sequence it
        identifies. Stops at a context with no recorded choice
        (unseen, or the end of every sentence it appeared in) or if a
        context repeats (the choice function would just loop forever)."""
        context = tuple(start_context)[-(self.ngram_n - 1):] if self.ngram_n > 1 else ()
        seq = list(context)
        seen = set()
        for _ in range(max_len):
            if context not in self.ngram_choice or context in seen:
                break
            seen.add(context)
            nxt = self.ngram_choice[context]
            seq.append(nxt)
            context = tuple(seen[-(self.ngram_n - 1):]) if self.ngram_n > 1 else ()
        return seq

    def all_correct_sequences(self, max_len=50):
        """Apply the choice function starting from every context that
        appears in the dataset — the complete set of 'correct' n-gram
        sequences it identifies, one canonical continuation path per
        starting context."""
        return {
            context: self.trace_sequence(context, max_len=max_len)
            for context in self.ngram_choice
        }

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
model.build_ngram_choice(n=3)

while True:
    prompt = input("USER: ")
    print(
        model.generate(
            start=prompt.split()[-1],
            chunk_size=6,
            length=1000,
            goal_strength=3.0
        )
    )