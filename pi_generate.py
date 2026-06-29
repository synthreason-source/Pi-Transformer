"""
trigram_markov.py
------------------
A word-level trigram language model, backed by an O(1) LFU-style
vocabulary (FrequencyTapper).

Two pieces:

1. FrequencyTapper
   An online word<->id vocab. Every time a word is seen its frequency
   bumps by one and it's moved to the matching "bucket" in O(1) -
   no re-sorting, no recomputation of a global counter. This is the
   classic bucket-list structure used in O(1) LFU caches: each integer
   frequency has a bucket (an ordered set of word ids at that
   frequency), and the set of occupied frequencies is tracked in
   sorted order so "most/least frequent" queries are cheap.

2. TrigramMarkov
   A standard trigram model (with bigram/unigram backoff) built on
   top of a FrequencyTapper, so the vocabulary itself tracks word
   frequency as it learns.

Run directly for a demo:
    python trigram_markov.py
"""

import random
from bisect import insort
from collections import defaultdict


# --------------------------------------------------------------------------
# FrequencyTapper: O(1) word<->id vocab with LFU-style frequency buckets
# --------------------------------------------------------------------------

class FrequencyTapper:
    """Online word<->id vocab with O(1) frequency bumps."""

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.freq: dict[int, int] = {}            # idx -> current frequency
        self.buckets: dict[int, dict[int, None]] = {}  # freq -> ordered set of idx (dict preserves insertion order)
        self.bucket_order: list[int] = []          # sorted list of frequencies that have >=1 word

    # -- internal helpers ---------------------------------------------------

    def _ensure_bucket(self, f: int):
        if f not in self.buckets:
            self.buckets[f] = {}
            insort(self.bucket_order, f)

    def _drop_bucket_if_empty(self, f: int):
        if f in self.buckets and not self.buckets[f]:
            del self.buckets[f]
            self.bucket_order.remove(f)

    # -- public API -----------------------------------------------------------

    def bump(self, word: str) -> int:
        """Register a single occurrence of `word`. Returns its id."""
        word = word.lower()

        if word not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
            self.freq[idx] = 0
        else:
            idx = self.word2idx[word]

        old_f = self.freq[idx]
        new_f = old_f + 1
        self.freq[idx] = new_f

        if old_f > 0:
            self.buckets[old_f].pop(idx, None)
            self._drop_bucket_if_empty(old_f)

        self._ensure_bucket(new_f)
        self.buckets[new_f][idx] = None
        return idx

    def bump_many(self, words):
        """Bump every word in an iterable. Returns the list of ids."""
        if not words:
            return []
        return [self.bump(w) for w in words]

    def encode(self, words):
        """Map known words to ids (does NOT register new words / bump freq)."""
        return [self.word2idx[w.lower()] for w in words if w.lower() in self.word2idx]

    def decode(self, ids):
        return " ".join(self.idx2word[i] for i in ids)

    def most_frequent(self, n=1):
        """Top-n (word, freq) pairs, highest frequency first."""
        out = []
        for f in reversed(self.bucket_order):
            for idx in self.buckets[f]:
                out.append((self.idx2word[idx], f))
                if len(out) >= n:
                    return out
        return out

    def least_frequent(self, n=1):
        """Top-n (word, freq) pairs, lowest frequency first."""
        out = []
        for f in self.bucket_order:
            for idx in self.buckets[f]:
                out.append((self.idx2word[idx], f))
                if len(out) >= n:
                    return out
        return out

    def __len__(self):
        return len(self.idx2word)

    def __contains__(self, word):
        return word.lower() in self.word2idx


# --------------------------------------------------------------------------
# TrigramMarkov: trigram language model with bigram/unigram backoff
# --------------------------------------------------------------------------

_BOS = "<s>"   # sentence-start marker
_EOS = "</s>"  # sentence-end marker


class TrigramMarkov:
    """
    Trigram word model with bigram and unigram backoff, built on a
    FrequencyTapper vocabulary (so the vocab itself tracks word
    frequency as the model learns).
    """

    def __init__(self):
        self.vocab = FrequencyTapper()

        # counts keyed by id-tuples, so lookups are cheap ints not strings
        self.trigram_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self.bigram_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.unigram_counts: dict[int, int] = defaultdict(int)

        # trigram_next[(w1, w2)] -> {w3: count}, for fast sampling of continuations
        self.trigram_next: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.bigram_next: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # -- training ---------------------------------------------------------

    def fit(self, text: str):
        """Train on a blob of text, one sentence per line (or just one
        big blob - it will all be treated as one stream of sentences
        split on '.', '!', '?')."""
        import re
        sentences = text.split(".")
        for sent in sentences:
            words = [w for w in sent.lower().split()]
            if words:
                self.learn_sentence(words)
        return self

    def learn_sentence(self, words):
        """Update counts and vocab frequencies from a single tokenized sentence."""
        ids = self.vocab.bump_many(words)
        bos = self.vocab.bump(_BOS)
        eos = self.vocab.bump(_EOS)

        seq = [bos, bos] + ids + [eos]

        for i in range(len(seq)):
            self.unigram_counts[seq[i]] += 1

        for i in range(len(seq) - 1):
            w1, w2 = seq[i], seq[i + 1]
            self.bigram_counts[(w1, w2)] += 1
            self.bigram_next[w1][w2] += 1

        for i in range(len(seq) - 2):
            w1, w2, w3 = seq[i], seq[i + 1], seq[i + 2]
            self.trigram_counts[(w1, w2, w3)] += 1
            self.trigram_next[(w1, w2)][w3] += 1

    # -- generation ---------------------------------------------------------

    def _sample_next(self, w1, w2):
        """Sample the next word id given the previous two, backing off
        trigram -> bigram -> unigram as needed."""
        candidates = self.trigram_next.get((w1, w2))
        if candidates:
            return self._weighted_choice(candidates)

        candidates = self.bigram_next.get(w2)
        if candidates:
            return self._weighted_choice(candidates)

        if self.unigram_counts:
            return self._weighted_choice(self.unigram_counts)

        return None

    @staticmethod
    def _weighted_choice(counts: dict[int, int]):
        ids = list(counts.keys())
        weights = list(counts.values())
        return random.choices(ids, weights=weights, k=1)[0]

    def generate(self, max_words=30, seed_words=None):
        """Generate a sentence. Optionally seed it with the first one or
        two words (as a list of strings)."""
        bos = self.vocab.word2idx[_BOS]
        eos = self.vocab.word2idx[_EOS]

        if seed_words:
            seed_ids = [self.vocab.word2idx[w.lower()] for w in seed_words if w.lower() in self.vocab]
        else:
            seed_ids = []

        seq = [bos, bos] + seed_ids
        out_words = list(seed_ids)

        for _ in range(max_words):
            w1, w2 = seq[-2], seq[-1]
            nxt = self._sample_next(w1, w2)
  
            seq.append(nxt)
            out_words.append(nxt)

        return self.vocab.decode(out_words)

    # -- introspection --------------------------------------------------------

    def most_common_words(self, n=10):
        return self.vocab.most_frequent(n)

    def least_common_words(self, n=10):
        return self.vocab.least_frequent(n)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

with open(input("Filename: "), "r", encoding="utf-8") as file:
    SAMPLE_TEXT = file.read()


def main():
    model = TrigramMarkov()
    model.fit(SAMPLE_TEXT)

    print(f"Vocab size: {len(model.vocab)}")
    print("Most frequent words:", model.most_common_words(5))
    print("Least frequent words:", model.least_common_words(5))
    while True:
        prompt = input("USER: ")
        print(model.generate(max_words=550, seed_words=prompt.split()[:-2]))



if __name__ == "__main__":
    main()

