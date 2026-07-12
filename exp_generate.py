import numpy as np
import re

def rank_transform(x, scale, freq):
    """Same shape as the original exp_sin: exponential-sine weighting,
    now framed as a rank-inflation transform over a ground set."""
    return np.exp(scale * x) * (1 + np.sin(freq * x))

class GroundElement:
    """One element of the ground set E (a word occurrence in the prompt)."""
    def __init__(self, text, index):
        self.text = text
        self.index = index
        self.prev = None
        self.next = None
        self.weight = 0.0
        self.phase = 0.0
        self.rank_contribution = 0.0

class MatroidMarkov:
    """
    Reframing of InfluenceSpaceMarkov in matroid-theory vocabulary:

    - ground set E : the corpus vocabulary
    - independent sets I : subsets of transitions kept during generation
    - rank function r(S) : exponential-sine weighted transition strength
    - bases B : the best-scoring generated sequences

    Note: this is a *relabeling* of the same computation, not a literal
    matroid (no independence axioms are checked). The math is identical
    to the original file.
    """

    def __init__(self, beta=2.0, alpha=3.0, rank_amplitude=1.0, rank_freq=1.0, rank_phase=0.0):
        self.beta = beta
        self.alpha = alpha
        self.rank_amplitude = rank_amplitude
        self.rank_freq = rank_freq
        self.rank_phase = rank_phase

        self.cognitive_tokens = [
            "attention", "memory", "reasoning", "perception", "judgment",
            "inference", "belief", "concept", "awareness", "focus",
            "thought", "learning", "knowledge", "understanding", "recognition",
            "association", "analysis", "synthesis", "reflection", "intuition",
            "evaluation", "comparison", "abstraction", "imagination", "prediction",
            "planning", "decision", "interpretation", "categorization", "comprehension",
            "curiosity", "insight", "observation", "recall", "anticipation",
            "deliberation", "representation", "generalization", "adaptation", "problem",
            "solution", "strategy", "expectation", "context", "meaning",
            "intent", "logic", "pattern", "model", "conclusion"
        ]

    def _tokenize(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentence_words = [[w.lower() for w in s.split() if w.strip()] for s in sentences if s.strip()]
        words = [w for s in sentence_words for w in s]
        return sentence_words, words

    def _build_ground_set(self, prompt):
        """Build the ground set E from the prompt's tokens, as a doubly
        linked chain, with each element's rank contribution computed from
        its position (phase) in the sequence."""
        toks = [w.lower() for w in prompt.split() if w.strip()]
        elements = [GroundElement(w, i) for i, w in enumerate(toks)]
        for i in range(len(elements) - 1):
            elements[i].next = elements[i + 1]
            elements[i + 1].prev = elements[i]
        n = len(elements)
        if n:
            denom = max(1, n - 1)
            for i, el in enumerate(elements):
                el.phase = i / denom
                el.weight = 1.0
                el.rank_contribution = el.weight * self.rank_amplitude * np.sin(
                    2 * np.pi * self.rank_freq * el.phase + self.rank_phase
                )
        return elements

    def _independence_bias(self, prompt):
        """Aggregate rank contributions per distinct word -> a bias dict
        used to favor certain ground-set elements during generation."""
        elements = self._build_ground_set(prompt)
        bias = {}
        for el in elements:
            bias[el.text] = bias.get(el.text, 0.0) + float(el.rank_contribution)
        return bias, elements

    def fit(self, text):
        sentence_words, words = self._tokenize(text)
        self.sentences = sentence_words
        self.vocab = sorted(set(words))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(self.vocab)

        incidence = np.zeros((n, n), dtype=float)
        for sent in sentence_words:
            for a, b in zip(sent[:-1], sent[1:]):
                i = self.word_to_idx[a]
                j = self.word_to_idx[b]
                incidence[i, j] += 1
        self.incidence = incidence

        x1 = np.log1p(incidence)
        rank_weights = rank_transform(x1, scale=4.0, freq=0.5)
        rank_weights[incidence == 0] = 0
        self.rank_weights = rank_weights

        rmax = max(rank_weights.max(), 1)
        x2 = rank_weights / rmax
        normalized_rank = rank_transform(x2, scale=4.0, freq=2.0)
        normalized_rank[rank_weights == 0] = 0
        self.normalized_rank = normalized_rank

        sums = normalized_rank.sum(axis=1, keepdims=True)
        self.P = np.divide(normalized_rank, sums, out=np.zeros_like(normalized_rank), where=sums != 0)

        window = 4
        cooc = np.zeros((n, n), dtype=float)
        for sent in sentence_words:
            for i, w in enumerate(sent):
                wi = self.word_to_idx[w]
                left = max(0, i - window)
                right = min(len(sent), i + window + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    wj = self.word_to_idx[sent[j]]
                    cooc[wi, wj] += 1
        self.cooc = cooc

        norms = np.linalg.norm(cooc, axis=1, keepdims=True)
        norms[norms == 0] = 1
        E = cooc / norms
        self.embedding = E
        self.cosine = E @ E.T
        return self

    def rank_walk(self, start, length=50, bias=None):
        """Random walk over the transition distribution, biased toward
        elements with higher aggregated rank contribution."""
        start = start.lower()
        if start not in self.word_to_idx:
            start = np.random.choice(self.vocab)
        current = start
        result = [current]
        for _ in range(length):
            i = self.word_to_idx[current]
            p = self.P[i].copy()
            if bias:
                for w, b in bias.items():
                    if w in self.word_to_idx:
                        p[self.word_to_idx[w]] = min(1.0, p[self.word_to_idx[w]] * (1.0 + b))
            if p.sum() == 0:
                break
            p = p / p.sum()
            current = np.random.choice(self.vocab, p=p)
            result.append(current)
        return result

    def similarity_walk(self, start, length=50, bias=None):
        """Random walk over cosine similarity in embedding space."""
        start = start.lower()
        if start not in self.word_to_idx:
            start = np.random.choice(self.vocab)
        current = start
        result = [current]
        for _ in range(length):
            i = self.word_to_idx[current]
            sim = self.cosine[i].copy()
            sim[i] = 0
            sim = np.maximum(sim, 0)
            if bias:
                for w, b in bias.items():
                    if w in self.word_to_idx:
                        sim[self.word_to_idx[w]] = max(0.0, sim[self.word_to_idx[w]] * (1.0 + b))
            if sim.sum() == 0:
                break
            sim /= sim.sum()
            current = np.random.choice(self.vocab, p=sim)
            result.append(current)
        return result

    def sequence_vector(self, words):
        vectors = []
        for w in words:
            if w in self.word_to_idx:
                vectors.append(self.embedding[self.word_to_idx[w]])
        if not vectors:
            return np.zeros(len(self.vocab))
        return np.mean(vectors, axis=0)

    def cosine_similarity(self, a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def basis_generate(self, start, candidates=50, length=50, bias=None):
        """Generate several candidate walks and keep the one whose vector
        is closest to the similarity-walk target."""
        target_path = self.similarity_walk(start, length, bias=bias)
        target_vector = self.sequence_vector(target_path)
        best = None
        for _ in range(candidates):
            candidate = self.rank_walk(start, length, bias=bias)
            candidate_vector = self.sequence_vector(candidate)
            score = self.cosine_similarity(candidate_vector, target_vector)
            if best is None or score > best[0]:
                best = (score, candidate)
        return best

    def spread_prob_word_pairs(self, pairs, matrix, frag_count=3):
        out = []
        prev_prob = 0.0
        rows, cols = matrix.shape

        for i, (prob, word) in enumerate(pairs):
            base = 0.5 * prev_prob + 0.5 * prob
            frags = []
            for k in range(frag_count):
                frac = (k + 1) / (frag_count + 1)
                frag_prob = base * frac + prob * (1 - frac)
                frag_word = f"{word}_{k}"
                frags.append((frag_prob, frag_word))

            for k, (frag_prob, frag_word) in enumerate(frags):
                r = (len(out) + k) % rows
                c = (len(out) * frag_count + k) % cols
                matrix[r, c] = frag_prob
                out.append((frag_prob, frag_word))

            prev_prob = prob

        return out, matrix

    def generate_from_prompt(self, prompt, candidates=30, length=60, p=0.01, seed=None):
        rng = np.random.default_rng(seed)
        bias, elements = self._independence_bias(prompt)
        seed_word = prompt.split()[-1].lower() if prompt.split() else str(rng.choice(self.vocab))
        score, result = self.basis_generate(seed_word, candidates=candidates, length=length, bias=bias)
        result = self.intersperse_cognitive_tokens(result, p=p, rng=rng)
        return score, result, elements, bias

    def intersperse_cognitive_tokens(self, words, p=0.01, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        result = []
        for i, word in enumerate(words):
            result.append(word)
            if i < len(words) - 1 and rng.random() < p:
                token_idx = rng.integers(len(self.cognitive_tokens))
                result.append(self.cognitive_tokens[token_idx])
        return result

if __name__ == '__main__':
    with open(input("Filename: "), 'r', encoding='utf8') as f:
        corpus = f.read()

    model = MatroidMarkov(beta=2.0, alpha=3.0, rank_amplitude=1.0, rank_freq=4.0, rank_phase=0.1)
    model.fit(corpus)

    while True:
        prompt = input('USER: ').strip()
        if not prompt:
            continue
        score, result, elements, bias = model.generate_from_prompt(prompt, candidates=30, length=600)
        print()
        print(' '.join(result))
        print('-' * 80)
