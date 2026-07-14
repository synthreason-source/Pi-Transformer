import numpy as np
import re


def rank_transform(x, scale, freq):
    return np.exp(scale * x) * (1 + np.sin(freq * x))


def apply_density_volatility(p, beta=2.0, eps=1e-8, vol_clip=10.0):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    density = p * (1.0 - p)
    volatility = np.abs(np.tan(1.0 - p))
    volatility = np.clip(volatility, 0.0, vol_clip)
    p = p * (1.0 + beta * density) * (1.0 + volatility)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return p / s


class GroundElement:
    def __init__(self, text, index):
        self.text = text
        self.index = index
        self.prev = None
        self.next = None
        self.weight = 0.0
        self.phase = 0.0
        self.rank_contribution = 0.0
        self.reduced = False


class MatroidMarkov:
    def __init__(self, beta=2.0, alpha=3.0, rank_amplitude=1.0, rank_freq=1.0, rank_phase=0.0,
                 use_hyponym_reduction=False, hyponym_levels=1, wordnet_penalty=0.5):
        self.beta = beta
        self.alpha = alpha
        self.rank_amplitude = rank_amplitude
        self.rank_freq = rank_freq
        self.rank_phase = rank_phase
        self.use_hyponym_reduction = use_hyponym_reduction
        self.hyponym_levels = hyponym_levels
        self.wordnet_penalty = wordnet_penalty
        self._hyponym_cache = {}
        self._wordnet_ready = False
        self.reduced_vocab = set()

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

    def _ensure_wordnet(self):
        if self._wordnet_ready:
            return
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        self._wordnet_ready = True

    def _hyponym_reduce_word(self, word):
        if word in self._hyponym_cache:
            return self._hyponym_cache[word]
        from nltk.corpus import wordnet as wn
        reduced = word
        synsets = wn.synsets(word)
        if synsets:
            synset = synsets[0]
            for _ in range(self.hyponym_levels):
                hypernyms = synset.hypernyms()
                if not hypernyms:
                    break
                synset = hypernyms[0]
            lemma_name = synset.lemmas()[0].name().replace("_", " ")
            reduced = lemma_name.lower()
        self._hyponym_cache[word] = reduced
        return reduced

    def hyponym_reduce_tokens(self, tokens):
        self._ensure_wordnet()
        reduced = []
        for w in tokens:
            rw = self._hyponym_reduce_word(w)
            reduced.append(rw)
            if rw != w:
                self.reduced_vocab.add(rw)
        return reduced

    def _tokenize(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentence_words = [[w.lower() for w in s.split() if w.strip()] for s in sentences if s.strip()]
        if self.use_hyponym_reduction:
            sentence_words = [self.hyponym_reduce_tokens(s) for s in sentence_words]
        words = [w for s in sentence_words for w in s]
        return sentence_words, words

    def _build_ground_set(self, prompt):
        toks = [w.lower() for w in prompt.split() if w.strip()]
        if self.use_hyponym_reduction:
            toks = self.hyponym_reduce_tokens(toks)
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
                el.rank_contribution = el.weight * self.rank_amplitude * np.sin(2 * np.pi * self.rank_freq * el.phase + self.rank_phase)
        return elements

    def _independence_bias(self, prompt):
        elements = self._build_ground_set(prompt)
        bias = {}
        for el in elements:
            bias[el.text] = bias.get(el.text, 0.0) + float(el.rank_contribution)
        return bias, elements

    def fit(self, text):
        self.reduced_vocab = set()
        sentence_words, words = self._tokenize(text)
        self.sentences = sentence_words
        self.vocab = sorted(set(words))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(self.vocab)

        incidence = np.zeros((n, n), dtype=float)
        for sent in sentence_words:
            for a, b in zip(sent[:-1], sent[1:]):
                incidence[self.word_to_idx[a], self.word_to_idx[b]] += 1
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

    def _apply_wordnet_penalty(self, p):
        if not self.use_hyponym_reduction or not self.reduced_vocab:
            return p
        p = np.asarray(p, dtype=float).copy()
        for w in self.reduced_vocab:
            if w in self.word_to_idx:
                p[self.word_to_idx[w]] *= self.wordnet_penalty
        s = p.sum()
        if s <= 0 or not np.isfinite(s):
            return None
        return p / s

    def rank_walk(self, start, length=50, bias=None):
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
            p = self._apply_wordnet_penalty(p)
            if p is None:
                break
            p = apply_density_volatility(p, beta=self.beta)
            if p is None:
                break
            current = np.random.choice(self.vocab, p=p)
            result.append(current)
        return result

    def similarity_walk(self, start, length=50, bias=None):
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
            sim = self._apply_wordnet_penalty(sim)
            if sim is None or sim.sum() == 0:
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

    def generate_from_prompt(self, prompt, candidates=30, length=60, p=0.51, seed=None):
        rng = np.random.default_rng(seed)
        bias, elements = self._independence_bias(prompt)
        seed_word = prompt.split()[-1].lower() if prompt.split() else str(rng.choice(self.vocab))
        if self.use_hyponym_reduction:
            seed_word = self._hyponym_reduce_word(seed_word)
        score, result = self.basis_generate(seed_word, candidates=candidates, length=length, bias=bias)
        result = self.intersperse_cognitive_tokens(result, p=p, rng=rng)
        return score, result, elements, bias

    def intersperse_cognitive_tokens(self, words, p=0.81, rng=None):
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
    with open(input('Filename: '), 'r', encoding='utf8') as f:
        corpus = f.read()

    model = MatroidMarkov(beta=2.0, alpha=1.0, rank_amplitude=11.0, rank_freq=4.0, rank_phase=0.1,
                          use_hyponym_reduction=True, hyponym_levels=2, wordnet_penalty=1.55)
    model.fit(corpus)

    while True:
        prompt = input('USER: ').strip()
        if not prompt:
            continue
        score, result, elements, bias = model.generate_from_prompt(prompt, candidates=30, length=600)
        print()
        print(' '.join(result))
        print('-' * 80)
