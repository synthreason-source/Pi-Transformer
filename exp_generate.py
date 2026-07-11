import numpy as np
import re


def exp_sin(x, scale, freq):
    return np.exp(scale * x) * (1 + np.sin(freq * x))


class WordNode:
    def __init__(self, text, index):
        self.text = text
        self.index = index
        self.prev = None
        self.next = None
        self.score = 0.0
        self.phase = 0.0
        self.modulated = 0.0


class InfluenceSpaceMarkov:
    def __init__(self, beta=2.0, alpha=3.0, sine_amplitude=1.0, sine_freq=1.0, sine_phase=0.0):
        self.beta = beta
        self.alpha = alpha
        self.sine_amplitude = sine_amplitude
        self.sine_freq = sine_freq
        self.sine_phase = sine_phase

    def _tokenize(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentence_words = [[w.lower() for w in s.split() if w.strip()] for s in sentences if s.strip()]
        words = [w for s in sentence_words for w in s]
        return sentence_words, words

    def _build_word_nodes(self, prompt):
        toks = [w.lower() for w in prompt.split() if w.strip()]
        nodes = [WordNode(w, i) for i, w in enumerate(toks)]
        for i in range(len(nodes) - 1):
            nodes[i].next = nodes[i + 1]
            nodes[i + 1].prev = nodes[i]
        n = len(nodes)
        if n:
            denom = max(1, n - 1)
            for i, node in enumerate(nodes):
                node.phase = i / denom
                node.score = 1.0
                node.modulated = node.score * self.sine_amplitude * np.sin(2 * np.pi * self.sine_freq * node.phase + self.sine_phase)
        return nodes

    def _prompt_bias(self, prompt):
        nodes = self._build_word_nodes(prompt)
        bias = {}
        for node in nodes:
            bias[node.text] = bias.get(node.text, 0.0) + float(node.modulated)
        return bias, nodes

    def fit(self, text):
        sentence_words, words = self._tokenize(text)
        self.sentences = sentence_words
        self.vocab = sorted(set(words))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(self.vocab)

        counts = np.zeros((n, n), dtype=float)
        for sent in sentence_words:
            for a, b in zip(sent[:-1], sent[1:]):
                i = self.word_to_idx[a]
                j = self.word_to_idx[b]
                counts[i, j] += 1
        self.counts = counts

        x1 = np.log1p(counts)
        Y = exp_sin(x1, scale=4.0, freq=0.5)
        Y[counts == 0] = 0
        self.Y = Y

        ymax = max(Y.max(), 1)
        x2 = Y / ymax
        Z = exp_sin(x2, scale=4.0, freq=2.0)
        Z[Y == 0] = 0
        self.Z = Z

        sums = Z.sum(axis=1, keepdims=True)
        self.P = np.divide(Z, sums, out=np.zeros_like(Z), where=sums != 0)

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

    def influence_generate(self, start, length=50, prompt_bias=None):
        start = start.lower()
        if start not in self.word_to_idx:
            start = np.random.choice(self.vocab)
        current = start
        result = [current]
        for _ in range(length):
            i = self.word_to_idx[current]
            p = self.P[i].copy()
            if prompt_bias:
                for w, b in prompt_bias.items():
                    if w in self.word_to_idx:
                        p[self.word_to_idx[w]] = min(1.0, p[self.word_to_idx[w]] * (1.0 + b))
            if p.sum() == 0:
                break
            p = p / p.sum()
            current = np.random.choice(self.vocab, p=p)
            result.append(current)
        return result

    def semantic_generate(self, start, length=50, prompt_bias=None):
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
            if prompt_bias:
                for w, b in prompt_bias.items():
                    if w in self.word_to_idx:
                        sim[self.word_to_idx[w]] = max(0.0, sim[self.word_to_idx[w]] * (1.0 + b))
            if sim.sum() == 0:
                break
            sim /= sim.sum()
            current = np.random.choice(self.vocab, p=sim)
            result.append(current)
        return result

    def sentence_vector(self, words):
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

    def intersection_generate(self, start, candidates=50, length=50, prompt_bias=None):
        semantic_path = self.semantic_generate(start, length, prompt_bias=prompt_bias)
        semantic_vector = self.sentence_vector(semantic_path)
        best = None
        for _ in range(candidates):
            candidate = self.influence_generate(start, length, prompt_bias=prompt_bias)
            candidate_vector = self.sentence_vector(candidate)
            score = self.cosine_similarity(candidate_vector, semantic_vector)
            if best is None or score > best[0]:
                best = (score, candidate)
        return best

    def generate_from_prompt(self, prompt, candidates=30, length=60):
        bias, nodes = self._prompt_bias(prompt)
        seed = prompt.split()[-1].lower() if prompt.split() else np.random.choice(self.vocab)
        score, result = self.intersection_generate(seed, candidates=candidates, length=length, prompt_bias=bias)
        return score, result, nodes, bias


if __name__ == '__main__':
    with open('singlekb.txt', 'r', encoding='utf8') as f:
        corpus = f.read()

    model = InfluenceSpaceMarkov(beta=2.0, alpha=3.0, sine_amplitude=1.0, sine_freq=1.0, sine_phase=0.0)
    model.fit(corpus)

    while True:
        prompt = input('USER: ').strip()
        if not prompt:
            continue
        score, result, nodes, bias = model.generate_from_prompt(prompt, candidates=30, length=600)
        print()
        print(' '.join(result))
        print('-' * 80)
