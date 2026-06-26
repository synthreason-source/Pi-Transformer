from pathlib import Path
import re
import random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


class Vocab:
    def __init__(self, words):
        self.word2idx = {}
        self.idx2word = []
        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)

    def __len__(self):
        return len(self.idx2word)

    def encode(self, words):
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def decode(self, ids):
        return " ".join(self.idx2word[i] for i in ids)


class CorpusIndexer:
    def __init__(self, words, window_size=400, overlap=200):
        window_size = int(window_size)
        overlap = int(overlap)
        self.windows = []
        start = 0
        words = [w.lower() for w in words]
        while start < len(words):
            chunk = words[start:start + window_size]
            self.windows.append({
                "words": chunk,
                "counter": Counter(chunk)
            })
            start += overlap

    def search(self, query, breadth=10):
        breadth = int(breadth)
        terms = re.findall(r"\w+", (query or "").lower())
        scored = []
        for win in self.windows:
            score = sum(win["counter"].get(t, 0) for t in terms)
            scored.append((score, win))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for _, win in scored[:breadth]:
            result.extend(win["words"])
        return result

    def proximal_objects(self, query, breadth=5, min_count=2):
        breadth = int(breadth)
        min_count = int(min_count)
        terms = re.findall(r"\w+", (query or "").lower())
        scored = []
        for win in self.windows:
            score = sum(win["counter"].get(t, 0) for t in terms)
            scored.append((score, win))
        scored.sort(key=lambda x: x[0], reverse=True)
        prox = Counter()
        for _, win in scored[:breadth]:
            for w, c in win["counter"].items():
                if c >= min_count and w.isalpha() and len(w) > 2:
                    prox[w.lower()] += c
        return prox


class ClusterSteerer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.cluster_ids = None
        self.cluster_members = defaultdict(list)
        self.cluster_scores = None
        self.cluster_centers = None

    def _word_vector(self, word):
        v = np.zeros(3, dtype=np.float32)
        v[0] = len(word)
        v[1] = sum(ord(c) for c in word) % 1024
        v[2] = sum(1 for c in word if c in "aeiou")
        return v

    def fit_words(self, words, n_clusters=4):
        words = [w for w in words if w in self.vocab.word2idx]
        if not words:
            return self
        vecs = np.vstack([self._word_vector(w) for w in words]).astype(np.float32)
        n_clusters = max(1, min(int(n_clusters), len(words)))
        if KMeans is None or n_clusters == 1:
            self.cluster_ids = np.zeros(len(words), dtype=int)
            self.cluster_members.clear()
            for i, w in enumerate(words):
                self.cluster_members[0].append(w)
            self.cluster_centers = vecs.mean(axis=0, keepdims=True)
            return self
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        self.cluster_ids = km.fit_predict(vecs)
        self.cluster_centers = km.cluster_centers_
        self.cluster_members.clear()
        for i, cid in enumerate(self.cluster_ids):
            self.cluster_members[int(cid)].append(words[i])
        return self

    def score_clusters(self, context_words):
        if self.cluster_centers is None or len(self.cluster_members) == 0:
            return None
        if not context_words:
            scores = np.ones(len(self.cluster_centers), dtype=np.float32)
            self.cluster_scores = scores / scores.sum()
            return self.cluster_scores
        ctx_vec = np.mean([self._word_vector(w) for w in context_words if w], axis=0)
        dists = np.linalg.norm(self.cluster_centers - ctx_vec[None, :], axis=1)
        scores = 1.0 / (dists + 1e-6)
        scores = scores / scores.sum()
        self.cluster_scores = scores
        return scores

    def allowed_words(self, top_clusters=12):
        if self.cluster_scores is None:
            return None
        top_clusters = max(1, int(top_clusters))
        chosen = np.argsort(-self.cluster_scores)[:top_clusters]
        allowed = set()
        for cid in chosen:
            allowed.update(self.cluster_members.get(int(cid), []))
        return allowed


class TrigramMarkov:
    def __init__(self, words):
        words = [w.lower() for w in words]

        self.vocab = Vocab(words)
        ids = self.vocab.encode(words)

        self.trigram = defaultdict(Counter)
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()

        for i, tok in enumerate(ids):
            self.unigram[tok] += 1
            if i >= 1:
                self.bigram[(ids[i - 1],)][tok] += 1
            if i >= 2:
                self.trigram[(ids[i - 2], ids[i - 1])][tok] += 1

        total = sum(self.unigram.values())
        self.baseline_probs = {tok: count / total for tok, count in self.unigram.items()} if total > 0 else {}

    def build_3x3_probability_cell(self, counter):
        items = list(counter.items())
        if not items:
            return None, None, None, None
        items = sorted(items, key=lambda kv: kv[1], reverse=True)[:9]
        vals = np.array([float(v) for _, v in items], dtype=np.float64)
        if len(vals) < 9:
            vals = np.pad(vals, (0, 9 - len(vals)), mode="constant", constant_values=0.0)
        cell = vals.reshape(3, 3)
        flat = cell.flatten()
        order = np.argsort(-flat)
        seq = flat[order]
        diag_seq = np.diag(seq)
        ordered_tokens = [items[i][0] for i in order if i < len(items)]
        return cell, seq, diag_seq, ordered_tokens

    def sample_counter(self, counter, temperature=0.9, top_k=800, baseline_weight=0.3, allowed_ids=None, min_support=0.0):
        if not counter:
            return None

        cell, seq, diag_seq, ordered_tokens = self.build_3x3_probability_cell(counter)

        if ordered_tokens:
            limited_counter = Counter({tok: counter[tok] for tok in ordered_tokens if tok in counter})
        else:
            limited_counter = counter

        items = list(limited_counter.items())
        if not items:
            return None

        tokens = torch.tensor([k for k, _ in items], dtype=torch.long)
        counts = torch.tensor([v for _, v in items], dtype=torch.float32)
        context_probs = counts / counts.sum()

        baseline_vals = torch.tensor(
            [self.baseline_probs.get(int(t), 1e-9) for t in tokens],
            dtype=torch.float32,
        )
        if baseline_vals.sum() > 0:
            baseline_vals = baseline_vals / baseline_vals.sum()
        else:
            baseline_vals = torch.full_like(context_probs, 1.0 / len(context_probs))

        bw = float(min(max(baseline_weight, 0.0), 1.0))
        combined_probs = (1 - bw) * context_probs + bw * baseline_vals

        if allowed_ids is not None:
            allowed_ids = set(int(x) for x in allowed_ids)
            mask = torch.tensor([1.0 if int(t) in allowed_ids else 0.0 for t in tokens], dtype=torch.float32)
            support = combined_probs * mask
            total_support = float(support.sum().item())
            if total_support <= float(min_support):
                return None
            combined_probs = support / total_support

        if diag_seq is not None and diag_seq.size > 0:
            diag_scale = float(np.trace(diag_seq))
            if diag_scale > 0:
                combined_probs = combined_probs * (1.0 + torch.tensor(diag_scale, dtype=torch.float32) * 1e-6)
                combined_probs = combined_probs / combined_probs.sum()

        logits = torch.log(combined_probs + 1e-9) / max(float(temperature), 1e-6)

        if top_k and len(logits) > top_k:
            vals, idx = torch.topk(logits, int(top_k))
            logits = vals
            tokens = tokens[idx]

        probs = F.softmax(logits, dim=0)
        return tokens[torch.multinomial(probs, 1).item()].item()

    def sample_baseline(self):
        if not self.baseline_probs:
            return None
        toks = torch.tensor(list(self.baseline_probs.keys()), dtype=torch.long)
        probs = torch.tensor(list(self.baseline_probs.values()), dtype=torch.float32)
        probs = probs / probs.sum()
        return toks[torch.multinomial(probs, 1).item()].item()

    def next_token(self, context, temperature=0.9, top_k=800, baseline_weight=0.3, allowed_ids=None, min_support=0.0):
        if len(context) >= 2:
            key = (context[-2], context[-1])
            if key in self.trigram:
                tok = self.sample_counter(self.trigram[key], temperature, top_k, baseline_weight, allowed_ids, min_support)
                if tok is not None:
                    return tok

        if len(context) >= 1:
            key = (context[-1],)
            if key in self.bigram:
                tok = self.sample_counter(self.bigram[key], temperature, top_k, baseline_weight, allowed_ids, min_support)
                if tok is not None:
                    return tok

        tok = self.sample_counter(self.unigram, temperature, top_k, baseline_weight, allowed_ids, min_support)
        if tok is not None:
            return tok

        return self.sample_baseline()


def load_words(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().split()


cognitive_words = [
    "sense", "detect", "observe", "notice", "perceive", "recognize", "identify", "attend", "focus", "awareness",
    "monitor", "track", "encode", "remember", "recall", "retrieve", "associate", "connect", "compare", "categorize",
    "generalize", "abstract", "conceptualize", "interpret", "understand", "comprehend", "reason", "infer", "deduce",
    "analyze", "evaluate", "estimate", "predict", "anticipate", "simulate", "imagine", "hypothesize", "explore",
    "plan", "strategize", "prioritize", "choose", "decide", "act", "verify", "reflect", "learn", "adapt", "optimize",
    "improve",
]


def generate_text(corpus_file, dataset_object_file, prompt, baseline_weight=0.3, object_gate=0.15, prox_breadth=5, temperature=0.9, target_length=400, cluster_count=42, top_clusters=12):
    corpus_path = corpus_file.name if corpus_file else "corpus.txt"
    object_path = dataset_object_file.name if dataset_object_file else "dataset_object.txt"

    corpus_words = load_words(corpus_path)
    object_words = load_words(object_path)

    indexer = CorpusIndexer(corpus_words)
    object_indexer = CorpusIndexer(object_words)

    global_model = TrigramMarkov(corpus_words)

    prompt = (prompt or "").strip().lower()
    if not prompt:
        prompt = "the"

    filtered = indexer.search(prompt, breadth=10)
    if len(filtered) < 100:
        filtered = corpus_words

    model = TrigramMarkov(filtered)
    model.baseline_probs = global_model.baseline_probs

    output_words = prompt.split()
    if len(output_words) < 2:
        output_words.insert(0, random.choice(corpus_words).lower())

    cluster_steerer = ClusterSteerer(model.vocab)

    refresh_interval = 15
    trailing_tokens = 5
    target_length = int(target_length)

    while len(output_words) < target_length:
        context_ids = model.vocab.encode(output_words[-2:])

        allowed_ids = None
        prox_query = " ".join(output_words[-trailing_tokens:] + [prompt])
        prox_words = object_indexer.proximal_objects(prox_query, breadth=prox_breadth)

        if prox_words:
            prox_tokens = [w for w in prox_words.keys() if w in model.vocab.word2idx]
            cluster_steerer.fit_words(prox_tokens, n_clusters=cluster_count)
            cluster_steerer.score_clusters(output_words[-trailing_tokens:] + prompt.split())

            allowed_words = cluster_steerer.allowed_words(top_clusters=top_clusters)
            if allowed_words:
                allowed_ids = {
                    model.vocab.word2idx[w]
                    for w in allowed_words
                    if w in model.vocab.word2idx
                }

            if allowed_ids:
                allowed_ids = {
                    tid for tid in allowed_ids
                    if prox_words.get(model.vocab.idx2word[tid], 0) / max(sum(prox_words.values()), 1) >= object_gate
                }
                if not allowed_ids:
                    allowed_ids = None

        if len(context_ids) < 2:
            next_word = random.choice(filtered).lower()
        else:
            nxt = model.next_token(
                context_ids,
                temperature=temperature,
                baseline_weight=baseline_weight,
                allowed_ids=allowed_ids,
                min_support=0.0
            )
            next_word = model.vocab.idx2word[nxt] if nxt is not None else random.choice(filtered).lower()

        output_words.append(next_word)

        if len(output_words) % refresh_interval == 0:
            query = f" {cognitive_words[len(output_words) % len(cognitive_words)]} {prompt}"
            _ = indexer.search(query, breadth=3)

    return " ".join(output_words)


demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.File(label="Upload corpus.txt", file_types=[]),
        gr.File(label="Upload dataset_object.txt", file_types=[]),
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.05, label="Baseline weight"),
        gr.Slider(minimum=0.0, maximum=1.0, value=1.00, step=0.01, label="Object gate"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.9, step=0.05, label="Temperature"),
        gr.Slider(minimum=50, maximum=2000, value=400, step=50, label="Target length"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Guided Walk Markov (Baseline-Blended + 3x3 Diagonalization + Cluster Steering)",
)

if __name__ == "__main__":
    demo.launch()
