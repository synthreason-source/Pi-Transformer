from pathlib import Path
import re
import random
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
import gradio as gr


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
        self.windows = []
        start = 0
        while start < len(words):
            chunk = words[start:start + window_size]
            self.windows.append({
                "words": chunk,
                "counter": Counter(w.lower() for w in chunk)
            })
            start += overlap

    def search(self, query, breadth=10):
        terms = re.findall(r"\w+", query.lower())
        scored = []
        for win in self.windows:
            score = sum(win["counter"].get(t, 0) for t in terms)
            scored.append((score, win))

        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for _, win in scored[:breadth]:
            result.extend(win["words"])

        return result


class TrigramMarkov:
    def __init__(self, words):
        words = [w.lower() for w in words]

        self.words = words
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

    def sample_counter(self, counter, temperature=0.9, top_k=800):
        if not counter:
            return None

        items = list(counter.items())

        tokens = torch.tensor([k for k, _ in items])
        counts = torch.tensor([v for _, v in items], dtype=torch.float32)

        logits = torch.log(counts + 1e-9) / max(temperature, 1e-6)

        if top_k and len(logits) > top_k:
            vals, idx = torch.topk(logits, top_k)
            logits = vals
            tokens = tokens[idx]

        probs = F.softmax(logits, dim=0)

        return tokens[torch.multinomial(probs, 1).item()].item()

    def next_token(self, context, temperature=0.9, top_k=800):
        if len(context) >= 2:
            key = (context[-2], context[-1])
            if key in self.trigram:
                tok = self.sample_counter(self.trigram[key], temperature, top_k)
                if tok is not None:
                    return tok

        if len(context) >= 1:
            key = (context[-1],)
            if key in self.bigram:
                tok = self.sample_counter(self.bigram[key], temperature, top_k)
                if tok is not None:
                    return tok

        return self.sample_counter(self.unigram, temperature, top_k)


def load_words(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read().split()
cognitive_words = [
    "think",
    "reason",
    "understand",
    "analyze",
    "infer",
    "deduce",
    "evaluate",
    "judge",
    "compare",
    "predict",
    "reflect",
    "remember",
    "recall",
    "recognize",
    "perceive",
    "interpret",
    "conceptualize",
    "abstract",
    "generalize",
    "focus",
    "attention",
    "awareness",
    "knowledge",
    "belief",
    "decision"
]

def generate_text(corpus_file, prompt):
    corpus_path = corpus_file.name if corpus_file else "corpus.txt"

    corpus_words = load_words(corpus_path)

    indexer = CorpusIndexer(corpus_words)

    prompt = (prompt or "").strip().lower()
    if not prompt:
        prompt = "the"

    filtered = indexer.search(prompt, breadth=10)
    if len(filtered) < 100:
        filtered = corpus_words

    model = TrigramMarkov(filtered)

    output_words = prompt.split()

    if len(output_words) < 2:
        output_words.insert(0, random.choice(corpus_words).lower())

    refresh_interval = 25
    trailing_tokens = 20
    target_length = 400

    while len(output_words) < target_length:

        context_ids = model.vocab.encode(output_words[-2:])

        if len(context_ids) < 2:
            next_word = random.choice(filtered).lower()
        else:
            nxt = model.next_token(context_ids)
            if nxt is None:
                next_word = random.choice(filtered).lower()
            else:
                next_word = model.vocab.idx2word[nxt]

        output_words.append(next_word)

        if len(output_words) % refresh_interval == 0:
            
            trailing = " ".join(output_words[-trailing_tokens:])
            query = f" {cognitive_words[len(output_words)%len(cognitive_words)]} {prompt}"

            refreshed = indexer.search(query, breadth=3)

            if len(refreshed) > 100:
                filtered = refreshed
                model = TrigramMarkov(filtered)

    text = " ".join(output_words)

    return text

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.File(label="Upload corpus.txt", file_types=[".txt"]),
        gr.Textbox(label="Prompt")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Automatic Trailing Context Markov"
)

if __name__ == "__main__":
    demo.launch()
