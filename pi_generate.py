import re
import argparse
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import random

class MemoryStore:
    def __init__(self):
        self.values = []

    def add(self, text):
        text = text.strip()
        if text:
            self.values.append(text)

    def retrieve(self, prompt):
        if not self.values:
            return ""
        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        best = self.values[0]
        best_score = -1
        for item in self.values:
            item_words = set(re.findall(r"\w+", item.lower()))
            score = len(prompt_words.intersection(item_words))
            if score > best_score:
                best_score = score
                best = item
        return best

class TrigramWordEngine:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.vocab = list()
        self.lru = OrderedDict()
        self.bitshift_clusters = defaultdict(set)
        self.latent_curve = []
        self.global_shift = 0
        self.seed_history = []

    def _build_clusters(self, contexts):
        total_occurrences = sum(sum(ctx.values()) for ctx in contexts.values())
        curve_exponent = 1.5 
        for word, ctx in contexts.items():
            local_freq = sum(ctx.values())
            prob = (local_freq / total_occurrences) ** (1 / curve_exponent)
            bucket = int(prob * 1024)
            self.bitshift_clusters[bucket << 2].add(word)
        
        sorted_keys = sorted(self.bitshift_clusters.keys())
        self.latent_curve = [self.bitshift_clusters[k] for k in sorted_keys]

    def _remap_vocab(self):
        if not self.latent_curve:
            return set()

        # advance to next curve position
        self.global_shift = (self.global_shift + 1) % len(self.latent_curve)

        # rotate the latent curve so future lookups see the updated ordering
        self.latent_curve = (
            self.latent_curve[self.global_shift:]
            + self.latent_curve[:self.global_shift]
        )

        # reset shift because the curve itself has been rotated
        self.global_shift = 0

        return self.latent_curve[0]

    def cluster_fallback(self, word):
        idx = self.global_shift % len(self.latent_curve)
        words = self.latent_curve[idx]
        return random.choice(list(words)) if words else np.random.choice(self.vocab)

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def train(self, texts):
        contexts = defaultdict(Counter)
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            if not tokens: continue
            all_tokens.extend(tokens)
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                self.bigrams[w1][w2] += 1
                self.lru[w1] = True; self.lru.move_to_end(w1)
            for i in range(len(tokens) - 2):
                self.trigrams[(tokens[i], tokens[i+1])][tokens[i+2]] += 1
            for i in range(1, len(tokens) - 1):
                contexts[tokens[i]][tokens[i-1]] += 1
                contexts[tokens[i]][tokens[i+1]] += 1
        
        self.vocab = list(set(all_tokens))
        self._build_clusters(contexts)
        while len(self.bigrams) > self.capacity:
            oldest, _ = self.lru.popitem(last=False)
            if oldest in self.bigrams: del self.bigrams[oldest]

    def _sample(self, counter, temperature=0.8):
        if not counter: return None
        words = list(counter.keys())
        counts = np.sqrt(np.array(list(counter.values()), dtype=np.float64))
        logits = np.log(counts + 1e-12) / max(float(temperature), 1e-6)
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(words, p=probs)

    def next_word(self, w1, w2, temp):
        if (w1, w2) in self.trigrams:
            nxt = self._sample(self.trigrams[(w1, w2)], temp)
            if nxt: return nxt
        if w2 in self.bigrams:
            nxt = self._sample(self.bigrams[w2], temp)
            if nxt: return nxt
        return self.cluster_fallback(w2)

    def generate_response(self, prompt, length=30, temperature=0.8):
        current_bucket = self._remap_vocab()

        if not current_bucket:
            current_bucket = set(self.vocab)

        prompt_tokens = self.tokenize(prompt)

        remapped_seeds = [
            t if t in current_bucket
            else random.choice(list(current_bucket))
            for t in prompt_tokens
        ]

        self.seed_history.append({
            "shift": self.global_shift,
            "seeds": remapped_seeds
        })
        
        if self.seed_history:
            historical = random.choice(self.seed_history)["seeds"]
            w1, w2 = historical[0], historical[1] if len(historical) > 1 else remapped_seeds[0]
        elif len(remapped_seeds) >= 2:
            w1, w2 = remapped_seeds[0], remapped_seeds[1]
        else:
            w1 = random.choice(list(current_bucket))
            w2 = random.choice(list(current_bucket))

        output = [w1, w2]
        for _ in range(length):
            nxt = self.next_word(w1, w2, temperature)
            if not nxt: break
            output.append(nxt)
            w1, w2 = w2, nxt
        return self.detokenize(output)

    def detokenize(self, tokens):
        out = []
        for tok in tokens:
            if not out: out.append(tok)
            elif re.fullmatch(r"[^\w\s]", tok): out[-1] += tok
            else: out.append(" " + tok)
        return "".join(out).strip()

def load_text_file(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        return [s.strip() for s in re.split(r"[.!?\n]+", f.read()) if s.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    engine = TrigramWordEngine()
    engine.train(load_text_file(args.training_file))
    while True:
        user = input("USER: ").strip()
        if not user: break
        print("AI:", engine.generate_response(user, args.length, args.temperature))

if __name__ == "__main__":
    main()
