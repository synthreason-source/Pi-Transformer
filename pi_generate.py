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
        self.cluster_signatures = {}
        self.cluster_words = defaultdict(set)
        self.bitshift_clusters = defaultdict(set)
        # Added state for linear remapping
        self.latent_curve = []
        self.global_shift = 0

    def _build_clusters(self, contexts):
        total_occurrences = sum(sum(ctx.values()) for ctx in contexts.values())
        curve_exponent = 1.5 

        for word, ctx in contexts.items():
            local_freq = sum(ctx.values())
            prob = (local_freq / total_occurrences) ** (1 / curve_exponent)
            bucket = int(prob * 1024)
            bitshift_key = bucket << 2
            self.bitshift_clusters[bitshift_key].add(word)
        
        # Build linear curve for remapping
        sorted_keys = sorted(self.bitshift_clusters.keys())
        self.latent_curve = [self.bitshift_clusters[k] for k in sorted_keys]

    def _remap_vocab(self):
        # Linearly shift the vocabulary window
        self.global_shift = (self.global_shift + 1) % len(self.latent_curve)

    def cluster_fallback(self, word):
        # Use remapped vocab based on global shift
        idx = self.global_shift % len(self.latent_curve)
        words = self.latent_curve[idx]
        if words:
            return random.choice(list(words))
        return np.random.choice(self.vocab) if self.vocab else ""

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
                self.lru[w1] = True
                self.lru.move_to_end(w1)

            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
                self.trigrams[(w1, w2)][w3] += 1

            for i in range(1, len(tokens) - 1):
                contexts[tokens[i]][tokens[i - 1]] += 1
                contexts[tokens[i]][tokens[i + 1]] += 1
        
        self.vocab = list(set(all_tokens))
        self._build_clusters(contexts)

        while len(self.bigrams) > self.capacity:
            oldest, _ = self.lru.popitem(last=False)
            if oldest in self.bigrams: del self.bigrams[oldest]

    def _sample(self, counter, temperature=0.8):
        if not counter: return None
        words = list(counter.keys())
        counts = np.array(list(counter.values()), dtype=np.float64)
        counts = np.sqrt(counts)
        temperature = max(float(temperature), 1e-6)
        logits = np.log(counts + 1e-12) / temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return np.random.choice(words, p=probs)

    def next_word(self, previous_word, current_word, temperature=0.8):
        trigram_key = (previous_word, current_word)
        if trigram_key in self.trigrams:
            nxt = self._sample(self.trigrams[trigram_key], temperature)
            if nxt: return nxt
        if current_word in self.bigrams:
            nxt = self._sample(self.bigrams[current_word], temperature)
            if nxt: return nxt
        return self.cluster_fallback(current_word)

    def generate_response(self, prompt, length=30, temperature=0.8):
        # 1. Advance the linear vocabulary shift
        self._remap_vocab()
        
        # 2. Get the current active vocabulary bucket
        current_bucket = self.latent_curve[self.global_shift % len(self.latent_curve)]
        
        # 3. Prompt Remapping: Transform prompt tokens to match the current bucket
        prompt_tokens = self.tokenize(prompt)
        remapped_seeds = []
        
        if prompt_tokens and current_bucket:
            # Map each prompt token to a relevant token from the current bucket
            # If the token exists in the system, we align it; otherwise, we map to bucket content
            for token in prompt_tokens:
                if token in current_bucket:
                    remapped_seeds.append(token)
                else:
                    # Inject a token from the current remapped bucket to align the seed
                    remapped_seeds.append(random.choice(list(current_bucket)))
        
        # 4. Use remapped_seeds to initialize the trigram generator
        if len(remapped_seeds) >= 2:
            w1, w2 = remapped_seeds[0], remapped_seeds[1]
        else:
            w1 = random.choice(list(current_bucket)) if current_bucket else random.choice(self.vocab)
            w2 = random.choice(list(current_bucket)) if current_bucket else random.choice(self.vocab)

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
            if not out:
                out.append(tok)
                continue
            if re.fullmatch(r"[^\w\s]", tok):
                out[-1] += tok
            else:
                out.append(" " + tok)
        return "".join(out).strip()

def load_text_file(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    text_data = load_text_file(args.training_file)
    engine = TrigramWordEngine()
    engine.train(text_data)

    print("Model ready. Type prompt:")
    while True:
        user_input = input("USER: ").strip()
        if not user_input: break
        print("AI:", engine.generate_response(user_input, length=args.length, temperature=args.temperature))

if __name__ == "__main__":
    main()
