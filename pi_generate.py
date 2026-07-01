import re
import argparse
from collections import defaultdict, Counter, OrderedDict
import numpy as np

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
        self.vocab = set()
        self.lru = OrderedDict()
        self.cluster_signatures = {}
        self.cluster_words = defaultdict(set)
        # New cluster storage
        self.bitshift_clusters = defaultdict(set)

    def _build_clusters(self, contexts):
        next_cluster_id = 0
        total_occurrences = sum(sum(ctx.values()) for ctx in contexts.values())
        
        # Power-law exponent to flatten the curve
        # A value > 1 pushes more words into the "high-probability" bin
        curve_exponent = 1.5 

        for word, ctx in contexts.items():
            # [Standard clustering logic remains here]
            
            # CURVE-BASED PROBABILITY BITSHIFTING
            local_freq = sum(ctx.values())
            prob = (local_freq / total_occurrences) ** (1 / curve_exponent)
            
            # Convert to a discrete logarithmic bucket
            # The curve ensures that the distance between high-prob and 
            # low-prob words is distributed more effectively
            bucket = int(prob * 1024)
            bitshift_key = bucket << 2
            
            self.bitshift_clusters[bitshift_key].add(word)

    def cluster_fallback(self, word):
        # 1. Try standard cluster
        for cid, words in self.cluster_words.items():
            if word in words:
                return np.random.choice(list(words))
        
        # 2. Try bitshift cluster fallback
        for key, words in self.bitshift_clusters.items():
            if word in words:
                return np.random.choice(list(words))
                
        return self.random_word()

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def train(self, texts):
        contexts = defaultdict(Counter)

        for text in texts:
            tokens = self.tokenize(text)
            if len(tokens) < 1:
                continue

            self.vocab.update(tokens)

            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                self.bigrams[w1][w2] += 1
                self.lru[w1] = True
                self.lru.move_to_end(w1)

            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
                self.trigrams[(w1, w2)][w3] += 1

            for i in range(1, len(tokens) - 1):
                word = tokens[i]
                left = tokens[i - 1]
                right = tokens[i + 1]
                contexts[word][left] += 1
                contexts[word][right] += 1

        self._build_clusters(contexts)

        while len(self.bigrams) > self.capacity:
            oldest, _ = self.lru.popitem(last=False)
            if oldest in self.bigrams:
                del self.bigrams[oldest]

    

    def random_word(self):
        if not self.vocab:
            return ""
        return np.random.choice(list(self.vocab))

    def _sample(self, counter, temperature=0.8):
        if not counter:
            return None
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
            if nxt is not None:
                return nxt

        if current_word in self.bigrams:
            nxt = self._sample(self.bigrams[current_word], temperature)
            if nxt is not None:
                return nxt

        return self.cluster_fallback(current_word)

    def generate_response(self, memory_store, prompt, length=30, temperature=0.8):
        memory = memory_store.retrieve(prompt)
        seed_tokens = self.tokenize(memory)

        if len(seed_tokens) >= 2:
            w1, w2 = seed_tokens[-2], seed_tokens[-1]
        elif len(seed_tokens) == 1:
            w1, w2 = seed_tokens[0], seed_tokens[0]
        else:
            w1, w2 = self.random_word(), self.random_word()

        if not w1:
            w1 = self.random_word()
        if not w2:
            w2 = self.random_word()

        output = [w1, w2]

        for _ in range(length):
            nxt = self.next_word(w1, w2, temperature)
            if not nxt:
                break
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
    sentences = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in sentences if s.strip()]

def main():
    parser = argparse.ArgumentParser(description="Word-level trigram text generator")
    parser.add_argument("training_file", nargs="?", help="Path to UTF-8 training text file")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    if not args.training_file:
        args.training_file = input("Training file: ").strip()

    text_data = load_text_file(args.training_file)

    memory_store = MemoryStore()
    for text in text_data:
        memory_store.add(text)

    engine = TrigramWordEngine(capacity=100000)
    engine.train(text_data)

    print(f"Loaded {len(text_data)} memories.")
    print("Model ready.")
    print("Press ENTER on an empty line to quit.")
    print()

    if args.no_interactive:
        prompt = args.prompt or ""
        print(engine.generate_response(memory_store, prompt, length=args.length, temperature=args.temperature))
        return

    while True:
        try:
            user_input = input("USER: ").strip()
        except EOFError:
            break

        if not user_input:
            break

        response = engine.generate_response(
            memory_store,
            user_input,
            length=args.length,
            temperature=args.temperature
        )

        print()
        print("AI:", response)
        print()

if __name__ == "__main__":
    main()
