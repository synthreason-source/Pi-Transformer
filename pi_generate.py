import re
import numpy as np
from collections import defaultdict, Counter, OrderedDict

# ============================================================
# MEMORY STORE
# ============================================================

class MnemoticStore:
    def __init__(self):
        self.values = []

    def add_to_memory(self, text):
        text = text.strip()
        if text:
            self.values.append(text)

    def retrieve(self, prompt):
        if not self.values:
            return ""

        prompt_words = set(prompt.lower().split())

        best_match = None
        best_score = -1

        for memory in self.values:
            memory_words = set(memory.lower().split())

            overlap = len(
                prompt_words.intersection(memory_words)
            )

            if overlap > best_score:
                best_score = overlap
                best_match = memory

        if best_match:
            return best_match

        return np.random.choice(self.values)


# ============================================================
# TRIGRAM ENGINE
# ============================================================

class TrigramEngine:

    def __init__(self, capacity=100000):

        self.capacity = capacity

        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)

        self.vocab = set()

        self.lru = OrderedDict()

        self.cluster_signatures = {}
        self.cluster_words = defaultdict(set)

    # --------------------------------------------------------

    def train(self, texts):

        contexts = defaultdict(Counter)

        for text in texts:

            tokens = text.lower().split()

            if len(tokens) < 1:
                continue

            self.vocab.update(tokens)

            # -------------------------
            # BIGRAMS
            # -------------------------

            for i in range(len(tokens) - 1):

                w1 = tokens[i]
                w2 = tokens[i + 1]

                self.bigrams[w1][w2] += 1

                self.lru[w1] = True
                self.lru.move_to_end(w1)

            # -------------------------
            # TRIGRAMS
            # -------------------------

            for i in range(len(tokens) - 2):

                w1 = tokens[i]
                w2 = tokens[i + 1]
                w3 = tokens[i + 2]

                self.trigrams[(w1, w2)][w3] += 1

            # -------------------------
            # CONTEXT COLLECTION
            # -------------------------

            for i in range(1, len(tokens) - 1):

                word = tokens[i]

                left = tokens[i - 1]
                right = tokens[i + 1]

                contexts[word][left] += 1
                contexts[word][right] += 1

        self._build_clusters(contexts)

        # -------------------------
        # LRU PRUNING
        # -------------------------

        while len(self.bigrams) > self.capacity:

            oldest, _ = self.lru.popitem(last=False)

            if oldest in self.bigrams:
                del self.bigrams[oldest]

    # --------------------------------------------------------

    def _build_clusters(self, contexts):

        next_cluster_id = 0

        for word, ctx in contexts.items():

            signature = tuple(
                sorted(
                    [w for w, _ in ctx.most_common(3)]
                )
            )

            if signature not in self.cluster_signatures:

                self.cluster_signatures[
                    signature
                ] = next_cluster_id

                next_cluster_id += 1

            cid = self.cluster_signatures[
                signature
            ]

            self.cluster_words[cid].add(word)

    # --------------------------------------------------------

    def random_word(self):

        if not self.vocab:
            return ""

        return np.random.choice(
            list(self.vocab)
        )

    # --------------------------------------------------------

    def _sample(self, counter, temperature=0.8):

        words = list(counter.keys())

        counts = np.array(
            list(counter.values()),
            dtype=np.float32
        )

        # sqrt damping
        counts = np.sqrt(counts)

        temperature = max(
            temperature,
            1e-6
        )

        logits = np.log(counts + 1e-9)

        logits /= temperature

        logits -= np.max(logits)

        probs = np.exp(logits)

        probs /= probs.sum()

        return np.random.choice(
            words,
            p=probs
        )

    # --------------------------------------------------------

    def cluster_fallback(self, word):

        for cid, words in self.cluster_words.items():

            if word in words:

                candidates = list(words)

                if candidates:
                    return np.random.choice(
                        candidates
                    )

        return self.random_word()

    # --------------------------------------------------------

    def next_word(
        self,
        previous_word,
        current_word,
        temperature=0.8
    ):

        trigram_key = (
            previous_word,
            current_word
        )

        if trigram_key in self.trigrams:

            return self._sample(
                self.trigrams[trigram_key],
                temperature
            )

        if current_word in self.bigrams:

            return self._sample(
                self.bigrams[current_word],
                temperature
            )

        return self.cluster_fallback(
            current_word
        )


# ============================================================
# TEXT LOADER
# ============================================================

def load_text_file(filename):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    sentences = re.split(
        r"[.!?\n]+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_response(
    engine,
    memory_store,
    prompt,
    length=30,
    temperature=0.8
):

    memory = memory_store.retrieve(prompt)

    seed = memory.lower().split()

    if len(seed) >= 2:

        w1 = seed[-2]
        w2 = seed[-1]

    elif len(seed) == 1:

        w1 = seed[0]
        w2 = seed[0]

    else:

        w1 = engine.random_word()
        w2 = engine.random_word()

    output = [w1, w2]

    for _ in range(length):

        nxt = engine.next_word(
            w1,
            w2,
            temperature
        )

        if not nxt:
            break

        output.append(nxt)

        w1 = w2
        w2 = nxt

    return " ".join(output)


# ============================================================
# MAIN
# ============================================================

def main():

    filename = input(
        "Training file: "
    ).strip()

    text_data = load_text_file(
        filename
    )

    memory_store = MnemoticStore()

    for text in text_data:
        memory_store.add_to_memory(
            text
        )

    engine = TrigramEngine(
        capacity=100000
    )

    engine.train(text_data)

    print()
    print(
        f"Loaded {len(text_data)} memories."
    )
    print("Model ready.")
    print(
        "Press ENTER on an empty line to quit."
    )
    print()

    while True:

        user_input = input(
            "USER: "
        ).strip()

        if not user_input:
            break

        response = generate_response(
            engine,
            memory_store,
            user_input,
            length=250,
            temperature=0.7
        )

        print()
        print("AI:", response)
        print()


if __name__ == "__main__":
    main()
