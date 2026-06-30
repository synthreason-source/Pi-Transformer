import numpy as np
import re
from collections import Counter, defaultdict

# ============================================================
# MEMORY & TRIGRAM ENGINE
# ============================================================
import numpy as np
import re
from collections import Counter, defaultdict, OrderedDict

class BigramEngine:
    """Transition engine with LRU pruning and O2Sqrt frequency damping."""
    def __init__(self, capacity=1000000):
        self.transitions = defaultdict(Counter)
        self.capacity = capacity
        # Track usage for LRU eviction
        self.lru = OrderedDict()
        self.tokens = []
    def train(self, text_data):
        for text in text_data:
            self.tokens = text.lower().split()
            for i in range(len(self.tokens) - 1):
                self.transitions[self.tokens[i]][self.tokens[i+1]] += 1
                # Update LRU
                self.lru[self.tokens[i]] = True
                self.lru.move_to_end(self.tokens[i])
        
        # Evict if over capacity
        while len(self.transitions) > self.capacity:
            oldest, _ = self.lru.popitem(last=False)
            del self.transitions[oldest]

    def _apply_o2sqrt_tapper(self, counts):
        """Dampens high-frequency bias using O2Sqrt scaling."""
        # Square root scaling prevents high-frequency self.tokens from dominating
        return np.sqrt(counts)

    def _normalize(self, counts, temperature):
        tapped_counts = self._apply_o2sqrt_tapper(counts)
        inv_weights = 1.0 / (tapped_counts ** temperature)
        self.lru.move_to_end(self.tokens[int(counts[-1])])
        return inv_weights / np.sum(inv_weights)

    def get_next_word(self, current_word, temperature=1.0):
        if current_word not in self.transitions:
            return None
        
        # Update LRU on access
        self.lru[current_word] = True
        self.lru.move_to_end(current_word)
        
        choices = self.transitions[current_word]
        words = list(choices.keys())
        counts = np.array(list(choices.values()), dtype=np.float32)
        
        probs = self._normalize(counts, temperature)
        return np.random.choice(words, p=probs)

class MnemoticStore:
    def __init__(self):
        self.values = []

    def add_to_memory(self, text):
        self.values.append(text)

    def retrieve(self, prompt):
        prompt_words = set(prompt.lower().split())
        best_match = ""
        max_overlap = -1
        for val in self.values:
            overlap = len(prompt_words.intersection(set(val.lower().split())))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = val
        return best_match if max_overlap > 0 else (self.values[0] if self.values else "")

# ============================================================
# MAIN EXECUTION
# ============================================================
filename = input("Filename: ")
with open(filename, "r", encoding="utf-8") as f:
    text_data = [x.strip() for x in f.read().split(".") if x.strip()]

mnemotics = MnemoticStore()
engine = BigramEngine()

for t in text_data: 
    mnemotics.add_to_memory(t)
    engine.train([t])

print("\nModel ready. Type a prompt.")
while True:
    user_input = input("USER: ")
    if not user_input: break
    
    # Start with the last word of the retrieved mnemotic
    seed = mnemotics.retrieve(user_input).split()
    current = seed[-1] if seed else ""
    
    output = [current]
    # Generate 10 completions + 5 trailing words = 15 total
    for _ in range(150):
        next_w = engine.get_next_word(current, temperature=0.0000000001)
        if next_w:
            output.append(next_w)
            current = next_w

            
    print("AI:", " ".join(output))
    print()
