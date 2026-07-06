import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import os

# 1. Setup and Preprocessing
file_path = 'singlekb.txt'


with open(file_path, 'r', encoding="utf-8") as f:
    text = f.read().lower().split()

# 2. Matrix and Log-Sorting
word_counts = Counter(text)
vocab = sorted(word_counts.keys())
matrix_data = np.array([word_counts[word] for word in vocab], dtype=float).reshape(1, -1)

log_projected = np.log1p(matrix_data)
log_indices = np.argsort(log_projected, axis=1)

sorted_vocab = [vocab[i] for i in log_indices[0]]
sorted_log_values = np.take_along_axis(log_projected, log_indices, axis=1)[0]

# 3. Markovian Linking Logic
transitions = defaultdict(Counter)
for i in range(len(text) - 1):
    transitions[text[i]][text[i+1]] += 1

markov_chain = {w: {next_w: count/sum(follows.values()) 
                for next_w, count in follows.items()} 
                for w, follows in transitions.items()}

# 4. Generative Engine
def generate_linked_trigram(last_word=None):
    # Start with the highest weighted word if no history exists
    if last_word is None or last_word not in markov_chain:
        current = sorted_vocab[-1] 
    else:
        current = last_word
    
    trigram = [current]
    
    # Generate 2 linked words based on transition probabilities
    for _ in range(2):
        if current in markov_chain:
            choices = list(markov_chain[current].keys())
            probs = list(markov_chain[current].values())
            current = np.random.choice(choices, p=probs)
        else:
            current = np.random.choice(sorted_vocab)
        trigram.append(current)
        
    return " ".join(trigram), current

# 5. Output
print("--- Linked Markovian Trigram Sequence ---")
while True:
    last_word = input("USER: ")[:-1]
    for _ in range(100):
        trigram, last_word = generate_linked_trigram(last_word)
        print(' '.join(trigram.split()[:2]), end=" ")
    print()
