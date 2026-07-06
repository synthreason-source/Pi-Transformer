import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.linalg import expm

# 1. Setup and Preprocessing
file_path = 'singlekb.txt'
try:
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read().lower().split()
except:
    text = ["the", "neural", "matrix", "is", "a", "projection", "of", "data"]

# 2. Matrix and Log-Sorting with Exponential Weighting
word_counts = Counter(text)
vocab = sorted(word_counts.keys())
v_map = {w: i for i, w in enumerate(vocab)}
n = len(vocab)

# Frequency Vector
freq_vec = np.array([word_counts[w] for w in vocab], dtype=float)
log_projected = np.log1p(freq_vec)

# Matrix Exponential Sort: Transforming the diagonal representation
# We create a simple influence matrix where A_ij = log(count_i) * log(count_j)
# Then apply matrix exponential to get the 'propagator' of word importance
influence_matrix = expm(np.outer(log_projected, log_projected) / n)
influence_scores = influence_matrix.sum(axis=1)

sorted_indices = np.argsort(influence_scores)
sorted_vocab = [vocab[i] for i in sorted_indices]

# 3. Markovian Linking Logic
transitions = defaultdict(Counter)
for i in range(len(text) - 1):
    transitions[text[i]][text[i+1]] += 1

markov_chain = {w: {next_w: count/sum(follows.values()) 
                for next_w, count in follows.items()} 
                for w, follows in transitions.items()}

# 4. Generative Engine
def generate_linked_trigram(last_word=None):
    if last_word is None or last_word not in markov_chain:
        # Use the highest influence word from our exponential sort as anchor
        current = sorted_vocab[-1] 
    else:
        current = last_word
    
    trigram = [current]
    for _ in range(2):
        if current in markov_chain:
            choices = list(markov_chain[current].keys())
            probs = list(markov_chain[current].values())
            current = np.random.choice(choices, p=probs)
        else:
            current = np.random.choice(sorted_vocab)
        trigram.append(current)
        
    return trigram, trigram[-1]

# 5. Output
print("--- Exponentially Weighted Markovian Engine ---")
while True:
    user_input = input("USER: ")
    if not user_input: break
    last_word = user_input.split()[-1]
    
    for _ in range(20):
        trigram, last_word = generate_linked_trigram(last_word)
        print(' '.join(trigram[:2]), end=" ")
    print()