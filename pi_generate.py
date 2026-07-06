import numpy as np
import random
import re

class SurjectiveStructure:
    def __init__(self):
        self.trigram_counts = {}        # Store counts of trigram pairs
        self.sorted_trigrams = []       # Sorted list based on log counts
        self.special_pairs = set()      # Pairs where meaning survives

    def add_trigram(self, pair, count=1, special=False):
        # Add pairs with counts, mark special if needed
        self.trigram_counts[pair] = self.trigram_counts.get(pair, 0) + count
        if special:
            self.special_pairs.add(pair)

    def sort_trigrams_by_log(self):
        # Apply natural log to counts and sort
        log_counts = {pair: np.log(count + 1e-12) for pair, count in self.trigram_counts.items()}
        self.sorted_trigrams = sorted(log_counts.items(), key=lambda x: x[1], reverse=True)

    def check_surjectivity(self, factor='A'):
        # Check if the projection covers entire factors
        if factor == 'A':
            return set(pair[0] for pair in self.trigram_counts)
        elif factor == 'B':
            return set(pair[1] for pair in self.trigram_counts)

    def is_meaning_preserved(self, pair):
        return pair in self.special_pairs

    def generate_text(self, seed_text=None, length=50, tuning=1.0):
        """
        Generate text of specified length starting from seed_text.
        If seed_text is None, start from a random pair.
        """
        if not self.sorted_trigrams:
            return ""

        if seed_text:
            seed_tokens = self.tokenize(seed_text)
            if len(seed_tokens) >= 2:
                current_pair = (seed_tokens[-2], seed_tokens[-1])
            elif len(seed_tokens) == 1:
                # If only one token, duplicate as pair
                current_pair = (seed_tokens[0], seed_tokens[0])
            else:
                # Empty seed, pick random pair
                current_pair = random.choice(list(self.trigram_counts.keys()))
        else:
            current_pair = random.choice(list(self.trigram_counts.keys()))

        output_tokens = [current_pair[0], current_pair[1]]

        for _ in range(length - 2):
            # Find candidate pairs where first element matches the last token
            candidates = [(pair, count) for pair, count in self.sorted_trigrams if pair[0] == current_pair[1]]
            if not candidates:
                break
            weights = np.array([np.exp(count[1] * tuning) for count in candidates])
            weights /= weights.sum()
            next_pair = random.choices([pair for pair, _ in candidates], weights=weights, k=1)[0]
            output_tokens.append(next_pair[1])
            current_pair = next_pair

        return ' '.join(output_tokens)

    def load_dataset_from_txt(self, filename, min_freq=3):
        # Load and process text file into trigram pairs
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = self.tokenize(text)
        # Generate trigram pairs
        for i in range(len(tokens) - 2):
            pair = (tokens[i], tokens[i+1])
            self.add_trigram(pair, count=1)
        # Mark pairs with frequency >= min_freq as special
        counts = {}
        for i in range(len(tokens) - 2):
            pair = (tokens[i], tokens[i+1])
            counts[pair] = counts.get(pair, 0) + 1
        for pair, count in counts.items():
            if count >= min_freq:
                self.add_trigram(pair, count=count, special=True)
        # After loading, sort pairs
        self.sort_trigrams_by_log()

    def tokenize(self, text):
        # Simple tokenizer splitting on non-word characters
        return text.lower().split()

# Usage example
structure = SurjectiveStructure()

# Load dataset from a text file
structure.load_dataset_from_txt(input("Filename: "), min_freq=3)

# Generate text starting from a seed
while True:
    seed = input("USER: ")
    generated_text = structure.generate_text(seed_text=seed, length=1000, tuning=2.0)
    print("Generated Text starting from seed:\n", generated_text)
