import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import difflib
from collections import defaultdict

# ==========================================================
# 1. Dataset Loading & Frequency Amplification
# ==========================================================

def load_and_analyze_dataset(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    tokens = raw_text.lower().split()
    trigram_sequence = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]

    unique_trigrams = sorted(list(set(trigram_sequence)))
    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    if unk_trigram not in unique_trigrams:
        unique_trigrams.append(unk_trigram)
    vocab_size = len(unique_trigrams)

    word_to_idx = {t: i for i, t in enumerate(unique_trigrams)}
    idx_to_word = {i: t for i, t in enumerate(unique_trigrams)}

    context_to_row = {}
    row_count = 0
    trigram_counts = defaultdict(lambda: defaultdict(int))

    for t in trigram_sequence:
        w1, w2, w3 = t
        context = (w1, w2)
        if context not in context_to_row:
            context_to_row[context] = row_count
            row_count += 1
        target_idx = word_to_idx.get(t, word_to_idx[unk_trigram])
        trigram_counts[context][target_idx] += 1

    freq_matrix = torch.zeros((row_count, vocab_size))
    for context, targets in trigram_counts.items():
        r_idx = context_to_row[context]
        for target_idx, count in targets.items():
            freq_matrix[r_idx, target_idx] = count

    col_sums = freq_matrix.sum(dim=0)
    max_freq = col_sums.max() + 1e-5
    normalized_inv_freq = 1.0 - (col_sums / max_freq)
    exp_boost = torch.exp(normalized_inv_freq * 2.0)

    row_indices = torch.linspace(0, 2 * math.pi, steps=row_count)
    sine_smoother = 0.5 * (1.0 + torch.sin(row_indices)).unsqueeze(1)

    smoothed_matrix = freq_matrix * sine_smoother
    row_smoothed_weight = smoothed_matrix.sum(dim=0) / (smoothed_matrix.sum() + 1e-5)

    amplification_boost = exp_boost * (1.0 + row_smoothed_weight)
    amplification_boost[word_to_idx[unk_trigram]] = 0.0

    top_indices = torch.argsort(col_sums, descending=True)
    frequent_trigrams = [idx_to_word[i.item()] for i in top_indices if idx_to_word[i.item()] != unk_trigram]

    return word_to_idx, idx_to_word, vocab_size, amplification_boost, frequent_trigrams, unique_trigrams


# ==========================================================
# 2. Stage 1: Markov Neural Generator (Markov In)
# ==========================================================

class MarkovSeedingLayer(nn.Module):
    """
    Stage 1: Generates initial population seeds for the GA via neural Markov forward passes.
    """
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.context_size = 3
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(self.context_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim * self.context_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        w_emb = self.word_embedding(idx)
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
        p_emb = self.pos_embedding(positions)
        x = w_emb + p_emb
        x = x.view(B, -1)
        return self.lm_head(x)

    @torch.no_grad()
    def generate_population_seeds(self, seed_context_idx, sequence_length, pop_size, 
                                 amp_boost, idx_to_word, unk_id, temperature=1.2):
        seeds = []
        amp_boost = amp_boost.to(seed_context_idx.device)

        for _ in range(pop_size):
            curr_idx = seed_context_idx.clone()
            candidate_sequence = []

            for _ in range(sequence_length):
                cond = curr_idx[:, -self.context_size:]
                logits = self(cond) + amp_boost
                logits[:, unk_id] = -float('inf')

                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

                curr_idx = torch.cat([curr_idx, next_id], dim=1)
                candidate_sequence.append(idx_to_word[next_id.item()])

            seeds.append(candidate_sequence)

        return seeds


# ==========================================================
# 3. Stage 2: Genetic Algorithm Engine (Genetic In/Out)
# ==========================================================

class GeneticOptimizerEngine:
    """
    Stage 2: Consumes neural seeds, performs selection, crossover, and mutation.
    """
    def __init__(self, population_size=1000, mutation_rate=0.1, generations=500):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def fitness(self, candidate, target_trigrams):
        score = 0.0
        for i in range(min(len(candidate), len(target_trigrams))):
            if candidate[i] == target_trigrams[i]:
                score += 1.0
        return score

    def crossover(self, parent1, parent2):
        if len(parent1) <= 1:
            return parent1
        split_point = random.randint(0, len(parent1) - 1)
        return parent1[:split_point] + parent2[split_point:]

    def mutate(self, candidate, corpus_trigrams):
        mutated = []
        for tg in candidate:
            if random.random() < self.mutation_rate:
                mutated.append(random.choice(corpus_trigrams))
            else:
                mutated.append(tg)
        return mutated

    def evolve(self, initial_population, target_trigrams, corpus_trigrams):
        population = initial_population

        for _ in range(self.generations):
            fitness_scores = [self.fitness(cand, target_trigrams) for cand in population]

            top_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True
            )[:10]

            parents = [population[i] for i in top_indices]

            offspring = []
            while len(offspring) < self.pop_size - len(parents):
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.crossover(p1, p2)
                offspring.append(self.mutate(child, corpus_trigrams))

            population = parents + offspring

        best_candidate = max(population, key=lambda c: self.fitness(c, target_trigrams))
        return best_candidate


# ==========================================================
# 4. Pipeline Execution Loop
# ==========================================================

if __name__ == "__main__":
    filename = input("Filename: ").strip()
    (word_to_idx, idx_to_word, vocab_size, amp_boost, 
     frequent_trigrams, corpus_trigrams) = load_and_analyze_dataset(filename)

    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    unk_id = word_to_idx[unk_trigram]

    # Instantiate Pipeline Layers
    markov_seed_layer = MarkovSeedingLayer(vocab_size=vocab_size)
    genetic_engine = GeneticOptimizerEngine(population_size=100, mutation_rate=0.1, generations=50)

    known_words = sorted(list({w for tg in word_to_idx.keys() if tg != unk_trigram for w in tg}))

    while True:
        raw_input = input("\nUSER: ").strip()
        if raw_input.lower() in {"quit", "exit", "stop"}:
            break

        tokens = raw_input.lower().split() if raw_input else []
        corrected_tokens = []
        for w in tokens:
            matches = difflib.get_close_matches(w, known_words, n=100, cutoff=0.0)
            if matches:
                corrected_tokens.append(matches[0])

        raw_target_trigrams = [tuple(corrected_tokens[i:i+3]) for i in range(len(corrected_tokens) - 2)]
        target_trigrams = [tg if tg in word_to_idx else unk_trigram for tg in raw_target_trigrams]

        if not target_trigrams or all(tg == unk_trigram for tg in target_trigrams):
            print("Fallback: Using frequent corpus trigrams as target...")
            target_trigrams = frequent_trigrams[:2]

        seed_context = target_trigrams[:markov_seed_layer.context_size]
        while len(seed_context) < markov_seed_layer.context_size:
            seed_context.append(frequent_trigrams[0])

        # Safe key lookup with unk_id fallback
        context_tensor = torch.tensor(
            [[word_to_idx.get(tg, unk_id) for tg in seed_context]], 
            dtype=torch.long
        )

        # STEP 1: MARKOV GENERATE (Seeds Generation 0)
        print("[1/2] MARKOV GENERATE: Generating neural seed candidates...")
        neural_seeds = markov_seed_layer.generate_population_seeds(
            seed_context_idx=context_tensor,
            sequence_length=99,
            pop_size=genetic_engine.pop_size,
            amp_boost=amp_boost,
            idx_to_word=idx_to_word,
            unk_id=unk_id,
            temperature=0.8
        )

        # STEP 2: GENETIC IN / GENETIC OUT (Evolutionary Filtering)
        print("[2/2] GENETIC IN/OUT: Evolving neural seeds across generations...")
        evolved_ga_trigrams = genetic_engine.evolve(
            initial_population=neural_seeds,
            target_trigrams=target_trigrams,
            corpus_trigrams=corpus_trigrams
        )

        print("\n--- FINAL PIPELINE OUTPUT ---")
        flattened_words = [word for trigram in evolved_ga_trigrams for word in trigram]
        print(' '.join(flattened_words))
        print("-" * 35)
