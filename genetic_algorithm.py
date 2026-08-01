import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import difflib
from collections import defaultdict, Counter

# ==========================================================
# 1. Dataset Loading & Context-Aware Frequency Analysis
# ==========================================================

def load_and_analyze_dataset(filename, context_window=5):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    tokens = raw_text.lower().split()
    if len(tokens) < 3:
        raise ValueError("Dataset must contain at least 3 tokens.")

    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    unique_trigrams = sorted(set(trigrams))
    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    if unk_trigram not in unique_trigrams:
        unique_trigrams.append(unk_trigram)

    vocab_size = len(unique_trigrams)
    word_to_idx = {t: i for i, t in enumerate(unique_trigrams)}
    idx_to_word = {i: t for i, t in enumerate(unique_trigrams)}

    context_to_row = {}
    row_count = 0
    trigram_counts = defaultdict(Counter)

    for i, t in enumerate(trigrams):
        context = tuple(tokens[max(0, i - context_window):i])
        if len(context) == 0:
            continue
        if context not in context_to_row:
            context_to_row[context] = row_count
            row_count += 1
        trigram_counts[context][word_to_idx.get(t, word_to_idx[unk_trigram])] += 1

    freq_matrix = torch.zeros((max(row_count, 1), vocab_size), dtype=torch.float32)
    for context, targets in trigram_counts.items():
        r_idx = context_to_row[context]
        for target_idx, count in targets.items():
            freq_matrix[r_idx, target_idx] = float(count)

    col_sums = freq_matrix.sum(dim=0)
    max_freq = col_sums.max().clamp_min(1e-5)
    normalized_inv_freq = 1.0 - (col_sums / max_freq)
    exp_boost = torch.exp(normalized_inv_freq * 2.0)

    row_sums = freq_matrix.sum(dim=1, keepdim=True).clamp_min(1e-5)
    row_distribution = freq_matrix / row_sums
    row_weights = row_distribution.mean(dim=0)

    row_indices = torch.linspace(0, 2 * math.pi, steps=freq_matrix.shape[0])
    sine_smoother = 0.5 * (1.0 + torch.sin(row_indices)).unsqueeze(1)

    smoothed_matrix = freq_matrix * sine_smoother
    smoothed_col_sums = smoothed_matrix.sum(dim=0)
    smoothed_total = smoothed_matrix.sum().clamp_min(1e-5)
    row_smoothed_weight = smoothed_col_sums / smoothed_total

    amplification_boost = exp_boost * (1.0 + row_smoothed_weight + row_weights)
    amplification_boost[word_to_idx[unk_trigram]] = 0.0

    top_indices = torch.argsort(col_sums, descending=True)
    frequent_trigrams = [idx_to_word[i.item()] for i in top_indices if idx_to_word[i.item()] != unk_trigram]

    frequent_words = [w for tg in frequent_trigrams for w in tg]
    frequent_words = list(dict.fromkeys(frequent_words))

    return (
        word_to_idx,
        idx_to_word,
        vocab_size,
        amplification_boost,
        frequent_trigrams,
        unique_trigrams,
        frequent_words,
        context_to_row,
    )


# ==========================================================
# 2. Stage 1: Contextual Markov Neural Generator
# ==========================================================

class MarkovSeedingLayer(nn.Module):
    """
    Stage 1: Generates initial population seeds for the GA via contextual neural passes.
    Uses both token embeddings and a learnable context summary.
    """
    def __init__(self, vocab_size, embed_dim=64, context_size=5):
        super().__init__()
        self.context_size = context_size
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_size, embed_dim)
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.lm_head = nn.Sequential(
            nn.Linear(embed_dim * context_size + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(self, idx):
        B, T = idx.shape
        T_use = min(T, self.context_size)
        idx = idx[:, -T_use:]

        w_emb = self.word_embedding(idx)
        positions = torch.arange(0, T_use, dtype=torch.long, device=idx.device)
        p_emb = self.pos_embedding(positions)

        x = w_emb + p_emb
        x_flat = x.reshape(B, -1)

        context_summary = x.mean(dim=1)
        context_summary = self.context_proj(context_summary)

        logits = self.lm_head(torch.cat([x_flat, context_summary], dim=-1))
        return logits

    @torch.no_grad()
    def generate_population_seeds(
        self,
        seed_context_idx,
        sequence_length,
        pop_size,
        amp_boost,
        idx_to_word,
        unk_id,
        temperature=1.2,
        top_k=25,
    ):
        seeds = []
        amp_boost = amp_boost.to(seed_context_idx.device)

        for _ in range(pop_size):
            curr_idx = seed_context_idx.clone()
            candidate_sequence = []

            for _ in range(sequence_length):
                cond = curr_idx[:, -self.context_size:]
                if cond.shape[1] < self.context_size:
                    pad = cond[:, :1].repeat(1, self.context_size - cond.shape[1])
                    cond = torch.cat([pad, cond], dim=1)

                logits = self(cond) + amp_boost
                logits[:, unk_id] = -float('inf')

                if top_k is not None and top_k < logits.shape[-1]:
                    topv, topi = torch.topk(logits, k=top_k, dim=-1)
                    filtered = torch.full_like(logits, -float("inf"))
                    filtered.scatter_(1, topi, topv)
                    logits = filtered

                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

                curr_idx = torch.cat([curr_idx, next_id], dim=1)
                candidate_sequence.append(idx_to_word[next_id.item()])

            seeds.append(candidate_sequence)

        return seeds


# ==========================================================
# 3. Stage 2: Genetic Algorithm Engine
# ==========================================================

class GeneticOptimizerEngine:
    """
    Stage 2: Consumes neural seeds, performs selection, crossover, and mutation.

    Adds a lightweight backward-propagation loop: a learnable per-word score
    vector (self.word_scores) is nudged with a real torch optimizer each
    generation. `fitness()` reads the current scores as a bonus term, and
    `mutate()` uses them (softmax-weighted) to bias which replacement words
    get sampled — so the GA's mutation/fitness behavior is steered by a
    gradient-trained signal rather than being purely random/heuristic.
    """
    def __init__(self, population_size=1000, mutation_rate=0.1, generations=500,
                 elite_size=10, word_vocab=None, learning_rate=0.05):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elite_size = elite_size

        # --- backward-propagation setup -----------------------------------
        # word_vocab: list of unique surface-form words the GA is allowed to
        # mutate towards. If provided, we keep a learnable score per word and
        # an optimizer to update it via backprop each generation.
        self.word_vocab = word_vocab
        if word_vocab:
            self.word_to_idx_word = {w: i for i, w in enumerate(word_vocab)}
            self.word_scores = torch.zeros(len(word_vocab), requires_grad=True)
            self.score_optimizer = torch.optim.Adam([self.word_scores], lr=learning_rate)
        else:
            self.word_to_idx_word = {}
            self.word_scores = None
            self.score_optimizer = None

    def fitness(self, candidate, target_trigrams, corpus_word_freq=None):
        score = 0.0
        n = min(len(candidate), len(target_trigrams))
        for i in range(n):
            if candidate[i] == target_trigrams[i]:
                score += 2.0
            overlap = len(set(candidate[i]) & set(target_trigrams[i]))
            score += 0.25 * overlap

        if corpus_word_freq is not None:
            score += 0.01 * sum(corpus_word_freq.get(w, 0) for tg in candidate for w in tg)

        # Bonus term driven by the backprop-trained word scores. Detached
        # (no_grad) since fitness() itself is used purely for ranking, not
        # for building the backward graph — that happens in backpropagate().
        if self.word_scores is not None:
            with torch.no_grad():
                bonus = 0.0
                for tg in candidate:
                    for w in tg:
                        idx = self.word_to_idx_word.get(w)
                        if idx is not None:
                            bonus += self.word_scores[idx].item()
                score += 0.05 * bonus

        return score

    def backpropagate(self, population, target_trigrams, corpus_word_freq=None):
        """
        Real backward-propagation step. Builds a differentiable proxy for
        population fitness out of self.word_scores (summed over every word
        that appears in every candidate), then backpropagates the negative
        of that signal so gradient descent pushes scores for
        frequently-appearing / target-aligned words upward and scores for
        words that drag fitness down, downward. Call this once per
        generation before ranking, so both fitness() and mutate() act on the
        freshly-updated scores.
        """
        if self.word_scores is None or self.score_optimizer is None:
            return

        target_word_set = {w for tg in target_trigrams for w in tg}

        self.score_optimizer.zero_grad()
        total = torch.zeros(1)
        count = 0

        for candidate in population:
            cand_score = torch.zeros(1)
            for tg in candidate:
                for w in tg:
                    idx = self.word_to_idx_word.get(w)
                    if idx is None:
                        continue
                    weight = 1.5 if w in target_word_set else 1.0
                    cand_score = cand_score + weight * self.word_scores[idx]
            total = total + cand_score
            count += 1

        if count == 0:
            return

        mean_score = total / count
        loss = -mean_score  # ascend fitness == descend negative fitness
        loss.backward()
        self.score_optimizer.step()

    def crossover(self, parent1, parent2):
        if len(parent1) <= 1:
            return parent1
        split_point = random.randint(1, len(parent1) - 1)
        return parent1[:split_point] + parent2[split_point:]

    def _score_weighted_words(self, corpus_words):
        """Softmax-normalized sampling weights over corpus_words, derived
        from the current (backprop-trained) word_scores. Falls back to None
        (uniform sampling) if scores aren't enabled or none of the words are
        in the learnable vocab."""
        if self.word_scores is None or not corpus_words:
            return None

        with torch.no_grad():
            idxs = [self.word_to_idx_word.get(w) for w in corpus_words]
            known = [i for i in idxs if i is not None]
            if not known:
                return None
            gathered = torch.stack([self.word_scores[i] for i in idxs if i is not None])
            probs = F.softmax(gathered, dim=0).tolist()

        # Map back to full-length weight list aligned with corpus_words,
        # giving unknown words a small floor weight so they can still appear.
        weights = []
        p_iter = iter(probs)
        for i in idxs:
            weights.append(next(p_iter) if i is not None else 1e-6)
        return weights

    def mutate(self, candidate, corpus_trigrams, corpus_words=None):
        mutated = []
        weights = self._score_weighted_words(corpus_words) if corpus_words else None

        for tg in candidate:
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:
                    mutated.append(random.choice(corpus_trigrams))
                else:
                    if corpus_words is None:
                        mutated.append(random.choice(corpus_trigrams))
                    else:
                        if weights is not None:
                            words = random.choices(corpus_words, weights=weights, k=3)
                        else:
                            words = [random.choice(corpus_words) for _ in range(3)]
                        mutated.append(tuple(words))
            else:
                mutated.append(tg)
        return mutated

    def evolve(self, initial_population, target_trigrams, corpus_trigrams, corpus_words=None, corpus_word_freq=None):
        population = initial_population
        elite_size = min(self.elite_size, len(population))

        for _ in range(self.generations):
            # backward-propagation pass: update learnable word scores before
            # this generation's fitness ranking / mutation happen
            self.backpropagate(population, target_trigrams, corpus_word_freq)

            fitness_scores = [self.fitness(cand, target_trigrams, corpus_word_freq) for cand in population]

            top_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True
            )[:elite_size]

            parents = [population[i] for i in top_indices]
            offspring = []

            while len(offspring) < self.pop_size - len(parents):
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.crossover(p1, p2)
                offspring.append(self.mutate(child, corpus_trigrams, corpus_words))

            population = parents + offspring

        best_candidate = max(population, key=lambda c: self.fitness(c, target_trigrams, corpus_word_freq))
        return best_candidate


# ==========================================================
# 4. Pipeline Execution Loop
# ==========================================================

if __name__ == "__main__":
    filename = input("Filename: ").strip()
    (
        word_to_idx,
        idx_to_word,
        vocab_size,
        amp_boost,
        frequent_trigrams,
        corpus_trigrams,
        frequent_words,
        context_to_row,
    ) = load_and_analyze_dataset(filename, context_window=5)

    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    unk_id = word_to_idx[unk_trigram]
    corpus_word_freq = Counter(w for tg in corpus_trigrams for w in tg)

    markov_seed_layer = MarkovSeedingLayer(vocab_size=vocab_size, embed_dim=64, context_size=5)

    known_words = sorted(set(frequent_words) | {w for tg in word_to_idx.keys() if tg != unk_trigram for w in tg})

    genetic_engine = GeneticOptimizerEngine(
        population_size=100,
        mutation_rate=0.12,
        generations=50,
        elite_size=12,
        word_vocab=known_words,
        learning_rate=0.05,
    )

    print("\nEnter text to guide generation. Type 'quit' to exit.")
    while True:
        raw_input_text = input("\nUSER: ").strip()
        if raw_input_text.lower() in {"quit", "exit", "stop"}:
            break

        tokens = raw_input_text.lower().split() if raw_input_text else []
        corrected_tokens = []
        for w in tokens:
            matches = difflib.get_close_matches(w, known_words, n=1, cutoff=0.0)
            corrected_tokens.append(matches[0] if matches else w)

        raw_target_trigrams = [tuple(corrected_tokens[i:i+3]) for i in range(len(corrected_tokens) - 2)]
        target_trigrams = [tg if tg in word_to_idx else unk_trigram for tg in raw_target_trigrams]

        if not target_trigrams or all(tg == unk_trigram for tg in target_trigrams):
            print("Fallback: Using frequent corpus trigrams as target...")
            target_trigrams = frequent_trigrams[:max(2, min(5, len(frequent_trigrams)))]

        seed_context_words = corrected_tokens[-markov_seed_layer.context_size:]
        if len(seed_context_words) < markov_seed_layer.context_size:
            fallback_words = frequent_words[: markov_seed_layer.context_size - len(seed_context_words)]
            seed_context_words = fallback_words + seed_context_words

        while len(seed_context_words) < markov_seed_layer.context_size:
            seed_context_words.insert(0, frequent_words[0] if frequent_words else "<unk>")

        seed_context = [(seed_context_words[i], seed_context_words[i + 1], seed_context_words[i + 2])
                        for i in range(max(1, len(seed_context_words) - 2))]
        seed_context = seed_context[-1] if seed_context else frequent_trigrams[0]

        context_tensor = torch.tensor(
            [[word_to_idx.get(seed_context[0], unk_id),
              word_to_idx.get(seed_context[1], unk_id),
              word_to_idx.get(seed_context[2], unk_id),
              word_to_idx.get(seed_context[0], unk_id),
              word_to_idx.get(seed_context[1], unk_id)]],
            dtype=torch.long
        )

        print("[1/2] MARKOV GENERATE: Generating contextual neural seed candidates...")
        neural_seeds = markov_seed_layer.generate_population_seeds(
            seed_context_idx=context_tensor,
            sequence_length=99,
            pop_size=genetic_engine.pop_size,
            amp_boost=amp_boost,
            idx_to_word=idx_to_word,
            unk_id=unk_id,
            temperature=0.8,
            top_k=25,
        )

        print("[2/2] GENETIC IN/OUT: Evolving neural seeds across generations (with backprop-guided scoring)...")
        evolved_ga_trigrams = genetic_engine.evolve(
            initial_population=neural_seeds,
            target_trigrams=target_trigrams,
            corpus_trigrams=corpus_trigrams,
            corpus_words=frequent_words,
            corpus_word_freq=corpus_word_freq,
        )

        print("\n--- FINAL PIPELINE OUTPUT ---")
        flattened_words = [word for trigram in evolved_ga_trigrams for word in trigram]
        print(' '.join(flattened_words))
        print("-" * 35)
