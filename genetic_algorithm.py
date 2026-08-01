import random
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & CORPUS INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

POPULATION_SIZE = 100
NUM_GENERATIONS = 1000
MUTATION_RATE   = 0.1

# Load base dataset to extract trigrams pool
with open("questions.conf", encoding="ISO-8859-1") as f:
    raw_lines = f.readlines()

corpus_tokens = " ".join(raw_lines).lower().split()

# Extract all valid (w1, w2, w3) trigram tuples
BASE_TRIGRAMS = [
    tuple(corpus_tokens[i : i + 3]) for i in range(len(corpus_tokens) - 2)
]

if not BASE_TRIGRAMS:
    BASE_TRIGRAMS = [("<UNK>", "<UNK>", "<UNK>")]


# ═══════════════════════════════════════════════════════════════════════════════
# GENETIC ALGORITHM HELPERS FOR TRIGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def text_to_trigrams(text: str):
    tokens = text.lower().split()
    return [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]


def fitness(candidate, target_trigrams):
    """Measures sequence match accuracy on a per-trigram basis."""
    score = 0
    for i in range(min(len(candidate), len(target_trigrams))):
        if candidate[i] == target_trigrams[i]:
            score += 1
    return score


def crossover(parent1, parent2):
    """Splits parent trigram lists at a random sequence boundary."""
    if len(parent1) <= 1:
        return parent1
    split_point = random.randint(0, len(parent1) - 1)
    return parent1[:split_point] + parent2[split_point:]


def mutate(candidate):
    """Randomly swaps candidate trigram tokens with valid corpus trigrams."""
    mutated = []
    for tg in candidate:
        if random.random() < MUTATION_RATE:
            mutated.append(random.choice(BASE_TRIGRAMS))
        else:
            mutated.append(tg)
    return mutated


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GA LOOP
# ═══════════════════════════════════════════════════════════════════════════════

while True:
    raw_input_text = input("USER: ").strip()
    if raw_input_text.lower() in {"quit", "exit", "stop"}:
        break

    target_trigrams = text_to_trigrams(raw_input_text)

    # Fallback if input is too short to produce at least 1 trigram
    if not target_trigrams:
        tokens = raw_input_text.lower().split()
        pad_word = tokens[0] if tokens else "<UNK>"
        target_trigrams = [(pad_word, pad_word, pad_word)]

    target_len = len(target_trigrams)

    # Initialize population of trigram sequences
    population = [
        [random.choice(BASE_TRIGRAMS) for _ in range(target_len)]
        for _ in range(POPULATION_SIZE)
    ]

    # Evolutionary loop
    for generation in range(NUM_GENERATIONS):
        # Calculate fitness against target trigrams
        fitness_scores = [fitness(cand, target_trigrams) for cand in population]

        # Top 10 parent selection
        top_indices = sorted(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i],
            reverse=True,
        )[:10]
        parents = [population[i] for i in top_indices]

        # Early exit if perfect target match is evolved
        if fitness_scores[top_indices[0]] == target_len:
            population = parents
            break

        # Recombine & Mutate
        offspring = []
        while len(offspring) < POPULATION_SIZE - len(parents):
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = crossover(p1, p2)
            offspring.append(mutate(child))

        population = parents + offspring

    # Select best candidate sequence
    best_trigram_seq = max(population, key=lambda c: fitness(c, target_trigrams))

    # Convert best trigram sequence back to readable string format
    flattened_words = [w for tg in best_trigram_seq for w in tg]
    result_text = " ".join(flattened_words)

    print("\n--- EVOLVED TRIGRAM OUTPUT ---")
    print(result_text)
    print("-" * 31 + "\n")