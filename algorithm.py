import math
import random
import re
from collections import Counter

# --- ORIGINAL SOLVER KEEPT EXACTLY THE SAME ---
class PCMMaxCut:
    def __init__(self, graph, noise=0.02, drift_nu=0.08):
        self.graph = graph
        self.n = len(graph)
        self.noise = noise
        self.drift_nu = drift_nu
        self.spins = [random.choice([-1, 1]) for _ in range(self.n)]
        self.age = 1

    def local_field(self, i):
        return sum(
            weight * self.spins[j]
            for j, weight in self.graph[i].items()
        )

    def delta_energy(self, i):
        return -2 * self.spins[i] * self.local_field(i)

    def energy(self):
        value = 0.0
        for i in range(self.n):
            for j, weight in self.graph[i].items():
                if j > i:
                    value += weight * self.spins[i] * self.spins[j]
        return value

    def cut_size(self):
        total = 0
        for i in range(self.n):
            for j, weight in self.graph[i].items():
                if j > i and self.spins[i] != self.spins[j]:
                    total += weight
        return total

    def pcm_flip_attempt(self, temperature):
        i = random.randrange(self.n)
        delta = self.delta_energy(i)

        drift = self.age ** (-self.drift_nu)
        thermal_noise = random.gauss(0.0, self.noise) * max(1.0, abs(self.local_field(i)))
        effective_delta = delta + thermal_noise * drift

        if effective_delta <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-effective_delta / max(temperature, 1e-9))

        if accept:
            self.spins[i] *= -1
            return i, True, delta

        return i, False, delta

    def run(self, iterations=1000, temperature=1.0, cooling=0.995):
        history = []
        for _ in range(iterations):
            _, accepted, _ = self.pcm_flip_attempt(temperature)
            temperature *= cooling
            self.age += 1
            history.append({
                "energy": self.energy(),
                "cut": self.cut_size(),
                "accepted": accepted,
                "temperature": temperature,
                "spins": self.spins[:],
            })
        return history
# ----------------------------------------------

def process_text_to_trigram_graph(filepath, max_nodes=200):
    """Builds a graph where nodes are Bigrams and edges represent Trigrams."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().lower()
    except FileNotFoundError:
        print(f"File {filepath} not found. Using a fallback dataset.")
        text = "the quick brown fox jumps over the lazy dog. the lazy dog barks at the brown fox. the quick brown fox runs away quickly into the deep brown forest."

    words = text.split()
    
    # 1. Extract all sequential bigrams from the text
    all_bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    
    # 2. Keep only the most frequent bigrams to control graph size
    top_bigrams = [b for b, _ in Counter(all_bigrams).most_common(max_nodes)]
    bigram_set = set(top_bigrams)
    
    node_to_id = {b: i for i, b in enumerate(top_bigrams)}
    id_to_node = {i: b for i, b in enumerate(top_bigrams)}
    
    n = len(top_bigrams)
    graph = {i: {} for i in range(n)}
    
    # 3. Build Trigram Edges
    # If Bigram A (w1, w2) is followed immediately by Bigram B (w2, w3), they form an edge.
    for i in range(len(words) - 2):
        b1 = (words[i], words[i+1])
        b2 = (words[i+1], words[i+2])
        
        if b1 in bigram_set and b2 in bigram_set:
            id1, id2 = node_to_id[b1], node_to_id[b2]
            if id1 != id2:
                # Undirected graph for Max-Cut
                graph[id1][id2] = graph[id1].get(id2, 0) + 1
                graph[id2][id1] = graph[id2].get(id1, 0) + 1
                
    return graph, node_to_id, id_to_node

def generate_trigram_text(graph, id_to_node, spins, length=30):
    """Walks the trigram graph, bouncing across the Max-Cut partition."""
    n = len(graph)
    
    # Start with a random valid bigram
    current_node = random.choice(range(n))
    # Output stores single words. We start by adding both words of the initial bigram.
    output = list(id_to_node[current_node]) 
    
    for _ in range(length - 2):
        current_spin = spins[current_node]
        neighbors = graph[current_node]
        
        # We only want to cross the cut AND maintain the Markov overlap 
        # (the next bigram's first word MUST match our current bigram's second word)
        current_word_2 = id_to_node[current_node][1]
        
        valid_neighbors = {}
        for neighbor, weight in neighbors.items():
            neighbor_bigram = id_to_node[neighbor]
            # Cross-cut check AND directional overlap check
            if spins[neighbor] != current_spin and neighbor_bigram[0] == current_word_2:
                valid_neighbors[neighbor] = weight
        
        if not valid_neighbors:
            # Fallback: If trapped, jump to a random bigram in the opposite spin 
            # that starts with our last word to maintain grammatical flow.
            opposite_nodes = [i for i, s in enumerate(spins) if s != current_spin and id_to_node[i][0] == current_word_2]
            if opposite_nodes:
                next_node = random.choice(opposite_nodes)
            else:
                # Hard fallback if completely stuck (breaks grammar slightly, keeps algorithm moving)
                next_node = random.choice([i for i, s in enumerate(spins) if s != current_spin])
        else:
            # Probabilistically pick the next bigram based on trigram frequency
            population = list(valid_neighbors.keys())
            weights = list(valid_neighbors.values())
            next_node = random.choices(population, weights=weights, k=1)[0]
            
        # Append only the SECOND word of the new bigram, since the first word overlaps
        output.append(id_to_node[next_node][1])
        current_node = next_node
        
    return " ".join(output)

if __name__ == "__main__":
    dataset_file = "singlekb.txt" 
    
    print("Building trigram graph from text...")
    # 150 nodes = Top 150 most common word PAIRS. 
    graph, node_to_id, id_to_node = process_text_to_trigram_graph(dataset_file, max_nodes=1500)
    
    if len(graph) == 0:
        print("Graph is empty. Check your dataset text.")
        exit()

    print(f"Graph built with {len(graph)} unique nodes (bigrams).")
    print("Running PCMMaxCut Solver...")
    
    solver = PCMMaxCut(graph, noise=0.05, drift_nu=0.08)
    # Trigram graphs are often sparser, so we might need more iterations
    history = solver.run(iterations=8000, temperature=2.5, cooling=0.999)
    
    best = max(history, key=lambda item: item["cut"])
    
    print(f"Max Cut Size: {best['cut']}")
    print(f"Final Energy: {history[-1]['energy']}\n")
    
    print("--- GENERATED TEXT (Trigram Max-Cut Walk) ---")
    generated = generate_trigram_text(graph, id_to_node, best["spins"], length=500)
    print(generated.capitalize() + ".")
