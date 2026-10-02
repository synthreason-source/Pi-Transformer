import math
import random
import re
from collections import Counter

# --- ORIGINAL SOLVER KEPT EXACTLY THE SAME ---
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
        text = "the quick brown fox jumps over the lazy dog. the lazy dog barks at the brown fox. the quick brown fox runs away quickly into the deep dark brown forest. the dark forest is full of mystery. a dog barks loudly in the dark."

    words = text.split()
    all_bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    top_bigrams = [b for b, _ in Counter(all_bigrams).most_common(max_nodes)]
    bigram_set = set(top_bigrams)
    
    node_to_id = {b: i for i, b in enumerate(top_bigrams)}
    id_to_node = {i: b for i, b in enumerate(top_bigrams)}
    
    n = len(top_bigrams)
    graph = {i: {} for i in range(n)}
    
    for i in range(len(words) - 2):
        b1 = (words[i], words[i+1])
        b2 = (words[i+1], words[i+2])
        
        if b1 in bigram_set and b2 in bigram_set:
            id1, id2 = node_to_id[b1], node_to_id[b2]
            if id1 != id2:
                graph[id1][id2] = graph[id1].get(id2, 0) + 1
                graph[id2][id1] = graph[id2].get(id1, 0) + 1
                
    return graph, node_to_id, id_to_node

def generate_markovian_prompt_text(graph, id_to_node, spins, prompt_vocab, length=30):
    """
    Populates an intermediate probability array dynamically based on Markov edge weights, 
    Max-Cut alignment, and Prompt vocabulary matching.
    """
    n = len(graph)
    
    # Try to start on a node that contains a prompt word, otherwise random
    valid_starts = [i for i in range(n) if any(w in prompt_vocab for w in id_to_node[i])]
    current_node = random.choice(valid_starts) if valid_starts else random.choice(range(n))
    
    output = list(id_to_node[current_node]) 
    
    for _ in range(length - 2):
        current_spin = spins[current_node]
        neighbors = graph[current_node]
        current_word_2 = id_to_node[current_node][1]
        
        # 1. Find all grammatically valid overlapping next bigrams
        valid_neighbors = [
            neighbor for neighbor in neighbors.keys() 
            if id_to_node[neighbor][0] == current_word_2
        ]
        
        if not valid_neighbors:
            # Fallback if trapped
            fallback_nodes = [i for i, s in enumerate(spins) if s != current_spin and id_to_node[i][0] == current_word_2]
            next_node = random.choice(fallback_nodes) if fallback_nodes else random.choice(range(n))
        else:
            # 2. Populate the intermediate probabilities array
            probabilities = []
            
            for neighbor in valid_neighbors:
                base_weight = graph[current_node][neighbor]
                
                # Max-Cut influence: 5x more likely to pick a path that crosses the Cut
                cut_multiplier = 5.0 if spins[neighbor] != current_spin else 1.0
                
                # Prompt influence: 50x more likely to pick a path leading to a prompt word
                neighbor_words = id_to_node[neighbor]
                prompt_multiplier = 50.0 if any(w in prompt_vocab for w in neighbor_words) else 1.0
                
                # Calculate intermediate probability score
                score = base_weight * cut_multiplier * prompt_multiplier
                probabilities.append(score)
            
            # Pick the next node proportionally based on our constructed probability array
            next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
            
        output.append(id_to_node[next_node][1])
        current_node = next_node
        
    return " ".join(output)

if __name__ == "__main__":
    dataset_file = input("Filename: ")
    
    print("Building trigram graph from text...")
    graph, node_to_id, id_to_node = process_text_to_trigram_graph(dataset_file, max_nodes=15000)
    
    if len(graph) == 0:
        print("Graph is empty. Check your dataset text.")
        exit()
    print(f"Graph built with {len(graph)} unique nodes (bigrams).\n")
    
    # Run Max Cut once to establish the structural partitions (Spins)
    print("Running PCMMaxCut Solver once for structural baseline...")
    solver = PCMMaxCut(graph, noise=0.01, drift_nu=0.00001)
    history = solver.run(iterations=5000, temperature=0.0005, cooling=0.0000199)
    best_cut = max(history, key=lambda item: item["cut"])
    best_spins = best_cut["spins"]
    print(f"Base partition created. Max Cut Size: {best_cut['cut']}\n")

    # Dynamic Prompt Testing Using Intermediate Probabilities
    
    while True:  
        # Generation is now instantly Markovian based on the prompt array
        final_text = generate_markovian_prompt_text(
            graph, 
            id_to_node, 
            best_spins, 
            input("USER: ").split(), 
            length=400
       )
            
        print("Generated Output:")
        print(f"> {final_text.capitalize()}.\n")
