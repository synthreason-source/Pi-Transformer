import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli

# --- 1. QUANTUM FEATURE SEEDER ---
class QuantumFeatureSeeder(nn.Module):
    def __init__(self, script_lines: list[str], num_qubits: int = 13):
        super().__init__()
        self.script_lines = script_lines
        self.num_qubits = num_qubits
        self.quantum_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.theta_param = Parameter("theta")
        self.base_qc = self.build_base_circuit()
        self.pauli_z = Pauli("Z")

    def build_base_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for line in self.script_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            op = parts[0].lower()

            if op in {"h", "x", "y", "z", "s", "t"}:
                qc.__getattribute__(op)(int(parts[1]))
            elif op in {"cx", "cz", "swap"}:
                qc.__getattribute__(op)(int(parts[1]), int(parts[2]))
            elif op in {"rx", "ry", "rz"}:
                base_angle = float(parts[1])
                q = int(parts[2]) if len(parts) > 2 else 0
                if op == "ry" and abs(base_angle - 0.346) < 1e-5:
                    qc.ry(self.theta_param, q)
                else:
                    qc.__getattribute__(op)(base_angle, q)
            elif op == "u":
                qc.u(float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]))
        return qc

    def forward(self, scalar_input: torch.Tensor) -> torch.Tensor:
        bound_angle = (self.quantum_weight * scalar_input).item()
        bound_qc = self.base_qc.assign_parameters({self.theta_param: bound_angle})
        
        state = Statevector(bound_qc)
        exp_z1 = state.expectation_value(self.pauli_z, qargs=[1]).real
        exp_z12 = state.expectation_value(self.pauli_z, qargs=[12]).real
        
        return torch.tensor([exp_z1, exp_z12], dtype=torch.float32)


# --- 2. CORRELATED N-GRAM & QUANTUM HYBRID MODEL ---
class QuantumCorrelatedTextGenerator(nn.Module):
    def __init__(self, script_lines: list[str], vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.quantum_seeder = QuantumFeatureSeeder(script_lines)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text_tensor: torch.Tensor, hidden=None):
        batch_size, seq_len = text_tensor.shape
        embedded = self.embedding(text_tensor)
        
        scalar_seed = text_tensor[:, 0].float().mean() / 1000.0
        q_features = self.quantum_seeder(scalar_seed)
        
        q_broadcast = q_features.repeat(batch_size, seq_len, 1)
        lstm_input = torch.cat([embedded, q_broadcast], dim=-1)
        
        out, hidden = self.lstm(lstm_input, hidden)
        logits = self.fc(out)
        return logits, hidden


# --- 3. EXTRACT WORD CORRELATIONS (Trigrams/Bigrams) ---
def build_correlation_matrix(words):
    """Builds a statistical transition map of correlated words from the dataset."""
    trigrams = defaultdict(Counter)
    bigrams = defaultdict(Counter)
    
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i+1], words[i+2]
        trigrams[(w1, w2)][w3] += 1
        bigrams[w1][w2] += 1
        
    return dict(trigrams), dict(bigrams)


# --- 4. HYBRID GENERATION (LSTM + Quantum + Correlation Fallback) ---
def generate_correlated_text(model, prompt_str, trigrams, bigrams, vocab_to_int, int_to_vocab, max_words=15, temperature=0.7):
    model.eval()
    words = prompt_str.split()
    
    with torch.no_grad():
        for _ in range(max_words):
            # Try Trigram correlation lookup first if context allows
            next_word = None
            if len(words) >= 2:
                context = (words[-2], words[-1])
                if context in trigrams:
                    possible_next = trigrams[context]
                    candidates, counts = zip(*possible_next.items())
                    total = sum(counts)
                    probs = [c / total for c in counts]
                    # Select correlated word statistically
                    next_word = torch.multinomial(torch.tensor(probs), num_samples=1).item()
                    next_word = candidates[next_word]

            # Fallback to Bigram if trigram misses
            if not next_word and len(words) >= 1:
                context = words[-1]
                if context in bigrams:
                    possible_next = bigrams[context]
                    candidates, counts = zip(*possible_next.items())
                    total = sum(counts)
                    probs = [c / total for c in counts]
                    next_word = torch.multinomial(torch.tensor(probs), num_samples=1).item()
                    next_word = candidates[next_word]

            # If statistical correlations fail, use the Neural + Quantum model
            if not next_word:
                prompt_indices = [vocab_to_int.get(w, 0) for w in words[-5:]]
                input_tensor = torch.tensor([prompt_indices], dtype=torch.long)
                logits, _ = model(input_tensor)
                next_word_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_word_logits, dim=-1)
                next_word_idx = torch.multinomial(probs, num_samples=1).item()
                next_word = int_to_vocab.get(next_word_idx, "the")

            words.append(next_word)
            
    return " ".join(words)


if __name__ == "__main__":
    # Load dataset corpus
    try:
        with open("singlekb.txt", "r", encoding="utf-8") as file:
            dataset_corpus = file.read()[:20000]
    except FileNotFoundError:
        dataset_corpus = "quantum computing and neural networks integration allows hybrid workflow optimization cascades. quantum machines process data using superposition and entanglement principles."

    words = dataset_corpus.split()
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i: word for i, word in enumerate(vocab)}

    # Extract correlated word dictionaries (Trigrams & Bigrams)
    trigrams, bigrams = build_correlation_matrix(words)

    # Prepare dataset training sequences for the neural network component
    seq_length = 8
    inputs, targets = [], []
    for i in range(len(words) - seq_length):
        seq_in = words[i:i + seq_length]
        seq_out = words[i + 1:i + seq_length + 1]
        inputs.append([vocab_to_int[w] for w in seq_in])
        targets.append([vocab_to_int[w] for w in seq_out])

    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)

    raw_cascade_text = [
        "h:1", "h:12", "h:4", 
        "cx:10:1", "cx:6:2", "cx:2:11", "cx:0:1", "cx:4:5", 
        "h:1", "h:3", 
        "ry:0.346:0", 
        "cx:5:1", "cx:2:5", 
        "h:1", "h:3"
    ]

    model = QuantumCorrelatedTextGenerator(raw_cascade_text, vocab_size, embed_dim=32, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("--- Training Hybrid Quantum-Correlated Text Generator ---")
    for epoch in range(25):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/25 | Loss: {loss.item():.4f}")
    while True:
        test_prompt = input("USER: ")
        result = generate_correlated_text(model, test_prompt, trigrams, bigrams, vocab_to_int, int_to_vocab, max_words=600, temperature=0.7)
    
        print("\n--- Natural Language Correlation Result ---")
        print(f"Prompt: '{test_prompt}'")
        print(f"Generated Output: {result}")
