import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import sys
from typing import List, Tuple, Dict

# ==========================================================
# 1. Standardized Word Tokenizer (Fixes Vocabulary Breakdown)
# ==========================================================

def load_and_tokenize(filename: str, context_size: int = 5):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    # Clean non-alphanumeric noise that introduces corrupted tokens
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.!?]', '', raw_text)
    words = cleaned_text.lower().split()

    if len(words) < context_size + 2:
        raise ValueError("Dataset is too small for the selected context size.")

    vocab = sorted(list(set(words)))
    vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + vocab
    
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}

    token_ids = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in words]
    dataset_tensor = torch.tensor(token_ids, dtype=torch.long)

    return word_to_idx, idx_to_word, len(vocab), dataset_tensor

# ==========================================================
# 2. Dynamic Hypernetwork Topological Layer
# ==========================================================

class RobustTopologicalLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, context_size: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim

        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(context_size, embed_dim)
        
        in_features = embed_dim * context_size

        # Dynamic Hypernetwork for prompt-conditioned trajectory generation
        self.hyper_W = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim * in_features)
        )
        self.hyper_b = nn.Linear(in_features, embed_dim)

        self.lm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size)
        )

    def extract_latent_state(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        idx_sub = idx[:, -self.context_size:]

        w_emb = self.word_embedding(idx_sub)
        positions = torch.arange(0, self.context_size, device=idx.device)
        p_emb = self.pos_embedding(positions)

        x = (w_emb + p_emb).reshape(B, -1)  # [B, embed_dim * context_size]

        W_dyn = self.hyper_W(x).reshape(B, self.embed_dim, -1)
        b_dyn = self.hyper_b(x)

        h_trans = torch.bmm(x.unsqueeze(1), W_dyn.transpose(1, 2)).squeeze(1) + b_dyn
        return h_trans

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_state = self.extract_latent_state(idx)
        logits = self.lm_head(h_state)
        return logits, h_state

# ==========================================================
# 3. Training Loop with Topological Loss Curriculum
# ==========================================================

def train_topological_knot(
    model: RobustTopologicalLayer,
    dataset_tensor: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 32,
    rollout_steps: int = 6,
    margin_delta: float = 1.5
):
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion_lm = nn.CrossEntropyLoss(ignore_index=0)

    seq_len = model.context_size
    num_samples = len(dataset_tensor) - seq_len - rollout_steps

    print(f"\n=== TRAINING WITH TOPOLOGICAL CURRICULUM ===")

    for epoch in range(1, epochs + 1):
        # Curriculum: Gradually ramp up closure penalties
        lambda_iso = min(1.0, epoch / 5.0)
        lambda_repel = min(0.5, epoch / 8.0)

        total_loss, total_lm, total_iso, total_repel = 0.0, 0.0, 0.0, 0.0
        permutation = torch.randperm(num_samples)

        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            if len(indices) < 2:
                continue

            batch_inputs = torch.stack([dataset_tensor[j:j + seq_len] for j in indices])
            batch_targets = dataset_tensor[indices + seq_len]

            optimizer.zero_grad()

            logits, h_start = model(batch_inputs)
            lm_loss = criterion_lm(logits, batch_targets)

            curr_ctx = batch_inputs.clone()
            intermediate_states = []

            for step in range(rollout_steps):
                step_logits, h_t = model(curr_ctx)
                next_tokens = torch.argmax(step_logits, dim=-1, keepdim=True)
                curr_ctx = torch.cat([curr_ctx[:, 1:], next_tokens], dim=1)

                if step < rollout_steps - 1:
                    intermediate_states.append(h_t)
                else:
                    h_end = h_t

            # 1. Isomorphic Boundary Closure
            iso_loss = F.mse_loss(h_end, h_start)

            # 2. Non-Isomorphic Repulsion (Avoid Self-Enclosure during rollout)
            repel_loss = torch.tensor(0.0, device=dataset_tensor.device)
            for h_inter in intermediate_states:
                dist = torch.norm(h_inter - h_start, p=2, dim=-1)
                violation = F.relu(margin_delta - dist)
                repel_loss = repel_loss + torch.mean(violation ** 2)

            step_loss = lm_loss + (lambda_iso * iso_loss) + (lambda_repel * repel_loss)
            step_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += step_loss.item()
            total_lm += lm_loss.item()
            total_iso += iso_loss.item()
            total_repel += repel_loss.item()

        num_batches = max(1, num_samples // batch_size)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Loss: {total_loss/num_batches:.4f} | "
            f"LM: {total_lm/num_batches:.4f} | "
            f"Iso (\u03BB={lambda_iso:.1f}): {total_iso/num_batches:.4f} | "
            f"Repel (\u03BB={lambda_repel:.1f}): {total_repel/num_batches:.4f}"
        )

# ==========================================================
# 4. Interactive Execution & Metrics Output
# ==========================================================

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else input("Dataset Text Filename: ").strip()

    word_to_idx, idx_to_word, vocab_size, dataset_tensor = load_and_tokenize(filename, context_size=5)

    model = RobustTopologicalLayer(vocab_size=vocab_size, embed_dim=64, context_size=5)
    train_topological_knot(model, dataset_tensor, epochs=2, batch_size=32, rollout_steps=6)

    model.eval()
    unk_id = word_to_idx["<UNK>"]

    while True:
        raw_input_text = input("\nUSER Prompt: ").strip()
        if raw_input_text.lower() in {"quit", "exit"}:
            break

        tokens = re.sub(r'[^a-zA-Z0-9\s]', '', raw_input_text).lower().split()
        tokens = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        if not tokens:
            continue

        prompt_ids = [word_to_idx.get(w, unk_id) for w in tokens]
        while len(prompt_ids) < 5:
            prompt_ids.insert(0, word_to_idx["<PAD>"])
        
        ctx_tensor = torch.tensor([prompt_ids[-5:]], dtype=torch.long)

        generated_tokens = []
        curr_ctx = ctx_tensor.clone()
        distances = []

        with torch.no_grad():
            _, h_start = model(ctx_tensor)

            for step in range(256):
                logits, h_t = model(curr_ctx)
                
                # Temperature sampling to maintain valid word dynamics
                probs = F.softmax(logits / 0.8, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

                word = idx_to_word[next_id.item()]
                generated_tokens.append(word)

                dist = torch.norm(h_t - h_start, p=2).item()
                distances.append(dist)

                curr_ctx = torch.cat([curr_ctx[:, 1:], next_id], dim=1)

                # Early termination if loop closes on boundary state
                if dist < 0.4 and step > 2:
                    break

        prompt_words = [idx_to_word[i] for i in ctx_tensor[0].tolist() if i != 0]

        print("\n=== TOPOLOGICAL TRAJECTORY RESULTS ===")
        print(f"Prompt Start State  : {' '.join(prompt_words)}")
        print(f"Generated Sequence  : {' '.join(generated_tokens)}")
        print(f"Latent Distances h_t: {[round(d, 3) for d in distances]}")
        print("=" * 65)
