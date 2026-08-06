import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
import difflib
import pandas as pd

# ==========================================
# 1. Dataset Loading & Preprocessing
# ==========================================
def load_and_prepare_data(filename):
    """
    Reads text, builds vocabulary, and prepares a standard Trigram dataset 
    (Context: 2 words -> Target: 1 word).
    """
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    tokens = raw_text.lower().split()
    
    # Build word-level vocabulary
    unique_words = sorted(list(set(tokens)))
    unk_token = "<UNK>"
    if unk_token not in unique_words:
        unique_words.append(unk_token)
        
    vocab_size = len(unique_words)
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    idx_to_word = {i: w for i, w in enumerate(unique_words)}
    
    # Build dataset (Context = 2 previous words, Target = next word)
    contexts, targets = [], []
    for i in range(len(tokens) - 2):
        ctx = [word_to_idx[tokens[i]], word_to_idx[tokens[i+1]]]
        tgt = word_to_idx[tokens[i+2]]
        contexts.append(ctx)
        targets.append(tgt)
        
    X = torch.tensor(contexts, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    
    return X, Y, word_to_idx, idx_to_word, vocab_size


# ==========================================
# 2. CSV Frequency Updater (Applying Hats)
# ==========================================
def csv_freq_updater(logits, csv_filename, word_to_idx, vocab_size, hat_power=0.5):
    """
    Reads ai_instructions_diverse_2.csv, extracts vocabulary counts, and 
    applies a 'hat' (power) to the frequencies to create a logits boost.
    """
    freq_boost = torch.zeros(vocab_size)
    try:
        df = pd.read_csv(csv_filename)
        
        # Flatten all text columns to extract raw word occurrences
        text = " ".join(df.astype(str).values.flatten()).lower()
        tokens = text.split()
        
        counts = defaultdict(int)
        for t in tokens:
            counts[t] += 1
            
        for word, idx in word_to_idx.items():
            if word in counts:
                # Apply the "hat" power scale
                
                freq_boost[idx] = float(counts[word]) ** hat_power
                
        # Normalize the boost to a sensible range to avoid blowing up logits
        max_val = freq_boost.max()
        if max_val > 0:
            log_hat = torch.log(freq_boost.to(logits.device) + eps)

            freq_boost = (freq_boost+log_hat / max_val) * 2.0 
            
        print(f"--- Successfully applied frequency hats from {csv_filename} ---")
        
    except Exception as e:
        print(f"Warning: Could not process {csv_filename} for frequency updating: {e}")
        
    return freq_boost


# ==========================================
# 3. Upgraded Neural Architecture (with Attention)
# ==========================================
class AttentionTrigramLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.context_size = 2 
        
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(self.context_size, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.lm_head = nn.Linear(embed_dim * self.context_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        w_emb = self.word_embedding(idx) 
        
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
        p_emb = self.pos_embedding(positions) 
        
        x = w_emb + p_emb 
        
        # Apply Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out) # Residual connection
        
        x = x.reshape(B, -1) 
        logits = self.lm_head(x) 
        
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_words, freq_boost=None, unk_id=None, temperature=1.0):
        for _ in range(max_new_words):
            idx_cond = idx[:, -self.context_size:]
            logits = self(idx_cond)
            
            # Inject the CSV frequency hats to bias the generation
            if freq_boost is not None:
                logits = csv_freq_updater(logits, "ai_instructions_diverse.csv", word_to_idx, vocab_size, hat_power=1.5)

            if unk_id is not None:
                logits[:, unk_id] = -float('inf')
                
            logits = logits / temperature 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# ==========================================
# 4. Training Loop
# ==========================================
def train_model(model, X, Y, epochs=5, batch_size=128, lr=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\n--- Training Model ---")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
    model.eval()


# ==========================================
# 5. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    filename = input("Filename for initial training: ")
    X, Y, word_to_idx, idx_to_word, vocab_size = load_and_prepare_data(filename)
    
    model = AttentionTrigramLM(vocab_size=vocab_size)
    train_model(model, X, Y)
    
    # Process ai_instructions_diverse_2.csv and retrieve the hat matrix

    while True:
        raw_input = input("\nUSER (Seed with at least 2 words): ").strip()
        if not raw_input: continue
        
        tokens = raw_input.lower().split()
        unk_id = word_to_idx["<UNK>"]
        known_words = list(word_to_idx.keys())
    
        # Fuzzy match to ensure words exist in vocabulary
        corrected_tokens = []
        for w in tokens:
            matches = difflib.get_close_matches(w, known_words, n=1, cutoff=0.6)#conceptual control
            corrected_tokens.append(matches[0] if matches else known_words[0])
    
        # Ensure we have exactly 2 words for the context window
        if len(corrected_tokens) < model.context_size:
            corrected_tokens = (known_words[:2] + corrected_tokens)[-2:]
        else:
            corrected_tokens = corrected_tokens[-2:]
    
        context = torch.tensor([[word_to_idx[w] for w in corrected_tokens]], dtype=torch.long)
        
        print("Generating sequence...")
        generated_indices = model.generate(
            context, 
            max_new_words=500, 
            unk_id=unk_id,
            temperature=0.8
        )[0].tolist()
        
        generated_text = " ".join([idx_to_word[i] for i in generated_indices])
        
        print("\n--- GENERATED OUTPUT ---")
        print(generated_text)
