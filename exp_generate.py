import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import defaultdict
import math
import difflib

# ==========================================
# 1. Dataset Loading, Tokenization & Frequencies
# ==========================================
def load_and_analyze_dataset(filename):
    """
    Reads text, builds a 2D context-by-target trigram frequency matrix,
    applies exponential scaling to the frequencies, and performs
    sine-wave smoothing along the context (row) axis.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    tokens = raw_text.lower().split()
    
    # Create the sequence of Trigrams
    trigram_sequence = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    
    # Build Vocabulary
    unique_trigrams = sorted(list(set(trigram_sequence)))
    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    if unk_trigram not in unique_trigrams:
        unique_trigrams.append(unk_trigram) 
    vocab_size = len(unique_trigrams)

    word_to_idx = {t: i for i, t in enumerate(unique_trigrams)}
    idx_to_word = {i: t for i, t in enumerate(unique_trigrams)}

    # Frequency Matrix Calculation
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
    print(f"Matrix analyzed: {row_count} unique rows (contexts), {vocab_size} trigram columns.")
    
    # Exponential frequency boost
    max_freq = col_sums.max() + 1e-5
    normalized_inv_freq = 1.0 - (col_sums / max_freq)
    exp_boost = torch.exp(normalized_inv_freq * 2.0)

    # Sine-wave smoothing on the row dimension
    row_indices = torch.linspace(0, 2 * math.pi, steps=row_count)
    sine_smoother = 0.5 * (1.0 + torch.sin(row_indices)).unsqueeze(1)

    smoothed_matrix = freq_matrix * sine_smoother
    row_smoothed_weight = smoothed_matrix.sum(dim=0) / (smoothed_matrix.sum() + 1e-5)

    amplification_boost = exp_boost * (1.0 + row_smoothed_weight)
    amplification_boost[word_to_idx[unk_trigram]] = 0.0

    # Extract top most frequent trigrams for deterministic fallback
    top_indices = torch.argsort(col_sums, descending=True)
    frequent_trigrams = [idx_to_word[i.item()] for i in top_indices if idx_to_word[i.item()] != unk_trigram]

    return word_to_idx, idx_to_word, vocab_size, amplification_boost, frequent_trigrams

# ==========================================
# 2. Defining the Trigram Neural Architecture
# ==========================================
class TrigramNeuralLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.context_size = 2 
        
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
        logits = self.lm_head(x) 
        
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_words, amplification_boost, unk_id=None, temperature=1.0):
        for _ in range(max_new_words):
            idx_cond = idx[:, -self.context_size:]
            logits = self(idx_cond)
            
            logits = logits + amplification_boost.to(logits.device)
            
            if unk_id is not None:
                logits[:, unk_id] = -float('inf')
                
            logits = logits / temperature 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# ==========================================
# 3. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    word_to_idx, idx_to_word, vocab_size, amp_boost, frequent_trigrams = load_and_analyze_dataset("singlekb.txt")
    
    # 2. Initialize Model
    model = TrigramNeuralLM(vocab_size=vocab_size)

    while True:
        # 3. Process User Input Deterministically
        raw_input = input("USER: ").strip()
        tokens = raw_input.lower().split() if raw_input else []
    
        unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
        unk_id = word_to_idx[unk_trigram]
    
        # Collect all single words in vocabulary
        known_words = sorted(list({w for tg in word_to_idx.keys() if tg != unk_trigram for w in tg}))
    
        # STEP A: Fuzzy match unknown words to nearest known word deterministically
        corrected_tokens = []
        for w in tokens:
            matches = difflib.get_close_matches(w, known_words, n=1, cutoff=0.0)
            if matches:
                corrected_tokens.append(matches[0])
    
        # STEP B: Construct trigrams from corrected tokens
        seed_trigrams = [tuple(corrected_tokens[i:i+3]) for i in range(len(corrected_tokens) - 2)]
        valid_trigrams = [tg for tg in seed_trigrams if tg in word_to_idx]
    
        # STEP C: Deterministic Fallback to most frequent trigram in dataset if no matches found
        if not valid_trigrams:
            print("No exact matches found. Falling back to highest frequency dataset trigrams...")
            seed_trigrams = frequent_trigrams[:model.context_size]
        else:
            seed_trigrams = valid_trigrams
    
        # Pad deterministically by repeating the last valid trigram
        while len(seed_trigrams) < model.context_size:
            seed_trigrams.append(seed_trigrams[-1])
    
        context = torch.tensor(
            [[word_to_idx[tg] for tg in seed_trigrams]], 
            dtype=torch.long
        )
        
        # 4. Generate Trigram Sequence
        print("\nGenerating trigram text sequence...")
        generated_indices = model.generate(
            context, 
            max_new_words=50, 
            amplification_boost=amp_boost, 
            unk_id=unk_id,
            temperature=0.8
        )[0].tolist()
        
        # 5. Decode output
        generated_trigrams = [idx_to_word[i] for i in generated_indices]
        words = [w for tg in generated_trigrams for w in tg if w != "<UNK>"]
        generated_text = " ".join(words)
        
        print("\n--- GENERATED OUTPUT ---")
        print(generated_text)
