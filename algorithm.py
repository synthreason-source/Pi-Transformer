import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import difflib
from collections import defaultdict, Counter

# ==========================================================
# 1. Dataset Loading & Context Analysis
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

    top_indices = torch.argsort(col_sums, descending=True)
    frequent_trigrams = [idx_to_word[i.item()] for i in top_indices if idx_to_word[i.item()] != unk_trigram]
    frequent_words = list(dict.fromkeys([w for tg in frequent_trigrams for w in tg]))

    return (
        word_to_idx,
        idx_to_word,
        vocab_size,
        exp_boost,
        frequent_trigrams,
        unique_trigrams,
        frequent_words,
        context_to_row,
    )


# ==========================================================
# 2. Logit Triangular Transpose Transformation
# ==========================================================

def apply_triangular_logit_transpose(logits, mode="standard"):
    """
    Applies a geometric triangular transpose on predicted logits.
    Logits are reshaped into local triangular triples (v1, v2, v3) across channels/vocab,
    and transposed.
    
    Modes:
      - 'standard': Reverses vertex orientation (v1 <-> v3 transpose).
      - 'anti': Flips along secondary axis (v1 <-> v2 anti-transpose).
      - 'full_transpose': Performs 2D matrix transpose on packed triangle logit grid.
    """
    if mode == "none" or logits is None:
        return logits

    B, V = logits.shape
    
    # Pad vocabulary dimension to nearest multiple of 3 to form 3-vertex triangles
    pad_len = (3 - (V % 3)) % 3
    if pad_len > 0:
        padded_logits = F.pad(logits, (0, pad_len), value=-1e9)
    else:
        padded_logits = logits

    # Reshape into [B, N, 3] triangular vertex triples
    triangles = padded_logits.view(B, -1, 3)


    # Full 2D triangular matrix transposition
    triangles_transposed = triangles.transpose(1, 2).contiguous()
    triangles_transposed = triangles_transposed.view(B, -1, 3)
   
    # Flatten back to original logit vocabulary shape
    transposed_logits = triangles_transposed.view(B, -1)[:, :V]
    return transposed_logits


# ==========================================================
# 3. Contextual Markov Neural Generator with Logit Transpose
# ==========================================================

class MarkovSeedingLayer(nn.Module):
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

    def forward(self, idx, transpose_mode="none"):
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
        
        # Apply Triangular Transpose directly to the output logits
        logits = apply_triangular_logit_transpose(logits, mode=transpose_mode)
        
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
        transpose_mode="standard",
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

                # Forward pass + Logit Triangular Transpose
                logits = self(cond, transpose_mode=transpose_mode) + amp_boost
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

    markov_seed_layer = MarkovSeedingLayer(vocab_size=vocab_size, embed_dim=64, context_size=5)
    known_words = sorted(set(frequent_words) | {w for tg in word_to_idx.keys() if tg != unk_trigram for w in tg})

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


        print("\n=== GENERATING SAMPLES WITH LOGIT TRIANGULAR TRANSPOSE ===")
        seeds = markov_seed_layer.generate_population_seeds(
            seed_context_idx=context_tensor,
            sequence_length=150,
            pop_size=1,
            amp_boost=amp_boost,
            idx_to_word=idx_to_word,
            unk_id=unk_id,
            temperature=0.8,
            top_k=25,
        )

        flattened_words = [word for trigram in seeds[0] for word in trigram]
        output_text = ' '.join(flattened_words)
        print(f"{output_text}")

        print("-" * 65)
