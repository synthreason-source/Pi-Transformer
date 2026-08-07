import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
import re

EPS = 1e-8

# ==========================================
# 1. Horizontal Sizing: find the best context width
# ==========================================
def build_contexts_for_size(tokens, word_to_idx, unk_id, ctx_size):
    """Plain (context -> target) pairs for one fixed context width."""
    contexts, targets = [], []
    for i in range(len(tokens) - ctx_size):
        ctx = [word_to_idx.get(tokens[i + k], unk_id) for k in range(ctx_size)]
        tgt = word_to_idx.get(tokens[i + ctx_size], unk_id)
        contexts.append(ctx)
        targets.append(tgt)
    if not contexts:
        return None, None
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def compute_richness_accuracy(X, Y):
    """
    richness  = how many distinct contexts this width produces relative to
                the number of examples (wider context -> more distinct
                situations -> richer signal, up to a point).
    accuracy  = how predictable the target is *given* that context
                (majority-vote fraction per context). Too-wide contexts are
                mostly unique (accuracy -> 1 trivially) so we balance both.
    """
    ctx_to_targets = defaultdict(list)
    for ctx, tgt in zip(X.tolist(), Y.tolist()):
        ctx_to_targets[tuple(ctx)].append(tgt)

    total = len(Y)
    richness = len(ctx_to_targets) / total

    correct = 0
    for tgts in ctx_to_targets.values():
        counts = defaultdict(int)
        for t in tgts:
            counts[t] += 1
        correct += max(counts.values())
    accuracy = correct / total

    return richness, accuracy


def find_optimal_context_size(tokens, word_to_idx, unk_id,
                               min_ctx=1, max_ctx=6, patience=2, richness_weight=0.5):
    """
    Grows the context window width (horizontal sizing) one word at a time,
    scoring each width by a blend of richness and accuracy, and stops
    automatically once the score stops improving for `patience` steps.
    """
    best_size, best_score, stale = min_ctx, -1.0, 0
    history = []

    print("--- Horizontal sizing: searching for optimal context width ---")
    for ctx_size in range(min_ctx, max_ctx + 1):
        X_c, Y_c = build_contexts_for_size(tokens, word_to_idx, unk_id, ctx_size)
        if X_c is None:
            break

        richness, accuracy = compute_richness_accuracy(X_c, Y_c)
        score = richness_weight * richness + (1 - richness_weight) * accuracy
        history.append((ctx_size, richness, accuracy, score))
        print(f"  ctx_size={ctx_size:2d}  richness={richness:.4f}  "
              f"accuracy={accuracy:.4f}  score={score:.4f}")

        if score > best_score + 1e-4:
            best_score, best_size, stale = score, ctx_size, 0
        else:
            stale += 1
        if stale >= patience:
            print(f"  -> plateaued after ctx_size={ctx_size}, stopping search")
            break

    print(f"--- Selected context width = {best_size} (score={best_score:.4f}) ---")
    return best_size, history


# ==========================================
# 2. Vertical Generate: stack every context depth per target
# ==========================================
def build_vertical_stack_dataset(tokens, word_to_idx, unk_id, optimal_ctx, pad_id):
    """
    For every target position t, generate one training row for EACH depth
    d = 1..optimal_ctx (left-padded with pad_id up to optimal_ctx). This is
    the 'vertical' stack: the same horizontal position in the text produces
    multiple rows of increasing context richness, so the model learns to
    predict correctly whether it's handed a short or a full-width context.
    """
    idx_tokens = [word_to_idx.get(t, unk_id) for t in tokens]
    contexts, targets = [], []

    for t in range(1, len(idx_tokens)):
        max_d = min(optimal_ctx, t)
        for d in range(1, max_d + 1):
            raw_ctx = idx_tokens[t - d:t]
            pad_len = optimal_ctx - d
            ctx = [pad_id] * pad_len + raw_ctx
            contexts.append(ctx)
            targets.append(idx_tokens[t])

    X = torch.tensor(contexts, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    return X, Y


# ==========================================
# 3. Dataset Loading & Preprocessing (adaptive width IS the training feature)
# ==========================================
def load_and_prepare_data(filename, min_ctx=1, max_ctx=6):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    tokens = raw_text.lower().split()

    unique_words = sorted(set(tokens))
    for special in ("<UNK>", "<PAD>"):
        if special not in unique_words:
            unique_words.append(special)

    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    idx_to_word = {i: w for i, w in enumerate(unique_words)}
    unk_id = word_to_idx["<UNK>"]
    pad_id = word_to_idx["<PAD>"]
    vocab_size = len(unique_words)

    # 1) Horizontal sizing: auto-discover the best context width
    optimal_ctx, history = find_optimal_context_size(
        tokens, word_to_idx, unk_id, min_ctx=min_ctx, max_ctx=max_ctx
    )

    # 2) Vertical generate: stack every depth 1..optimal_ctx per position
    X, Y = build_vertical_stack_dataset(tokens, word_to_idx, unk_id, optimal_ctx, pad_id)
    print(f"--- Built {len(X)} vertically-stacked examples "
          f"(context width={optimal_ctx}, {len(tokens)} tokens) ---")

    return X, Y, word_to_idx, idx_to_word, unk_id, pad_id, vocab_size, optimal_ctx


# ==========================================
# 4. CSV Frequency Updater (Applying Hats)
# ==========================================
def csv_freq_updater(logits, csv_filename, word_to_idx, vocab_size, hat_power=0.5):
    freq_boost = torch.zeros(vocab_size)
    try:
        df = pd.read_csv(csv_filename)
        text = " ".join(df.astype(str).values.flatten()).lower()
        tokens = text.split()

        counts = defaultdict(int)
        for t in tokens:
            counts[t] += 1

        for word, idx in word_to_idx.items():
            if word in counts:
                freq_boost[idx] = float(counts[word]) ** hat_power

        max_val = freq_boost.max()
        if max_val > 0:
            log_hat = torch.log(freq_boost.to(logits.device) + EPS)
            freq_boost = (freq_boost + log_hat / max_val) * 2.0

        print(f"--- Successfully applied frequency hats from {csv_filename} ---")

    except Exception as e:
        print(f"Warning: Could not process {csv_filename} for frequency updating: {e}")

    return freq_boost


# ==========================================
# 5. Architecture (context_size is now dynamic, driven by horizontal sizing)
# ==========================================
class AttentionTrigramLM(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim=64):
        super().__init__()
        self.context_size = context_size

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
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)

        x = x.reshape(B, -1)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_words, pad_id, unk_id=None, temperature=1.0,
                 freq_boost_csv=None, word_to_idx=None, vocab_size=None):
        for _ in range(max_new_words):
            idx_cond = idx[:, -self.context_size:]
            if idx_cond.shape[1] < self.context_size:
                pad = torch.full(
                    (idx_cond.shape[0], self.context_size - idx_cond.shape[1]),
                    pad_id, dtype=torch.long, device=idx.device
                )
                idx_cond = torch.cat([pad, idx_cond], dim=1)

            logits = self(idx_cond)

            if freq_boost_csv is not None and word_to_idx is not None:
                boost = csv_freq_updater(logits, freq_boost_csv, word_to_idx, vocab_size, hat_power=1.5)
                logits = logits + boost.to(logits.device)

            if unk_id is not None:
                logits[:, unk_id] = -float('inf')
            logits[:, pad_id] = -float('inf')

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ==========================================
# 6. Training Loop
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

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")
    model.eval()


# ==========================================
# 7. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    filename = input("Filename for initial training: ")
    (X, Y, word_to_idx, idx_to_word, unk_id, pad_id,
     vocab_size, optimal_ctx) = load_and_prepare_data(filename)

    model = AttentionTrigramLM(vocab_size=vocab_size, context_size=optimal_ctx)
    train_model(model, X, Y)

    known_words = list(word_to_idx.keys())

    while True:
        raw_input_text = input("\nUSER (seed text): ").strip()
        tokens = raw_input_text.lower().split()

        import difflib
        corrected_tokens = []
        for w in tokens:
            matches = difflib.get_close_matches(w, known_words, n=1, cutoff=0.6)
            corrected_tokens.append(matches[0] if matches else "<UNK>")

        seed_ids = [word_to_idx.get(w, unk_id) for w in corrected_tokens]
        if len(seed_ids) < optimal_ctx:
            seed_ids = [pad_id] * (optimal_ctx - len(seed_ids)) + seed_ids
        else:
            seed_ids = seed_ids[-optimal_ctx:]

        context = torch.tensor([seed_ids], dtype=torch.long)

        print("Generating sequence...")
        generated_indices = model.generate(
            context,
            max_new_words=500,
            pad_id=pad_id,
            unk_id=unk_id,
            temperature=0.8,
        )[0].tolist()

        generated_text = " ".join(idx_to_word[i] for i in generated_indices if i != pad_id)

        print("\n--- GENERATED OUTPUT ---")
        print(generated_text)
