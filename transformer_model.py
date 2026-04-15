import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import os
import sys

print("LLM Loading...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------------------------------
# 1. TRIGRAM DATASET
#    input: 2 words
#    label: next word
# -------------------------------------------------
class TrigramDataset(Dataset):
    def __init__(self, text: str, vocab_size=8192, max_samples=None):
        toks = text.lower().split()
        if len(toks) < 3:
            raise ValueError("Need at least 3 words for trigram training.")

        counts = Counter(toks)
        vocab_list = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(vocab_size - 2)]
        self.word_to_id = {w: i for i, w in enumerate(vocab_list)}
        self.id_to_word = vocab_list
        self.vocab_size = len(vocab_list)

        def encode(w):
            return self.word_to_id.get(w, 1)  # <unk>

        self.data = []
        for i in range(len(toks) - 2):
            x0 = encode(toks[i])
            x1 = encode(toks[i + 1])
            y = encode(toks[i + 2])
            self.data.append((x0, x1, y))

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"TrigramDataset: {len(self.data)} samples | vocab_size={self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x0, x1, y = self.data[idx]
        return {
            "input_ids": torch.tensor([x0, x1], dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
        }

# -------------------------------------------------
# 2. TRIGRAM MODEL
#    predicts next word from previous 2 words
# -------------------------------------------------
class TrigramLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, vocab_size)
        )

    def forward(self, x):
        # x: (B, 2)
        emb = self.embed(x)          # (B, 2, D)
        emb = emb.reshape(x.size(0), -1)  # (B, 2D)
        logits = self.mlp(emb)       # (B, V)
        return logits

# -------------------------------------------------
# 3. GENERATION
# -------------------------------------------------
@torch.inference_mode()
def generate(model, seed_text, max_new_words=64, temperature=0.9, top_k=40, device=DEVICE):
    model.eval()

    if not hasattr(model, "word_to_id") or not hasattr(model, "id_to_word"):
        raise ValueError("Attach word_to_id and id_to_word to model before generation.")

    toks = seed_text.lower().split()
    if len(toks) < 2:
        raise ValueError("Seed text must contain at least 2 words for trigram generation.")

    word_to_id = model.word_to_id
    id_to_word = model.id_to_word

    ids = [word_to_id.get(w, 1) for w in toks]  # 1 = <unk>

    for _ in range(max_new_words):
        ctx = torch.tensor([ids[-2], ids[-1]], dtype=torch.long, device=device).unsqueeze(0)  # (1,2)
        logits = model(ctx) / temperature  # (1,V)

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits = torch.where(
                logits >= topk_vals[:, -1].unsqueeze(-1),
                logits,
                torch.tensor(-float("inf"), device=device)
            )

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)

        next_word = id_to_word[next_id] if 0 <= next_id < len(id_to_word) else "<unk>"
        if next_word == "<pad>":
            break

    words = [id_to_word[i] if 0 <= i < len(id_to_word) else "<unk>" for i in ids]
    return " ".join(words)

# -------------------------------------------------
# 4. TRAINING
# -------------------------------------------------
def main():
    paste_path = "singlekb.txt"
    if os.path.exists(paste_path):
        with open(paste_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        print(f"✓ Loaded {paste_path} ({len(text)} chars)")
    else:
        text = "NeuroSymbolic V18 fallback text corpus for testing with enough repeated words for trigram training."
        print("⚠ singlekb.txt not found - using fallback corpus")

    ds = TrigramDataset(text, vocab_size=8192, max_samples=None)
    if len(ds) == 0:
        print("Dataset is empty; cannot build DataLoader.")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = TrigramLLM(vocab_size=ds.vocab_size, d_model=128, hidden=256).to(DEVICE)
    model.word_to_id = ds.word_to_id
    model.id_to_word = ds.id_to_word

    opt = optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)   # (B,2)
            labels = batch["labels"].to(DEVICE)         # (B,)

            logits = model(input_ids)                   # (B,V)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f}")

    os.makedirs("output", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "word_to_id": ds.word_to_id,
        "id_to_word": ds.id_to_word,
        "vocab_size": ds.vocab_size
    }, "output/trigram_llm.pth")

    print(f"✓ Model ready: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print("🎉 COMPLETE - Trigram model saved to output/trigram_llm.pth")

    ckpt = torch.load("output/trigram_llm.pth", map_location=DEVICE)
    model = TrigramLLM(vocab_size=ckpt["vocab_size"], d_model=128, hidden=256).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.word_to_id = ckpt["word_to_id"]
    model.id_to_word = ckpt["id_to_word"]
    model.eval()

    while True:
        user_in = input("\nUSER: ").strip()
        if user_in.lower() in {"exit", "quit"}:
            break
        try:
            print(generate(model, user_in, max_new_words=64, temperature=0.9, top_k=40))
        except Exception as e:
            print(f"Generation error: {e}")

if __name__ == "__main__":
    main()
