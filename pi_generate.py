import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. VOCAB BUILDER
# =========================
class Vocab:
    def __init__(self, tokens):
        self.word2idx = {}
        self.idx2word = []

        for w in tokens:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)

    def __len__(self):
        return len(self.idx2word)

    def encode(self, words):
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def decode(self, idxs):
        return " ".join(self.idx2word[i] for i in idxs)


# =========================
# 2. DATASET (TRIGRAMS)
# =========================
class TrigramDataset(Dataset):
    def __init__(self, path):
        text = open(path, "r", encoding="utf-8").read().lower()
        words = text.split()

        self.vocab = Vocab(words)
        ids = self.vocab.encode(words)

        if len(ids) < 3:
            raise ValueError("Corpus too small")

        self.data = []
        for i in range(len(ids) - 2):
            self.data.append((ids[i], ids[i+1], ids[i+2]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        w1, w2, w3 = self.data[idx]
        return (
            torch.tensor([w1, w2], dtype=torch.long),
            torch.tensor(w3, dtype=torch.long)
        )


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return x, y


# =========================
# 3. TRIGRAM MODEL
# =========================
class TrigramModel(nn.Module):
    def __init__(self, vocab_size, d=128):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d)
        self.fc1 = nn.Linear(d * 2, 256)
        self.fc2 = nn.Linear(256, vocab_size)

    def forward(self, x):
        e = self.emb(x)              # (B, 2, d)
        e = e.view(x.size(0), -1)    # (B, 2d)

        h = F.relu(self.fc1(e))
        logits = self.fc2(h)

        return logits


# =========================
# 4. TRAIN LOOP
# =========================
def train(model, loader, opt, device):
    model.train()
    total = 0

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

    return total / len(loader)


# =========================
# 5. LOGITS CONSTRAINT ENGINE (TRIGRAM VERSION)
# =========================
class IsomorphismLogitsProcessor:
    def __init__(self, target_mass=0.5):
        self.target_mass = target_mass

    def __call__(self, logits):
        logits = logits.clone()
        probs = F.softmax(logits, dim=-1)

        half = logits.size(-1) // 2
        mass = probs[:, :half].sum()

        diff = float(mass - self.target_mass)

        if abs(diff) > 1e-3:
            logits[:, :half] -= diff * 1.5

        return logits


# =========================
# 6. GENERATION (TRIGRAM SAMPLING)
# =========================
@torch.no_grad()
def generate(model, vocab, prompt, device, processor, max_len=200):
    model.eval()

    words = prompt.lower().split()

    # ensure at least 2 words
    if len(words) < 2:
        words = ["<s>", words[0] if words else "<s>"]

    ids = vocab.encode(words)

    if len(ids) < 2:
        ids = [0, 0]

    for _ in range(max_len):
        x = torch.tensor(ids[-2:], dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        logits = processor(logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)

    return vocab.decode(ids)


# =========================
# 7. MAIN
# =========================
def main(txt_path="corpus.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrigramDataset(txt_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate)

    model = TrigramModel(len(dataset.vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("🚀 Training trigram model...")
    for epoch in range(2):
        loss = train(model, loader, opt, device)
        print(f"\nEpoch {epoch} | loss {loss:.4f}\n")

    processor = IsomorphismLogitsProcessor(target_mass=0.01)

    print("🧠 Generating...\n")
    while True:
        out = generate(
            model,
            dataset.vocab,
            input("USER: "),
            device,
            processor
        )

        print(out)


if __name__ == "__main__":
    main(input("Filename: "))
