import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. VOCAB
# =========================
class Vocab:
    def __init__(self, words):
        self.word2idx = {}
        self.idx2word = []

        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)

    def __len__(self):
        return len(self.idx2word)

    def encode(self, words):
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def decode(self, ids):
        return " ".join(self.idx2word[i] for i in ids)


# =========================
# 2. N-GRAM DATASET
# =========================
class NGramDataset(Dataset):
    def __init__(self, file_path, n=5):
        text = open(file_path, "r", encoding="utf-8").read().lower()
        words = text.split()

        self.vocab = Vocab(words)
        self.n = n

        ids = self.vocab.encode(words)

        if len(ids) < n + 1:
            raise ValueError("Corpus too small for given n")

        self.data = []
        for i in range(len(ids) - n):
            self.data.append((
                ids[i:i+n],     # context
                ids[i+n]        # target
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return x, y


# =========================
# 3. N-GRAM MODEL
# =========================
class NGramModel(nn.Module):
    def __init__(self, vocab_size, n=5, d=128):
        super().__init__()
        self.n = n

        self.emb = nn.Embedding(vocab_size, d)
        self.fc1 = nn.Linear(n * d, 256)
        self.fc2 = nn.Linear(256, vocab_size)

    def forward(self, x):
        # x: (B, n)
        e = self.emb(x)                 # (B, n, d)
        e = e.reshape(x.size(0), -1)    # (B, n*d)

        h = F.relu(self.fc1(e))
        return self.fc2(h)


# =========================
# 4. TRAIN LOOP
# =========================
def train(model, loader, opt, device):
    model.train()
    total_loss = 0

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        batch_size = y.size(0)
        vocab_size = logits.size(1)

        target = torch.zeros(
            batch_size,
            vocab_size,
            device=device
        )

        target.scatter_(1, y.unsqueeze(1), 0.5)
        loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

    return total_loss / len(loader)


# =========================
# 5. LOGITS CONSTRAINT ENGINE
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
# 6. GENERATION (USING ids[-n:])
# =========================
@torch.no_grad()
def generate(model, vocab, prompt, device, processor, max_len=300):
    model.eval()

    words = prompt.lower().split()
    ids = vocab.encode(words)

    if len(ids) < model.n:
        ids = [0] * model.n

    for _ in range(max_len):
        context = ids[-model.n:]   # 🔥 THIS is the corrected sliding window

        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        logits = processor(logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)
        

    return vocab.decode(ids)


# =========================
# 7. MAIN
# =========================
def main(txt_path="corpus.txt", n=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NGramDataset(txt_path, n=n)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate)

    model = NGramModel(len(dataset.vocab), n=n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("🚀 Training n-gram model...")
    for epoch in range(2):
        loss = train(model, loader, opt, device)
        print(f"\nEpoch {epoch} | loss {loss:.4f}\n")

    processor = IsomorphismLogitsProcessor(target_mass=0.00005)

    print("🧠 Generating...\n")
    while True:
        output = generate(
            model,
            dataset.vocab,
            input("USER: "),
            device,
            processor
        )

        print(output)
        print()

if __name__ == "__main__":
    main(input("Filename: "), n=3)
