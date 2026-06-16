import re
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

        self.data = [
            (ids[i:i+n], ids[i+n])
            for i in range(len(ids) - n)
        ]

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
        ce_loss = F.cross_entropy(logits, y, reduction='none').mean()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Small entropy bonus
        loss = ce_loss - 0.001 * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"  step {step} | loss {loss.item():.4f}")

    return total_loss / len(loader)


# =========================
# 5. LOGITS CONSTRAINT ENGINE (FIXED)
# =========================
class IsomorphismLogitsProcessor:
    def __init__(self, target_mass=0.5):
        # target_mass must be in [0, 1] because it's a probability mass
        self.target_mass = target_mass

    def __call__(self, logits):
        logits = logits.clone()

        probs = F.softmax(logits, dim=-1)
        half = logits.size(-1) // 2
        # Per-row mass over the first half of vocab
        mass = probs[:, :half].sum(dim=-1, keepdim=True)
        diff = mass - self.target_mass

        # Only adjust if we're significantly off
        if diff.abs().max() > 1e-3:
            logits[:, :half] -= diff * 1.5

        return logits


# =========================
# 6. BENCHMARK LOGITS PROCESSOR
# =========================
class BenchmarkLogitsProcessor:
    def __init__(self, token_loss_map, alpha=0.3, max_boost=5.0):
        self.token_loss_map = token_loss_map
        self.alpha = alpha
        self.max_boost = max_boost

    def update(self, token_loss_map):
        self.token_loss_map = token_loss_map

    def __call__(self, logits):
        if not self.token_loss_map:
            return logits
        logits = logits.clone()
        for vocab_id, loss in self.token_loss_map.items():
            boost = min(loss * self.alpha, self.max_boost)
            logits[:, vocab_id] += boost
        return logits


# =========================
# 7. STREAK TRACKER
# =========================
class StreakTracker:
    def __init__(self, threshold=0.5, alpha=0.08, max_streak=8):
        self.threshold = threshold
        self.alpha = alpha
        self.max_streak = max_streak
        self.streak = 0

    def reset(self):
        self.streak = 0

    def __call__(self, logits):
        logits = logits.clone()

        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1).values.item()

        if top_prob >= self.threshold:
            self.streak = min(self.streak + 1, self.max_streak)
        else:
            self.streak = 0

        if self.streak > 0:
            multiplier = 1.0 + self.streak * self.alpha
            logits = logits * multiplier

        return logits

    @property
    def status(self):
        bar = "█" * self.streak + "░" * (self.max_streak - self.streak)
        return f"streak [{bar}] {self.streak}/{self.max_streak}"


# =========================
# 8. TEST BENCHMARK
# =========================
class TestBenchmark:
    def __init__(self, qa_pairs, vocab, n):
        self.qa_pairs = qa_pairs
        self.vocab = vocab
        self.n = n

    @torch.no_grad()
    def evaluate(self, model, device, epoch):
        model.eval()
        print(f"\n── Benchmark after epoch {epoch} ──")

        all_token_losses = []
        token_loss_map = {}

        for q_str, a_str in self.qa_pairs:
            q_ids = self.vocab.encode(re.sub(r"[^\w\s]", "", q_str.lower()).split())
            if len(q_ids) < self.n:
                q_ids = [0] * (self.n - len(q_ids)) + q_ids

            a_ids = self.vocab.encode(re.sub(r"[^\w\s]", "", a_str.lower()).split())
            if not a_ids:
                continue

            context = q_ids[-self.n:]
            token_losses = []

            for target_id in a_ids:
                x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(x)
                target = torch.tensor([target_id], device=device)
                loss = F.cross_entropy(logits, target).item()
                token_losses.append(loss)

                token_loss_map[target_id] = token_loss_map.get(target_id, 0.0) + loss

                context = context[1:] + [target_id]

            cumsum = 0.0
            tokens = re.sub(r"[^\w\s]", "", a_str.lower()).split()
            print(f"  Q: {q_str!r}")
            for i, (tok, l) in enumerate(zip(tokens, token_losses)):
                cumsum += l
                known = tok in self.vocab.word2idx
                flag = "" if known else " [OOV]"
                print(f"    [{i+1:3d}] {tok:<20s} loss={l:6.3f}  cumsum={cumsum:8.3f}{flag}")

            all_token_losses.extend(token_losses)

        if all_token_losses:
            global_cumsum = 0.0
            print(f"\n  Global cumsum stream ({len(all_token_losses)} tokens):")
            for i, l in enumerate(all_token_losses):
                global_cumsum += l
                if (i + 1) % 10 == 0 or i == len(all_token_losses) - 1:
                    pct = (i + 1) / len(all_token_losses)
                    filled = int(pct * 30)
                    bar = "█" * filled + "░" * (30 - filled)
                    print(f"\r  [{bar}] {i+1}/{len(all_token_losses)} cumsum={global_cumsum:.3f}", end="", flush=True)
            print()
            avg = global_cumsum / len(all_token_losses)
            print(f"  avg token loss = {avg:.4f} | total cumsum = {global_cumsum:.3f}")

        print("────────────────────────────────")
        model.train()
        return all_token_losses, token_loss_map


# =========================
# 9. GENERATION
# =========================
@torch.no_grad()
def generate(model, vocab, prompt, device, iso_processor, bench_processor, streak_tracker, max_len=300):
    model.eval()
    streak_tracker.reset()

    words = re.sub(r"[^\w\s]", "", prompt.lower()).split()
    ids = vocab.encode(words)

    # Pad with zeros only if we have fewer than n tokens
    if len(ids) < model.n:
        ids = [0] * (model.n - len(ids)) + ids

    for step in range(max_len):
        context = ids[-model.n:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        logits = iso_processor(logits)
        logits = bench_processor(logits)
        logits = streak_tracker(logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)

    return vocab.decode(ids)


# =========================
# 10. MAIN
# =========================
def main(txt_path="corpus.txt", n=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NGramDataset(txt_path, n=n)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate)

    model = NGramModel(len(dataset.vocab), n=n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    qa_pairs = [
        ("what is the meaning of", "life is a journey full of purpose"),
        ("the quick brown fox",    "jumps over the lazy dog"),
        ("to be or not to",        "be that is the question"),
    ]

    benchmark     = TestBenchmark(qa_pairs, vocab=dataset.vocab, n=n)
    bench_processor  = BenchmarkLogitsProcessor(token_loss_map={}, alpha=0.1, max_boost=1.0)
    # Fixed: target_mass must be in [0, 1]
    iso_processor    = IsomorphismLogitsProcessor(target_mass=0.5)
    streak_tracker   = StreakTracker(threshold=0.5, alpha=0.08, max_streak=8)

    print("Training n-gram model...")
    for epoch in range(2):
        loss = train(model, loader, opt, device)
        print(f"\nEpoch {epoch} | train loss {loss:.4f}")

        _, token_loss_map = benchmark.evaluate(model, device, epoch)
        bench_processor.update(token_loss_map)

    print("\nGenerating...\n")
    while True:
        output = generate(
            model,
            dataset.vocab,
            input("USER: "),
            device,
            iso_processor,
            bench_processor,
            streak_tracker,
        )
        print(output)
        print()


if __name__ == "__main__":
    filepath = input("Filename: ")
    main(filepath, n=2)  # Use n=5 as default; change if you want n=2
