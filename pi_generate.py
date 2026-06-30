import re
import math
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import numpy as np
KB_LEN = 9999
# ============================================================
# POLYMORPHIC ROUTER: Arithmetic Logic Layer
# ============================================================
class PreprocessingRouter:
    def __init__(self):
        # Strategies define the specific arithmetic formula for context modulation
        self.arithmetic_strategies = {
            "sum_mod": lambda x: int(np.sum(x) % 10),
            "max_mod": lambda x: int(np.max(x) % 15),
            "mean_mod": lambda x: int(np.mean(x) % 5)
        }

    def select_strategy(self, features):
        # Choose the arithmetic logic based on entropy/wc complexity
        if features['entropy'] > 3.0: return self.arithmetic_strategies["max_mod"]
        if features['wc'] < 5: return self.arithmetic_strategies["mean_mod"]
        return self.arithmetic_strategies["sum_mod"]

# ============================================================
# UTILS & PREPROCESSING
# ============================================================
def tokenize(text): return re.findall(r"\{[^}]+\}|\w+|[^\w\s]", text.lower())

def get_features(text):
    words = re.findall(r"\w+", text)
    wc = len(words)
    entropy = -sum((c/len(text))*math.log2(c/len(text)) for c in Counter(text).values() if c > 0) if len(text) > 0 else 0
    return {"wc": wc, "entropy": entropy, "raw_vec": np.array([wc, entropy, len(text)], dtype=np.float32)}

# ============================================================
# DATASET: Polymorphic Context Window Modulation
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, stoi):
        self.router = PreprocessingRouter()
        self.samples = []
        for t in texts:
            feats = get_features(t)
            # Strategy selection returns the arithmetic function to apply
            arithmetic_func = self.router.select_strategy(feats)
            
            ids = [2] + [stoi.get(tok, 1) for tok in tokenize(t)] + [3]
            for i in range(1, len(ids)):
                # Apply the polymorphic arithmetic directly to modulate context
                context_mod = arithmetic_func(feats['raw_vec'])
                start_idx = max(0, i - (64 - context_mod))
                x = torch.tensor(ids[start_idx:i], dtype=torch.long)
                y = torch.tensor(ids[i], dtype=torch.long)
                self.samples.append((x, y))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    x_padded = torch.stack([torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in xs])
    return x_padded, torch.stack(ys)

# ============================================================
# MODEL
# ============================================================
class GRUModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.gru = nn.GRU(256, 512, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)
    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.fc(h.squeeze(0))

# ============================================================
# TRAINING LOOP & GENERATION
# ============================================================
filename = input("Filename: ")
with open(filename, "r", encoding="utf-8") as f:
    text_data = [x.strip() for x in f.read().split(".") if x.strip()][:KB_LEN]

vocab = sorted(list(set(tok for t in text_data for tok in tokenize(t))))
stoi = {t: i+4 for i, t in enumerate(vocab)}
stoi.update({"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3})
itos = {i: t for t, i in stoi.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TextDataset(text_data, stoi)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
model = GRUModel(len(stoi)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(5):
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

def generate(prompt, max_tokens=50):
    tokens = [2] + [stoi.get(t, 1) for t in tokenize(prompt)]
    for _ in range(max_tokens):
        x = torch.tensor([tokens[-64:]], device=device)
        next_tok = torch.multinomial(torch.softmax(model(x), dim=-1), 1).item()
        tokens.append(next_tok)
    return " ".join([itos[i] for i in tokens if i > 3])

while True:
    p = input("\nPrompt: ")
    if not p: break
    print(generate(p))
