import torch
import torch.nn as nn
import numpy as np
import re
import math
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ============================================================
# MNEMOTIC MEMORY & ROUTER
# ============================================================
class PreprocessingRouter:
    def __init__(self):
        self.arithmetic_strategies = {
            "sum_mod": lambda x: int(np.sum(x) % 10),
            "max_mod": lambda x: int(np.max(x) % 15),
            "mean_mod": lambda x: int(np.mean(x) % 5)
        }
    def select_strategy(self, features):
        if features['entropy'] > 3.0: return self.arithmetic_strategies["max_mod"]
        if features['wc'] < 5: return self.arithmetic_strategies["mean_mod"]
        return self.arithmetic_strategies["sum_mod"]

class MnemoticStore:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.keys, self.values = [], []

    def encode(self, text, stoi):
        self.model.eval()
        ids = torch.tensor([[stoi.get(t, 1) for t in tokenize(text)]]).to(self.device)
        with torch.no_grad():
            emb = self.model.emb(ids)
            _, h = self.model.gru(emb)
            return h.squeeze(0).cpu().numpy().flatten()

    def add_to_memory(self, text, stoi):
        self.keys.append(self.encode(text, stoi))
        self.values.append(text)

    def retrieve(self, prompt, stoi):
        prompt_vec = self.encode(prompt, stoi)
        scores = [np.dot(prompt_vec, k) / (np.linalg.norm(prompt_vec) * np.linalg.norm(k) + 1e-8) for k in self.keys]
        return self.values[np.argmax(scores)]

# ============================================================
# UTILS & MODEL
# ============================================================
def tokenize(text): return text.lower().split()

def get_features(text):
    words = re.findall(r"\w+", text)
    wc = len(words)
    entropy = -sum((c/len(text))*math.log2(c/len(text)) for c in Counter(text).values() if c > 0) if len(text) > 0 else 0
    return {"wc": wc, "entropy": entropy, "raw_vec": np.array([wc, entropy, len(text)], dtype=np.float32)}

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
# MAIN EXECUTION
# ============================================================
filename = input("Filename: ")
with open(filename, "r", encoding="utf-8") as f:
    text_data = [x.strip() + "." for x in f.read().split(".") if x.strip()]

vocab = sorted(list(set(tok for t in text_data for tok in tokenize(t))))
stoi = {t: i+4 for i, t in enumerate(vocab)}
stoi.update({"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3})
itos = {i: t for t, i in stoi.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(len(stoi)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 1. Training (Polymorphic Modulation)
print("Training polymorphic model...")
# (Assuming dataset/loader implementation from previous steps)
# [Training loop proceeds here...]

# 2. Memory Population
mnemotics = MnemoticStore(model, device)
for t in text_data: mnemotics.add_to_memory(t, stoi)

def generate(prompt, max_tokens=20):
    # Retrieve closest mnemotic vector
    ctx = mnemotics.retrieve(prompt, stoi)
    tokens = [2] + [stoi.get(t, 1) for t in tokenize(ctx + " " + prompt)]
    for _ in range(max_tokens):
        x = torch.tensor([tokens[-64:]], device=device)
        logits = model(x)
        next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        tokens.append(next_tok)
    return " ".join([itos.get(i, "<unk>") for i in tokens if i > 3])


with open("questions.conf", "r", encoding="UTF-8") as file:
    content = file.readlines()
    prompt = input("USER: ")
for question in content:
    print(question, generate(question).split(".")[0] + ".")
    print()
