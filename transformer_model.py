import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import random
import numpy as np
import re
import pickle

max_new_tokens = 200
D = 2048

# ============================
# 1. SEEDING
# ============================
def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================
# 2. TRIGRAM TOKENIZER
# ============================

class TrigramTokenizer:
    def __init__(self, text):
        words = text.lower().split()
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(" ".join(words[i:i+3]))
        vocab = list(set(trigrams))
        random.shuffle(vocab)
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        words = text.lower().split()
        tokens = []
        for i in range(len(words) - 2):
            tri = " ".join(words[i:i+3])
            if tri in self.stoi:
                tokens.append(self.stoi[tri])
        if len(tokens) == 0:
            tokens = [random.randint(0, self.vocab_size - 1)]
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens):
        return " ".join([self.itos[int(t)] for t in tokens])
        
def save_tokenizer(tokenizer, path="tokenizer.pkl"):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
        
def load_tokenizer(path="tokenizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
        
# ============================
# 3. LOAD TEXT DATASET/MODEL
# ============================
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_model(model, path="model.pt"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="model.pt"):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model
    
# ============================
# 4. KERNEL MODULE
# ============================
class EfferenceKernelStack(nn.Module):
    def __init__(self, d_model=128, device="cpu", seed=42):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([8.0, 4.0, 4.0]))
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.omega_eff = nn.Parameter(torch.randn(3, d_model, generator=g, device=device))
        self.bias_eff = nn.Parameter(torch.randn(d_model, generator=g, device=device))

    def efference_features(self, rho, theta, sigma):
        B = rho.size(0)
        rho_eff = rho * torch.cos(theta)
        components = torch.stack([rho_eff, theta, sigma], dim=1)  # [B, 3]
        
        # Lambda dot iterations
        dot_prods = torch.zeros(B, 3, self.omega_eff.size(1), device=rho.device)
        for i in range(3):
            comp_i = components[:, i:i+1] * self.lambdas[i]
            dot_prods[:, i] = torch.sum(comp_i.unsqueeze(-1) * self.omega_eff[i], dim=1)
        
        proj = dot_prods.sum(dim=1) + self.bias_eff  # Stack → sum iterations
        return torch.exp(proj)
# ============================
# 5. TRANSFORMER BLOCK
# ============================
class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x

# ============================
# 6. FULL MODEL
# ============================
class KernelLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(D, d_model)
        self.kernel = EfferenceKernelStack(d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        # Add a DNN block after the Transformer
        self.dnn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        if T > D:
            idx = idx[:, -D:]
            T = D
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        emb = self.tok_emb(idx).mean(dim=1)  # [B, d_model]
        rho   = torch.sigmoid(emb[:, 0])
        theta = torch.sigmoid(emb[:, 1])
        sigma = torch.sigmoid(emb[:, 2])
        kernel_feat = self.kernel.efference_features(rho, theta, sigma)
        x = x + kernel_feat.unsqueeze(1)
        x = self.blocks(x)

        x = self.dnn(x) # <-- DNN added here
        x = self.ln(x) 
        return self.head(x)

# ============================
# 7. GENERATION
# ============================
@torch.no_grad()
def generate(model, idx, max_new_tokens=100, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx

# ============================
# 8. MAIN
# ============================
if __name__ == "__main__":

    set_seed(42)
    mode = input("train or load? (t/l): ")

    if mode == "t":
        text = load_text("singlekb.txt")

        tokenizer = TrigramTokenizer(text)
        data = tokenizer.encode(text).unsqueeze(0)

        model = KernelLLM(tokenizer.vocab_size)

        # ---- TRAINING LOOP (ADDED) ----
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("training...")
        for step in range(3):
            model.train()

            logits = model(data)

            T = min(logits.size(1), data.size(1) - 1)

            loss = F.cross_entropy(
                logits[:, :T, :].reshape(-1, tokenizer.vocab_size),
                data[:, 1:T+1].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f}")

        save_model(model)
        save_tokenizer(tokenizer)

    else:
        tokenizer = load_tokenizer()

        model = KernelLLM(tokenizer.vocab_size)
        model = load_model(model)

    # ---- INTERACTIVE LOOP ----
    while True:
        seed_text = input("USER: ")

        prompt = tokenizer.encode(seed_text).unsqueeze(0)

        generated = generate(model, prompt, max_new_tokens)

        print("\n--- GENERATED TEXT ---\n")
        print(tokenizer.decode(generated[0].tolist()))
