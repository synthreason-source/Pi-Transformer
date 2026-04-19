import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import pickle
from typing import List

# ============================
# CONFIG
# ============================
max_new_tokens = 200
D = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# SEED
# ============================
def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================
# TOKENIZER
# ============================
class TrigramTokenizer:
    def __init__(self, text):
        words = text.lower().split()
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        vocab = list(set(trigrams))
        random.shuffle(vocab)
        self.stoi = {t:i for i,t in enumerate(vocab)}
        self.itos = {i:t for t,i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        words = text.lower().split()
        tokens = []
        for i in range(len(words)-2):
            tri = " ".join(words[i:i+3])
            if tri in self.stoi:
                tokens.append(self.stoi[tri])
        if not tokens:
            tokens = [hash(text) % self.vocab_size]
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens):
        return " ".join([self.itos[int(t)] for t in tokens])

def save_tokenizer(tok):
    with open("tokenizer.pkl","wb") as f:
        pickle.dump(tok,f)

def load_tokenizer():
    with open("tokenizer.pkl","rb") as f:
        return pickle.load(f)

# ============================
# MEMORY SYSTEM
# ============================
class SentenceVector:
    def __init__(self, tokens, emb):
        self.tokens = tokens
        self.emb = emb

class IsomorphicSyntaxStacker:
    def __init__(self, top_k=3, max_stored=64):
        self.top_k = top_k
        self.max_stored = max_stored
        self.store: List[SentenceVector] = []

    def add(self, tokens, emb):
        self.store.append(SentenceVector(tokens, emb.detach()))
        if len(self.store) > self.max_stored:
            self.store.pop(0)

    def syntax_echo_bonus(self, current_emb):
        if not self.store:
            return torch.zeros_like(current_emb)

        sims = []
        for sv in self.store:
            L = min(current_emb.size(0), sv.emb.size(0))
            sim = F.cosine_similarity(current_emb[:L], sv.emb[:L], dim=-1).mean()
            sims.append(sim)

        sims = torch.stack(sims)
        topk = torch.topk(sims, min(self.top_k, len(self.store)))

        bonus = torch.zeros_like(current_emb)
        for i in range(len(topk.values)):
            sv = self.store[topk.indices[i]]
            L = min(current_emb.size(0), sv.emb.size(0))
            bonus[:L] += topk.values[i] * sv.emb[:L]

        return bonus * 0.3

# ============================
# MODEL COMPONENTS
# ============================
class FineAlterableMonad(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pre = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, 2*d_model)
        self.down = nn.Linear(2*d_model, d_model)
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, strength=1.0):
        y = self.pre(x)
        y = F.gelu(self.up(y))
        y = self.down(y)
        g = torch.sigmoid(self.gate).view(1,1,-1)
        return x + strength * g * y

class CardanGrilleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(D))

    def forward(self, x):
        T = x.size(1)
        mask = torch.sigmoid(self.logits[:T])
        return x * mask.unsqueeze(0).unsqueeze(-1)

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out,_ = self.attn(x,x,x,attn_mask=mask)
        x = self.ln1(x + -attn_out)
        x = self.ln2(x + self.ff(x))
        return x

# ============================
# FULL MODEL
# ============================
class KernelLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, 128)
        self.pos_emb = nn.Embedding(D, 128)

        self.syntax = IsomorphicSyntaxStacker()

        self.grille = CardanGrilleLayer()
        self.monad1 = FineAlterableMonad(128)
        self.blocks = nn.Sequential(*[Block(128,4) for _ in range(4)])
        self.monad2 = FineAlterableMonad(128)

        self.ln = nn.LayerNorm(128)
        self.head = nn.Linear(128, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        if T > D:
            idx = idx[:, -D:]
            T = D

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # === MEMORY SYSTEM ===
        tokens = idx[0].tolist()
        self.syntax.add(tokens, x[0])
        bonus = self.syntax.syntax_echo_bonus(x[0])
        x = x + bonus.unsqueeze(0)

        # === MODEL CORE ===
        x = self.grille(x)
        x = self.monad1(x, 0.5)
        x = self.blocks(x)
        x = self.monad2(x, 0.8)
        x = self.grille(x)

        x = self.ln(x)
        return self.head(x)

# ============================
# GENERATION
# ============================
@torch.no_grad()
def generate(model, idx):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx)
        probs = F.softmax(logits[:,-1,:], dim=-1)
        next_token = torch.multinomial(probs,1)
        idx = torch.cat([idx,next_token], dim=1)
    return idx

# ============================
# MAIN
# ============================
if __name__ == "__main__":

    set_seed(42)
    mode = input("train or load? (t/l): ")

    if mode == "t":
        text = open("singlekb.txt","r",encoding="utf-8").read()

        tokenizer = TrigramTokenizer(text)
        data = tokenizer.encode(text).unsqueeze(0)

        model = KernelLLM(tokenizer.vocab_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("training...")

        for step in range(20):
            model.train()

            # random chunk training
            start = random.randint(0, data.size(1)-512-1)
            x = data[:, start:start+512].to(DEVICE)
            y = data[:, start+1:start+513].to(DEVICE)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                y.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"step {step} loss {loss.item():.4f}")

        torch.save(model.state_dict(),"model.pt")
        save_tokenizer(tokenizer)

    else:
        tokenizer = load_tokenizer()
        model = KernelLLM(tokenizer.vocab_size).to(DEVICE)
        model.load_state_dict(torch.load("model.pt", map_location=DEVICE))

    while True:
        seed = input("USER: ")
        idx = tokenizer.encode(seed).unsqueeze(0).to(DEVICE)

        out = generate(model, idx)
        print("\n--- GENERATED ---\n")
        print(tokenizer.decode(out[0].tolist()))
