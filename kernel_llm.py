import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
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
            tokens = [hash(text) % self.vocab_size]

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens):
        out = []
        for t in tokens:
            t = int(t)
            if t in self.itos:
                out.append(self.itos[t])
        return " ".join(out)


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


def load_model(model, path="model.pt", device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {path}")
    return model


# ============================
# 4. KERNEL MODULE
# ============================
class EfferenceKernelStack(nn.Module):
    def __init__(self, d_model=128, device="cpu", seed=42):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([8.0, 4.0, 4.0], device=device))

        g = torch.Generator(device=device)
        g.manual_seed(seed)

        self.omega_eff = nn.Parameter(
            torch.randn(3, d_model, generator=g, device=device)
        )
        self.bias_eff = nn.Parameter(
            torch.randn(d_model, generator=g, device=device)
        )

    def efference_features(self, rho, theta, sigma):
        B = rho.size(0)
        rho_eff = rho * torch.cos(theta)
        components = torch.stack([rho_eff, theta, sigma], dim=1)

        dot_prods = torch.zeros(B, 3, self.omega_eff.size(1), device=rho.device)

        for i in range(3):
            comp_i = components[:, i:i+1] * self.lambdas[i]
            dot_prods[:, i] = torch.sum(
                comp_i.unsqueeze(-1) * self.omega_eff[i], dim=1
            )

        proj = dot_prods.sum(dim=1) + self.bias_eff
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
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + attn_out)

        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


# ============================
# 6. FULL MODEL
# ============================
class KernelLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, device="cpu"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(D, d_model)
        self.kernel = EfferenceKernelStack(d_model=d_model, device=device)
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)

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
        tok = self.tok_emb(idx)
        x = tok + self.pos_emb(pos)

        emb = tok.mean(dim=1)
        rho = torch.sigmoid(emb[:, 0])
        theta = torch.sigmoid(emb[:, 1])
        sigma = torch.sigmoid(emb[:, 2])

        kernel_feat = self.kernel.efference_features(rho, theta, sigma)
        x = x + kernel_feat.unsqueeze(1)

        x = self.blocks(x)
        x = self.dnn(x)
        x = self.ln(x)

        return self.head(x)  # [B, T, V]

    def _unpack_state(self, state, C):
        return state[:C], state[C:2*C], state[2*C:3*C], state[3*C:4*C]

    def loss(self, features: torch.Tensor, gold_indices: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(features, dim=-1)          # [B, V]
        C = features.size(-1)
        targets = F.one_hot(gold_indices.long().clamp(0, C - 1), C).float()  # [B, V]
        return F.binary_cross_entropy(probs, targets)


# ============================
# 8. GENERATION
# ============================
@torch.no_grad()
def generate(model, idx, max_new_tokens=100, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -D:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


# ============================
# 8. TRAIN
# ============================
def train_model(
    text_path="singlekb.txt",
    model_path="model.pt",
    tok_path="tokenizer.pkl",
    steps=300,
    lr=1e-3,
    device="cpu"
):
    text = load_text(text_path)
    tokenizer = TrigramTokenizer(text)
    data = tokenizer.encode(text).unsqueeze(0).to(device)

    if data.numel() < 2:
        raise ValueError("Not enough tokens to train.")

    model = KernelLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("training...")

    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        logits = model(data)

        T = min(logits.size(1), data.size(1)) - 1
        V = tokenizer.vocab_size
        t = min(step, T - 1)
        probs = F.softmax(logits[0, t, :], dim=-1)              # [V]
        logp, c_rho, c_theta, c_sigma = model._unpack_state(probs, step+1)
        gold = data[0, t + 1].unsqueeze(0)                      # [1]
        loss = model.loss(c_rho.unsqueeze(0), gold)             # [1, V//4], [1]

        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"step {step} | loss {loss.item():.6f}")

    save_model(model, model_path)
    save_tokenizer(tokenizer, tok_path)
    return model, tokenizer


# ============================
# 9. LOAD
# ============================
def load_trained_model(model_path="model.pt", tok_path="tokenizer.pkl", device="cpu"):
    tokenizer = load_tokenizer(tok_path)

    model = KernelLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        device=device
    ).to(device)

    model = load_model(model, model_path, device=device)
    return model, tokenizer


# ============================
# 10. MAIN
# ============================
if __name__ == "__main__":
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    mode = input("train or load? (t/l): ").strip().lower()

    if mode == "t":
        model, tokenizer = train_model(
            text_path="singlekb.txt",
            model_path="model.pt",
            tok_path="tokenizer.pkl",
            steps=10,
            lr=1e-3,
            device=device
        )
    else:
        model, tokenizer = load_trained_model(
            model_path="model.pt",
            tok_path="tokenizer.pkl",
            device=device
        )

    while True:
        seed_text = input("USER: ").strip()

        if seed_text.lower() in {"quit", "exit", "q"}:
            break

        prompt = tokenizer.encode(seed_text).unsqueeze(0).to(device)

        generated = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
        )

        print("\n--- GENERATED TEXT ---\n")
        print(tokenizer.decode(generated[0].detach().cpu().tolist()))
        print()
