import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pickle
from pathlib import Path

max_new_tokens = 200
D = 2048


def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrigramTokenizer:
    def __init__(self, text):
        words = text.lower().split()
        self.trigrams = [" ".join(words[i:i+3]) for i in range(max(0, len(words)-2))]
        vocab = sorted(set(self.trigrams))
        random.shuffle(vocab)
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        words = text.lower().split()
        tokens = [
            self.stoi[" ".join(words[i:i+3])]
            for i in range(max(0, len(words)-2))
            if " ".join(words[i:i+3]) in self.stoi
        ]
        if len(tokens) == 0:
            tokens = [abs(hash(text)) % max(self.vocab_size, 1)] if self.vocab_size > 0 else [0]
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


class EfferenceKernelStack(nn.Module):
    def __init__(self, d_model=128, device="cpu", seed=42):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([8.0, 4.0, 4.0], device=device))
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.omega_eff = nn.Parameter(torch.randn(3, d_model, generator=g, device=device))
        self.bias_eff = nn.Parameter(torch.randn(d_model, generator=g, device=device))

    def efference_features(self, rho, theta, sigma):
        B = rho.size(0)
        rho_eff = rho * torch.cos(theta)
        components = torch.stack([rho_eff, theta, sigma], dim=1)
        dot_prods = torch.zeros(B, 3, self.omega_eff.size(1), device=rho.device)
        for i in range(3):
            comp_i = components[:, i:i+1] * self.lambdas[i]
            dot_prods[:, i] = torch.sum(comp_i.unsqueeze(-1) * self.omega_eff[i], dim=1)
        proj = dot_prods.sum(dim=1) + self.bias_eff
        return torch.exp(proj.clamp(max=20))


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
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class KernelLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(max(vocab_size, 1), d_model)
        self.pos_emb = nn.Embedding(D, d_model)
        self.kernel = EfferenceKernelStack(d_model=d_model, device=device)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.dnn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, max(vocab_size, 1))
        # State projection: maps d_model -> 4 state slices of d_model each
        self.state_proj = nn.Linear(d_model, 4 * d_model)

    def forward(self, idx):
        B, T = idx.shape
        if T > D:
            idx = idx[:, -D:]
            T = D
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1))
        x = tok + pos
        emb = tok.mean(dim=1)
        rho   = torch.sigmoid(emb[:, 0:1])
        theta = torch.sigmoid(emb[:, 1:2])
        sigma = torch.sigmoid(emb[:, 2:3])
        kernel_feat = self.kernel.efference_features(rho, theta, sigma)
        x = x + kernel_feat.unsqueeze(1)
        x = self.blocks(x)
        x = self.dnn(x)
        x = self.ln(x)
        logits = self.head(x)
        # Build flat state tensor: shape (B, 4*d_model) -> flatten to (4*d_model*B,)
        state_vec = self.state_proj(x.mean(dim=1))   # (B, 4*d_model)
        state = state_vec.reshape(-1)                 # flat: (B*4*d_model,)
        return logits, state


def _unpack_state(state, C):
    return state[:C], state[C:2*C], state[2*C:3*C], state[3*C:4*C]


def manhattan_distance(pred_logits, target_ids):
    V = pred_logits.size(-1)
    probs = F.softmax(pred_logits, dim=-1)
    targets = F.one_hot(target_ids.clamp(0, V - 1), V).float()
    return torch.abs(probs - targets).sum(dim=-1).mean()


def trigram_coverage(tokenizer, text):
    words = text.lower().split()
    tris = [" ".join(words[i:i+3]) for i in range(max(0, len(words)-2))]
    if not tris:
        return 0.0
    covered = sum(1 for tri in tris if tri in tokenizer.stoi)
    return covered / len(tris)


def make_batch(data, seq_len):
    if data.numel() <= seq_len + 1:
        return data[:-1].unsqueeze(0), data[1:].unsqueeze(0)
    start = random.randint(0, data.numel() - seq_len - 1)
    x = data[start:start+seq_len].unsqueeze(0)
    y = data[start+1:start+seq_len+1].unsqueeze(0)
    return x, y


@torch.no_grad()
def generate(model, idx, max_new_tokens=100, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -D:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


def train_model(text_path="singlekb.txt", model_path="model.pt", tok_path="tokenizer.pkl",
                steps=300, lr=1e-3, device="cpu", seq_len=64):
    text = load_text(text_path)
    tokenizer = TrigramTokenizer(text)
    data = tokenizer.encode(text).to(device)
    if data.numel() < 2:
        raise ValueError("Not enough tokens to train.")
    if tokenizer.vocab_size == 0:
        raise ValueError("Tokenizer vocabulary is empty.")

    model = KernelLLM(
        vocab_size=tokenizer.vocab_size, d_model=128,
        n_layers=4, n_heads=4, device=device
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    C = model.d_model  # state slice size
    print(f"training on {data.numel()} tokens | vocab={tokenizer.vocab_size} | coverage={trigram_coverage(tokenizer, text):.4f}")

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        x, y = make_batch(data, seq_len)
        x, y = x.to(device), y.to(device)

        logits, state = model(x)
        # Unpack state: loss_state used as auxiliary scalar, _1 as soft targets
        loss_state, _1, _2, _3 = _unpack_state(state, C)
        # CE loss on logits vs real targets
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        # Manhattan distance: logits vs state-derived soft signal (_1 mapped to token indices)
        state_target = (_1.detach().abs() * (tokenizer.vocab_size - 1)).long().clamp(0, tokenizer.vocab_size - 1)
        # Pad/trim state_target to match logits batch*seq dimension
        BT = logits.size(0) * logits.size(1)
        state_target = state_target[:BT] if state_target.numel() >= BT else state_target.repeat(BT // state_target.numel() + 1)[:BT]
        md = manhattan_distance(logits.reshape(-1, logits.size(-1)), state_target)
        total_loss = loss + 0.71 * md
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"step {step} | ce {loss.item():.6f} | manhattan {md.item():.6f} | total {total_loss.item():.6f}")

    save_model(model, model_path)
    save_tokenizer(tokenizer, tok_path)
    return model, tokenizer


def load_trained_model(model_path="model.pt", tok_path="tokenizer.pkl", device="cpu"):
    tokenizer = load_tokenizer(tok_path)
    model = KernelLLM(
        vocab_size=tokenizer.vocab_size, d_model=128,
        n_layers=4, n_heads=4, device=device
    ).to(device)
    model = load_model(model, model_path, device=device)
    return model, tokenizer


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    mode = input("train or load? (t/l): ").strip().lower()
    if mode == "t":
        model, tokenizer = train_model(
            text_path=input("Filename: "),
            model_path="model.pt",
            tok_path="tokenizer.pkl",
            steps=10,
            lr=1e-3,
            device=device,
            seq_len=128,
        )
    else:
        model, tokenizer = load_trained_model(
            model_path="model.pt", tok_path="tokenizer.pkl", device=device
        )
    while True:
        seed_text = input("USER: ").strip()
        if seed_text.lower() in {"quit", "exit", "q"}:
            break
        prompt = tokenizer.encode(seed_text).unsqueeze(0).to(device)
        generated = generate(model, prompt, max_new_tokens=max_new_tokens, temperature=1.0)
        print("\n--- GENERATED TEXT ---\n")
        print(tokenizer.decode(generated[0].detach().cpu().tolist()))
        print()
