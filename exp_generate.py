import os
import math
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import plotly.graph_objects as go


class CurvePriorNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.head(out)
        return logits, h


def tokenize(text):
    return re.findall(r"[A-Za-z']+|[.,!?;:]", text.lower())


def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab += [w for w, c in counts.items() if c >= min_freq and w not in vocab]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos


def encode(tokens, stoi):
    return [stoi.get(t, stoi["<unk>"]) for t in tokens]


def make_dataset(ids, seq_len=32):
    xs, ys = [], []
    for i in range(len(ids) - seq_len):
        xs.append(ids[i:i + seq_len])
        ys.append(ids[i + 1:i + seq_len + 1])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def load_curve_prior(path=None, top_k=50):
    if path is None:
        probs = np.array([
            0.0390, 0.0384, 0.0368, 0.0335, 0.0275,
            0.0218, 0.0208, 0.0183, 0.0164, 0.0156,
            0.0138, 0.0134, 0.0112, 0.0088, 0.0080,
            0.0071, 0.0068, 0.0065, 0.0058, 0.0057,
            0.0055, 0.0054, 0.0053, 0.0050, 0.0048,
            0.0046, 0.0046, 0.0045, 0.0043, 0.0042,
            0.0042, 0.0042, 0.0041, 0.0041, 0.0040,
            0.0039, 0.0039, 0.0038, 0.0038, 0.0037,
            0.0037, 0.0035, 0.0034, 0.0033, 0.0033,
            0.00325, 0.00322, 0.00320, 0.00318, 0.00315
        ], dtype=np.float32)
    else:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        probs = data[:, 1] if data.shape[1] > 1 else data[:, 0]
        probs = probs.astype(np.float32)

    if top_k is not None:
        probs = probs[:top_k]
    probs = probs / probs.sum()
    return probs


def train_model(model, x, y, curve_prior=None, curve_weight=0.0, epochs=20, batch_size=64, lr=3e-4, device="cpu"):
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n = x.shape[0]
    steps = max(1, math.ceil(n / batch_size))

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            xb = x[idx]
            yb = y[idx]
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            if curve_prior is not None and curve_weight > 0:
                vocab_slice = min(logits.size(-1), len(curve_prior))
                prior = torch.tensor(curve_prior[:vocab_slice], device=device)
                prior = prior / prior.sum()
                logp = F.log_softmax(logits[:, -1, :vocab_slice], dim=-1).mean(dim=0)
                prior_loss = F.kl_div(logp, prior, reduction="batchmean")
                loss = loss + curve_weight * prior_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        print(f"epoch {epoch+1:03d} loss {total/steps:.4f}")

    return model


@torch.no_grad()
def generate_text(model, stoi, itos, prime="the", length=80, temperature=1.0, device="cpu"):
    model.eval()
    tokens = tokenize(prime)
    ids = [stoi.get("<bos>", 1)] + [stoi.get(t, stoi["<unk>"]) for t in tokens]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    h = None

    logits, h = model(x, h)
    out = tokens[:]

    cur = x[:, -1:]
    for _ in range(length):
        logits, h = model(cur, h)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        next_tok = itos[next_id]
        if next_tok == "<eos>":
            break
        out.append(next_tok)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)

    text = []
    for t in out:
        if t in ".,!?;:":
            if text:
                text[-1] = text[-1] + t
            else:
                text.append(t)
        else:
            text.append(t)
    return " ".join(text)


def save_curve_png(probs, out_path="output/curve.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(probs)), y=probs, mode="lines+markers", name="curve"))
    fig.update_layout(
        title="Probability curve prior",
        xaxis_title="Rank",
        yaxis_title="Probability",
        template="plotly_white",
        width=1000,
        height=600,
    )
    fig.write_image(out_path)
    return out_path


def main():
    os.makedirs("output", exist_ok=True)

    with open(input('Filename: '), 'r', encoding='utf8') as f:
        text = f.read()

    tokens = tokenize(text)
    vocab, stoi, itos = build_vocab(tokens, min_freq=1)
    ids = encode(["<bos>"] + tokens + ["<eos>"], stoi)

    seq_len = 32
    x, y = make_dataset(ids, seq_len=seq_len)

    curve_prior = load_curve_prior(None, top_k=min(50, len(vocab)))
    save_curve_png(curve_prior, "output/curve.png")

    model = CurvePriorNet(vocab_size=len(vocab), emb_dim=64, hidden=128, layers=2)
    model = train_model(
        model,
        x,
        y,
        curve_prior=curve_prior,
        curve_weight=0.05,
        epochs=20,
        batch_size=64,
        lr=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    while True:
        sample = generate_text(
            model,
            stoi,
            itos,
            prime=input("USER: "),
            length=100,
            temperature=0.9,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        with open("output/sample.txt", "w", encoding="utf-8") as f:
            f.write(sample)

        print(sample)


if __name__ == "__main__":
    main()
