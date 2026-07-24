import os
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
KB_LEN = -1
# -----------------------------------------------------------------------------
# 1. Dataset extracted directly from Image 1
# -----------------------------------------------------------------------------
with open(input("Filename: "), "r", encoding="utf-8") as f:
    dataset_rows = f.read().split(".")[:KB_LEN]

# -----------------------------------------------------------------------------
# 2. Model Definition & Helper Functions
# -----------------------------------------------------------------------------
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
    return text.lower().split()


def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab += [w for w, c in counts.items() if c >= min_freq and w not in vocab]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos


def encode(tokens, stoi):
    return [stoi.get(t, stoi["<unk>"]) for t in tokens]


def make_dataset(ids, seq_len=8):
    """
    Creates input (x) and target (y) sequences.
    Adjusted seq_len default to fit smaller example datasets cleanly.
    """
    xs, ys = [], []
    if len(ids) <= seq_len:
        # Pad sequence if total token count is smaller than seq_len
        pad_id = 0
        ids = ids + [pad_id] * (seq_len + 1 - len(ids))

    for i in range(len(ids) - seq_len):
        xs.append(ids[i : i + seq_len])
        ys.append(ids[i + 1 : i + seq_len + 1])

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def load_curve_prior(path=None, top_k=50):
    probs = np.array([
        0.0390, 0.0384, 0.0368, 0.0335, 0.0275, 0.0218, 0.0208, 0.0183, 0.0164, 0.0156,
        0.0138, 0.0134, 0.0112, 0.0088, 0.0080, 0.0071, 0.0068, 0.0065, 0.0058, 0.0057,
        0.0055, 0.0054, 0.0053, 0.0050, 0.0048, 0.0046, 0.0046, 0.0045, 0.0043, 0.0042,
        0.0042, 0.0042, 0.0041, 0.0041, 0.0040, 0.0039, 0.0039, 0.0038, 0.0038, 0.0037,
        0.0037, 0.0035, 0.0034, 0.0033, 0.0033, 0.00325, 0.00322, 0.00320, 0.00318, 0.00315
    ], dtype=np.float32)

    if top_k is not None:
        probs = probs[:top_k]
    probs = probs / probs.sum()
    return probs


def train_model(model, x, y, curve_prior=None, curve_weight=0.05, sensitivity=50, epochs=30, batch_size=16, lr=1e-3, device="cpu"):
    model.to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n = x.shape[0]
    steps = max(1, math.ceil(n / batch_size))

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0

        for s in range(steps):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            xb, yb = x[idx], y[idx]

            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=0)

            if curve_prior is not None and curve_weight > 0:
                vocab_slice = min(logits.size(-1), len(curve_prior))
                prior = torch.tensor(curve_prior[:vocab_slice], device=device)
                prior = prior / prior.sum().clamp_min(1e-12)

                last_logits = logits[:, -1, :vocab_slice]
                pred = F.softmax(last_logits, dim=-1).mean(dim=0)
                
                # Applying LogE (Natural Logarithm) to predictions and prior before MSE
                eps = 1e-12 # small epsilon to avoid log(0)
                log_pred = torch.exp(pred + eps)
                log_prior = torch.log(prior + eps)
                
                prior_loss = F.mse_loss(log_pred, log_prior)

                loss = loss + curve_weight * sensitivity * prior_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d} | Loss: {total/steps:.4f}")

    return model

@torch.no_grad()
def generate_text(
    model,
    stoi,
    itos,
    prime="the",
    length=80,
    temperature=1.0,
    sensitivity=1.0,
    device="cpu"
):
    model.eval()
    tokens = tokenize(prime)
    prompt_ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    ids = [stoi.get("<bos>", 1)] + prompt_ids

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    h = None
    logits, h = model(x, h)

    vocab_size = logits.size(-1)
    prompt_bias = prompt_bias_from_tokens(prompt_ids, vocab_size, device, sensitivity=sensitivity)

    out = tokens[:]
    cur = x[:, -1:]

    for _ in range(length):
        logits, h = model(cur, h)
        step_logits = logits[:, -1, :] / max(1e-6, temperature)
        step_logits = step_logits + prompt_bias.unsqueeze(0)

        probs = F.softmax(step_logits, dim=-1).squeeze(0)

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

def prompt_bias_from_tokens(prompt_ids, vocab_size, device, sensitivity=1.0):
    bias = torch.zeros(vocab_size, device=device)
    if len(prompt_ids) == 0:
        return bias
    counts = torch.bincount(torch.tensor(prompt_ids, device=device), minlength=vocab_size).float()
    counts = counts / counts.sum().clamp_min(1.0)
    bias = sensitivity * counts
    return bias


# -----------------------------------------------------------------------------
# 3. Execution Pipeline
# -----------------------------------------------------------------------------
def run_dataset_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Combine sentences into sequence tokens
    full_text = ".".join(dataset_rows)
    tokens = tokenize(full_text)
    
    vocab, stoi, itos = build_vocab(tokens, min_freq=1)
    ids = encode(["<bos>"] + tokens + ["<eos>"], stoi)

    seq_len = 8  # Adjusted sequence length for chunked sentences
    x, y = make_dataset(ids, seq_len=seq_len)

    curve_prior = load_curve_prior(None, top_k=min(50, len(vocab)))

    config = {"emb_dim": 64, "hidden": 128, "layers": 2, "seq_len": seq_len}
    model = CurvePriorNet(
        vocab_size=len(vocab),
        emb_dim=config["emb_dim"],
        hidden=config["hidden"],
        layers=config["layers"]
    )

    print("Training model on dataset...")
    model = train_model(
        model,
        x,
        y,
        curve_prior=curve_prior,
        curve_weight=0.05,
        sensitivity=10.0,
        epochs=10,
        batch_size=16,
        lr=1e-3,
        device=device
    )

    print("\nGenerating sample output from trained prior net:")
    while True:    
        sample = generate_text(
            model,
            stoi,
            itos,
            prime=input("USER: "),
            length=600,
            temperature=0.8,
            device=device
        )
        print(f"\nGenerated Result: '{sample}'")


if __name__ == "__main__":
    run_dataset_pipeline()
