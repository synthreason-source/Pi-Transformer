import os
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

KB_LEN = -1
SEQ_LEN = 16
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
MARKOV_ORDER = 2
NGRAM_BLOCK = 3

# -----------------------------------------------------------------------------
# 1. Dataset extraction, uniqueness filtering, and combinatorial pruning
# -----------------------------------------------------------------------------
def normalize_row(row):
    row = row.strip().lower()
    row = re.sub(r"\s+", " ", row)
    return row

def load_unique_rows(path, kb_len=-1):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw_rows = [r.strip() for r in raw.split(".")]

    seen = set()
    unique_rows = []
    for r in raw_rows:
        if not r:
            continue
        key = normalize_row(r)
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    if kb_len != -1:
        unique_rows = unique_rows[:kb_len]

    return unique_rows

def tokenize(text):
    return text.lower().split()

def build_combinatorial_token_stream(rows, n=3):
    tokens = []
    seen = set()
    for row in rows:
        toks = tokenize(row)
        if len(toks) == 0:
            continue
        for i in range(max(1, len(toks) - n + 1)):
            ng = tuple(toks[i:i+n]) if len(toks) >= n else tuple(toks)
            if ng in seen:
                continue
            seen.add(ng)
            if len(toks) <= n:
                tokens.extend(list(toks[-1]))
            else:
                tokens.extend(toks)
    return tokens

# -----------------------------------------------------------------------------
# 2. Model Definition & Helper Functions
# -----------------------------------------------------------------------------
class CurvePriorNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, layers=2, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.head(out)
        return logits, h

def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    vocab = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    vocab += [w for w, c in counts.items() if c >= min_freq and w not in vocab]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos

def encode(tokens, stoi):
    return [stoi.get(t, stoi[UNK_TOKEN]) for t in tokens]

def make_dataset(ids, seq_len=8, pad_id=0):
    xs, ys = [], []
    if len(ids) <= seq_len:
        ids = ids + [pad_id] * (seq_len + 1 - len(ids))
    for i in range(len(ids) - seq_len):
        xs.append(ids[i:i+seq_len])
        ys.append(ids[i+1:i+seq_len+1])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

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
    return probs / probs.sum()

def prompt_bias_from_tokens(prompt_ids, vocab_size, device, sensitivity=1.0):
    bias = torch.zeros(vocab_size, device=device)
    if len(prompt_ids) == 0:
        return bias
    counts = torch.bincount(torch.tensor(prompt_ids, device=device), minlength=vocab_size).float()
    counts = counts / counts.sum().clamp_min(1.0)
    return sensitivity * counts

def to_nilpotent_ideal(t):
    *batch, k = t.shape
    N = torch.zeros(*batch, k, k, dtype=t.dtype, device=t.device)
    if k > 1:
        i0 = torch.arange(k - 1, device=t.device)
        i1 = torch.arange(1, k, device=t.device)
        N[..., i0, i1] = t[..., :-1]
    return N

def from_nilpotent_ideal(N):
    return N.sum(dim=-1)

def probs_to_nilpotent_ideal(probs):
    if isinstance(probs, torch.Tensor):
        return to_nilpotent_ideal(probs)
    return to_nilpotent_ideal(torch.from_numpy(np.asarray(probs))).numpy()

def nilpotent_ideal_to_probs(N, eps=1e-12):
    if isinstance(N, torch.Tensor):
        vec = from_nilpotent_ideal(N)
        return vec / vec.sum().clamp_min(eps)
    vec = from_nilpotent_ideal(torch.from_numpy(np.asarray(N))).numpy()
    s = vec.sum()
    return np.full_like(vec, 0.1 / len(vec)) if s < eps else vec / s

# -----------------------------------------------------------------------------
# 3. Markov chain
# -----------------------------------------------------------------------------
def build_markov_chain(token_ids, order=2):
    chain = defaultdict(Counter)
    if len(token_ids) <= order:
        return chain
    for i in range(len(token_ids) - order):
        state = tuple(token_ids[i:i+order])
        nxt = token_ids[i + order]
        chain[state][nxt] += 1
    return chain

def markov_prior_vector(chain, state, vocab_size, device="cpu", eps=1e-12):
    vec = torch.zeros(vocab_size, device=device)
    counter = chain.get(tuple(state), None)
    if not counter:
        return vec
    for k, v in counter.items():
        if 0 <= k < vocab_size:
            vec[k] = float(v)
    return vec / vec.sum().clamp_min(eps)

# -----------------------------------------------------------------------------
# 4. Training
# -----------------------------------------------------------------------------
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
            idx = perm[s * batch_size:(s + 1) * batch_size]
            xb, yb = x[idx], y[idx]
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=0)
            if curve_prior is not None and curve_weight > 0:
                vocab_slice = min(logits.size(-1), len(curve_prior))
                prior = torch.tensor(curve_prior[:vocab_slice], device=device)
                prior = prior / prior.sum().clamp_min(1e-12)
                last_logits = logits[:, -1, :vocab_slice]
                pred = F.softmax(last_logits, dim=-1).mean(dim=0)
                loss = loss + curve_weight * sensitivity * F.mse_loss(pred, prior)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d} | Loss: {total/steps:.4f}")
    return model

# -----------------------------------------------------------------------------
# 5. Decoding helpers
# -----------------------------------------------------------------------------
def apply_repetition_penalties(logits, generated_ids, ngram_block=NGRAM_BLOCK, penalty=1.2):
    if len(generated_ids) == 0:
        return logits
    logits = logits.clone()
    for gid in set(generated_ids):
        logits[..., gid] /= penalty
    if len(generated_ids) >= ngram_block - 1:
        prefix = tuple(generated_ids[-(ngram_block - 1):])
        blocked = set()
        for i in range(len(generated_ids) - ngram_block + 1):
            if tuple(generated_ids[i:i+ngram_block-1]) == prefix:
                blocked.add(generated_ids[i+ngram_block-1])
        for gid in blocked:
            logits[..., gid] = -1e9
    return logits

@torch.no_grad()
def generate_text(model, stoi, itos, markov_chain=None, prime="the", length=80, temperature=1.0, sensitivity=1.0, device="cpu"):
    model.eval()
    tokens = tokenize(prime)
    prompt_ids = [stoi.get(t, stoi[UNK_TOKEN]) for t in tokens]
    ids = [stoi[BOS_TOKEN]] + prompt_ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    h = None
    logits, h = model(x, h)
    vocab_size = logits.size(-1)
    prompt_bias = prompt_bias_from_tokens(prompt_ids, vocab_size, device, sensitivity=sensitivity)
    out = tokens[:]
    cur = x[:, -1:]
    state = tuple(ids[-MARKOV_ORDER:]) if len(ids) >= MARKOV_ORDER else (stoi[BOS_TOKEN],) * MARKOV_ORDER
    generated_ids = ids[:]
    for _ in range(length):
        logits, h = model(cur, h)
        step_logits = logits[:, -1, :] / max(1e-6, temperature)
        step_logits = step_logits + prompt_bias.unsqueeze(0)
        step_logits = apply_repetition_penalties(step_logits, generated_ids, ngram_block=NGRAM_BLOCK, penalty=1.2)
        probs = F.softmax(step_logits, dim=-1).squeeze(0)
        if markov_chain is not None:
            mprior = markov_prior_vector(markov_chain, state, vocab_size, device=device)
            if mprior.sum() > 0:
                probs = 0.65 * probs + 0.35 * mprior
                probs = probs / probs.sum().clamp_min(1e-12)
        probs = nilpotent_ideal_to_probs(probs_to_nilpotent_ideal(probs))
        next_id = torch.distributions.Categorical(probs=probs).sample().item()
        next_tok = itos[next_id]
        if next_tok == EOS_TOKEN:
            break
        if next_tok != out[-1]:
            out.append(next_tok)
            generated_ids.append(next_id)
            cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        if markov_chain is not None:
            state = tuple((list(state) + [next_id])[-MARKOV_ORDER:])
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

# -----------------------------------------------------------------------------
# 6. Execution Pipeline
# -----------------------------------------------------------------------------
def run_dataset_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    filename = input("Filename: ")
    dataset_rows = load_unique_rows(filename, kb_len=KB_LEN)
    cleaned_rows = []
    for row in dataset_rows:
        toks = tokenize(row)
        toks = build_combinatorial_token_stream([row], n=3) if toks else []
        if toks:
            cleaned_rows.append(" ".join(toks))
    if not cleaned_rows:
        cleaned_rows = dataset_rows
    full_text = ". ".join(cleaned_rows)
    tokens = tokenize(full_text)
    vocab, stoi, itos = build_vocab(tokens, min_freq=1)
    ids = encode([BOS_TOKEN] + tokens + [EOS_TOKEN], stoi)
    x, y = make_dataset(ids, seq_len=SEQ_LEN, pad_id=stoi[PAD_TOKEN])
    curve_prior = load_curve_prior(None, top_k=min(50, len(vocab)))
    markov_chain = build_markov_chain(ids, order=MARKOV_ORDER)
    model = CurvePriorNet(vocab_size=len(vocab), emb_dim=64, hidden=128, layers=2, pad_idx=stoi[PAD_TOKEN])
    print("Training model on dataset...")
    model = train_model(
        model, x, y,
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
        prime = input("USER: ")
        sample = generate_text(
            model,
            stoi,
            itos,
            markov_chain=markov_chain,
            prime=prime,
            length=600,
            temperature=0.8,
            device=device
        )
        print(f"\nGenerated Result: '{sample}'")

if __name__ == "__main__":
    run_dataset_pipeline()
