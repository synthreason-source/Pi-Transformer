import os
import math
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tokenization / vocab
# ---------------------------------------------------------------------------

def tokenize(text):
    # same as your original
    return re.findall(r"[A-Za-z']+|[.,!?;:]", text.lower())


def tokenize_with_source(text, source_id):
    toks = tokenize(text)
    return [(t, source_id) for t in toks]


def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab += [w for w, c in counts.items() if c >= min_freq and w not in vocab]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos


def encode(tokens, stoi):
    return [stoi.get(t, stoi["<unk>"]) for t in tokens]


# ---------------------------------------------------------------------------
# Interleaving and dataset construction
# ---------------------------------------------------------------------------

def interleave_ids(ids1, ids2, block_len=64):
    """
    Interleave two token-id streams in fixed-size blocks:
    A(block_len), B(block_len), A(block_len), B(block_len), ...
    then append remaining from whichever corpus is longer.
    """
    combined = []
    i1 = i2 = 0
    turn = 0  # 0 -> first corpus, 1 -> second corpus
    n1, n2 = len(ids1), len(ids2)

    while i1 < n1 or i2 < n2:
        if turn == 0 and i1 < n1:
            block = ids1[i1:i1 + block_len]
            combined.extend(block)
            i1 += block_len
        elif turn == 1 and i2 < n2:
            block = ids2[i2:i2 + block_len]
            combined.extend(block)
            i2 += block_len

        # flip turn
        turn = 1 - turn

        # if one stream is exhausted, append remaining from the other
        if i1 >= n1 and i2 < n2:
            combined.extend(ids2[i2:])
            break
        if i2 >= n2 and i1 < n1:
            combined.extend(ids1[i1:])
            break

    return combined


def build_source_stream(tokens1, tokens2, block_len=64):
    """
    Same interleaving pattern, but keeps (token_str, source_id) pairs.
    Used only for visualization.
    """
    stream = []
    i1 = i2 = 0
    turn = 0
    n1, n2 = len(tokens1), len(tokens2)

    while i1 < n1 or i2 < n2:
        if turn == 0 and i1 < n1:
            block = tokens1[i1:i1 + block_len]
            stream.extend(block)
            i1 += block_len
        elif turn == 1 and i2 < n2:
            block = tokens2[i2:i2 + block_len]
            stream.extend(block)
            i2 += block_len

        turn = 1 - turn

        if i1 >= n1 and i2 < n2:
            stream.extend(tokens2[i2:])
            break
        if i2 >= n2 and i1 < n1:
            stream.extend(tokens1[i1:])
            break

    return stream


def make_dataset(ids, seq_len=32, stride=1):
    """
    Sliding-window dataset over a unified id stream.
    stride controls how far the window advances each step.
    """
    xs, ys = [], []
    n = len(ids)
    for i in range(0, n - seq_len - 1, stride):
        xs.append(ids[i:i + seq_len])
        ys.append(ids[i + 1:i + seq_len + 1])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def show_sliding_windows(inter_stream, seq_len=32, stride=8, num_windows=5):
    """
    'Show and tell' visualization of how the sliding window moves
    over the interleaved stream. Prints tokens with source labels.
    """
    print("\n=== Sliding window 'show and tell' ===")
    n = len(inter_stream)
    count = 0
    for start in range(0, n - seq_len, stride):
        if count >= num_windows:
            break
        window = inter_stream[start:start + seq_len]
        line = " | ".join(f"{t} (S{src})" for t, src in window)
        print(f"Window {count} [start={start}]:")
        print(line)
        print()
        count += 1
    print("=== End of show and tell ===\n")


# ---------------------------------------------------------------------------
# Curve prior
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, x, y, curve_prior=None, curve_weight=0.0,
                epochs=20, batch_size=64, lr=3e-4, device="cpu"):
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
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1)
            )

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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, vocab, config):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "config": config,
        },
        path,
    )
    print(f"[checkpoint] saved to {path}")


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    vocab = ckpt["vocab"]
    config = ckpt["config"]

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}

    model = CurvePriorNet(
        vocab_size=len(vocab),
        emb_dim=config.get("emb_dim", 64),
        hidden=config.get("hidden", 128),
        layers=config.get("layers", 2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"[checkpoint] loaded from {path}")
    return model, vocab, stoi, itos, config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("output", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_ckpt_path = "output/model.pt"

    ckpt_path = input(
        f"Checkpoint path to load (blank to skip, default '{default_ckpt_path}'): "
    ).strip()
    if ckpt_path == "":
        ckpt_path = default_ckpt_path

    model = None
    vocab = stoi = itos = None
    config = None

    # Try to load existing checkpoint
    if os.path.exists(ckpt_path):
        load_choice = input(
            f"Found checkpoint at '{ckpt_path}'. Load it? [Y/n]: "
        ).strip().lower()
        if load_choice in ("", "y", "yes"):
            model, vocab, stoi, itos, config = load_checkpoint(ckpt_path, device=device)

    # If no model loaded, train a new one using TWO text files
    if model is None:
        fname1 = input("Primary filename: ").strip()
        fname2 = "science_corpus.txt"

        with open(fname1, "r", encoding="utf8") as f:
            text1 = f.read()
        with open(fname2, "r", encoding="utf8") as f:
            text2 = f.read()

        tokens1 = tokenize_with_source(text1, source_id=0)
        tokens2 = tokenize_with_source(text2, source_id=1)

        raw_tokens1 = [t for (t, _) in tokens1]
        raw_tokens2 = [t for (t, _) in tokens2]

        vocab, stoi, itos = build_vocab(raw_tokens1 + raw_tokens2, min_freq=1)

        ids1 = encode(["<bos>"] + raw_tokens1 + ["<eos>"], stoi)
        ids2 = encode(["<bos>"] + raw_tokens2 + ["<eos>"], stoi)

        block_len = 64        # size of chunks when interleaving
        seq_len = 32          # sliding window length
        stride = 8            # stride of sliding window

        combined_ids = interleave_ids(ids1, ids2, block_len=block_len)
        inter_stream = build_source_stream(tokens1, tokens2, block_len=block_len)

        # Visualize a few sliding windows over the interspersed stream
        show_sliding_windows(inter_stream, seq_len=seq_len, stride=stride, num_windows=5)

        x, y = make_dataset(combined_ids, seq_len=seq_len, stride=stride)

        curve_prior = load_curve_prior(None, top_k=min(50, len(vocab)))

        config = {
            "emb_dim": 64,
            "hidden": 128,
            "layers": 2,
            "seq_len": seq_len,
            "block_len": block_len,
            "stride": stride,
        }
        model = CurvePriorNet(
            vocab_size=len(vocab),
            emb_dim=config["emb_dim"],
            hidden=config["hidden"],
            layers=config["layers"],
        )
        model = train_model(
            model,
            x,
            y,
            curve_prior=curve_prior,
            curve_weight=0.05,
            epochs=20,
            batch_size=64,
            lr=3e-4,
            device=device,
        )

        save_path = input(f"Save trained model as: ").strip() or ckpt_path
        save_checkpoint(save_path, model, vocab, config)

    # Interactive generation loop
    while True:
        prompt = input("USER: ")
        cmd = prompt.strip().lower()
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/save":
            save_path = input(f"Save model to [{ckpt_path}]: ").strip() or ckpt_path
            save_checkpoint(save_path, model, vocab, config)
            continue

        sample = generate_text(
            model,
            stoi,
            itos,
            prime=prompt,
            length=600,
            temperature=0.9,
            device=device,
        )
        with open("output/sample.txt", "w", encoding="utf-8") as f:
            f.write(sample)

        print(sample)


if __name__ == "__main__":
    main()
