import os
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class LieBracketLayer(nn.Module):
    """
    Computes the Lie Bracket f(x, y) = [x, y] to model causal influence.
    f(x, y) = xWy^T - yWx^T (Skew-symmetric / Alternating)
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(emb_dim, emb_dim) * 0.1)

    def forward(self, x, y):
        # x, y shape: (batch, emb_dim)
        # Restrict W to the nilpotent ideal n (strictly upper-triangular
        # matrices) before using it, so the parameter itself is always a
        # genuine element of the ideal: W_n^emb_dim = 0.
        W_n = torch.triu(self.W, diagonal=1)

        # Bilinear interaction: xWy^T
        term1 = torch.matmul(torch.matmul(x, W_n), y.transpose(-1, -2))
        term2 = torch.matmul(torch.matmul(y, W_n), x.transpose(-1, -2))

        # Alternating Property: f(x, y) = -f(y, x)
        return term1 - term2

class CurvePriorNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden, num_layers=layers, batch_first=True)

        # Lie Bracket module to capture interaction dynamics
        self.lie_bracket = LieBracketLayer(emb_dim)

        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        # 1. Standard RNN processing
        embs = self.embed(x)
        # Route embeddings through the nilpotent ideal (embed/read-back
        # roundtrip on the last dimension, batched over every leading dim).
        embs = from_nilpotent_ideal(to_nilpotent_ideal(embs))

        out, h_new = self.rnn(embs, h)
        out = from_nilpotent_ideal(to_nilpotent_ideal(out))
        # h_new is the GRU's carried state; route it too so no numeric
        # variable in the forward pass skips the ideal.
        h_new = from_nilpotent_ideal(to_nilpotent_ideal(h_new))

        # 2. Synthesis with Lie Bracket
        # We look at the relationship between the last two tokens to compute influence
        if x.size(1) > 1:
            prev_token = embs[:, -2, :]
            curr_token = embs[:, -1, :]

            # Compute Lie Bracket as "influence score"
            influence = self.lie_bracket(prev_token, curr_token) # Shape: (batch, batch) or similar

            # Apply Exponential mapping to scale influence
            # Map the influence into the logit space (exp/log transformation)
            influence_map = torch.exp(influence.mean(dim=-1, keepdim=True))

            # Modulate logits
            logits = self.head(out) * influence_map.unsqueeze(-1)
        else:
            logits = self.head(out)

        logits = from_nilpotent_ideal(to_nilpotent_ideal(logits))

        return logits, h_new


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


# ---------------------------------------------------------------------------
# Nilpotent-ideal embedding
#
# In the Borel subalgebra b = t ⊕ n of gl_k (t = diagonal matrices, n =
# strictly-upper-triangular matrices), n is a nilpotent ideal: it is closed
# under the bracket with all of b, and N^k = 0 for any N in n, since each
# matrix product pushes the nonzero band strictly further off the diagonal
# until it falls off the edge of the k x k matrix.
#
# Any vector v = (v_0, ..., v_{k-1}) -- along the last dimension of any
# tensor, batch dims included -- can be embedded into this ideal by placing
# it on the superdiagonal of an otherwise-zero k x k matrix. The resulting
# matrix N is nilpotent by construction, and no extra machinery is needed
# to guarantee that; it falls out of the shape of the ideal itself.
#
# This is a general tensor operation, not just something for probability
# vectors: anything with a last dimension can be routed through it
# (embeddings, hidden states, logits, gradients, ...). Objects that are NOT
# numeric tensors -- strings, dicts, ints, module references -- have no
# ring for an ideal to live in, so they are left untouched.
# ---------------------------------------------------------------------------

def to_nilpotent_ideal(t):
    """
    Embed the last dimension of an arbitrary tensor into the nilpotent
    ideal n (strictly upper-triangular matrices) of the Borel subalgebra b
    of gl_k, batched over all leading dimensions.

    Input shape (..., k) -> output shape (..., k, k), with
    N[..., i, i+1] = t[..., i] for i < k - 1. N^k = 0 automatically.
    """
    *batch, k = t.shape
    N = torch.zeros(*batch, k, k, dtype=t.dtype, device=t.device)
    if k > 1:
        i0 = torch.arange(k - 1, device=t.device)
        i1 = torch.arange(1, k, device=t.device)
        N[..., i0, i1] = t[..., :-1]
    return N


def from_nilpotent_ideal(N):
    """
    Recover a vector from an element of the nilpotent ideal by reading back
    its superdiagonal (row sums, since each row of N has at most one
    nonzero entry). Inverse of to_nilpotent_ideal (up to the dropped last
    coordinate, which the ideal has no room to store).
    """
    return N.sum(dim=-1)


def probs_to_nilpotent_ideal(probs):
    """Vector-only convenience wrapper (numpy or torch) used for the
    curve-prior distribution, kept for backward compatibility."""
    if isinstance(probs, torch.Tensor):
        return to_nilpotent_ideal(probs)
    probs_t = torch.from_numpy(np.asarray(probs))
    return to_nilpotent_ideal(probs_t).numpy()


def nilpotent_ideal_to_probs(N, eps=1e-12):
    """Vector-only convenience wrapper that also renormalizes back into a
    valid probability distribution, kept for backward compatibility."""
    if isinstance(N, torch.Tensor):
        vec = from_nilpotent_ideal(N)
        return vec / vec.sum().clamp_min(eps)
    N_t = torch.from_numpy(np.asarray(N))
    vec = from_nilpotent_ideal(N_t).numpy()
    s = vec.sum()
    if s < eps:
        return np.full_like(vec, 1.0 / len(vec))
    return vec / s


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

    # Embed into the nilpotent ideal, then read back a normalized vector.
    # N^k = 0 holds for the intermediate matrix by construction.
    N = probs_to_nilpotent_ideal(probs)
    probs = nilpotent_ideal_to_probs(N).astype(np.float32)

    return probs


def prompt_bias_from_tokens(prompt_ids, vocab_size, device, sensitivity=1.0):
    bias = torch.zeros(vocab_size, device=device)
    if len(prompt_ids) == 0:
        return bias
    counts = torch.bincount(torch.tensor(prompt_ids, device=device), minlength=vocab_size).float()
    counts = counts / counts.sum().clamp_min(1.0)
    bias = sensitivity * counts
    return bias


def train_model(
    model,
    x,
    y,
    curve_prior=None,
    curve_weight=0.0,
    sensitivity=1.0,
    epochs=20,
    batch_size=64,
    lr=3e-4,
    device="cpu"
):
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
                prior = prior / prior.sum().clamp_min(1e-12)

                last_logits = logits[:, -1, :vocab_slice]
                pred = F.softmax(last_logits, dim=-1).mean(dim=0)
                prior_loss = F.mse_loss(pred, prior)

                loss = loss + curve_weight * sensitivity * prior_loss

            opt.zero_grad()
            loss.backward()

            # Route every gradient through the nilpotent ideal too, so the
            # optimizer never touches a raw, un-embedded numeric variable.
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.copy_(from_nilpotent_ideal(to_nilpotent_ideal(p.grad)))

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        print(f"epoch {epoch+1:03d} loss {total/steps:.4f}")

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

        # Route the sampling distribution through the nilpotent ideal before
        # drawing a sample -- same embed/read-back roundtrip used for the
        # curve prior.
        N = probs_to_nilpotent_ideal(probs)
        probs = nilpotent_ideal_to_probs(N).unsqueeze(0)

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


def main():
    os.makedirs("output", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_ckpt_path = "output/model.pt"

    ckpt_path = input(f"Checkpoint path to load (blank to skip, default '{default_ckpt_path}'): ").strip()
    if ckpt_path == "":
        ckpt_path = default_ckpt_path

    model = None
    vocab = stoi = itos = None
    config = None

    if os.path.exists(ckpt_path):
        load_choice = input(f"Found checkpoint at '{ckpt_path}'. Load it? [Y/n]: ").strip().lower()
        if load_choice in ("", "y", "yes"):
            model, vocab, stoi, itos, config = load_checkpoint(ckpt_path, device=device)

    if model is None:
        filename = input("Filename: ").strip()
        with open(filename, "r", encoding="utf8") as f:
            text = f.read()[:9999]

        tokens = tokenize(text)
        vocab, stoi, itos = build_vocab(tokens, min_freq=1)
        ids = encode(["<bos>"] + tokens + ["<eos>"], stoi)

        seq_len = 32
        x, y = make_dataset(ids, seq_len=seq_len)

        curve_prior = load_curve_prior(None, top_k=min(50, len(vocab)))

        config = {"emb_dim": 64, "hidden": 128, "layers": 2, "seq_len": seq_len}
        model = CurvePriorNet(
            vocab_size=len(vocab),
            emb_dim=config["emb_dim"],
            hidden=config["hidden"],
            layers=config["layers"]
        )

        model = train_model(
            model,
            x,
            y,
            curve_prior=curve_prior,
            curve_weight=0.05,
            sensitivity=50,
            epochs=20,
            batch_size=64,
            lr=3e-4,
            device=device,
        )

        save_path = input(f"Save trained model to [{ckpt_path}]: ").strip() or ckpt_path
        save_checkpoint(save_path, model, vocab, config)

    while True:
        prompt = input("USER: ")
        if prompt.strip().lower() in ("/quit", "/exit"):
            break
        if prompt.strip().lower() == "/save":
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
            sensitivity=10.1,
            device=device,
        )

        with open("output/sample.txt", "w", encoding="utf-8") as f:
            f.write(sample)

        print(sample)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Surjection Matrix (rows grouped by sentence length -> columns A,B,C,...)
# Demonstration structure based on the supplied diagram.
# ---------------------------------------------------------------------------

SURJECTION_COLUMNS = ["A","B","C","D","E"]

SURJECTION_MATRIX = {
    3: [("r1","A"),("r2","C"),("r3","B"),("r4","E")],
    4: [("r5","A"),("r6","D"),("r7","B"),("r8","C")],
    5: [("r9","B"),("r10","D"),("r11","A")],
    6: [("r12","C"),("r13","E")],
}

def apply_surjection_matrix(sentence_chunks):
    """
    sentence_chunks: iterable of strings.
    Groups rows by sentence length (word count), sorts by size,
    then surjectively assigns rows to columns A,B,C,...
    Returns list of dictionaries.
    """
    grouped = {}
    for s in sentence_chunks:
        n = len(s.split())
        grouped.setdefault(n, []).append(s)

    result = []
    for length in sorted(grouped):
        mapping = SURJECTION_MATRIX.get(length, [])
        for i, sent in enumerate(grouped[length]):
            col = mapping[i % max(1,len(mapping))][1] if mapping else SURJECTION_COLUMNS[i % len(SURJECTION_COLUMNS)]
            result.append({
                "length": length,
                "row": f"r{i+1}",
                "column": col,
                "sentence": sent,
            })
    return result
