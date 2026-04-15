
# =====================================================
# KERNEL-LLM v1.3 - COMPLETE & FIXED (paste.txt → TRAINED MODEL)
# 3x5 Grid Compression of V18 NeuroSymbolic | No Errors | Ready to Run
# =====================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import math
import os
import sys

print("LLM Loading...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------------------------------
# 1. 3x5  KERNEL GRID (15% of original paste.txt math)
# -------------------------------------------------
def aniso_kern(drho, dtheta, lr=0.1, lt=0.9):
    return torch.exp(-lr * drho**-2 - lt * dtheta**2)

def l1_proj(x, eps=1e-12):
    min_vals = x.min(dim=-1, keepdim=True).values  # ← grab values only
    x = F.softplus(x - min_vals)
    return x / (x.sum(dim=-1, keepdim=True) + eps)

def layer_norm(x, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mu) / (std + eps)

def mobius_shift(a, b, c=0.35):
    ta = torch.tanh(a * 0.1)
    tb = torch.tanh(b * 0.1)
    denom = (1 + c * ta * tb).clamp(min=1e-6)
    return torch.atanh((ta + c * tb) / denom)

def simplex_proj(x):
    x = F.relu(x - x.min(dim=-1, keepdim=True))
    return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-12)

def orbit_bonus(theta, n, sector):
    return 0.5 * torch.cos(2 * math.pi * (theta / sector - n)) + 0.5

# -------------------------------------------------
# 2.  KERNEL LAYER (ALL CLASSES PROPERLY DEFINED)
# -------------------------------------------------
class KernelLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.grid_bias = nn.Parameter(torch.randn(15, d_model) * 0.1)  # 3x5=15

    def forward(self, x, rho, theta, sigma):
        B, L, D = x.shape


        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Ensure rho, theta are (B, L)
        assert rho.shape == (B, L), f"rho: {rho.shape}"
        assert theta.shape == (B, L), f"theta: {theta.shape}"

        # FIXED 3x5 GRID INJECTION (safe broadcasting)
        drho = rho.unsqueeze(-1) - rho.mean(dim=1, keepdim=True).unsqueeze(-1)  # (B,L,1)
        dtheta = theta.unsqueeze(-1) - theta.mean(dim=1, keepdim=True).unsqueeze(-1)  # (B,L,1)

        kern_map = aniso_kern(drho, dtheta)  # (B,L,1)
        kern_map = kern_map.squeeze(-1).unsqueeze(-1)  # explicit (B,L,1) view

        grid_size = self.grid_bias.shape[0]
        kern_expanded = kern_map.unsqueeze(-1).expand(B, L, grid_size, 1)  # (B,L,15,1)
        kern_grid = l1_proj(kern_expanded).squeeze(-1)  # (B,L,15)

        kern_ff = torch.bmm(kern_grid, self.grid_bias.unsqueeze(0).expand(B, -1, -1))  # (B,L,D)
        kern_ff_roll = kern_ff.roll(grid_size, dims=1)
        kern_ff = mobius_shift(kern_ff, kern_ff_roll) * orbit_bonus(theta.unsqueeze(-1), 0, 4)
        x = self.norm2(x + layer_norm(kern_ff))
        return x

# -------------------------------------------------
# 3. MAIN KERNEL-LLM
# -------------------------------------------------
class KernelLLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(512, d_model))
        nn.init.normal_(self.pos_enc, std=0.02)
        self.layers = nn.ModuleList([KernelLayer(d_model, nhead) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, rho=None, theta=None, sigma=None):
        B, L = x.shape
        assert L >= 2, f"Trigram model expects at least 2 input tokens, got {L}"
        emb = self.embed(x) + self.pos_enc[:L].unsqueeze(0)

        if rho is None: rho = torch.abs(torch.randn(B, L, device=x.device)*0.5 + 0.5)
        if theta is None: theta = (torch.rand(B, L, device=x.device)*2*math.pi - math.pi)
        if sigma is None: sigma = torch.ones(B, L, device=x.device)

        for layer in self.layers:
            emb = layer(emb, rho, theta, sigma)

        out = self.norm(emb[:, -1, :])   # (B, D)
        logits = self.head(out)          # (B, V)
        return logits
   
# -------------------------------------------------
# 4. PASTE.TXT DATASET
# -------------------------------------------------
class V18Dataset(Dataset):
    def __init__(self, paste_text: str, vocab_size=8192, max_samples=None):
        toks = paste_text.lower().split()
        if len(toks) < 3:
            raise ValueError("Need at least 3 words for trigram training.")

        counts = Counter(toks)
        vocab_list = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(vocab_size - 2)]
        self.word_to_id = {w: i for i, w in enumerate(vocab_list)}
        self.id_to_word = vocab_list
        self.vocab_size = len(vocab_list)

        def encode(w):
            return self.word_to_id.get(w, 1)  # <unk>

        self.data = []
        for i in range(len(toks) - 2):
            x0 = encode(toks[i])
            x1 = encode(toks[i + 1])
            y = encode(toks[i + 2])
            self.data.append((x0, x1, y))

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"V18Dataset: {len(self.data)} trigrams | vocab_size={self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x0, x1, y = self.data[idx]
        return {
            "input_ids": torch.tensor([x0, x1], dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
            "rho": torch.rand(2, device=DEVICE),
            "theta": torch.rand(2, device=DEVICE) * 2 * math.pi - math.pi,
            "sigma": torch.ones(2, device=DEVICE),
        }
# =====================================================
# 6. TEXT GENERATION (add to end of your script)
# =====================================================
@torch.inference_mode()
def generate(model: KernelLLM, seed_text: str, max_new_words: int = 128,
             temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
             device=DEVICE):
    model.eval()

    # 1) Assume model.head.weight.shape[0] == vocab_size
    vocab_size = model.head.weight.shape[0]

    # 2) Recover dataset‑level vocab from model (if you stored it)
    #    In practice you’d pass ds.id_to_word explicitly; here we fake it:
    if not hasattr(model, "id_to_word"):
        # you can also pass this as an arg to generate
        raise ValueError("Attach ds.id_to_word to model or pass vocab mapping.")

    id_to_word = model.id_to_word

    # 3) Tokenize seed into words
    toks = seed_text.lower().split()
    if not toks:
        raise ValueError("seed_text contains no words")

    # 4) Map to ints (use 0 for OOV)
    if not hasattr(model, "word_to_id"):
        raise ValueError("also attach word_to_id dict to model")

    word_to_id = model.word_to_id
    ids = [word_to_id.get(w, 0) for w in toks]
    if not ids:
        ids = [0]  # dummy token

    # 5) Start as sequence of word IDs
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

    for _ in range(max_new_words):
        L = x.shape[1]
        rho = (torch.rand(1, L, device=device) * 0.01 + 0.01)
        theta = (torch.rand(1, L, device=device) * 2 * math.pi - math.pi)
        sigma = torch.ones(1, L, device=device)

        logits = model(x, rho, theta, sigma)        # (1, L, V)
        next_logits = logits / temperature  # logits is already (1, V)
        # Top‑k + top‑p (same as before)
        # After temperature scaling, before top_p:
        if top_k > 0:
            top_k_vals = torch.topk(next_logits, top_k, dim=-1).values
            next_logits = next_logits.masked_fill(next_logits < top_k_vals[:, -1:], -float("inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            next_logits = next_logits.masked_fill(indices_to_remove, -float("inf"))

        probs = F.softmax(next_logits, dim=-1)          # (1, V)
        next_id = torch.multinomial(probs, 1)           # (1, 1)
        # clip to vocab range
        next_id = next_id.clamp(0, vocab_size - 1)
        x = torch.cat([x, next_id], dim=1)              # (1, L+1)

        # Stop heuristic: stop when you hit repeated “stop‑word” or max_new_words
        if id_to_word[next_id.item()] in {"<pad>", ".", "!", "?"}:
            break

    # 6) Convert new IDs back to words
    prefix_len = len(toks)
    new_ids = x[0, prefix_len:].tolist()
    new_words = [id_to_word[i] for i in new_ids if 0 <= i < len(id_to_word)]
    # skip <pad> at the end
    while new_words and new_words[-1] == "<pad>":
        new_words.pop()

    full_text = seed_text + " " + " ".join(new_words).strip()
    return full_text
# -------------------------------------------------
# 5. TRAINING & SAVE
# -------------------------------------------------
def main():
    # Load paste.txt
    paste_path = 'singlekb.txt'
    if os.path.exists(paste_path):
        with open(paste_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"✓ Loaded {paste_path} ({len(text)} chars)")
    else:
        text = "NeuroSymbolic V18 fallback text corpus for testing."
        print("⚠ singlekb.txt not found - using fallback corpus")

    # Dataset + Model
    ds = V18Dataset(text)
    print(f"Dataset length: {len(ds)}")
    if len(ds) == 0:
        print("Dataset is empty; cannot build DataLoader.")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=45, shuffle=True)

    model = KernelLLM(vocab_size=ds.vocab_size).to(DEVICE)
    model.word_to_id = ds.word_to_id   # attach for generation
    model.id_to_word = ds.id_to_word
    opt = optim.AdamW(model.parameters(), lr=3e-4)

    # Train
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            rho, theta, sigma = [b.to(DEVICE) for b in [batch['rho'], batch['theta'], batch['sigma']]]

            logits = model(input_ids, rho, theta, sigma)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/3 | Loss: {total_loss/len(loader):.4f}")

    # Save
    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), 'output/_llm_trained.pth')
    print(f"✓ Model ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print("🎉 COMPLETE - Trained model saved to output/_llm_trained.pth")

    model.eval()
    model.word_to_id = ds.word_to_id   # reattach vocab
    model.id_to_word = ds.id_to_word

    while True:
        print("\nGenerating text...")
        print(generate(
            model,
            seed_text=input("USER: "),
            max_new_words=640,
            temperature=111000000000.2,
            top_k=340,
            top_p=170.1
        ))
if __name__ == "__main__":
    main()
