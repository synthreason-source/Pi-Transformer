"""
Noetic LM (Standard Transformer version without Multi-View Memory).
Word-level tokenization with full file corpus loading.

Run: python noetic_lm.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict


# --------------------------------------------------------------------------- #
# Transformer Block
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class NoeticLM(nn.Module):
    def __init__(self, vocab, d=128, heads=4, layers=3, ctx=64, ground_dim=0):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.ground = nn.Linear(ground_dim, d) if ground_dim else None   # grounding channel
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, idx, ground=None, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        if self.ground is not None and ground is not None:
            x = x + self.ground(ground).unsqueeze(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, ground=None, temperature=0.8):
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -self.ctx:], ground)
            p = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx


# --------------------------------------------------------------------------- #
# Demo: corpus loading & vocabulary building
# --------------------------------------------------------------------------- #
with open(input("Filename: "), "r", encoding="utf8") as f:
    _CORPORA = f.read()
    CORPORA = [_CORPORA] * 40


def main(steps=600, seed=0):
    torch.manual_seed(seed)
    text = "".join(CORPORA)
    
    # Tokenize corpus into words using simple whitespace splitting
    all_words = text.split()
    vocab_words = sorted(set(all_words))
    
    stoi = {w: i for i, w in enumerate(vocab_words)}
    itos = {i: w for w, i in stoi.items()}
    
    # Encoder safely maps known words, ignoring out-of-vocab entries
    enc = lambda s: torch.tensor([stoi[w] for w in s.split() if w in stoi])
    
    data = [enc(c) for c in CORPORA]
    data = [d for d in data if len(d) > 65]  # Ensure sequences exceed context window
    if len(data) == 0:
        data = [enc(c) for c in CORPORA]

    model = NoeticLM(len(vocab_words), ground_dim=len(CORPORA), ctx=32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ctx, bs = model.ctx, min(16, len(data))
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, steps + 1):
        model.train()
        which = torch.randint(0, len(CORPORA), (bs,))
        xs, ys = [], []
        for w in which:
            if len(data[w]) <= ctx + 1:
                padded = F.pad(data[w], (0, ctx + 2 - len(data[w])))
                xs.append(padded[:ctx])
                ys.append(padded[1:ctx + 1])
            else:
                i = torch.randint(0, len(data[w]) - ctx - 1, (1,)).item()
                xs.append(data[w][i:i + ctx])
                ys.append(data[w][i + 1:i + ctx + 1])
                
        g = F.one_hot(which, len(CORPORA)).float()
        _, loss = model(torch.stack(xs), g, torch.stack(ys))
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0:
            print(f"step {step:4d} loss {loss.item():.3f}")
            
    while True:
        user_prompt = input("\nUSER: ")
        prompt = enc(user_prompt).unsqueeze(0)
        if prompt.size(1) == 0:
            print("Prompt words out of vocabulary. Try again.")
            continue
        # Use first grounding index for generation testing
        w = 0
        g = F.one_hot(torch.tensor([w]), len(CORPORA)).float()
        out = model.generate(prompt, 600, g)[0].tolist()
        print(" ".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
