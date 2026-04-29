from __future__ import annotations

import math, os, json
from typing import List, Optional, Tuple, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
VOCAB       = 10000
D_MODEL     = 64
MAX_SEQ_LEN = 128
SEQ_LEN     = 64
DROPOUT     = 0.1
LR          = 6e-4
EPOCHS      = 1
KB_LEN      = 9999
CHECKPOINT  = "v18_geo.pt"
TOKENIZER_F = "v18_tokenizer.json"


# ─────────────────────────────────────────────────────────────
# 84 VECTOR MAGNET ENCODING
# ─────────────────────────────────────────────────────────────
class VectorMagnetEncoding(nn.Module):
    """
    Eighty-four-vector magnetic field style encoding.
    Each magnet contributes a 2D field pair, giving 168 raw features.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.phase = nn.Parameter(torch.tensor(0.983))
        self.scale = nn.Parameter(torch.tensor(1.0))

        mags = torch.linspace(0.55, 4.75, 111184)
        axes = torch.linspace(1.0, -1.0, 111184)

        self.magnets = nn.Parameter(mags)
        self.axes = nn.Parameter(axes)

        self.proj = nn.Linear(168, d_model)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """
        pos_ids: (B, T)
        returns: (B, T, D_MODEL)
        """
        pos = pos_ids.float()
        B, T = pos.shape

        feats = []
        for i in range(84):
            w = self.magnets[i]
            a = self.axes[i]
            x = pos * w + self.phase + (i * 0.01)

            bx = torch.sin(x) * a
            by = torch.cos(x) * (1.0 - 0.1 * a)

            feats.append(bx)
            feats.append(by)

        feats = torch.stack(feats, dim=-1)
        return self.proj(feats) * self.scale / math.sqrt(self.d_model)


# ─────────────────────────────────────────────────────────────
# TOKENIZER (minimal trigram version)
# ─────────────────────────────────────────────────────────────
class WordTokenizer:
    def __init__(self, vocab_size=VOCAB):
        self.vocab_size = vocab_size
        self.t2i = {"<pad>": 0, "<unk>": 1}
        self.i2t = {0: "<pad>", 1: "<unk>"}

    def _tok(self, text):
        w = text.lower().split()
        return [" ".join(w[i:i+3]) for i in range(len(w)-2)]

    def build(self, texts):
        freq = {}
        for t in texts:
            for tri in self._tok(t):
                freq[tri] = freq.get(tri, 0) + 1

        for k,_ in sorted(freq.items(), key=lambda x:-x[1])[:self.vocab_size-2]:
            idx = len(self.t2i)
            self.t2i[k] = idx
            self.i2t[idx] = k

    def encode(self, text):
        toks = self._tok(text[:KB_LEN])
        if len(toks) == 0:
            toks = ["<unk>"]
        return torch.tensor([self.t2i.get(t,1) for t in toks], dtype=torch.long)

    def decode(self, ids):
        out = []
        for i in ids:
            out.extend(self.i2t.get(int(i),"<unk>").split())
        return " ".join(out)

    def save(self, p):
        json.dump(self.t2i, open(p,"w"))

    @staticmethod
    def load(p):
        t = WordTokenizer()
        t.t2i = json.load(open(p))
        t.i2t = {v:k for k,v in t.t2i.items()}
        return t


# ─────────────────────────────────────────────────────────────
# MODEL CORE
# ─────────────────────────────────────────────────────────────
class V18Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok = nn.Embedding(VOCAB, D_MODEL)

        # 84-vector magnetic encoding replaces positional embedding
        self.pos = VectorMagnetEncoding(D_MODEL, MAX_SEQ_LEN)

        self.drop = nn.Dropout(DROPOUT)

        self.tr = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL)
        )

        self.out = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        B,T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B,-1)

        x = self.tok(x) + self.pos(pos)
        x = self.drop(x)
        x = self.tr(x)

        return self.out(x)

    @torch.no_grad()
    def generate(self, x, steps=150):
        for _ in range(steps):
            logits = self(x)
            probs = torch.softmax(logits[:,-1], dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return x


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
class TextDS(Dataset):
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return max(0, len(self.t)-SEQ_LEN-1)

    def __getitem__(self,i):
        x = self.t[i:i+SEQ_LEN+1]
        return x[:-1], x[1:]


def collate(batch):
    x = pad_sequence([b[0] for b in batch], True, 0)
    y = pad_sequence([b[1] for b in batch], True, 0)
    return x,y


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
def train(text):
    tok = WordTokenizer()
    tok.build([text])

    tokens = tok.encode(text)

    ds = TextDS(tokens)
    if len(ds) == 0:
        raise ValueError("Text too short for current SEQ_LEN")

    dl = DataLoader(ds, batch_size=16, collate_fn=collate, shuffle=True)

    m = V18Model()
    opt = optim.AdamW(m.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    m.train()
    for e in range(EPOCHS):
        last_loss = None
        for x,y in dl:
            logits = m(x)
            loss = loss_fn(logits.transpose(1,2), y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()

        print("epoch", e, "loss", last_loss)

    torch.save(m.state_dict(), CHECKPOINT)
    tok.save(TOKENIZER_F)


# ─────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────
def generate(prompt):
    tok = WordTokenizer.load(TOKENIZER_F)

    m = V18Model()
    m.load_state_dict(torch.load(CHECKPOINT))
    m.eval()

    ids = tok.encode(prompt).unsqueeze(0)

    out = m.generate(ids, 180)

    print(tok.decode(out[0]))


# ─────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────
def gui():
    tok = WordTokenizer.load(TOKENIZER_F)
    m = V18Model()
    m.load_state_dict(torch.load(CHECKPOINT))
    m.eval()

    print("V18 GUI READY")

    while True:
        s = input(">>> ")
        if s in ("q","quit"):
            break

        ids = tok.encode(s).unsqueeze(0)
        out = m.generate(ids, 60)
        print(tok.decode(out[0]))


# ─────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        c = input("(t)rain (g)en (i)gui > ")

        if c == "t":
            text = open(input("File: "), encoding="utf-8").read()
            train(text)

        if c == "g":
            generate(input("prompt:"))

        if c == "i":
            gui()
