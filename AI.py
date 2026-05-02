import re
import json
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE = 5000
SEQ_LEN = 16
EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

MODEL_FILE = "simple_lm.pt"
TOKENIZER_FILE = "simple_tokenizer.json"


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def length_similarity(a: str, b: str) -> float:
    la = len(a.split())
    lb = len(b.split())
    if la == 0 and lb == 0:
        return 1.0
    return 1.0 - abs(la - lb) / max(la, lb, 1)


def cosine_length_sort_sentences(sentences: List[str], length_weight: float = 0.35) -> List[str]:
    if len(sentences) <= 1:
        return sentences

    vocab = {}
    for s in sentences:
        for w in s.lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)

    def bow(text: str) -> List[float]:
        vec = [0.0] * len(vocab)
        for w in text.lower().split():
            if w in vocab:
                vec[vocab[w]] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    vecs = [bow(s) for s in sentences]
    ref = vecs[0]

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    scored = []
    for s, v in zip(sentences, vecs):
        cos = cosine(v, ref)
        ls = length_similarity(s, sentences[0])
        score = (1.0 - length_weight) * cos + length_weight * ls
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


class Tokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.t2i = {"<pad>": 0, "<unk>": 1}
        self.i2t = {0: "<pad>", 1: "<unk>"}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build(self, texts: List[str]):
        freq = {}
        for text in texts:
            for tok in self.tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1
        items = sorted(freq.items(), key=lambda x: -x[1])[: self.vocab_size - 2]
        for tok, _ in items:
            idx = len(self.t2i)
            self.t2i[tok] = idx
            self.i2t[idx] = tok

    def encode(self, text: str) -> List[int]:
        return [self.t2i.get(tok, 1) for tok in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.i2t.get(int(i), "<unk>") for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.t2i, f)

    @staticmethod
    def load(path: str):
        tok = Tokenizer()
        with open(path, "r", encoding="utf-8") as f:
            tok.t2i = json.load(f)
        tok.i2t = {v: k for k, v in tok.t2i.items()}
        return tok


class TextDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int = SEQ_LEN):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx + self.seq_len]
        y = self.token_ids[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.rnn(x)
        return self.fc(out)

    @torch.no_grad()
    def generate(self, start_ids: List[int], max_new_tokens: int = 30, device: str = "cpu") -> List[int]:
        self.eval()
        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            logits = self(x)
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)

        return x[0].tolist()


def save_all(model: SimpleLM, tokenizer: Tokenizer):
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": len(tokenizer.t2i),
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
        },
        MODEL_FILE,
    )
    tokenizer.save(TOKENIZER_FILE)


def load_all(device: str = "cpu"):
    tokenizer = Tokenizer.load(TOKENIZER_FILE)
    ckpt = torch.load(MODEL_FILE, map_location=device)
    model = SimpleLM(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer


def train(text: str):
    sentences = split_sentences(text)
    if sentences:
        text = " ".join(cosine_length_sort_sentences(sentences))

    tokenizer = Tokenizer()
    tokenizer.build([text])
    token_ids = tokenizer.encode(text)

    if len(token_ids) <= SEQ_LEN:
        raise ValueError("Text too short for training.")

    dataset = TextDataset(token_ids, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleLM(vocab_size=len(tokenizer.t2i)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch {epoch + 1}/{EPOCHS} loss={total_loss / len(loader):.6f}")

    save_all(model, tokenizer)
    print(f"saved model -> {MODEL_FILE}")
    print(f"saved tokenizer -> {TOKENIZER_FILE}")


def generate_once(prompt: str, max_new_tokens: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_all(device=device)

    start_ids = tokenizer.encode(prompt)
    if not start_ids:
        start_ids = [tokenizer.t2i["<unk>"]]

    out_ids = model.generate(start_ids, max_new_tokens=max_new_tokens, device=device)
    print(tokenizer.decode(out_ids))


def generate_loop(max_new_tokens: int = 300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_all(device=device)

    print("Generation loop ready. Type q to exit.")
    while True:
        prompt = input("prompt: ").strip()
        if prompt.lower() in ("q"):
            break

        start_ids = tokenizer.encode(prompt)
        if not start_ids:
            start_ids = [tokenizer.t2i["<unk>"]]

        out_ids = model.generate(start_ids, max_new_tokens=max_new_tokens, device=device)
        print(tokenizer.decode(out_ids))
        print()


if __name__ == "__main__":
    while True:
        c = input("(t)rain (g)en-once (i)nteractive (l)oad-test (q)uit > ").strip().lower()

        if c == "t":
            path = input("File: ").strip()
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            train(text)

        elif c == "g":
            prompt = input("prompt: ").strip()
            generate_once(prompt)

        elif c == "i":
            generate_loop()

        elif c == "l":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer = load_all(device=device)
            print("model and tokenizer loaded successfully")

        elif c in ("q", "quit"):
            break
