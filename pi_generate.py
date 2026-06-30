import re
import math
import random
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
KB_LEN = 10000
EMBED_DIM = 256
HIDDEN_DIM = 512
SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILE = "plaintext_feature_model.pt"

if torch.cuda.is_available():
    print("CUDA ENABLED")
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
else:
    print("Running on CPU")

# ============================================================
# LOAD DATA
# ============================================================

filename = input("Filename: ")

with open(filename, "r", encoding="utf-8") as f:
    raw_text = f.read()[:KB_LEN]

TEXTS = [
    x.strip()
    for x in raw_text.split(".")
    if x.strip()
]

# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    return re.findall(
        r"<[^>]+>|\w+|[^\w\s]",
        text.lower()
    )

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def safe_entropy(text):
    if not text:
        return 0.0

    counts = Counter(text)

    total = len(text)

    ent = 0.0

    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def extract_feature_tokens(text):

    words = re.findall(r"\w+", text)

    chars = len(text)

    word_count = len(words)

    unique_words = len(
        set(w.lower() for w in words)
    )

    avg_word_len = (
        sum(len(w) for w in words)
        / max(word_count, 1)
    )

    longest_word = max(
        [len(w) for w in words] + [0]
    )

    shortest_word = min(
        [len(w) for w in words] + [0]
    )

    punct_count = len(
        re.findall(r"[^\w\s]", text)
    )

    digit_count = len(
        re.findall(r"\d", text)
    )

    upper_count = sum(
        1 for c in text if c.isupper()
    )

    lower_count = sum(
        1 for c in text if c.islower()
    )

    vowels = sum(
        1 for c in text.lower()
        if c in "aeiou"
    )

    consonants = sum(
        1 for c in text.lower()
        if c.isalpha() and c not in "aeiou"
    )

    entropy = safe_entropy(text)

    raw = np.array([
        word_count,
        unique_words,
        avg_word_len,
        longest_word,
        shortest_word,
        punct_count,
        digit_count,
        upper_count,
        lower_count,
        vowels,
        consonants,
        entropy
    ], dtype=np.float32)

    feature_names = [
        "wc",
        "uniq",
        "avglen",
        "maxlen",
        "minlen",
        "punct",
        "digits",
        "upper",
        "lower",
        "vowels",
        "cons",
        "entropy"
    ]

    transforms = {}

    transforms["raw"] = raw

    transforms["sig"] = np.array(
        [sigmoid(float(x)) for x in raw],
        dtype=np.float32
    )

    transforms["tanh"] = np.tanh(raw)

    transforms["log"] = np.log1p(
        np.maximum(raw, 0)
    )

    transforms["sqrt"] = np.sqrt(
        np.maximum(raw, 0)
    )

    transforms["square"] = raw ** 2

    transforms["cube"] = raw ** 3

    clipped = np.clip(raw, -5, 5)

    transforms["exp"] = np.exp(clipped)

    soft = np.exp(raw - np.max(raw))
    soft /= soft.sum()

    transforms["soft"] = soft

    mean = raw.mean()
    std = raw.std() + 1e-8

    transforms["z"] = (
        raw - mean
    ) / std

    mn = raw.min()
    mx = raw.max()

    transforms["minmax"] = (
        raw - mn
    ) / (mx - mn + 1e-8)

    transforms["sin"] = np.sin(raw)
    transforms["cos"] = np.cos(raw)

    feature_tokens = []

    for transform_name, values in transforms.items():

        for feature_name, value in zip(
            feature_names,
            values
        ):
            feature_tokens.append(
                f"<{transform_name}_{feature_name}_{value:.4f}>"
            )

    return feature_tokens

# ============================================================
# EMBED FEATURES INTO DATASET AS PLAINTEXT
# ============================================================

ENRICHED_TEXTS = []

for text in TEXTS:

    feature_tokens = extract_feature_tokens(text)

    enriched = (
        " ".join(feature_tokens)
        + " "
        + text
    )

    ENRICHED_TEXTS.append(enriched)

TEXTS = ENRICHED_TEXTS

print()
print("Examples:")
print(TEXTS[0][:500])
print()

# ============================================================
# VOCAB
# ============================================================

SPECIAL = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>"
]

counter = Counter()

for text in TEXTS:
    counter.update(tokenize(text))

vocab = SPECIAL + sorted(counter.keys())

stoi = {
    token: idx
    for idx, token in enumerate(vocab)
}

itos = {
    idx: token
    for token, idx in stoi.items()
}

PAD = stoi["<pad>"]
UNK = stoi["<unk>"]
BOS = stoi["<bos>"]
EOS = stoi["<eos>"]

print("Vocabulary:", len(vocab))

# ============================================================
# DATASET
# ============================================================

class TextDataset(Dataset):

    def __init__(self, texts):

        self.samples = []

        for text in texts:

            ids = [BOS]

            ids.extend(
                stoi.get(tok, UNK)
                for tok in tokenize(text)
            )

            ids.append(EOS)

            for i in range(1, len(ids)):

                start = max(
                    0,
                    i - SEQ_LEN
                )

                x = ids[start:i]

                y = ids[
                    start + 1:
                    i + 1
                ]

                self.samples.append(
                    (x, y)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate(batch):

    xs, ys = zip(*batch)

    max_len = max(
        len(x)
        for x in xs
    )

    xpad = []
    ypad = []

    for x, y in zip(xs, ys):

        pad = max_len - len(x)

        xpad.append(
            x + [PAD] * pad
        )

        ypad.append(
            y + [PAD] * pad
        )

    return (
        torch.tensor(xpad),
        torch.tensor(ypad)
    )

dataset = TextDataset(TEXTS)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
    pin_memory=torch.cuda.is_available(),
    num_workers=0
)

# ============================================================
# MODEL
# ============================================================

class PlaintextFeatureGRU(nn.Module):

    def __init__(
        self,
        vocab_size
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=PAD
        )

        self.gru = nn.GRU(
            EMBED_DIM,
            HIDDEN_DIM,
            batch_first=True
        )

        self.output = nn.Linear(
            HIDDEN_DIM,
            vocab_size
        )

    def forward(self, x):

        emb = self.embedding(x)

        out, _ = self.gru(emb)

        return self.output(out)

model = PlaintextFeatureGRU(
    len(vocab)
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits.reshape(
                -1,
                len(vocab)
            ),
            y.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={total_loss/len(loader):.4f}"
    )

torch.save(
    {
        "model": model.state_dict(),
        "stoi": stoi,
        "itos": itos
    },
    "plaintext_feature_model.pt"
)

print()
print("Saved: plaintext_feature_model.pt")
print()

# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_tokens=100,
    temperature=0.8
):

    model.eval()

    features = extract_feature_tokens(
        prompt
    )

    enriched_prompt = (
        " ".join(features)
        + " "
        + prompt
    )

    current = [BOS]

    current.extend(
        stoi.get(tok, UNK)
        for tok in tokenize(
            enriched_prompt
        )
    )

    for _ in range(max_tokens):

        x = torch.tensor(
            [current[-SEQ_LEN:]],
            device=DEVICE
        )

        logits = model(x)

        logits = (
            logits[0, -1]
            / temperature
        )

        probs = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            1
        ).item()

        if next_token == EOS:
            break

        current.append(
            next_token
        )

    output_tokens = [
        itos[i]
        for i in current
        if i in itos
    ]

    text = " ".join(output_tokens)

    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    text = text.replace(
        "<bos>",
        ""
    )

    text = text.replace(
        "<eos>",
        ""
    )

    return text.strip()

# ============================================================
# INTERACTIVE
# ============================================================

while True:

    prompt = input(
        "\nPrompt (blank quits): "
    )

    if not prompt.strip():
        break

    print()
    print(
        generate(
            prompt,
            max_tokens=100,
            temperature=0.9
        )
    )
