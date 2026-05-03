import re
import json
import os
import shutil
import tempfile
import threading
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gradio as gr
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import HfApi, whoami, hf_hub_download

# ── Constants ─────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 500_000_000
SEQ_LEN     = 16
EMBED_DIM   = 128
HIDDEN_DIM  = 256
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3

MODEL_FILE     = "simple_lm.pt"
TOKENIZER_FILE = "simple_tokenizer.json"

# Accepted MIME types / extensions for training file uploads
ACCEPTED_EXTENSIONS = [".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".rst", ".text"]

# ── Text helpers ───────────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def length_similarity(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    if la == 0 and lb == 0:
        return 1.0
    return 1.0 - abs(la - lb) / max(la, lb, 1)


def cosine_length_sort_sentences(sentences: List[str], length_weight: float = 0.35) -> List[str]:
    if len(sentences) <= 1:
        return sentences
    vocab: dict = {}
    for s in sentences:
        for w in s.lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)

    def bow(text: str) -> List[float]:
        vec = [0.9] * len(vocab)
        for w in text.lower().split():
            if w in vocab:
                vec[vocab[w]] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    vecs = [bow(s) for s in sentences]
    ref  = vecs[0]

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    scored = []
    for s, v in zip(sentences, vecs):
        cos   = cosine(v, ref)
        ls    = length_similarity(s, sentences[0])
        score = (1.0 - length_weight) * cos + length_weight * ls
        scored.append((score, s))

    scored.sort(key=lambda x: x[0])
    return [s for _, s in scored]


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.t2i = {"<pad>": 0, "<unk>": 1}
        self.i2t = {0: "<pad>", 1: "<unk>"}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build(self, texts: List[str]):
        freq: dict = {}
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


# ── Model ─────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int = SEQ_LEN):
        self.token_ids = token_ids
        self.seq_len   = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.rnn(x)
        return self.fc(out)

    @torch.no_grad()
    def generate(self, start_ids: List[int], max_new_tokens: int = 30, device: str = "cpu") -> List[int]:
        self.eval()
        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new_tokens):
            logits  = self(x)
            probs   = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x       = torch.cat([x, next_id], dim=1)
        return x[0].tolist()


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_all(model: SimpleLM, tokenizer: Tokenizer):
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size":  len(tokenizer.t2i),
            "embed_dim":   EMBED_DIM,
            "hidden_dim":  HIDDEN_DIM,
            "seq_len":     SEQ_LEN,
        },
        MODEL_FILE,
    )
    tokenizer.save(TOKENIZER_FILE)


def load_all(device: str = "cpu"):
    tokenizer = Tokenizer.load(TOKENIZER_FILE)
    ckpt      = torch.load(MODEL_FILE, map_location=device)
    model     = SimpleLM(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def ranked_generate(
    model: SimpleLM,
    tokenizer: Tokenizer,
    prompt: str,
    length_weight: float = 0.35,
    n_samples: int = 5,
    max_new_tokens: int = 30,
    device: str = "cpu",
) -> str:
    start_ids  = tokenizer.encode(prompt) or [tokenizer.t2i["<unk>"]]
    prompt_len = len(prompt.split())

    completions = []
    for _ in range(n_samples):
        out_ids    = model.generate(start_ids, max_new_tokens=max_new_tokens, device=device)
        completion = tokenizer.decode(out_ids)
        completions.append(completion)

    def score(c: str) -> float:
        c_len = len(c.split())
        if prompt_len == 0 and c_len == 0:
            return 1.0
        return 1.0 - abs(c_len - prompt_len) / max(c_len, prompt_len, 1)

    scored = [(score(c), c) for c in completions]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


# ── Training ──────────────────────────────────────────────────────────────────

def train_on_text(text: str, epochs: int, lr: float, batch_size: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "❌ No sentences found in the provided text."

    orig_text   = " ".join(sentences)
    sorted_text = " ".join(cosine_length_sort_sentences(sentences))

    tokenizer = Tokenizer()
    tokenizer.build([orig_text])
    token_ids = tokenizer.encode(sorted_text)

    if len(token_ids) <= SEQ_LEN:
        return f"❌ Text too short — need more than {SEQ_LEN} tokens."

    dataset   = TextDataset(token_ids, seq_len=SEQ_LEN)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = SimpleLM(vocab_size=len(tokenizer.t2i)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    log_lines = [f"Training on {device} | vocab={len(tokenizer.t2i):,} | tokens={len(token_ids):,}"]

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            loss    = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        log_lines.append(f"Epoch {epoch+1}/{epochs} — loss: {avg:.6f}")
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg:.6f}")

    save_all(model, tokenizer)
    log_lines.append(f"✅ Saved → {MODEL_FILE}, {TOKENIZER_FILE}")
    return "\n".join(log_lines)


# ── File upload helper ────────────────────────────────────────────────────────

def _extract_text_from_jsonl(raw: str) -> str:
    """Best-effort: concatenate all string values from each JSON-lines row."""
    lines_out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                parts = [str(v) for v in obj.values() if isinstance(v, str) and v.strip()]
                lines_out.append(" ".join(parts))
            elif isinstance(obj, str):
                lines_out.append(obj)
        except json.JSONDecodeError:
            lines_out.append(line)
    return "\n".join(lines_out)


def _extract_text_from_json(raw: str) -> str:
    """Flatten a JSON array / object into plain text strings."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # fall back to raw

    parts: List[str] = []

    def collect(node):
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(data)
    return "\n".join(p for p in parts if p.strip())


def load_uploaded_file(file_obj) -> tuple[str, str]:
    """
    Read an uploaded file and return (text_content, status_message).
    Supports .txt / .md / .rst / .text / .csv / .tsv / .json / .jsonl
    """
    if file_obj is None:
        return "", "No file uploaded."

    path = file_obj if isinstance(file_obj, str) else file_obj.name
    ext  = os.path.splitext(path)[1].lower()

    if ext not in ACCEPTED_EXTENSIONS:
        return (
            "",
            f"❌ Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(ACCEPTED_EXTENSIONS)}",
        )

    try:
        # Try UTF-8 first, fall back to latin-1
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    raw = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            return "", "❌ Could not decode file — please use a UTF-8 encoded text file."

        # Format-specific extraction
        if ext == ".jsonl":
            text = _extract_text_from_jsonl(raw)
        elif ext == ".json":
            text = _extract_text_from_json(raw)
        elif ext in (".csv", ".tsv"):
            # Strip delimiter columns; keep all cell text concatenated per row
            sep = "\t" if ext == ".tsv" else ","
            rows = []
            for line in raw.splitlines():
                cells = [c.strip().strip('"') for c in line.split(sep)]
                row_text = " ".join(c for c in cells if c)
                if row_text:
                    rows.append(row_text)
            text = "\n".join(rows)
        else:
            text = raw  # plain text variants

        n_chars = len(text)
        n_words = len(text.split())
        fname   = os.path.basename(path)
        status  = (
            f"✅ Loaded '{fname}'  —  "
            f"{n_chars:,} chars · {n_words:,} words"
        )
        return text, status

    except Exception as e:
        return "", f"❌ Error reading file: {e}"


# ── HuggingFace Dataset helpers ───────────────────────────────────────────────

def fetch_hf_configs(dataset_name: str):
    dataset_name = dataset_name.strip()
    if not dataset_name:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), "Enter a dataset name first."
    try:
        configs = get_dataset_config_names(dataset_name) or ["default"]
        return (
            gr.update(choices=configs, value=configs[0]),
            gr.update(choices=[], value=None),
            f"Found {len(configs)} config(s).",
        )
    except Exception as e:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), f"❌ {e}"


def fetch_hf_fields(dataset_name: str, config: str, split: str = "train"):
    dataset_name = dataset_name.strip()
    if not dataset_name or not config:
        return gr.update(choices=[], value=None), "Provide dataset name and config."
    try:
        cfg     = None if config in ("default", "") else config
        ds      = load_dataset(dataset_name, cfg, split=split, streaming=True, trust_remote_code=False)
        example = next(iter(ds))
        str_fields = [k for k, v in example.items() if isinstance(v, str)] or list(example.keys())
        return gr.update(choices=str_fields, value=str_fields[0] if str_fields else None), f"Fields: {str_fields}"
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ {e}"


def load_hf_text(dataset_name: str, config: str, split: str, text_field: str, max_samples: int) -> str:
    dataset_name = dataset_name.strip()
    if not dataset_name or not text_field:
        return ""
    try:
        cfg   = None if config in ("default", "") else config
        ds    = load_dataset(dataset_name, cfg, split=split, streaming=True, trust_remote_code=False)
        texts = []
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            val = ex.get(text_field, "")
            if isinstance(val, str) and val.strip():
                texts.append(val.strip())
        return "\n\n".join(texts)
    except Exception as e:
        return f"❌ Error loading dataset: {e}"


def ui_load_hf_and_preview(dataset_name, config, split, text_field, max_samples):
    text    = load_hf_text(dataset_name, config, split, text_field, int(max_samples))
    preview = text[:2000] + ("\n…(truncated)" if len(text) > 2000 else "")
    return text, preview


# ── HuggingFace Hub push / pull ───────────────────────────────────────────────

def validate_token(hf_token: str) -> str:
    hf_token = hf_token.strip()
    if not hf_token:
        return "—"
    try:
        info = whoami(token=hf_token)
        return f"✅ Logged in as: {info['name']}"
    except Exception as e:
        return f"❌ Invalid token: {e}"


def push_to_hub(repo_id: str, hf_token: str, commit_message: str, private: bool) -> str:
    repo_id        = repo_id.strip()
    hf_token       = hf_token.strip()
    commit_message = commit_message.strip() or "Upload SimpleLM checkpoint"

    if not repo_id:
        return "❌ Provide a repo ID, e.g.  your-username/my-simple-lm"
    if not hf_token:
        return "❌ Provide a HuggingFace token with write access."
    if not os.path.exists(MODEL_FILE) or not os.path.exists(TOKENIZER_FILE):
        return "❌ No saved model found locally — train first."

    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

        card = f"""---
language: en
tags:
  - text-generation
  - gru
  - pytorch
---
# SimpleLM — {repo_id}

A compact GRU language model trained with [SimpleLM](https://github.com/).

## Files
| File | Description |
|------|-------------|
| `{MODEL_FILE}` | PyTorch checkpoint (weights + arch config) |
| `{TOKENIZER_FILE}` | Word-level vocabulary (token → id) |

## Architecture
| Param | Value |
|-------|-------|
| Embedding dim | {EMBED_DIM} |
| Hidden dim | {HIDDEN_DIM} |
| Sequence length | {SEQ_LEN} |

## Usage
```python
from app import load_all, ranked_generate
model, tokenizer = load_all()
print(ranked_generate(model, tokenizer, "The quick brown"))
```
"""
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(card)
            card_path = f.name

        uploads = [
            (MODEL_FILE,     MODEL_FILE),
            (TOKENIZER_FILE, TOKENIZER_FILE),
            (card_path,      "README.md"),
        ]
        for local, remote in uploads:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
        os.unlink(card_path)

        visibility = "private" if private else "public"
        url = f"https://huggingface.co/{repo_id}"
        return (
            f"✅ Pushed to {url}  [{visibility}]\n\n"
            f"Uploaded:\n"
            f"  • {MODEL_FILE}\n"
            f"  • {TOKENIZER_FILE}\n"
            f"  • README.md (model card)"
        )
    except Exception as e:
        return f"❌ Upload failed: {e}"


def pull_from_hub(repo_id: str, hf_token: str) -> str:
    global _model, _tokenizer
    repo_id  = repo_id.strip()
    hf_token = hf_token.strip() or None
    if not repo_id:
        return "❌ Provide a repo ID."
    try:
        kwargs     = dict(repo_id=repo_id, repo_type="model", token=hf_token)
        model_path = hf_hub_download(filename=MODEL_FILE,     **kwargs)
        tok_path   = hf_hub_download(filename=TOKENIZER_FILE, **kwargs)
        shutil.copy(model_path, MODEL_FILE)
        shutil.copy(tok_path,   TOKENIZER_FILE)
        _model, _tokenizer = load_all(device=DEVICE)
        n_params = sum(p.numel() for p in _model.parameters())
        return (
            f"✅ Loaded from {repo_id}\n"
            f"Vocab: {len(_tokenizer.t2i):,} tokens | Params: {n_params:,}"
        )
    except Exception as e:
        return f"❌ Download failed: {e}"


# ── App state ─────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model: SimpleLM | None      = None
_tokenizer: Tokenizer | None = None


def _try_load():
    global _model, _tokenizer
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        _model, _tokenizer = load_all(device=DEVICE)
        return "✅ Existing model loaded."
    return "No saved model found yet."


_startup_msg = _try_load()


def ui_train(text: str, epochs: int, lr: float, batch_size: int):
    global _model, _tokenizer
    if not text or not text.strip():
        return "❌ No text provided."
    result = train_on_text(text, int(epochs), float(lr), int(batch_size))
    if "✅" in result:
        _model, _tokenizer = load_all(device=DEVICE)
    return result


def ui_generate(prompt: str, max_new_tokens: int, n_samples: int):
    if _model is None or _tokenizer is None:
        return "❌ No model loaded. Train one first."
    if not prompt.strip():
        return "❌ Enter a prompt."
    return ranked_generate(
        _model, _tokenizer, prompt,
        n_samples=int(n_samples),
        max_new_tokens=int(max_new_tokens),
        device=DEVICE,
    )


def model_info() -> str:
    lines = [f"Device: {DEVICE}"]
    if _model is not None and _tokenizer is not None:
        n_params = sum(p.numel() for p in _model.parameters())
        lines += [
            f"Vocab size:  {len(_tokenizer.t2i):,}",
            f"Parameters:  {n_params:,}",
            f"Embed dim:   {EMBED_DIM}",
            f"Hidden dim:  {HIDDEN_DIM}",
            f"Seq length:  {SEQ_LEN}",
        ]
    else:
        lines.append("No model loaded yet.")
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="SimpleLM — Train & Generate",
    theme=gr.themes.Base(
        primary_hue="slate",
        secondary_hue="zinc",
        neutral_hue="zinc",
        font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ),
    css="""
    .startup-msg { font-size: 0.8rem; color: #71717a; }
    .upload-status { font-size: 0.8rem; }
    """,
) as demo:

    gr.Markdown(
        """
# 🧠 SimpleLM — GRU Language Model
Train a compact GRU language model on your own text or a Hugging Face dataset, then generate completions.
        """
    )
    gr.Markdown(f"**Runtime:** `{DEVICE.upper()}` | {_startup_msg}", elem_classes="startup-msg")

    with gr.Tabs():

        # ── Tab 1: HuggingFace Dataset ─────────────────────────────────────
        with gr.Tab("📦 HF Dataset"):
            gr.Markdown("### Load text from a Hugging Face dataset")
            with gr.Row():
                hf_name  = gr.Textbox(label="Dataset name", placeholder="wikitext", scale=3)
                hf_split = gr.Textbox(label="Split", value="train", scale=1)
            with gr.Row():
                btn_configs = gr.Button("1 — Fetch configs", variant="secondary")
                hf_status   = gr.Textbox(label="Status", interactive=False, scale=3)
            with gr.Row():
                hf_config  = gr.Dropdown(label="Config", choices=[], interactive=True, scale=2)
                btn_fields = gr.Button("2 — Fetch fields", variant="secondary", scale=1)
            hf_field    = gr.Dropdown(label="Text field", choices=[], interactive=True)
            hf_max      = gr.Slider(label="Max samples", minimum=10, maximum=5000, value=500, step=10)
            btn_load_hf = gr.Button("3 — Load text into editor ⬇", variant="primary")
            hf_preview  = gr.Textbox(label="Preview (first 2 000 chars)", lines=6, interactive=False)
            hf_full     = gr.State("")

            btn_configs.click(fetch_hf_configs, [hf_name], [hf_config, hf_field, hf_status])
            btn_fields.click(fetch_hf_fields, [hf_name, hf_config, hf_split], [hf_field, hf_status])
            btn_load_hf.click(ui_load_hf_and_preview, [hf_name, hf_config, hf_split, hf_field, hf_max], [hf_full, hf_preview])

        # ── Tab 2: Train ────────────────────────────────────────────────────
        with gr.Tab("🏋 Train"):
            gr.Markdown("### Train on text — paste, upload a file, or load from the HF Dataset tab")

            # ── File upload section ──────────────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "#### 📂 Upload a training file\n"
                    f"_Accepted formats: {', '.join(ACCEPTED_EXTENSIONS)}_"
                )
                with gr.Row():
                    upload_file   = gr.File(
                        label="Drop or click to upload",
                        file_types=ACCEPTED_EXTENSIONS,
                        file_count="single",
                        scale=4,
                    )
                    btn_load_file = gr.Button("Load file into editor ⬇", variant="primary", scale=1)
                upload_status = gr.Textbox(
                    label="File status",
                    interactive=False,
                    value="—",
                    elem_classes="upload-status",
                )

            gr.Markdown("---")

            # ── Text editor ─────────────────────────────────────────────────
            train_text = gr.Textbox(label="Training text", placeholder="Paste text here…", lines=12)

            with gr.Row():
                btn_use_hf   = gr.Button("⬆  Use HuggingFace text loaded above", variant="secondary")
                btn_clear    = gr.Button("🗑  Clear editor", variant="secondary")

            btn_use_hf.click(lambda t: t, [hf_full], [train_text])
            btn_clear.click(lambda: ("", "—"), outputs=[train_text, upload_status])

            # Wire file upload → text area
            btn_load_file.click(
                load_uploaded_file,
                inputs=[upload_file],
                outputs=[train_text, upload_status],
            )
            # Also auto-load when a file is selected (convenience)
            upload_file.change(
                load_uploaded_file,
                inputs=[upload_file],
                outputs=[train_text, upload_status],
            )

            gr.Markdown("---")

            # ── Hyperparams + train ─────────────────────────────────────────
            with gr.Row():
                t_epochs = gr.Slider(label="Epochs",     minimum=1,    maximum=50,  value=EPOCHS,     step=1)
                t_lr     = gr.Slider(label="LR",         minimum=1e-5, maximum=1e-1, value=LR,        step=1e-5)
                t_bs     = gr.Slider(label="Batch size", minimum=8,    maximum=256, value=BATCH_SIZE,  step=8)
            btn_train = gr.Button("Train 🚀", variant="primary")
            train_log = gr.Textbox(label="Training log", lines=14, interactive=False)
            btn_train.click(ui_train, [train_text, t_epochs, t_lr, t_bs], [train_log])

        # ── Tab 3: Generate ─────────────────────────────────────────────────
        with gr.Tab("✨ Generate"):
            gr.Markdown("### Generate text completions")
            gen_prompt = gr.Textbox(label="Prompt", placeholder="The quick brown fox…", lines=2)
            with gr.Row():
                gen_tokens  = gr.Slider(label="Max new tokens",    minimum=10, maximum=300, value=60, step=5)
                gen_samples = gr.Slider(label="Samples (ranked)",  minimum=1,  maximum=20,  value=5,  step=1)
            btn_gen    = gr.Button("Generate ✨", variant="primary")
            gen_output = gr.Textbox(label="Completion", lines=6, interactive=False)
            btn_gen.click(ui_generate, [gen_prompt, gen_tokens, gen_samples], [gen_output])

        # ── Tab 4: HuggingFace Hub ──────────────────────────────────────────
        with gr.Tab("🤗 HF Hub"):
            gr.Markdown("### Push your trained model to the Hugging Face Hub")

            # ── Token section ──
            with gr.Group():
                gr.Markdown("#### 🔑 Authentication")
                with gr.Row():
                    hub_token    = gr.Textbox(
                        label="HF Token (write access)",
                        placeholder="hf_…",
                        type="password",
                        scale=4,
                    )
                    btn_validate = gr.Button("Validate token", variant="secondary", scale=1)
                token_status = gr.Textbox(label="Token status", interactive=False, value="—")
                btn_validate.click(validate_token, [hub_token], [token_status])
                gr.Markdown(
                    "_Get a write token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)_"
                )

            gr.Markdown("---")

            # ── Push section ──
            with gr.Group():
                gr.Markdown("#### ⬆  Push model")
                with gr.Row():
                    push_repo = gr.Textbox(
                        label="Repo ID",
                        placeholder="your-username/my-simple-lm",
                        scale=4,
                    )
                    push_private = gr.Checkbox(label="Private repo", value=False, scale=1)
                push_commit = gr.Textbox(label="Commit message", value="Upload SimpleLM checkpoint")
                btn_push    = gr.Button("Push to Hub ⬆", variant="primary")
                push_status = gr.Textbox(label="Push status", lines=5, interactive=False)
                btn_push.click(push_to_hub, [push_repo, hub_token, push_commit, push_private], [push_status])

            gr.Markdown("---")

            # ── Pull section ──
            with gr.Group():
                gr.Markdown("#### ⬇  Load model from Hub")
                pull_repo   = gr.Textbox(label="Repo ID", placeholder="your-username/my-simple-lm")
                btn_pull    = gr.Button("Load from Hub ⬇", variant="secondary")
                pull_status = gr.Textbox(label="Load status", lines=4, interactive=False)
                btn_pull.click(pull_from_hub, [pull_repo, hub_token], [pull_status])

        # ── Tab 5: Model info ───────────────────────────────────────────────
        with gr.Tab("ℹ Model info"):
            info_box = gr.Textbox(label="Model stats", lines=8, interactive=False, value=model_info())
            gr.Button("Refresh").click(model_info, outputs=[info_box])


if __name__ == "__main__":
    demo.launch(share=False)
