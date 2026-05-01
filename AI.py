from __future__ import annotations

import math, os, json, random
from typing import List, Optional, Tuple, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
VOCAB       = 1000000
D_MODEL     = 64
MAX_SEQ_LEN = 512
SEQ_LEN     = 16
DROPOUT     = 0.9
LR          = 6e-4
EPOCHS      = 2
KB_LEN      = -1
CHECKPOINT  = "v18.pt"
TOKENIZER_F = "v18_tokenizer.json"


# ═══════════════════════════════════════════════════════════════
#  PART 1: DATA SAMPLING ENGINE (atomic)
# ═══════════════════════════════════════════════════════════════

def sample_domain(n: int = 100, low: int = -10, high: int = 10) -> List[Tuple[int, int]]:
    """Generate n random (a,b) integer pairs."""
    return [(random.randint(low, high), random.randint(low, high)) for _ in range(n)]


def traces_to_tensors(traces):
    """Convert trace list to (x_tensor, y_tensor, z_tensor)."""
    xs, ys, zs = [], [], []
    for (x, y), z, _ in traces:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
        torch.tensor(zs, dtype=torch.float32),
    )


# ═══════════════════════════════════════════════════════════════
#  PART 2: AST GRAMMAR ENGINE (atomic)
# ═══════════════════════════════════════════════════════════════

class Expr:
    """Atomic AST node."""
    def __init__(self, op: str, *args):
        self.op = op
        self.args = args

    def __repr__(self):
        return f"{self.op}({','.join(map(str, self.args))})"


def all_constants():
    """Yield small integer constants as AST nodes."""
    for c in range(-2, 3):
        yield Expr("const", c)


# ═══════════════════════════════════════════════════════════════
#  PART 3: SCORING ENGINE (atomic) — stubs
# ═══════════════════════════════════════════════════════════════

def score_ast(expr, traces, ignore_name=None):
    return 0.0   # dummy

def robust_ast_search(traces_A, traces_B, min_accuracy=0.01, max_candidates=50):
    return []    # dummy


# ═══════════════════════════════════════════════════════════════
#  PART 4: NEURAL LOGIC UNIT (atomic)
# ═══════════════════════════════════════════════════════════════

class NALULike(nn.Module):
    """Learnable arithmetic logic unit."""
    def __init__(self):
        super().__init__()
        self.add_linear = nn.Linear(2, 1)
        self.mul_linear = nn.Linear(2, 1)
        self.add_gate = nn.Linear(2, 1)
        self.mul_gate = nn.Linear(2, 1)

    def forward(self, x, y):
        inp = torch.stack([x, y], dim=-1)
        a = self.add_linear(inp)
        m = self.mul_linear(inp)
        a_g = torch.sigmoid(self.add_gate(inp))
        m_g = torch.sigmoid(self.mul_gate(inp))
        return (a_g * a + m_g * m).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
#  PART 5: VECTOR MAGNET ENCODING (unchanged core)
# ═══════════════════════════════════════════════════════════════

class VectorMagnetEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512,
                 n_low: int = 130, n_mid: int = 40, n_high: int = 50,
                 cage_size: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_vecs = n_low + n_mid + n_high
        self.phase = nn.Parameter(torch.tensor(0.983))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))
        self.cage_size = cage_size   # half‑edge: cube spans [-cage_size, +cage_size]

        mags_low = torch.linspace(0.55, 1.8, n_low)
        axes_low = torch.linspace(1.0, -0.3, n_low)

        mags_mid = torch.linspace(1.9, 2.9, n_mid)
        axes_mid = torch.linspace(-0.2, -0.6, n_mid)

        mags_high = torch.linspace(3.0, 4.75, n_high)
        axes_high = torch.linspace(-0.7, -1.0, n_high)

        self.magnets = nn.Parameter(torch.cat([mags_low, mags_mid, mags_high], dim=0))
        self.axes = nn.Parameter(torch.cat([axes_low, axes_mid, axes_high], dim=0))

        self.proj = nn.Linear(2 * self.n_vecs, d_model)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        pos = pos_ids.float() + self.shift
        B, T = pos.shape

        # 1. Build 2D magnet‑axis activation map (bx, by)
        feats = []
        for i in range(self.n_vecs):
            w = self.magnets[i]
            a = self.axes[i]
            x = pos * w + self.phase + (i * 0.91)

            bx = torch.sin(x) * a
            by = torch.cos(x) * (1.0 - 0.1 * a)

            feats.append(-bx)
            feats.append(by)

        # Shape: [B, T, 2*n_vecs]
        feats = torch.stack(feats, dim=-1)

        # 2. fold into 3D‑like “x,y,z” coordinates (conceptually)
        n_3d = (feats.shape[-1] // 3) * 3
        xyz = feats[..., :n_3d].reshape(B, T, -1, 3)  # [..., stacks, 3]

        # 3. cubic cage mask: 1 inside, 0 outside
        max_feat = xyz.abs().max(dim=-1, keepdim=True)[0]
        xyz_norm = xyz / (max_feat + 1e-6)

        box_margin = 10.90
        inside = (xyz_norm.abs() <= 1 + box_margin).float()
        smooth_mask_per_point = inside.min(dim=-1, keepdim=True)[0]  # [..., stacks, 1]

        # 4. reshape back to 2D latent vector and apply mask
        mask_3d = smooth_mask_per_point.expand_as(xyz)
        mask_flat = mask_3d.reshape(B, T, n_3d)

        if feats.shape[-1] > n_3d:
            ones = torch.ones(B, T, feats.shape[-1] - n_3d, device=feats.device)
            mask_flat = torch.cat([mask_flat, ones], dim=-1)

        feats = torch.roll(feats, shifts=int(self.shift.item()) if self.shift.numel() == 1 else 1, dims=-1)
        feats = feats * mask_flat

        return self.proj(feats) * self.scale / math.sqrt(self.d_model)


# ═══════════════════════════════════════════════════════════════
#  PART 6: MINI‑KERNEL HEAD (per‑index latent kernels)
# ═══════════════════════════════════════════════════════════════

class MiniKernelHead(nn.Module):
    """
    Per‑sample mini‑kernel that modulates token embeddings.
    For each sequence (index) in a batch, it looks at a small
    window of that sequence's embeddings and outputs a
    [B, d_model] modulation vector.
    """
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.proj = nn.Sequential(
            nn.Linear(kernel_size * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Linear(kernel_size * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, d_model] — kernel context per sample.
        returns: [B, d_model] modulation per sample.
        """
        B, K, D = x.shape
        flat = x.reshape(B, K * D)
        gate = torch.sigmoid(self.gate(flat))
        mod  = self.proj(flat)
        return gate * mod


# ═══════════════════════════════════════════════════════════════
#  PART 7: TOKENIZER
# ═══════════════════════════════════════════════════════════════

class WordTokenizer:
    def __init__(self, vocab_size=VOCAB):
        self.vocab_size = vocab_size
        self.t2i = {"<pad>": 0, "　": 1}
        self.i2t = {0: "<pad>", 1: "　"}

    def _tok(self, text):
        w = text.lower().split()
        return [" ".join(w[i:i+3]) for i in range(len(w)-2)]

    def build(self, texts):
        freq = {}
        for t in texts:
            for tri in self._tok(t):
                freq[tri] = freq.get(tri, 0) + 1
        for k,_ in sorted(freq.items(), key=lambda x: -x[1])[:self.vocab_size-2]:
            idx = len(self.t2i)
            self.t2i[k] = idx
            self.i2t[idx] = k

    def encode(self, text):
        toks = self._tok(text[:KB_LEN])
        if len(toks) == 0:
            toks = ["　"]
        return torch.tensor([self.t2i.get(t, 1) for t in toks], dtype=torch.long)

    def decode(self, ids):
        out = []
        for i in ids:
            out.extend(self.i2t.get(int(i), "　").split())
        return " ".join(out)

    def save(self, p):
        json.dump(self.t2i, open(p, "w"))

    @staticmethod
    def load(p):
        t = WordTokenizer()
        t.t2i = json.load(open(p))
        t.i2t = {v: k for k, v in t.t2i.items()}
        return t


# ═══════════════════════════════════════════════════════════════
#  PART 8: V18 MODEL + MINI‑KERNEL INTEGRATION
# ═══════════════════════════════════════════════════════════════
class V18Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = VectorMagnetEncoding(
            d_model=D_MODEL, max_len=MAX_SEQ_LEN,
            n_low=2, n_mid=3, n_high=10   # your current settings
        )
        self.drop = nn.Dropout(DROPOUT)

        # per‑index mini‑kernel head
        self.k_head = MiniKernelHead(D_MODEL, kernel_size=4)

        self.tr = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.out = nn.Linear(D_MODEL, VOCAB)

        # <-- activation‑checkpointing flag (set to True to stunt training)
        self.grad_checkpoint = False

    def _forward_core(self, x_emb):
        """This is the part we will checkpoint."""
        x_emb = self.drop(x_emb)
        x_emb = self.tr(x_emb)
        return x_emb

    def forward(self, x, kernel_ctx: Optional[torch.Tensor] = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x_emb = self.tok(x) + self.pos(pos)

        if kernel_ctx is not None:
            mod = self.k_head(kernel_ctx)        # [B, D_MODEL]
            mod = mod.unsqueeze(1).expand(-1, T, -1)
            x_emb = x_emb + mod

        # Apply checkpointing only to the core (drop + trunk)
        if self.grad_checkpoint:
            # checkpoint will recompute _forward_core during backward
            x_emb = checkpoint(self._forward_core, x_emb, use_reentrant=False)
        else:
            x_emb = self._forward_core(x_emb)

        return self.out(x_emb)

    @torch.no_grad()
    def generate(
        self,
        x,
        steps: int = 450,
        use_minikernels: bool = True,
        kernel_ctx: Optional[torch.Tensor] = None,
    ):
        device = x.device
        B, T = x.shape
        for _ in range(steps):
            if _ % 5 and use_minikernels and kernel_ctx is None:
                pos = torch.arange(5, device=device).unsqueeze(0).expand(B, -1)
                emb = self.tok(x) + self.pos(pos)
                K = min(4, T)
                kernel_ctx = emb[:, :K, :].detach()
            logits = self(x, kernel_ctx=kernel_ctx if use_minikernels else None)
            probs = torch.softmax(logits[:, -1], dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)

        return x


# ═══════════════════════════════════════════════════════════════
#  PART 9: DATASET & COLLATE
# ═══════════════════════════════════════════════════════════════

class TextDS(Dataset):
    def __init__(self, t):
        self.t = t
    def __len__(self):
        return max(0, len(self.t) - SEQ_LEN - 1)
    def __getitem__(self, i):
        x = self.t[i:i + SEQ_LEN + 1]
        return x[:-1], x[1:]


def collate(batch):
    x = pad_sequence([b[0] for b in batch], True, 0)
    y = pad_sequence([b[1] for b in batch], True, 0)
    return x, y


# ─────── stub math_recognise so generate compiles ───────
def math_recognise(func: Callable,
                   nA: int = 20, nB: int = 10,
                   lowA: int = -15, highA: int = 15,
                   lowB: int = -6, highB: int = 6,
                   device: Optional[str] = None):
    best_ast = Expr("add", "a", "b")
    nalu = NALULike()
    style = "additive"
    return best_ast, nalu, style


# ═══════════════════════════════════════════════════════════════
#  PART 10: TRAIN — with per‑index mini‑kernels
# ═══════════════════════════════════════════════════════════════

def train(text):
    tok = WordTokenizer()
    tok.build([text])
    tokens = tok.encode(text)
    ds = TextDS(tokens)
    if len(ds) == 0:
        raise ValueError("Text too short for current SEQ_LEN")
    dl = DataLoader(ds, batch_size=16, collate_fn=collate, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = V18Model().to(device)

    # language model + mini‑kernels
    opt_lm = optim.Adam(m.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss(ignore_index=0)

    m.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            B, T = xb.shape

            # build kernel context per index (sequence) from the sequence itself
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            with torch.no_grad():
                emb = m.tok(xb) + m.pos(pos)
            K = min(4, T)
            k_ctx = emb[:, :K, :].detach()   # [B, K, D_MODEL]

            logits = m(xb, kernel_ctx=k_ctx)
            loss = ce(logits.reshape(-1, VOCAB), yb.reshape(-1))

            opt_lm.zero_grad()
            loss.backward()
            opt_lm.step()
            total_loss += loss.item()

        print(f"[LM+miniK] epoch {epoch+1}/{EPOCHS}, loss={total_loss/len(dl):.6f}")

    # math / NALU dummy setup (unchanged scaffolding)
    num_epochs = 5000
    nA = 20
    nB = 10
    lowA = -5
    highA = 5
    lowB = -6
    highB = 6

    def dummy_func(a, b):
        return a + b

    domain_A = sample_domain(n=nA, low=lowA, high=highA)
    domain_B = sample_domain(n=nB, low=lowB, high=highB)

    traces_A = [((a, b), dummy_func(a, b), "A") for a, b in domain_A]
    traces_B = [((a, b), dummy_func(a, b), "B") for a, b in domain_B]

    xA, yA, zA = traces_to_tensors(traces_A)
    xB, yB, zB = traces_to_tensors(traces_B)

    x = torch.cat([xA, xB]).to(device)
    y = torch.cat([yA, yB]).to(device)
    z = torch.cat([zA, zB]).to(device)

    nalu = NALULike().to(device)
    opt_nalu = optim.Adam(nalu.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    nalu.train()
    for epoch in range(num_epochs):
        opt_nalu.zero_grad()
        pred = nalu(x, y)
        loss = loss_fn(pred, z)
        loss.backward()
        opt_nalu.step()
        if epoch % 50 == 0:
            print(f"[NALU] epoch {epoch}, loss={loss.item():.6f}")

    torch.save(m.state_dict(), CHECKPOINT)
    tok.save(TOKENIZER_F)


# ═══════════════════════════════════════════════════════════════
#  PART 11: GENERATE — using mini‑kernels from the prompt
# ═══════════════════════════════════════════════════════════════

def generate(prompt: str,
             math_func: Optional[Callable] = None,
             steps: int = 180,
             use_minikernels: bool = True):
    """
    Generate text. If math_func is provided, run math_recognise
    and inject the inferred style as an ingredient.
    Mini‑kernels are built from the prompt as per‑index kernels.
    """
    tok = WordTokenizer.load(TOKENIZER_F)
    m = V18Model()
    m.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    m.eval()

    style = "neutral"
    if math_func is not None:
        _, _, style = math_recognise(
            math_func,
            nA=20, nB=10,
            lowA=-5, highA=5,
            lowB=-6, highB=6
        )
        print(f"→ math ingredient = {style}")

    ingredient_tag = f"ingredient={style}"
    augmented_prompt = f"{ingredient_tag} | user: {prompt}"

    ids = tok.encode(augmented_prompt).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = ids.to(device)
    m.to(device)

    # build kernel context from the prompt (acts as “index mini‑kernel”)
    with torch.no_grad():
        B, T = ids.shape
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        emb = m.tok(ids) + m.pos(pos)
        K = min(4, T)
        k_ctx = emb[:, :K, :].detach()

    out = m.generate(ids, steps=steps, use_minikernels=use_minikernels, kernel_ctx=k_ctx)
    generated = tok.decode(out[0].cpu())
    print(generated)


# ═══════════════════════════════════════════════════════════════
#  PART 12: GUI — math wheel + mini‑kernels
# ═══════════════════════════════════════════════════════════════

def gui():
    tok = WordTokenizer.load(TOKENIZER_F)
    m = V18Model()
    m.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    m.eval()

    def black_box(a, b):
        return a + b + 1   # ← change this to any function

    best_ast, nalu, style = math_recognise(
        black_box,
        nA=20, nB=10,
        lowA=-5, highA=5,
        lowB=-6, highB=6
    )
    print(f"WHEEL ready: style={style} (from {best_ast})")
    print("V18 GUI + MATH WHEEL + MINI‑KERNELS READY")
    print("Commands: q/quit to exit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)

    while True:
        s = input(">>> ").strip()
        if s.lower() in ("q", "quit"):
            break

        augmented = f"ingredient={style} | user: {s}"
        ids = tok.encode(augmented).unsqueeze(0).to(device)

        with torch.no_grad():
            B, T = ids.shape
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            emb = m.tok(ids) + m.pos(pos)
            K = min(4, T)
            k_ctx = emb[:, :K, :].detach()

        out = m.generate(ids, 280, use_minikernels=True, kernel_ctx=k_ctx)
        print(tok.decode(out[0].cpu()))


# ═══════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    while True:
        c = input("(t)rain (g)en (i)gui > ").strip().lower()
        if c == "t":
            path = input("File: ")
            text = open(path, encoding="utf-8").read()
            train(text)
        if c == "g":
            prompt = input("prompt: ")
            generate(prompt)
        if c == "i":
            gui()
