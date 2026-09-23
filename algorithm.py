"""CUDA-first stochastic polymorphic number generator and bijective automorphism filter.

Run:
  pip install torch gradio plotly
  python app_cuda_automorphism.py
  python app_cuda_automorphism.py --share

Transforms vector spaces via a bijective, isometric orthogonal automorphism
(matrix exponential of a skew-symmetric generator) before computing GPU similarity
and subset-sum routing over compact polymorphic numbers.
"""
from __future__ import annotations
import argparse, json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import plotly.graph_objects as go

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = DEVICE.type == 'cuda'
BOS, EOS = '<bos>', '<eos>'
SPECIAL = {BOS, EOS}
MODEL_PATH = 'model_auto.json'
BINDINGS_PATH = 'bindings_auto.json'
DEFAULT_CORPUS = 'corpus_auto.txt'

ALPHA = .05; TEMP = .8; TOP_K = 20; MAX_NEW = 800; MAX_SUBSET = 5; BEAM = 24
Vec = Dict[str, float]

def tokenize(text):
    # Tokenize numbers, hex codes, or polymorphic symbols
    return text.lower().split()

def sentences(text):
    out = []
    for x in text.split("."):
        x = x.strip()
        if x:
            out.append(x+".")
    return out

def bow(tokens):
    return Counter(x for x in tokens if x not in SPECIAL)

def matrix(vectors, keys):
    return torch.tensor([[v.get(k, 0.) for k in keys] for v in vectors], dtype=torch.float32, device=DEVICE)

def gpu_cosine(query: Vec, vectors: List[Vec]):
    keys = sorted(set(query).union(*(v.keys() for v in vectors))) if vectors else []
    if not keys: return torch.zeros(len(vectors), device=DEVICE)
    q = torch.tensor([query.get(k, 0.) for k in keys], dtype=torch.float32, device=DEVICE)
    x = matrix(vectors, keys)
    return (x @ q) / (torch.linalg.vector_norm(x, dim=1) * torch.linalg.vector_norm(q)).clamp_min(1e-12)

def gpu_gain(target: Vec, vectors: List[Vec]):
    keys = sorted(target)
    if not keys: return torch.zeros(len(vectors), device=DEVICE)
    t = torch.tensor([target[k] for k in keys], dtype=torch.float32, device=DEVICE)
    return torch.minimum(matrix(vectors, keys), t).sum(1)

def gpu_automorphism_matrix(vectors, theta=0.5):
    """
    Transforms the vector space via a bijective, isometric automorphism
    before computing the similarity matrix.

    Constructs an orthogonal rotation matrix Q via matrix exponential:
    Q = exp(theta * (A - A^T)), where A is a random seed matrix.
    Since Q^T Q = I, the mapping x -> xQ is strictly bijective and invertible.
    """
    if not vectors:
        return torch.empty((0, 0), device=DEVICE)

    keys = sorted(set().union(*(v.keys() for v in vectors)))
    x = matrix(vectors, keys)
    num_vectors, num_keys = x.shape

    if num_keys > 1:
        torch.manual_seed(42)
        A = torch.randn(num_keys, num_keys, device=DEVICE) * 0.01
        A = A - A.T  # Skew-symmetric generator

        try:
            Q = torch.linalg.matrix_exp(theta * A)
        except AttributeError:
            Q = torch.eye(num_keys, device=DEVICE) + theta * A + (theta**2 / 2.0) * (A @ A)

        x_transformed = x @ Q
    else:
        x_transformed = x

    n = torch.linalg.vector_norm(x_transformed, dim=1, keepdim=True).clamp_min(1e-12)
    return (x_transformed @ x_transformed.T) / (n @ n.T)

class Bindings:
    def __init__(self, threshold=.35, momentum=.85):
        self.threshold = threshold
        self.momentum = momentum
        self.contexts = {}
        self.links = {}

    def update(self, text, incremental=False):
        incoming = defaultdict(Counter)
        for s in sentences(text):
            t = tokenize(s)
            for i, w in enumerate(t):
                incoming[w].update(t[max(0, i-4):min(len(t), i+5)])
        if incremental and self.contexts:
            merged = defaultdict(Counter)
            for w, v in self.contexts.items(): merged[w].update(v)
            for w, v in incoming.items():
                for k, x in v.items():
                    merged[w][k] = self.momentum * merged[w][k] + (1 - self.momentum) * x
            self.contexts = dict(merged)
        else:
            self.contexts = dict(incoming)
        self.rebuild()

    def rebuild(self):
        words = list(self.contexts)
        vectors = [{k: float(x) for k, x in self.contexts[w].items()} for w in words]
        if not words:
            self.links = {}
            return
        # Utilizing the Bijective Automorphism Matrix computation on CUDA
        scores = gpu_automorphism_matrix(vectors)
        links = {}
        for i, w in enumerate(words):
            row = scores[i].clone()
            row[i] = -1
            values, idx = torch.topk(row, k=min(8, len(words) - 1))
            links[w] = {words[int(j)]: float(v) for v, j in zip(values.detach().cpu(), idx.detach().cpu()) if float(v) >= self.threshold}
        self.links = links

    def expand(self, v):
        out = defaultdict(float)
        for w, x in v.items():
            out[w] += x
            for z, s in self.links.get(w, {}).items():
                out[z] += x * s
        return dict(out)

    def save(self):
        Path(BINDINGS_PATH).write_text(json.dumps({
            'threshold': self.threshold, 'momentum': self.momentum,
            'links': self.links, 'contexts': {w: dict(v) for w, v in self.contexts.items()}
        }, indent=2), encoding='utf8')

    @classmethod
    def load(cls):
        x = cls()
        if not Path(BINDINGS_PATH).exists(): return x
        d = json.loads(Path(BINDINGS_PATH).read_text(encoding='utf8'))
        x.threshold = d.get('threshold', .35)
        x.momentum = d.get('momentum', .85)
        x.links = d.get('links', {})
        x.contexts = {w: Counter(v) for w, v in d.get('contexts', {}).items()}
        return x

    def summary(self):
        return f'Device: {DEVICE}\nBijective Automorphism Nodes: {len(self.links)}\nActive Tensor Links: {sum(len(x) for x in self.links.values())}'

class Ref:
    def __init__(self, i, s, t, freq, features):
        self.i = i; self.s = s; self.t = t; self.freq = freq; self.f = features

class Match:
    def __init__(self, rank, ref, score, vector, selected, endpoint):
        self.rank = rank; self.ref = ref; self.score = score; self.vector = vector; self.selected = selected; self.endpoint = endpoint

class Corpus:
    def __init__(self, text='', bindings=None):
        self.bindings = bindings or Bindings()
        if text: self.bindings.update(text)
        ss = sentences(text)
        freq = Counter(s.lower() for s in ss)
        self.refs = []
        for i, s in enumerate(ss):
            t = tokenize(s)
            f = self.bindings.expand({k: float(v) for k, v in bow(t).items()})
            self.refs.append(Ref(i, s, t, freq[s.lower()], f))

    def search(self, prompt, limit=5):
        target = self.bindings.expand({k: float(v) for k, v in bow(tokenize(prompt)).items()})
        vectors = [r.f for r in self.refs]
        scores = gpu_cosine(target, vectors)
        gains = gpu_gain(target, vectors)
        chosen = []
        covered = defaultdict(float)
        for _ in range(min(MAX_SUBSET, len(self.refs))):
            residual = {k: max(0., v - covered[k]) for k, v in target.items()}
            rg = gpu_gain(residual, vectors)
            rg[[i for i in range(len(self.refs)) if i in chosen]] = 0
            j = int(torch.argmax(rg).item())
            if float(rg[j]) <= 0: break
            chosen.append(j)
            for k, v in self.refs[j].f.items(): covered[k] += v
        rows = []
        for i, r in enumerate(self.refs):
            if i in chosen or float(gains[i]) > 0:
                overlap = len(set(tokenize(prompt)) & set(r.t)) / max(1, len(set(tokenize(prompt)) | set(r.t)))
                score = .7 * float(scores[i]) + .2 * overlap + .1 * math.log1p(r.freq)
                rows.append(Match(0, r, score, float(scores[i]), i in chosen, i not in chosen))
        rows.sort(key=lambda x: x.score, reverse=True)
        for i, x in enumerate(rows[:limit], 1): x.rank = i
        return rows[:limit]

class NGram:
    def __init__(self):
        self.vocab = {}; self.inv = []; self.uni = Counter(); self.bi = defaultdict(Counter); self.tri = defaultdict(Counter)

    def ingest(self, text):
        for s in sentences(text):
            q = [BOS, BOS, *tokenize(s), EOS]
            self.uni.update(q)
            for a, b in zip(q, q[1:]): self.bi[a][b] += 1
            for a, b, c in zip(q, q[1:], q[2:]): self.tri[(a, b)][c] += 1
        self.inv = sorted(self.uni)
        self.vocab = {w: i for i, w in enumerate(self.inv)}
        return self

    def gpu_logits(self, history):
        a, b = history[-2:]
        counts = self.tri.get((a, b)) or self.bi.get(b) or self.uni
        v = torch.full((len(self.inv),), -float('inf'), device=DEVICE)
        total = sum(counts.values()) + ALPHA * len(counts)
        for w, c in counts.items():
            v[self.vocab[w]] = math.log((c + ALPHA) / total)
        return v

    def bias_from(self, seeds):
        out = {}
        for w in seeds:
            c = self.bi.get(w)
            total = sum(c.values()) if c else 0
            if total:
                for z, n in c.items(): out[z] = out.get(z, 0.) + n / total
        return out

    def sample(self, history, bias=None):
        logits = self.gpu_logits(history)
        if bias:
            for w, x in bias.items():
                if w in self.vocab: logits[self.vocab[w]] += x
        vals, ids = torch.topk(logits, k=min(TOP_K, len(logits)))
        p = torch.softmax(vals / TEMP, 0)
        return self.inv[int(ids[torch.multinomial(p, 1)].item())]

    def generate(self, prompt, biasfn=None):
        out = [BOS, BOS, *tokenize(prompt)]
        for i in range(MAX_NEW):
            b = biasfn(out) if biasfn and i % 8 == 0 else None
            x = self.sample(out, b)
            out.append(x)
        return ' '.join(x for x in out if x not in SPECIAL)

    def save(self):
        Path(MODEL_PATH).write_text(json.dumps({
            'vocab': self.inv, 'uni': dict(self.uni),
            'bi': {str(k): dict(v) for k, v in self.bi.items()},
            'tri': {str(k): dict(v) for k, v in self.tri.items()}
        }, indent=2), encoding='utf8')

NGram.model_bias = lambda self, seeds: self.bias_from(seeds) if hasattr(self, 'bias_from') else {}


# ---------------------------------------------------------------------------
# Sparse Manhattan-distance connectome
#
# Reuses the same context vectors already built by Bindings (token -> Counter
# of co-occurring tokens). Computes a dense pairwise L1 (Manhattan) distance
# matrix on GPU, then sparsifies each node down to its k-nearest neighbours
# (a "connectome" rather than a dense similarity matrix).
#
# Axes:
#   x = log1p(token frequency), normalized 0..1  (dataset frequency)
#   y = mean L1 distance to a node's kept neighbours, normalized 0..1
#       (complementary to x: hub words cluster low, isolated words sit high)
#   z = log(sorted(exp(d))) of each node's own kept-neighbour distance vector.
#       Note: log and exp are exact inverses, so this is algebraically
#       identical to sorted(d) -- implemented literally rather than swapped
#       out, with the median of the sorted vector taken as the node's z value.
#
# Marginal peak envelopes: histogram x and y separately, find each bin that
# is a local maximum, and connect those peaks with straight (piecewise
# linear) segments -- drawn as extra traces projected onto the z=0 floor.
# ---------------------------------------------------------------------------

def _local_peaks(values: List[float], bins: int = 16):
    if not values:
        return [], 0.0, 1.0
    mn, mx = min(values), max(values)
    span = (mx - mn) or 1.0
    hist = [0] * bins
    for v in values:
        b = min(bins - 1, int((v - mn) / span * bins))
        hist[b] += 1
    peaks = []
    for i in range(bins):
        prev = hist[i - 1] if i > 0 else -1
        nxt = hist[i + 1] if i < bins - 1 else -1
        if hist[i] >= prev and hist[i] >= nxt and hist[i] > 0:
            peaks.append((mn + (i + 0.5) / bins * span, hist[i]))
    return peaks, mn, mx

def build_connectome(bindings: 'Bindings', vocab_size: int = 45, k: int = 3) -> Tuple[go.Figure, str]:
    if not bindings.contexts:
        return go.Figure(), 'Train the filter first (no contexts to build a connectome from).'

    words_all = list(bindings.contexts)
    freq_all = {w: sum(bindings.contexts[w].values()) for w in words_all}
    words = sorted(words_all, key=lambda w: freq_all[w], reverse=True)[:vocab_size]
    wset = set(words)

    # context vectors restricted to the chosen vocabulary, on GPU
    keys = words
    vecs = torch.tensor(
        [[float(bindings.contexts[w].get(k2, 0.)) for k2 in keys] for w in words],
        dtype=torch.float32, device=DEVICE,
    )

    # dense pairwise Manhattan distance, computed on GPU
    dist = torch.cdist(vecs, vecs, p=1.0)
    n = len(words)
    kk = min(k, max(1, n - 1))

    dist_filled = dist.clone()
    dist_filled.fill_diagonal_(float('inf'))
    nn_vals, nn_idx = torch.topk(dist_filled, k=kk, largest=False, dim=1)

    edges = set()
    for i in range(n):
        for j in nn_idx[i].tolist():
            key = (i, j) if i < j else (j, i)
            edges.add(key)

    freq = [freq_all[w] for w in words]
    mean_nn = nn_vals.mean(dim=1)  # y: mean distance to kept neighbours

    # z = log(sorted(exp(d))) per node's kept-neighbour distances == sorted(d);
    # take the median of that sorted vector as the node's scalar z.
    sorted_nn, _ = torch.sort(nn_vals, dim=1)
    z_literal = torch.log(torch.exp(sorted_nn))  # identity, kept explicit per spec
    z_med = z_literal[:, kk // 2]

    def norm(t: torch.Tensor) -> torch.Tensor:
        mn, mx = t.min(), t.max()
        return (t - mn) / (mx - mn) if (mx - mn).item() > 1e-9 else torch.full_like(t, 0.5)

    x_raw = torch.log1p(torch.tensor(freq, dtype=torch.float32, device=DEVICE))
    xN = norm(x_raw).tolist()
    yN = norm(mean_nn).tolist()
    zN = norm(z_med).tolist()

    fig = go.Figure()

    # sparse edges
    ex, ey, ez = [], [], []
    for i, j in edges:
        ex += [xN[i], xN[j], None]
        ey += [yN[i], yN[j], None]
        ez += [zN[i], zN[j], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color='rgba(120,140,160,.55)', width=2),
        name=f'sparse Manhattan edges (k={kk})', hoverinfo='skip',
    ))

    # nodes
    fig.add_trace(go.Scatter3d(
        x=xN, y=yN, z=zN, mode='markers+text',
        marker=dict(
            size=[6 + 10 * (f / max(freq)) for f in freq],
            color=zN, colorscale='Tealrose', showscale=True,
            colorbar=dict(title='z (sorted dist.)'),
            line=dict(color='black', width=0.5),
        ),
        text=words, textposition='top center', textfont=dict(size=9),
        hovertemplate='%{text}<br>freq=%{customdata[0]}<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>',
        customdata=[[freq_all[w]] for w in words],
        name='tokens',
    ))

    # piecewise-linear peak envelope on the x marginal, projected onto the z=0 floor
    xpeaks, _, _ = _local_peaks(xN)
    if len(xpeaks) > 1:
        xpeaks.sort(key=lambda p: p[0])
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in xpeaks],
            y=[0.0] * len(xpeaks),
            z=[0.0] * len(xpeaks),
            mode='lines+markers', line=dict(color='#e7a45c', width=5),
            marker=dict(size=4, color='#e7a45c'),
            name='x-axis peak envelope',
        ))

    # piecewise-linear peak envelope on the y marginal, projected onto the z=0 floor
    ypeaks, _, _ = _local_peaks(yN)
    if len(ypeaks) > 1:
        ypeaks.sort(key=lambda p: p[0])
        fig.add_trace(go.Scatter3d(
            x=[0.0] * len(ypeaks),
            y=[p[0] for p in ypeaks],
            z=[0.0] * len(ypeaks),
            mode='lines+markers', line=dict(color='#c77dd2', width=5),
            marker=dict(size=4, color='#c77dd2'),
            name='y-axis peak envelope',
        ))

    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='x: frequency (log, normalized)',
            yaxis_title='y: mean neighbour distance (normalized)',
            zaxis_title='z: sorted per-node distance (log(sorted(exp(d))) == sorted(d))',
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', y=-0.05),
        height=650,
    )

    summary = (
        f'Nodes: {n} · sparse edges: {len(edges)} (k={kk}) · '
        f'x-peaks: {len(xpeaks)} · y-peaks: {len(ypeaks)} · device: {DEVICE}'
    )
    return fig, summary


class State:
    def __init__(self):
        self.model = None; self.corpus = None; self.bindings = Bindings()

    def train(self, text):
        self.model = NGram().ingest(text)
        self.model.save()
        self.bindings = Bindings()
        self.bindings.update(text)
        self.bindings.save()
        self.corpus = Corpus(text, self.bindings)

S = State()

def train(file):
    if file is None: return 'No file', '', ''
    text = Path(getattr(file, 'name', file)).read_text(encoding='utf8', errors='replace')
    S.train(text)
    return 'Automorphism Filter Trained', text[:1000], S.bindings.summary()

def generate(prompt, vocab_size=45, k=3):
    if not S.model:
        return 'Train filter first', '', '', go.Figure(), 'Train the filter first (need a trained corpus).'
    target = S.corpus.bindings.expand({k2: float(v) for k2, v in bow(tokenize(prompt)).items()})
    state = dict(target)
    def bias(history):
        seeds = sorted(state, key=lambda x: state[x] * S.corpus.bindings.threshold, reverse=True)[:12]
        return S.model_bias(seeds) if hasattr(S, 'model_bias') else {}
    text = S.model.generate(prompt, bias)
    fig, conn_summary = build_connectome(S.bindings, int(vocab_size), int(k))
    return 'CUDA device: ' + str(DEVICE), format_matches(S.corpus.search(prompt)), text, fig, conn_summary

def format_matches(rows):
    return '\n'.join(f'{x.rank}. {x.ref.s} score={x.score:.3f} vector={x.vector:.3f}' + (' [subset]' if x.selected else ' [endpoint]') for x in rows) or 'No matches.'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--share', action='store_true')
    p.add_argument('--server-name', default='127.0.0.1')
    p.add_argument('--server-port', type=int, default=7860)
    a = p.parse_args()
    import gradio as gr
    with gr.Blocks(title="CUDA Bijective Automorphism Filter") as ui:
        gr.Markdown(f"# CUDA-first Bijective Automorphism Filter\nDevice: `{DEVICE}`")
        gr.Markdown("## 1. Corpus Ingestion")
        corpus_file = gr.File(label="Corpus File (.txt)", file_types=["file"])
        train_button = gr.Button("Train Automorphism Filter", variant="primary")
        training_status = gr.Textbox(label="Status", lines=2)
        corpus_preview = gr.Textbox(label="Preview", lines=6)
        binding_summary = gr.Textbox(label="GPU Automorphism Summary", lines=5)
        train_button.click(fn=train, inputs=[corpus_file], outputs=[training_status, corpus_preview, binding_summary])

        gr.Markdown("## 2. CUDA Stochastic Generation & Filtering")
        gr.Markdown(
            "One pass builds the generated series *and* the sparse Manhattan distance connectome "
            "over the same trained token vectors: x = frequency, y = mean neighbour distance "
            "(complementary to x), z = `log(sorted(exp(d)))` of each node's distance vector "
            "(algebraically == sorted(d)). Peak envelopes trace local maxima on the x/y marginals."
        )
        prompt_input = gr.Textbox(label="Filter Seed Prompt", placeholder="Enter a seed phrase...", lines=2)
        with gr.Row():
            vocab_slider = gr.Slider(10, 100, value=45, step=1, label="Connectome vocab size (top-N tokens)")
            k_slider = gr.Slider(1, 8, value=3, step=1, label="Connectome k-nearest neighbours")
        generate_button = gr.Button("Execute Automorphic Stream", variant="primary")
        generation_status = gr.Textbox(label="CUDA Execution Status", lines=2)
        corpus_matches = gr.Textbox(label="Subset-sum Filter Matches", lines=8)
        generated_text = gr.Textbox(label="Automorphically Filtered Series Output", lines=8)
        connectome_plot = gr.Plot(label="Sparse Manhattan Connectome")
        connectome_summary = gr.Textbox(label="Connectome Summary", lines=2)
        generate_button.click(
            fn=generate,
            inputs=[prompt_input, vocab_slider, k_slider],
            outputs=[generation_status, corpus_matches, generated_text, connectome_plot, connectome_summary],
        )

        ui.launch(server_name=a.server_name, server_port=a.server_port, share=a.share)

if __name__ == '__main__':
    main()
