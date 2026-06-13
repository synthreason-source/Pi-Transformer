"""
fork_gradio.py
──────────────
Gradio interface for the fork-weight text generator.

Run:
    python fork_gradio.py
"""

import math
import random
import textwrap
from collections import Counter

import gradio as gr
import torch
from nltk.probability import (
    ConditionalFreqDist,
    ConditionalProbDist,
    LidstoneProbDist,
)

EPS = 1e-12

DEFAULT_CORPUS = """
the quick brown fox jumps over the lazy dog and the fox ran quickly through the forest
a dog chased the fox but the fox was too quick and jumped over the fence
the lazy dog slept while the brown fox explored the quiet forest paths
animals in the forest run and jump and play all day long
the fox found food near the old oak tree beside the running stream
quick animals jump over fences and run through fields in the morning
the brown dog barked at the jumping fox near the tall fence
foxes are quick and clever animals that run through the green forest
dogs and foxes often chase each other through the open fields near the trees
the old oak tree stood tall beside the quiet stream in the deep forest
running through fields the quick brown fox leapt over every obstacle
the lazy dog watched the clever fox disappear into the dark forest path
curious foxes leap over sleeping dogs beside the mossy stream
the stream runs cold beside the ancient oak where animals gather
every morning the fox and the dog meet beside the tall oak tree
""".strip()

# ── model cache so we don't rebuild on every click ───────────────────────────
_cpd_cache: dict = {}


def _norm(t: torch.Tensor) -> torch.Tensor:
    t = t.to(torch.float64)
    return t / t.sum().clamp(EPS)


def build_cpd(corpus: str, n: int = 3, gamma: float = 0.1):
    key = (corpus[:200], n, gamma)
    if key in _cpd_cache:
        return _cpd_cache[key]
    tokens = corpus.lower().split()
    padded = [""] * (n - 1) + tokens + [""]
    cfd = ConditionalFreqDist()
    for ng in zip(*[padded[i:] for i in range(n)]):
        cfd[ng[:-1]][ng[-1]] += 1
    vocab = set(tokens) | {""}
    cpd = ConditionalProbDist(
        cfd, lambda fd: LidstoneProbDist(fd, gamma, bins=max(1, len(vocab)))
    )
    _cpd_cache[key] = (cpd, vocab, tokens)
    return cpd, vocab, tokens


# ── fork weight redistribution ────────────────────────────────────────────────

def fork_weights(
    pairs: list,
    min_prob: float = 0.15,
    min_len: int = 6,
    alpha: float = 0.5,
    max_forks: int = 8,
) -> tuple:
    n = len(pairs)
    if n < 2:
        return pairs, []
    words = [w for w, _ in pairs]
    raw = torch.tensor([float(p) for _, p in pairs], dtype=torch.float64)
    probs = raw / raw.sum().clamp(EPS)

    order = torch.argsort(probs, descending=True)
    forked: list = []
    for i in order.tolist():
        if float(probs[i]) < min_prob:
            break
        if len(words[i]) >= min_len:
            forked.append(i)
        if len(forked) >= max_forks:
            break

    if not forked:
        return list(zip(words, probs.tolist())), []

    new_probs = probs.clone()
    fork_log: list = []
    for fi in forked:
        mass = float(new_probs[fi])
        keep = mass * alpha
        spread = mass * (1.0 - alpha)
        new_probs[fi] = keep
        left = fi - 1 if fi > 0 else None
        right = fi + 1 if fi < n - 1 else None
        if left is not None and right is not None:
            new_probs[left] += spread * 0.5
            new_probs[right] += spread * 0.5
        elif left is not None:
            new_probs[left] += spread
        elif right is not None:
            new_probs[right] += spread
        fork_log.append(words[fi])

    new_probs = new_probs.clamp(min=EPS)
    new_probs = new_probs / new_probs.sum()
    return list(zip(words, new_probs.tolist())), fork_log


# ── generator ─────────────────────────────────────────────────────────────────

def generate(
    cpd,
    prompt: str,
    n_words: int,
    seed: int,
    context_window: int,
    temperature: float,
    rep_penalty: float,
    fork_min_prob: float,
    fork_min_len: int,
    fork_alpha: float,
    fork_enabled: bool,
) -> tuple:
    rng = random.Random(seed)
    toks = prompt.lower().split()
    cw = context_window
    ctx = list(toks[-cw:]) if len(toks) >= cw else [""] * (cw - len(toks)) + toks
    out = list(toks)
    history: Counter = Counter()
    fork_events: list = []

    def _dist(ctx_list):
        for cut in range(len(ctx_list), 0, -1):
            key = tuple([""] * (cw - cut) + list(ctx_list[-cut:]))
            try:
                d = cpd[key]
                if list(d.samples()):
                    return d
            except Exception:
                pass
        try:
            d = cpd[tuple([""] * cw)]
            if list(d.samples()):
                return d
        except Exception:
            pass
        return None

    for _ in range(n_words * 4):
        if len(out) - len(toks) >= n_words:
            break
        dist = _dist(ctx)
        if dist is None:
            ctx = [""] * cw
            continue
        pairs = [(w, max(1e-12, dist.prob(w))) for w in dist.samples() if w]
        if not pairs:
            continue

        raw = torch.tensor([p for _, p in pairs], dtype=torch.float64)
        probs = _norm(raw).pow(1.0 / max(temperature, 0.01))
        probs = probs / probs.sum().clamp(EPS)
        pairs = list(zip([w for w, _ in pairs], probs.tolist()))

        fork_log: list = []
        if fork_enabled:
            pairs, fork_log = fork_weights(
                pairs,
                min_prob=fork_min_prob,
                min_len=fork_min_len,
                alpha=fork_alpha,
            )
            if fork_log:
                tok_idx = len(out) - len(toks)
                fork_events.append((tok_idx, fork_log))

        pairs = [(w, p / (rep_penalty ** history[w])) for w, p in pairs]
        s = sum(p for _, p in pairs)
        if s < EPS:
            continue
        pairs = [(w, p / s) for w, p in pairs]

        draw = rng.random()
        chosen = pairs[-1][0]
        cum = 0.0
        for w, p in sorted(pairs, key=lambda x: -x[1]):
            cum += p
            if draw < cum:
                chosen = w
                break

        history[chosen] += 1
        ctx = ctx[1:] + [chosen]
        out.append(chosen)

    result, cap = [], True
    for w in out:
        result.append(w.capitalize() if cap else w)
        cap = w.rstrip("\"'")[-1:] in {".", "!", "?"}

    text = " ".join(result)
    return text, fork_events


# ── gradio callback ───────────────────────────────────────────────────────────

def run(
    corpus_file,  # Added file input
    corpus_text: str,
    prompt: str,
    n_words: int,
    seed: int,
    ngram_n: int,
    temperature: float,
    rep_penalty: float,
    fork_enabled: bool,
    fork_min_prob: float,
    fork_min_len: int,
    fork_alpha: float,
):
    # 1. Prioritize uploaded file text, fall back to manual textbox, then default dataset
    if corpus_file is not None:
        try:
            with open(corpus_file.name, "r", encoding="utf-8") as f:
                corpus = f.read().strip()
        except Exception as e:
            return f"⚠️ File read error: {e}", "—", "—"
    else:
        corpus = (corpus_text or "").strip() or DEFAULT_CORPUS

    if not corpus:
        return "⚠️ Error: Corpus is empty. Please upload a file or write text.", "—", "—"

    try:
        cpd, _, _ = build_cpd(corpus, n=int(ngram_n))
    except Exception as e:
        return f"⚠️ Model error: {e}", "—", "—"

    text, fork_events = generate(
        cpd,
        prompt=prompt or "the quick fox",
        n_words=int(n_words),
        seed=int(seed),
        context_window=int(ngram_n) - 1,
        temperature=float(temperature),
        rep_penalty=float(rep_penalty),
        fork_min_prob=float(fork_min_prob),
        fork_min_len=int(fork_min_len),
        fork_alpha=float(fork_alpha),
        fork_enabled=bool(fork_enabled),
    )

    # stats
    generated = len(text.split()) - len(prompt.split())
    stats = f"Generated: {generated} tokens · Fork events: {len(fork_events)}"

    # fork log
    if fork_events:
        lines = []
        for tok_idx, words in fork_events[:12]:
            lines.append(f"  token {tok_idx:>3}  ←  {', '.join(words)}")
        fork_log = "Mass spread from:\n" + "\n".join(lines)
    else:
        fork_log = "No fork events (try lowering min prob or enabling forks)."

    return text, stats, fork_log


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Fork-Weight Generator") as demo:

    gr.Markdown(
        """
# 🌿 Fork-Weight Text Generator
Probability forks are detected at each sampling step — high-mass long tokens
spread half their weight to neighbours, softening peaks and widening the path.
        """
    )

    with gr.Row():
        # ── left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Corpus Source")
            
            # Added file upload component
            corpus_file = gr.File(
                label="Upload Training Corpus (.txt)", 
                file_types=[".txt"],
                file_count="single"
            )
            
            corpus_box = gr.Textbox(
                value="",
                label="Or paste plain text directly (Ignored if file is uploaded)",
                lines=2,
                placeholder="Paste any plain text here…",
            )

            gr.Markdown("### Prompt & length")
            prompt_box = gr.Textbox(
                value="the quick fox",
                label="Seed prompt",
                lines=1,
            )
            run_btn = gr.Button("▶  Generate", variant="primary", size="lg")

            with gr.Row():
                n_words_sl = gr.Slider(10, 200, value=50, step=5, label="Words to generate")
                seed_num   = gr.Number(value=42, label="Seed", precision=0)

            gr.Markdown("### Model")
            with gr.Row():
                ngram_sl = gr.Slider(2, 5, value=3, step=1, label="N-gram order")
                temp_sl  = gr.Slider(0.5, 10.0, value=4.5, step=0.1, label="Temperature")
                rep_sl   = gr.Slider(1.0, 3.0, value=1.18, step=0.01, label="Repetition penalty")

            gr.Markdown("### Fork settings")
            fork_chk = gr.Checkbox(value=True, label="Enable fork weight redistribution")
            with gr.Row():
                fp_sl  = gr.Slider(0.05, 0.5, value=0.15, step=0.01, label="Min prob to fork")
                fl_sl  = gr.Slider(3, 12,    value=6,    step=1,    label="Min word length")
                fa_sl  = gr.Slider(0.1, 0.9, value=0.5,  step=0.05, label="Alpha (mass kept)")


        # ── right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            out_text = gr.Textbox(
                label="Generated text",
                lines=10,
                interactive=False,
                elem_classes=["output-text"],
            )
            stats_box = gr.Textbox(
                label="Stats",
                lines=1,
                interactive=False,
                elem_classes=["stats-box"],
            )
            fork_log_box = gr.Textbox(
                label="Fork log",
                lines=10,
                interactive=False,
                elem_classes=["fork-log"],
            )

    run_btn.click(
        fn=run,
        inputs=[
            corpus_file,  # Included as the first input
            corpus_box, prompt_box, n_words_sl, seed_num,
            ngram_sl, temp_sl, rep_sl,
            fork_chk, fp_sl, fl_sl, fa_sl,
        ],
        outputs=[out_text, stats_box, fork_log_box],
    )

    gr.Examples(
        examples=[
            [None, DEFAULT_CORPUS, "the quick fox",     50, 42, 3, 4.5, 1.18, True, 0.15, 6, 0.5],
            [None, DEFAULT_CORPUS, "a dog chased",     40,  7, 3, 4.5, 1.18, True, 0.12, 5, 0.5],
            [None, DEFAULT_CORPUS, "the old oak tree", 45, 99, 3, 3.0, 1.18, True, 0.10, 5, 0.7],
            [None, DEFAULT_CORPUS, "the quick fox",     50, 42, 3, 4.5, 1.18, False, 0.15, 6, 0.5],
        ],
        inputs=[
            corpus_file,  # Map None placeholder for examples
            corpus_box, prompt_box, n_words_sl, seed_num,
            ngram_sl, temp_sl, rep_sl,
            fork_chk, fp_sl, fl_sl, fa_sl,
        ],
        label="Quick examples (last one has forks disabled for comparison)",
    )

if __name__ == "__main__":
    demo.launch(share=False)
