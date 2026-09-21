"""Stochastic n-gram text generator -- everything in one file.

    python app.py                 # launch the Gradio UI
    python app.py --share         # public link

Sections: text primitives -> corpus search -> n-gram model -> chain of thought
-> reasoning bias -> pipeline -> app state/handlers -> UI.

Every probability used for generation is a smoothed count observed in the
corpus; text is sampled stochastically from that distribution.  gradio is
imported lazily inside build_ui(), so the logic is importable without it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

Vec = Dict[str, float]
BiasFn = Callable[[List[str]], Tuple[Optional[Vec], float]]

BOS, EOS = "<bos>", "<eos>"
SPECIAL = {BOS, EOS, "<unk>"}

# sampling
ALPHA = 0.05                 # additive smoothing over observed support
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.8
TOP_K = 20
REFRESH_EVERY = 8            # re-derive the bias every N generated tokens
BIAS_WEIGHT = 1.2            # base strength of the reasoning bias

# search scoring
LEXICAL_WEIGHT = VECTOR_WEIGHT = 0.5

# chain of thought / reasoning bias
COT_STEPS = 4
COT_DECAY = 0.5
COT_MIN_RELEVANCE = 0.05
COT_REDUNDANCY = 0.9
COT_TOP_N = 3
COT_SEED_WORDS = 12
COT_BIAS_WEIGHT = 0.6


# ---------------------------------------------------------------- text ---

def tokenize(text: str) -> List[str]:
    return text.lower().split()


def split_sentences(text: str) -> List[str]:
    return [p.strip() + "." for p in text.split(".") if p.strip()]


def bag_of_words(tokens) -> Counter:
    return Counter(t for t in tokens if t not in SPECIAL)


def cosine(a: Vec, b: Vec) -> float:
    """|cos| of two sparse vectors. All vectors here are non-negative, so the
    abs() is a no-op; it is kept only so behaviour is bit-identical."""
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    return abs(dot / (na * nb))


def jaccard(a, b) -> float:
    sa, sb = set(a) - SPECIAL, set(b) - SPECIAL
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# -------------------------------------------------------------- corpus ---

class Ref(NamedTuple):
    sentence: str
    tokens: List[str]
    vec: Vec
    freq: int


class Match(NamedTuple):
    rank: int
    sentence: str
    overlap: float
    vector: float
    freq: int
    score: float


class Corpus:
    """Sentence index: lexical (Jaccard) + term-frequency cosine search."""

    def __init__(self, text: str = "") -> None:
        sents = split_sentences(text)
        freq = Counter(s.lower() for s in sents)
        self.doc_freq: Counter = Counter()
        self.refs: List[Ref] = []
        for s in sents:
            tokens = tokenize(s)
            bag = bag_of_words(tokens)
            self.doc_freq.update(bag.keys())
            self.refs.append(Ref(s, tokens, {t: float(c) for t, c in bag.items()}, freq[s.lower()]))

    def idf(self, token: str) -> float:
        return math.log((1 + len(self.refs)) / (1 + self.doc_freq.get(token, 0)))

    def search(self, prompt: str, limit: int = 5) -> List[Match]:
        tokens = tokenize(prompt)
        vec = {t: float(c) for t, c in bag_of_words(tokens).items()}
        scored = []
        for r in self.refs:
            lex, cos = jaccard(tokens, r.tokens), cosine(vec, r.vec)
            scored.append((LEXICAL_WEIGHT * lex + VECTOR_WEIGHT * cos, r, lex, cos))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [Match(i, r.sentence, lex, cos, r.freq, score)
                for i, (score, r, lex, cos) in enumerate(scored[:limit], start=1)]


# --------------------------------------------------------------- model ---

class NGram:
    """Trigram -> bigram -> unigram backoff; every probability is a smoothed
    corpus count, and sampling draws from it with random.choices."""

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = alpha
        self.unigram: Counter = Counter()
        self.bigram: Dict[str, Counter] = defaultdict(Counter)
        self.trigram: Dict[str, Counter] = defaultdict(Counter)

    # training
    def ingest(self, text: str) -> "NGram":
        for s in split_sentences(text):
            seq = [BOS, BOS, *tokenize(s), EOS]
            self.unigram.update(seq)
            for a, b in zip(seq, seq[1:]):
                self.bigram[a][b] += 1
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                self.trigram[f"{a}\t{b}"][c] += 1
        return self

    # probabilities
    def _smooth(self, counts: Counter) -> Vec:
        total = sum(counts.values()) + self.alpha * len(counts)
        return {t: (c + self.alpha) / total for t, c in counts.items()}

    def distribution(self, tokens: List[str]) -> Vec:
        prev2, prev = ([BOS, BOS, *tokens])[-2:]
        counts = self.trigram.get(f"{prev2}\t{prev}") or self.bigram.get(prev) or self.unigram
        return self._smooth(counts)

    def bias_from(self, seeds: Vec) -> Vec:
        """Real bigram follow-on mass of the seed words: P(next | seed) * weight."""
        bias: Dict[str, float] = defaultdict(float)
        for word, w in seeds.items():
            followers = self.bigram.get(word)
            if w <= 0 or not followers:
                continue
            total = sum(followers.values())
            for t, c in followers.items():
                bias[t] += w * (c / total)
        return dict(bias)

    # sampling
    def _top_weights(self, tokens, temperature, top_k, bias, bias_weight):
        """Unnormalised top-k sampling weights -- the single source of truth
        for both sample() and next_distribution()."""
        dist = self.distribution(tokens)
        if not dist:
            return [], []
        temp = max(temperature, 1e-5)
        use_bias = bool(bias and bias_weight)
        logits = {}
        for t, p in dist.items():
            lp = math.log(max(p, 1e-12))
            if use_bias:
                lp += bias_weight * bias.get(t, 0.0)
            logits[t] = lp / temp
        top = sorted(logits, key=logits.get, reverse=True)[:top_k]
        peak = max(logits[t] for t in top)
        return top, [math.exp(logits[t] - peak) for t in top]

    def sample(self, tokens, temperature=TEMPERATURE, top_k=TOP_K, bias=None, bias_weight=0.0) -> str:
        top, weights = self._top_weights(tokens, temperature, top_k, bias, bias_weight)
        return random.choices(top, weights=weights, k=1)[0] if top else EOS

    def next_distribution(self, tokens, temperature=TEMPERATURE, top_k=TOP_K,
                          bias=None, bias_weight=0.0) -> Vec:
        """Exact probabilities sample() draws from (for curves / tests)."""
        top, weights = self._top_weights(tokens, temperature, top_k, bias, bias_weight)
        z = sum(weights)
        return {t: w / z for t, w in zip(top, weights)}

    def generate(self, prompt: str, n: int = MAX_NEW_TOKENS, temperature=TEMPERATURE,
                 top_k=TOP_K, bias_fn: Optional[BiasFn] = None,
                 refresh_every: int = REFRESH_EVERY) -> str:
        out = tokenize(prompt)
        bias, weight = None, 0.0
        for step in range(n):
            if bias_fn is not None and step % max(1, refresh_every) == 0:
                bias, weight = bias_fn(out)
            out.append(self.sample(out, temperature, top_k, bias, weight))
        return " ".join(t for t in out if t not in SPECIAL)

    # persistence (file format stays loadable by the original)
    def save(self, path) -> None:
        data = {"smoothing_alpha": self.alpha, "unigram": dict(self.unigram),
                "bigram": {k: dict(v) for k, v in self.bigram.items()},
                "trigram": {k: dict(v) for k, v in self.trigram.items()}}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "NGram":
        d = json.loads(Path(path).read_text(encoding="utf-8"))   # extra legacy keys ignored
        m = cls(d.get("smoothing_alpha", ALPHA))
        m.unigram = Counter(d.get("unigram", {}))
        m.bigram = defaultdict(Counter, {k: Counter(v) for k, v in d.get("bigram", {}).items()})
        m.trigram = defaultdict(Counter, {k: Counter(v) for k, v in d.get("trigram", {}).items()})
        return m


# ----------------------------------------------------- chain of thought ---

class Step(NamedTuple):
    index: int
    sentence: str
    relevance: float
    novelty: float


def reason(corpus: Corpus, prompt: str, steps: int = COT_STEPS) -> Tuple[List[Step], Vec]:
    """Chain of real corpus sentences chosen against a compounding state.
    Returns (chain, final_state)."""
    state = {t: float(c) for t, c in bag_of_words(tokenize(prompt)).items()}
    chosen: List[Vec] = []
    used: set = set()
    chain: List[Step] = []
    for _ in range(steps):
        scored = []
        for i, ref in enumerate(corpus.refs):
            if i in used:
                continue
            rel = cosine(state, ref.vec)
            if rel < COT_MIN_RELEVANCE:
                continue
            overlap = max((cosine(ref.vec, v) for v in chosen), default=0.0)
            if overlap >= COT_REDUNDANCY:
                continue
            nov = 1.0 - overlap
            scored.append((rel * nov, i, rel, nov))
        if not scored:
            break
        scored.sort(key=lambda s: s[0], reverse=True)
        pool = scored[:COT_TOP_N]
        _, i, rel, nov = random.choices(pool, weights=[s[0] for s in pool], k=1)[0]
        ref = corpus.refs[i]
        used.add(i)
        chosen.append(ref.vec)
        chain.append(Step(len(chain) + 1, ref.sentence, rel, nov))
        state = {t: COT_DECAY * w for t, w in state.items()}
        for t, w in ref.vec.items():
            state[t] = state.get(t, 0.0) + w
    return chain, state


def format_chain(chain: List[Step]) -> str:
    if not chain:
        return "No reasoning steps found for that prompt."
    lines = [f"Step {s.index}: {s.sentence}.  (relevance={s.relevance:.3f}, novelty={s.novelty:.3f})"
             for s in chain]
    lines.append(f"Conclusion: {chain[-1].sentence}.")
    return "\n".join(lines)


class ReasoningBias:
    """A BiasFn. Keeps a decaying state of the text so far, turns its most
    distinctive words (state x idf) into seeds, and pulls harder the further
    the text drifts (1 - cos) from the original reasoning state."""

    def __init__(self, model: NGram, corpus: Corpus, anchor: Vec, prompt_len: int,
                 base_weight: float = BIAS_WEIGHT) -> None:
        self.model, self.corpus = model, corpus
        self.anchor, self.state = dict(anchor), dict(anchor)
        self.seen = prompt_len
        self.base_weight = base_weight

    def seeds(self) -> Vec:
        weighted = {t: w * self.corpus.idf(t) for t, w in self.state.items()
                    if t in self.model.unigram and t not in SPECIAL}
        top = sorted(((t, w) for t, w in weighted.items() if w > 0),
                     key=lambda kv: kv[1], reverse=True)[:COT_SEED_WORDS]
        return {t: w / top[0][1] for t, w in top} if top else {}

    def __call__(self, generated: List[str]):
        new = generated[self.seen:]
        self.seen = len(generated)
        if new:
            self.state = {t: COT_DECAY * w for t, w in self.state.items()}
            for t, c in bag_of_words(new).items():
                self.state[t] = self.state.get(t, 0.0) + float(c)
        seeds = {t: COT_BIAS_WEIGHT * w for t, w in self.seeds().items()}
        scale = 1.0 + (1.0 - cosine(self.state, self.anchor))
        return self.model.bias_from(seeds), self.base_weight * scale


# ---------------------------------------------------------------- pipeline ---

def format_matches(matches: List[Match]) -> str:
    return "\n".join(f"{m.rank}. {m.sentence} (score={m.score:.3f}, "
                     f"overlap={m.overlap:.3f}, vector={m.vector:.3f})" for m in matches)


def _fmt_seeds(seeds: Vec) -> str:
    return ", ".join(f"{w} ({x:.2f})" for w, x in seeds.items())


def generate_text(model: NGram, corpus: Corpus, prompt: str = "",
                  n: int = MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    """Returns (seed_summary, corpus_matches, generated_text)."""
    prompt = (prompt or "").strip()

    chain: List[Step] = []
    seeds: Vec = {}
    bias_fn: Optional[BiasFn] = None
    if prompt and corpus.refs:
        chain, state = reason(corpus, prompt)
        bias_fn = ReasoningBias(model, corpus, state, len(tokenize(prompt)))
        seeds = bias_fn.seeds()

    text = model.generate(prompt, n, temperature, top_k, bias_fn)

    if not prompt:
        matches = "No prompt provided for corpus search."
    else:
        found = corpus.search(prompt, 5)
        matches = format_matches(found) if found else "No corpus file loaded."
        if chain:
            matches += "\n\nReasoning chain used for modifiers:\n" + format_chain(chain)

    summary = ("Reasoning seed words (refreshed during generation): " + _fmt_seeds(seeds)
               if seeds else "No reasoning seed words (enter a prompt that matches the corpus).")
    return summary, matches, text


# ------------------------------------------------------------ app state ---

MODEL_PATH = "model.json"
DEFAULT_CORPUS = "corpus.txt"
NOT_LOADED = "Model not loaded. Upload a corpus and click 'Train model' first."


class State:
    """Model + corpus index, loaded lazily from disk."""

    def __init__(self) -> None:
        self.model: Optional[NGram] = None
        self.corpus: Optional[Corpus] = None
        self.corpus_text: Optional[str] = None

    def load(self) -> None:
        self.model = NGram.load(MODEL_PATH)
        if self.corpus_text is None and Path(DEFAULT_CORPUS).exists():
            self.corpus_text = Path(DEFAULT_CORPUS).read_text(encoding="utf-8", errors="replace")
        self.corpus = Corpus(self.corpus_text or "")

    def train(self, text: str) -> NGram:
        self.corpus_text = text
        model = NGram().ingest(text)
        model.save(MODEL_PATH)
        self.load()
        return model


S = State()


def train(corpus_file):
    if corpus_file is None:
        return "No corpus file uploaded.", "", ""
    path = Path(getattr(corpus_file, "name", corpus_file))
    if not path.exists():
        return f"Corpus file not found: {path}", "", ""
    text = path.read_text(encoding="utf-8", errors="replace")
    model = S.train(text)
    summary = (f"Vocabulary: {len(model.unigram)}\n"
               f"Bigram contexts: {len(model.bigram)}\n"
               f"Trigram contexts: {len(model.trigram)}")
    return f"Model trained and saved to {MODEL_PATH}.", "\n".join(text.splitlines()[:5]), summary


def generate(prompt):
    if S.model is None or S.corpus is None:
        return NOT_LOADED, "", ""
    return generate_text(S.model, S.corpus, prompt)


# ------------------------------------------------------------------- UI ---

def build_ui():
    import gradio as gr                      # lazy: logic above needs no gradio

    with gr.Blocks(title="Stochastic N-Gram Model") as demo:
        gr.Markdown(
            "# Stochastic N-Gram Model\n"
            "1. Upload any text file as corpus.  2. Click **Train model**.  "
            "3. Enter a prompt -- a corpus-derived reasoning chain biases generation "
            "using real bigram follow-on counts.\n\n"
            "Sampling is stochastic, so re-running a prompt can give different text."
        )

        gr.Markdown("## 1. Corpus & Training")
        with gr.Row():
            with gr.Column(scale=1):
                corpus_in = gr.File(label="Corpus file (any type)", file_types=["file"])
                train_btn = gr.Button("Train model", variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(label="Training status", lines=2)
                sample = gr.Textbox(label="Corpus sample (first 5 lines)", lines=5)
                summary = gr.Textbox(label="Model summary", lines=4)
        train_btn.click(train, [corpus_in], [status, sample, summary])

        gr.Markdown("## 2. Stochastic Generation")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Text prompt", lines=3)
                gen_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                seeds_out = gr.Textbox(label="Reasoning seed words", lines=4)
                matches_out = gr.Textbox(label="Corpus matches", lines=6)
                text_out = gr.Textbox(label="Generated text", lines=8)
        gen_btn.click(generate, [prompt], [seeds_out, matches_out, text_out])

    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7860)
    args = p.parse_args()
    try:
        S.load()
    except FileNotFoundError:
        pass
    build_ui().launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
