from __future__ import annotations

"""
Slimmed kernel version of the original app.

What's kept (the actual generation kernel):
  - tokenize / n-gram counting (unigram, bigram, trigram)
  - trigram -> bigram -> unigram backoff distribution
  - lexical context vectors (bigram-context bag-of-words per token)
  - cosine-similarity bias from an "instruction vector" folded into
    next-token scores (still usable by any caller, just no longer fed by
    a camera image -- see below)
  - temperature + top-k sampling
  - corpus lexical search (unchanged, it was already minimal)
  - the Gradio UI, now text-only (no image/webcam input)

Also removed in this pass: the camera/webcam pipeline (image capture,
extract_raw_image_tensor, project_image_to_lexical_vector) and the torch
dependency it existed for. The kernel never needed torch -- it was only
used to turn a webcam frame into an instruction vector. With that gone the
app is pure Python + numpy (numpy is kept for the entropy-matrix formulas
below).

What's removed (previously ~90% of the "math" in the file), because none of
it changed the kernel's behavior in a way the app's own callers exercised,
or because it was provably dead code:
  - context_dimension_reduction  (agglomerative context merging)
  - deep_bilinear_activation     (32-layer hashed elementwise-square stack)
  - influence_and_isomorphism    (duplicate-token thresholding)
  - transitivity_mask            (dead: no caller ever passed a mask)
  - instruction_compound_growth  (per-token compounding multiplier)

Each of those is recorded below as an ENTROPY_MATRIX: not its hyperparameters
(that would encode the *code*), but the actual numeric output each formula
produces when run on a canonical sample context-vector (that encodes the
*math*). The Shannon entropy of each output row is a single number standing
in for "how much this transformation actually did to the vector." Nothing
else in the file reads this matrix at runtime -- it's a record of what was
deleted, computed once at import time for reference.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import gradio as gr

MODEL_PATH = "model.json"
DEFAULT_CORPUS_FILE = "corpus.txt"

MAX_NEW_TOKENS = 800
TEMPERATURE = 0.8
TOP_K = 20
MIN_COUNT = 1
INSTRUCTION_WEIGHT = 1.5

LEXICAL_WEIGHT = 0.15
VECTOR_WEIGHT = 0.15

IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}


# ------------------- entropy matrix of the removed math ------------------
#
# These five functions are faithful, minimal re-implementations of the
# formulas that used to run inside the model. They are not called by the
# kernel below -- they exist only so ENTROPY_MATRIX can be computed by
# actually executing the math once, on one canonical sample vector, instead
# of just listing the constants those formulas used to take as arguments.

_SAMPLE_CONTEXT_VECTOR = np.array(
    [0.40, 0.25, 0.15, 0.12, 0.05, 0.02, 0.01], dtype=np.float64
)


def _math_context_dimension_reduction(x: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    v = x.copy()
    target = max(1, math.ceil(len(v) * fraction))
    while len(v) > target:
        i, j = np.argsort(v)[-2:]  # greedily merge the two largest bins
        merged = v[i] + v[j]
        v = np.append(np.delete(v, [i, j]), merged)
    out = np.zeros_like(x)
    out[: len(v)] = v
    return out


def _math_deep_bilinear_activation(x: np.ndarray, layers: int = 32, scale: float = 14.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    v = x.copy()
    for _ in range(layers):
        projection = rng.standard_normal((len(v), len(v)))
        projection /= np.linalg.norm(projection, axis=0, keepdims=True) + 1e-12
        u = v @ projection
        v = np.tanh(scale * u * u)  # u and v share one projection -> elementwise square
        if not np.any(v):
            break
    return v


def _math_influence_and_isomorphism(x: np.ndarray, influence_tau: float = 0.7) -> np.ndarray:
    spread = x.max() - x.min() + 1e-12
    similarity = 1.0 - np.abs(np.subtract.outer(x, x)) / spread  # pairwise closeness, diag=1
    np.fill_diagonal(similarity, -1.0)  # a token never influences itself
    influence = np.where(similarity >= influence_tau, similarity, 0.0)
    return influence.sum(axis=1)


def _math_transitivity_mask(x: np.ndarray, weight: float = 0.71, k: float = 13.0) -> np.ndarray:
    return weight * (np.exp(k * x) - 1.0)


def _math_instruction_compound_growth(
    x: np.ndarray, curve_k: float = 2.0, midpoint: float = 0.5, growth: float = 0.65, cap: float = 4.0
) -> np.ndarray:
    out = x.copy()
    factor = 1.0
    for i in range(len(out)):
        likeness = 1.0 / (1.0 + math.exp(-curve_k * (out[i] - midpoint)))
        factor = min(cap, factor * (1.0 + growth * likeness))
        out[i] = out[i] * factor
    return out


def _shannon_entropy_bits(v: np.ndarray) -> float:
    magnitudes = np.abs(v)
    total = magnitudes.sum()
    if total <= 0:
        return 0.0
    probabilities = magnitudes[magnitudes > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _build_entropy_matrix() -> Dict[str, Dict[str, Any]]:
    formulas = {
        "context_dimension_reduction": _math_context_dimension_reduction,
        "deep_bilinear_activation": _math_deep_bilinear_activation,
        "influence_and_isomorphism": _math_influence_and_isomorphism,
        "transitivity_mask": _math_transitivity_mask,
        "instruction_compound_growth": _math_instruction_compound_growth,
    }
    matrix = {}
    for name, fn in formulas.items():
        output_row = fn(_SAMPLE_CONTEXT_VECTOR)
        matrix[name] = {
            "output_row": [round(float(v), 4) for v in output_row],
            "entropy_bits": round(_shannon_entropy_bits(output_row), 4),
        }
    return matrix


# Computed once at import time from the sample vector above: rows are what
# each removed formula actually did to a context vector, not the arguments
# it used to be called with.
ENTROPY_MATRIX: Dict[str, Dict[str, Any]] = _build_entropy_matrix()


# --------------------------- shared helpers --------------------------

def tokenize(text: str) -> List[str]:
    return text.lower().split()


def split_sentences(text: str) -> List[str]:
    return [p.strip() for p in text.split(".") if p.strip()]


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t not in IGNORED_TOKENS)


def cosine_similarity(a: Dict[str, float], b: Dict[str, float], eps: float = 1e-12) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a < eps or norm_b < eps:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a) - IGNORED_TOKENS, set(b) - IGNORED_TOKENS
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def strip_structural_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in IGNORED_TOKENS]


# --------------------------- corpus search ----------------------------

@dataclass
class CorpusReference:
    sentence: str
    tokens: List[str]
    vector: Dict[str, float]
    frequency: int = 1


@dataclass
class Candidate:
    sentence: str
    symbolic_overlap: float
    vector_similarity: float
    frequency: int
    score: float
    rank: int = 0


class CorpusSearch:
    def __init__(self, lexical_weight=LEXICAL_WEIGHT, vector_weight=VECTOR_WEIGHT):
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.references: List[CorpusReference] = []

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(s.lower() for s in sentences)
        self.references = []
        for sentence in sentences:
            tokens = tokenize(sentence)
            if not tokens:
                continue
            bow = bag_of_words(tokens)
            self.references.append(
                CorpusReference(
                    sentence=sentence,
                    tokens=tokens,
                    vector={t: float(c) for t, c in bow.items()},
                    frequency=counts[sentence.lower()],
                )
            )

    def analyze(self, prompt: str, limit: int = 5) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)
        prompt_vector = {t: float(c) for t, c in bag_of_words(prompt_tokens).items()}

        candidates = []
        for ref in self.references:
            symbolic = lexical_overlap(prompt_tokens, ref.tokens)
            vec_sim = cosine_similarity(prompt_vector, ref.vector)
            score = self.lexical_weight * symbolic + self.vector_weight * vec_sim
            candidates.append(
                Candidate(ref.sentence, symbolic, vec_sim, ref.frequency, score)
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:limit]
        for i, c in enumerate(candidates, start=1):
            c.rank = i
        return candidates


# --------------------------- the kernel --------------------------------

@dataclass
class NGramModel:
    """Trigram-backoff language model with cosine-similarity steering.

    This is the entire kernel: count n-grams, back off trigram -> bigram ->
    unigram for the next-token distribution, and optionally nudge scores
    toward tokens whose bigram-context vector is similar to an external
    "instruction vector" (kept as a general hook; nothing in this app
    currently supplies one now that the image pipeline is gone).
    """

    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    # ---- training ----

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)
            if not words:
                continue
            self._add_sequence(["<bos>", "<bos>", *words, self.eos_token])

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        # lexical vector for a token = normalized distribution of the
        # bigram contexts it appears after. This is what the
        # instruction-bias cosine similarity operates on.
        token_contexts: Dict[str, Counter] = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}
        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {c: n / total for c, n in counts.items()}

        self.finalized = True

    # ---- scoring / sampling ----

    def backoff_distribution(self, previous: str, previous_previous: Optional[str]) -> Dict[str, float]:
        if previous_previous is not None:
            counts = self.trigram.get(f"{previous_previous}\t{previous}")
            if counts:
                return self.normalize(counts)
        counts = self.bigram.get(previous)
        if counts:
            return self.normalize(counts)
        return self.normalize(self.unigram)

    @staticmethod
    def normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()} if total else {}

    def resolve_context(self, prompt: str) -> Tuple[str, Optional[str]]:
        tokens = tokenize(prompt)
        if not tokens:
            return "<bos>", None
        previous = tokens[-1]
        previous_previous = tokens[-2] if len(tokens) >= 2 else None
        return previous, previous_previous

    def score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        previous, previous_previous = self.resolve_context(prompt)
        base = self.backoff_distribution(previous, previous_previous)
        if not base:
            return {}

        candidates = sorted(base, key=base.get, reverse=True)[:candidate_limit]

        scores: Dict[str, float] = {}
        for token in candidates:
            score = safe_log(base[token])

            if instruction_vector and instruction_weight:
                likeness = cosine_similarity(instruction_vector, self.lexical_vectors.get(token, {}))
                score += instruction_weight * likeness

            scores[token] = score

        return scores

    def probabilities(
        self,
        prompt: str,
        temperature: float,
        candidate_limit: int,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
    ) -> Dict[str, float]:
        scores = self.score_next_token(
            prompt, candidate_limit, instruction_vector, instruction_weight
        )
        if not scores:
            return {}

        temperature = max(temperature, 1e-5)
        scaled = {t: s / temperature for t, s in scores.items()}
        maximum = max(scaled.values())
        exps = {t: math.exp(s - maximum) for t, s in scaled.items()}
        total = sum(exps.values())
        return {t: v / total for t, v in exps.items()} if total else {}

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
    ) -> str:
        probs = self.probabilities(
            prompt, temperature, max(top_k, 1), instruction_vector, instruction_weight
        )
        if not probs:
            return self.eos_token
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
    ) -> str:
        generated = tokenize(prompt)
        for _ in range(max_new_tokens):
            token = self.sample_next(
                " ".join(generated), temperature, top_k, instruction_vector, instruction_weight
            )
            generated.append(token)
        return self.detokenize(strip_structural_tokens(generated))

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        return " ".join(tokens)

    # ---- persistence ----

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {k: dict(v) for k, v in self.trigram.items()},
            "lexical_vectors": self.lexical_vectors,
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data.get("eos_token", "<eos>"),
            unk_token=data.get("unk_token", "<unk>"),
            min_count=data.get("min_count", MIN_COUNT),
        )
        model.unigram = Counter(data.get("unigram", {}))
        model.bigram = defaultdict(Counter, {k: Counter(v) for k, v in data.get("bigram", {}).items()})
        model.trigram = defaultdict(Counter, {k: Counter(v) for k, v in data.get("trigram", {}).items()})
        model.lexical_vectors = data.get("lexical_vectors", {})
        model.vocabulary = data.get("vocabulary", [])
        model.finalized = data.get("finalized", False)
        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------- globals & UI wiring ---------------------------

TEXT_MODEL: NGramModel | None = None
CORPUS_SEARCH: CorpusSearch | None = None
CORPUS_TEXT_CACHE: str | None = None


def load_text_model() -> NGramModel:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Upload a corpus and click 'Train model' first.")
    model = NGramModel.load_json(model_path)
    if not model.finalized:
        model.finalize()
    return model


def load_corpus_search(corpus_text: str) -> CorpusSearch:
    search = CorpusSearch()
    search.build_index(corpus_text)
    return search


def reload_globals():
    global TEXT_MODEL, CORPUS_SEARCH, CORPUS_TEXT_CACHE
    TEXT_MODEL = load_text_model()

    if CORPUS_TEXT_CACHE:
        corpus_text = CORPUS_TEXT_CACHE
    else:
        default_path = Path(DEFAULT_CORPUS_FILE)
        corpus_text = default_path.read_text(encoding="utf-8", errors="replace") if default_path.exists() else ""
        CORPUS_TEXT_CACHE = corpus_text

    CORPUS_SEARCH = load_corpus_search(corpus_text)


def format_matches(prompt: str, limit: int = 5) -> str:
    if CORPUS_SEARCH is None or not CORPUS_SEARCH.references:
        return "No corpus file loaded."
    candidates = CORPUS_SEARCH.analyze(prompt, limit=limit)
    if not candidates:
        return "No corpus matches found."
    return "\n".join(
        f"{c.rank}. {c.sentence} (score={c.score:.3f}, overlap={c.symbolic_overlap:.3f}, vector={c.vector_similarity:.3f})"
        for c in candidates
    )


def train_model_from_file(corpus_file):
    global CORPUS_TEXT_CACHE
    if corpus_file is None:
        return "No corpus file uploaded.", "", ""

    path = Path(corpus_file.name) if hasattr(corpus_file, "name") else Path(corpus_file)
    if not path.exists():
        return f"Corpus file not found: {path}", "", ""

    corpus_text = path.read_text(encoding="utf-8", errors="replace")
    CORPUS_TEXT_CACHE = corpus_text

    model = NGramModel()
    model.ingest_text(corpus_text)
    model.finalize()
    model.save_json(Path(MODEL_PATH))

    reload_globals()

    sample_lines = "\n".join(corpus_text.splitlines()[:5])
    summary = (
        f"Vocabulary: {len(model.vocabulary)}\n"
        f"Unigrams: {len(model.unigram)}\n"
        f"Bigram contexts: {len(model.bigram)}\n"
        f"Trigram contexts: {len(model.trigram)}"
    )
    return f"Model trained and saved to {MODEL_PATH}.", sample_lines, summary


def generate_from_prompt(user_prompt: str):
    if TEXT_MODEL is None or CORPUS_SEARCH is None:
        return "Model not loaded. Upload a corpus and click 'Train model' first.", ""

    prompt = user_prompt.strip() if user_prompt and user_prompt.strip() else "<bos>"

    generated = TEXT_MODEL.generate(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )
    corpus_matches = format_matches(prompt, limit=5) if prompt != "<bos>" else "No prompt provided for corpus search."

    return corpus_matches, generated


with gr.Blocks(title="Tau Model") as demo:
    gr.Markdown(
        """
# Tau Model

1. Upload any file as corpus.
2. Click **Train model**.
3. Enter a prompt and generate text.
"""
    )

    gr.Markdown("## 1. Corpus & Training")
    with gr.Row():
        with gr.Column(scale=1):
            corpus_file_input = gr.File(label="Corpus file (any type)", file_types=["file"])
            train_button = gr.Button("Train model", variant="primary")
        with gr.Column(scale=2):
            train_status = gr.Textbox(label="Training status", lines=2)
            corpus_sample = gr.Textbox(label="Corpus sample (first 5 lines)", lines=5)
            model_summary = gr.Textbox(label="Model summary", lines=6)

    train_button.click(
        fn=train_model_from_file,
        inputs=[corpus_file_input],
        outputs=[train_status, corpus_sample, model_summary],
    )

    gr.Markdown("## 2. Text Generation")
    with gr.Row():
        with gr.Column(scale=1):
            user_prompt = gr.Textbox(label="Prompt", lines=3)
            generate_button = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            corpus_output = gr.Textbox(label="Corpus matches", lines=6)
            generated_output = gr.Textbox(label="Tau model response", lines=8)

    generate_button.click(
        fn=generate_from_prompt,
        inputs=[user_prompt],
        outputs=[corpus_output, generated_output],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    try:
        reload_globals()
    except FileNotFoundError:
        pass

    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
