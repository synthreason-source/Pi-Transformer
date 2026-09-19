from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import gradio as gr


MODEL_PATH = "model.json"
DEFAULT_CORPUS_FILE = "corpus.txt"

MAX_NEW_TOKENS = 800
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 20

MIN_COUNT = 1
# Additive ("Laplace-style") smoothing applied to observed counts before
# they're turned into a probability distribution. This is the only knob
# that shapes how "flat" vs "peaky" sampling is -- it is applied directly
# to real corpus counts, nothing else.
SMOOTHING_ALPHA = 0.05

IMAGE_BIAS_WEIGHT = 1.2

# A small, fixed vocabulary of "descriptor" words tied to simple visual
# properties (brightness / warmth / greenness). None of these words are
# forced into the model -- at generation time we only ever use the ones
# that actually appear in the trained corpus, and we bias generation
# using their *real* bigram follow-on counts from that corpus. So the
# camera feature never injects anything that isn't grounded in dataset
# statistics.
IMAGE_DESCRIPTOR_WORDS: Dict[str, List[str]] = {
    "bright": ["bright", "light", "sunny", "white", "day"],
    "dark": ["dark", "night", "shadow", "black", "dim"],
    "warm": ["warm", "red", "orange", "fire", "hot"],
    "cool": ["cool", "blue", "cold", "ice", "water"],
    "green": ["green", "grass", "forest", "leaf", "plant"],
}

RANDOM_SEED = None
random.seed(RANDOM_SEED)

IGNORED_TOKENS = {
    "<bos>",
    "<eos>",
    "<unk>",
}


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def split_sentences(text: str) -> List[str]:
    parts = text.split(".")
    return [part.strip() for part in parts if part.strip()]


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(token for token in tokens if token not in IGNORED_TOKENS)


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)

    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def lexical_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a) - IGNORED_TOKENS
    set_b = set(b) - IGNORED_TOKENS

    if not set_a or not set_b:
        return 0.0

    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


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
    """Finds corpus sentences related to a prompt using plain lexical
    overlap (Jaccard) and a term-frequency cosine similarity. Both
    signals come directly from counting words in the dataset -- no
    learned embeddings involved.
    """

    def __init__(self, lexical_weight: float = 0.5, vector_weight: float = 0.5) -> None:
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.references: List[CorpusReference] = []

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(sentence.lower() for sentence in sentences)

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
                    vector={token: float(count) for token, count in bow.items()},
                    frequency=counts[sentence.lower()],
                )
            )

    def analyze(self, prompt: str, limit: int = 5) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)
        prompt_vector = {
            token: float(count) for token, count in bag_of_words(prompt_tokens).items()
        }

        candidates: List[Candidate] = []

        for reference in self.references:
            symbolic = lexical_overlap(prompt_tokens, reference.tokens)
            vector_similarity = cosine_similarity(prompt_vector, reference.vector)

            score = (
                self.lexical_weight * symbolic + self.vector_weight * vector_similarity
            )

            candidates.append(
                Candidate(
                    sentence=reference.sentence,
                    symbolic_overlap=symbolic,
                    vector_similarity=vector_similarity,
                    frequency=reference.frequency,
                    score=score,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        candidates = candidates[:limit]

        for index, candidate in enumerate(candidates, start=1):
            candidate.rank = index

        return candidates


@dataclass
class NGramModel:
    """A trigram-with-backoff language model.

    Every probability used for sampling is computed directly from counts
    observed in the training corpus (with light additive smoothing).
    Generation is genuinely stochastic: at each step we draw the next
    token from that real probability distribution via `random.choices`,
    so the same prompt can legitimately produce different continuations
    from one run to the next.
    """

    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT
    smoothing_alpha: float = SMOOTHING_ALPHA

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    # ---------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------- #

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)

            if not words:
                continue

            sequence = ["<bos>", "<bos>", *words, self.eos_token]
            self._add_sequence(sequence)

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return

        for token in sequence:
            self.unigram[token] += 1

        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1

        for first, second, third in zip(sequence, sequence[1:], sequence[2:]):
            key = f"{first}\t{second}"
            self.trigram[key][third] += 1

        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(
            token for token, count in self.unigram.items() if count >= self.min_count
        )

        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        self.finalized = True

    # ---------------------------------------------------------------- #
    # Probability estimation (this is the whole "algorithm" now)
    # ---------------------------------------------------------------- #

    def _smoothed_distribution(self, counts: Counter) -> Dict[str, float]:
        """Turn a Counter of observed next-token counts into a probability
        distribution using additive smoothing over the observed support.
        """
        if not counts:
            return {}

        alpha = self.smoothing_alpha
        total = sum(counts.values()) + alpha * len(counts)

        return {token: (count + alpha) / total for token, count in counts.items()}

    def backoff_distribution(
        self, previous: str, previous_previous: Optional[str]
    ) -> Dict[str, float]:
        """Trigram -> bigram -> unigram backoff. Each level's probabilities
        are estimated purely from how often that continuation was actually
        seen in the corpus.
        """
        if previous_previous is not None:
            key = f"{previous_previous}\t{previous}"
            counts = self.trigram.get(key)

            if counts:
                return self._smoothed_distribution(counts)

        counts = self.bigram.get(previous)

        if counts:
            return self._smoothed_distribution(counts)

        return self._smoothed_distribution(self.unigram)

    def cooccurrence_bias(self, seed_weights: Dict[str, float]) -> Dict[str, float]:
        """Given a set of seed words (e.g. derived from an image) and their
        weights, return a bias distribution built entirely from real bigram
        follow-on counts of those seed words in the corpus: "what actually
        tends to come after this word in the dataset".
        """
        bias: Dict[str, float] = defaultdict(float)

        for word, weight in seed_weights.items():
            if weight <= 0:
                continue

            followers = self.bigram.get(word)

            if not followers:
                continue

            total = sum(followers.values())

            for token, count in followers.items():
                bias[token] += weight * (count / total)

        return dict(bias)

    def resolve_context(self, tokens: List[str]) -> Tuple[str, Optional[str]]:
        if not tokens:
            return "<bos>", "<bos>"

        previous = tokens[-1]
        previous_previous = tokens[-2] if len(tokens) >= 2 else "<bos>"

        return previous, previous_previous

    # ---------------------------------------------------------------- #
    # Sampling
    # ---------------------------------------------------------------- #

    def sample_next(
        self,
        tokens: List[str],
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        bias: Optional[Dict[str, float]] = None,
        bias_weight: float = 0.0,
    ) -> str:
        if not self.finalized:
            self.finalize()

        previous, previous_previous = self.resolve_context(tokens)
        distribution = self.backoff_distribution(previous, previous_previous)

        if not distribution:
            return self.eos_token

        temperature = max(temperature, 1e-5)

        # Combine the dataset-derived log-probability with an optional
        # dataset-derived bias term (also just real bigram statistics),
        # then temperature-scale and renormalize -- standard stochastic
        # sampling, nothing invented.
        adjusted: Dict[str, float] = {}

        for token, probability in distribution.items():
            log_prob = math.log(max(probability, 1e-12))

            if bias and bias_weight:
                log_prob += bias_weight * bias.get(token, 0.0)

            adjusted[token] = log_prob / temperature

        top_tokens = sorted(adjusted, key=adjusted.get, reverse=True)[:top_k]

        maximum = max(adjusted[token] for token in top_tokens)
        weights = [math.exp(adjusted[token] - maximum) for token in top_tokens]

        return random.choices(top_tokens, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        bias: Optional[Dict[str, float]] = None,
        bias_weight: float = 0.0,
    ) -> str:
        generated = tokenize(prompt)

        for _ in range(max_new_tokens):
            token = self.sample_next(
                generated,
                temperature=temperature,
                top_k=top_k,
                bias=bias,
                bias_weight=bias_weight,
            )



            generated.append(token)

        visible_tokens = [token for token in generated if token not in IGNORED_TOKENS]

        return " ".join(visible_tokens)

    # ---------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "smoothing_alpha": self.smoothing_alpha,
            "unigram": dict(self.unigram),
            "bigram": {key: dict(value) for key, value in self.bigram.items()},
            "trigram": {key: dict(value) for key, value in self.trigram.items()},
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data.get("eos_token", "<eos>"),
            unk_token=data.get("unk_token", "<unk>"),
            min_count=data.get("min_count", MIN_COUNT),
            smoothing_alpha=data.get("smoothing_alpha", SMOOTHING_ALPHA),
        )

        model.unigram = Counter(data.get("unigram", {}))

        model.bigram = defaultdict(
            Counter,
            {key: Counter(value) for key, value in data.get("bigram", {}).items()},
        )

        model.trigram = defaultdict(
            Counter,
            {key: Counter(value) for key, value in data.get("trigram", {}).items()},
        )

        model.vocabulary = data.get("vocabulary", [])
        model.finalized = data.get("finalized", False)

        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------- Image -> dataset bias -----------------------
#
# The camera feature no longer hashes pixels into a random high-dimensional
# "lexical space" via an untrained projection matrix. Instead it computes a
# few simple, interpretable image statistics (brightness, warmth, greenness),
# maps each one to a short list of descriptor words, keeps only the
# descriptor words that actually occur in the trained corpus, and biases
# generation using those words' *real* bigram follow-on statistics.


def extract_image_descriptors(image: Any) -> Dict[str, float]:
    if image is None:
        return {}

    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    image = np.asarray(image).astype(np.float32)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    r = float(image[:, :, 0].mean()) / 255.0
    g = float(image[:, :, 1].mean()) / 255.0
    b = float(image[:, :, 2].mean()) / 255.0

    brightness = (r + g + b) / 3.0
    warmth = r - b
    greenness = g - (r + b) / 2.0

    return {
        "bright": max(0.0, (brightness - 0.5) * 2.0),
        "dark": max(0.0, (0.5 - brightness) * 2.0),
        "warm": max(0.0, warmth),
        "cool": max(0.0, -warmth),
        "green": max(0.0, greenness),
    }


def image_seed_words(model: NGramModel, image: Any) -> Dict[str, float]:
    descriptors = extract_image_descriptors(image)
    seed_weights: Dict[str, float] = {}

    for descriptor_key, intensity in descriptors.items():
        if intensity <= 0:
            continue

        for word in IMAGE_DESCRIPTOR_WORDS.get(descriptor_key, []):
            if word in model.unigram:  # only use words that exist in the corpus
                seed_weights[word] = seed_weights.get(word, 0.0) + intensity

    return seed_weights


# ------------------- Globals & Inference Helpers --------------------

TEXT_MODEL: NGramModel | None = None
CORPUS_SEARCH: CorpusSearch | None = None
CORPUS_TEXT_CACHE: str | None = None


def load_text_model() -> NGramModel:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Upload a corpus and click 'Train model' first."
        )
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
        if default_path.exists():
            corpus_text = default_path.read_text(encoding="utf-8", errors="replace")
            CORPUS_TEXT_CACHE = corpus_text
        else:
            corpus_text = ""

    CORPUS_SEARCH = load_corpus_search(corpus_text)


def format_matches(prompt: str, limit: int = 5) -> str:
    if CORPUS_SEARCH is None or not CORPUS_SEARCH.references:
        return "No corpus file loaded."

    candidates = CORPUS_SEARCH.analyze(prompt, limit=limit)
    if not candidates:
        return "No corpus matches found."

    lines = []
    for candidate in candidates:
        lines.append(
            f"{candidate.rank}. "
            f"{candidate.sentence} "
            f"(score={candidate.score:.3f}, "
            f"overlap={candidate.symbolic_overlap:.3f}, "
            f"vector={candidate.vector_similarity:.3f})"
        )
    return "\n".join(lines)


def train_model_from_file(corpus_file):
    global CORPUS_TEXT_CACHE

    if corpus_file is None:
        return "No corpus file uploaded.", "", ""

    if hasattr(corpus_file, "name"):
        path = Path(corpus_file.name)
    else:
        path = Path(corpus_file)

    if not path.exists():
        return f"Corpus file not found: {path}", "", ""

    corpus_text = path.read_text(encoding="utf-8", errors="replace")
    CORPUS_TEXT_CACHE = corpus_text

    model = NGramModel()
    model.ingest_text(corpus_text)
    model.finalize()

    model_path = Path(MODEL_PATH)
    model.save_json(model_path)

    reload_globals()

    sample_lines = "\n".join(corpus_text.splitlines()[:5])

    summary = (
        f"Vocabulary: {len(model.vocabulary)}\n"
        f"Unigrams: {len(model.unigram)}\n"
        f"Bigram contexts: {len(model.bigram)}\n"
        f"Trigram contexts: {len(model.trigram)}"
    )

    return f"Model trained and saved to {MODEL_PATH}.", sample_lines, summary


def process_camera_image(image, user_prompt: str):
    if TEXT_MODEL is None or CORPUS_SEARCH is None:
        return (
            image,
            "Model not loaded. Upload a corpus and click 'Train model' first.",
            "",
            "",
        )

    seed_weights = image_seed_words(TEXT_MODEL, image)
    bias = TEXT_MODEL.cooccurrence_bias(seed_weights)

    prompt = user_prompt.strip() if user_prompt and user_prompt.strip() else ""

    generated = TEXT_MODEL.generate(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        bias=bias,
        bias_weight=IMAGE_BIAS_WEIGHT,
    )

    if prompt:
        corpus_matches = format_matches(prompt, limit=5)
    else:
        corpus_matches = "No prompt provided for corpus search."

    if seed_weights:
        bias_summary = (
            "Image-derived seed words (found in corpus): "
            + ", ".join(f"{word} ({weight:.2f})" for word, weight in seed_weights.items())
        )
    else:
        bias_summary = "No image-derived seed words matched the corpus vocabulary."

    return image, bias_summary, corpus_matches, generated


# --------------------------- UI -------------------------------------

with gr.Blocks(title="Stochastic N-Gram Model") as demo:
    gr.Markdown(
        """
# Stochastic N-Gram Model

1. Upload any text file as corpus.
2. Click **Train model**.
3. Optionally use the webcam/upload -- simple image stats (brightness /
   warmth / greenness) bias generation toward words that actually follow
   related seed words in your corpus.

Every probability used for generation is computed directly from n-gram
counts observed in the corpus; text is sampled stochastically from that
distribution, so re-running the same prompt can give different results.
"""
    )

    gr.Markdown("## 1. Corpus & Training")

    with gr.Row():
        with gr.Column(scale=1):
            corpus_file_input = gr.File(
                label="Corpus file (any type)",
                file_types=["file"],
            )
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

    gr.Markdown("## 2. Camera + Stochastic Generation")

    with gr.Row():
        with gr.Column(scale=1):
            camera = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Camera image",
                webcam_options=gr.WebcamOptions(mirror=False),
            )
            user_prompt = gr.Textbox(label="Optional text prompt", lines=3)
            recognize_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            processed_image = gr.Image(label="Processed image", type="numpy")
            feature_tokens_output = gr.Textbox(
                label="Image-derived bias words", lines=2
            )
            corpus_output = gr.Textbox(label="Corpus matches", lines=6)
            generated_output = gr.Textbox(label="Generated text", lines=8)

    recognize_button.click(
        fn=process_camera_image,
        inputs=[camera, user_prompt],
        outputs=[
            processed_image,
            feature_tokens_output,
            corpus_output,
            generated_output,
        ],
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

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
