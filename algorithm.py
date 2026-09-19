from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

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

# When True, every cosine similarity in this file ignores a global sign flip
# of either vector (cos(a, b) == cos(a, -b) == cos(-a, b)), i.e. it returns
# |cos|, in [0, 1]. Set to False to get the plain signed cosine in [-1, 1].
SIGN_INVARIANT_SIMILARITY = True

# Chain-of-thought structuring (see ChainOfThought below).
COT_STEPS = 4
COT_STATE_DECAY = 0.5        # how much of the earlier reasoning state carries forward
COT_MIN_RELEVANCE = 0.05     # a step must be at least this similar to the running state
COT_REDUNDANCY_LIMIT = 0.9   # skip steps this similar to an already-chosen step
COT_TOP_N = 3                # sample the next step from the best N candidates

# Reasoning -> generation modifiers (see ReasoningSubstrate below).
COT_SEED_WORDS = 12          # how many reasoning-state words become bias seeds
COT_BIAS_WEIGHT = 0.6        # weight of reasoning seeds relative to image seeds
GENERATE_REFRESH_EVERY = 8   # re-derive the bias from the evolving state every N tokens

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


def cosine_similarity(
    a: Dict[str, float],
    b: Dict[str, float],
    sign_invariant: bool = SIGN_INVARIANT_SIMILARITY,
) -> float:
    """Cosine similarity between two sparse vectors (dicts).

    With sign_invariant=True (default) the result is |cos|, so a global
    sign flip of either vector doesn't change it. Works for signed vectors
    too (e.g. the image descriptor axes), not just non-negative counts.
    """
    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)

    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    value = dot / (norm_a * norm_b)

    return abs(value) if sign_invariant else value


def align_sign(
    a: Dict[str, float], b: Dict[str, float]
) -> Tuple[Dict[str, float], float]:
    """Flip b (if needed) so that dot(a, b) >= 0. Returns (aligned_b, sign)."""
    dot = sum(a[key] * b[key] for key in set(a) & set(b))
    sign = 1.0 if dot >= 0 else -1.0

    return {key: sign * value for key, value in b.items()}, sign


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

    The cosine term is sign-invariant by default (see SIGN_INVARIANT_SIMILARITY).
    """

    def __init__(
        self,
        lexical_weight: float = 0.5,
        vector_weight: float = 0.5,
        sign_invariant: bool = SIGN_INVARIANT_SIMILARITY,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.sign_invariant = sign_invariant
        self.references: List[CorpusReference] = []
        self.doc_freq: Counter = Counter()

    def idf(self, token: str) -> float:
        """How distinctive a word is, from real sentence counts in the corpus.
        A word in every sentence gets 0; rarer words get more.
        """
        n = len(self.references)
        return math.log((1 + n) / (1 + self.doc_freq.get(token, 0)))

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(sentence.lower() for sentence in sentences)

        self.references = []
        self.doc_freq = Counter()

        for sentence in sentences:
            tokens = tokenize(sentence)

            if not tokens:
                continue

            bow = bag_of_words(tokens)
            self.doc_freq.update(bow.keys())

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
            vector_similarity = cosine_similarity(
                prompt_vector, reference.vector, sign_invariant=self.sign_invariant
            )

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
        bias_provider: Optional[
            Callable[[List[str]], Tuple[Optional[Dict[str, float]], float]]
        ] = None,
        refresh_every: int = GENERATE_REFRESH_EVERY,
    ) -> str:
        generated = tokenize(prompt)

        for step in range(max_new_tokens):
            # Optionally re-derive the bias from the text generated so far.
            if bias_provider is not None and step % max(1, refresh_every) == 0:
                bias, bias_weight = bias_provider(generated)

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


# ------------------------- Chain of thought --------------------------
#
# Structured reasoning built from the corpus itself. Each step is a real
# corpus sentence, chosen by sign-invariant cosine similarity against a
# running "reasoning state". The state compounds: after every step it is
# decayed and the chosen sentence's vector is added, so later steps follow
# from everything said so far, not just the prompt. A step is skipped if it
# is nearly identical (also by sign-invariant cosine) to a step already
# taken, which keeps the chain moving forward instead of repeating itself.


@dataclass
class ThoughtStep:
    index: int
    sentence: str
    relevance: float  # cosine to the running state when chosen
    novelty: float    # 1 - max cosine to earlier steps (1.0 for the first step)


class ChainOfThought:
    def __init__(
        self,
        search: CorpusSearch,
        decay: float = COT_STATE_DECAY,
        min_relevance: float = COT_MIN_RELEVANCE,
        redundancy_limit: float = COT_REDUNDANCY_LIMIT,
        top_n: int = COT_TOP_N,
    ) -> None:
        self.search = search
        self.decay = decay
        self.min_relevance = min_relevance
        self.redundancy_limit = redundancy_limit
        self.top_n = max(1, top_n)
        self.last_state: Dict[str, float] = {}

    def _cos(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return cosine_similarity(a, b, sign_invariant=self.search.sign_invariant)

    def run(self, prompt: str, steps: int = COT_STEPS) -> List[ThoughtStep]:
        state: Dict[str, float] = {
            token: float(count)
            for token, count in bag_of_words(tokenize(prompt)).items()
        }
        chosen_vectors: List[Dict[str, float]] = []
        used: set = set()
        chain: List[ThoughtStep] = []

        for _ in range(steps):
            scored: List[Tuple[float, int, float, float]] = []

            for idx, reference in enumerate(self.search.references):
                if idx in used:
                    continue

                relevance = self._cos(state, reference.vector)

                if relevance < self.min_relevance:
                    continue

                overlap = max(
                    (self._cos(reference.vector, v) for v in chosen_vectors),
                    default=0.0,
                )

                if overlap >= self.redundancy_limit:
                    continue

                novelty = 1.0 - overlap
                scored.append((relevance * novelty, idx, relevance, novelty))

            if not scored:
                break

            scored.sort(key=lambda item: item[0], reverse=True)
            pool = scored[: self.top_n]

            # Stochastic like the rest of the app, but only among the best few.
            _, idx, relevance, novelty = random.choices(
                pool, weights=[item[0] for item in pool], k=1
            )[0]

            reference = self.search.references[idx]
            used.add(idx)
            chosen_vectors.append(reference.vector)

            chain.append(
                ThoughtStep(
                    index=len(chain) + 1,
                    sentence=reference.sentence,
                    relevance=relevance,
                    novelty=novelty,
                )
            )

            # Compound the state: decay what we had, add what we just said.
            state = {token: self.decay * weight for token, weight in state.items()}
            for token, weight in reference.vector.items():
                state[token] = state.get(token, 0.0) + weight

        self.last_state = state

        return chain


def format_chain(chain: List[ThoughtStep]) -> str:
    if not chain:
        return "No reasoning steps found for that prompt."

    lines = []
    for step in chain:
        lines.append(
            f"Step {step.index}: {step.sentence}.  "
            f"(relevance={step.relevance:.3f}, novelty={step.novelty:.3f})"
        )
    lines.append(f"Conclusion: {chain[-1].sentence}.")

    return "\n".join(lines)


def merge_seed_weights(
    base: Dict[str, float], extra: Dict[str, float], extra_weight: float
) -> Dict[str, float]:
    merged = dict(base)

    for word, weight in extra.items():
        merged[word] = merged.get(word, 0.0) + extra_weight * weight

    return merged


class ReasoningSubstrate:
    """Feeds the reasoning state into generation as corpus modifiers.

    1. Seeds: the most distinctive words in the reasoning state (state weight
       x idf, so "the"/"is" drop out) become seed words. They go through the
       same NGramModel.cooccurrence_bias as the image seeds, i.e. real bigram
       follow-on counts from the corpus.
    2. Compounding: as text is generated, the state is decayed and the new
       tokens are added, so the seeds keep tracking what has been said.
    3. Adaptive strength: the sign-invariant cosine between the evolving state
       and the original reasoning state measures how far generation has
       drifted. The bias weight is scaled by 1 + (1 - alignment), so it
       pulls harder as the text drifts away from the reasoning.
    """

    def __init__(
        self,
        model: NGramModel,
        search: CorpusSearch,
        initial_state: Dict[str, float],
        prompt_length: int = 0,
        decay: float = COT_STATE_DECAY,
        top_words: int = COT_SEED_WORDS,
    ) -> None:
        self.model = model
        self.search = search
        self.anchor = dict(initial_state)
        self.state = dict(initial_state)
        self.decay = decay
        self.top_words = top_words
        self._seen = prompt_length  # prompt tokens are already in the state

    def seed_weights(self) -> Dict[str, float]:
        weighted = {
            token: weight * self.search.idf(token)
            for token, weight in self.state.items()
            if token in self.model.unigram and token not in IGNORED_TOKENS
        }
        weighted = {token: w for token, w in weighted.items() if w > 0}

        top = sorted(weighted.items(), key=lambda item: item[1], reverse=True)
        top = top[: self.top_words]

        if not top:
            return {}

        peak = top[0][1]

        return {token: weight / peak for token, weight in top}

    def alignment(self) -> float:
        return cosine_similarity(
            self.state, self.anchor, sign_invariant=self.search.sign_invariant
        )

    def update(self, new_tokens: List[str]) -> None:
        if not new_tokens:
            return

        self.state = {token: self.decay * w for token, w in self.state.items()}

        for token, count in bag_of_words(new_tokens).items():
            self.state[token] = self.state.get(token, 0.0) + float(count)

    def provider(
        self, base_seeds: Dict[str, float], base_weight: float
    ) -> Callable[[List[str]], Tuple[Optional[Dict[str, float]], float]]:
        """Build the callable NGramModel.generate uses to refresh its bias."""

        def refresh(generated: List[str]) -> Tuple[Optional[Dict[str, float]], float]:
            self.update(generated[self._seen :])
            self._seen = len(generated)

            seeds = merge_seed_weights(base_seeds, self.seed_weights(), COT_BIAS_WEIGHT)
            scale = 1.0 + (1.0 - self.alignment())

            return self.model.cooccurrence_bias(seeds), base_weight * scale

        return refresh


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

    prompt = user_prompt.strip() if user_prompt and user_prompt.strip() else ""

    # Reasoning substrate: run the chain over the prompt, then let its
    # compounded state drive the generation modifiers alongside the image.
    chain: List[ThoughtStep] = []
    substrate: Optional[ReasoningSubstrate] = None

    if prompt and CORPUS_SEARCH.references:
        cot = ChainOfThought(CORPUS_SEARCH)
        chain = cot.run(prompt, steps=COT_STEPS)
        substrate = ReasoningSubstrate(
            TEXT_MODEL,
            CORPUS_SEARCH,
            cot.last_state,
            prompt_length=len(tokenize(prompt)),
        )

    reasoning_seeds = substrate.seed_weights() if substrate else {}
    combined_seeds = merge_seed_weights(seed_weights, reasoning_seeds, COT_BIAS_WEIGHT)
    bias = TEXT_MODEL.cooccurrence_bias(combined_seeds)

    generated = TEXT_MODEL.generate(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        bias=bias,
        bias_weight=IMAGE_BIAS_WEIGHT,
        bias_provider=(
            substrate.provider(seed_weights, IMAGE_BIAS_WEIGHT) if substrate else None
        ),
    )

    if prompt:
        corpus_matches = format_matches(prompt, limit=5)
        if chain:
            corpus_matches += "\n\nReasoning chain used for modifiers:\n" + format_chain(chain)
    else:
        corpus_matches = "No prompt provided for corpus search."

    bias_lines = []
    if seed_weights:
        bias_lines.append(
            "Image-derived seed words (found in corpus): "
            + ", ".join(f"{word} ({weight:.2f})" for word, weight in seed_weights.items())
        )
    else:
        bias_lines.append("No image-derived seed words matched the corpus vocabulary.")

    if reasoning_seeds:
        bias_lines.append(
            "Reasoning seed words (refreshed during generation): "
            + ", ".join(f"{word} ({weight:.2f})" for word, weight in reasoning_seeds.items())
        )

    bias_summary = "\n".join(bias_lines)

    return image, bias_summary, corpus_matches, generated


def run_chain_of_thought(prompt: str, n_steps: int) -> str:
    if CORPUS_SEARCH is None or not CORPUS_SEARCH.references:
        return "No corpus loaded. Upload a corpus and click 'Train model' first."

    if not prompt or not prompt.strip():
        return "Enter a prompt first."

    chain = ChainOfThought(CORPUS_SEARCH).run(prompt, steps=int(n_steps))

    return format_chain(chain)


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
                label="Bias words (image + reasoning)", lines=4
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

    gr.Markdown("## 3. Structured reasoning (chain of thought)")

    with gr.Row():
        with gr.Column(scale=1):
            cot_prompt = gr.Textbox(label="Question or topic", lines=3)
            cot_steps = gr.Slider(
                minimum=1, maximum=8, value=COT_STEPS, step=1, label="Steps"
            )
            cot_button = gr.Button("Reason", variant="primary")

        with gr.Column(scale=2):
            cot_output = gr.Textbox(label="Reasoning chain", lines=12)

    cot_button.click(
        fn=run_chain_of_thought,
        inputs=[cot_prompt, cot_steps],
        outputs=[cot_output],
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
