from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = "model_cuda.json"
BINDINGS_PATH = "bindings_cuda.json"
CELLULAR_PATH = "cellular_cuda.pt"
DEFAULT_CORPUS = "corpus.txt"

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

SPECIAL = {
    BOS,
    EOS,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CUDA = DEVICE.type == "cuda"

ALPHA = 0.05
DEFAULT_TEMP = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW = 200
DEFAULT_MAX_SUBSET = 5

DEFAULT_CA_STEPS = 6
DEFAULT_CA_WIDTH = 48
DEFAULT_CA_BIAS_SCALE = 0.55
DEFAULT_CA_NOISE = 0.005

DEFAULT_BINDING_THRESHOLD = 0.35
DEFAULT_BINDING_MOMENTUM = 0.85

Vec = Dict[str, float]


# ---------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(
    r"""
    [A-Za-z_]\w*
    |\d+(?:\.\d+)?
    |"(?:\\.|[^"\\])*"
    |'(?:\\.|[^'\\])*'
    |==|!=|<=|>=|->|:=|\+=|-=|\*=|/=|//=|\*\*
    |&&|\|\||<<|>>|&=|\|=|\^=
    |[{}\[\]():,.;+\-*/%<>=!?]
    |[^\s]
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> List[str]:
    """
    Tokenises ordinary text and source code.

    Punctuation, operators, identifiers, numbers and quoted strings
    remain separate tokens.
    """
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
    ]


def sentences(text: str) -> List[str]:
    """
    Produces corpus units.

    For prose, paragraphs are split into sentence-like units.
    For code, non-empty lines are retained, which avoids destroying
    constructs such as object.method() and decimal literals.
    """
    output = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            continue

        stripped = line.strip()

        if stripped.startswith(
            (
                "def ",
                "class ",
                "if ",
                "elif ",
                "else:",
                "for ",
                "while ",
                "try:",
                "except",
                "finally:",
                "with ",
                "return ",
                "import ",
                "from ",
                "#",
            )
        ):
            output.append(line)
            continue

        parts = re.split(r"(?<=[.!?])\s+", line.strip())

        for part in parts:
            part = part.strip()

            if not part:
                continue

            if part[-1] not in ".!?:":
                part += "."

            output.append(part)

    return output


def bow(tokens: List[str]) -> Counter:
    return Counter(
        token
        for token in tokens
        if token not in SPECIAL
    )


# ---------------------------------------------------------------------
# CUDA tensor helpers
# ---------------------------------------------------------------------

def tensor_vector(values: List[float]) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.float32,
        device=DEVICE,
    )


def matrix(
    vectors: List[Vec],
    keys: List[str],
) -> torch.Tensor:
    if not vectors:
        return torch.empty(
            (0, len(keys)),
            dtype=torch.float32,
            device=DEVICE,
        )

    return torch.tensor(
        [
            [
                float(vector.get(key, 0.0))
                for key in keys
            ]
            for vector in vectors
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


def gpu_cosine(
    query: Vec,
    vectors: List[Vec],
) -> torch.Tensor:
    if not vectors:
        return torch.empty(
            0,
            dtype=torch.float32,
            device=DEVICE,
        )

    keys = sorted(
        set(query).union(
            *(vector.keys() for vector in vectors)
        )
    )

    if not keys:
        return torch.zeros(
            len(vectors),
            dtype=torch.float32,
            device=DEVICE,
        )

    query_tensor = tensor_vector(
        [
            query.get(key, 0.0)
            for key in keys
        ]
    )

    vector_tensor = matrix(vectors, keys)

    numerator = vector_tensor @ query_tensor

    denominator = (
        torch.linalg.vector_norm(
            vector_tensor,
            dim=1,
        )
        * torch.linalg.vector_norm(
            query_tensor,
        )
    ).clamp_min(1e-12)

    return numerator / denominator


def gpu_cosine_matrix(
    vectors: List[Vec],
) -> torch.Tensor:
    if not vectors:
        return torch.empty(
            (0, 0),
            dtype=torch.float32,
            device=DEVICE,
        )

    keys = sorted(
        set().union(
            *(vector.keys() for vector in vectors)
        )
    )

    if not keys:
        return torch.eye(
            len(vectors),
            dtype=torch.float32,
            device=DEVICE,
        )

    values = matrix(vectors, keys)

    norms = torch.linalg.vector_norm(
        values,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)

    normalised = values / norms

    return normalised @ normalised.T


def gpu_gain(
    target: Vec,
    vectors: List[Vec],
) -> torch.Tensor:
    if not vectors:
        return torch.empty(
            0,
            dtype=torch.float32,
            device=DEVICE,
        )

    keys = sorted(target)

    if not keys:
        return torch.zeros(
            len(vectors),
            dtype=torch.float32,
            device=DEVICE,
        )

    target_tensor = tensor_vector(
        [
            target[key]
            for key in keys
        ]
    )

    vector_tensor = matrix(vectors, keys)

    return torch.minimum(
        vector_tensor,
        target_tensor,
    ).sum(dim=1)


# ---------------------------------------------------------------------
# Adaptive token bindings
# ---------------------------------------------------------------------

class Bindings:
    def __init__(
        self,
        threshold: float = DEFAULT_BINDING_THRESHOLD,
        momentum: float = DEFAULT_BINDING_MOMENTUM,
    ):
        self.threshold = threshold
        self.momentum = momentum
        self.contexts: Dict[str, Counter] = {}
        self.links: Dict[str, Dict[str, float]] = {}

    def update(
        self,
        text: str,
        incremental: bool = False,
    ) -> None:
        incoming = defaultdict(Counter)

        for line in sentences(text):
            tokens = tokenize(line)

            for index, word in enumerate(tokens):
                start = max(0, index - 4)
                end = min(len(tokens), index + 5)

                incoming[word].update(
                    tokens[start:end]
                )

        if incremental and self.contexts:
            merged = defaultdict(Counter)

            for word, values in self.contexts.items():
                merged[word].update(values)

            for word, values in incoming.items():
                for key, value in values.items():
                    merged[word][key] = (
                        self.momentum * merged[word][key]
                        + (1.0 - self.momentum) * value
                    )

            self.contexts = dict(merged)
        else:
            self.contexts = dict(incoming)

        self.rebuild()

    def rebuild(self) -> None:
        words = list(self.contexts)

        if not words:
            self.links = {}
            return

        vectors = [
            {
                key: float(value)
                for key, value in self.contexts[word].items()
            }
            for word in words
        ]

        scores = gpu_cosine_matrix(vectors)
        links = {}

        for index, word in enumerate(words):
            row = scores[index].clone()
            row[index] = -1.0

            count = min(8, max(1, len(words) - 1))

            values, indices = torch.topk(
                row,
                k=count,
            )

            result = {}

            for value, other_index in zip(
                values.detach().cpu().tolist(),
                indices.detach().cpu().tolist(),
            ):
                if value >= self.threshold:
                    result[words[other_index]] = float(value)

            links[word] = result

        self.links = links

    def expand(self, vector: Vec) -> Vec:
        output = defaultdict(float)

        for word, value in vector.items():
            output[word] += value

            for linked_word, similarity in self.links.get(
                word,
                {},
            ).items():
                output[linked_word] += value * similarity

        return dict(output)

    def save(self) -> None:
        data = {
            "threshold": self.threshold,
            "momentum": self.momentum,
            "links": self.links,
            "contexts": {
                word: dict(values)
                for word, values in self.contexts.items()
            },
        }

        Path(BINDINGS_PATH).write_text(
            json.dumps(data, indent=2),
            encoding="utf8",
        )

    @classmethod
    def load(cls) -> "Bindings":
        path = Path(BINDINGS_PATH)

        if not path.exists():
            return cls()

        data = json.loads(
            path.read_text(encoding="utf8")
        )

        instance = cls(
            threshold=data.get(
                "threshold",
                DEFAULT_BINDING_THRESHOLD,
            ),
            momentum=data.get(
                "momentum",
                DEFAULT_BINDING_MOMENTUM,
            ),
        )

        instance.links = data.get("links", {})

        instance.contexts = {
            word: Counter(values)
            for word, values in data.get(
                "contexts",
                {},
            ).items()
        }

        return instance

    def summary(self) -> str:
        return (
            f"Device: {DEVICE}\n"
            f"Binding tokens: {len(self.links)}\n"
            f"Binding links: "
            f"{sum(len(value) for value in self.links.values())}"
        )


# ---------------------------------------------------------------------
# Corpus retrieval
# ---------------------------------------------------------------------

class Ref:
    def __init__(
        self,
        index: int,
        source: str,
        tokens: List[str],
        frequency: int,
        features: Vec,
    ):
        self.index = index
        self.source = source
        self.tokens = tokens
        self.frequency = frequency
        self.features = features


class Match:
    def __init__(
        self,
        rank: int,
        reference: Ref,
        score: float,
        vector_score: float,
        selected: bool,
        endpoint: bool,
    ):
        self.rank = rank
        self.reference = reference
        self.score = score
        self.vector_score = vector_score
        self.selected = selected
        self.endpoint = endpoint


class Corpus:
    def __init__(
        self,
        text: str = "",
        bindings: Optional[Bindings] = None,
    ):
        self.bindings = bindings or Bindings()
        self.references: List[Ref] = []

        if text:
            self.bindings.update(text)

        units = sentences(text)
        frequency = Counter(
            unit.lower()
            for unit in units
        )

        for index, source in enumerate(units):
            tokens = tokenize(source)

            features = self.bindings.expand(
                {
                    key: float(value)
                    for key, value in bow(tokens).items()
                }
            )

            self.references.append(
                Ref(
                    index=index,
                    source=source,
                    tokens=tokens,
                    frequency=frequency[source.lower()],
                    features=features,
                )
            )

    def search(
        self,
        prompt: str,
        limit: int = 8,
        max_subset: int = DEFAULT_MAX_SUBSET,
    ) -> List[Match]:
        target = self.bindings.expand(
            {
                key: float(value)
                for key, value in bow(
                    tokenize(prompt)
                ).items()
            }
        )

        if not self.references:
            return []

        vectors = [
            reference.features
            for reference in self.references
        ]

        cosine_scores = gpu_cosine(
            target,
            vectors,
        )

        gains = gpu_gain(
            target,
            vectors,
        )

        selected = []
        covered = defaultdict(float)

        for _ in range(
            min(max_subset, len(self.references))
        ):
            residual = {
                key: max(
                    0.0,
                    value - covered[key],
                )
                for key, value in target.items()
            }

            residual_gains = gpu_gain(
                residual,
                vectors,
            )

            if selected:
                residual_gains[
                    torch.tensor(
                        selected,
                        dtype=torch.long,
                        device=DEVICE,
                    )
                ] = 0.0

            if residual_gains.numel() == 0:
                break

            index = int(
                torch.argmax(
                    residual_gains
                ).item()
            )

            if float(residual_gains[index]) <= 0.0:
                break

            selected.append(index)

            for key, value in self.references[
                index
            ].features.items():
                covered[key] += value

        prompt_tokens = set(tokenize(prompt))
        rows = []

        for index, reference in enumerate(
            self.references
        ):
            overlap_tokens = (
                prompt_tokens
                & set(reference.tokens)
            )

            union_tokens = (
                prompt_tokens
                | set(reference.tokens)
            )

            overlap = len(overlap_tokens) / max(
                1,
                len(union_tokens),
            )

            score = (
                0.62 * float(cosine_scores[index])
                + 0.25 * overlap
                + 0.13 * math.log1p(
                    reference.frequency
                )
            )

            if (
                index in selected
                or float(gains[index]) > 0.0
                or overlap > 0.0
            ):
                rows.append(
                    Match(
                        rank=0,
                        reference=reference,
                        score=score,
                        vector_score=float(
                            cosine_scores[index]
                        ),
                        selected=index in selected,
                        endpoint=index not in selected,
                    )
                )

        rows.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        for rank, item in enumerate(
            rows[:limit],
            start=1,
        ):
            item.rank = rank

        return rows[:limit]


# ---------------------------------------------------------------------
# Neural cellular automaton
# ---------------------------------------------------------------------

class CellularTextField:
    """
    A one-dimensional neural cellular automaton.

    Each token is a cell. Every cell is updated from:
      - its left neighbour,
      - its own state,
      - its right neighbour,
      - the global sequence context.

    All dense state evolution occurs on DEVICE.
    """

    def __init__(
        self,
        vocabulary: Dict[str, int],
        width: int = DEFAULT_CA_WIDTH,
        steps: int = DEFAULT_CA_STEPS,
        noise: float = DEFAULT_CA_NOISE,
    ):
        self.vocabulary = vocabulary
        self.inverse_vocabulary = [
            word
            for word, _ in sorted(
                vocabulary.items(),
                key=lambda item: item[1],
            )
        ]

        self.width = width
        self.steps = steps
        self.noise = noise

        scale = 1.0 / math.sqrt(width)

        self.embedding = (
            torch.randn(
                len(self.inverse_vocabulary),
                width,
                device=DEVICE,
            )
            * 0.05
        )

        self.left = (
            torch.randn(
                width,
                width,
                device=DEVICE,
            )
            * scale
        )

        self.self_projection = (
            torch.randn(
                width,
                width,
                device=DEVICE,
            )
            * scale
        )

        self.right = (
            torch.randn(
                width,
                width,
                device=DEVICE,
            )
            * scale
        )

        self.context_projection = (
            torch.randn(
                width,
                width,
                device=DEVICE,
            )
            * scale
        )

        self.update_gate = torch.randn(
            width,
            device=DEVICE,
        ) * 0.02

        self.output_projection = (
            torch.randn(
                width,
                width,
                device=DEVICE,
            )
            * scale
        )

    def encode(
        self,
        tokens: List[str],
    ) -> torch.Tensor:
        unknown = self.vocabulary.get(
            UNK,
            0,
        )

        ids = [
            self.vocabulary.get(
                token,
                unknown,
            )
            for token in tokens
        ]

        if not ids:
            return torch.empty(
                (0, self.width),
                dtype=torch.float32,
                device=DEVICE,
            )

        token_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=DEVICE,
        )

        return self.embedding[token_ids]

    def perceive(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape[0] == 0:
            return state

        left = torch.roll(
            state,
            shifts=1,
            dims=0,
        )

        right = torch.roll(
            state,
            shifts=-1,
            dims=0,
        )

        left[0] = state[0]
        right[-1] = state[-1]

        local = (
            left @ self.left.T
            + state @ self.self_projection.T
            + right @ self.right.T
        )

        return torch.tanh(local)

    def evolve(
        self,
        tokens: List[str],
        context: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        state = self.encode(tokens)

        if state.shape[0] == 0:
            return state

        if context is None:
            context = torch.zeros(
                self.width,
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            context = context.to(DEVICE)

        context_signal = (
            context @ self.context_projection.T
        )

        actual_steps = (
            self.steps
            if steps is None
            else max(0, int(steps))
        )

        for _ in range(actual_steps):
            perception = self.perceive(state)

            candidate = torch.tanh(
                perception
                + context_signal.unsqueeze(0)
            )

            gate = torch.sigmoid(
                self.update_gate
            ).unsqueeze(0)

            state = (
                (1.0 - gate) * state
                + gate * candidate
            )

            if self.noise > 0.0:
                state = state + (
                    torch.randn_like(state)
                    * self.noise
                )

        return state

    def context_vector(
        self,
        tokens: List[str],
    ) -> torch.Tensor:
        state = self.encode(tokens)

        if state.shape[0] == 0:
            return torch.zeros(
                self.width,
                dtype=torch.float32,
                device=DEVICE,
            )

        return state.mean(dim=0)

    def token_bias(
        self,
        tokens: List[str],
        context: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        state = self.evolve(
            tokens,
            context=context,
            steps=steps,
        )

        if state.shape[0] == 0:
            return torch.zeros(
                len(self.inverse_vocabulary),
                dtype=torch.float32,
                device=DEVICE,
            )

        current = (
            state[-1]
            @ self.output_projection.T
        )

        scores = self.embedding @ current

        return torch.tanh(scores)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embedding.detach().cpu(),
            "left": self.left.detach().cpu(),
            "self_projection": self.self_projection.detach().cpu(),
            "right": self.right.detach().cpu(),
            "context_projection": self.context_projection.detach().cpu(),
            "update_gate": self.update_gate.detach().cpu(),
            "output_projection": self.output_projection.detach().cpu(),
        }

    def load_state_dict(
        self,
        state: Dict[str, torch.Tensor],
    ) -> None:
        names = (
            "embedding",
            "left",
            "self_projection",
            "right",
            "context_projection",
            "update_gate",
            "output_projection",
        )

        for name in names:
            if name in state:
                setattr(
                    self,
                    name,
                    state[name].to(DEVICE),
                )

    def save(self) -> None:
        torch.save(
            {
                "width": self.width,
                "steps": self.steps,
                "noise": self.noise,
                "vocabulary": self.vocabulary,
                "state": self.state_dict(),
            },
            CELLULAR_PATH,
        )

    @classmethod
    def load_if_compatible(
        cls,
        vocabulary: Dict[str, int],
        width: int,
        steps: int,
        noise: float,
    ) -> "CellularTextField":
        field = cls(
            vocabulary=vocabulary,
            width=width,
            steps=steps,
            noise=noise,
        )

        path = Path(CELLULAR_PATH)

        if not path.exists():
            return field

        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            saved_vocabulary = checkpoint.get(
                "vocabulary",
                {},
            )

            if saved_vocabulary != vocabulary:
                return field

            saved_state = checkpoint.get(
                "state",
                {},
            )

            field.load_state_dict(saved_state)

        except Exception:
            return field

        return field


# ---------------------------------------------------------------------
# N-gram model
# ---------------------------------------------------------------------

class NGram:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocabulary: List[str] = []
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.cellular: Optional[CellularTextField] = None

        self.temperature = DEFAULT_TEMP
        self.top_k = DEFAULT_TOP_K
        self.max_new = DEFAULT_MAX_NEW
        self.ca_steps = DEFAULT_CA_STEPS
        self.ca_bias_scale = DEFAULT_CA_BIAS_SCALE

    def ingest(self, text: str) -> "NGram":
        self.unigram.clear()
        self.bigram.clear()
        self.trigram.clear()

        for unit in sentences(text):
            sequence = [
                BOS,
                BOS,
                *tokenize(unit),
                EOS,
            ]

            self.unigram.update(sequence)

            for first, second in zip(
                sequence,
                sequence[1:],
            ):
                self.bigram[first][second] += 1

            for first, second, third in zip(
                sequence,
                sequence[1:],
                sequence[2:],
            ):
                self.trigram[
                    (first, second)
                ][third] += 1

        self.inverse_vocabulary = sorted(
            self.unigram
        )

        if UNK not in self.inverse_vocabulary:
            self.inverse_vocabulary.append(UNK)

        self.vocab = {
            word: index
            for index, word in enumerate(
                self.inverse_vocabulary
            )
        }

        self.cellular = CellularTextField(
            vocabulary=self.vocab,
            width=DEFAULT_CA_WIDTH,
            steps=self.ca_steps,
        )

        return self

    def gpu_logits(
        self,
        history: List[str],
    ) -> torch.Tensor:
        logits = torch.full(
            (
                len(self.inverse_vocabulary),
            ),
            -float("inf"),
            dtype=torch.float32,
            device=DEVICE,
        )

        if not history:
            return logits

        if len(history) == 1:
            first = BOS
            second = history[-1]
        else:
            first, second = history[-2:]

        counts = (
            self.trigram.get(
                (first, second)
            )
            or self.bigram.get(second)
            or self.unigram
        )

        denominator = (
            sum(counts.values())
            + ALPHA * len(counts)
        )

        for word, count in counts.items():
            index = self.vocab.get(word)

            if index is None:
                continue

            probability = (
                count + ALPHA
            ) / max(
                denominator,
                1e-12,
            )

            logits[index] = math.log(
                probability
            )

        return logits

    def bias_from(
        self,
        seeds: List[str],
    ) -> Dict[str, float]:
        output = {}

        for word in seeds:
            counts = self.bigram.get(word)

            if not counts:
                continue

            total = sum(counts.values())

            if total <= 0:
                continue

            for next_word, count in counts.items():
                output[next_word] = (
                    output.get(next_word, 0.0)
                    + count / total
                )

        return output

    def model_bias(
        self,
        seeds: List[str],
    ) -> Dict[str, float]:
        return self.bias_from(seeds)

    def sample(
        self,
        history: List[str],
        bias: Optional[Dict[str, float]] = None,
        cellular_bias: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        logits = self.gpu_logits(history)

        if bias:
            for word, value in bias.items():
                index = self.vocab.get(word)

                if index is not None:
                    logits[index] += float(value)

        if cellular_bias is not None:
            logits = logits + (
                self.ca_bias_scale
                * cellular_bias
            )

        finite = torch.isfinite(logits)

        if not bool(finite.any()):
            return EOS

        logits = logits.clone()
        logits[~finite] = -float("inf")

        actual_temperature = (
            self.temperature
            if temperature is None
            else float(temperature)
        )

        actual_temperature = max(
            0.05,
            actual_temperature,
        )

        actual_top_k = (
            self.top_k
            if top_k is None
            else int(top_k)
        )

        actual_top_k = max(
            1,
            min(
                actual_top_k,
                int(finite.sum().item()),
            ),
        )

        values, indices = torch.topk(
            logits,
            k=actual_top_k,
        )

        probabilities = torch.softmax(
            values / actual_temperature,
            dim=0,
        )

        selected = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        vocabulary_index = int(
            indices[selected].item()
        )

        return self.inverse_vocabulary[
            vocabulary_index
        ]

    def generate(
        self,
        prompt: str,
        bias_function=None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new: Optional[int] = None,
        ca_steps: Optional[int] = None,
    ) -> str:
        output = [
            BOS,
            BOS,
            *tokenize(prompt),
        ]

        limit = (
            self.max_new
            if max_new is None
            else max(1, int(max_new))
        )

        for _ in range(limit):
            cellular_bias = None
            cellular_context = None

            if self.cellular is not None:
                cellular_context = (
                    self.cellular.context_vector(
                        output
                    )
                )

                cellular_bias = (
                    self.cellular.token_bias(
                        output,
                        context=cellular_context,
                        steps=ca_steps,
                    )
                )

            bias = None

            if bias_function is not None:
                bias = bias_function(
                    output,
                    cellular_context,
                )

            next_token = self.sample(
                history=output,
                bias=bias,
                cellular_bias=cellular_bias,
                temperature=temperature,
                top_k=top_k,
            )

            output.append(next_token)


        return " ".join(
            token
            for token in output
            if token not in SPECIAL
        )

    def save(self) -> None:
        data = {
            "vocabulary": self.inverse_vocabulary,
            "unigram": dict(self.unigram),
            "bigram": {
                first: dict(values)
                for first, values in self.bigram.items()
            },
            "trigram": {
                f"{first}\t{second}": dict(values)
                for (first, second), values
                in self.trigram.items()
            },
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_new": self.max_new,
            "ca_steps": self.ca_steps,
            "ca_bias_scale": self.ca_bias_scale,
        }

        Path(MODEL_PATH).write_text(
            json.dumps(data, indent=2),
            encoding="utf8",
        )

        if self.cellular is not None:
            self.cellular.save()

    @classmethod
    def load(cls) -> Optional["NGram"]:
        path = Path(MODEL_PATH)

        if not path.exists():
            return None

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf8"
                )
            )

            model = cls()

            model.inverse_vocabulary = data.get(
                "vocabulary",
                [],
            )

            model.vocab = {
                word: index
                for index, word in enumerate(
                    model.inverse_vocabulary
                )
            }

            model.unigram = Counter(
                data.get("unigram", {})
            )

            model.bigram = defaultdict(
                Counter,
                {
                    word: Counter(values)
                    for word, values in data.get(
                        "bigram",
                        {},
                    ).items()
                },
            )

            model.trigram = defaultdict(
                Counter
            )

            for key, values in data.get(
                "trigram",
                {},
            ).items():
                first, second = key.split(
                    "\t",
                    maxsplit=1,
                )

                model.trigram[
                    (first, second)
                ] = Counter(values)

            model.temperature = data.get(
                "temperature",
                DEFAULT_TEMP,
            )

            model.top_k = data.get(
                "top_k",
                DEFAULT_TOP_K,
            )

            model.max_new = data.get(
                "max_new",
                DEFAULT_MAX_NEW,
            )

            model.ca_steps = data.get(
                "ca_steps",
                DEFAULT_CA_STEPS,
            )

            model.ca_bias_scale = data.get(
                "ca_bias_scale",
                DEFAULT_CA_BIAS_SCALE,
            )

            model.cellular = (
                CellularTextField.load_if_compatible(
                    vocabulary=model.vocab,
                    width=DEFAULT_CA_WIDTH,
                    steps=model.ca_steps,
                    noise=DEFAULT_CA_NOISE,
                )
            )

            return model

        except Exception:
            return None


# ---------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------

class State:
    def __init__(self):
        self.model: Optional[NGram] = None
        self.corpus: Optional[Corpus] = None
        self.bindings = Bindings()
        self.raw_text = ""

    def train(
        self,
        text: str,
        ca_steps: int = DEFAULT_CA_STEPS,
        ca_bias_scale: float = DEFAULT_CA_BIAS_SCALE,
    ) -> None:
        self.raw_text = text

        self.bindings = Bindings()
        self.bindings.update(text)
        self.bindings.save()

        self.model = NGram()
        self.model.ca_steps = max(
            0,
            int(ca_steps),
        )
        self.model.ca_bias_scale = float(
            ca_bias_scale
        )
        self.model.ingest(text)
        self.model.save()

        self.corpus = Corpus(
            text=text,
            bindings=self.bindings,
        )

    def load_saved(self) -> bool:
        model = NGram.load()

        if model is None:
            return False

        self.model = model
        self.bindings = Bindings.load()

        corpus_path = Path(DEFAULT_CORPUS)

        if corpus_path.exists():
            self.raw_text = corpus_path.read_text(
                encoding="utf8",
                errors="replace",
            )

            self.corpus = Corpus(
                text=self.raw_text,
                bindings=self.bindings,
            )
        else:
            self.corpus = Corpus(
                text="",
                bindings=self.bindings,
            )

        return True


S = State()


# ---------------------------------------------------------------------
# Formatting and UI functions
# ---------------------------------------------------------------------

def read_uploaded_file(file_value) -> str:
    if file_value is None:
        return ""

    if isinstance(file_value, str):
        path = Path(file_value)
    elif hasattr(file_value, "name"):
        path = Path(file_value.name)
    else:
        path = Path(str(file_value))

    return path.read_text(
        encoding="utf8",
        errors="replace",
    )


def train_from_file(
    file_value,
    ca_steps,
    ca_bias_scale,
):
    if file_value is None:
        return (
            "No corpus file selected.",
            "",
            "",
        )

    try:
        text = read_uploaded_file(file_value)

        if not text.strip():
            return (
                "The selected file is empty.",
                "",
                "",
            )

        S.train(
            text=text,
            ca_steps=int(ca_steps),
            ca_bias_scale=float(ca_bias_scale),
        )

        preview = text[:3000]

        summary = (
            f"{S.bindings.summary()}\n"
            f"Cellular steps: {ca_steps}\n"
            f"Cellular bias scale: {ca_bias_scale}\n"
            f"N-gram vocabulary: "
            f"{len(S.model.inverse_vocabulary)}\n"
            f"Corpus units: "
            f"{len(S.corpus.references)}"
        )

        return (
            "Training complete.",
            preview,
            summary,
        )

    except Exception as error:
        return (
            f"Training failed: {type(error).__name__}: {error}",
            "",
            "",
        )


def format_matches(
    rows: List[Match],
) -> str:
    if not rows:
        return "No matches."

    lines = []

    for item in rows:
        label = []

        if item.selected:
            label.append("subset")

        if item.endpoint:
            label.append("endpoint")

        marker = (
            f" [{', '.join(label)}]"
            if label
            else ""
        )

        source = item.reference.source.replace(
            "\n",
            " ",
        )

        lines.append(
            f"{item.rank}. "
            f"{source} "
            f"score={item.score:.4f} "
            f"vector={item.vector_score:.4f}"
            f"{marker}"
        )

    return "\n".join(lines)


def generate_text(
    prompt: str,
    temperature: float,
    top_k: int,
    max_new: int,
    ca_steps: int,
    ca_bias_scale: float,
):
    if S.model is None or S.corpus is None:
        return (
            "Train a model first.",
            "",
            "",
        )

    prompt = prompt or ""

    S.model.temperature = max(
        0.05,
        float(temperature),
    )

    S.model.top_k = max(
        1,
        int(top_k),
    )

    S.model.max_new = max(
        1,
        int(max_new),
    )

    S.model.ca_steps = max(
        0,
        int(ca_steps),
    )

    S.model.ca_bias_scale = float(
        ca_bias_scale
    )

    target = S.corpus.bindings.expand(
        {
            key: float(value)
            for key, value in bow(
                tokenize(prompt)
            ).items()
        }
    )

    def bias_function(
        history,
        cellular_context,
    ):
        if not target:
            return None

        seeds = sorted(
            target,
            key=target.get,
            reverse=True,
        )[:16]

        return S.model.model_bias(seeds)

    generated = S.model.generate(
        prompt=prompt,
        bias_function=bias_function,
        temperature=temperature,
        top_k=top_k,
        max_new=max_new,
        ca_steps=ca_steps,
    )

    matches = S.corpus.search(prompt)

    status = (
        f"Device: {DEVICE}\n"
        f"CUDA available: {CUDA}\n"
        f"Cellular steps: {ca_steps}\n"
        f"Cellular bias scale: {ca_bias_scale}\n"
        f"Vocabulary: "
        f"{len(S.model.inverse_vocabulary)}"
    )

    return (
        status,
        format_matches(matches),
        generated,
    )


def load_existing_model():
    if not S.load_saved():
        return (
            "No saved model loaded.",
            "",
            S.bindings.summary(),
        )

    return (
        "Saved model loaded.",
        S.raw_text[:3000],
        S.bindings.summary(),
    )


# ---------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------

def build_ui():
    import gradio as gr

    with gr.Blocks(
        title="CUDA Cellular N-Gram Generator"
    ) as interface:
        gr.Markdown(
            f"""
# CUDA Cellular N-Gram Generator

A stochastic n-gram generator whose token state evolves through
a one-dimensional cellular update field.

**Device:** `{DEVICE}`  
**CUDA available:** `{CUDA}`

Text and source code are both accepted as corpus data.
"""
        )

        with gr.Tab("Training"):
            corpus_file = gr.File(
                label="Text or source-code corpus",
                file_types=[
                    ".txt",
                    ".md",
                    ".py",
                    ".cpp",
                    ".c",
                    ".h",
                    ".hpp",
                    ".cu",
                    ".js",
                    ".ts",
                    ".html",
                    ".css",
                    ".json",
                    ".csv",
                ],
                type="filepath",
            )

            with gr.Row():
                ca_steps_input = gr.Slider(
                    minimum=0,
                    maximum=32,
                    value=DEFAULT_CA_STEPS,
                    step=1,
                    label="Cellular evolution steps",
                )

                ca_scale_input = gr.Slider(
                    minimum=0.0,
                    maximum=3.0,
                    value=DEFAULT_CA_BIAS_SCALE,
                    step=0.05,
                    label="Cellular logit scale",
                )

            with gr.Row():
                train_button = gr.Button(
                    "Train corpus",
                    variant="primary",
                )

                load_button = gr.Button(
                    "Load saved model",
                )

            training_status = gr.Textbox(
                label="Training status",
                lines=3,
            )

            corpus_preview = gr.Textbox(
                label="Corpus preview",
                lines=12,
            )

            binding_summary = gr.Textbox(
                label="Model summary",
                lines=8,
            )

            train_button.click(
                fn=train_from_file,
                inputs=[
                    corpus_file,
                    ca_steps_input,
                    ca_scale_input,
                ],
                outputs=[
                    training_status,
                    corpus_preview,
                    binding_summary,
                ],
            )

            load_button.click(
                fn=load_existing_model,
                inputs=[],
                outputs=[
                    training_status,
                    corpus_preview,
                    binding_summary,
                ],
            )

        with gr.Tab("Generation"):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder=(
                    "For code, try: "
                    "def train ("
                ),
                lines=5,
            )

            with gr.Row():
                temperature_input = gr.Slider(
                    minimum=0.05,
                    maximum=2.5,
                    value=DEFAULT_TEMP,
                    step=0.05,
                    label="Temperature",
                )

                top_k_input = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Top-k",
                )

            with gr.Row():
                max_new_input = gr.Slider(
                    minimum=1,
                    maximum=2000,
                    value=DEFAULT_MAX_NEW,
                    step=1,
                    label="Maximum generated tokens",
                )

                generation_steps_input = gr.Slider(
                    minimum=0,
                    maximum=32,
                    value=DEFAULT_CA_STEPS,
                    step=1,
                    label="Cellular steps",
                )

            generation_scale_input = gr.Slider(
                minimum=0.0,
                maximum=3.0,
                value=DEFAULT_CA_BIAS_SCALE,
                step=0.05,
                label="Cellular logit scale",
            )

            generate_button = gr.Button(
                "Generate",
                variant="primary",
            )

            generation_status = gr.Textbox(
                label="Generation status",
                lines=7,
            )

            corpus_matches = gr.Textbox(
                label="Corpus subset matches and endpoints",
                lines=14,
            )

            generated_text = gr.Textbox(
                label="Generated text",
                lines=20,
            )

            generate_button.click(
                fn=generate_text,
                inputs=[
                    prompt_input,
                    temperature_input,
                    top_k_input,
                    max_new_input,
                    generation_steps_input,
                    generation_scale_input,
                ],
                outputs=[
                    generation_status,
                    corpus_matches,
                    generated_text,
                ],
            )

    return interface


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "CUDA-first stochastic n-gram generator "
            "with cellular token dynamics."
        )
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a Gradio public share link.",
    )

    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Server bind address.",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port.",
    )

    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Do not attempt to load saved model files.",
    )

    args = parser.parse_args()

    if not args.no_load:
        S.load_saved()

    interface = build_ui()

    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
