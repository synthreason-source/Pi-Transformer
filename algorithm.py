from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import torch
import numpy as np
import gradio as gr


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

MODEL_PATH = "model.json"
DEFAULT_CORPUS_FILE = "corpus.txt"

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 20

MIN_COUNT = 1
INFLUENCE_TAU = 0.7

CURVE_K = 18.0
CURVE_MIDPOINT = 0.5

CANDIDATE_LIMIT = 15
LEXICAL_WEIGHT = 0.15
VECTOR_WEIGHT = 0.15

ISOMORPHISM_TAU = 0.97
ISOMORPHISM_SHARPNESS = 0.8

CONTEXT_REDUCTION_FRACTION = 0.1
CONTEXT_REDUCTION_MAX_DIMS = 200

ENABLE_DEEP_BILINEAR_ACTIVATION = True
BILINEAR_LAYERS = 32
BILINEAR_DIM = 128
BILINEAR_SCALE = 14.0

TRANSITIVITY_DECAY = 0.01
TRANSITIVITY_WEIGHT = 0.71
TRANSITIVITY_MASK_PENALTY = 1.0
TRANSITIVITY_SUPERPOLY_K = 13.0

INSTRUCTION_COMPOUND_CURVE_K = 2.0
INSTRUCTION_COMPOUND_MIDPOINT = 1110.5
INSTRUCTION_COMPOUND_GROWTH = 0.65
INSTRUCTION_COMPOUND_CAP = 4.0
INSTRUCTION_COMPOUND_WEIGHT = 1.4

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


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(token for token in tokens if token not in IGNORED_TOKENS)


def cosine_similarity(a, b, eps: float = 1e-12) -> float:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a = a / torch.clamp(
            torch.linalg.vector_norm(a, dim=-1, keepdim=True),
            min=eps,
        )
        b = b / torch.clamp(
            torch.linalg.vector_norm(b, dim=-1, keepdim=True),
            min=eps,
        )
        return float((a * b).sum(dim=-1).item())

    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)

    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def lexical_overlap(
    a: Iterable[str],
    b: Iterable[str],
) -> float:
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
    def __init__(
        self,
        lexical_weight: float = LEXICAL_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> None:
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

    def analyze(
        self,
        prompt: str,
        limit: int = 5,
    ) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)
        prompt_vector = {
            token: float(count)
            for token, count in bag_of_words(prompt_tokens).items()
        }

        candidates: List[Candidate] = []

        for reference in self.references:
            symbolic = lexical_overlap(prompt_tokens, reference.tokens)
            vector_similarity = cosine_similarity(
                prompt_vector,
                reference.vector,
            )

            score = (
                self.lexical_weight * symbolic
                + self.vector_weight * vector_similarity
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
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT
    influence_tau: float = INFLUENCE_TAU
    curve_k: float = CURVE_K
    curve_midpoint: float = CURVE_MIDPOINT
    isomorphism_tau: float = ISOMORPHISM_TAU
    isomorphism_sharpness: float = ISOMORPHISM_SHARPNESS
    context_reduction_fraction: float = CONTEXT_REDUCTION_FRACTION
    context_reduction_max_dims: int = CONTEXT_REDUCTION_MAX_DIMS
    enable_deep_bilinear_activation: bool = ENABLE_DEEP_BILINEAR_ACTIVATION
    bilinear_layers: int = BILINEAR_LAYERS
    bilinear_dim: int = BILINEAR_DIM
    bilinear_scale: float = BILINEAR_SCALE

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    trigram: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    lexical_vectors: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    influence_vectors: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    isomorphism_map: Dict[str, str] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)

            if not words:
                continue

            sequence = [
                "<bos>",
                "<bos>",
                *words,
                self.eos_token,
            ]

            self._add_sequence(sequence)

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return

        for token in sequence:
            self.unigram[token] += 1

        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1

        for first, second, third in zip(
            sequence,
            sequence[1:],
            sequence[2:],
        ):
            key = f"{first}\t{second}"
            self.trigram[key][third] += 1

        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(
            token
            for token, count in self.unigram.items()
            if count >= self.min_count
        )

        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts: Dict[str, Counter] = defaultdict(Counter)

        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}

        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1

            self.lexical_vectors[token] = {
                context: count / total
                for context, count in counts.items()
            }

        self.reduce_context_dimensions()

        if self.enable_deep_bilinear_activation:
            self.apply_deep_bilinear_activation()

        self.build_influence_and_isomorphism()

        self.finalized = True

    def reduce_context_dimensions(
        self,
        fraction: Optional[float] = None,
        max_dims: Optional[int] = None,
    ) -> None:
        fraction = (
            self.context_reduction_fraction
            if fraction is None
            else fraction
        )

        max_dims = (
            self.context_reduction_max_dims
            if max_dims is None
            else max_dims
        )

        context_totals: Counter = Counter()

        for vector in self.lexical_vectors.values():
            for context, weight in vector.items():
                context_totals[context] += weight

        if len(context_totals) <= 1:
            return

        ranked_contexts = [
            context
            for context, _ in context_totals.most_common()
        ]

        active_contexts = ranked_contexts[:max_dims]

        target_count = max(
            1,
            math.ceil(len(active_contexts) * fraction),
        )

        if target_count >= len(active_contexts):
            return

        token_order = list(self.lexical_vectors.keys())
        token_index = {
            token: index
            for index, token in enumerate(token_order)
        }

        context_index = {
            context: index
            for index, context in enumerate(active_contexts)
        }

        rows = torch.zeros(
            (
                len(active_contexts),
                len(token_order),
            ),
            dtype=DTYPE,
            device=DEVICE,
        )

        for token, vector in self.lexical_vectors.items():
            token_position = token_index[token]

            for context, weight in vector.items():
                context_position = context_index.get(context)

                if context_position is not None:
                    rows[context_position, token_position] = weight

        names = list(active_contexts)
        members: List[List[str]] = [
            [context]
            for context in active_contexts
        ]

        while len(names) > target_count:
            norms = torch.clamp(
                torch.linalg.vector_norm(
                    rows,
                    dim=1,
                    keepdim=True,
                ),
                min=1e-12,
            )

            normalized = rows / norms
            similarities = normalized @ normalized.T
            similarities.fill_diagonal_(-1.0)

            current_count = similarities.shape[0]
            flat_index = int(torch.argmax(similarities).item())

            first_index, second_index = divmod(
                flat_index,
                current_count,
            )

            if first_index > second_index:
                first_index, second_index = (
                    second_index,
                    first_index,
                )

            merged_name = (
                f"{names[first_index]}-{names[second_index]}"
            )

            merged_vector = (
                rows[first_index] + rows[second_index]
            ).unsqueeze(0)

            merged_members = (
                members[first_index]
                + members[second_index]
            )

            keep = [
                index
                for index in range(current_count)
                if index not in (first_index, second_index)
            ]

            keep_index = torch.tensor(
                keep,
                dtype=torch.long,
                device=DEVICE,
            )

            rows = torch.cat(
                [
                    rows.index_select(0, keep_index),
                    merged_vector,
                ],
                dim=0,
            )

            names = [
                names[index]
                for index in keep
            ] + [merged_name]

            members = [
                members[index]
                for index in keep
            ] + [merged_members]

        remap: Dict[str, str] = {
            leaf: group_name
            for group_name, leaves in zip(names, members)
            for leaf in leaves
        }

        new_vectors: Dict[str, Dict[str, float]] = {}

        for token, vector in self.lexical_vectors.items():
            new_vector: Dict[str, float] = defaultdict(float)

            for context, weight in vector.items():
                new_context = remap.get(context, context)
                new_vector[new_context] += weight

            new_vectors[token] = dict(new_vector)

        self.lexical_vectors = new_vectors

    @staticmethod
    def hash_feature(
        key: str,
        layer: int,
        projection: int,
        dimension: int,
    ) -> Tuple[int, float]:
        value = hash((layer, projection, key))

        index = value % dimension
        sign = 1.0 if (value // dimension) % 2 == 0 else -1.0

        return index, sign

    def hash_projection_matrix(
        self,
        keys: List[str],
        layer: int,
        projection: int,
        dimension: int,
    ) -> torch.Tensor:
        matrix = torch.zeros(
            (
                len(keys),
                dimension,
            ),
            dtype=DTYPE,
            device=DEVICE,
        )

        for key_index, key in enumerate(keys):
            feature_index, sign = self.hash_feature(
                key,
                layer,
                projection,
                dimension,
            )

            matrix[key_index, feature_index] = sign

        return matrix

    def apply_deep_bilinear_activation(self) -> None:
        token_order = list(self.lexical_vectors.keys())

        if not token_order:
            return

        input_keys = sorted(
            {
                key
                for vector in self.lexical_vectors.values()
                for key in vector
            }
        )

        key_index = {
            key: index
            for index, key in enumerate(input_keys)
        }

        x = torch.zeros(
            (
                len(token_order),
                len(input_keys),
            ),
            dtype=DTYPE,
            device=DEVICE,
        )

        for token_index, token in enumerate(token_order):
            for key, weight in self.lexical_vectors[token].items():
                x[token_index, key_index[key]] = weight

        dimension = self.bilinear_dim
        current_keys = input_keys

        for layer in range(self.bilinear_layers):
            # u and v now share the SAME hash projection, so their product
            # is an elementwise square (>= 0 wherever the projection is
            # nonzero) instead of the product of two uncorrelated random
            # projections, which cancelled to zero almost immediately.
            projection = self.hash_projection_matrix(
                current_keys, layer, 0, dimension,
            )

            u = x @ projection
            v = u

            x = torch.tanh(self.bilinear_scale * u * v)

            current_keys = [f"L{layer}:{index}" for index in range(dimension)]

            if not torch.any(x):
                break


        x_cpu = x.detach().cpu()
        nonzero_rows, nonzero_columns = torch.nonzero(
            x_cpu,
            as_tuple=True,
        )

        per_token: Dict[int, Dict[str, float]] = defaultdict(dict)

        for row, column in zip(
            nonzero_rows.tolist(),
            nonzero_columns.tolist(),
        ):
            per_token[row][current_keys[column]] = float(
                x_cpu[row, column]
            )

        self.lexical_vectors = {
            token: per_token.get(token_index, {})
            for token_index, token in enumerate(token_order)
        }

    def vocab_similarity_matrix(
        self,
    ) -> Tuple[List[str], torch.Tensor]:
        vocabulary = self.vocabulary

        keys = sorted(
            {
                key
                for vector in self.lexical_vectors.values()
                for key in vector
            }
        )

        key_index = {
            key: index
            for index, key in enumerate(keys)
        }

        matrix = torch.zeros(
            (
                len(vocabulary),
                len(keys),
            ),
            dtype=DTYPE,
            device=DEVICE,
        )

        for row_index, token in enumerate(vocabulary):
            for key, weight in self.lexical_vectors.get(
                token,
                {},
            ).items():
                column_index = key_index.get(key)

                if column_index is not None:
                    matrix[row_index, column_index] = weight

        norms = torch.clamp(
            torch.linalg.vector_norm(
                matrix,
                dim=1,
                keepdim=True,
            ),
            min=1e-12,
        )

        normalized = matrix / norms
        similarities = normalized @ normalized.T

        return vocabulary, similarities

    def build_influence_and_isomorphism(self) -> None:
        vocabulary, similarities = self.vocab_similarity_matrix()
        similarities_cpu = similarities.detach().cpu()

        self.influence_vectors = {}

        for row_index, source in enumerate(vocabulary):
            row = similarities_cpu[row_index]
            scores: Dict[str, float] = {}

            hits = (
                row >= self.influence_tau
            ).nonzero(as_tuple=True)[0].tolist()

            for column_index in hits:
                if column_index == row_index:
                    continue

                scores[vocabulary[column_index]] = float(
                    row[column_index]
                )

            self.influence_vectors[source] = scores

        self.isomorphism_map = (
            self.isomorphism_map_from_similarity(
                vocabulary,
                similarities_cpu,
            )
        )

    def isomorphism_map_from_similarity(
        self,
        vocabulary: List[str],
        similarities_cpu: torch.Tensor,
    ) -> Dict[str, str]:
        count = len(vocabulary)

        clamped = torch.clamp(
            similarities_cpu,
            min=0.0,
            max=1.0,
        )

        sharpened = clamped ** self.isomorphism_sharpness

        canonical: Dict[str, str] = {}
        assigned = [False] * count

        for first_index in range(count):
            if assigned[first_index]:
                continue

            first_token = vocabulary[first_index]
            canonical[first_token] = first_token
            assigned[first_index] = True

            if not self.lexical_vectors.get(first_token):
                continue

            row = sharpened[first_index]

            for second_index in range(first_index + 1, count):
                if assigned[second_index]:
                    continue

                second_token = vocabulary[second_index]

                if not self.lexical_vectors.get(second_token):
                    continue

                if float(row[second_index]) >= self.isomorphism_tau:
                    canonical[second_token] = first_token
                    assigned[second_index] = True

        return canonical

    def build_isomorphism_classes(self) -> None:
        vocabulary, similarities = self.vocab_similarity_matrix()

        self.isomorphism_map = (
            self.isomorphism_map_from_similarity(
                vocabulary,
                similarities.detach().cpu(),
            )
        )

    def sigmoid_curve(
        self,
        value: float,
        k: float,
        midpoint: float,
    ) -> float:
        return 1.0 / (
            1.0 + math.exp(-k * (value - midpoint))
        )

    def instruction_vector(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        tokens = [
            token
            for token in tokenize(prompt)
            if token not in IGNORED_TOKENS
        ]

        aggregate: Dict[str, float] = defaultdict(float)

        for token in tokens:
            vector = self.lexical_vectors.get(token, {})

            for context, weight in vector.items():
                aggregate[context] += weight

        return dict(aggregate)

    def backoff_distribution(
        self,
        previous: str,
        previous_previous: Optional[str],
    ) -> Dict[str, float]:
        if previous_previous is not None:
            key = f"{previous_previous}\t{previous}"
            counts = self.trigram.get(key)

            if counts:
                return self.normalize(counts)

        counts = self.bigram.get(previous)

        if counts:
            return self.normalize(counts)

        return self.normalize(self.unigram)

    @staticmethod
    def normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())

        if not total:
            return {}

        return {
            token: count / total
            for token, count in counts.items()
        }

    def curve_weight(self, eos_probability: float) -> float:
        eos_probability = min(
            1.0,
            max(0.0, eos_probability),
        )

        z = self.curve_k * (
            eos_probability - self.curve_midpoint
        )

        return 1.0 / (1.0 + math.exp(z))

    def resolve_context(
        self,
        prompt: str,
    ) -> Tuple[str, Optional[str]]:
        tokens = tokenize(prompt)

        if not tokens:
            return "<bos>", None

        previous = tokens[-1]
        previous_previous = (
            tokens[-2]
            if len(tokens) >= 2
            else None
        )

        return previous, previous_previous

    def score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
        mask_penalty: float = TRANSITIVITY_MASK_PENALTY,
        superpoly_k: float = TRANSITIVITY_SUPERPOLY_K,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_factor: float = 1.0,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        previous, previous_previous = self.resolve_context(prompt)

        base = self.backoff_distribution(
            previous,
            previous_previous,
        )

        if not base:
            return {}

        candidates = sorted(
            base,
            key=base.get,
            reverse=True,
        )[:candidate_limit]

        source_vector = self.lexical_vectors.get(
            previous,
            {},
        )

        influences = self.influence_vectors.get(
            previous,
            {},
        )

        curve = self.curve_weight(
            base.get(self.eos_token, 0.0)
        )

        scores: Dict[str, float] = {}

        for token in candidates:
            similarity = cosine_similarity(
                source_vector,
                self.lexical_vectors.get(token, {}),
            )

            influence = influences.get(token, 0.0)

            score = (
                safe_log(base[token])
                + curve * 0.35 * similarity
                + curve * 0.65 * influence
            )

            if (
                transitivity_mask is not None
                and transitivity_weight
            ):
                mask_weight = transitivity_mask.get(token)

                if mask_weight is not None:
                    score += transitivity_weight * (
                        math.exp(superpoly_k * mask_weight)
                        - 1.0
                    )
                else:
                    score -= transitivity_weight * (
                        math.exp(superpoly_k * mask_penalty)
                        - 1.0
                    )

            if instruction_vector and instruction_weight:
                likeness = cosine_similarity(
                    instruction_vector,
                    self.lexical_vectors.get(token, {}),
                )

                curved_likeness = self.sigmoid_curve(
                    likeness,
                    INSTRUCTION_COMPOUND_CURVE_K,
                    INSTRUCTION_COMPOUND_MIDPOINT,
                )

                score += (
                    instruction_weight
                    * compound_factor
                    * curved_likeness
                )

            scores[token] = score

        return scores

    def probabilities(
        self,
        prompt: str,
        temperature: float,
        candidate_limit: int,
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_factor: float = 1.0,
    ) -> Dict[str, float]:
        scores = self.score_next_token(
            prompt=prompt,
            candidate_limit=candidate_limit,
            transitivity_mask=transitivity_mask,
            transitivity_weight=transitivity_weight,
            instruction_vector=instruction_vector,
            instruction_weight=instruction_weight,
            compound_factor=compound_factor,
        )

        if not scores:
            return {}

        temperature = max(temperature, 1e-5)

        scaled = {
            token: score / temperature
            for token, score in scores.items()
        }

        maximum = max(scaled.values())

        exponentials = {
            token: math.exp(score - maximum)
            for token, score in scaled.items()
        }

        total = sum(exponentials.values())

        if not total:
            return {}

        return {
            token: value / total
            for token, value in exponentials.items()
        }

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_factor: float = 1.0,
    ) -> str:
        probabilities = self.probabilities(
            prompt=prompt,
            temperature=temperature,
            candidate_limit=max(top_k, 1),
            transitivity_mask=transitivity_mask,
            transitivity_weight=transitivity_weight,
            instruction_vector=instruction_vector,
            instruction_weight=instruction_weight,
            compound_factor=compound_factor,
        )

        if not probabilities:
            return self.eos_token

        items = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        tokens, weights = zip(*items)

        return random.choices(
            tokens,
            weights=weights,
            k=1,
        )[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_growth: float = INSTRUCTION_COMPOUND_GROWTH,
        compound_cap: float = INSTRUCTION_COMPOUND_CAP,
        reform_window: int = 0,
        reform_passes: int = 1,
    ) -> str:
        generated = tokenize(prompt)
        prompt_length = len(generated)

        instruction_target = instruction_vector

        compound_factor = 1.0

        def resample_at(position: int) -> str:
            left_context = " ".join(
                generated[:position]
            )

            return self.sample_next(
                prompt=left_context,
                temperature=temperature,
                top_k=top_k,
                transitivity_mask=transitivity_mask,
                transitivity_weight=transitivity_weight,
                instruction_vector=instruction_target,
                instruction_weight=instruction_weight,
                compound_factor=compound_factor,
            )

        for _ in range(max_new_tokens):
            token = self.sample_next(
                prompt=" ".join(generated),
                temperature=temperature,
                top_k=top_k,
                transitivity_mask=transitivity_mask,
                transitivity_weight=transitivity_weight,
                instruction_vector=instruction_target,
                instruction_weight=instruction_weight,
                compound_factor=compound_factor,
            )

            generated.append(token)

            if instruction_target and instruction_weight:
                likeness = cosine_similarity(
                    instruction_target,
                    self.lexical_vectors.get(token, {}),
                )

                curved_likeness = self.sigmoid_curve(
                    likeness,
                    INSTRUCTION_COMPOUND_CURVE_K,
                    INSTRUCTION_COMPOUND_MIDPOINT,
                )

                compound_factor = min(
                    compound_cap,
                    compound_factor
                    * (
                        1.0
                        + compound_growth * curved_likeness
                    ),
                )

            if reform_window > 0:
                window_start = max(
                    prompt_length,
                    len(generated) - 1 - reform_window,
                )

                window_end = len(generated) - 1

                for _ in range(max(reform_passes, 1)):
                    for position in range(
                        window_start,
                        window_end,
                    ):
                        generated[position] = resample_at(
                            position
                        )

        visible_tokens = strip_structural_tokens(generated)

        return self.detokenize(visible_tokens)

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        return " ".join(tokens)

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "influence_tau": self.influence_tau,
            "curve_k": self.curve_k,
            "curve_midpoint": self.curve_midpoint,
            "isomorphism_tau": self.isomorphism_tau,
            "isomorphism_sharpness": self.isomorphism_sharpness,
            "context_reduction_fraction": (
                self.context_reduction_fraction
            ),
            "context_reduction_max_dims": (
                self.context_reduction_max_dims
            ),
            "enable_deep_bilinear_activation": (
                self.enable_deep_bilinear_activation
            ),
            "bilinear_layers": self.bilinear_layers,
            "bilinear_dim": self.bilinear_dim,
            "bilinear_scale": self.bilinear_scale,
            "unigram": dict(self.unigram),
            "bigram": {
                key: dict(value)
                for key, value in self.bigram.items()
            },
            "trigram": {
                key: dict(value)
                for key, value in self.trigram.items()
            },
            "lexical_vectors": self.lexical_vectors,
            "influence_vectors": self.influence_vectors,
            "isomorphism_map": self.isomorphism_map,
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data.get("eos_token", "<eos>"),
            unk_token=data.get("unk_token", "<unk>"),
            min_count=data.get("min_count", MIN_COUNT),
            influence_tau=data.get(
                "influence_tau",
                INFLUENCE_TAU,
            ),
            curve_k=data.get("curve_k", CURVE_K),
            curve_midpoint=data.get(
                "curve_midpoint",
                CURVE_MIDPOINT,
            ),
            isomorphism_tau=data.get(
                "isomorphism_tau",
                ISOMORPHISM_TAU,
            ),
            isomorphism_sharpness=data.get(
                "isomorphism_sharpness",
                ISOMORPHISM_SHARPNESS,
            ),
            context_reduction_fraction=data.get(
                "context_reduction_fraction",
                CONTEXT_REDUCTION_FRACTION,
            ),
            context_reduction_max_dims=data.get(
                "context_reduction_max_dims",
                CONTEXT_REDUCTION_MAX_DIMS,
            ),
            enable_deep_bilinear_activation=data.get(
                "enable_deep_bilinear_activation",
                ENABLE_DEEP_BILINEAR_ACTIVATION,
            ),
            bilinear_layers=data.get(
                "bilinear_layers",
                BILINEAR_LAYERS,
            ),
            bilinear_dim=data.get(
                "bilinear_dim",
                BILINEAR_DIM,
            ),
            bilinear_scale=data.get(
                "bilinear_scale",
                BILINEAR_SCALE,
            ),
        )

        model.unigram = Counter(data.get("unigram", {}))

        model.bigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value in data.get(
                    "bigram",
                    {},
                ).items()
            },
        )

        model.trigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value in data.get(
                    "trigram",
                    {},
                ).items()
            },
        )

        model.lexical_vectors = data.get(
            "lexical_vectors",
            {},
        )

        model.influence_vectors = data.get(
            "influence_vectors",
            {},
        )

        model.isomorphism_map = data.get(
            "isomorphism_map",
            {},
        )

        model.vocabulary = data.get(
            "vocabulary",
            [],
        )

        model.finalized = data.get(
            "finalized",
            False,
        )

        if (
            not model.isomorphism_map
            and model.vocabulary
            and model.lexical_vectors
        ):
            model.build_isomorphism_classes()

        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(
            json.loads(
                Path(path).read_text(
                    encoding="utf-8"
                )
            )
        )


def strip_structural_tokens(
    tokens: List[str],
) -> List[str]:
    return [
        token
        for token in tokens
        if token not in IGNORED_TOKENS
    ]


# ------------------- Direct Tensor Image Projection -----------------

def flip_image_horizontal(image: Any) -> np.ndarray | None:
    if image is None:
        return None

    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    image = np.asarray(image)

    if image.ndim == 2:
        return np.fliplr(image)
    if image.ndim == 3:
        return np.fliplr(image)
    return image


def extract_raw_image_tensor(image: Any, n_bins: int = 8) -> torch.Tensor:
    if image is None:
        return torch.zeros(3 + 3 + 2 + (3 * n_bins), dtype=DTYPE, device=DEVICE)

    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    img = image.astype(np.float32)

    means = [float(img[:, :, c].mean()) / 255.0 for c in range(3)]
    stds = [float(img[:, :, c].std()) / 255.0 for c in range(3)]
    
    gray = img.mean(axis=-1)
    brightness = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 255.0

    hist_features = []
    for c in range(3):
        channel = img[:, :, c].ravel()
        hist, _ = np.histogram(channel, bins=n_bins, range=(0.0, 255.0))
        hist = hist.astype(np.float32)
        total = hist.sum()
        probs = hist / total if total else np.zeros_like(hist)
        hist_features.extend(probs.tolist())

    feature_vector = means + stds + [brightness, contrast] + hist_features
    return torch.tensor(feature_vector, dtype=DTYPE, device=DEVICE)


def project_image_to_lexical_vector(
    model: NGramModel,
    image: Any,
) -> Dict[str, float]:
    if not model.finalized:
        model.finalize()

    context_keys = sorted(
        {
            key
            for vector in model.lexical_vectors.values()
            for key in vector
        }
    )

    if not context_keys:
        return {}

    img_tensor = extract_raw_image_tensor(image)
    img_dim = img_tensor.shape[0]
    ctx_dim = len(context_keys)

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(42)

    projection_matrix = torch.randn(
        (img_dim, ctx_dim),
        generator=generator,
        dtype=DTYPE,
        device=DEVICE,
    )
    projection_matrix = torch.nn.functional.normalize(projection_matrix, dim=0)

    projected = img_tensor @ projection_matrix
    projected = torch.tanh(projected)

    result = {}
    projected_cpu = projected.detach().cpu()
    for i, key in enumerate(context_keys):
        val = float(projected_cpu[i].item())
        if abs(val) > 1e-6:
            result[key] = val

    return result


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
            corpus_text = default_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
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
        f"Trigram contexts: {len(model.trigram)}\n"
        f"Isomorphism classes: "
        f"{len(set(model.isomorphism_map.values()))} "
        f"from {len(model.isomorphism_map)} tokens"
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

    flipped = flip_image_horizontal(image)
    image_vector = project_image_to_lexical_vector(TEXT_MODEL, flipped)

    if user_prompt and user_prompt.strip():
        generated = TEXT_MODEL.generate(
            prompt=user_prompt.strip(),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            instruction_vector=image_vector,
            instruction_weight=1.5,
        )
        corpus_matches = format_matches(user_prompt.strip(), limit=5)
    else:
        generated = TEXT_MODEL.generate(
            prompt="<bos>",
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            instruction_vector=image_vector,
            instruction_weight=1.5,
        )
        corpus_matches = "No prompt provided for corpus search."

    vector_summary = f"Projected {len(image_vector)} active dimensions into model lexical space."

    return flipped, vector_summary, corpus_matches, generated


# --------------------------- UI -------------------------------------

with gr.Blocks(title="Camera Tensor-Faceted Tau Model") as demo:
    gr.Markdown(
        """
# Camera Tensor-Faceted Tau Model

1. Upload any file as corpus.
2. Click **Train model**.
3. Use the webcam/upload to project image tensors directly into model inference.
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
            train_status = gr.Textbox(
                label="Training status",
                lines=2,
            )
            corpus_sample = gr.Textbox(
                label="Corpus sample (first 5 lines)",
                lines=5,
            )
            model_summary = gr.Textbox(
                label="Model summary",
                lines=6,
            )

    train_button.click(
        fn=train_model_from_file,
        inputs=[corpus_file_input],
        outputs=[train_status, corpus_sample, model_summary],
    )

    gr.Markdown("## 2. Camera + Tensor Inference")

    with gr.Row():
        with gr.Column(scale=1):
            camera = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Camera image",
                webcam_options=gr.WebcamOptions(mirror=False),
            )
            user_prompt = gr.Textbox(
                label="Optional text prompt",
                lines=3,
            )
            recognize_button = gr.Button(
                "Process image tensor",
                variant="primary",
            )

        with gr.Column(scale=2):
            processed_image = gr.Image(
                label="Processed image",
                type="numpy",
            )
            feature_tokens_output = gr.Textbox(
                label="Image Tensor Mapping Status",
                lines=2,
            )
            corpus_output = gr.Textbox(
                label="Corpus matches",
                lines=6,
            )
            generated_output = gr.Textbox(
                label="Tau model response",
                lines=8,
            )

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
