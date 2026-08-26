#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal Degrees-of-Abstraction Engine
=======================================

A domain-independent symbolic sequence model.

The engine accepts arbitrary sequences of symbols and constructs multiple
abstraction levels:

    Level 0: raw symbols
    Level 1: unigram statistics
    Level 2: pairwise transitions
    Level 3: n-gram context transitions
    Level 4: sparse symbolic feature vectors
    Level 5: similarity and influence relations
    Level 6: constrained candidate selection
    Level 7: adaptive sequential inference

The implementation does not assume:
    - natural language
    - words
    - stems
    - characters
    - sentences
    - a particular domain ontology

Example:

    python universal_abstraction.py \
        --input sequences.json \
        --prompt "A B" \
        --order 3 \
        --new-items 20 \
        --threshold 0.60

Input JSON formats:

1. A list of sequences:

    [
      ["A", "B", "C"],
      ["A", "C", "D"]
    ]

2. An object containing sequences:

    {
      "sequences": [
        ["A", "B", "C"],
        ["D", "E", "F"]
      ]
    }

3. A plain text file:

    Each non-empty line is treated as one sequence and split on whitespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


SymbolId = int
Context = Tuple[SymbolId, ...]
SparseVector = Dict[int, float]


@dataclass
class SymbolStatistics:
    count: int = 0
    density: float = 0.0
    out_degree: int = 0
    relation: float = 0.0
    entropy: float = 0.0
    coherence: float = 0.0
    mean_position: float = 0.0
    depth: float = 0.0


@dataclass
class SymbolEntry:
    symbol: str
    statistics: SymbolStatistics = field(default_factory=SymbolStatistics)
    weight: float = 1.0


@dataclass
class GenerationStep:
    step: int
    context: List[str]
    selected: Optional[str] = None
    source: str = ""
    requested_threshold: float = 0.0
    effective_threshold: float = 0.0
    similarity: float = 0.0
    influence: float = 0.0
    probability: float = 0.0
    combined_score: float = 0.0
    passed_filter: bool = False
    used_fallback: bool = False
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "context": self.context,
            "selected": self.selected,
            "source": self.source,
            "requested_threshold": self.requested_threshold,
            "effective_threshold": self.effective_threshold,
            "similarity": self.similarity,
            "influence": self.influence,
            "probability": self.probability,
            "combined_score": self.combined_score,
            "passed_filter": self.passed_filter,
            "used_fallback": self.used_fallback,
            "stop_reason": self.stop_reason,
        }


class UniversalAbstractionEngine:
    """
    Domain-independent symbolic sequence model.

    The model is based on a finite symbol space and supports arbitrary
    context orders. The default order is three, corresponding to a
    context containing up to two preceding symbols.
    """

    VERSION = 1
    START_SYMBOL = "<START>"
    END_SYMBOL = "<END>"

    def __init__(
        self,
        order: int = 3,
        max_symbols: int = 100_000,
        feature_buckets: int = 8192,
    ) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if max_symbols < 2:
            raise ValueError("max_symbols must be at least 2")
        if feature_buckets < 64:
            raise ValueError("feature_buckets must be at least 64")

        self.order = order
        self.max_symbols = max_symbols
        self.feature_buckets = feature_buckets

        self.symbol_to_id: Dict[str, SymbolId] = {}
        self.id_to_symbol: Dict[SymbolId, str] = {}
        self.entries: Dict[SymbolId, SymbolEntry] = {}

        self.next_id = 0
        self.total_items = 0
        self.sequences: List[List[SymbolId]] = []

        self.unigram_counts: DefaultDict[SymbolId, int] = defaultdict(int)

        self.transition_counts: DefaultDict[
            SymbolId,
            DefaultDict[SymbolId, int],
        ] = defaultdict(lambda: defaultdict(int))

        self.context_counts: DefaultDict[
            Context,
            DefaultDict[SymbolId, int],
        ] = defaultdict(lambda: defaultdict(int))

        self.transition_probabilities: Dict[
            SymbolId,
            Dict[SymbolId, float],
        ] = {}

        self.context_probabilities: Dict[
            Context,
            Dict[SymbolId, float],
        ] = {}

    # ------------------------------------------------------------------
    # Symbol management
    # ------------------------------------------------------------------

    def register(self, symbol: Any) -> SymbolId:
        """
        Register a symbol after converting it to a stable string form.

        JSON-compatible primitive values preserve their normal textual
        representation. Other values use repr().
        """
        normalized = self.normalize_symbol(symbol)

        if normalized in self.symbol_to_id:
            return self.symbol_to_id[normalized]

        if self.next_id >= self.max_symbols:
            raise ValueError(
                f"symbol capacity exceeded: max_symbols={self.max_symbols}"
            )

        symbol_id = self.next_id
        self.next_id += 1

        self.symbol_to_id[normalized] = symbol_id
        self.id_to_symbol[symbol_id] = normalized
        self.entries[symbol_id] = SymbolEntry(symbol=normalized)

        return symbol_id

    @staticmethod
    def normalize_symbol(symbol: Any) -> str:
        if isinstance(symbol, str):
            value = symbol
        elif symbol is None:
            value = "null"
        elif isinstance(symbol, bool):
            value = "true" if symbol else "false"
        else:
            value = str(symbol)

        value = value.strip()

        if not value:
            raise ValueError("empty symbols are not allowed")

        return value

    def symbol_id(self, symbol: Any) -> Optional[SymbolId]:
        return self.symbol_to_id.get(self.normalize_symbol(symbol))

    def symbol(self, symbol_id: SymbolId) -> str:
        return self.id_to_symbol[symbol_id]

    def content_ids(self) -> List[SymbolId]:
        excluded = {self.START_SYMBOL, self.END_SYMBOL}
        return [
            symbol_id
            for symbol_id, value in self.id_to_symbol.items()
            if value not in excluded
        ]

    # ------------------------------------------------------------------
    # Sequence preparation
    # ------------------------------------------------------------------

    def prepare_sequence(self, sequence: Sequence[Any]) -> List[SymbolId]:
        values = [
            self.normalize_symbol(value)
            for value in sequence
            if self.normalize_symbol(value)
        ]

        if not values:
            return []

        boundary = [self.START_SYMBOL] * max(0, self.order - 1)
        padded = boundary + values + [self.END_SYMBOL]

        return [self.register(value) for value in padded]

    def add_sequence(self, sequence: Sequence[Any]) -> List[SymbolId]:
        encoded = self.prepare_sequence(sequence)

        if not encoded:
            return []

        self.sequences.append(encoded)

        for position, symbol_id in enumerate(encoded):
            self.unigram_counts[symbol_id] += 1
            self.total_items += 1

            stats = self.entries[symbol_id].statistics
            stats.count += 1

            for width in range(1, self.order):
                if position - width < 0:
                    continue

                context = tuple(encoded[position - width:position])
                target = encoded[position]
                self.context_counts[context][target] += 1

            if position > 0:
                previous = encoded[position - 1]
                self.transition_counts[previous][symbol_id] += 1

        return encoded

    def ingest(self, sequences: Iterable[Sequence[Any]]) -> None:
        for sequence in sequences:
            self.add_sequence(sequence)

        self.finalize()

    # ------------------------------------------------------------------
    # Finalization and statistics
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_counts(
        counts: Mapping[SymbolId, int],
    ) -> Dict[SymbolId, float]:
        total = sum(counts.values())

        if total <= 0:
            return {}

        return {
            symbol_id: count / total
            for symbol_id, count in counts.items()
            if count > 0
        }

    def finalize(self) -> None:
        self.transition_probabilities = {
            source: self.normalize_counts(targets)
            for source, targets in self.transition_counts.items()
        }

        self.context_probabilities = {
            context: self.normalize_counts(targets)
            for context, targets in self.context_counts.items()
        }

        self.update_statistics()

    def update_statistics(self) -> None:
        if not self.total_items:
            return

        vocabulary_size = max(1, len(self.entries))
        max_degree = max(1, vocabulary_size - 1)

        position_totals: DefaultDict[SymbolId, float] = defaultdict(float)
        position_counts: DefaultDict[SymbolId, int] = defaultdict(int)
        maximum_length = max(
            (len(sequence) for sequence in self.sequences),
            default=1,
        )

        for sequence in self.sequences:
            for position, symbol_id in enumerate(sequence):
                position_totals[symbol_id] += position
                position_counts[symbol_id] += 1

        for symbol_id, entry in self.entries.items():
            stats = entry.statistics
            stats.density = stats.count / self.total_items

            outgoing = self.transition_probabilities.get(symbol_id, {})
            stats.out_degree = len(outgoing)
            stats.relation = min(1.0, stats.out_degree / max_degree)

            if len(outgoing) <= 1:
                stats.entropy = 0.0
                stats.coherence = 1.0
            else:
                entropy = -sum(
                    probability * math.log(probability + 1e-12)
                    for probability in outgoing.values()
                )

                maximum_entropy = math.log(len(outgoing))
                stats.entropy = entropy
                stats.coherence = max(
                    0.0,
                    1.0 - entropy / max(maximum_entropy, 1e-12),
                )

            if position_counts[symbol_id]:
                stats.mean_position = (
                    position_totals[symbol_id] / position_counts[symbol_id]
                )
                stats.depth = stats.mean_position / maximum_length

            entry.weight = 0.5 * stats.density + 0.5 * stats.coherence

    # ------------------------------------------------------------------
    # Sparse vector representation
    # ------------------------------------------------------------------

    def _hash_feature(self, feature: str) -> int:
        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=8,
        ).digest()

        return int.from_bytes(digest, "big") % self.feature_buckets

    def _hashed_identity_vector(self, symbol: str) -> SparseVector:
        vector: DefaultDict[int, float] = defaultdict(float)
        text = f"<{symbol}>"

        for width in range(1, 4):
            for offset in range(0, max(0, len(text) - width + 1)):
                feature = text[offset:offset + width]
                bucket = 10_000 + self._hash_feature(feature)
                vector[bucket] += 1.0

        return dict(vector)

    @staticmethod
    def normalize_vector(vector: SparseVector) -> SparseVector:
        norm = math.sqrt(sum(value * value for value in vector.values()))

        if norm <= 0.0:
            return dict(vector)

        return {
            key: value / norm
            for key, value in vector.items()
        }

    def vector(self, symbol_id: SymbolId) -> SparseVector:
        if symbol_id not in self.entries:
            return {}

        stats = self.entries[symbol_id].statistics

        vector: SparseVector = {
            0: stats.density,
            1: stats.relation,
            2: stats.coherence,
            3: stats.entropy,
            4: stats.depth,
            5: self.entries[symbol_id].weight,
        }

        for target, probability in self.transition_probabilities.get(
            symbol_id,
            {},
        ).items():
            vector[100 + target] = probability

        lexical = self._hashed_identity_vector(
            self.id_to_symbol[symbol_id]
        )

        for feature, value in lexical.items():
            vector[feature] = vector.get(feature, 0.0) + value

        return self.normalize_vector(vector)

    @staticmethod
    def cosine(
        left: SparseVector,
        right: SparseVector,
    ) -> float:
        if not left or not right:
            return 0.0

        if len(left) > len(right):
            left, right = right, left

        dot = sum(
            value * right.get(feature, 0.0)
            for feature, value in left.items()
        )

        left_norm = math.sqrt(
            sum(value * value for value in left.values())
        )
        right_norm = math.sqrt(
            sum(value * value for value in right.values())
        )

        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0

        return dot / (left_norm * right_norm)

    # ------------------------------------------------------------------
    # Influence model
    # ------------------------------------------------------------------

    def influence(
        self,
        source_id: SymbolId,
        target_id: SymbolId,
    ) -> float:
        source_entry = self.entries.get(source_id)
        target_entry = self.entries.get(target_id)

        if source_entry is None or target_entry is None:
            return 0.0

        transition = self.transition_probabilities.get(
            source_id,
            {},
        ).get(target_id, 0.0)

        source_stats = source_entry.statistics
        target_stats = target_entry.statistics

        structural = 1.0 - abs(
            source_stats.coherence *target_stats.coherence
        )

        density_alignment = 1.0 - abs(
            source_stats.density ** target_stats.density
        )

        compatibility = 0.1 + structural * 0.5 * density_alignment

        return max(0.0, source_entry.weight) * max(
            0.0,
            target_entry.weight,
        ) * (
            0.5 * transition + 0.5 * compatibility
        )

    def relation(
        self,
        source_id: SymbolId,
        target_id: SymbolId,
    ) -> Dict[str, float]:
        similarity = self.cosine(
            self.vector(source_id),
            self.vector(target_id),
        )

        return {
            "similarity": similarity,
            "influence": self.influence(source_id, target_id),
            "transition_probability": self.transition_probabilities.get(
                source_id,
                {},
            ).get(target_id, 0.0),
        }

    def influence_matrix(
        self,
        threshold: float = 0.60,
    ) -> Dict[str, Any]:
        threshold = max(-1.0, min(1.0, float(threshold)))
        ids = self.content_ids()
        vectors = {symbol_id: self.vector(symbol_id) for symbol_id in ids}

        rows: List[Dict[str, Any]] = []

        for source_id in ids:
            for target_id in ids:
                if source_id == target_id:
                    continue

                similarity = self.cosine(
                    vectors[source_id],
                    vectors[target_id],
                )

                influence = self.influence(source_id, target_id)
                probability = self.transition_probabilities.get(
                    source_id,
                    {},
                ).get(target_id, 0.0)

                rows.append({
                    "source": self.symbol(source_id),
                    "target": self.symbol(target_id),
                    "similarity": similarity,
                    "influence": influence,
                    "transition_probability": probability,
                    "retained": similarity >= threshold,
                })

        retained = [
            row for row in rows
            if row["retained"]
        ]

        retained.sort(
            key=lambda row: (
                row["similarity"],
                row["influence"],
            ),
            reverse=True,
        )

        return {
            "threshold": threshold,
            "domain_size": len(ids),
            "relation_count": len(rows),
            "retained_count": len(retained),
            "rows": rows,
            "retained": retained,
        }

    # ------------------------------------------------------------------
    # Context and backoff
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: Sequence[Any],
    ) -> Tuple[List[str], List[SymbolId], List[str]]:
        prompt_values = [
            self.normalize_symbol(value)
            for value in prompt
        ]

        known: List[SymbolId] = []
        unknown: List[str] = []

        for value in prompt_values:
            symbol_id = self.symbol_to_id.get(value)

            if symbol_id is None:
                unknown.append(value)
            else:
                known.append(symbol_id)

        return prompt_values, known, unknown

    def resolve_context(
        self,
        known_ids: Sequence[SymbolId],
    ) -> Context:
        if not known_ids:
            raise ValueError("prompt contains no known symbols")

        maximum_width = min(self.order - 1, len(known_ids))

        for width in range(maximum_width, 0, -1):
            context = tuple(known_ids[-width:])

            if context in self.context_probabilities:
                return context

        return tuple(known_ids[-maximum_width:])

    def unigram_distribution(self) -> Dict[SymbolId, float]:
        excluded = {
            self.symbol_to_id.get(self.START_SYMBOL),
            self.symbol_to_id.get(self.END_SYMBOL),
        }

        counts = {
            symbol_id: count
            for symbol_id, count in self.unigram_counts.items()
            if symbol_id not in excluded
        }

        return self.normalize_counts(counts)

    def backoff_distributions(
        self,
        context: Context,
    ) -> List[Tuple[str, Dict[SymbolId, float]]]:
        distributions: List[Tuple[str, Dict[SymbolId, float]]] = []

        for width in range(len(context), 0, -1):
            reduced = context[-width:]
            distribution = self.context_probabilities.get(
                reduced,
                {},
            )

            if distribution:
                distributions.append(
                    (f"context_{width}", distribution)
                )

        if context:
            current = context[-1]
            distribution = self.transition_probabilities.get(
                current,
                {},
            )

            if distribution:
                distributions.append(
                    ("pairwise", distribution)
                )

        unigram = self.unigram_distribution()

        if unigram:
            distributions.append(("unigram", unigram))

        return distributions

    def usable_distribution(
        self,
        distribution: Mapping[SymbolId, float],
    ) -> Dict[SymbolId, float]:
        excluded = {
            self.symbol_to_id.get(self.START_SYMBOL),
            self.symbol_to_id.get(self.END_SYMBOL),
        }

        return {
            symbol_id: probability
            for symbol_id, probability in distribution.items()
            if symbol_id not in excluded
            and probability > 0.0
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Sequence[Any],
        max_new_items: int = 30,
        threshold: float = 0.60,
        threshold_floor: float = 0.05,
        threshold_step: float = 0.05,
        temperature: float = 0.8,
        candidate_count: int = 32,
        stochastic: bool = True,
        preserve_prompt: bool = True,
        adaptive_threshold: bool = True,
        unfiltered_fallback: bool = True,
    ) -> Tuple[List[str], float, List[GenerationStep]]:
        if max_new_items < 0:
            raise ValueError("max_new_items cannot be negative")

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        prompt_values, known_ids, _ = self.encode_prompt(prompt)

        if not known_ids:
            raise ValueError(
                "no prompt symbols were found in the model vocabulary"
            )

        if max_new_items == 0:
            return (
                prompt_values if preserve_prompt else [],
                0.0,
                [],
            )

        requested_threshold = max(
            -1.0,
            min(1.0, float(threshold)),
        )

        threshold_floor = max(
            -1.0,
            min(requested_threshold, float(threshold_floor)),
        )

        threshold_step = max(
            1e-9,
            float(threshold_step),
        )

        context = self.resolve_context(known_ids)
        output = list(prompt_values) if preserve_prompt else []

        minimum_similarity = float("inf")
        diagnostics: List[GenerationStep] = []

        for step_number in range(1, max_new_items + 1):
            distributions = self.backoff_distributions(context)

            selected_source = ""
            distribution: Dict[SymbolId, float] = {}

            for source_name, candidate_distribution in distributions:
                usable = self.usable_distribution(
                    candidate_distribution
                )

                if usable:
                    selected_source = source_name
                    distribution = usable
                    break

            if not distribution:
                diagnostics.append(
                    GenerationStep(
                        step=step_number,
                        context=[
                            self.symbol(symbol_id)
                            for symbol_id in context
                        ],
                        stop_reason="no_candidates",
                    )
                )
                break

            ranked_distribution = sorted(
                distribution.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )[:max(1, candidate_count)]

            source_id = context[-1]
            source_vector = self.vector(source_id)

            candidates: List[
                Tuple[float, float, float, float, SymbolId]
            ] = []

            for target_id, probability in ranked_distribution:
                similarity = self.cosine(
                    source_vector,
                    self.vector(target_id),
                )

                influence = self.influence(
                    source_id,
                    target_id,
                )

                combined = (
                    0.45 * similarity
                    + 0.35 * influence
                    + 0.20 * probability
                )

                candidates.append(
                    (
                        combined,
                        similarity,
                        influence,
                        probability,
                        target_id,
                    )
                )

            candidates.sort(reverse=True)

            effective_threshold = requested_threshold

            passing = [
                candidate
                for candidate in candidates
                if candidate[1] >= effective_threshold
            ]

            if not passing and adaptive_threshold:
                while effective_threshold > threshold_floor:
                    effective_threshold = max(
                        threshold_floor,
                        effective_threshold - threshold_step,
                    )

                    passing = [
                        candidate
                        for candidate in candidates
                        if candidate[1] >= effective_threshold
                    ]

                    if passing:
                        break

            used_fallback = False

            if passing:
                pool = passing
            elif unfiltered_fallback:
                pool = candidates
                used_fallback = True
            else:
                diagnostics.append(
                    GenerationStep(
                        step=step_number,
                        context=[
                            self.symbol(symbol_id)
                            for symbol_id in context
                        ],
                        source=selected_source,
                        requested_threshold=requested_threshold,
                        effective_threshold=effective_threshold,
                        stop_reason="threshold_rejected_all_candidates",
                    )
                )
                break

            if stochastic and len(pool) > 1:
                weights = [
                    max(1e-12, candidate[0])
                    ** (1.0 / temperature)
                    for candidate in pool
                ]

                chosen = random.choices(
                    pool,
                    weights=weights,
                    k=1,
                )[0]
            else:
                chosen = pool[0]

            combined, similarity, influence, probability, target_id = (
                chosen
            )

            selected_symbol = self.symbol(target_id)
            output.append(selected_symbol)

            minimum_similarity = min(
                minimum_similarity,
                similarity,
            )

            diagnostics.append(
                GenerationStep(
                    step=step_number,
                    context=[
                        self.symbol(symbol_id)
                        for symbol_id in context
                    ],
                    selected=selected_symbol,
                    source=selected_source,
                    requested_threshold=requested_threshold,
                    effective_threshold=effective_threshold,
                    similarity=similarity,
                    influence=influence,
                    probability=probability,
                    combined_score=combined,
                    passed_filter=not used_fallback,
                    used_fallback=used_fallback,
                )
            )

            context = (*context, target_id)[-(self.order - 1):]

        if minimum_similarity == float("inf"):
            minimum_similarity = 0.0

        return output, minimum_similarity, diagnostics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "config": {
                "order": self.order,
                "max_symbols": self.max_symbols,
                "feature_buckets": self.feature_buckets,
            },
            "vocabulary": [
                self.id_to_symbol[index]
                for index in range(self.next_id)
            ],
            "sequences": self.sequences,
            "unigram_counts": {
                str(symbol_id): count
                for symbol_id, count in self.unigram_counts.items()
            },
            "transition_counts": {
                str(source): {
                    str(target): count
                    for target, count in targets.items()
                }
                for source, targets in self.transition_counts.items()
            },
            "context_counts": {
                ",".join(map(str, context)): {
                    str(target): count
                    for target, count in targets.items()
                }
                for context, targets in self.context_counts.items()
            },
        }

    def save(self, filename: str) -> None:
        path = Path(filename)
        path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "UniversalAbstractionEngine":
        config = state["config"]

        engine = cls(
            order=int(config["order"]),
            max_symbols=int(config["max_symbols"]),
            feature_buckets=int(config["feature_buckets"]),
        )

        for symbol in state["vocabulary"]:
            engine.register(symbol)

        engine.sequences = [
            [int(symbol_id) for symbol_id in sequence]
            for sequence in state["sequences"]
        ]

        for symbol_id, count in state["unigram_counts"].items():
            engine.unigram_counts[int(symbol_id)] = int(count)

        for source, targets in state["transition_counts"].items():
            source_id = int(source)

            for target, count in targets.items():
                target_id = int(target)
                engine.transition_counts[source_id][target_id] = int(
                    count
                )

        for context_text, targets in state["context_counts"].items():
            context = tuple(
                int(value)
                for value in context_text.split(",")
                if value
            )

            for target, count in targets.items():
                engine.context_counts[context][int(target)] = int(
                    count
                )

        engine.total_items = sum(engine.unigram_counts.values())
        engine.finalize()

        return engine

    @classmethod
    def load(cls, filename: str) -> "UniversalAbstractionEngine":
        path = Path(filename)
        state = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(state)


# ----------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------


def load_sequences(filename: str) -> List[List[str]]:
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(filename)

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, dict):
            data = data.get("sequences", [])

        if not isinstance(data, list):
            raise ValueError(
                "JSON input must be a sequence list or an object containing "
                "'sequences'"
            )

        sequences: List[List[str]] = []

        for item in data:
            if isinstance(item, list):
                sequences.append([str(value) for value in item])
            else:
                sequences.append([str(item)])

        return sequences

    text = path.read_text(encoding="utf-8")
    sequences = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if " " in line or "\t" in line:
            sequences.append(re.split(r"\s+", line))
        else:
            sequences.append(list(line))

    return sequences


def parse_prompt(value: str) -> List[str]:
    value = value.strip()

    if not value:
        return []

    if value.startswith("["):
        parsed = json.loads(value)

        if not isinstance(parsed, list):
            raise ValueError("prompt JSON must be a list")

        return [str(item) for item in parsed]

    return value.split()


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------


def print_diagnostics(
    diagnostics: Sequence[GenerationStep],
) -> None:
    print("\nDiagnostics")
    print("-" * 110)

    for diagnostic in diagnostics:
        if diagnostic.stop_reason:
            print(
                f"{diagnostic.step:04d} | STOP | "
                f"{diagnostic.stop_reason}"
            )
            continue

        context = " ".join(diagnostic.context)

        print(
            f"{diagnostic.step:04d} | "
            f"{context:30s} -> "
            f"{str(diagnostic.selected):20s} | "
            f"source={diagnostic.source:12s} | "
            f"sim={diagnostic.similarity:.5f} | "
            f"threshold={diagnostic.effective_threshold:.5f} | "
            f"influence={diagnostic.influence:.5f} | "
            f"probability={diagnostic.probability:.5f} | "
            f"fallback={diagnostic.used_fallback}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Universal domain-independent symbolic sequence abstraction "
            "engine"
        )
    )

    parser.add_argument(
        "--input",
        help="Input JSON or line-oriented sequence file",
    )

    parser.add_argument(
        "--load",
        help="Load an existing model",
    )

    parser.add_argument(
        "--save",
        help="Save the resulting model",
    )

    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Maximum context order",
    )

    parser.add_argument(
        "--new-items",
        type=int,
        default=300,
        help="Number of generated symbols",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Initial similarity lower bound",
    )

    parser.add_argument(
        "--threshold-floor",
        type=float,
        default=0.05,
        help="Minimum adaptive similarity lower bound",
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Adaptive threshold decrement",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Stochastic selection temperature",
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=32,
        help="Maximum ranked candidates",
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Always select the highest-ranked candidate",
    )

    parser.add_argument(
        "--no-adaptive-threshold",
        action="store_true",
        help="Disable threshold relaxation",
    )

    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable unconstrained candidate fallback",
    )

    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Print retained influence relations",
    )

    parser.add_argument(
        "--matrix-limit",
        type=int,
        default=20,
        help="Maximum matrix relations to print",
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print generation diagnostics",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.load:
        engine = UniversalAbstractionEngine.load(args.load)
        source = f"loaded {args.load}"
    else:
        if not args.input:
            parser.error("--input is required unless --load is used")

        engine = UniversalAbstractionEngine(order=args.order)
        sequences = load_sequences(args.input)
        engine.ingest(sequences)
        source = f"trained from {args.input}"

    if args.save:
        engine.save(args.save)
        print(f"Saved model: {args.save}")

    print("=" * 110)
    print("UNIVERSAL DEGREES-OF-ABSTRACTION ENGINE")
    print("=" * 110)
    print(f"Source: {source}")
    print(f"Symbols: {len(engine.content_ids())}")
    print(f"Sequences: {len(engine.sequences)}")
    print(f"Context order: {engine.order}")


    while True:
        try:
            generated, minimum_similarity, diagnostics = engine.generate(
                prompt=input("prompt: ").split(),
                max_new_items=args.new_items,
                threshold=args.threshold,
                threshold_floor=args.threshold_floor,
                threshold_step=args.threshold_step,
                temperature=args.temperature,
                candidate_count=args.candidate_count,
                stochastic=not args.greedy,
                preserve_prompt=True,
                adaptive_threshold=not args.no_adaptive_threshold,
                unfiltered_fallback=not args.no_fallback,
            )
        except ValueError as error:
            parser.error(str(error))
            return

        print("\nGenerated sequence")
        print("-" * 110)
        print(" ".join(generated))
        print(f"\nMinimum selected similarity: {minimum_similarity:.6f}")

        if args.diagnostics:
            print_diagnostics(diagnostics)

        if args.matrix:
            matrix = engine.influence_matrix(args.threshold)

            print("\nRetained influence relations")
            print("-" * 110)

            for row in matrix["retained"][:args.matrix_limit]:
                print(
                    f"{row['source']} -> {row['target']} | "
                    f"similarity={row['similarity']:.5f} | "
                    f"influence={row['influence']:.5f} | "
                    f"probability={row['transition_probability']:.5f}"
                )


if __name__ == "__main__":
    main()
