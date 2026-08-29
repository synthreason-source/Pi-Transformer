#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Sparse Symbol Manifest (v2)
===================================

Same prompt-seeded trigram generator as v1, with three structural changes:

  1. INSTRUMENTATION
     Every similarity comparison can report which feature family
     (intrinsic / beam_state / frequency / transitions / lexical)
     contributed how much to the result, instead of returning one
     opaque combined number.

  2. NON-MERGED (COMPONENT) VECTORS
     `influence_vector_components()` returns each feature family as its
     own separate sub-vector (an `InfluenceComponents` object) rather
     than flattening everything into one dict and normalizing it as a
     single unit. The old flat `influence_vector()` still exists (built
     by merging the components) for anything that needs a single vector,
     but the seam between families is now preserved upstream and can be
     inspected before it's collapsed.

  3. HONEST NO-MATCH EXIT
     The generator no longer *always* produces a token. When
     `require_meaningful_match=True` (the new default), if no candidate
     clears `tau` even after adaptive relaxation to `tau_floor`, the
     generator stops and reports `no_meaningful_match_found` with the
     best candidate's per-family breakdown attached, instead of silently
     falling back to an unfiltered top-probability pick. The old
     always-succeed behavior is still available via
     `require_meaningful_match=False` for anyone who wants it.

Everything else (tokenization, counting, trigram backoff order, beam/
state assignment, persistence format) is unchanged from v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

SymbolIndex = int
SparseVector = Dict[int, float]
Context = Tuple[SymbolIndex, SymbolIndex]

FEATURE_FAMILY_RANGES: Dict[str, Tuple[int, int]] = {
    "intrinsic": (0, 4),
    "beam_state": (5, 7),
    "frequency": (8, 8),
}
TRANSITIONS_OFFSET = 9
LEXICAL_OFFSET = 100000
VALID_FAMILIES = ("intrinsic", "beam_state", "frequency", "transitions", "lexical")


@dataclass
class SymbolIntrinsic:
    density: float = 0.0
    relation: float = 0.0
    coherence: float = 0.0
    volatility: float = 0.0
    depth: float = 0.0


@dataclass
class SymbolBeam:
    beam_id: int = 0
    affinity: float = 0.0


@dataclass
class MarkovState:
    state_id: int = 0


@dataclass
class SymbolManifestEntry:
    intrinsic: SymbolIntrinsic = field(default_factory=SymbolIntrinsic)
    beam: SymbolBeam = field(default_factory=SymbolBeam)
    markov: MarkovState = field(default_factory=MarkovState)
    weight: float = 1.0
    raw_count: int = 0


@dataclass
class InfluenceComponents:
    """Per-family sub-vectors, kept separate rather than pre-merged.

    Each family's sub-vector is normalized independently, so the seam
    between "behaves like" (intrinsic/beam_state/frequency), "follows
    like" (transitions), and "is spelled like" (lexical) survives long
    enough to be inspected, weighted, or excluded per-family before any
    combination happens.
    """
    intrinsic: SparseVector = field(default_factory=dict)
    beam_state: SparseVector = field(default_factory=dict)
    frequency: SparseVector = field(default_factory=dict)
    transitions: SparseVector = field(default_factory=dict)
    lexical: SparseVector = field(default_factory=dict)

    def families(self) -> Dict[str, SparseVector]:
        return {
            "intrinsic": self.intrinsic,
            "beam_state": self.beam_state,
            "frequency": self.frequency,
            "transitions": self.transitions,
            "lexical": self.lexical,
        }

    def merged(self, exclude_family: Optional[str] = None) -> SparseVector:
        """Flatten to one dict, for callers that need a single vector."""
        merged: SparseVector = {}
        for name, vector in self.families().items():
            if exclude_family and name == exclude_family:
                continue
            for key, value in vector.items():
                merged[key] = merged.get(key, 0.0) + value
        return merged


class SparseSymbolManifest:
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    VERSION = 4

    def __init__(
        self,
        max_symbols: int = 64000,
        max_beams: int = 16,
        max_states: int = 32,
    ) -> None:
        self.max_symbols = max_symbols
        self.max_beams = max_beams
        self.max_states = max_states

        self.entries: Dict[SymbolIndex, SymbolManifestEntry] = {}
        self.symbol_counts: DefaultDict[SymbolIndex, int] = defaultdict(int)
        self.total_symbols = 0
        self.sequences: List[List[SymbolIndex]] = []
        self.successors: DefaultDict[SymbolIndex, Set[SymbolIndex]] = defaultdict(set)

        self.index_to_word: Dict[SymbolIndex, str] = {}
        self.word_to_index: Dict[str, SymbolIndex] = {}
        self.next_index = 0

        self.transition_counts: DefaultDict[SymbolIndex, DefaultDict[SymbolIndex, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        self.transition_probs: Dict[SymbolIndex, Dict[SymbolIndex, float]] = {}

        self.trigram_counts: DefaultDict[Context, DefaultDict[SymbolIndex, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        self.trigram_probs: Dict[Context, Dict[SymbolIndex, float]] = {}

        self._max_symbol_count: int = 0

    # -----------------------------------------------------------------
    # Tokenization, registration, training (unchanged from v1)
    # -----------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.lower().split()

    @classmethod
    def split_sentences(cls, text: str) -> List[List[str]]:
        sequences = []
        for part in re.split(r"[.!?]+", text):
            tokens = cls.tokenize(part)
            if tokens:
                sequences.append(tokens)
        return sequences

    def _ensure_entry(self, idx: SymbolIndex) -> None:
        if idx not in self.entries:
            self.entries[idx] = SymbolManifestEntry()

    def _register_word(self, word: str) -> SymbolIndex:
        if word in self.word_to_index:
            return self.word_to_index[word]
        if self.next_index >= self.max_symbols:
            raise ValueError(f"Vocabulary exceeds max_symbols={self.max_symbols}")
        idx = self.next_index
        self.next_index += 1
        self.word_to_index[word] = idx
        self.index_to_word[idx] = word
        self._ensure_entry(idx)
        return idx

    def add_sequence_words(self, words: List[str]) -> List[SymbolIndex]:
        words = [word.lower() for word in words if word.strip()]
        if not words:
            return []

        padded = [self.START_TOKEN, self.START_TOKEN, *words, self.END_TOKEN]
        sequence = [self._register_word(word) for word in padded]
        self.sequences.append(sequence)

        for idx in sequence:
            self.symbol_counts[idx] += 1
            self.total_symbols += 1
            if self.symbol_counts[idx] > self._max_symbol_count:
                self._max_symbol_count = self.symbol_counts[idx]

        for current_idx, next_idx in zip(sequence, sequence[1:]):
            self.transition_counts[current_idx][next_idx] += 1
            self.successors[current_idx].add(next_idx)

        for position in range(len(sequence) - 2):
            self.trigram_counts[(sequence[position], sequence[position + 1])][
                sequence[position + 2]
            ] += 1
        return sequence

    def ingest_dataset_words(self, sequences: Iterable[List[str]]) -> None:
        for words in sequences:
            self.add_sequence_words(words)
        self.finalize()

    def ingest_text(self, text: str) -> None:
        self.ingest_dataset_words(self.split_sentences(text))

    # -----------------------------------------------------------------
    # Manifest calculations (unchanged from v1)
    # -----------------------------------------------------------------

    def finalize(self) -> None:
        self._finalize_transitions()
        self._finalize_trigrams()
        self._update_intrinsics()
        self._assign_beams_and_states()

    def _finalize_transitions(self) -> None:
        self.transition_probs = {}
        for current, counts in self.transition_counts.items():
            total = sum(counts.values())
            if total:
                self.transition_probs[current] = {
                    nxt: count / total for nxt, count in counts.items()
                }

    def _finalize_trigrams(self) -> None:
        self.trigram_probs = {}
        for context, counts in self.trigram_counts.items():
            total = sum(counts.values())
            if total:
                self.trigram_probs[context] = {
                    nxt: count / total for nxt, count in counts.items()
                }

    def _update_intrinsics(self) -> None:
        if not self.total_symbols:
            return

        vocabulary_size = max(1, len(self.entries))
        max_degree = max(1, vocabulary_size - 1)

        for idx, entry in self.entries.items():
            entry.intrinsic.density = self.symbol_counts[idx] / self.total_symbols
            entry.raw_count = self.symbol_counts[idx]
            out_degree = len(self.successors[idx])
            entry.intrinsic.relation = out_degree / max_degree
            entry.intrinsic.volatility = out_degree / max_degree

            probabilities = list(self.transition_probs.get(idx, {}).values())
            if len(probabilities) <= 1:
                entry.intrinsic.coherence = 1.0
            else:
                entropy = -sum(p * math.log(p + 1e-12) for p in probabilities)
                entry.intrinsic.coherence = max(
                    0.0,
                    1.0 - entropy / math.log(len(probabilities)),
                )

        position_total: DefaultDict[SymbolIndex, float] = defaultdict(float)
        occurrence_count: DefaultDict[SymbolIndex, int] = defaultdict(int)
        maximum_length = max((len(seq) for seq in self.sequences), default=1)

        for sequence in self.sequences:
            for position, idx in enumerate(sequence):
                position_total[idx] += position
                occurrence_count[idx] += 1

        for idx, entry in self.entries.items():
            if occurrence_count[idx]:
                entry.intrinsic.depth = (
                    position_total[idx] / occurrence_count[idx]
                ) / maximum_length

    def _assign_beams_and_states(self) -> None:
        beam_scale = max(1, self.max_beams - 1)
        state_scale = max(1, self.max_states - 1)

        for entry in self.entries.values():
            beam_score = min(
                1.0,
                (entry.intrinsic.relation + entry.intrinsic.coherence) / 2.0,
            )
            entry.beam.beam_id = max(0, min(self.max_beams - 1, int(beam_score * beam_scale)))
            entry.beam.affinity = beam_score

            state_score = min(
                1.0,
                (entry.intrinsic.volatility + entry.intrinsic.depth) / 2.0,
            )
            entry.markov.state_id = max(0, min(self.max_states - 1, int(state_score * state_scale)))
            entry.weight = 0.5 * entry.intrinsic.density + 0.5 * entry.intrinsic.coherence

    # -----------------------------------------------------------------
    # Sparse vectors: (2) component form kept separate, flat form merged
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize(vector: SparseVector) -> SparseVector:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0.0:
            return dict(vector)
        return {key: value / norm for key, value in vector.items()}

    @staticmethod
    def cosine_similarity(
        vec_a: SparseVector,
        vec_b: SparseVector,
        exclude_family: Optional[str] = None,
    ) -> float:
        if exclude_family:
            vec_a = SparseSymbolManifest._zero_family(vec_a, exclude_family)
            vec_b = SparseSymbolManifest._zero_family(vec_b, exclude_family)
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _zero_family(vector: SparseVector, family: str) -> SparseVector:
        if family not in VALID_FAMILIES:
            raise ValueError(f"Unknown feature family: {family!r}. Valid: {VALID_FAMILIES}")

        if family in FEATURE_FAMILY_RANGES:
            start, end = FEATURE_FAMILY_RANGES[family]
            return {key: value for key, value in vector.items() if not (start <= key <= end)}

        if family == "transitions":
            return {
                key: value
                for key, value in vector.items()
                if not (TRANSITIONS_OFFSET <= key < LEXICAL_OFFSET)
            }

        if family == "lexical":
            return {key: value for key, value in vector.items() if key < LEXICAL_OFFSET}

        return dict(vector)

    def _lexical_vector(self, word: str) -> SparseVector:
        vector: DefaultDict[int, float] = defaultdict(float)
        bounded = f"^{word.lower()}$"
        for size in (1, 2, 3):
            for offset in range(max(0, len(bounded) - size + 1)):
                gram = bounded[offset:offset + size].encode("utf-8")
                bucket = int.from_bytes(
                    hashlib.blake2b(gram, digest_size=8).digest(),
                    "big",
                ) % 8192
                vector[LEXICAL_OFFSET + bucket] += 1.0
        return self._normalize(dict(vector))

    def influence_vector_components(self, idx: SymbolIndex) -> InfluenceComponents:
        """(2) Return each feature family as its own separate, independently
        normalized sub-vector, instead of pre-merging everything into one
        dict. Nothing here decides how families should be weighted or
        combined -- that decision is left to whoever consumes this, and
        can be inspected before it happens.
        """
        entry = self.entries.get(idx)
        if entry is None:
            return InfluenceComponents()

        intrinsic = {
            0: entry.intrinsic.density,
            1: entry.intrinsic.relation,
            2: entry.intrinsic.coherence,
            3: entry.intrinsic.volatility,
            4: entry.intrinsic.depth,
        }
        beam_state = {
            5: entry.beam.affinity,
            6: entry.beam.beam_id / max(1, self.max_beams - 1),
            7: entry.markov.state_id / max(1, self.max_states - 1),
        }
        frequency = {
            8: entry.raw_count / self._max_symbol_count if self._max_symbol_count else 0.0,
        }
        transitions = {
            TRANSITIONS_OFFSET + next_idx: probability
            for next_idx, probability in self.transition_probs.get(idx, {}).items()
        }
        lexical = self._lexical_vector(self.index_to_word[idx])

        return InfluenceComponents(
            intrinsic=self._normalize(intrinsic),
            beam_state=self._normalize(beam_state),
            frequency=self._normalize(frequency),
            transitions=self._normalize(transitions),
            lexical=self._normalize(lexical),
        )

    def influence_vector(
        self,
        idx: SymbolIndex,
        exclude_family: Optional[str] = None,
    ) -> SparseVector:
        """Flat vector, built by merging components. Kept for callers
        (e.g. the whole-vocabulary influence_matrix) that legitimately
        want one combined similarity number rather than a breakdown.
        """
        components = self.influence_vector_components(idx)
        return self._normalize(components.merged(exclude_family=exclude_family))

    def influence_score(self, source_idx: SymbolIndex, target_idx: SymbolIndex) -> float:
        source = self.entries.get(source_idx)
        target = self.entries.get(target_idx)
        if source is None or target is None:
            return 0.0

        transition = self.transition_probs.get(source_idx, {}).get(target_idx, 0.0)
        beam_alignment = 1.0 - abs(source.beam.beam_id - target.beam.beam_id) / max(1, self.max_beams - 1)
        state_alignment = 1.0 - abs(source.markov.state_id - target.markov.state_id) / max(1, self.max_states - 1)
        structural_alignment = 0.5 * beam_alignment + 0.5 * state_alignment

        return max(0.0, source.weight) * max(0.0, target.weight) * (
            0.5 * transition + 0.5 * structural_alignment
        )

    # -----------------------------------------------------------------
    # (1) Instrumentation: per-family similarity breakdown
    # -----------------------------------------------------------------

    def similarity_breakdown(
        self,
        source_idx: SymbolIndex,
        target_idx: SymbolIndex,
        exclude_family: Optional[str] = None,
    ) -> Dict[str, object]:
        """Return which feature family contributed how much to the
        similarity between two words, instead of one opaque number.

        `per_family_cosine`: cosine similarity computed using ONLY that
            family's sub-vectors (so a value near 1.0 in "lexical" and
            near 0.0 in "transitions" tells you the words look alike by
            spelling but have unrelated adjacency histories, etc).
        `combined_cosine`: cosine similarity on the merged flat vector,
            for comparison against the old single-number behavior.
        `dominant_family`: whichever family had the highest per-family
            cosine, i.e. which history the resemblance actually comes from.
        """
        source_components = self.influence_vector_components(source_idx)
        target_components = self.influence_vector_components(target_idx)

        per_family_cosine: Dict[str, float] = {}
        for family in VALID_FAMILIES:
            if exclude_family == family:
                per_family_cosine[family] = 0.0
                continue
            vec_a = source_components.families()[family]
            vec_b = target_components.families()[family]
            per_family_cosine[family] = self.cosine_similarity(vec_a, vec_b)

        combined_a = source_components.merged(exclude_family=exclude_family)
        combined_b = target_components.merged(exclude_family=exclude_family)
        combined_cosine = self.cosine_similarity(
            self._normalize(combined_a), self._normalize(combined_b)
        )

        dominant_family = max(per_family_cosine, key=per_family_cosine.get) if per_family_cosine else None

        return {
            "source_word": self.index_to_word.get(source_idx),
            "target_word": self.index_to_word.get(target_idx),
            "per_family_cosine": per_family_cosine,
            "combined_cosine": combined_cosine,
            "dominant_family": dominant_family,
        }

    # -----------------------------------------------------------------
    # Image-style A x B influence matrix and codomain Y
    # -----------------------------------------------------------------

    def content_indices(self) -> List[SymbolIndex]:
        special = {self.START_TOKEN, self.END_TOKEN}
        return [idx for idx, word in self.index_to_word.items() if word not in special]

    def influence_matrix(
        self,
        tau: float = 0.60,
        exclude_family: Optional[str] = None,
        include_breakdown: bool = False,
    ) -> Dict[str, object]:
        tau = max(-1.0, min(1.0, float(tau)))
        indices = self.content_indices()
        vectors = {idx: self.influence_vector(idx, exclude_family=exclude_family) for idx in indices}
        rows: List[Dict[str, object]] = []

        for source in indices:
            for target in indices:
                if source == target:
                    continue
                cosine = self.cosine_similarity(vectors[source], vectors[target])
                row: Dict[str, object] = {
                    "source_word": self.index_to_word[source],
                    "target_word": self.index_to_word[target],
                    "influence_score": self.influence_score(source, target),
                    "cosine_similarity": cosine,
                    "transition_probability": self.transition_probs.get(source, {}).get(target, 0.0),
                    "kept": cosine >= tau,
                }
                if include_breakdown:
                    row["family_breakdown"] = self.similarity_breakdown(
                        source, target, exclude_family=exclude_family
                    )["per_family_cosine"]
                rows.append(row)

        kept = [row for row in rows if row["kept"]]
        rejected = [row for row in rows if not row["kept"]]
        kept.sort(key=lambda row: (float(row["cosine_similarity"]), float(row["influence_score"])), reverse=True)
        return {
            "tau": tau,
            "exclude_family": exclude_family,
            "domain_size": len(indices) * len(indices),
            "rows": rows,
            "kept_rows": kept,
            "rejected_rows": rejected,
        }

    # -----------------------------------------------------------------
    # Prompt context and backoff
    # -----------------------------------------------------------------

    def _prompt_indices(self, prompt: str) -> Tuple[List[str], List[SymbolIndex], List[str]]:
        words = self.tokenize(prompt)
        known = [self.word_to_index[word] for word in words if word in self.word_to_index]
        unknown = [word for word in words if word not in self.word_to_index]
        return words, known, unknown

    def _resolve_prompt_context(self, known: List[SymbolIndex]) -> Context:
        start = self.word_to_index[self.START_TOKEN]
        if len(known) >= 2 and (known[-2], known[-1]) in self.trigram_probs:
            return known[-2], known[-1]
        if not known:
            raise ValueError("No prompt terms found in vocabulary.")

        current = known[-1]
        matching = [context for context in self.trigram_probs if context[1] == current]
        if matching:
            matching.sort(key=lambda context: sum(self.trigram_counts[context].values()), reverse=True)
            return matching[0]
        return start, current

    def _backoff_distributions(
        self,
        previous_idx: SymbolIndex,
        current_idx: SymbolIndex,
    ) -> List[Tuple[str, Dict[SymbolIndex, float]]]:
        sources: List[Tuple[str, Dict[SymbolIndex, float]]] = []

        exact = self.trigram_probs.get((previous_idx, current_idx), {})
        if exact:
            sources.append(("trigram", exact))

        bigram = self.transition_probs.get(current_idx, {})
        if bigram:
            sources.append(("bigram_backoff", bigram))

        merged: DefaultDict[SymbolIndex, float] = defaultdict(float)
        for (prev, current), successors in self.trigram_probs.items():
            if current != current_idx:
                continue
            mass = sum(self.trigram_counts[(prev, current)].values())
            for next_idx, probability in successors.items():
                merged[next_idx] += probability * max(1, mass)

        total = sum(merged.values())
        if total:
            sources.append((
                "merged_context_backoff",
                {idx: value / total for idx, value in merged.items()},
            ))

        special = {self.START_TOKEN, self.END_TOKEN}
        unigram = {
            idx: count / max(1, self.total_symbols)
            for idx, count in self.symbol_counts.items()
            if self.index_to_word.get(idx) not in special
        }
        if unigram:
            sources.append(("unigram_backoff", unigram))

        return sources

    # -----------------------------------------------------------------
    # (3) Generation with an honest no-meaningful-match exit
    # -----------------------------------------------------------------

    def generate_from_seed_prompt(
        self,
        seed_prompt: str,
        max_new_words: int = 30,
        temperature: float = 0.8,
        tau: float = 0.60,
        candidate_count: int = 32,
        stochastic: bool = True,
        preserve_prompt: bool = True,
        adaptive_tau: bool = True,
        tau_floor: float = 0.05,
        tau_step: float = 0.05,
        require_meaningful_match: bool = True,
        exclude_family: Optional[str] = None,
    ) -> Tuple[List[str], float, List[Dict[str, object]]]:
        """Generate a continuation.

        `require_meaningful_match` (new, default True): if no candidate
        clears `tau` even after relaxing to `tau_floor`, generation stops
        with an honest `no_meaningful_match_found` diagnostic instead of
        silently accepting the best unfiltered candidate. Set this to
        False to restore the old always-succeed fallback behavior.
        """
        prompt_words, known, unknown = self._prompt_indices(seed_prompt)
        if not known:
            raise ValueError(
                "No seed-prompt words were found in the trained vocabulary."
            )
        if max_new_words <= 0:
            return (prompt_words if preserve_prompt else []), 0.0, []

        previous_idx, current_idx = self._resolve_prompt_context(known)
        requested_tau = max(-1.0, min(1.0, float(tau)))
        tau_floor = max(-1.0, min(requested_tau, float(tau_floor)))
        tau_step = max(1e-4, float(tau_step))
        temperature = max(1e-6, float(temperature))

        output = list(prompt_words) if preserve_prompt else []
        selected_cosines: List[float] = []
        diagnostics: List[Dict[str, object]] = []

        for step in range(max_new_words):
            source_name = ""
            distribution: Dict[SymbolIndex, float] = {}

            for name, candidate_distribution in self._backoff_distributions(previous_idx, current_idx):
                usable = {
                    idx: probability
                    for idx, probability in candidate_distribution.items()
                    if self.index_to_word.get(idx) not in {self.START_TOKEN, self.END_TOKEN}
                }
                if usable:
                    source_name = name
                    distribution = usable
                    break

            if not distribution:
                diagnostics.append({
                    "step": step + 1,
                    "stop_reason": "no_candidates_after_all_backoff_levels",
                })
                break

            source_components = self.influence_vector_components(current_idx)
            source_vector = self._normalize(source_components.merged(exclude_family=exclude_family))

            candidates: List[Tuple[float, float, float, float, SymbolIndex, Dict[str, float]]] = []
            ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:max(1, candidate_count)]

            for next_idx, probability in ranked:
                breakdown = self.similarity_breakdown(current_idx, next_idx, exclude_family=exclude_family)
                cosine = breakdown["combined_cosine"]
                influence = self.influence_score(current_idx, next_idx)
                combined = 0.45 * cosine + 0.35 * influence + 0.20 * probability
                candidates.append((combined, cosine, influence, probability, next_idx, breakdown["per_family_cosine"]))

            candidates.sort(key=lambda c: c[0], reverse=True)
            effective_tau = requested_tau
            passing = [candidate for candidate in candidates if candidate[1] >= effective_tau]

            if not passing and adaptive_tau:
                while effective_tau > tau_floor:
                    effective_tau = max(tau_floor, effective_tau - tau_step)
                    passing = [candidate for candidate in candidates if candidate[1] >= effective_tau]
                    if passing:
                        break

            if not passing:
                if require_meaningful_match:
                    best = candidates[0] if candidates else None
                    diagnostics.append({
                        "step": step + 1,
                        "stop_reason": "no_meaningful_match_found",
                        "distribution_source": source_name,
                        "requested_tau": requested_tau,
                        "effective_tau": effective_tau,
                        "best_candidate_word": self.index_to_word[best[4]] if best else None,
                        "best_candidate_cosine": best[1] if best else 0.0,
                        "best_candidate_family_breakdown": best[5] if best else {},
                    })
                    break
                else:
                    # Legacy always-succeed behavior, explicitly opted into.
                    pool = candidates
                    used_fallback = True
            else:
                pool = passing
                used_fallback = False

            if stochastic and len(pool) > 1:
                weights = [max(1e-12, candidate[0]) ** (1.0 / temperature) for candidate in pool]
                combined, cosine, influence, probability, next_idx, family_breakdown = random.choices(
                    pool, weights=weights, k=1
                )[0]
            else:
                combined, cosine, influence, probability, next_idx, family_breakdown = pool[0]

            next_word = self.index_to_word[next_idx]
            output.append(next_word)
            selected_cosines.append(cosine)
            diagnostics.append({
                "step": step + 1,
                "context": [self.index_to_word[previous_idx], self.index_to_word[current_idx]],
                "next_word": next_word,
                "distribution_source": source_name,
                "requested_tau": requested_tau,
                "effective_tau": effective_tau,
                "cosine": cosine,
                "family_breakdown": family_breakdown,
                "dominant_family": max(family_breakdown, key=family_breakdown.get) if family_breakdown else None,
                "influence_score": influence,
                "probability": probability,
                "combined_score": combined,
                "used_unfiltered_fallback": used_fallback,
                "exclude_family": exclude_family,
            })
            previous_idx, current_idx = current_idx, next_idx

        return output, min(selected_cosines) if selected_cosines else 0.0, diagnostics

    # -----------------------------------------------------------------
    # Safe JSON persistence (unchanged from v1)
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "config": {
                "max_symbols": self.max_symbols,
                "max_beams": self.max_beams,
                "max_states": self.max_states,
            },
            "vocabulary": [self.index_to_word[i] for i in range(self.next_index)],
            "sequences": self.sequences,
            "symbol_counts": {str(i): c for i, c in self.symbol_counts.items()},
            "transition_counts": {
                str(source): {str(target): count for target, count in targets.items()}
                for source, targets in self.transition_counts.items()
            },
            "trigram_counts": {
                f"{previous},{current}": {str(target): count for target, count in targets.items()}
                for (previous, current), targets in self.trigram_counts.items()
            },
        }

    def save_json(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, filename: str) -> "SparseSymbolManifest":
        with open(filename, "r", encoding="utf-8") as file:
            state = json.load(file)
        config = state["config"]
        manifest = cls(
            max_symbols=int(config["max_symbols"]),
            max_beams=int(config["max_beams"]),
            max_states=int(config["max_states"]),
        )

        for word in state["vocabulary"]:
            manifest._register_word(str(word))
        manifest.sequences = [[int(index) for index in seq] for seq in state["sequences"]]
        manifest.symbol_counts = defaultdict(
            int,
            {int(index): int(count) for index, count in state["symbol_counts"].items()},
        )
        manifest.total_symbols = sum(manifest.symbol_counts.values())
        manifest._max_symbol_count = max(manifest.symbol_counts.values(), default=0)

        for source, targets in state["transition_counts"].items():
            for target, count in targets.items():
                source_idx, target_idx = int(source), int(target)
                manifest.transition_counts[source_idx][target_idx] = int(count)
                manifest.successors[source_idx].add(target_idx)

        for context, targets in state["trigram_counts"].items():
            previous, current = (int(value) for value in context.split(",", 1))
            for target, count in targets.items():
                manifest.trigram_counts[(previous, current)][int(target)] = int(count)

        manifest.finalize()
        return manifest


def get_corpus(path: Optional[str]) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    with open(input("Filename: "), "r", encoding="utf-8") as file:
        return file.read()


def print_diagnostics(rows: List[Dict[str, object]]) -> None:
    print("\nStep diagnostics")
    print("-" * 110)
    for row in rows:
        if "stop_reason" in row:
            if row["stop_reason"] == "no_meaningful_match_found":
                print(
                    f"STOP | no_meaningful_match_found | best candidate was "
                    f"{row.get('best_candidate_word')!r} at cosine="
                    f"{float(row.get('best_candidate_cosine', 0.0)):.4f} "
                    f"(needed >= {float(row.get('effective_tau', 0.0)):.4f}) | "
                    f"breakdown={row.get('best_candidate_family_breakdown')}"
                )
            else:
                print(f"STOP | {row['stop_reason']}")
            continue
        context = " ".join(row["context"])  # type: ignore[arg-type]
        print(
            f"{int(row['step']):02d} | {context:30s} -> {str(row['next_word']):16s} | "
            f"src={str(row['distribution_source']):22s} | "
            f"cos={float(row['cosine']):.4f} | "
            f"tau={float(row['effective_tau']):.4f} | "
            f"dominant={str(row.get('dominant_family')):12s} | "
            f"fallback={row['used_unfiltered_fallback']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust prompt-seeded sparse cosine influence text generator (v2)")
    parser.add_argument("--input", help="Corpus file path")
    parser.add_argument("--load", help="Load manifest JSON")
    parser.add_argument("--save", help="Save manifest JSON")
    parser.add_argument("--prompt", default="adiabatic dark state", help="Seed prompt")
    parser.add_argument("--new-words", type=int, default=300)
    parser.add_argument("--tau", type=float, default=0.60, help="Requested cosine lower bound")
    parser.add_argument("--tau-floor", type=float, default=0.05, help="Adaptive minimum cosine lower bound")
    parser.add_argument("--tau-step", type=float, default=0.05, help="Adaptive tau decrement")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-adaptive-tau", action="store_true")
    parser.add_argument(
        "--allow-unfiltered-fallback",
        action="store_true",
        help="Restore the old always-succeed behavior instead of stopping honestly on no match.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--exclude-family",
        choices=VALID_FAMILIES,
        default=None,
        help="Zero out one feature family before cosine scoring",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.load:
        manifest = SparseSymbolManifest.load_json(args.load)
        source = f"loaded {args.load}"
    else:
        manifest = SparseSymbolManifest(max_symbols=64000, max_beams=8, max_states=8)
        manifest.ingest_text(get_corpus(args.input))
        source = f"trained from {args.input or 'singlekb.txt / embedded corpus'}"

    if args.save:
        manifest.save_json(args.save)
        print(f"Saved manifest: {args.save}")

    prompt_words, known, unknown = manifest._prompt_indices(args.prompt)
    print("=" * 110)
    print("ROBUST PROMPT-SEEDED COSINE LOWER-BOUND GENERATION (v2: instrumented, componentized, honest-stop)")
    print("=" * 110)
    print(f"Source: {source}")
    print(f"Vocabulary: {len(manifest.content_indices())} words")
    print(f"Prompt: {args.prompt!r}")
    print(f"Known prompt words: {[manifest.index_to_word[idx] for idx in known]}")
    if unknown:
        print(f"Unknown prompt words: {unknown}")
    print(
        f"Requested tau={args.tau:.3f}; adaptive floor={args.tau_floor:.3f}; "
        f"require_meaningful_match={not args.allow_unfiltered_fallback}"
    )
    if args.exclude_family:
        print(f"Excluding feature family from cosine scoring: {args.exclude_family}")

    while True:
        try:
            prompt = input("USER: ")
        except EOFError:
            break

        try:
            words, min_cosine, diagnostics = manifest.generate_from_seed_prompt(
                seed_prompt=prompt,
                max_new_words=args.new_words,
                temperature=args.temperature,
                tau=args.tau,
                candidate_count=args.candidate_count,
                stochastic=not args.greedy,
                preserve_prompt=True,
                adaptive_tau=not args.no_adaptive_tau,
                tau_floor=args.tau_floor,
                tau_step=args.tau_step,
                require_meaningful_match=not args.allow_unfiltered_fallback,
                exclude_family=args.exclude_family,
            )
        except ValueError as exc:
            print(f"Generation failed: {exc}")
            continue

        print("\nGenerated text")
        print("-" * 110)
        print(" ".join(words))


if __name__ == "__main__":
    main()
