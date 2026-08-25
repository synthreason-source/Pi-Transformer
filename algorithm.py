#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Sparse Symbol Manifest
=============================

Prompt-seeded trigram text generation with cosine lower-bound influence
mapping and robust backoff. No external packages are required.

Influence workflow:
    A x B      : source/current-word x target/next-word pairs
    s_ij       : weighted influence score
    c_ij       : cosine similarity between sparse influence vectors
    retain     : c_ij >= tau
    Y          : retained influence space

Prompt generation recovery path:
    exact trigram -> bigram -> merged compatible trigrams -> unigram
    -> adaptive cosine threshold relaxation -> optional unfiltered fallback

Examples:
    python robust_sparse_symbol_manifest.py --input singlekb.txt \
      --prompt "adiabatic dark state" --tau 0.55 --new-words 30

    python robust_sparse_symbol_manifest.py --input corpus.txt \
      --prompt "quantum computing" --tau 0.75 --fallback

    python robust_sparse_symbol_manifest.py --load manifest.json \
      --prompt "neural networks" --tau 0.50 --greedy
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


class SparseSymbolManifest:
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    VERSION = 2

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

    # -----------------------------------------------------------------
    # Tokenization, registration, training
    # -----------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text.lower())

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
    # Manifest calculations
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
    # Sparse influence vectors
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize(vector: SparseVector) -> SparseVector:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0.0:
            return dict(vector)
        return {key: value / norm for key, value in vector.items()}

    @staticmethod
    def cosine_similarity(vec_a: SparseVector, vec_b: SparseVector) -> float:
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

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
                vector[100000 + bucket] += 1.0
        return self._normalize(dict(vector))

    def influence_vector(self, idx: SymbolIndex) -> SparseVector:
        entry = self.entries.get(idx)
        if entry is None:
            return {}

        vector: SparseVector = {
            0: entry.intrinsic.density,
            1: entry.intrinsic.relation,
            2: entry.intrinsic.coherence,
            3: entry.intrinsic.volatility,
            4: entry.intrinsic.depth,
            5: entry.beam.affinity,
            6: entry.beam.beam_id / max(1, self.max_beams - 1),
            7: entry.markov.state_id / max(1, self.max_states - 1),
        }

        for next_idx, probability in self.transition_probs.get(idx, {}).items():
            vector[8 + next_idx] = probability

        for feature, value in self._lexical_vector(self.index_to_word[idx]).items():
            vector[feature] = vector.get(feature, 0.0) + value

        return self._normalize(vector)

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
    # Image-style A x B influence matrix and codomain Y
    # -----------------------------------------------------------------

    def content_indices(self) -> List[SymbolIndex]:
        special = {self.START_TOKEN, self.END_TOKEN}
        return [idx for idx, word in self.index_to_word.items() if word not in special]

    def influence_matrix(self, tau: float = 0.60) -> Dict[str, object]:
        tau = max(-1.0, min(1.0, float(tau)))
        indices = self.content_indices()
        vectors = {idx: self.influence_vector(idx) for idx in indices}
        rows: List[Dict[str, object]] = []

        for source in indices:
            for target in indices:
                if source == target:
                    continue
                cosine = self.cosine_similarity(vectors[source], vectors[target])
                rows.append({
                    "source_word": self.index_to_word[source],
                    "target_word": self.index_to_word[target],
                    "influence_score": self.influence_score(source, target),
                    "cosine_similarity": cosine,
                    "transition_probability": self.transition_probs.get(source, {}).get(target, 0.0),
                    "kept": cosine >= tau,
                })

        kept = [row for row in rows if row["kept"]]
        rejected = [row for row in rows if not row["kept"]]
        kept.sort(key=lambda row: (float(row["cosine_similarity"]), float(row["influence_score"])), reverse=True)
        return {
            "tau": tau,
            "domain_size": len(indices) * len(indices),
            "rows": rows,
            "kept_rows": kept,
            "rejected_rows": rejected,
        }

    # -----------------------------------------------------------------
    # Prompt context and robust generation fallback
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
        """Ordered exact trigram, bigram, merged context, and unigram backoff."""
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
        fallback_to_unfiltered: bool = True,
    ) -> Tuple[List[str], float, List[Dict[str, object]]]:
        """Generate a continuation, preventing zero-token failures via backoff."""
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

            source_vector = self.influence_vector(current_idx)
            candidates: List[Tuple[float, float, float, float, SymbolIndex]] = []
            ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:max(1, candidate_count)]

            for next_idx, probability in ranked:
                cosine = self.cosine_similarity(source_vector, self.influence_vector(next_idx))
                influence = self.influence_score(current_idx, next_idx)
                combined = 0.45 * cosine + 0.35 * influence + 0.20 * probability
                candidates.append((combined, cosine, influence, probability, next_idx))

            candidates.sort(reverse=True)
            effective_tau = requested_tau
            passing = [candidate for candidate in candidates if candidate[1] >= effective_tau]

            if not passing and adaptive_tau:
                while effective_tau > tau_floor:
                    effective_tau = max(tau_floor, effective_tau - tau_step)
                    passing = [candidate for candidate in candidates if candidate[1] >= effective_tau]
                    if passing:
                        break

            used_fallback = False
            if passing:
                pool = passing
            elif fallback_to_unfiltered:
                pool = candidates
                used_fallback = True
            else:
                diagnostics.append({
                    "step": step + 1,
                    "stop_reason": "no_candidate_passed_cosine_threshold",
                    "distribution_source": source_name,
                    "requested_tau": requested_tau,
                    "effective_tau": effective_tau,
                    "best_available_cosine": candidates[0][1] if candidates else 0.0,
                })
                break

            if stochastic and len(pool) > 1:
                weights = [max(1e-12, candidate[0]) ** (1.0 / temperature) for candidate in pool]
                combined, cosine, influence, probability, next_idx = random.choices(pool, weights=weights, k=1)[0]
            else:
                combined, cosine, influence, probability, next_idx = pool[0]

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
                "influence_score": influence,
                "probability": probability,
                "combined_score": combined,
                "passed_cosine_filter": not used_fallback,
                "used_unfiltered_fallback": used_fallback,
            })
            previous_idx, current_idx = current_idx, next_idx

        return output, min(selected_cosines) if selected_cosines else 0.0, diagnostics

    # -----------------------------------------------------------------
    # Safe JSON persistence
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
    if os.path.exists("singlekb.txt"):
        with open("singlekb.txt", "r", encoding="utf-8") as file:
            return file.read()
    return (
        "adiabatic dark state transfer moves population between quantum sites. "
        "adiabatic dark state transfer protects coherence from a lossy middle site. "
        "quantum computing processes information through coherent quantum states. "
        "quantum computing uses superposition to process information. "
        "neural networks learn patterns from data through optimization. "
        "neural networks optimize parameters using gradient descent. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox runs past the lazy cat."
    )


def print_diagnostics(rows: List[Dict[str, object]]) -> None:
    print("\nStep diagnostics")
    print("-" * 100)
    for row in rows:
        if "stop_reason" in row:
            print(f"STOP | {row['stop_reason']}")
            continue
        context = " ".join(row["context"])  # type: ignore[arg-type]
        print(
            f"{int(row['step']):02d} | {context:30s} -> {str(row['next_word']):16s} | "
            f"src={str(row['distribution_source']):22s} | "
            f"cos={float(row['cosine']):.4f} | "
            f"tau={float(row['effective_tau']):.4f} | "
            f"s={float(row['influence_score']):.5f} | "
            f"fallback={row['used_unfiltered_fallback']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust prompt-seeded sparse cosine influence text generator")
    parser.add_argument("--input", help="Corpus file path")
    parser.add_argument("--load", help="Load manifest JSON")
    parser.add_argument("--save", help="Save manifest JSON")
    parser.add_argument("--prompt", default="adiabatic dark state", help="Seed prompt")
    parser.add_argument("--new-words", type=int, default=30)
    parser.add_argument("--tau", type=float, default=0.60, help="Requested cosine lower bound")
    parser.add_argument("--tau-floor", type=float, default=0.05, help="Adaptive minimum cosine lower bound")
    parser.add_argument("--tau-step", type=float, default=0.05, help="Adaptive tau decrement")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-adaptive-tau", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--show-mapping", type=int, default=10)
    parser.add_argument("--seed", type=int)
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
    print("=" * 100)
    print("ROBUST PROMPT-SEEDED COSINE LOWER-BOUND GENERATION")
    print("=" * 100)
    print(f"Source: {source}")
    print(f"Vocabulary: {len(manifest.content_indices())} words")
    print(f"Prompt: {args.prompt!r}")
    print(f"Known prompt words: {[manifest.index_to_word[idx] for idx in known]}")
    if unknown:
        print(f"Unknown prompt words: {unknown}")
    print(f"Requested tau={args.tau:.3f}; adaptive floor={args.tau_floor:.3f}; fallback={not args.no_fallback}")

    try:
        words, min_cosine, diagnostics = manifest.generate_from_seed_prompt(
            seed_prompt=args.prompt,
            max_new_words=args.new_words,
            temperature=args.temperature,
            tau=args.tau,
            candidate_count=args.candidate_count,
            stochastic=not args.greedy,
            preserve_prompt=True,
            adaptive_tau=not args.no_adaptive_tau,
            tau_floor=args.tau_floor,
            tau_step=args.tau_step,
            fallback_to_unfiltered=not args.no_fallback,
        )
    except ValueError as exc:
        print(f"Generation failed: {exc}")
        return

    generated_count = sum(1 for row in diagnostics if "next_word" in row)
    print("\nGenerated text")
    print("-" * 100)
    print(" ".join(words))
   

if __name__ == "__main__":
    main()
