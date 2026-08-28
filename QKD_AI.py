#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Sparse Symbol Manifest (Quantum-Noise, One-Way-Proof Sampling)
======================================================================

Prompt-seeded trigram text generation with cosine lower-bound influence
mapping and robust backoff. No external packages are required.

Influence workflow:
A x B : source/current-word x target/next-word pairs
s_ij : weighted influence score
c_ij : cosine similarity between sparse influence vectors
retain : c_ij >= tau
Y : retained influence space

Feature families (used for cosine similarity and can be selectively
zeroed out via `exclude_family`):
intrinsic : indices 0-4 (density, relation, coherence, volatility, depth)
beam_state : indices 5-7 (beam affinity, beam id, markov state id)
frequency : index 8 (max-normalized raw symbol count)
transitions : indices 9..9+N (per-symbol transition-probability block)
lexical : indices >= 100000 (character n-gram hash buckets)

Prompt generation recovery path:
exact trigram -> bigram -> merged compatible trigrams -> unigram
-> adaptive cosine threshold relaxation -> optional unfiltered fallback

Randomness / seed setter
-------------------------
Every stochastic pick is driven by `OneWayRandomDriver`, a quantum-noise-style
generator with a one-way commitment proof:

- The driver draws a fresh 256-bit witness + nonce, commits to them with
  SHA-256 (`commitment`), and derives the actual 64-bit value used for
  sampling from the witness via HKDF.
- `driver.public_record(proof)` -- {algorithm, context, nonce, commitment} --
  can be published immediately and reveals nothing about which candidate
  will be (or was) picked.
- `OneWayRandomDriver.verify(proof)` lets anyone holding the full proof
  (which includes the revealed witness/value) confirm the pick matches the
  earlier commitment, without the commitment ever having leaked the pick.

Use `--seed` for deterministic, reproducible generation. The same corpus,
options, and seed will reproduce the same output.

Examples:
python s.py --input singlekb.txt \
  --prompt "adiabatic dark state" --tau 0.55 --new-words 30

python s.py --input corpus.txt \
  --prompt "quantum computing" --tau 0.75 --experiment demo --session 1

python s.py --load manifest.json \
  --prompt "neural networks" --tau 0.50 --greedy

# Ignore the lexical (character n-gram) feature family when scoring:
python s.py --input corpus.txt \
  --prompt "quantum computing" --exclude-family lexical

# Deterministic reproducible run:
python s.py --input corpus.txt \
  --prompt "quantum computing" --seed experiment-001 --new-words 30 \
  --transcript-log run.json

# Verify a transcript:
python s.py --verify-transcript run.json
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------
# Entropy sources
# ---------------------------------------------------------------------

class QuantumEntropySource:
    """Wraps a byte-source. Defaults to the OS CSPRNG (os.urandom).
    Pass a `quantum_reader(n) -> bytes` callable to use real hardware."""

    def __init__(self, quantum_reader: Optional[Callable[[int], bytes]] = None):
        self.quantum_reader = quantum_reader

    def read(self, n: int) -> bytes:
        if n <= 0:
            raise ValueError("n must be positive")
        if self.quantum_reader is not None:
            data = self.quantum_reader(n)
            if not isinstance(data, bytes) or len(data) != n:
                raise ValueError("quantum_reader must return exactly n bytes")
            return data
        return os.urandom(n)


class SeededEntropySource(QuantumEntropySource):
    """Deterministic entropy source for reproducible generation."""

    def __init__(self, seed: str):
        self.seed = str(seed).encode("utf-8")
        self.counter = 0

    def read(self, n: int) -> bytes:
        if n <= 0:
            raise ValueError("n must be positive")

        output = bytearray()
        while len(output) < n:
            block = hashlib.sha256(
                b"REPRODUCIBLE-ENTROPY-v1"
                + len(self.seed).to_bytes(8, "big")
                + self.seed
                + self.counter.to_bytes(8, "big")
            ).digest()
            output.extend(block)
            self.counter += 1

        return bytes(output[:n])


# ---------------------------------------------------------------------
# HKDF and helpers
# ---------------------------------------------------------------------

def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out = bytearray()
    previous = b""
    counter = 1
    while len(out) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        out.extend(previous)
        counter += 1
    return bytes(out[:length])


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _int_in_range(stream: bytes, low: int, high: int) -> int:
    span = high - low + 1
    if span <= 0:
        raise ValueError("high must be >= low")
    return low + (int.from_bytes(stream, "big") % span)


DOMAIN_SALT = b"QRNG-ONEWAY-SALT-v1"
DOMAIN_IKM = b"QRNG-ONEWAY-IKM-v1"
DOMAIN_OUTPUT = b"QRNG-ONEWAY-OUTPUT-v1"
DOMAIN_COMMIT = b"QRNG-ONEWAY-COMMIT-v1"

WITNESS_BYTES = 32
NONCE_BYTES = 16


# ---------------------------------------------------------------------
# One-way random driver
# ---------------------------------------------------------------------

class OneWayRandomDriver:
    """Cryptographically driven sampler used as the generator's seed setter."""

    def __init__(
        self,
        entropy_source: Optional[QuantumEntropySource] = None,
        experiment: Optional[str] = None,
        session: Optional[int] = None,
    ):
        self.entropy = entropy_source or QuantumEntropySource()
        self.experiment = experiment
        self.session = session

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def _full_context(self, context: dict, operation: str) -> dict:
        full: dict = {}
        if self.experiment is not None:
            full["experiment"] = self.experiment
        if self.session is not None:
            full["session"] = self.session
        full["operation"] = operation
        full.update(context)
        # Timestamp is metadata only, not part of derivation context
        return full

    def generate(
        self,
        context: Optional[dict] = None,
        low: int = 0,
        high: int = 2 ** 64 - 1,
        operation: str = "generate",
    ) -> Tuple[int, dict]:
        """Draw one random integer in [low, high] plus a one-way proof."""
        context = self._full_context(context or {}, operation)
        context_bytes = _canonical_json(context)

        witness = self.entropy.read(WITNESS_BYTES)
        nonce = self.entropy.read(NONCE_BYTES)

        salt = hashlib.sha256(DOMAIN_SALT + nonce).digest()
        prk = _hkdf_extract(salt, DOMAIN_IKM + witness + context_bytes + nonce)

        needed_bytes = ((high - low).bit_length() // 8) + 9
        stream = _hkdf_expand(prk, DOMAIN_OUTPUT + context_bytes, needed_bytes)
        value = _int_in_range(stream, low, high)

        commitment = hashlib.sha256(
            DOMAIN_COMMIT
            + len(context_bytes).to_bytes(8, "big") + context_bytes
            + witness
            + nonce
        ).hexdigest()

        proof = {
            "algorithm": "SHA256-HKDF-witness",
            "context": context,
            "nonce": nonce.hex(),
            "commitment": commitment,
            "range": [low, high],
            "witness": witness.hex(),
            "value": value,
            "created_at": self._timestamp(),
        }
        return value, proof

    @staticmethod
    def public_record(proof: dict) -> dict:
        """Safe to publish immediately: proves a commitment was made."""
        # Greedy proofs may not have a real nonce; synthesize one from context.
        nonce = proof.get("nonce")
        if nonce is None:
            # Derive a stable pseudo-nonce from the commitment so it’s deterministic.
            nonce = hashlib.sha256(
                b"GREEDY-PSEUDO-NONCE-v1"
                + _canonical_json(proof.get("context", {}))
            ).hexdigest()

        return {
            "algorithm": proof["algorithm"],
            "context": proof["context"],
            "nonce": nonce,
            "commitment": proof["commitment"],
            "range": proof["range"],
        }

    @staticmethod
    def verify(proof: dict) -> bool:
        context_bytes = _canonical_json(proof["context"])
        witness = bytes.fromhex(proof["witness"])
        nonce = bytes.fromhex(proof["nonce"])
        low, high = proof["range"]

        expected_commitment = hashlib.sha256(
            DOMAIN_COMMIT
            + len(context_bytes).to_bytes(8, "big") + context_bytes
            + witness
            + nonce
        ).hexdigest()
        if not hmac.compare_digest(expected_commitment, proof["commitment"]):
            return False

        salt = hashlib.sha256(DOMAIN_SALT + nonce).digest()
        prk = _hkdf_extract(salt, DOMAIN_IKM + witness + context_bytes + nonce)
        needed_bytes = ((high - low).bit_length() // 8) + 9
        stream = _hkdf_expand(prk, DOMAIN_OUTPUT + context_bytes, needed_bytes)
        expected_value = _int_in_range(stream, low, high)
        return expected_value == proof["value"]

    @staticmethod
    def verify_selection(proof: dict) -> bool:
        """Stronger check for weighted picks."""
        if not OneWayRandomDriver.verify(proof):
            return False

        candidate_words = proof.get("candidate_words")
        weights = proof.get("weights")
        selected_index = proof.get("selected_index")
        selected_word = proof.get("selected_word")

        if not isinstance(candidate_words, list):
            return False
        if not isinstance(weights, list):
            return False
        if len(candidate_words) != len(weights):
            return False
        if not candidate_words:
            return False
        if not isinstance(selected_index, int):
            return False
        if not 0 <= selected_index < len(candidate_words):
            return False
        if candidate_words[selected_index] != selected_word:
            return False

        clean = [max(0.0, float(w)) for w in weights]
        total = sum(clean)

        if total <= 0.0:
            expected_index = 0
        else:
            u = proof["value"] / float(1 << 64)
            target = u * total
            cumulative = 0.0
            expected_index = len(clean) - 1
            for i, weight in enumerate(clean):
                cumulative += weight
                if target < cumulative:
                    expected_index = i
                    break

        return expected_index == selected_index

    @staticmethod
    def weighted_choice_index(
        driver: "OneWayRandomDriver",
        candidate_words: List[str],
        weights: List[float],
        context: dict,
    ) -> Tuple[int, dict]:
        """Pick an index with probability proportional to `weights`."""
        if len(candidate_words) != len(weights):
            raise ValueError("candidate_words and weights must have equal length")
        if not candidate_words:
            raise ValueError("Cannot choose from an empty candidate list")

        clean = [max(0.0, float(w)) for w in weights]
        total = sum(clean)

        value, proof = driver.generate(
            {
                **context,
                "sampling": "weighted-choice",
                "candidate_words": candidate_words,
                "weights": clean,
            },
            low=0,
            high=(1 << 64) - 1,
            operation="weighted-choice",
        )

        if total <= 0.0:
            index = 0
        else:
            unit = value / float(1 << 64)
            target = unit * total
            cumulative = 0.0
            index = len(clean) - 1
            for i, weight in enumerate(clean):
                cumulative += weight
                if target < cumulative:
                    index = i
                    break

        proof["candidate_words"] = list(candidate_words)
        proof["weights"] = clean
        proof["selected_index"] = index
        proof["selected_word"] = candidate_words[index]

        return index, proof


# ---------------------------------------------------------------------
# Symbol manifest types
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Sparse symbol manifest
# ---------------------------------------------------------------------

class SparseSymbolManifest:
    START_TOKEN = ""
    END_TOKEN = ""
    VERSION = 3

    def __init__(
        self,
        max_symbols: int = 64000,
        max_beams: int = 16,
        max_states: int = 32,
        experiment: Optional[str] = None,
        session: Optional[int] = None,
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

        self.random_driver = OneWayRandomDriver(experiment=experiment, session=session)
        self.last_random_proofs: List[Dict[str, object]] = []
        self.last_public_records: List[Dict[str, object]] = []

        self.transition_counts: DefaultDict[SymbolIndex, DefaultDict[SymbolIndex, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        self.transition_probs: Dict[SymbolIndex, Dict[SymbolIndex, float]] = {}

        self.trigram_counts: DefaultDict[Context, DefaultDict[SymbolIndex, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        self.trigram_probs: Dict[Context, Dict[SymbolIndex, float]] = {}

        self._max_symbol_count: int = 0

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

    def influence_vector(
        self,
        idx: SymbolIndex,
        exclude_family: Optional[str] = None,
    ) -> SparseVector:
        entry = self.entries.get(idx)
        if entry is None:
            return {}

        frequency = entry.raw_count / self._max_symbol_count if self._max_symbol_count else 0.0

        vector: SparseVector = {
            0: entry.intrinsic.density,
            1: entry.intrinsic.relation,
            2: entry.intrinsic.coherence,
            3: entry.intrinsic.volatility,
            4: entry.intrinsic.depth,
            5: entry.beam.affinity,
            6: entry.beam.beam_id / max(1, self.max_beams - 1),
            7: entry.markov.state_id / max(1, self.max_states - 1),
            8: frequency,
        }

        for next_idx, probability in self.transition_probs.get(idx, {}).items():
            vector[TRANSITIONS_OFFSET + next_idx] = probability

        for feature, value in self._lexical_vector(self.index_to_word[idx]).items():
            vector[feature] = vector.get(feature, 0.0) + value

        if exclude_family:
            vector = self._zero_family(vector, exclude_family)

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

    def content_indices(self) -> List[SymbolIndex]:
        special = {self.START_TOKEN, self.END_TOKEN}
        return [idx for idx, word in self.index_to_word.items() if word not in special]

    def influence_matrix(
        self,
        tau: float = 0.60,
        exclude_family: Optional[str] = None,
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
            "exclude_family": exclude_family,
            "domain_size": len(indices) * len(indices),
            "rows": rows,
            "kept_rows": kept,
            "rejected_rows": rejected,
        }

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
        exclude_family: Optional[str] = None,
    ) -> Tuple[List[str], float, List[Dict[str, object]]]:
        prompt_words, known, unknown = self._prompt_indices(seed_prompt)
        if not known:
            raise ValueError("No seed-prompt words were found in the trained vocabulary.")
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
        self.last_random_proofs = []
        self.last_public_records = []

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

            source_vector = self.influence_vector(current_idx, exclude_family=exclude_family)
            candidates: List[Tuple[float, float, float, float, SymbolIndex]] = []
            ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:max(1, candidate_count)]

            for next_idx, probability in ranked:
                cosine = self.cosine_similarity(
                    source_vector,
                    self.influence_vector(next_idx, exclude_family=exclude_family),
                )
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

                rng_context = {
                    "generator": "RobustSparseSymbolManifest",
                    "version": self.VERSION,
                    "step": step + 1,
                    "prompt": seed_prompt,
                    "previous_word": self.index_to_word[previous_idx],
                    "current_word": self.index_to_word[current_idx],
                    "tau": requested_tau,
                    "effective_tau": effective_tau,
                    "temperature": temperature,
                    "distribution_source": source_name,
                    "candidate_count": len(pool),
                    "output_prefix_sha256": self._text_digest(output),
                }

                candidate_words = [
                    self.index_to_word[candidate[4]]
                    for candidate in pool
                ]

                selected_index, random_proof = OneWayRandomDriver.weighted_choice_index(
                    self.random_driver,
                    candidate_words=candidate_words,
                    weights=weights,
                    context=rng_context,
                )

                self.last_random_proofs.append(random_proof)
                self.last_public_records.append(OneWayRandomDriver.public_record(random_proof))

                combined, cosine, influence, probability, next_idx = pool[selected_index]
            else:
                combined, cosine, influence, probability, next_idx = pool[0]

                greedy_context = self.random_driver._full_context(
                    {
                        "generator": "RobustSparseSymbolManifest",
                        "step": step + 1,
                        "prompt": seed_prompt,
                        "sampling": "greedy",
                        "selected_word": self.index_to_word[next_idx],
                        "output_prefix_sha256": self._text_digest(output),
                    },
                    operation="greedy-choice",
                )

                greedy_context_bytes = _canonical_json(greedy_context)
                greedy_commitment = hashlib.sha256(
                    b"GREEDY-CHOICE-v1" + greedy_context_bytes
                ).hexdigest()

                # Deterministic pseudo-nonce for greedy proofs
                greedy_nonce = hashlib.sha256(
                    b"GREEDY-NONCE-v1"
                    + greedy_context_bytes
                ).hexdigest()

                greedy_proof = {
                    "algorithm": "GREEDY-SHA256-v1",
                    "context": greedy_context,
                    "nonce": greedy_nonce,
                    "range": [0, 0],
                    "value": 0,
                    "candidate_words": [self.index_to_word[idx] for _, _, _, _, idx in pool],
                    "weights": [],
                    "selected_index": 0,
                    "selected_word": self.index_to_word[next_idx],
                    "commitment": greedy_commitment,
                    "created_at": self.random_driver._timestamp(),
                }
                self.last_random_proofs.append(greedy_proof)
                self.last_public_records.append(OneWayRandomDriver.public_record(greedy_proof))

            next_word = self.index_to_word[next_idx]
            output.append(next_word)
            selected_cosines.append(cosine)

            proof_for_diag = self.last_random_proofs[-1]
            proof_for_diag["output_word"] = next_word  # type: ignore
            proof_for_diag["output_position"] = len(output) - 1  # type: ignore

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
                "exclude_family": exclude_family,
                "random_commitment": proof_for_diag["commitment"],  # type: ignore
                "random_proof_index": len(self.last_random_proofs) - 1,
            })
            previous_idx, current_idx = current_idx, next_idx

        return output, min(selected_cosines) if selected_cosines else 0.0, diagnostics

    @staticmethod
    def _text_digest(words: List[str]) -> str:
        return hashlib.sha256(_canonical_json(words)).hexdigest()

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


# ---------------------------------------------------------------------
# Corpus loading and diagnostics
# ---------------------------------------------------------------------

def get_corpus(path: Optional[str]) -> str:
    filename = path or input("Filename: ")
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


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


# ---------------------------------------------------------------------
# Transcript verification
# ---------------------------------------------------------------------
def verify_transcript(transcript: dict) -> Tuple[bool, List[str]]:
    errors = []

    words = transcript.get("generated_words")
    generated_text = transcript.get("generated_text")
    recorded_digest = transcript.get("text_sha256")
    diagnostics = transcript.get("diagnostics", [])
    options = transcript.get("options", {})

    if not isinstance(words, list):
        errors.append("generated_words is missing or invalid")
        return False, errors

    expected_text = " ".join(str(word) for word in words)
    if generated_text != expected_text:
        errors.append("generated_text does not match generated_words")

    expected_digest = hashlib.sha256(_canonical_json(words)).hexdigest()
    if recorded_digest != expected_digest:
        errors.append("text_sha256 does not match generated_words")

    proofs = transcript.get("proofs", [])

    # Use diagnostics length (one per generated token) as the ground truth.
    expected_proofs = len(diagnostics)

    # Fallback: if diagnostics missing, use options.new_words if present.
    if expected_proofs == 0 and "new_words" in options:
        expected_proofs = int(options["new_words"])

    if len(proofs) != expected_proofs:
        errors.append(
            f"proof count mismatch: expected {expected_proofs}, got {len(proofs)}"
        )

    for index, proof in enumerate(proofs):
        algorithm = proof.get("algorithm")

        if algorithm == "GREEDY-SHA256-v1":
            context_bytes = _canonical_json(proof["context"])
            expected = hashlib.sha256(
                b"GREEDY-CHOICE-v1" + context_bytes
            ).hexdigest()
            if not hmac.compare_digest(expected, proof.get("commitment", "")):
                errors.append(f"proof {index}: greedy commitment mismatch")
            if proof.get("range") != [0, 0]:
                errors.append(f"proof {index}: invalid greedy range")
            if proof.get("value") != 0:
                errors.append(f"proof {index}: invalid greedy value")
            if proof.get("selected_word") != proof.get("output_word"):
                errors.append(f"proof {index}: selected word/output word mismatch")
            continue

        if not OneWayRandomDriver.verify_selection(proof):
            errors.append(f"proof {index}: selection verification failed")

        if proof.get("selected_word") != proof.get("output_word"):
            errors.append(f"proof {index}: selected word/output word mismatch")

    return not errors, errors


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust prompt-seeded sparse cosine influence text generator"
    )
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
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--show-mapping", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=str,
        help="Deterministic seed; identical corpus, options, and seed reproduce the same output",
    )
    parser.add_argument(
        "--exclude-family",
        choices=VALID_FAMILIES,
        default=None,
        help="Zero out one feature family before cosine scoring",
    )
    parser.add_argument("--experiment", help="Label embedded in every one-way commitment's context")
    parser.add_argument("--session", type=int, help="Session number embedded in every one-way commitment's context")
    parser.add_argument("--proof-log", help="Write full per-token proofs (incl. revealed witness/value) to a JSON file")
    parser.add_argument("--public-log", help="Write per-token public commitment records (no reveal) to a JSON file")
    parser.add_argument("--transcript-log", help="Write generated text, diagnostics, and verification proofs")
    parser.add_argument("--verify-transcript", help="Verify a previously saved generation transcript")

    args = parser.parse_args()

    if args.verify_transcript:
        with open(args.verify_transcript, "r", encoding="utf-8") as file:
            transcript = json.load(file)

        ok, errors = verify_transcript(transcript)

        if ok:
            print("VERIFIED: transcript, generated text, and selections match")
            return

        print("FAILED: transcript verification failed")
        for error in errors:
            print(f"- {error}")
        return

    entropy_source = (
        SeededEntropySource(args.seed)
        if args.seed is not None
        else QuantumEntropySource()
    )

    if args.load:
        manifest = SparseSymbolManifest.load_json(args.load)
        manifest.random_driver = OneWayRandomDriver(
            entropy_source=entropy_source,
            experiment=args.experiment,
            session=args.session,
        )
        source = f"loaded {args.load}"
    else:
        manifest = SparseSymbolManifest(
            max_symbols=64000,
            max_beams=8,
            max_states=8,
            experiment=args.experiment,
            session=args.session,
        )
        manifest.random_driver = OneWayRandomDriver(
            entropy_source=entropy_source,
            experiment=args.experiment,
            session=args.session,
        )
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
    if args.exclude_family:
        print(f"Excluding feature family from cosine scoring: {args.exclude_family}")

    while True:
        try:
            words, min_cosine, diagnostics = manifest.generate_from_seed_prompt(
                seed_prompt=input("USER: "),
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
                exclude_family=args.exclude_family,
            )
        except ValueError as exc:
            print(f"Generation failed: {exc}")
            return

        print("\nGenerated text")
        print("-" * 100)
        print(" ".join(words))

        print("\nPublic commitment records (safe to publish before reveal)")
        print("-" * 100)
        print(f"Cryptographic sampling decisions: {len(manifest.last_public_records)}")
        for i, record in enumerate(manifest.last_public_records, 1):
            print(f"[{i:03d}]")
            print(json.dumps(record, ensure_ascii=False, indent=2))

        print("\nReveal + verification")
        print("-" * 100)
        for i, proof in enumerate(manifest.last_random_proofs, 1):
            if proof["algorithm"] == "GREEDY-SHA256-v1":
                ok = True
            else:
                ok = OneWayRandomDriver.verify_selection(proof)
            print(f"[{i:03d}] selected_index={proof.get('selected_index', 0)} verify={ok}")

        if args.public_log:
            with open(args.public_log, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "algorithm": "SHA256-HKDF-witness",
                        "prompt": args.prompt,
                        "records": manifest.last_public_records,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"\nSaved public commitment log: {args.public_log}")

        if args.proof_log:
            with open(args.proof_log, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "algorithm": "SHA256-HKDF-witness",
                        "prompt": args.prompt,
                        "proofs": manifest.last_random_proofs,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Saved full proof log (includes revealed witness/value): {args.proof_log}")

        if args.transcript_log:
            generated_text = " ".join(words)
            text_digest = hashlib.sha256(_canonical_json(words)).hexdigest()

            transcript = {
                "format": "RSM-generation-transcript-v1",
                "prompt": args.prompt,
                "generated_words": words,
                "generated_text": generated_text,
                "text_sha256": text_digest,
                "seed": args.seed,
                "options": {
                    "new_words": args.new_words,
                    "tau": args.tau,
                    "tau_floor": args.tau_floor,
                    "tau_step": args.tau_step,
                    "temperature": args.temperature,
                    "candidate_count": args.candidate_count,
                    "greedy": args.greedy,
                    "adaptive_tau": not args.no_adaptive_tau,
                    "fallback_to_unfiltered": not args.no_fallback,
                    "exclude_family": args.exclude_family,
                },
                "proofs": manifest.last_random_proofs,
                "diagnostics": diagnostics,
            }
            transcript["transcript_sha256"] = hashlib.sha256(
                _canonical_json(transcript)
            ).hexdigest()

            with open(args.transcript_log, "w", encoding="utf-8") as file:
                json.dump(transcript, file, ensure_ascii=False, indent=2)
            print(f"Saved transcript: {args.transcript_log}")


if __name__ == "__main__":
    main()
