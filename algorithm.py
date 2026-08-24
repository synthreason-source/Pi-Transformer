#!/usr/bin/env python3
"""
Optical-bench generation driven by a sparse trigram symbol manifest.

This merges two things:

1. SparseSymbolManifest — a trigram word model that learns
       P(word_n | word_(n-2), word_(n-1))
   from a text dataset (no torch/transformers needed).

2. The optical sampling bench from the GPT script — instead of a
   transformer's logits, the trigram model's P(next | context) is
   used as the "digital" probability vector. That vector is handed
   to the optical emulator (m = p + noise, renormalized), and the
   optical measurement — not argmax/softmax sampling — decides the
   next token.

Usage:
    python optical_symbol_gpt.py --file corpus.txt --num-sequences 10
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

SymbolIndex = int


# ===============================================================
# SPARSE TRIGRAM SYMBOL MANIFEST  (unchanged model, from symbol file)
# ===============================================================

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

    def __init__(self, max_symbols: int = 256, max_beams: int = 16, max_states: int = 32):
        self.max_symbols = max_symbols
        self.max_beams = max_beams
        self.max_states = max_states

        self.entries: Dict[SymbolIndex, SymbolManifestEntry] = {}
        self.symbol_counts: Dict[SymbolIndex, int] = defaultdict(int)
        self.total_symbols = 0
        self.sequences: List[List[SymbolIndex]] = []

        self.successors: Dict[SymbolIndex, Set[SymbolIndex]] = defaultdict(set)

        self.index_to_word: Dict[SymbolIndex, str] = {}
        self.word_to_index: Dict[str, SymbolIndex] = {}
        self.next_index = 0

        self.transition_counts: Dict[SymbolIndex, Dict[SymbolIndex, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_probs: Dict[SymbolIndex, Dict[SymbolIndex, float]] = {}

        self.trigram_counts: Dict[Tuple[SymbolIndex, SymbolIndex], Dict[SymbolIndex, int]] = defaultdict(lambda: defaultdict(int))
        self.trigram_probs: Dict[Tuple[SymbolIndex, SymbolIndex], Dict[SymbolIndex, float]] = {}

    def _ensure_entry(self, idx: SymbolIndex) -> None:
        if idx not in self.entries:
            self.entries[idx] = SymbolManifestEntry()

    def _register_word(self, word: str) -> SymbolIndex:
        if word not in self.word_to_index:
            if self.next_index >= self.max_symbols:
                raise ValueError(f"Vocabulary exceeds max_symbols={self.max_symbols}")
            idx = self.next_index
            self.next_index += 1
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
            self._ensure_entry(idx)
            return idx
        return self.word_to_index[word]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.split()

    def add_sequence_words(self, words: List[str]) -> List[SymbolIndex]:
        words = [word for word in words if word.strip()]
        if not words:
            return []

        padded_words = [self.START_TOKEN, self.START_TOKEN, *words, self.END_TOKEN]
        seq = [self._register_word(word) for word in padded_words]
        self.sequences.append(seq)

        for symbol in seq:
            self.symbol_counts[symbol] += 1
            self.total_symbols += 1

        for current, nxt in zip(seq, seq[1:]):
            self.transition_counts[current][nxt] += nxt
            self.successors[current].add(nxt)

        for i in range(len(seq) - 2):
            previous_word = seq[i]
            current_word = seq[i + 1]
            next_word = seq[i + 2]
            self.trigram_counts[(previous_word, current_word)][next_word] += 1

        return seq

    def ingest_dataset_words(self, sequences: List[List[str]]) -> None:
        for words in sequences:
            self.add_sequence_words(words)

        self._finalize_transitions()
        self._finalize_trigrams()
        self._update_intrinsics()
        self._assign_beams_and_states()

    def _finalize_transitions(self) -> None:
        self.transition_probs = {}
        for current, next_words in self.transition_counts.items():
            total = sum(next_words.values())
            if total:
                self.transition_probs[current] = {nxt: count / total for nxt, count in next_words.items()}

    def _finalize_trigrams(self) -> None:
        self.trigram_probs = {}
        for context, next_words in self.trigram_counts.items():
            total = sum(next_words.values())
            if total:
                self.trigram_probs[context] = {nxt: count / total for nxt, count in next_words.items()}

    def _update_intrinsics(self) -> None:
        if self.total_symbols == 0:
            return

        vocabulary_size = max(1, len(self.entries))

        for symbol, entry in self.entries.items():
            entry.intrinsic.density = self.symbol_counts[symbol] / self.total_symbols

            out_degree = len(self.successors[symbol])
            max_possible = max(1, vocabulary_size - 1)

            entry.intrinsic.relation = out_degree / max_possible
            entry.intrinsic.volatility = out_degree / max_possible

            if symbol not in self.transition_probs:
                entry.intrinsic.coherence = 1.0
            else:
                probabilities = list(self.transition_probs[symbol].values())
                if len(probabilities) <= 1:
                    entry.intrinsic.coherence = 1.0
                else:
                    entropy = -sum(p * math.log(p + 1e-12) for p in probabilities)
                    max_entropy = math.log(len(probabilities))
                    entry.intrinsic.coherence = max(0.0, 1.0 - entropy / max_entropy)

        depth_sums = defaultdict(float)
        depth_counts = defaultdict(int)

        for sequence in self.sequences:
            for position, symbol in enumerate(sequence):
                depth_sums[symbol] += position
                depth_counts[symbol] += 1

        max_length = max((len(sequence) for sequence in self.sequences), default=1)

        for symbol, entry in self.entries.items():
            if depth_counts[symbol]:
                average_position = depth_sums[symbol] / depth_counts[symbol]
                entry.intrinsic.depth = average_position / max_length

    def _assign_beams_and_states(self) -> None:
        beam_denominator = max(1, self.max_beams - 1)
        state_denominator = max(1, self.max_states - 1)

        for entry in self.entries.values():
            relation = entry.intrinsic.relation
            coherence = entry.intrinsic.coherence

            beam_score = min(1.0, (relation + coherence) / 2.0)
            beam_id = int(beam_score * beam_denominator)
            entry.beam.beam_id = max(0, min(self.max_beams - 1, beam_id))
            entry.beam.affinity = beam_score

            volatility = entry.intrinsic.volatility
            depth = entry.intrinsic.depth

            state_score = min(1.0, (volatility + depth) / 2.0)
            state_id = int(state_score * state_denominator)
            entry.markov.state_id = max(0, min(self.max_states - 1, state_id))

            entry.weight = 0.5 * entry.intrinsic.density + 0.5 * entry.intrinsic.coherence

    def vocabulary(self, include_special_tokens: bool = False) -> List[str]:
        words = list(self.word_to_index.keys())
        if include_special_tokens:
            return words
        return [w for w in words if w not in {self.START_TOKEN, self.END_TOKEN}]


# ===============================================================
# MINIMAL OPTICAL EMULATOR  (from the optical GPT script)
# ===============================================================

class SimpleOpticalEmulator:
    """
    Minimal optical emulator:
      - Takes a probability vector p (length K).
      - Returns m = p + noise, renormalized.
    """

    def __init__(self, dim: int, noise_std: float = 0.02, seed: int = 2026):
        self.dim = int(dim)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def measure(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 1 and len(p) == self.dim:
            m = p + self.rng.normal(0.0, self.noise_std, size=p.shape)
        else:
            p_flat = np.reshape(p, -1)[: self.dim]
            if len(p_flat) < self.dim:
                p_flat = np.pad(p_flat, (0, self.dim - len(p_flat)))
            m = p_flat + self.rng.normal(0.0, self.noise_std, size=(self.dim,))
        m = np.maximum(m, 0.0)
        s = m.sum()
        if s <= 0:
            return np.ones(self.dim, dtype=np.float64) / self.dim
        return m / s


class OpticalProbabilitySampler:
    """
    Optical sampler using the minimal emulator. The optical measurement
    is the ONLY source of randomness used to pick a token.
    """

    def __init__(self, dim: int, noise_std: float = 0.02, seed: int = 2026):
        self.dim = int(dim)
        self.emulator = SimpleOpticalEmulator(dim, noise_std, seed)
        self.calibration_matrix: Optional[np.ndarray] = None

    def _normalise(self, probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.maximum(p, 0.0)
        total = p.sum()
        if total <= 0:
            raise ValueError("Probability vector has zero mass")
        return p / total

    def measure_distribution(self, probabilities):
        p = self._normalise(probabilities)
        m = self.emulator.measure(p)
        if self.calibration_matrix is not None and self.calibration_matrix.shape == (len(m), len(m)):
            p_est = np.maximum(self.calibration_matrix @ m, 0.0)
            total = p_est.sum()
            if total > 0:
                return p_est / total
        return m


# ===============================================================
# OPTICAL TRIGRAM GENERATOR  (bridges the two files)
# ===============================================================

class OpticalSymbolGenerator:
    """
    Generates word sequences from a SparseSymbolManifest, but instead of
    drawing the next word directly from the trigram distribution, the
    distribution is treated exactly like GPT logits in the optical
    script: it's restricted to its top-k support, sent to the optical
    emulator, and the *optical measurement* chooses the next symbol.
    """

    def __init__(
        self,
        manifest: SparseSymbolManifest,
        optical_sampler: OpticalProbabilitySampler,
        temperature: float = 1.0,
        optical_topk: int = 8,
        seed: int = 2026,
    ):
        self.manifest = manifest
        self.optical_sampler = optical_sampler
        self.temperature = max(float(temperature), 1e-6)
        self.optical_topk = int(optical_topk)
        self.rng = np.random.default_rng(seed)

    def _context_distribution(
        self, previous_idx: SymbolIndex, current_idx: SymbolIndex
    ) -> Optional[Tuple[List[SymbolIndex], np.ndarray]]:
        probs_dict = self.manifest.trigram_probs.get((previous_idx, current_idx))
        if not probs_dict:
            return None

        symbols = list(probs_dict.keys())
        probs = np.array(list(probs_dict.values()), dtype=np.float64)

        # temperature-adjust the "digital" distribution, same as the GPT script's softmax/temperature step
        adjusted = probs ** (1.0 / self.temperature)
        total = adjusted.sum()
        if total <= 0:
            return None
        adjusted /= total

        return symbols, adjusted

    def _optical_choose(self, symbols: List[SymbolIndex], probs: np.ndarray) -> SymbolIndex:
        k = min(self.optical_topk, len(symbols))
        top_local = np.argsort(probs)[-k:]
        candidate_symbols = [symbols[i] for i in top_local]
        candidate_probs = probs[top_local]
        candidate_probs = np.maximum(candidate_probs, 0.0)
        candidate_probs /= candidate_probs.sum()

        # pad to the optical bench's fixed dimensionality so the emulator
        # takes its exact-length code path, then measure and restrict
        # back down to the k real candidates.
        padded = np.zeros(self.optical_sampler.dim, dtype=np.float64)
        padded[:k] = candidate_probs

        measured = self.optical_sampler.measure_distribution(padded)
        measured_valid = measured[:k]
        s = measured_valid.sum()

        if s <= 0:
            measured_valid = candidate_probs
        else:
            measured_valid = measured_valid / s

        local_choice = int(self.rng.choice(k, p=measured_valid))
        return candidate_symbols[local_choice]

    def generate(self, start_word: Optional[str] = None, max_len: int = 30) -> List[str]:
        manifest = self.manifest
        start_token_idx = manifest.word_to_index[manifest.START_TOKEN]

        previous_idx = start_token_idx
        current_idx = start_token_idx
        generated: List[SymbolIndex] = []

        if start_word is not None:
            if start_word not in manifest.word_to_index:
                raise ValueError(f"Start word '{start_word}' is not in the learned vocabulary.")
            start_symbol = manifest.word_to_index[start_word]
            generated.append(start_symbol)
            current_idx = start_symbol

            contexts = [c for c in manifest.trigram_probs if c[1] == start_symbol]
            previous_idx = contexts[0][0] if contexts else start_token_idx

        for _ in range(max_len):
            distribution = self._context_distribution(previous_idx, current_idx)
            if distribution is None:
                break

            symbols, probs = distribution
            next_symbol = self._optical_choose(symbols, probs)
            next_word = manifest.index_to_word[next_symbol]

            if next_word == manifest.END_TOKEN:
                break
            if next_word != manifest.START_TOKEN:
                generated.append(next_symbol)

            previous_idx, current_idx = current_idx, next_symbol

        return [manifest.index_to_word[i] for i in generated]


# ===============================================================
# MAIN
# ===============================================================

@dataclass
class TopicCluster:
    beam_id: int
    affinity: float
    words: List[str]  # all candidate words in this cluster, highest weight first


def get_associated_words(
    manifest: SparseSymbolManifest,
    seed_words: List[str],
    depth: int = 1,
    breadth: int = 10,
) -> Set[str]:
    """
    Expand a set of seed words (e.g. the words literally typed in a
    prompt) out into the dataset's own association graph, rather than
    staying confined to the seeds themselves.

    Association here means the bigram transition graph learned over the
    WHOLE corpus (manifest.transition_probs): for each word in the
    current frontier, follow its top `breadth` most probable successors
    — the words that actually followed it somewhere in the training
    data — and add them to the pool. Repeating this `depth` times walks
    `depth` hops out from the prompt into genuinely dataset-derived
    territory.

    Returns the seed words plus everything reached by the walk.
    """
    associated: Set[str] = set()
    frontier: Set[str] = set()

    for word in seed_words:
        if word in manifest.word_to_index:
            associated.add(word)
            frontier.add(word)

    for _ in range(max(0, depth)):
        next_frontier: Set[str] = set()

        for word in frontier:
            idx = manifest.word_to_index[word]
            successor_probs = manifest.transition_probs.get(idx, {})

            top_successors = sorted(
                successor_probs.items(), key=lambda kv: kv[1], reverse=True
            )[:breadth]

            for succ_idx, _ in top_successors:
                succ_word = manifest.index_to_word[succ_idx]
                if succ_word in (manifest.START_TOKEN, manifest.END_TOKEN):
                    continue
                next_frontier.add(succ_word)

        associated |= next_frontier
        frontier = next_frontier

    return associated


def get_topk_clusters(
    manifest: SparseSymbolManifest, candidate_words: List[str], topk: int = 10
) -> List[TopicCluster]:
    """
    Group `candidate_words` (already deduped, already filtered to the
    learned vocabulary) by their whole-dataset beam_id — the clustering
    computed over the entire training corpus during ingest_dataset_words()
    — then rank the resulting clusters by beam affinity and return the
    top `topk` clusters.

    Unlike a plain top-k word list, this keeps every candidate word that
    fell into each top cluster (sorted by weight), so each returned
    cluster can be explored: every member is a legitimate start word for
    that topic, not just a single representative.
    """
    cluster_members: Dict[int, List[Tuple[float, str]]] = defaultdict(list)

    for word in candidate_words:
        if word not in manifest.word_to_index:
            continue
        idx = manifest.word_to_index[word]
        entry = manifest.entries[idx]
        cluster_members[entry.beam.beam_id].append((entry.weight, word))

    clusters: List[TopicCluster] = []

    for beam_id, members in cluster_members.items():
        members.sort(reverse=True)  # highest weight first within the cluster
        words_sorted = [word for _, word in members]

        _, top_word = members[0]
        affinity = manifest.entries[manifest.word_to_index[top_word]].beam.affinity

        clusters.append(TopicCluster(beam_id=beam_id, affinity=affinity, words=words_sorted))

    clusters.sort(key=lambda c: c.affinity, reverse=True)  # most central cluster first

    return clusters[:topk]


def get_topk_prompt_clusters(
    manifest: SparseSymbolManifest,
    prompt: str,
    topk: int = 10,
    assoc_depth: int = 1,
    assoc_breadth: int = 10,
) -> List[TopicCluster]:
    """
    Tokenize the prompt, then walk out `assoc_depth` hops through the
    dataset's own transition graph (get_associated_words) so the
    candidate pool isn't limited to words literally typed in the prompt
    — it also includes whatever the corpus itself associates with them.
    That expanded, dataset-grounded pool is then grouped into
    whole-dataset topic clusters via get_topk_clusters().
    """
    seen: Set[str] = set()
    seed_words: List[str] = []

    for word in SparseSymbolManifest.tokenize(prompt):
        if word in seen:
            continue
        seen.add(word)
        seed_words.append(word)

    candidate_words = get_associated_words(
        manifest, seed_words, depth=assoc_depth, breadth=assoc_breadth
    )

    return get_topk_clusters(manifest, list(candidate_words), topk=topk)




def _spread_select(words: List[str], count: int) -> List[str]:
    """
    Pick `count` words spread evenly across `words` (which is assumed
    sorted highest-weight first), rather than just taking the first
    `count`. This guarantees the selection covers the full range from
    the most common member down to the rarest one, instead of clustering
    around a handful of dominant/common words.
    """
    n = len(words)
    if count <= 0 or n == 0:
        return []
    if count >= n:
        return list(words)
    if count == 1:
        return [words[0]]

    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]

    seen_idx: Set[int] = set()
    selected: List[str] = []
    for idx in indices:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        selected.append(words[idx])

    return selected


def explore_clusters(
    generator: "OpticalSymbolGenerator",
    clusters: List[TopicCluster],
    max_len: int,
    lines_per_cluster: int = 3,
) -> List[Tuple[TopicCluster, List[Tuple[str, List[str]]]]]:
    """
    Step 2: explore each top-k cluster by generating one optical-sampled
    line per selected member word. Member words are spread across the
    cluster's full weight range (common to rare) via _spread_select, so
    exploration surfaces new/less-common words instead of repeating the
    same few high-weight ones every time. Returns, per cluster, a list of
    (start_word, generated_words) pairs.
    """
    exploration: List[Tuple[TopicCluster, List[Tuple[str, List[str]]]]] = []

    for cluster in clusters:
        words_to_try = _spread_select(cluster.words, lines_per_cluster)
        lines = generate_lines_from_words(generator, words_to_try, max_len=max_len)
        exploration.append((cluster, lines))

    return exploration


def generate_lines_from_words(
    generator: "OpticalSymbolGenerator", start_words: List[str], max_len: int
) -> List[Tuple[str, List[str]]]:
    """
    Step 2: run the optical-bench generator once per start word, producing
    one generated line per word. Returns (start_word, generated_words) pairs
    in the same order as start_words.
    """
    lines: List[Tuple[str, List[str]]] = []

    for start_word in start_words:
        try:
            words = generator.generate(start_word=start_word, max_len=max_len)
        except ValueError:
            words = []
        lines.append((start_word, words))

    return lines


def get_topk_vocabulary_clusters(
    manifest: SparseSymbolManifest, topk: int = 10
) -> List[TopicCluster]:
    """
    Fallback for when no --prompt is given: treat the whole learned
    vocabulary (minus <START>/<END>) as the candidate pool and cluster it
    via get_topk_clusters(), so the same exploration logic applies.
    """
    candidate_words = [
        word
        for word in manifest.word_to_index
        if word not in (manifest.START_TOKEN, manifest.END_TOKEN)
    ]
    return get_topk_clusters(manifest, candidate_words, topk=topk)


def build_manifest_from_file(path: str, max_symbols: int) -> SparseSymbolManifest:
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    raw_sequences = re.split(r"[.!?]+", text)

    dataset = []
    for sentence in raw_sequences:
        words = SparseSymbolManifest.tokenize(sentence)
        if words:
            dataset.append(words)

    manifest = SparseSymbolManifest(max_symbols=max_symbols, max_beams=8, max_states=8)
    manifest.ingest_dataset_words(dataset)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Trigram symbol manifest generation sampled through an optical bench."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to a text corpus file.")
   
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument(
        "--assoc-depth",
        type=int,
        default=1,
        help="How many hops to walk from the prompt's words through the "
             "dataset's own transition graph before clustering, so "
             "exploration isn't limited to words literally in the prompt.",
    )
    parser.add_argument(
        "--assoc-breadth",
        type=int,
        default=10,
        help="How many top successors per word to follow at each "
             "association hop.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top-ranked topic clusters (from --prompt words, or "
             "from the whole vocabulary with no prompt) to explore.",
    )
    parser.add_argument(
        "--lines-per-cluster",
        type=int,
        default=11,
        help="Number of member words per top-k cluster to generate a line for.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--optical-topk", type=int, default=8)
    parser.add_argument("--optical-noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-symbols", type=int, default=640000)

    args = parser.parse_args()

    print("Building trigram symbol manifest from:", args.file)
    manifest = build_manifest_from_file(args.file, args.max_symbols)
    print(f"Vocabulary size: {len(manifest.vocabulary())}")
    print(f"Trigram contexts learned: {len(manifest.trigram_probs)}")

    sampler = OpticalProbabilitySampler(
        dim=args.optical_topk,
        noise_std=args.optical_noise,
        seed=args.seed,
    )

    generator = OpticalSymbolGenerator(
        manifest=manifest,
        optical_sampler=sampler,
        temperature=args.temperature,
        optical_topk=args.optical_topk,
        seed=args.seed,
    )
    while True:
        # --- Step 1: get the top-k topic clusters ---------------------------
        clusters = get_topk_prompt_clusters(
            manifest,
            input("USER: "),
            topk=args.topk,
            assoc_depth=args.assoc_depth,
            assoc_breadth=args.assoc_breadth,
        )
        print(
            f"Top-{args.topk} clusters found after {args.assoc_depth}-hop "
            f"association from prompt words: {len(clusters)}"
        )
       
        if not clusters:
            print("No usable topic clusters found (none of the words are in the learned vocabulary).")
            return

        # --- Step 2: explore each top-k cluster ------------------------------
        print()
        print("=" * 70)
        print("OPTICAL-BENCH CLUSTER EXPLORATION")
        print("=" * 70)

        print()

        exploration = explore_clusters(
            generator, clusters, max_len=args.max_len, lines_per_cluster=args.lines_per_cluster
        )

        all_lines: List[List[str]] = []
        for cluster_num, (cluster, lines) in enumerate(exploration, start=1):
            for start_word, words in lines:
                if not words:
                    continue
                all_lines.append(words)

        # print in reverse order of appearance: last-generated line first
        for words in reversed(all_lines):
            print(" ".join(words), end=" ")

if __name__ == "__main__":
    main()
