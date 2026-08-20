import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple


SymbolIndex = int


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
    """
    Sparse trigram word manifest.

    The model learns:

        P(word_n | word_(n-2), word_(n-1))

    from the supplied dataset.
    """

    START_TOKEN = "<START>"
    END_TOKEN = "<END>"

    def __init__(
        self,
        max_symbols: int = 256,
        max_beams: int = 16,
        max_states: int = 32,
    ):
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

        # Bigram statistics used by manifest properties
        self.transition_counts: Dict[
            SymbolIndex, Dict[SymbolIndex, int]
        ] = defaultdict(lambda: defaultdict(int))

        self.transition_probs: Dict[
            SymbolIndex, Dict[SymbolIndex, float]
        ] = {}

        # Trigram counts:
        # (previous_word, current_word) -> next_word -> count
        self.trigram_counts: Dict[
            Tuple[SymbolIndex, SymbolIndex],
            Dict[SymbolIndex, int],
        ] = defaultdict(lambda: defaultdict(int))

        # Trigram probabilities:
        # (previous_word, current_word) -> next_word -> probability
        self.trigram_probs: Dict[
            Tuple[SymbolIndex, SymbolIndex],
            Dict[SymbolIndex, float],
        ] = {}

    # -------------------------------------------------
    # Registration and ingestion
    # -------------------------------------------------

    def _ensure_entry(self, idx: SymbolIndex) -> None:
        if idx not in self.entries:
            self.entries[idx] = SymbolManifestEntry()

    def _register_word(self, word: str) -> SymbolIndex:
        if word not in self.word_to_index:
            if self.next_index >= self.max_symbols:
                raise ValueError(
                    f"Vocabulary exceeds max_symbols={self.max_symbols}"
                )

            idx = self.next_index
            self.next_index += 1

            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
            self._ensure_entry(idx)

            return idx

        return self.word_to_index[word]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Basic tokenizer that keeps words and common punctuation.
        """
        return text.split()

    def add_sequence_words(self, words: List[str]) -> List[SymbolIndex]:
        """
        Add one word sequence.

        Start and end tokens are added internally so that trigram generation
        can learn sentence beginnings and endings.
        """
        words = [word for word in words if word.strip()]

        if not words:
            return []

        padded_words = [
            self.START_TOKEN,
            self.START_TOKEN,
            *words,
            self.END_TOKEN,
        ]

        seq = [self._register_word(word) for word in padded_words]
        self.sequences.append(seq)

        for symbol in seq:
            self.symbol_counts[symbol] += 1
            self.total_symbols += 1

        # Bigram statistics
        for current, nxt in zip(seq, seq[1:]):
            self.transition_counts[current][nxt] += 1
            self.successors[current].add(nxt)

        # Trigram statistics
        for i in range(len(seq) - 2):
            previous_word = seq[i]
            current_word = seq[i + 1]
            next_word = seq[i + 2]

            self.trigram_counts[
                (previous_word, current_word)
            ][next_word] += 1

        return seq

    def ingest_dataset_words(self, sequences: List[List[str]]) -> None:
        """
        Ingest a dataset containing tokenized word sequences.
        """
        for words in sequences:
            self.add_sequence_words(words)

        self._finalize_transitions()
        self._finalize_trigrams()
        self._update_intrinsics()
        self._assign_beams_and_states()

    # -------------------------------------------------
    # Model finalization
    # -------------------------------------------------

    def _finalize_transitions(self) -> None:
        self.transition_probs = {}

        for current, next_words in self.transition_counts.items():
            total = sum(next_words.values())

            if total:
                self.transition_probs[current] = {
                    nxt: count / total
                    for nxt, count in next_words.items()
                }

    def _finalize_trigrams(self) -> None:
        self.trigram_probs = {}

        for context, next_words in self.trigram_counts.items():
            total = sum(next_words.values())

            if total:
                self.trigram_probs[context] = {
                    nxt: count / total
                    for nxt, count in next_words.items()
                }

    def _update_intrinsics(self) -> None:
        if self.total_symbols == 0:
            return

        vocabulary_size = max(1, len(self.entries))

        for symbol, entry in self.entries.items():
            entry.intrinsic.density = (
                self.symbol_counts[symbol] / self.total_symbols
            )

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
                    entropy = -sum(
                        probability * math.log(probability + 1e-12)
                        for probability in probabilities
                    )

                    max_entropy = math.log(len(probabilities))
                    entry.intrinsic.coherence = max(
                        0.0,
                        1.0 - entropy / max_entropy,
                    )

        depth_sums = defaultdict(float)
        depth_counts = defaultdict(int)

        for sequence in self.sequences:
            for position, symbol in enumerate(sequence):
                depth_sums[symbol] += position
                depth_counts[symbol] += 1

        max_length = max(
            (len(sequence) for sequence in self.sequences),
            default=1,
        )

        for symbol, entry in self.entries.items():
            if depth_counts[symbol]:
                average_position = (
                    depth_sums[symbol] / depth_counts[symbol]
                )
                entry.intrinsic.depth = average_position / max_length

    def _assign_beams_and_states(self) -> None:
        beam_denominator = max(1, self.max_beams - 1)
        state_denominator = max(1, self.max_states - 1)

        for entry in self.entries.values():
            relation = entry.intrinsic.relation
            coherence = entry.intrinsic.coherence

            beam_score = min(1.0, (relation + coherence) / 2.0)
            beam_id = int(beam_score * beam_denominator)

            entry.beam.beam_id = max(
                0,
                min(self.max_beams - 1, beam_id),
            )

            entry.beam.affinity = beam_score

            volatility = entry.intrinsic.volatility
            depth = entry.intrinsic.depth

            state_score = min(1.0, (volatility + depth) / 2.0)
            state_id = int(state_score * state_denominator)

            entry.markov.state_id = max(
                0,
                min(self.max_states - 1, state_id),
            )

            entry.weight = (
                0.5 * entry.intrinsic.density
                + 0.5 * entry.intrinsic.coherence
            )

    # -------------------------------------------------
    # Trigram generation
    # -------------------------------------------------

    def sample_next_trigram(
        self,
        previous_idx: SymbolIndex,
        current_idx: SymbolIndex,
        temperature: float = 1.0,
    ) -> Optional[SymbolIndex]:
        """
        Sample:

            next_word ~ P(next_word | previous_word, current_word)
        """
        context = (previous_idx, current_idx)
        probabilities = self.trigram_probs.get(context)

        if not probabilities:
            return None

        temperature = max(float(temperature), 1e-6)

        symbols = list(probabilities.keys())

        adjusted = [
            probability ** (1.0 / temperature)
            for probability in probabilities.values()
        ]

        total = sum(adjusted)

        if total <= 0.0:
            return None

        adjusted = [value / total for value in adjusted]

        return random.choices(symbols, weights=adjusted, k=1)[0]

    def _choose_start_context(
        self,
    ) -> Optional[Tuple[SymbolIndex, SymbolIndex]]:
        start_idx = self.word_to_index.get(self.START_TOKEN)

        if start_idx is None:
            return None

        contexts = [
            context
            for context in self.trigram_probs
            if context[0] == start_idx and context[1] == start_idx
        ]

        if contexts:
            return contexts[0]

        return None

    def generate_sequence_indices(
        self,
        start: Optional[SymbolIndex] = None,
        max_len: int = 20,
        temperature: float = 1.0,
    ) -> List[SymbolIndex]:
        """
        Generate a sequence using the trigram model.
        """
        if max_len <= 0:
            return []

        start_idx = self.word_to_index[self.START_TOKEN]

        # Normal sentence generation starts with <START>, <START>.
        previous_idx = start_idx
        current_idx = start_idx

        generated = []

        if start is not None:
            generated.append(start)
            current_idx = start

            # Find a compatible context for the requested starting word.
            contexts = [
                context
                for context in self.trigram_probs
                if context[1] == start
            ]

            if contexts:
                previous_idx = contexts[0][0]
            else:
                previous_idx = start_idx

        for _ in range(max_len):
            next_idx = self.sample_next_trigram(
                previous_idx,
                current_idx,
                temperature=temperature,
            )

            if next_idx is None:
                break

            next_word = self.index_to_word[next_idx]

            if next_word == self.END_TOKEN:
                break

            if next_word != self.START_TOKEN:
                generated.append(next_idx)

            previous_idx, current_idx = current_idx, next_idx

        return generated

    def generate_sequence_words(
        self,
        start_word: Optional[str] = None,
        max_len: int = 20,
        temperature: float = 1.0,
    ) -> List[str]:
        if start_word is not None:
            if start_word not in self.word_to_index:
                raise ValueError(
                    f"Start word '{start_word}' is not in the learned vocabulary."
                )

            start_idx = self.word_to_index[start_word]
        else:
            start_idx = None

        index_sequence = self.generate_sequence_indices(
            start=start_idx,
            max_len=max_len,
            temperature=temperature,
        )

        return [
            self.index_to_word[index]
            for index in index_sequence
            if self.index_to_word[index] not in {
                self.START_TOKEN,
                self.END_TOKEN,
            }
        ]

    # -------------------------------------------------
    # Inspection
    # -------------------------------------------------

    def vocabulary(self, include_special_tokens: bool = False) -> List[str]:
        words = list(self.word_to_index.keys())

        if include_special_tokens:
            return words

        return [
            word
            for word in words
            if word not in {self.START_TOKEN, self.END_TOKEN}
        ]

    def trigram_table(self) -> List[Dict]:
        """
        Return learned trigrams as readable words.
        """
        rows = []

        for context, next_words in sorted(self.trigram_counts.items()):
            previous_idx, current_idx = context
            total = sum(next_words.values())

            for next_idx, count in sorted(next_words.items()):
                rows.append({
                    "word_1": self.index_to_word[previous_idx],
                    "word_2": self.index_to_word[current_idx],
                    "word_3": self.index_to_word[next_idx],
                    "count": count,
                    "probability": count / total,
                })

        return rows

    def manifest_table(self) -> List[Dict]:
        rows = []

        for idx in sorted(self.entries):
            entry = self.entries[idx]

            rows.append({
                "index": idx,
                "word": self.index_to_word.get(idx, "<unknown>"),
                "density": entry.intrinsic.density,
                "relation": entry.intrinsic.relation,
                "coherence": entry.intrinsic.coherence,
                "volatility": entry.intrinsic.volatility,
                "depth": entry.intrinsic.depth,
                "beam_id": entry.beam.beam_id,
                "beam_affinity": entry.beam.affinity,
                "state_id": entry.markov.state_id,
                "weight": entry.weight,
                "successors": [
                    self.index_to_word[index]
                    for index in sorted(self.successors.get(idx, []))
                ],
            })

        return rows


# -----------------------------
# Example usage / self-test
# -----------------------------

if __name__ == "__main__":
    with open(input("Filename: "), "r", encoding="utf-8") as file:
        text = file.read()

    # Split the file into sentence-like sequences.
    raw_sequences = re.split(r"[.!?]+", text)

    # Convert each sentence into a list of words.
    dataset = []

    for sentence in raw_sequences:
        words = SparseSymbolManifest.tokenize(sentence)

        if words:
            dataset.append(words)

    manifest = SparseSymbolManifest(
        max_symbols=64000,
        max_beams=8,
        max_states=8,
    )

    manifest.ingest_dataset_words(dataset)

    print("Vocabulary: symbol -> actual word")
    print("-" * 60)

    for symbol_index, word in sorted(manifest.index_to_word.items()):
        print(f"{symbol_index:6d} -> {word}")

    print("\nLearned trigrams: symbol sequence -> actual words")
    print("-" * 90)

    for row in manifest.trigram_table():
        word_1 = row["word_1"]
        word_2 = row["word_2"]
        word_3 = row["word_3"]

        symbol_1 = manifest.word_to_index[word_1]
        symbol_2 = manifest.word_to_index[word_2]
        symbol_3 = manifest.word_to_index[word_3]

        print(
            f"({symbol_1}, {symbol_2}, {symbol_3}) "
            f"-> ({word_1!r}, {word_2!r}, {word_3!r}) "
            f"| count={row['count']} "
            f"| probability={row['probability']:.6f}"
        )

    print("\nManifest table: symbol -> actual word")
    print("-" * 90)

    for row in manifest.manifest_table():
        print(
            f"symbol={row['index']:6d} "
            f"word={row['word']!r} "
            f"density={row['density']:.6f} "
            f"relation={row['relation']:.6f} "
            f"coherence={row['coherence']:.6f} "
            f"volatility={row['volatility']:.6f} "
            f"depth={row['depth']:.6f} "
            f"beam={row['beam_id']} "
            f"state={row['state_id']} "
            f"successors={row['successors']}"
        )

    print("\nGenerated trigram word sequences")
    print("-" * 90)

    for _ in range(5):
        sequence = manifest.generate_sequence_words(
            max_len=10000,
            temperature=1.0,
        )

        print(" ".join(sequence))
