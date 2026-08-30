from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional


# ============================================================
# SPARSE SYMBOL MANIFEST
# SYMBOLIC CANDIDATE ANALYSIS + CHINESE ROOM REBINDING
#
# Processing order:
#
#   singlekb.txt
#        |
#        v
#   symbolic model
#        |
#        v
#      USER PROMPT
#        |
#        v
#   1. SYMBOLIC CANDIDATE ANALYSIS
#        |
#        +--> potential prompts
#        +--> candidate references
#        +--> overlap scores
#        +--> vector scores
#        +--> ranking
#        |
#        v
#   2. FIRST GENERATION
#        |
#        v
#   3. POST-GENERATION REBINDING
#        |
#        v
#   4. SECOND SYMBOLIC PASS
#
# No command-line arguments.
# No external ground_truth.jsonl.
#
# "Candidate ground truth" means a corpus-derived reference.
# It is NOT independently verified ground truth.
#
# The "analysis steps" are explicit symbolic operations,
# not hidden chain-of-thought.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

CORPUS_PATH = input("Filename: ")
MODEL_PATH = "model.json"
RECORD_PATH = "chinese_room_record.json"

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 20

MIN_COUNT = 1
INFLUENCE_TAU = 0.5

CURVE_NAME = "sigmoid"
CURVE_K = 8.0
CURVE_MIDPOINT = 0.5


# ============================================================
# SYMBOLIC CANDIDATE ANALYSIS
# ============================================================

CANDIDATE_LIMIT = 5

LEXICAL_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55

ACCEPTANCE_THRESHOLD = 0.35
AMBIGUITY_MARGIN = 0.10


# ============================================================
# SECOND PASS
# ============================================================

SECOND_PASS_GENERATION = True

SECOND_PASS_MAX_NEW_TOKENS = 50
SECOND_PASS_TEMPERATURE = 0.8
SECOND_PASS_TOP_K = 20


# ============================================================
# TRAINING
# ============================================================

TRAIN_MODEL = True
RANDOM_SEED = 2026

random.seed(RANDOM_SEED)


# ============================================================
# TOKENIZATION
# ============================================================

TOKEN_RE = re.compile(
    r"[A-Za-z0-9_']+|[.,!?;:()\[\]{}\-]"
)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def safe_log(
    value: float,
    floor: float = 1e-12,
) -> float:
    return math.log(max(value, floor))


# ============================================================
# VECTOR FUNCTIONS
# ============================================================

def cosine_similarity(
    a: Dict[str, float],
    b: Dict[str, float],
) -> float:

    if not a or not b:
        return 0.0

    common = set(a).intersection(b)

    dot = sum(
        a[k] * b[k]
        for k in common
    )

    norm_a = math.sqrt(
        sum(
            v * v
            for v in a.values()
        )
    )

    norm_b = math.sqrt(
        sum(
            v * v
            for v in b.values()
        )
    )

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def bag_of_words(
    tokens: Iterable[str],
) -> Counter:

    ignored = {
        "<bos>",
        "<eos>",
        "<unk>",
    }

    return Counter(
        token
        for token in tokens
        if token not in ignored
    )


def lexical_overlap(
    a: Iterable[str],
    b: Iterable[str],
) -> float:

    ignored = {
        "<bos>",
        "<eos>",
        "<unk>",
    }

    sa = set(a)
    sb = set(b)

    sa -= ignored
    sb -= ignored

    if not sa or not sb:
        return 0.0

    intersection = len(
        sa.intersection(sb)
    )

    union = len(
        sa.union(sb)
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# CORPUS REFERENCE
# ============================================================

@dataclass
class CorpusReference:

    sentence: str

    tokens: List[str]

    vector: Dict[str, float]

    frequency: int = 1

    def to_dict(self) -> dict:

        return {
            "sentence": self.sentence,
            "tokens": self.tokens,
            "vector": self.vector,
            "frequency": self.frequency,
        }


# ============================================================
# SYMBOLIC CANDIDATE
# ============================================================

@dataclass
class RebindingCandidate:

    potential_prompt: str

    candidate_ground_truth: str

    symbolic_overlap: float

    vector_similarity: float

    frequency: int

    score: float

    rank: int = 0

    def to_dict(self) -> dict:

        return {
            "rank": self.rank,
            "potential_prompt": self.potential_prompt,
            "candidate_ground_truth":
                self.candidate_ground_truth,
            "symbolic_overlap":
                self.symbolic_overlap,
            "vector_similarity":
                self.vector_similarity,
            "frequency":
                self.frequency,
            "score":
                self.score,
        }


# ============================================================
# REBINDING RESULT
# ============================================================

@dataclass
class RebindingResult:

    generated_text: str

    potential_prompts: List[dict]

    rebound_prompt: Optional[str]

    rebound_ground_truth: Optional[str]

    grounding_score: float

    ambiguity: float

    symbolic_overlap: float

    semantic_overlap: float

    accepted: bool

    reason: str

    def to_dict(self) -> dict:

        return {
            "generated_text":
                self.generated_text,

            "potential_prompts":
                self.potential_prompts,

            "rebound_prompt":
                self.rebound_prompt,

            "rebound_ground_truth":
                self.rebound_ground_truth,

            "grounding_score":
                self.grounding_score,

            "ambiguity":
                self.ambiguity,

            "symbolic_overlap":
                self.symbolic_overlap,

            "semantic_overlap":
                self.semantic_overlap,

            "accepted":
                self.accepted,

            "reason":
                self.reason,
        }


# ============================================================
# SYMBOLIC CANDIDATE ANALYSIS
#
# This is deliberately explicit and observable.
# It does not expose hidden reasoning.
# ============================================================

@dataclass
class SymbolicCandidateAnalyzer:

    references: List[CorpusReference] = field(
        default_factory=list
    )

    lexical_weight: float = LEXICAL_WEIGHT

    vector_weight: float = VECTOR_WEIGHT

    acceptance_threshold: float = (
        ACCEPTANCE_THRESHOLD
    )

    ambiguity_margin: float = (
        AMBIGUITY_MARGIN
    )

    # --------------------------------------------------------
    # BUILD REFERENCE INDEX
    # --------------------------------------------------------

    def build_index(
        self,
        corpus_text: str,
    ) -> None:

        sentences = split_sentences(
            corpus_text
        )

        frequency = Counter(
            sentence.lower()
            for sentence in sentences
        )

        self.references = []

        for sentence in sentences:

            tokens = tokenize(
                sentence
            )

            if not tokens:
                continue

            bow = bag_of_words(
                tokens
            )

            vector = {
                token: float(count)
                for token, count
                in bow.items()
            }

            self.references.append(
                CorpusReference(
                    sentence=sentence,
                    tokens=tokens,
                    vector=vector,
                    frequency=frequency[
                        sentence.lower()
                    ],
                )
            )

    # --------------------------------------------------------
    # ANALYZE PROMPT
    #
    # These are symbolic candidate-generation steps:
    #
    #   Step 1: tokenize
    #   Step 2: construct bag of words
    #   Step 3: compare against corpus
    #   Step 4: calculate lexical overlap
    #   Step 5: calculate vector similarity
    #   Step 6: combine scores
    #   Step 7: rank candidates
    #
    # --------------------------------------------------------

    def analyze(
        self,
        prompt: str,
        limit: int = 5,
    ) -> List[RebindingCandidate]:

        prompt_tokens = tokenize(
            prompt
        )

        prompt_bow = bag_of_words(
            prompt_tokens
        )

        prompt_vector = {
            token: float(count)
            for token, count
            in prompt_bow.items()
        }

        candidates = []

        for reference in self.references:

            symbolic = lexical_overlap(
                prompt_tokens,
                reference.tokens,
            )

            vector_similarity = (
                cosine_similarity(
                    prompt_vector,
                    reference.vector,
                )
            )

            score = (
                self.lexical_weight
                * symbolic
                +
                self.vector_weight
                * vector_similarity
            )

            candidates.append(
                RebindingCandidate(
                    potential_prompt=
                        reference.sentence,

                    candidate_ground_truth=
                        reference.sentence,

                    symbolic_overlap=
                        symbolic,

                    vector_similarity=
                        vector_similarity,

                    frequency=
                        reference.frequency,

                    score=
                        score,
                )
            )

        candidates.sort(
            key=lambda candidate:
                candidate.score,
            reverse=True,
        )

        candidates = candidates[:limit]

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            candidate.rank = index

        return candidates

    # --------------------------------------------------------
    # REBIND GENERATED OUTPUT
    # --------------------------------------------------------

    def rebind(
        self,
        generated_text: str,
        candidate_limit: int = 5,
    ) -> RebindingResult:

        candidates = self.analyze(
            generated_text,
            candidate_limit,
        )

        candidate_dicts = [
            candidate.to_dict()
            for candidate in candidates
        ]

        if not candidates:

            return RebindingResult(
                generated_text=generated_text,
                potential_prompts=[],
                rebound_prompt=None,
                rebound_ground_truth=None,
                grounding_score=0.0,
                ambiguity=1.0,
                symbolic_overlap=0.0,
                semantic_overlap=0.0,
                accepted=False,
                reason=(
                    "The corpus contained "
                    "no candidate references."
                ),
            )

        best = candidates[0]

        best_score = best.score

        if len(candidates) > 1:
            second_score = candidates[1].score
        else:
            second_score = 0.0

        margin = (
            best_score
            - second_score
        )

        if best_score > 0:

            ambiguity = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        margin
                        / best_score
                    ),
                ),
            )

        else:
            ambiguity = 1.0

        accepted = (
            best_score
            >= self.acceptance_threshold
            and (
                len(candidates) == 1
                or margin
                >= self.ambiguity_margin
            )
        )

        if accepted:

            reason = (
                "Highest-scoring corpus "
                "reference accepted as a "
                "candidate symbolic rebinding."
            )

            rebound_prompt = (
                best.potential_prompt
            )

            rebound_ground_truth = (
                best.candidate_ground_truth
            )

        elif (
            best_score
            < self.acceptance_threshold
        ):

            reason = (
                "No candidate reached "
                "the grounding threshold."
            )

            rebound_prompt = None
            rebound_ground_truth = None

        else:

            reason = (
                "Top candidates were too close "
                "to establish an unambiguous "
                "symbolic rebinding."
            )

            rebound_prompt = None
            rebound_ground_truth = None

        return RebindingResult(
            generated_text=generated_text,

            potential_prompts=candidate_dicts,

            rebound_prompt=rebound_prompt,

            rebound_ground_truth=
                rebound_ground_truth,

            grounding_score=best_score,

            ambiguity=ambiguity,

            symbolic_overlap=
                best.symbolic_overlap,

            semantic_overlap=
                best.vector_similarity,

            accepted=accepted,

            reason=reason,
        )


# ============================================================
# SPARSE SYMBOL MANIFEST
# ============================================================

@dataclass
class SparseSymbolManifest:

    eos_token: str = "<eos>"

    unk_token: str = "<unk>"

    min_count: int = 1

    influence_tau: float = 0.5

    curve_name: str = "sigmoid"

    curve_k: float = 8.0

    curve_midpoint: float = 0.5

    unigram: Counter = field(
        default_factory=Counter
    )

    bigram: Dict[
        str,
        Counter
    ] = field(
        default_factory=lambda:
            defaultdict(Counter)
    )

    trigram: Dict[
        str,
        Counter
    ] = field(
        default_factory=lambda:
            defaultdict(Counter)
    )

    lexical_vectors: Dict[
        str,
        Dict[str, float]
    ] = field(
        default_factory=dict
    )

    influence_vectors: Dict[
        str,
        Dict[str, float]
    ] = field(
        default_factory=dict
    )

    vocabulary: List[str] = field(
        default_factory=list
    )

    finalized: bool = False

    # ========================================================
    # INGESTION
    # ========================================================

    def ingest_text(
        self,
        text: str,
    ) -> None:

        for sentence in split_sentences(
            text
        ):

            words = tokenize(
                sentence
            )

            if not words:
                continue

            sequence = (
                ["<bos>", "<bos>"]
                + words
                + [self.eos_token]
            )

            self.add_sequence_words(
                sequence
            )

    def add_sequence_words(
        self,
        sequence: List[str],
    ) -> None:

        if len(sequence) < 3:
            return

        for token in sequence:
            self.unigram[token] += 1

        for left, right in zip(
            sequence,
            sequence[1:],
        ):
            self.bigram[left][right] += 1

        for a, b, c in zip(
            sequence,
            sequence[1:],
            sequence[2:],
        ):

            self.trigram[
                f"{a}\t{b}"
            ][c] += 1

        self.finalized = False

    # ========================================================
    # VOCABULARY
    # ========================================================

    def register_vocabulary(
        self,
    ) -> None:

        self.vocabulary = sorted(
            token
            for token, count
            in self.unigram.items()
            if count >= self.min_count
        )

        if self.unk_token not in self.vocabulary:

            self.vocabulary.append(
                self.unk_token
            )

    # ========================================================
    # LEXICAL VECTORS
    # ========================================================

    def update_intrinsics(
        self,
    ) -> None:

        self.register_vocabulary()

        token_contexts = defaultdict(
            Counter
        )

        for context, counts in (
            self.bigram.items()
        ):

            for token, count in (
                counts.items()
            ):

                token_contexts[token][
                    context
                ] += count

        self.lexical_vectors = {}

        for token in self.vocabulary:

            counts = token_contexts.get(
                token,
                Counter(),
            )

            total = sum(
                counts.values()
            )

            if total == 0:
                total = 1

            self.lexical_vectors[
                token
            ] = {
                context:
                    count / total
                for context, count
                in counts.items()
            }

    # ========================================================
    # INFLUENCE VECTORS
    # ========================================================

    def build_influence_vectors(
        self,
    ) -> None:

        self.influence_vectors = {}

        for source in self.vocabulary:

            source_vector = (
                self.lexical_vectors.get(
                    source,
                    {},
                )
            )

            scores = {}

            for target in self.vocabulary:

                if source == target:
                    continue

                target_vector = (
                    self.lexical_vectors.get(
                        target,
                        {},
                    )
                )

                similarity = (
                    cosine_similarity(
                        source_vector,
                        target_vector,
                    )
                )

                if (
                    similarity
                    >= self.influence_tau
                ):

                    scores[target] = (
                        similarity
                    )

            self.influence_vectors[
                source
            ] = scores

    # ========================================================
    # FINALIZE
    # ========================================================

    def finalize(self) -> None:

        self.update_intrinsics()

        self.build_influence_vectors()

        self.finalized = True

    # ========================================================
    # BACKOFF
    # ========================================================

    def backoff_distributions(
        self,
        previous_token: str,
        previous_previous_token:
            str | None = None,
    ) -> Dict[str, float]:

        if previous_previous_token is not None:

            key = (
                f"{previous_previous_token}"
                f"\t{previous_token}"
            )

            counts = self.trigram.get(
                key
            )

            if counts:

                return self._normalize_counts(
                    counts
                )

        counts = self.bigram.get(
            previous_token
        )

        if counts:

            return self._normalize_counts(
                counts
            )

        return self._normalize_counts(
            self.unigram
        )

    @staticmethod
    def _normalize_counts(
        counts: Counter,
    ) -> Dict[str, float]:

        total = sum(
            counts.values()
        )

        if total == 0:
            return {}

        return {
            token:
                count / total
            for token, count
            in counts.items()
        }

    # ========================================================
    # PROMPT CONTEXT
    # ========================================================

    def resolve_prompt_context(
        self,
        prompt: str,
    ) -> Tuple[
        str,
        str | None,
    ]:

        tokens = tokenize(
            prompt
        )

        if not tokens:

            return (
                "<bos>",
                None,
            )

        previous = tokens[-1]

        previous_previous = (
            tokens[-2]
            if len(tokens) >= 2
            else None
        )

        return (
            previous,
            previous_previous,
        )

    # ========================================================
    # EOS CURVE
    # ========================================================

    def curve_weight(
        self,
        p_eos: float,
    ) -> float:

        p_eos = min(
            1.0,
            max(
                0.0,
                p_eos,
            ),
        )

        if self.curve_name == "linear":

            return 1.0 - p_eos

        if self.curve_name == "inverse":

            return 1.0 / (
                1.0 + p_eos
            )

        if self.curve_name == "sigmoid":

            z = (
                self.curve_k
                * (
                    p_eos
                    - self.curve_midpoint
                )
            )

            return 1.0 / (
                1.0
                + math.exp(z)
            )

        raise ValueError(
            f"Unknown curve: "
            f"{self.curve_name}"
        )

    # ========================================================
    # NEXT TOKEN
    # ========================================================

    def score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
    ) -> Dict[str, float]:

        if not self.finalized:
            self.finalize()

        previous, previous_previous = (
            self.resolve_prompt_context(
                prompt
            )
        )

        base = self.backoff_distributions(
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

        source = (
            self.lexical_vectors.get(
                previous,
                {},
            )
        )

        influences = (
            self.influence_vectors.get(
                previous,
                {},
            )
        )

        p_eos = base.get(
            self.eos_token,
            0.0,
        )

        curve = self.curve_weight(
            p_eos
        )

        scores = {}

        for token in candidates:

            similarity = (
                cosine_similarity(
                    source,
                    self.lexical_vectors.get(
                        token,
                        {},
                    ),
                )
            )

            influence = influences.get(
                token,
                0.0,
            )

            score = (
                safe_log(
                    base[token]
                )
                +
                curve
                * 0.35
                * similarity
                +
                curve
                * 0.65
                * influence
            )

            scores[token] = score

        return scores

    # ========================================================
    # PROBABILITIES
    # ========================================================

    def probabilities(
        self,
        prompt: str,
        temperature: float = 1.0,
        candidate_limit: int = 64,
    ) -> Dict[str, float]:

        scores = self.score_next_token(
            prompt,
            candidate_limit,
        )

        if not scores:
            return {}

        temperature = max(
            temperature,
            1e-5,
        )

        scaled = {
            token:
                score / temperature
            for token, score
            in scores.items()
        }

        maximum = max(
            scaled.values()
        )

        exponentials = {
            token:
                math.exp(
                    score - maximum
                )
            for token, score
            in scaled.items()
        }

        total = sum(
            exponentials.values()
        )

        if total == 0:
            return {}

        return {
            token:
                value / total
            for token, value
            in exponentials.items()
        }

    # ========================================================
    # SAMPLE
    # ========================================================

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> str:

        probabilities = self.probabilities(
            prompt,
            temperature,
            max(
                top_k,
                1,
            ),
        )

        if not probabilities:
            return self.eos_token

        items = sorted(
            probabilities.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )[:top_k]

        tokens = [
            token
            for token, _
            in items
        ]

        weights = [
            weight
            for _, weight
            in items
        ]

        return random.choices(
            tokens,
            weights=weights,
            k=1,
        )[0]

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> str:

        generated = tokenize(
            prompt
        )

        for _ in range(
            max_new_tokens
        ):

            current_prompt = " ".join(
                generated
            )

            token = self.sample_next(
                current_prompt,
                temperature,
                top_k,
            )

            if token == self.eos_token:
                break

            generated.append(
                token
            )

        return self.detokenize(
            generated
        )

    # ========================================================
    # DETOKENIZE
    # ========================================================

    @staticmethod
    def detokenize(
        tokens: List[str],
    ) -> str:

        text = " ".join(
            tokens
        )

        text = re.sub(
            r"\s+([.,!?;:)\]}])",
            r"\1",
            text,
        )

        text = re.sub(
            r"([(\[{])\s+",
            r"\1",
            text,
        )

        return text

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self) -> dict:

        return {
            "eos_token":
                self.eos_token,

            "unk_token":
                self.unk_token,

            "min_count":
                self.min_count,

            "influence_tau":
                self.influence_tau,

            "curve_name":
                self.curve_name,

            "curve_k":
                self.curve_k,

            "curve_midpoint":
                self.curve_midpoint,

            "unigram":
                dict(self.unigram),

            "bigram": {
                key: dict(value)
                for key, value
                in self.bigram.items()
            },

            "trigram": {
                key: dict(value)
                for key, value
                in self.trigram.items()
            },

            "lexical_vectors":
                self.lexical_vectors,

            "influence_vectors":
                self.influence_vectors,

            "vocabulary":
                self.vocabulary,

            "finalized":
                self.finalized,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "SparseSymbolManifest":

        model = cls(
            eos_token=data[
                "eos_token"
            ],

            unk_token=data[
                "unk_token"
            ],

            min_count=data[
                "min_count"
            ],

            influence_tau=data[
                "influence_tau"
            ],

            curve_name=data[
                "curve_name"
            ],

            curve_k=data[
                "curve_k"
            ],

            curve_midpoint=data[
                "curve_midpoint"
            ],
        )

        model.unigram = Counter(
            data["unigram"]
        )

        model.bigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value
                in data["bigram"].items()
            },
        )

        model.trigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value
                in data["trigram"].items()
            },
        )

        model.lexical_vectors = (
            data["lexical_vectors"]
        )

        model.influence_vectors = (
            data["influence_vectors"]
        )

        model.vocabulary = (
            data["vocabulary"]
        )

        model.finalized = (
            data["finalized"]
        )

        return model

    def save_json(
        self,
        path: str | Path,
    ) -> None:

        Path(path).write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "SparseSymbolManifest":

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8",
            )
        )

        return cls.from_dict(
            data
        )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    corpus_text: str,
) -> SparseSymbolManifest:

    model = SparseSymbolManifest(
        curve_name=CURVE_NAME,
        curve_k=CURVE_K,
        curve_midpoint=CURVE_MIDPOINT,
        min_count=MIN_COUNT,
        influence_tau=INFLUENCE_TAU,
    )

    model.ingest_text(
        corpus_text
    )

    model.finalize()

    model.save_json(
        MODEL_PATH
    )

    return model


# ============================================================
# SECOND SYMBOLIC PASS
# ============================================================

def second_pass(
    model: SparseSymbolManifest,
    result: RebindingResult,
) -> Optional[str]:

    if not result.accepted:
        return None

    if not result.rebound_ground_truth:
        return None

    reference = (
        result.rebound_ground_truth
    )

    return model.generate(
        reference,
        max_new_tokens=(
            SECOND_PASS_MAX_NEW_TOKENS
        ),
        temperature=(
            SECOND_PASS_TEMPERATURE
        ),
        top_k=(
            SECOND_PASS_TOP_K
        ),
    )


# ============================================================
# DISPLAY CANDIDATE ANALYSIS
# ============================================================

def display_candidates(
    title: str,
    candidates: List[RebindingCandidate],
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not candidates:

        print(
            "No symbolic candidates found."
        )

        return

    for candidate in candidates:

        print()
        print(
            f"STEP {candidate.rank}"
        )

        print(
            "  Potential prompt:"
        )

        print(
            f"    {candidate.potential_prompt}"
        )

        print(
            "  Candidate ground truth:"
        )

        print(
            f"    {candidate.candidate_ground_truth}"
        )

        print(
            "  Symbolic overlap:"
            f" {candidate.symbolic_overlap:.4f}"
        )

        print(
            "  Vector similarity:"
            f" {candidate.vector_similarity:.4f}"
        )

        print(
            "  Combined score:"
            f" {candidate.score:.4f}"
        )

        print(
            "  Corpus frequency:"
            f" {candidate.frequency}"
        )


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

def display_result(
    prompt: str,
    pre_candidates: List[RebindingCandidate],
    generated: str,
    result: RebindingResult,
    second_generation: Optional[str],
) -> None:

    print()
    print("=" * 70)
    print(
        "SPARSE SYMBOL MANIFEST"
    )
    print(
        "SYMBOLIC CANDIDATE ANALYSIS"
    )
    print("=" * 70)

    print()
    print(
        "ORIGINAL PROMPT"
    )

    print(prompt)

    # --------------------------------------------------------
    # FIRST PROCESSING STAGE
    # --------------------------------------------------------

    display_candidates(
        "1. SYMBOLIC CANDIDATE ANALYSIS",
        pre_candidates,
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "2. FIRST SYMBOLIC GENERATION"
    )
    print("=" * 70)

    print()
    print(generated)

    # --------------------------------------------------------
    # POST-GENERATION REBINDING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "3. POST-GENERATION REBINDING"
    )
    print("=" * 70)

    print()
    print(
        "Grounding score:"
        f" {result.grounding_score:.4f}"
    )

    print(
        "Symbolic overlap:"
        f" {result.symbolic_overlap:.4f}"
    )

    print(
        "Vector similarity:"
        f" {result.semantic_overlap:.4f}"
    )

    print(
        "Ambiguity:"
        f" {result.ambiguity:.4f}"
    )

    print(
        "Accepted:"
        f" {result.accepted}"
    )

    print(
        "Reason:"
        f" {result.reason}"
    )

    if result.accepted:

        print()
        print(
            "REBOUND PROMPT"
        )

        print(
            result.rebound_prompt
        )

        print()
        print(
            "CANDIDATE GROUND TRUTH"
        )

        print(
            result.rebound_ground_truth
        )

    # --------------------------------------------------------
    # SECOND PASS
    # --------------------------------------------------------

    if second_generation is not None:

        print()
        print("=" * 70)
        print(
            "4. SECOND SYMBOLIC PASS"
        )
        print("=" * 70)

        print()
        print(
            second_generation
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    corpus_path = Path(
        CORPUS_PATH
    )

    if not corpus_path.exists():

        print()
        print(
            f"ERROR: {CORPUS_PATH} "
            "does not exist."
        )

        print()
        print(
            "Create singlekb.txt in "
            "the same directory as this script."
        )

        return

  
    # --------------------------------------------------------
    # LOAD CORPUS
    # --------------------------------------------------------

    corpus_text = (
        corpus_path.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # TRAIN / LOAD MODEL
    # --------------------------------------------------------

    if (
        TRAIN_MODEL
        or not Path(
            MODEL_PATH
        ).exists()
    ):

        print()
        print(
            "Training sparse symbolic model..."
        )

        model = train_model(
            corpus_text
        )

    else:

        print()
        print(
            "Loading existing model..."
        )

        model = (
            SparseSymbolManifest.load_json(
                MODEL_PATH
            )
        )

    # --------------------------------------------------------
    # MODEL STATISTICS
    # --------------------------------------------------------

    print()
    print(
        f"Vocabulary: "
        f"{len(model.vocabulary)}"
    )

    print(
        f"Unigrams: "
        f"{len(model.unigram)}"
    )

    print(
        f"Bigram contexts: "
        f"{len(model.bigram)}"
    )

    print(
        f"Trigram contexts: "
        f"{len(model.trigram)}"
    )
    while True:
        # --------------------------------------------------------
        # USER INPUT
        # --------------------------------------------------------

        prompt = input(
            "USER: "
        ).strip()

        if not prompt:

            print(
                "Empty prompt."
            )


        # ========================================================
        # CREATE SYMBOLIC REFERENCE INDEX
        # ========================================================

        analyzer = SymbolicCandidateAnalyzer(
            lexical_weight=LEXICAL_WEIGHT,
            vector_weight=VECTOR_WEIGHT,
            acceptance_threshold=(
                ACCEPTANCE_THRESHOLD
            ),
            ambiguity_margin=(
                AMBIGUITY_MARGIN
            ),
        )

        analyzer.build_index(
            corpus_text
        )

        # ========================================================
        # FIRST PROCESSING STAGE
        #
        # Potential prompts are identified BEFORE generation.
        # ========================================================

        pre_candidates = analyzer.analyze(
            prompt,
            limit=CANDIDATE_LIMIT,
        )

        # ========================================================
        # GENERATION
        # ========================================================

        print()
        print(
            "Generating..."
        )

        generated = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

        # ========================================================
        # POST-GENERATION REBINDING
        # ========================================================

        result = analyzer.rebind(
            generated,
            candidate_limit=CANDIDATE_LIMIT,
        )

        # ========================================================
        # SECOND PASS
        # ========================================================

        second_generation = None

        if SECOND_PASS_GENERATION:

            second_generation = second_pass(
                model,
                result,
            )

        # ========================================================
        # DISPLAY
        # ========================================================

        display_result(
            prompt=prompt,
            pre_candidates=pre_candidates,
            generated=generated,
            result=result,
            second_generation=second_generation,
        )

        # ========================================================
        # COMPLETE MACHINE RECORD
        # ========================================================

        record = {

            "experiment": {

                "name":
                    "Sparse Symbol Manifest "
                    "Symbolic Candidate Analysis "
                    "Chinese Room Rebinding",

                "seed":
                    RANDOM_SEED,

                "prompt":
                    prompt,
            },

            "processing_order": [

                "symbolic_candidate_analysis",

                "first_generation",

                "post_generation_rebinding",

                "second_symbolic_pass",
            ],

            "model": {

                "model_path":
                    MODEL_PATH,

                "vocabulary_size":
                    len(model.vocabulary),

                "unigram_types":
                    len(model.unigram),

                "bigram_contexts":
                    len(model.bigram),

                "trigram_contexts":
                    len(model.trigram),
            },

            # ----------------------------------------------------
            # FIRST SYMBOLIC ANALYSIS
            # ----------------------------------------------------

            "symbolic_candidate_analysis": {

                "description":
                    "Observable symbolic candidate "
                    "generation performed before "
                    "language generation.",

                "candidate_limit":
                    CANDIDATE_LIMIT,

                "lexical_weight":
                    LEXICAL_WEIGHT,

                "vector_weight":
                    VECTOR_WEIGHT,

                "candidates": [
                    candidate.to_dict()
                    for candidate
                    in pre_candidates
                ],
            },

            # ----------------------------------------------------
            # GENERATION
            # ----------------------------------------------------

            "generation": {

                "temperature":
                    TEMPERATURE,

                "top_k":
                    TOP_K,

                "max_new_tokens":
                    MAX_NEW_TOKENS,

                "generated":
                    generated,
            },

            # ----------------------------------------------------
            # CHINESE ROOM
            # ----------------------------------------------------

            "chinese_room": {

                "candidate_limit":
                    CANDIDATE_LIMIT,

                "lexical_weight":
                    LEXICAL_WEIGHT,

                "vector_weight":
                    VECTOR_WEIGHT,

                "acceptance_threshold":
                    ACCEPTANCE_THRESHOLD,

                "ambiguity_margin":
                    AMBIGUITY_MARGIN,

                "rebinding":
                    result.to_dict(),
            },

            # ----------------------------------------------------
            # SECOND PASS
            # ----------------------------------------------------

            "second_pass": {

                "enabled":
                    SECOND_PASS_GENERATION,

                "input":
                    result.rebound_ground_truth,

                "generated":
                    second_generation,
            },

            # ----------------------------------------------------
            # INTERPRETATION
            # ----------------------------------------------------

            "interpretation": {

                "analysis_steps_are":
                    "Explicit symbolic scoring operations, "
                    "not hidden chain-of-thought.",

                "generated_symbols_are_not":
                    "Independently verified semantic understanding.",

                "candidate_ground_truth_is":
                    "A corpus-derived candidate reference.",

                "potential_prompt_is":
                    "A corpus sentence ranked as a possible "
                    "symbolic reference for the prompt or output.",

                "rebinding_measures":
                    "Lexical overlap and vector similarity "
                    "between symbolic text representations.",
            },
        }

        # ========================================================
        # SAVE RECORD
        # ========================================================

        Path(
            RECORD_PATH
        ).write_text(

            json.dumps(
                record,
                indent=2,
                ensure_ascii=False,
            ),

            encoding="utf-8",
        )

        print()
        print(
            f"Experiment record saved to "
            f"{RECORD_PATH}"
        )


    # ============================================================
    # ENTRY POINT
    # ============================================================

if __name__ == "__main__":
    main()
