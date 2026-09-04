from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ============================================================
# Small n-gram language model + corpus similarity search.
#
# Pipeline per user turn:
#   1. tokenize prompt, find closest corpus sentences (display only)
#   2. run several independent "scratch" generations from the model
#   3. combine those runs into a single candidate modifier: tokens
#      that show up consistently across runs reinforce each other,
#      tokens that only appear in a minority of runs cancel out
#   4. sample the final continuation, using that modifier (blended
#      50/50 with the raw baseline token probability) to bias the
#      per-token scores
#
# (The original script also computed a "post-generation rebinding"
#  pass and a "second generation" pass, but neither was ever
#  displayed or used — removed as dead code.)
# ============================================================

MODEL_PATH = "model.json"

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 20

MIN_COUNT = 1
INFLUENCE_TAU = 0.5

CURVE_K = 8.0
CURVE_MIDPOINT = 0.5

CANDIDATE_LIMIT = 15
LEXICAL_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55

# how hard the corpus evidence modifier (SUPPORTED/REFUTED sentences)
# pushes the final generation's per-token scores
EVIDENCE_WEIGHT = 0.4

# --- consensus / "cancel out" ensemble settings ---
NUM_GENERATIONS = 5        # how many scratch runs to generate per turn
CANCEL_STRENGTH = 1.0      # how hard disagreement between runs is penalized
MODIFIER_WEIGHT = 0.6      # how much the consensus modifier biases the final generation
CONSENSUS_BASELINE_SPLIT = 0.5  # 0.0 = pure baseline prob, 1.0 = pure consensus modifier

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\[\]{}\-]")
IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}


# ============================================================
# Text utilities
# ============================================================

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t not in IGNORED_TOKENS)


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a) - IGNORED_TOKENS
    sb = set(b) - IGNORED_TOKENS
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


# ============================================================
# Evidence
#
# Each corpus sentence is treated as a standing hypothesis about
# the world. Every user prompt is an observation that either
# supports it, contradicts it, or says nothing about it. Evidence
# accumulates across turns instead of being recomputed from
# scratch each time, so a sentence's status can evolve (or become
# CONFLICTED) as the conversation goes on.
# ============================================================

class Evidence(Enum):
    TRUE = "⊤"
    FALSE = "⊥"
    UNKNOWN = "?"
    CONFLICT = "!"


NEGATORS = ("not ", "never ", "did not ", "didn't ", "no ", "without ")


@dataclass
class EvidenceRecord:
    state: Evidence = Evidence.UNKNOWN
    true_count: int = 0
    false_count: int = 0
    similarities: List[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.true_count + self.false_count

    @property
    def consistency(self) -> float:
        if self.total == 0:
            return 0.0
        return max(self.true_count, self.false_count) / self.total

    def add(self, result: Evidence, similarity: float) -> None:
        if result is Evidence.UNKNOWN:
            return
        self.similarities.append(similarity)
        if result is Evidence.TRUE:
            self.true_count += 1
        elif result is Evidence.FALSE:
            self.false_count += 1
        if self.true_count and self.false_count:
            self.state = Evidence.CONFLICT
        elif self.true_count:
            self.state = Evidence.TRUE
        elif self.false_count:
            self.state = Evidence.FALSE

    @property
    def verdict(self) -> str:
        return {
            Evidence.TRUE: "SUPPORTED",
            Evidence.FALSE: "REFUTED",
            Evidence.UNKNOWN: "UNRESOLVED",
            Evidence.CONFLICT: "CONFLICTED",
        }[self.state]


# ============================================================
# Geometric text space (TF-IDF -> SVD -> normalized vectors)
#
# Replaces the old raw bag-of-words cosine similarity. Sparse
# term-count cosine tends to reward any shared word equally; TF-IDF
# downweights corpus-wide filler and SVD lets sentences that share
# no exact words still land close together if they share context.
# The similarity SCALE this pipeline produces is much smaller than
# a hand-picked constant like 0.35 would assume, so thresholds are
# calibrated per turn from what's actually achievable (see analyze).
# ============================================================

class GeometricDataset:
    def __init__(self, texts: List[str], dimensions: int = 16) -> None:
        self.texts = texts
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)

        max_dim = min(dimensions, max(1, X.shape[0] - 1), max(1, X.shape[1] - 1))

        if X.shape[1] <= 1:
            self.X = X.toarray().astype(float)
            self.svd = None
        else:
            self.svd = TruncatedSVD(n_components=max_dim, random_state=42)
            self.X = self.svd.fit_transform(X)

        self.X = normalize(self.X)

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        if self.svd is not None:
            X = self.svd.transform(X)
        return normalize(X)

    def vector(self, text: str) -> np.ndarray:
        return self.transform([text])[0]


# ============================================================
# Corpus similarity search (used for the "candidates" display)
# ============================================================

@dataclass
class CorpusReference:
    sentence: str
    tokens: List[str]
    vector: np.ndarray
    frequency: int = 1
    evidence: EvidenceRecord = field(default_factory=EvidenceRecord)


@dataclass
class Candidate:
    sentence: str
    symbolic_overlap: float
    vector_similarity: float
    frequency: int
    score: float
    state: Evidence
    verdict: str
    true_count: int
    false_count: int
    consistency: float
    samples: int
    rank: int = 0


class CorpusSearch:
    """
    Finds corpus sentences closest to a prompt, and classifies each
    one as supported, refuted, unresolved, or conflicted evidence
    given everything said so far this session.
    """

    def __init__(
        self,
        lexical_weight: float = LEXICAL_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.references: List[CorpusReference] = []
        self.dataset: Optional[GeometricDataset] = None

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(s.lower() for s in sentences)

        kept = [(s, tokenize(s)) for s in sentences]
        kept = [(s, t) for s, t in kept if t]
        if not kept:
            self.references = []
            self.dataset = None
            return

        self.dataset = GeometricDataset([s for s, _ in kept], dimensions=16)

        self.references = [
            CorpusReference(
                sentence=sentence,
                tokens=tokens,
                vector=self.dataset.X[i],
                frequency=counts[sentence.lower()],
            )
            for i, (sentence, tokens) in enumerate(kept)
        ]

    def analyze(self, prompt: str, limit: int = 5) -> Tuple[List[Candidate], Dict[str, float]]:
        if not self.references or self.dataset is None:
            return [], {}

        prompt_tokens = tokenize(prompt)
        prompt_vector = self.dataset.vector(prompt)

        similarities = [float(np.dot(prompt_vector, ref.vector)) for ref in self.references]

        # Calibrate this turn's support/contradiction bar against what
        # this prompt actually achieved, instead of a fixed constant
        # tuned for a different similarity scale (dense embeddings).
        # contradiction_threshold is kept <= support_threshold so a
        # negated-but-on-topic prompt is caught as a contradiction
        # before it can be mistaken for support.
        ceiling = max(similarities) if similarities else 0.0
        support_threshold = 0.55 * ceiling
        contradiction_threshold = 0.45 * ceiling

        text = " " + prompt.lower() + " "
        explicitly_negative = any(neg in text for neg in NEGATORS)

        candidates = []
        raw_modifier: Dict[str, float] = defaultdict(float)

        for ref, vector_sim in zip(self.references, similarities):
            if ceiling <= 1e-9:
                result = Evidence.UNKNOWN
            elif explicitly_negative and vector_sim >= contradiction_threshold:
                result = Evidence.FALSE
            elif vector_sim >= support_threshold:
                result = Evidence.TRUE
            else:
                result = Evidence.UNKNOWN

            ref.evidence.add(result, vector_sim)

            symbolic = lexical_overlap(prompt_tokens, ref.tokens)
            score = self.lexical_weight * symbolic + self.vector_weight * vector_sim

            candidates.append(
                Candidate(
                    sentence=ref.sentence,
                    symbolic_overlap=symbolic,
                    vector_similarity=vector_sim,
                    frequency=ref.frequency,
                    score=score,
                    state=ref.evidence.state,
                    verdict=ref.evidence.verdict,
                    true_count=ref.evidence.true_count,
                    false_count=ref.evidence.false_count,
                    consistency=ref.evidence.consistency,
                    samples=ref.evidence.total,
                )
            )

            # --- compartmental evidence modifier ---
            # Each reference is its own compartment: its (sim, symbolic)
            # -> score is computed independently, then spread onto only
            # that reference's own tokens, signed by its evidence state.
            # SUPPORTED sentences push their tokens up, REFUTED sentences
            # push theirs down, CONFLICTED sentences push proportionally
            # to which side currently outweighs the other, and UNRESOLVED
            # sentences contribute nothing. Compartments are then summed
            # additively per token, so a token repeated across several
            # supported sentences accumulates more bias than one that
            # only shows up once.
            state = ref.evidence.state
            if state is Evidence.TRUE:
                sign = 1.0
            elif state is Evidence.FALSE:
                sign = -1.0
            elif state is Evidence.CONFLICT:
                total = ref.evidence.total
                sign = (ref.evidence.true_count - ref.evidence.false_count) / total if total else 0.0
            else:  # UNKNOWN
                sign = 0.0

            if sign != 0.0:
                weight = sign * score
                for token in ref.tokens:
                    if token not in IGNORED_TOKENS:
                        raw_modifier[token] += weight

        evidence_modifier: Dict[str, float] = {}
        if raw_modifier:
            largest = max(abs(v) for v in raw_modifier.values())
            if largest > 0:
                evidence_modifier = {t: v / largest for t, v in raw_modifier.items()}

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:limit]
        for i, c in enumerate(candidates, start=1):
            c.rank = i
        return candidates, evidence_modifier


# ============================================================
# N-gram language model
# ============================================================

@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT
    influence_tau: float = INFLUENCE_TAU
    curve_k: float = CURVE_K
    curve_midpoint: float = CURVE_MIDPOINT

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    # ---------------- ingestion ----------------

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)
            if not words:
                continue
            sequence = ["<bos>", "<bos>"] + words + [self.eos_token]
            self._add_sequence(sequence)

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    # ---------------- training ----------------

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}
        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {ctx: c / total for ctx, c in counts.items()}

        self.influence_vectors = {}
        for source in self.vocabulary:
            source_vec = self.lexical_vectors.get(source, {})
            scores = {}
            for target in self.vocabulary:
                if source == target:
                    continue
                sim = cosine_similarity(source_vec, self.lexical_vectors.get(target, {}))
                if sim >= self.influence_tau:
                    scores[target] = sim
            self.influence_vectors[source] = scores

        self.finalized = True

    # ---------------- generation ----------------

    def _backoff_distribution(self, prev: str, prev_prev: Optional[str]) -> Dict[str, float]:
        if prev_prev is not None:
            counts = self.trigram.get(f"{prev_prev}\t{prev}")
            if counts:
                return self._normalize(counts)
        counts = self.bigram.get(prev)
        if counts:
            return self._normalize(counts)
        return self._normalize(self.unigram)

    @staticmethod
    def _normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()} if total else {}

    def _curve_weight(self, p_eos: float) -> float:
        p_eos = min(1.0, max(0.0, p_eos))
        z = self.curve_k * (p_eos - self.curve_midpoint)
        return 1.0 / (1.0 + math.exp(z))

    def _resolve_context(self, prompt: str) -> Tuple[str, Optional[str]]:
        tokens = tokenize(prompt)
        if not tokens:
            return "<bos>", None
        prev = tokens[-1]
        prev_prev = tokens[-2] if len(tokens) >= 2 else None
        return prev, prev_prev

    def _score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        prev, prev_prev = self._resolve_context(prompt)
        base = self._backoff_distribution(prev, prev_prev)
        if not base:
            return {}

        candidates = [t for t in sorted(base, key=base.get, reverse=True) if t not in IGNORED_TOKENS][:candidate_limit]
        source_vec = self.lexical_vectors.get(prev, {})
        influences = self.influence_vectors.get(prev, {})
        curve = self._curve_weight(base.get(self.eos_token, 0.0))

        scores = {}
        for token in candidates:
            similarity = cosine_similarity(source_vec, self.lexical_vectors.get(token, {}))
            influence = influences.get(token, 0.0)
            score = (
                safe_log(base[token])
                + curve * 0.35 * similarity
                + curve * 0.65 * influence
            )
            if candidate_modifier and modifier_weight:
                # Split the difference between the ensemble consensus
                # ("census") weight and the raw baseline probability for
                # this token, instead of using the consensus weight alone.
                consensus_weight = candidate_modifier.get(token, 0.0)
                baseline_prob = base.get(token, 0.0)
                blended_bias = (
                    CONSENSUS_BASELINE_SPLIT * consensus_weight
                    + (1.0 - CONSENSUS_BASELINE_SPLIT) * baseline_prob
                )
                score += modifier_weight * blended_bias
            if evidence_modifier and evidence_weight:
                # Signed compartmental bias from CorpusSearch: positive
                # for tokens belonging to SUPPORTED corpus sentences,
                # negative for REFUTED ones, applied directly (unlike
                # the consensus modifier this one is meaningfully
                # negative, so it isn't blended toward a nonnegative
                # baseline probability).
                score += evidence_weight * evidence_modifier.get(token, 0.0)
            scores[token] = score
        return scores

    def _probabilities(
        self,
        prompt: str,
        temperature: float,
        candidate_limit: int,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
    ) -> Dict[str, float]:
        scores = self._score_next_token(
            prompt, candidate_limit, candidate_modifier, modifier_weight,
            evidence_modifier, evidence_weight,
        )
        if not scores:
            return {}
        temperature = max(temperature, 1e-5)
        scaled = {t: s / temperature for t, s in scores.items()}
        maximum = max(scaled.values())
        exps = {t: math.exp(s - maximum) for t, s in scaled.items()}
        total = sum(exps.values())
        return {t: v / total for t, v in exps.items()} if total else {}

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
    ) -> str:
        probs = self._probabilities(
            prompt, temperature, max(top_k, 1), candidate_modifier, modifier_weight,
            evidence_modifier, evidence_weight,
        )
        if not probs:
            return self.eos_token
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
    ) -> str:
        generated = tokenize(prompt)
        for _ in range(max_new_tokens):
            token = self.sample_next(
                " ".join(generated), temperature, top_k, candidate_modifier, modifier_weight,
                evidence_modifier, evidence_weight,
            )
            #if token == self.eos_token:
                #break
            generated.append(token)
        return self.detokenize(generated)

    def generate_with_trace(
        self, prompt: str, max_new_tokens: int, temperature: float, top_k: int
    ) -> Tuple[str, List[str]]:
        """Generate once and also return just the newly generated tokens."""
        generated = tokenize(prompt)
        start = len(generated)
        for _ in range(max_new_tokens):
            token = self.sample_next(" ".join(generated), temperature, top_k)
            #if token == self.eos_token:
                #break
            generated.append(token)
        return self.detokenize(generated), generated[start:]

    def multi_generate(
        self,
        prompt: str,
        num_generations: int = NUM_GENERATIONS,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> List[List[str]]:
        """Run several independent scratch generations from the same prompt."""
        runs: List[List[str]] = []
        for _ in range(num_generations):
            _, new_tokens = self.generate_with_trace(prompt, max_new_tokens, temperature, top_k)
            runs.append(new_tokens)
        return runs

    @staticmethod
    def build_candidate_modifier(
        runs: List[List[str]], cancel_strength: float = CANCEL_STRENGTH
    ) -> Dict[str, float]:
        """
        Combine multiple generation runs ("arrays") into one candidate modifier.

        Each run is turned into a token-count vector. The vectors are
        combined by taking their mean and then subtracting the
        disagreement between them (population std-dev), scaled by
        cancel_strength:

            modifier[token] = mean_count(token) - cancel_strength * std(token)

        Tokens that appear consistently across runs (low variance) survive
        and reinforce each other. Tokens that only show up in a minority of
        runs (high variance relative to their mean) get cancelled out to
        zero or below and are dropped. The surviving values are normalized
        to 0..1 so they can be blended into the scoring function.
        """
        if not runs:
            return {}

        counters = [Counter(run) for run in runs]
        vocab = set()
        for c in counters:
            vocab.update(c.keys())

        n = len(counters)
        modifier: Dict[str, float] = {}
        for token in vocab:
            values = [c.get(token, 0) for c in counters]
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(variance)
            net = mean - cancel_strength * std
            if net > 0:
                modifier[token] = net

        if not modifier:
            return {}
        max_val = max(modifier.values())
        return {t: v / max_val for t, v in modifier.items()}

    def generate_consensus(
        self,
        prompt: str,
        num_generations: int = NUM_GENERATIONS,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        cancel_strength: float = CANCEL_STRENGTH,
        modifier_weight: float = MODIFIER_WEIGHT,
        evidence_modifier: Optional[Dict[str, float]] = None,
        evidence_weight: float = 0.0,
    ) -> Tuple[str, List[List[str]], Dict[str, float]]:
        """
        Full pipeline: run several scratch generations, cancel them out
        against each other into a candidate modifier, then do one more
        final generation. The final generation's per-token bias splits
        the difference between that consensus modifier and each token's
        raw baseline probability (see CONSENSUS_BASELINE_SPLIT), and is
        additionally pushed by the corpus evidence_modifier (from
        CorpusSearch.analyze) toward tokens belonging to currently
        SUPPORTED sentences and away from REFUTED ones. Scratch runs are
        left unbiased so they stay independent of each other.

        Returns (final_text, scratch_runs, modifier) so callers can inspect
        what survived the cancel-out pass.
        """
        runs = self.multi_generate(prompt, num_generations, max_new_tokens, temperature, top_k)
        modifier = self.build_candidate_modifier(runs, cancel_strength)
        final_text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            candidate_modifier=modifier,
            modifier_weight=modifier_weight,
            evidence_modifier=evidence_modifier,
            evidence_weight=evidence_weight,
        )
        return final_text, runs, modifier

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:)\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        return text

    # ---------------- persistence ----------------

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "influence_tau": self.influence_tau,
            "curve_k": self.curve_k,
            "curve_midpoint": self.curve_midpoint,
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {k: dict(v) for k, v in self.trigram.items()},
            "lexical_vectors": self.lexical_vectors,
            "influence_vectors": self.influence_vectors,
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data["eos_token"],
            unk_token=data["unk_token"],
            min_count=data["min_count"],
            influence_tau=data["influence_tau"],
            curve_k=data["curve_k"],
            curve_midpoint=data["curve_midpoint"],
        )
        model.unigram = Counter(data["unigram"])
        model.bigram = defaultdict(Counter, {k: Counter(v) for k, v in data["bigram"].items()})
        model.trigram = defaultdict(Counter, {k: Counter(v) for k, v in data["trigram"].items()})
        model.lexical_vectors = data["lexical_vectors"]
        model.influence_vectors = data["influence_vectors"]
        model.vocabulary = data["vocabulary"]
        model.finalized = data["finalized"]
        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ============================================================
# Display
# ============================================================

def display_candidates(candidates: List[Candidate]) -> None:
    print()
    print("=" * 70)
    print("REASONING")
    print("=" * 70)
    if not candidates:
        print("No candidates found.")
        return
    for c in candidates:
        print(f"\n[{c.rank}] score={c.score:.3f}  state={c.state.value} ({c.verdict})  {c.sentence}")
        print(
            f"      sim={c.vector_similarity:.3f}  symbolic={c.symbolic_overlap:.3f}  "
            f"true={c.true_count}  false={c.false_count}  "
            f"consistency={c.consistency:.2f}  samples={c.samples}"
        )


def display_ensemble(runs: List[List[str]], modifier: Dict[str, float]) -> None:
    print()
    print("=" * 70)
    print(f"ENSEMBLE ({len(runs)} scratch generations, cancelled against each other)")
    print("=" * 70)
    for i, run in enumerate(runs, start=1):
        preview = NGramModel.detokenize(run[:20])
        print(f"\n[run {i}] {preview}{' ...' if len(run) > 20 else ''}")
    top_survivors = sorted(modifier.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\nSurviving tokens after cancel-out (top 15):")
    if not top_survivors:
        print("  (none survived — runs disagreed on everything)")
    for token, weight in top_survivors:
        print(f"  {token!r:<15} weight={weight:.3f}")


def display_generation(generated: str) -> None:
    print()
    print("=" * 70)
    print("GENERATED")
    print("=" * 70)
    print()
    print(generated)


# ============================================================
# Main
# ============================================================

def main() -> None:
    corpus_path = Path(input("Filename: "))
    if not corpus_path.exists():
        print(f"\nERROR: {corpus_path} does not exist.")
        return

    corpus_text = corpus_path.read_text(encoding="utf-8")

    if Path(MODEL_PATH).exists():
        print("\nLoading existing model...")
        model = NGramModel.load_json(MODEL_PATH)
    else:
        print("\nTraining n-gram model...")
        model = NGramModel(min_count=MIN_COUNT, influence_tau=INFLUENCE_TAU)
        model.ingest_text(corpus_text)
        model.finalize()
        model.save_json(MODEL_PATH)

    print(f"\nVocabulary: {len(model.vocabulary)}")
    print(f"Unigrams: {len(model.unigram)}")
    print(f"Bigram contexts: {len(model.bigram)}")
    print(f"Trigram contexts: {len(model.trigram)}")

    search = CorpusSearch(lexical_weight=LEXICAL_WEIGHT, vector_weight=VECTOR_WEIGHT)
    search.build_index(corpus_text)  # built once, not on every turn

    while True:
        prompt = input("\nUSER: ").strip()
        if not prompt:
            print("Empty prompt.")
            continue

        candidates, evidence_modifier = search.analyze(prompt, limit=CANDIDATE_LIMIT)
        display_candidates(candidates)

        print(f"\nRunning {NUM_GENERATIONS} scratch generations to build a consensus modifier...")
        final_text, runs, modifier = model.generate_consensus(
            prompt,
            num_generations=NUM_GENERATIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            cancel_strength=CANCEL_STRENGTH,
            modifier_weight=MODIFIER_WEIGHT,
            evidence_modifier=evidence_modifier,
            evidence_weight=EVIDENCE_WEIGHT,
        )

        print("\nGenerating final (modifier- and evidence-biased)...")
        display_generation(final_text)


if __name__ == "__main__":
    main()
