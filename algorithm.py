from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ============================================================
# Small n-gram language model + corpus similarity search.
#
# Pipeline per user turn:
#   1. tokenize prompt, find closest corpus sentences (display only)
#   2. run several independent "scratch" generations from the model
#   3. combine those runs into a single candidate modifier using a
#      kernelized adversarial consensus pass (see below), instead of
#      the old mean/std cancel-out
#   4. sample the final continuation, using that modifier (blended
#      50/50 with the raw baseline token probability) to bias the
#      per-token scores
#
# --- Kernelized adversarial consensus ---
# Each scratch run is first canonicalized through the vocabulary's
# "isomorphism classes": tokens whose bigram-context distributions are
# near-identical (cosine similarity above ISOMORPHISM_TAU) play the same
# structural role in the grammar, so they're collapsed to one canonical
# representative before anything is counted. This means a vote for any
# member of a class reinforces the whole class, both when the modifier
# is built and when it's looked up during final scoring.
#
# Each (canonicalized) run is then treated as a point in token-count
# space. An RBF kernel measures how close every run sits to the others.
# Runs near the shared cluster are the ensemble's "promises" -- tokens
# they contain get amplified. Runs sitting apart, in the interstitial
# space between clusters, are adversarially suppressed: their
# kernel-agreement score is raised to a sharpening power before being
# used as a weight, which pushes outlier runs toward zero rather than
# just discounting them linearly (as the old std-based penalty did).
#
# (The original script also computed a "post-generation rebinding"
#  pass and a "second generation" pass, but neither was ever
#  displayed or used -- removed as dead code.)
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

# --- consensus / kernelized adversarial ensemble settings ---
NUM_GENERATIONS = 5        # how many scratch runs to generate per turn
KERNEL_SHARPNESS = 4.0     # adversarial sharpening exponent on kernel agreement
MODIFIER_WEIGHT = 0.6      # how much the consensus modifier biases the final generation
CONSENSUS_BASELINE_SPLIT = 0.5  # 0.0 = pure baseline prob, 1.0 = pure consensus modifier

# --- vocab isomorphism settings ---
ISOMORPHISM_TAU = 0.97     # cosine similarity threshold for treating two tokens
                           # as structurally interchangeable in the vocabulary

# --- Markovian transitivity masking settings ---
TRANSITIVITY_DECAY = 0.5   # decay applied per hop when composing A->B->C into A->C
TRANSITIVITY_WEIGHT = 0.5  # how strongly the prompt-pattern mask biases scoring
TRANSITIVITY_MASK_PENALTY = 1.0   # "deficit strength" for masked-out tokens (fed into the exponential, not a raw log-penalty anymore)
TRANSITIVITY_SUPERPOLY_K = 3.0    # exponential growth rate applied to promise strength; higher = more explosive gap between weakly- and strongly-promised tokens

RANDOM_SEED = None  # set to an int for reproducible runs; None = fresh entropy each run
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
# Corpus similarity search (used for the "candidates" display)
# ============================================================

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
    """Finds corpus sentences closest to a prompt (lexical + cosine)."""

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
        counts = Counter(s.lower() for s in sentences)

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
                    vector={t: float(c) for t, c in bow.items()},
                    frequency=counts[sentence.lower()],
                )
            )

    def analyze(self, prompt: str, limit: int = 5) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)
        prompt_vector = {t: float(c) for t, c in bag_of_words(prompt_tokens).items()}

        candidates = []
        for ref in self.references:
            symbolic = lexical_overlap(prompt_tokens, ref.tokens)
            vector_sim = cosine_similarity(prompt_vector, ref.vector)
            score = self.lexical_weight * symbolic + self.vector_weight * vector_sim
            candidates.append(
                Candidate(
                    sentence=ref.sentence,
                    symbolic_overlap=symbolic,
                    vector_similarity=vector_sim,
                    frequency=ref.frequency,
                    score=score,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:limit]
        for i, c in enumerate(candidates, start=1):
            c.rank = i
        return candidates


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
    isomorphism_tau: float = ISOMORPHISM_TAU

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    isomorphism_map: Dict[str, str] = field(default_factory=dict)
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

        self._build_isomorphism_classes()

        self.finalized = True

    def _build_isomorphism_classes(self) -> None:
        """
        Group tokens whose bigram-context distributions are (near) identical.
        Two tokens are treated as isomorphic in the vocabulary if their
        lexical vectors are nearly parallel (cosine >= isomorphism_tau) --
        i.e. they play the same structural role in the grammar even if
        they're literally different words. Each class collapses to a single
        canonical token (its first member, in sorted vocabulary order) so
        that a vote cast for any member reinforces the whole class.
        """
        canonical: Dict[str, str] = {}
        assigned = set()
        vocab = self.vocabulary
        for i, tok_a in enumerate(vocab):
            if tok_a in assigned:
                continue
            canonical[tok_a] = tok_a
            assigned.add(tok_a)
            vec_a = self.lexical_vectors.get(tok_a, {})
            if not vec_a:
                continue
            for tok_b in vocab[i + 1:]:
                if tok_b in assigned:
                    continue
                vec_b = self.lexical_vectors.get(tok_b, {})
                if not vec_b:
                    continue
                if cosine_similarity(vec_a, vec_b) >= self.isomorphism_tau:
                    canonical[tok_b] = tok_a
                    assigned.add(tok_b)
        self.isomorphism_map = canonical

    def _canonical(self, token: str) -> str:
        return self.isomorphism_map.get(token, token)

    def _canonicalize_run(self, run: List[str]) -> Counter:
        return Counter(self._canonical(t) for t in run)

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
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
        mask_penalty: float = TRANSITIVITY_MASK_PENALTY,
        superpoly_k: float = TRANSITIVITY_SUPERPOLY_K,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        prev, prev_prev = self._resolve_context(prompt)
        base = self._backoff_distribution(prev, prev_prev)
        if not base:
            return {}

        candidates = sorted(base, key=base.get, reverse=True)[:candidate_limit]
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
                # weight and the raw baseline probability for this token,
                # instead of using the consensus weight alone. The lookup
                # goes through the isomorphism map so a token benefits from
                # votes cast for any structurally-equivalent partner.
                consensus_weight = candidate_modifier.get(self._canonical(token), 0.0)
                baseline_prob = base.get(token, 0.0)
                blended_bias = (
                    CONSENSUS_BASELINE_SPLIT * consensus_weight
                    + (1.0 - CONSENSUS_BASELINE_SPLIT) * baseline_prob
                )
                score += modifier_weight * blended_bias
            if transitivity_mask is not None and transitivity_weight:
                # Superpolynomial masking: rather than adding the promise
                # weight linearly, run it through an exponential so the
                # gap between a weakly-promised and a strongly-promised
                # token blows up rather than scaling proportionally.
                # Masked-out tokens get the mirror-image exponential
                # penalty (using the fixed deficit strength) instead of a
                # flat linear subtraction.
                mask_weight = transitivity_mask.get(token)
                if mask_weight is not None:
                    score += transitivity_weight * (math.exp(superpoly_k * mask_weight) - 1.0)
                else:
                    score -= transitivity_weight * (math.exp(superpoly_k * mask_penalty) - 1.0)
            scores[token] = score
        return scores

    def _probabilities(
        self,
        prompt: str,
        temperature: float,
        candidate_limit: int,
        candidate_modifier: Optional[Dict[str, float]] = None,
        modifier_weight: float = 0.0,
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
    ) -> Dict[str, float]:
        scores = self._score_next_token(
            prompt,
            candidate_limit,
            candidate_modifier,
            modifier_weight,
            transitivity_mask,
            transitivity_weight,
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
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
    ) -> str:
        probs = self._probabilities(
            prompt,
            temperature,
            max(top_k, 1),
            candidate_modifier,
            modifier_weight,
            transitivity_mask,
            transitivity_weight,
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
        transitivity_mask: Optional[Dict[str, float]] = None,
        transitivity_weight: float = 0.0,
    ) -> str:
        generated = tokenize(prompt)
        for _ in range(max_new_tokens):
            token = self.sample_next(
                " ".join(generated),
                temperature,
                top_k,
                candidate_modifier,
                modifier_weight,
                transitivity_mask,
                transitivity_weight,
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
    def _sq_dist(a: Counter, b: Counter) -> float:
        keys = set(a) | set(b)
        return float(sum((a.get(k, 0) - b.get(k, 0)) ** 2 for k in keys))

    def build_kernel_consensus_modifier(
        self,
        runs: List[List[str]],
        sharpness: float = KERNEL_SHARPNESS,
    ) -> Dict[str, float]:
        """
        Kernelized adversarial replacement for the old mean/std cancel-out.

        Each scratch run is canonicalized through the vocabulary's
        isomorphism classes (structurally-equivalent tokens collapsed to
        one representative), then treated as a point in token-count space.

        An RBF kernel (bandwidth set by the median-distance heuristic)
        measures how close each run sits to the others. A run's
        "agreement" score is its average kernel similarity to every other
        run -- high if it resembles the shared cluster (a "promise" the
        ensemble is making), low if it sits apart in the interstitial
        space between clusters (an outlier).

        Those agreement scores are then raised to `sharpness` before being
        renormalized into weights. This is the adversarial step: it
        doesn't just discount outliers linearly, it pushes them toward
        zero influence while pushes consensus runs' tokens up, so the
        final modifier is dominated by tokens the ensemble actually
        agrees on.

        Returns a 0..1 normalized dict suitable for blending into scoring.
        """
        if not runs:
            return {}

        counters = [self._canonicalize_run(run) for run in runs]
        n = len(counters)

        if n == 1:
            total = sum(counters[0].values()) or 1
            return {t: c / total for t, c in counters[0].items()}

        sq_dists = [[0.0] * n for _ in range(n)]
        all_d2 = []
        for i in range(n):
            for j in range(i + 1, n):
                d2 = self._sq_dist(counters[i], counters[j])
                sq_dists[i][j] = sq_dists[j][i] = d2
                all_d2.append(d2)

        nonzero = [d for d in all_d2 if d > 0]
        if nonzero:
            median_d2 = sorted(nonzero)[len(nonzero) // 2]
            gamma = 1.0 / (2.0 * median_d2)
        else:
            gamma = 1.0  # every run produced an identical token multiset

        agreement = []
        for i in range(n):
            sims = [math.exp(-gamma * sq_dists[i][j]) for j in range(n) if j != i]
            agreement.append(sum(sims) / len(sims) if sims else 0.0)

        sharpened = [max(a, 0.0) ** sharpness for a in agreement]
        weight_total = sum(sharpened)
        weights = (
            [s / weight_total for s in sharpened]
            if weight_total > 0
            else [1.0 / n] * n
        )

        modifier: Dict[str, float] = defaultdict(float)
        for weight, counter in zip(weights, counters):
            for token, count in counter.items():
                modifier[token] += weight * count

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
        kernel_sharpness: float = KERNEL_SHARPNESS,
        modifier_weight: float = MODIFIER_WEIGHT,
    ) -> Tuple[str, List[List[str]], Dict[str, float]]:
        """
        Full pipeline: run several scratch generations, combine them with
        the kernelized adversarial consensus pass into a candidate
        modifier, then do one more final generation. The final
        generation's per-token bias splits the difference between that
        consensus modifier and each token's raw baseline probability
        (see CONSENSUS_BASELINE_SPLIT).

        Returns (final_text, scratch_runs, modifier) so callers can
        inspect what survived the kernel/isomorphism pass.
        """
        if not self.finalized:
            self.finalize()
        runs = self.multi_generate(prompt, num_generations, max_new_tokens, temperature, top_k)
        modifier = self.build_kernel_consensus_modifier(runs, kernel_sharpness)
        final_text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            candidate_modifier=modifier,
            modifier_weight=modifier_weight,
        )
        return final_text, runs, modifier

    # ---------------- non-conjoint context matrix / rhombus selection ----------------

    def build_disjointness_matrix(self, top_n: int = 30) -> Tuple[List[str], List[List[float]]]:
        """
        Build a square matrix over the `top_n` most frequent bigram contexts
        (rows/cols = contexts, i.e. this is a slice of the Markov transition
        table). Entry (i, j) is a disjointness score in [0, 1]: 1.0 means
        contexts i and j are "non-conjoint" -- their successor-token sets
        share nothing in common (Jaccard distance of the two rows) -- and
        0.0 means they always transition to the same tokens.
        """
        contexts = sorted(
            self.bigram.keys(),
            key=lambda c: sum(self.bigram[c].values()),
            reverse=True,
        )[:top_n]
        succ_sets = {c: set(self.bigram[c]) for c in contexts}

        n = len(contexts)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            a = succ_sets[contexts[i]]
            for j in range(n):
                b = succ_sets[contexts[j]]
                union = a | b
                matrix[i][j] = 1.0 if not union else 1.0 - len(a & b) / len(union)
        return contexts, matrix

    @staticmethod
    def rhombus_select_lateral(
        matrix: List[List[float]], radius: int = 2
    ) -> Dict[Tuple[int, int], float]:
        """
        Select the "lateral" part of a square matrix using a rhombus
        (diamond) band mask centered on the main diagonal: every cell
        (i, j) with |i - j| <= radius, EXCLUDING the diagonal itself
        (i == j). Plotted on a grid this is a diamond-shaped strip
        running along the diagonal; "lateral" means we keep only the
        wings of that strip (the off-diagonal transitions between
        distinct contexts/tokens) and drop the diagonal (self-transitions,
        i == j) which carry no lateral information.
        """
        n = len(matrix)
        selected: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if abs(i - j) <= radius:
                    selected[(i, j)] = matrix[i][j]
        return selected

    def non_conjoint_lateral_report(
        self,
        top_n: int = 30,
        radius: int = 2,
        threshold: float = 0.999,
    ) -> List[Tuple[str, str, float]]:
        """
        Full pipeline: build the context disjointness matrix, rhombus-select
        its lateral (off-diagonal, diamond-banded) region, then keep only
        the pairs that are (near-)perfectly non-conjoint (score >= threshold),
        sorted strongest-first.
        """
        contexts, matrix = self.build_disjointness_matrix(top_n)
        lateral = self.rhombus_select_lateral(matrix, radius)
        pairs = [
            (contexts[i], contexts[j], score)
            for (i, j), score in lateral.items()
            if score >= threshold
        ]
        pairs.sort(key=lambda t: -t[2])
        return pairs

    # ---------------- Markovian transitivity / prompt-pattern masking ----------------

    def transitive_successors(self, token: str, decay: float = 0.5) -> Dict[str, float]:
        """
        One hop of Markovian transitive closure. If A has a direct bigram
        successor B (prob p_ab), and B itself has a direct successor C
        (prob p_bc), then A relates to C transitively even if "A C" never
        literally occurs in the corpus. The transitive weight is
        p_ab * p_bc * decay, summed over all intermediate B's.
        """
        direct = self._normalize(self.bigram.get(token, Counter()))
        transitive: Dict[str, float] = defaultdict(float)
        for mid, p_ab in direct.items():
            if mid in IGNORED_TOKENS:
                continue
            mid_direct = self._normalize(self.bigram.get(mid, Counter()))
            for c, p_bc in mid_direct.items():
                transitive[c] += p_ab * p_bc * decay
        return dict(transitive)

    def prompt_transitivity_mask(self, prompt: str, decay: float = 0.5) -> Dict[str, float]:
        """
        Masking via prompt patterns: build a vocabulary mask from the
        tokens that actually appear in the prompt. For each distinct
        prompt token, compute its transitive (2-hop) Markov successors;
        the mask value for a candidate token is the max transitive weight
        reaching it from any prompt token. Tokens with no transitive path
        from anything in the prompt get no entry at all -- they're
        implicitly masked out when this is applied during scoring.
        """
        tokens = [t for t in tokenize(prompt) if t not in IGNORED_TOKENS]
        mask: Dict[str, float] = defaultdict(float)
        seen = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            for c, w in self.transitive_successors(t, decay).items():
                if w > mask[c]:
                    mask[c] = w
        return dict(mask)

    def generate_with_transitivity_mask(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        decay: float = TRANSITIVITY_DECAY,
        transitivity_weight: float = TRANSITIVITY_WEIGHT,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Convenience wrapper: build the prompt-pattern transitivity mask
        from the current prompt, then generate with it applied. Returns
        (generated_text, mask) so callers can inspect what was allowed
        through vs. masked out.
        """
        if not self.finalized:
            self.finalize()
        mask = self.prompt_transitivity_mask(prompt, decay)
        text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            transitivity_mask=mask,
            transitivity_weight=transitivity_weight,
        )
        return text, mask

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
            "isomorphism_tau": self.isomorphism_tau,
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {k: dict(v) for k, v in self.trigram.items()},
            "lexical_vectors": self.lexical_vectors,
            "influence_vectors": self.influence_vectors,
            "isomorphism_map": self.isomorphism_map,
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
            isomorphism_tau=data.get("isomorphism_tau", ISOMORPHISM_TAU),
        )
        model.unigram = Counter(data["unigram"])
        model.bigram = defaultdict(Counter, {k: Counter(v) for k, v in data["bigram"].items()})
        model.trigram = defaultdict(Counter, {k: Counter(v) for k, v in data["trigram"].items()})
        model.lexical_vectors = data["lexical_vectors"]
        model.influence_vectors = data["influence_vectors"]
        model.vocabulary = data["vocabulary"]
        model.finalized = data["finalized"]

        if "isomorphism_map" in data:
            model.isomorphism_map = data["isomorphism_map"]
        elif model.vocabulary and model.lexical_vectors:
            # Migrating an older model.json saved before isomorphism
            # classes existed: everything needed (vocabulary, lexical
            # vectors) is already loaded, so build the map now rather
            # than silently leaving it empty.
            model._build_isomorphism_classes()
        else:
            model.isomorphism_map = {}
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
        print(f"\n[{c.rank}] score={c.score:.3f}  {c.sentence}")


def display_ensemble(runs: List[List[str]], modifier: Dict[str, float]) -> None:
    print()
    print("=" * 70)
    print(f"ENSEMBLE ({len(runs)} scratch generations, kernel-weighted consensus)")
    print("=" * 70)
    for i, run in enumerate(runs, start=1):
        preview = NGramModel.detokenize(run[:20])
        print(f"\n[run {i}] {preview}{' ...' if len(run) > 20 else ''}")
    top_survivors = sorted(modifier.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\nSurviving tokens after kernel/isomorphism consensus (top 15):")
    if not top_survivors:
        print("  (none survived — runs disagreed on everything)")
    for token, weight in top_survivors:
        print(f"  {token!r:<15} weight={weight:.3f}")


def display_non_conjoint(pairs: List[Tuple[str, str, float]]) -> None:
    print()
    print("=" * 70)
    print("NON-CONJOINT CONTEXTS (rhombus-selected lateral band)")
    print("=" * 70)
    if not pairs:
        print("No fully non-conjoint context pairs found in this band.")
        return
    for a, b, score in pairs[:15]:
        print(f"  {a!r:<25} <-/-> {b!r:<25} disjointness={score:.3f}")


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
    iso_classes = len(set(model.isomorphism_map.values())) if model.isomorphism_map else 0
    print(f"Isomorphism classes: {iso_classes} (from {len(model.isomorphism_map)} vocab tokens)")

    non_conjoint_pairs = model.non_conjoint_lateral_report(top_n=30, radius=2, threshold=0.999)
    display_non_conjoint(non_conjoint_pairs)

    search = CorpusSearch(lexical_weight=LEXICAL_WEIGHT, vector_weight=VECTOR_WEIGHT)
    search.build_index(corpus_text)  # built once, not on every turn

    while True:
        prompt = input("\nUSER: ").strip()
        if not prompt:
            print("Empty prompt.")
            continue

        candidates = search.analyze(prompt, limit=CANDIDATE_LIMIT)
        display_candidates(candidates)

        print(f"\nRunning {NUM_GENERATIONS} scratch generations for kernelized consensus...")
        final_text, runs, modifier = model.generate_consensus(
            prompt,
            num_generations=NUM_GENERATIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            kernel_sharpness=KERNEL_SHARPNESS,
            modifier_weight=MODIFIER_WEIGHT,
        )
        display_ensemble(runs, modifier)

        print("\nGenerating final (modifier-biased)...")
        display_generation(final_text)


if __name__ == "__main__":
    main()
