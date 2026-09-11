from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

MODEL_PATH = "model.json"

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 120

MIN_COUNT = 1
INFLUENCE_TAU = 0.7

CURVE_K = 18.0
CURVE_MIDPOINT = 0.5

CANDIDATE_LIMIT = 15
LEXICAL_WEIGHT = 0.85
VECTOR_WEIGHT = 0.55

NUM_GENERATIONS = 5
KERNEL_SHARPNESS = 0.1
MODIFIER_WEIGHT = 0.6
CONSENSUS_BASELINE_SPLIT = 0.5

ISOMORPHISM_TAU = 0.07
ISOMORPHISM_SHARPNESS = 0.5

CONTEXT_REDUCTION_FRACTION = 0.9
CONTEXT_REDUCTION_MAX_DIMS = 2000

TRANSITIVITY_DECAY = 0.1
TRANSITIVITY_WEIGHT = 0.1
TRANSITIVITY_MASK_PENALTY = 1.0
TRANSITIVITY_SUPERPOLY_K = 3.0

INSTRUCTION_COMPOUND_CURVE_K = 12.0
INSTRUCTION_COMPOUND_MIDPOINT = 0.3
INSTRUCTION_COMPOUND_GROWTH = 0.15
INSTRUCTION_COMPOUND_CAP = 4.0
INSTRUCTION_COMPOUND_WEIGHT = 0.4

INVERSION_GATE_TOKENS = 60
RANDOM_SEED = None
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

TOKEN_RE = re.compile(r"[A-Za-z0-9_\']+|[.,!?;:()\[\]{}\-]")
IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}


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


class CUDANGramBackend:
    """Dense tensor backend for the original NGramModel equations."""

    def finalize_cuda(self, model):
        self.model = model
        self.vocab = list(model.vocabulary)
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.V = len(self.vocab)
        contexts = sorted(model.bigram.keys())
        self.contexts = contexts
        self.context_index = {c: i for i, c in enumerate(contexts)}
        C = len(contexts)

        lexical = torch.zeros((self.V, C), dtype=DTYPE, device=DEVICE)
        for token, vec in model.lexical_vectors.items():
            i = self.index.get(token)
            if i is None:
                continue
            for ctx, value in vec.items():
                j = self.context_index.get(ctx)
                if j is not None:
                    lexical[i, j] = float(value)
        self.lexical = lexical
        self.lexical_norm = torch.linalg.vector_norm(lexical, dim=1).clamp_min(1e-12)

        sims = (lexical @ lexical.T) / (self.lexical_norm[:, None] * self.lexical_norm[None, :])
        sims.fill_diagonal_(0.0)
        self.influence = torch.where(sims >= model.influence_tau, sims, torch.zeros_like(sims))

        self.bigram_probs = torch.zeros((self.V, self.V), dtype=DTYPE, device=DEVICE)
        self.trigram_probs = {}
        for left, counts in model.bigram.items():
            i = self.index.get(left)
            if i is not None:
                ids = [self.index[t] for t in counts if t in self.index]
                vals = torch.tensor([counts[t] for t in counts if t in self.index], dtype=DTYPE, device=DEVICE)
                if ids:
                    self.bigram_probs[i, torch.tensor(ids, device=DEVICE)] = vals / vals.sum()
        for key, counts in model.trigram.items():
            ids = [self.index[t] for t in counts if t in self.index]
            if ids:
                vals = torch.tensor([counts[t] for t in counts if t in self.index], dtype=DTYPE, device=DEVICE)
                self.trigram_probs[key] = (torch.tensor(ids, device=DEVICE), vals / vals.sum())

        unigram_ids = [self.index[t] for t in model.unigram if t in self.index]
        unigram_vals = torch.tensor([model.unigram[t] for t in model.unigram if t in self.index], dtype=DTYPE, device=DEVICE)
        self.unigram_probs = torch.zeros(self.V, dtype=DTYPE, device=DEVICE)
        self.unigram_probs[torch.tensor(unigram_ids, device=DEVICE)] = unigram_vals / unigram_vals.sum()
        self.eos_id = self.index.get(model.eos_token, -1)
        return self

    def _distribution(self, prev: str, prev_prev: Optional[str]):
        key = f"{prev_prev}\t{prev}" if prev_prev is not None else None
        if key in self.trigram_probs:
            ids, probs = self.trigram_probs[key]
            return ids, probs
        i = self.index.get(prev)
        if i is not None:
            ids = torch.nonzero(self.bigram_probs[i] > 0, as_tuple=False).flatten()
            if ids.numel():
                return ids, self.bigram_probs[i, ids]
        ids = torch.nonzero(self.unigram_probs > 0, as_tuple=False).flatten()
        return ids, self.unigram_probs[ids]

    def score_next(self, prev: str, prev_prev: Optional[str], temperature=0.8,
                   candidate_modifier=None, modifier_weight=0.0,
                   instruction_vector=None, instruction_weight=0.0,
                   compound_factor=1.0):
        ids, base = self._distribution(prev, prev_prev)
        if ids.numel() == 0:
            return ids, ids.to(DTYPE)
        p_eos = base[ids == self.eos_id].sum() if self.eos_id >= 0 else torch.zeros((), device=DEVICE)
        curve = torch.sigmoid(-self.model.curve_k * (p_eos - self.model.curve_midpoint))
        prev_i = self.index.get(prev, -1)
        source = self.lexical[prev_i] if prev_i >= 0 else torch.zeros(self.lexical.shape[1], device=DEVICE)
        target = self.lexical[ids]
        sim = (target @ source) / (torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12) * torch.linalg.vector_norm(source).clamp_min(1e-12))
        influence = self.influence[prev_i, ids] if prev_i >= 0 else torch.zeros_like(sim)
        scores = torch.log(torch.clamp(base, min=1e-12)) + curve * (0.35 * sim + 0.65 * influence)

        if candidate_modifier and modifier_weight:
            bias = torch.tensor([candidate_modifier.get(self.model._canonical(self.vocab[i]), 0.0) for i in ids.tolist()], dtype=DTYPE, device=DEVICE)
            scores = scores + modifier_weight * (0.5 * bias + 0.5 * base)
        if instruction_vector is not None and instruction_weight:
            iv = instruction_vector
            likeness = (target @ iv) / (torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12) * torch.linalg.vector_norm(iv).clamp_min(1e-12))
            curved = torch.sigmoid(INSTRUCTION_COMPOUND_CURVE_K * (likeness - INSTRUCTION_COMPOUND_MIDPOINT))
            scores = scores + instruction_weight * compound_factor * curved
        return ids, torch.softmax(scores / max(float(temperature), 1e-5), dim=-1)

    @torch.inference_mode()
    def sample_next(self, prompt_tokens: List[str], temperature=0.8, top_k=20, **kwargs):
        prev = prompt_tokens[-1] if prompt_tokens else "<bos>"
        prev_prev = prompt_tokens[-2] if len(prompt_tokens) > 1 else None
        ids, probs = self.score_next(prev, prev_prev, temperature, **kwargs)
        if ids.numel() == 0:
            return self.model.eos_token
        k = min(int(top_k), ids.numel())
        vals, pos = torch.topk(probs, k)
        chosen = pos[torch.multinomial(vals / vals.sum(), 1)]
        return self.vocab[int(ids[chosen].item())]

    @torch.inference_mode()
    def consensus_modifier(self, runs: List[List[str]], sharpness=0.1):
        if not runs:
            return {}
        counters = [Counter(self.model._canonical(t) for t in run) for run in runs]
        keys = sorted(set().union(*(c.keys() for c in counters)))
        mat = torch.zeros((len(counters), len(keys)), dtype=DTYPE, device=DEVICE)
        ki = {k: i for i, k in enumerate(keys)}
        for r, c in enumerate(counters):
            for token, count in c.items():
                mat[r, ki[token]] = count
        d2 = torch.cdist(mat, mat, p=2).square()
        nonzero = d2[d2 > 0]
        gamma = 1.0 / (2.0 * torch.median(nonzero)) if nonzero.numel() else torch.ones((), device=DEVICE)
        agreement = (torch.exp(-gamma * d2).sum(1) - 1.0) / max(len(counters) - 1, 1)
        weights = agreement.clamp_min(0).pow(sharpness)
        weights = weights / weights.sum().clamp_min(1e-12)
        modifier = (weights[:, None] * mat).sum(0)
        modifier = modifier / modifier.max().clamp_min(1e-12)
        return {k: float(v) for k, v in zip(keys, modifier.detach().cpu())}


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

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    isomorphism_map: Dict[str, str] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

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

        self._reduce_context_dimensions()

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

    def _reduce_context_dimensions(
        self,
        fraction: Optional[float] = None,
        max_dims: Optional[int] = None,
    ) -> None:
        fraction = self.context_reduction_fraction if fraction is None else fraction
        max_dims = self.context_reduction_max_dims if max_dims is None else max_dims

        context_totals: Counter = Counter()
        for vec in self.lexical_vectors.values():
            for ctx, w in vec.items():
                context_totals[ctx] += w

        if len(context_totals) <= 1:
            return

        ranked = [c for c, _ in context_totals.most_common()]
        active = ranked[:max_dims]

        target_count = max(1, math.ceil(len(active) * fraction))
        if target_count >= len(active):
            return

        columns: Dict[str, Dict[str, float]] = {c: {} for c in active}
        for token, vec in self.lexical_vectors.items():
            for ctx, w in vec.items():
                if ctx in columns:
                    columns[ctx][token] = w

        groups: Dict[str, List[str]] = {c: [c] for c in active}
        group_vecs: Dict[str, Dict[str, float]] = dict(columns)

        while len(groups) > target_count:
            ids = list(groups.keys())
            best_pair, best_sim = None, -1.0
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    sim = cosine_similarity(group_vecs[ids[i]], group_vecs[ids[j]])
                    if sim > best_sim:
                        best_sim, best_pair = sim, (ids[i], ids[j])
            if best_pair is None:
                break
            a, b = best_pair
            merged_name = f"{a}+{b}"
            merged_vec: Dict[str, float] = defaultdict(float)
            for tok, w in group_vecs[a].items():
                merged_vec[tok] += w
            for tok, w in group_vecs[b].items():
                merged_vec[tok] += w
            groups[merged_name] = groups.pop(a) + groups.pop(b)
            group_vecs.pop(a)
            group_vecs.pop(b)
            group_vecs[merged_name] = dict(merged_vec)

        remap: Dict[str, str] = {
            member: gid for gid, members in groups.items() for member in members
        }

        new_vectors: Dict[str, Dict[str, float]] = {}
        for token, vec in self.lexical_vectors.items():
            new_vec: Dict[str, float] = defaultdict(float)
            for ctx, w in vec.items():
                new_vec[remap.get(ctx, ctx)] += w
            new_vectors[token] = dict(new_vec)
        self.lexical_vectors = new_vectors

    def _build_isomorphism_classes(self) -> None:
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
                raw_sim = cosine_similarity(vec_a, vec_b)
                sharpened_sim = raw_sim ** self.isomorphism_sharpness
                if sharpened_sim >= self.isomorphism_tau:
                    canonical[tok_b] = tok_a
                    assigned.add(tok_b)
        self.isomorphism_map = canonical

    def _canonical(self, token: str) -> str:
        return self.isomorphism_map.get(token, token)

    def _canonicalize_run(self, run: List[str]) -> Counter:
        return Counter(self._canonical(t) for t in run)

    @staticmethod
    def _sigmoid_curve(x: float, k: float, midpoint: float) -> float:
        return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))

    def _instruction_vector(self, prompt: str) -> Dict[str, float]:
        tokens = [t for t in tokenize(prompt) if t not in IGNORED_TOKENS]
        agg: Dict[str, float] = defaultdict(float)
        for t in tokens:
            vec = self.lexical_vectors.get(t, {})
            for ctx, w in vec.items():
                agg[ctx] += w
        return dict(agg)

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
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_factor: float = 1.0,
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
                consensus_weight = candidate_modifier.get(self._canonical(token), 0.0)
                baseline_prob = base.get(token, 0.0)
                blended_bias = (
                    CONSENSUS_BASELINE_SPLIT * consensus_weight
                    + (1.0 - CONSENSUS_BASELINE_SPLIT) * baseline_prob
                )
                score += modifier_weight * blended_bias
            if transitivity_mask is not None and transitivity_weight:
                mask_weight = transitivity_mask.get(token)
                if mask_weight is not None:
                    score += transitivity_weight * (math.exp(superpoly_k * mask_weight) - 1.0)
                else:
                    score -= transitivity_weight * (math.exp(superpoly_k * mask_penalty) - 1.0)
            if instruction_vector and instruction_weight:
                likeness = cosine_similarity(
                    instruction_vector, self.lexical_vectors.get(token, {})
                )
                curved_likeness = self._sigmoid_curve(
                    likeness, INSTRUCTION_COMPOUND_CURVE_K, INSTRUCTION_COMPOUND_MIDPOINT
                )
                score += instruction_weight * compound_factor * curved_likeness
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
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_factor: float = 1.0,
    ) -> Dict[str, float]:
        scores = self._score_next_token(
            prompt,
            candidate_limit,
            candidate_modifier,
            modifier_weight,
            transitivity_mask,
            transitivity_weight,
            instruction_vector=instruction_vector,
            instruction_weight=instruction_weight,
            compound_factor=compound_factor,
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
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_factor: float = 1.0,
    ) -> str:
        probs = self._probabilities(
            prompt,
            temperature,
            max(top_k, 1),
            candidate_modifier,
            modifier_weight,
            transitivity_mask,
            transitivity_weight,
            instruction_vector=instruction_vector,
            instruction_weight=instruction_weight,
            compound_factor=compound_factor,
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
        instruction_weight: float = 0.0,
        compound_growth: float = INSTRUCTION_COMPOUND_GROWTH,
        compound_cap: float = INSTRUCTION_COMPOUND_CAP,
        inversion_gate_tokens: Optional[int] = INVERSION_GATE_TOKENS,
    ) -> str:
        generated = tokenize(prompt)

        # Fixed instruction target, built once from the original prompt
        # (never recomputed from the growing `generated` list). Off by
        # default (instruction_weight=0.0) -- callers that want the
        # compounding pull toward the instruction (generate_consensus's
        # final pass) opt in explicitly.
        instruction_vector = self._instruction_vector(prompt) if instruction_weight else None
        compound_factor = 1.0

        for step in range(max_new_tokens):
            # Length-gated inversion: once `step` reaches the gate, the
            # instruction-likeness term (below) flips sign for every
            # subsequent token. Set inversion_gate_tokens=None to disable
            # gating entirely and keep the original toward-instruction
            # behavior for the whole run.
            invert = inversion_gate_tokens is not None and step >= inversion_gate_tokens
            token = self.sample_next(
                " ".join(generated),
                temperature,
                top_k,
                candidate_modifier,
                modifier_weight,
                transitivity_mask,
                transitivity_weight,
                instruction_vector=instruction_vector,
                instruction_weight=instruction_weight,
                compound_factor=compound_factor,
                invert_instruction=invert,
            )
            generated.append(token)
            if instruction_vector and instruction_weight:
                likeness = cosine_similarity(
                    instruction_vector, self.lexical_vectors.get(token, {})
                )
                curved_likeness = self._sigmoid_curve(
                    likeness, INSTRUCTION_COMPOUND_CURVE_K, INSTRUCTION_COMPOUND_MIDPOINT
                )
                compound_factor = min(
                    compound_cap, compound_factor * (1.0 + compound_growth * curved_likeness)
                )
        # <bos>/<eos>/<unk> stay in `generated` for scoring purposes
        # (backoff distributions and the curve weight depend on seeing
        # them) but are stripped before the text is ever shown to the user.
        return self.detokenize(_strip_structural_tokens(generated))

    def generate_with_trace(
        self, prompt: str, max_new_tokens: int, temperature: float, top_k: int
    ) -> Tuple[str, List[str]]:
        # Deliberately unbiased: no candidate_modifier, no transitivity
        # mask, no instruction-compounding curve. These scratch runs need
        # to stay independent and diverse -- the kernel-consensus step
        # downstream is only meaningful if they disagree where the model
        # is genuinely uncertain, rather than all being pulled toward the
        # same instruction target ahead of time.
        generated = tokenize(prompt)
        start = len(generated)
        for _ in range(max_new_tokens):
            token = self.sample_next(" ".join(generated), temperature, top_k)
            generated.append(token)
        # Return the cleaned text for display, but the raw (unstripped)
        # new-token slice for the consensus/canonicalization machinery,
        # which needs <eos>/<bos> to correctly detect run boundaries.
        return self.detokenize(_strip_structural_tokens(generated)), generated[start:]
    def multi_generate(
        self,
        prompt: str,
        num_generations: int = NUM_GENERATIONS,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> List[List[str]]:
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
            gamma = 1.0

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
        instruction_weight: float = INSTRUCTION_COMPOUND_WEIGHT,
        compound_growth: float = INSTRUCTION_COMPOUND_GROWTH,
        compound_cap: float = INSTRUCTION_COMPOUND_CAP,
        inversion_gate_tokens: Optional[int] = INVERSION_GATE_TOKENS,
    ) -> Tuple[str, List[List[str]], Dict[str, float]]:
        if not self.finalized:
            self.finalize()
        # Scratch runs stay unbiased (see generate_with_trace's docstring).
        runs = self.multi_generate(prompt, num_generations, max_new_tokens, temperature, top_k)
        modifier = self.build_kernel_consensus_modifier(runs, kernel_sharpness)
        # Only the final generation gets the consensus modifier, the
        # compounding instruction-likeness curve, AND the length-gated
        # inversion of that curve, stacked together.
        final_text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            candidate_modifier=modifier,
            modifier_weight=modifier_weight,
            instruction_weight=instruction_weight,
            compound_growth=compound_growth,
            compound_cap=compound_cap,
            inversion_gate_tokens=inversion_gate_tokens,
        )
        return final_text, runs, modifier


    def build_disjointness_matrix(self, top_n: int = 30) -> Tuple[List[str], List[List[float]]]:
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
        contexts, matrix = self.build_disjointness_matrix(top_n)
        lateral = self.rhombus_select_lateral(matrix, radius)
        pairs = [
            (contexts[i], contexts[j], score)
            for (i, j), score in lateral.items()
            if score >= threshold
            and contexts[i] not in IGNORED_TOKENS
            and contexts[j] not in IGNORED_TOKENS
        ]
        pairs.sort(key=lambda t: -t[2])
        return pairs

    def transitive_successors(self, token: str, decay: float = 0.5) -> Dict[str, float]:
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
            "context_reduction_fraction": self.context_reduction_fraction,
            "context_reduction_max_dims": self.context_reduction_max_dims,
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
            isomorphism_sharpness=data.get("isomorphism_sharpness", ISOMORPHISM_SHARPNESS),
            context_reduction_fraction=data.get(
                "context_reduction_fraction", CONTEXT_REDUCTION_FRACTION
            ),
            context_reduction_max_dims=data.get(
                "context_reduction_max_dims", CONTEXT_REDUCTION_MAX_DIMS
            ),
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
            model._build_isomorphism_classes()
        else:
            model.isomorphism_map = {}
        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_ontology(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "concepts" not in data:
        raise ValueError(f"{path} does not look like a terminology_ontology_pipeline.py result")
    return data


def select_concept(ontology: dict, prompt: str) -> dict:
    prompt_tokens = tokenize(prompt)
    best, best_score = None, -1.0
    for concept in ontology["concepts"]:
        vocab_tokens = []
        for t in concept["terms"]:
            vocab_tokens.extend(t["term"].split())
        for p in concept["properties"]:
            vocab_tokens.append(p["word"])
        score = lexical_overlap(prompt_tokens, vocab_tokens)
        if score > best_score:
            best, best_score = concept, score
    if best is None or best_score <= 0.0:
        best = max(ontology["concepts"], key=lambda c: len(c["terms"]))
    return best


def concept_vocab_modifier(concept: dict) -> Dict[str, float]:
    weights: Dict[str, float] = defaultdict(float)
    for t in concept["terms"]:
        typ = t.get("typicality", 0.5)
        for word in t["term"].split():
            weights[word] = max(weights[word], typ)
    props = concept.get("properties", [])
    max_g2 = max((p["g2"] for p in props), default=1.0) or 1.0
    for p in props:
        norm_g2 = p["g2"] / max_g2
        weights[p["word"]] = max(weights[p["word"]], norm_g2)
    if not weights:
        return {}
    max_val = max(weights.values())
    return {w: v / max_val for w, v in weights.items()} if max_val > 0 else dict(weights)


def _strip_structural_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in IGNORED_TOKENS]


def generate_list(
    model: "NGramModel",
    concept: dict,
    modifier: Dict[str, float],
    modifier_weight: float,
    num_items: int,
    max_new_tokens_per_item: int,
    temperature: float,
    top_k: int,
    prompt: str = "",
    instruction_weight: float = INSTRUCTION_COMPOUND_WEIGHT,
    compound_growth: float = INSTRUCTION_COMPOUND_GROWTH,
    compound_cap: float = INSTRUCTION_COMPOUND_CAP,
) -> List[Tuple[str, str]]:
    ranked_terms = sorted(concept["terms"], key=lambda t: t["typicality"], reverse=True)
    seeds = [t["term"] for t in ranked_terms[:num_items]]
    while len(seeds) < num_items and ranked_terms:
        seeds.append(ranked_terms[len(seeds) % len(ranked_terms)]["term"])
    instruction_vector = (
        model._instruction_vector(prompt) if (prompt and instruction_weight) else None
    )
    items = []
    for seed in seeds:
        generated_tokens = tokenize(seed)
        compound_factor = 1.0
        for _ in range(max_new_tokens_per_item):
            next_tok = model.sample_next(
                " ".join(generated_tokens),
                temperature=temperature,
                top_k=top_k,
                candidate_modifier=modifier,
                modifier_weight=modifier_weight,
                instruction_vector=instruction_vector,
                instruction_weight=instruction_weight,
                compound_factor=compound_factor,
            )
            if next_tok == model.eos_token:
                break
            generated_tokens.append(next_tok)
            if instruction_vector and instruction_weight:
                likeness = cosine_similarity(
                    instruction_vector, model.lexical_vectors.get(next_tok, {})
                )
                curved_likeness = NGramModel._sigmoid_curve(
                    likeness, INSTRUCTION_COMPOUND_CURVE_K, INSTRUCTION_COMPOUND_MIDPOINT
                )
                compound_factor = min(
                    compound_cap, compound_factor * (1.0 + compound_growth * curved_likeness)
                )
        item_text = NGramModel.detokenize(_strip_structural_tokens(generated_tokens))
        items.append((seed, item_text))
    return items


def display_list(concept: dict, items: List[Tuple[str, str]]) -> None:
    print()
    print("=" * 70)
    print(f"GENERATED LIST -- concept {concept['concept_id']} "
          f"(parent {concept['parent_concept_id']})")
    print("=" * 70)
    for i, (seed, text) in enumerate(items, start=1):
        print(f"  {i}. [{seed}] {text}")


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
        preview = NGramModel.detokenize(_strip_structural_tokens(run[:20]))
        print(f"\n[run {i}] {preview}{' ...' if len(run) > 20 else ''}")
    visible_modifier = {t: w for t, w in modifier.items() if t not in IGNORED_TOKENS}
    top_survivors = sorted(visible_modifier.items(), key=lambda kv: kv[1], reverse=True)[:15]
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="N-gram consensus generator, optionally biased by a "
                     "terminology_ontology_pipeline.py result for list generation.")
    parser.add_argument("--corpus", type=str, default=None,
                         help="Corpus filename (skips the interactive prompt if given).")
    parser.add_argument("--ontology", type=str, default=None,
                         help="Path to a terminology_ontology_pipeline.py JSON result. "
                              "When set, each prompt also produces a generated list "
                              "from the best-matching concept's controlled vocabulary.")
    parser.add_argument("--list-items", type=int, default=500,
                         help="Number of list items to generate per prompt.")
    parser.add_argument("--list-tokens", type=int, default=120,
                         help="Max tokens generated per list item.")
    parser.add_argument("--list-modifier-weight", type=float, default=MODIFIER_WEIGHT,
                         help="How strongly the concept vocabulary biases list generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_path = Path(args.corpus) if args.corpus else Path(input("Filename: "))
    if not corpus_path.exists():
        print(f"\nERROR: {corpus_path} does not exist.")
        return

    ontology = None
    if args.ontology:
        try:
            ontology = load_ontology(args.ontology)
            print(f"\nLoaded ontology: {len(ontology['concepts'])} concepts "
                  f"from {args.ontology}")
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"\nERROR loading ontology: {e}")
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
    search.build_index(corpus_text)

    # Initialize CUDA backend after model is finalized
    print("\nInitializing CUDA backend...")
    cuda_model = CUDANGramBackend()
    cuda_model.finalize_cuda(model)
    print(f"Running on device: {DEVICE}")

    while True:
        prompt = input("\nUSER: ").strip()
        if not prompt:
            print("Empty prompt.")
            continue

        candidates = search.analyze(prompt, limit=CANDIDATE_LIMIT)
        display_candidates(candidates)

        print(f"\nRunning {NUM_GENERATIONS} scratch generations for kernelized consensus...")
        # Use CUDA consensus modifier
        runs = model.multi_generate(prompt, NUM_GENERATIONS, MAX_NEW_TOKENS, TEMPERATURE, TOP_K)
        modifier = cuda_model.consensus_modifier(runs, KERNEL_SHARPNESS)
        display_ensemble(runs, modifier)

        print("\nGenerating final (modifier-biased, compounding toward instruction)...")
        # Use CUDA sampling for final generation
        generated = tokenize(prompt)
        instruction_vector = model._instruction_vector(prompt)
        compound_factor = 1.0
        for _ in range(MAX_NEW_TOKENS):
            token = cuda_model.sample_next(
                generated,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                candidate_modifier=modifier,
                modifier_weight=MODIFIER_WEIGHT,
                instruction_vector=torch.tensor(list(instruction_vector.values()), dtype=DTYPE, device=DEVICE) if instruction_vector else None,
                instruction_weight=INSTRUCTION_COMPOUND_WEIGHT,
                compound_factor=compound_factor,
            )
            generated.append(token)
            if instruction_vector:
                tok_vec = model.lexical_vectors.get(token, {})
                likeness = cosine_similarity(instruction_vector, tok_vec)
                curved_likeness = NGramModel._sigmoid_curve(
                    likeness, INSTRUCTION_COMPOUND_CURVE_K, INSTRUCTION_COMPOUND_MIDPOINT
                )
                compound_factor = min(
                    INSTRUCTION_COMPOUND_CAP, compound_factor * (1.0 + INSTRUCTION_COMPOUND_GROWTH * curved_likeness)
                )
        final_text = model.detokenize(_strip_structural_tokens(generated))
        display_generation(final_text)

        if ontology is not None:
            concept = select_concept(ontology, prompt)
            list_modifier = concept_vocab_modifier(concept)
            items = generate_list(
                model, concept, list_modifier,
                modifier_weight=args.list_modifier_weight,
                num_items=args.list_items,
                max_new_tokens_per_item=args.list_tokens,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                prompt=prompt,
            )
            display_list(concept, items)


if __name__ == "__main__":
    main()
