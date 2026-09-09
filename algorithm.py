"""
neural_text_generator_cuda_features.py
======================================
One-piece pipeline: read a plain-text dataset, extract hand-tuned
math from the original n-gram/consensus generator as raw FEATURES,
train a small neural layer on the dataset's own bigram statistics,
and generate new text using corpus probabilities only.

CHANGES vs. original
--------------------
- All neural network code runs on CUDA when available (strict mode).
- The neural layer no longer scores tokens or modifies logits.
- The network outputs feature embeddings only.
- Token sampling uses corpus bigram/unigram probabilities directly.
- Legacy scoring, bias computation, and score bootstrapping removed.
- Syntax and indentation fully corrected.
- Training targets now match network output shape [batch, output_dim].

Usage
-----
python neural_text_generator_cuda_features.py --corpus mytext.txt --train-steps 0
python neural_text_generator_cuda_features.py --corpus mytext.txt --max-tokens 120 --temperature 0.9 --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\[\]{}\-]")
BOS, EOS = "<BOS>", "<EOS>"


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and ensure drivers are working."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def detokenize(tokens: List[str]) -> str:
    text = " ".join(t for t in tokens if t not in (BOS, EOS))
    text = re.sub(r"\s+([.,!?;:)\]}])", r"\1", text)
    text = re.sub(r"([({\[<])\s+", r"\1", text)
    return text


def cosine_similarity(
    a: Dict[str, float], b: Dict[str, float]
) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def sigmoid_curve(x: float, k: float, midpoint: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))


# --- isomorphism / context-reduction / kernel-consensus settings ---
ISOMORPHISM_TAU = 0.07
ISOMORPHISM_SHARPNESS = 0.5
CONTEXT_REDUCTION_FRACTION = 0.9
CONTEXT_REDUCTION_MAX_DIMS = 2000
NUM_GENERATIONS = 5
KERNEL_SHARPNESS = 0.1


# ============================================================
# 1. Feature definitions
# ============================================================

FEATURE_NAMES: List[str] = [
    "log_base_prob",
    "p_eos",
    "lexical_cos_prev",
    "influence_raw",
    "influence_above_tau",
    "consensus_modifier_raw",
    "baseline_prob",
    "kernel_vote_count",
    "iso_class_size",
    "transitivity_mask_raw",
    "transitivity_present",
    "instruction_cos_raw",
    "compound_factor",
    "unigram_log_freq",
]

FEATURE_DIM = len(FEATURE_NAMES)  # 14


@dataclass
class FeatureContext:
    base: Dict[str, float]
    source_vec: Dict[str, float]
    lexical_vectors: Dict[str, Dict[str, float]]
    influence_tau: float
    isomorphism_map: Dict[str, str]
    unigram: Dict[str, int]
    total_unigram: int
    candidate_modifier: Optional[Dict[str, float]] = None
    run_canonical_votes: Optional[Dict[str, int]] = None
    transitivity_mask: Optional[Dict[str, float]] = None
    instruction_vector: Optional[Dict[str, float]] = None
    compound_factor: float = 1.0


class FeatureExtractor:
    def _canonical(self, ctx: FeatureContext, token: str) -> str:
        return ctx.isomorphism_map.get(token, token)

    def _iso_class_size(self, ctx: FeatureContext, token: str) -> int:
        if not ctx.isomorphism_map:
            return 1
        counts: Dict[str, int] = {}
        for v in ctx.isomorphism_map.values():
            counts[v] = counts.get(v, 0) + 1
        return counts.get(self._canonical(ctx, token), 1)

    def extract(self, ctx: FeatureContext, token: str) -> np.ndarray:
        canon = self._canonical(ctx, token)
        tok_vec = ctx.lexical_vectors.get(token, {})

        log_base_prob = math.log(max(ctx.base.get(token, 0.0), 1e-12))
        p_eos = ctx.base.get(EOS, 0.0)

        lexical_cos = cosine_similarity(ctx.source_vec, tok_vec)
        influence_raw = lexical_cos
        influence_above_tau = (
            1.0 if influence_raw >= ctx.influence_tau else 0.0
        )

        consensus_modifier_raw = (
            ctx.candidate_modifier.get(canon, 0.0)
            if ctx.candidate_modifier
            else 0.0
        )
        baseline_prob = ctx.base.get(token, 0.0)

        kernel_vote_count = (
            float(ctx.run_canonical_votes.get(canon, 0))
            if ctx.run_canonical_votes
            else 0.0
        )
        iso_class_size = float(self._iso_class_size(ctx, token))

        transitivity_mask_raw = 0.0
        transitivity_present = 0.0
        if ctx.transitivity_mask is not None:
            v = ctx.transitivity_mask.get(token)
            if v is not None:
                transitivity_mask_raw = v
                transitivity_present = 1.0

        instruction_cos_raw = (
            cosine_similarity(ctx.instruction_vector, tok_vec)
            if ctx.instruction_vector is not None
            else 0.0
        )

        unigram_log_freq = math.log(
            max(ctx.unigram.get(token, 0), 1) / max(ctx.total_unigram, 1)
        )

        return np.array(
            [
                log_base_prob,
                p_eos,
                lexical_cos,
                influence_raw,
                influence_above_tau,
                consensus_modifier_raw,
                baseline_prob,
                kernel_vote_count,
                iso_class_size,
                transitivity_mask_raw,
                transitivity_present,
                instruction_cos_raw,
                ctx.compound_factor,
                unigram_log_freq,
            ],
            dtype=np.float64,
        )

    def extract_batch(
        self, ctx: FeatureContext, tokens: List[str]
    ) -> np.ndarray:
        return np.stack([self.extract(ctx, t) for t in tokens])


# ============================================================
# 2. CUDA-safe device resolution (strict)
# ============================================================


def resolve_device(requested: Optional[str] = None) -> torch.device:
    requested = (requested or "cuda").lower()

    if requested == "cpu":
        raise RuntimeError(
            "This build does not support CPU. Use --device cuda."
        )

    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and ensure drivers are working."
            )
        return torch.device("cuda")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise RuntimeError(
                "Device='auto' but no CUDA device found. "
                "Install a CUDA-enabled PyTorch build or use --device cuda on a GPU machine."
            )

    raise ValueError(
        f"Unknown device '{requested}'. Use 'auto', 'cuda', or 'cpu'."
    )


# ============================================================
# 3. Neural feature layer (CUDA, no scoring)
# ============================================================


class NeuralFeatureLayer:
    """
    CUDA-accelerated feature projector.

    This module does not score tokens, rank candidates, produce sampling
    logits, or make decisions. It converts feature vectors into learned
    feature embeddings.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 32,
        output_dim: int = 16,
        seed: int = 0,
        device: Optional[str] = None,
        lr: float = 1e-3,
        use_all_gpus: bool = True,
    ):
        self.device = resolve_device(device)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        base_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        ).to(self.device)

        self.gpu_count = (
            torch.cuda.device_count()
            if self.device.type == "cuda"
            else 0
        )

        self.multi_gpu = (
            use_all_gpus
            and self.device.type == "cuda"
            and self.gpu_count > 1
        )

        self.net = (
            nn.DataParallel(base_net)
            if self.multi_gpu
            else base_net
        )

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

    @property
    def _module(self) -> nn.Module:
        return (
            self.net.module
            if isinstance(self.net, nn.DataParallel)
            else self.net
        )

    @property
    def output_dim(self) -> int:
        # Last linear layer's output dimension
        return self._module[-2].out_features  # type: ignore[attr-defined]

    def forward(self, X: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            output = self.net(x)

        return output.detach().cpu().numpy()

    def extract_features(
        self,
        extractor: FeatureExtractor,
        ctx: FeatureContext,
        tokens: List[str],
    ) -> Dict[str, np.ndarray]:
        X = extractor.extract_batch(ctx, tokens)
        embeddings = self.forward(X)
        return dict(zip(tokens, embeddings))

    def extract_feature_matrix(
        self,
        extractor: FeatureExtractor,
        ctx: FeatureContext,
        tokens: List[str],
    ) -> np.ndarray:
        X = extractor.extract_batch(ctx, tokens)
        return self.forward(X)

    def train_step(
        self,
        X: np.ndarray,
        target: np.ndarray,
        lr: float = 1e-3,
    ) -> float:
        x = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        y = torch.as_tensor(
            target,
            dtype=torch.float32,
            device=self.device,
        )

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        output = self.net(x)
        loss = torch.mean((output - y) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(),
            max_norm=1.0,
        )
        self.optimizer.step()

        return float(loss.detach().cpu())


# ============================================================
# 4. Corpus statistics
# ============================================================


class CorpusModel:
    def __init__(
        self,
        isomorphism_tau: float = ISOMORPHISM_TAU,
        isomorphism_sharpness: float = ISOMORPHISM_SHARPNESS,
        context_reduction_fraction: float = CONTEXT_REDUCTION_FRACTION,
        context_reduction_max_dims: int = CONTEXT_REDUCTION_MAX_DIMS,
    ):
        self.unigram: Counter = Counter()
        self.bigram: Dict[str, Counter] = defaultdict(Counter)
        self.lexical_vectors: Dict[str, Dict[str, float]] = {}
        self.isomorphism_map: Dict[str, str] = {}
        self.isomorphism_tau = isomorphism_tau
        self.isomorphism_sharpness = isomorphism_sharpness
        self.context_reduction_fraction = context_reduction_fraction
        self.context_reduction_max_dims = context_reduction_max_dims

    def ingest(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)
            if not words:
                continue
            seq = [BOS] + words + [EOS]
            for tok in seq:
                self.unigram[tok] += 1
            for left, right in zip(seq, seq[1:]):
                self.bigram[left][right] += 1

    def finalize(self, build_isomorphisms: bool = True) -> None:
        token_contexts: Dict[str, Counter] = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {
            token: {
                c: v / (sum(counts.values()) or 1)
                for c, v in counts.items()
            }
            for token, counts in token_contexts.items()
        }

        if build_isomorphisms:
            self._reduce_context_dimensions()
            self._build_isomorphism_classes()

    def _reduce_context_dimensions(
        self,
        fraction: Optional[float] = None,
        max_dims: Optional[int] = None,
    ) -> None:
        fraction = (
            self.context_reduction_fraction
            if fraction is None
            else fraction
        )
        max_dims = (
            self.context_reduction_max_dims
            if max_dims is None
            else max_dims
        )

        context_totals: Counter = Counter()
        for vec in self.lexical_vectors.values():
            for ctx, w in vec.items():
                context_totals[ctx] += w

        if len(context_totals) <= 1:
            return

        active = [
            c for c, _ in context_totals.most_common()
        ][:max_dims]
        target_count = max(1, math.ceil(len(active) * fraction))
        if target_count >= len(active):
            return

        columns: Dict[str, Dict[str, float]] = {
            c: {} for c in active
        }
        for token, vec in self.lexical_vectors.items():
            for ctx, w in vec.items():
                if ctx in columns:
                    columns[ctx][token] = w

        groups: Dict[str, List[str]] = {
            c: [c] for c in active
        }
        group_vecs: Dict[str, Dict[str, float]] = dict(columns)

        while len(groups) > target_count:
            ids = list(groups.keys())
            best_pair = None
            best_sim = -1.0
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    sim = cosine_similarity(
                        group_vecs[ids[i]], group_vecs[ids[j]]
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (ids[i], ids[j])
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

        remap = {
            member: gid
            for gid, members in groups.items()
            for member in members
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
        vocab = sorted(self.lexical_vectors.keys())
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
                sharpened = (
                    cosine_similarity(vec_a, vec_b)
                    ** self.isomorphism_sharpness
                )
                if sharpened >= self.isomorphism_tau:
                    canonical[tok_b] = tok_a
                    assigned.add(tok_b)
        self.isomorphism_map = canonical

    def canonical(self, token: str) -> str:
        return self.isomorphism_map.get(token, token)

    def backoff_distribution(self, prev: str) -> Dict[str, float]:
        counts = self.bigram.get(prev) or self.unigram
        total = sum(counts.values())
        return (
            {t: c / total for t, c in counts.items()}
            if total
            else {}
        )

    def total_unigram(self) -> int:
        return sum(self.unigram.values())


def _sq_dist(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    return float(
        sum((a.get(k, 0) - b.get(k, 0)) ** 2 for k in keys)
    )


def unbiased_sample_next(
    model: CorpusModel,
    prompt: str,
    temperature: float,
    top_k: int,
) -> str:
    tokens = tokenize(prompt)
    prev = tokens[-1] if tokens else BOS
    base = model.backoff_distribution(prev)
    if not base:
        return EOS
    top = sorted(
        base.items(), key=lambda kv: kv[1], reverse=True
    )[:top_k]
    toks, probs = zip(*top)
    scaled = np.log(np.array(probs)) / max(temperature, 1e-5)
    weights = np.exp(scaled - scaled.max())
    weights = weights / weights.sum()
    return random.choices(toks, weights=weights.tolist(), k=1)[0]


def unbiased_generate(
    model: CorpusModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> List[str]:
    generated = tokenize(prompt)
    start = len(generated)
    for _ in range(max_new_tokens):
        tok = unbiased_sample_next(
            model, " ".join(generated), temperature, top_k
        )
        generated.append(tok)
        if tok == EOS:
            break
    return generated[start:]


def multi_generate(
    model: CorpusModel,
    prompt: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> List[List[str]]:
    return [
        unbiased_generate(
            model, prompt, max_new_tokens, temperature, top_k
        )
        for _ in range(num_generations)
    ]


def build_kernel_consensus(
    model: CorpusModel,
    runs: List[List[str]],
    sharpness: float = KERNEL_SHARPNESS,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    if not runs:
        return {}, {}

    counters = [
        Counter(model.canonical(t) for t in run) for run in runs
    ]
    vote_counts: Dict[str, int] = defaultdict(int)
    for c in counters:
        for tok in c:
            vote_counts[tok] += 1

    n = len(counters)
    if n == 1:
        total = sum(counters[0].values()) or 1
        return (
            {t: c / total for t, c in counters[0].items()},
            dict(vote_counts),
        )

    sq_dists = [[0.0] * n for _ in range(n)]
    all_d2 = []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _sq_dist(counters[i], counters[j])
            sq_dists[i][j] = sq_dists[j][i] = d2
            all_d2.append(d2)

    nonzero = [d for d in all_d2 if d > 0]
    gamma = (
        1.0 / (2.0 * sorted(nonzero)[len(nonzero) // 2])
        if nonzero
        else 1.0
    )

    agreement = []
    for i in range(n):
        sims = [
            math.exp(-gamma * sq_dists[i][j])
            for j in range(n)
            if j != i
        ]
        agreement.append(
            sum(sims) / len(sims) if sims else 0.0
        )

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
        return {}, dict(vote_counts)
    max_val = max(modifier.values())
    return (
        {t: v / max_val for t, v in modifier.items()},
        dict(vote_counts),
    )


def instruction_vector(
    model: CorpusModel, prompt: str
) -> Dict[str, float]:
    agg: Dict[str, float] = defaultdict(float)
    for tok in tokenize(prompt):
        for ctx, w in model.lexical_vectors.get(tok, {}).items():
            agg[ctx] += w
    return dict(agg)


# ============================================================
# 5. Self-supervised training data (target shape = [batch, output_dim])
# ============================================================


def make_training_batch(
    model: CorpusModel,
    extractor: FeatureExtractor,
    batch_size: int,
    candidate_limit: int,
    rng: random.Random,
    output_dim: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    contexts = [
        c for c in model.bigram.keys() if c != EOS
    ]
    if not contexts:
        raise ValueError(
            "Corpus too small to train on -- no bigram contexts found."
        )

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    total_unigram = model.total_unigram()
    attempts = 0

    while len(xs) < batch_size and attempts < batch_size * 10:
        attempts += 1
        prev = rng.choice(contexts)
        base = model.backoff_distribution(prev)
        if not base:
            continue

        actual = rng.choices(
            list(base.keys()),
            weights=list(base.values()),
            k=1,
        )[0]

        candidates = sorted(
            base, key=base.get, reverse=True
        )[:candidate_limit]
        if actual not in candidates:
            candidates.append(actual)
        negatives = [t for t in candidates if t != actual]
        if not negatives:
            continue
        negative = rng.choice(negatives)

        ctx = FeatureContext(
            base=base,
            source_vec=model.lexical_vectors.get(prev, {}),
            lexical_vectors=model.lexical_vectors,
            influence_tau=0.7,
            isomorphism_map=model.isomorphism_map,
            unigram=model.unigram,
            total_unigram=total_unigram,
        )

        xs.append(extractor.extract(ctx, actual))
        xs.append(extractor.extract(ctx, negative))

        # Positive -> ones, Negative -> zeros (shape [output_dim])
        ys.append(np.ones(output_dim, dtype=np.float32))
        ys.append(np.zeros(output_dim, dtype=np.float32))

    return np.stack(xs), np.stack(ys)


def train_feature_layer(
    feature_layer: NeuralFeatureLayer,
    extractor: FeatureExtractor,
    model: CorpusModel,
    steps: int,
    batch_size: int = 64,
    candidate_limit: int = 40,
    lr: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> None:
    rng = random.Random(seed)
    for step in tqdm(
        range(steps),
        desc="Training",
        disable=not verbose,
    ):
        X, y = make_training_batch(
            model,
            extractor,
            batch_size,
            candidate_limit,
            rng,
            output_dim=feature_layer.output_dim,
        )
        loss = feature_layer.train_step(X, y, lr=lr)
        if verbose:
            tqdm.write(
                f" train step {step:4d}/{steps} loss={loss:.4f}"
            )


# ============================================================
# 6. Generation (corpus-probability sampling, features only)
# ============================================================


def generate(
    model: CorpusModel,
    extractor: FeatureExtractor,
    feature_layer: NeuralFeatureLayer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 20,
    candidate_limit: int = 60,
    use_instruction_pull: bool = True,
    compound_growth: float = 0.15,
    compound_cap: float = 4.0,
    seed: Optional[int] = None,
    use_consensus: bool = True,
    num_generations: int = NUM_GENERATIONS,
    kernel_sharpness: float = KERNEL_SHARPNESS,
    verbose_features: bool = False,
) -> str:
    if seed is not None:
        random.seed(seed)

    generated = tokenize(prompt) if prompt.strip() else []
    prev = generated[-1] if generated else BOS

    instr_vec = (
        instruction_vector(model, prompt)
        if (prompt.strip() and use_instruction_pull)
        else None
    )
    compound_factor = 1.0

    modifier: Dict[str, float] = {}
    vote_counts: Dict[str, int] = {}
    if use_consensus and num_generations > 0:
        runs = multi_generate(
            model,
            prompt,
            num_generations,
            max_new_tokens,
            temperature,
            top_k,
        )
        modifier, vote_counts = build_kernel_consensus(
            model, runs, kernel_sharpness
        )

    for _ in range(max_new_tokens):
        base = model.backoff_distribution(prev)
        if not base:
            break
        candidates = sorted(
            base, key=base.get, reverse=True
        )[:candidate_limit]

        ctx = FeatureContext(
            base=base,
            source_vec=model.lexical_vectors.get(prev, {}),
            lexical_vectors=model.lexical_vectors,
            influence_tau=0.7,
            isomorphism_map=model.isomorphism_map,
            unigram=model.unigram,
            total_unigram=model.total_unigram(),
            candidate_modifier=modifier,
            run_canonical_votes=vote_counts,
            instruction_vector=instr_vec,
            compound_factor=compound_factor,
        )

        candidate_features = feature_layer.extract_features(
            extractor, ctx, candidates
        )

        if candidate_features and verbose_features:
            feature_matrix = np.stack(
                list(candidate_features.values()), axis=0
            )
            print(
                "feature matrix:",
                feature_matrix.shape,
                "device:",
                feature_layer.device,
            )

        top = sorted(
            base.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        if not top:
            break

        candidate_tokens, candidate_probs = zip(*top)
        candidate_probs = np.asarray(candidate_probs, dtype=np.float64)

        temperature = max(float(temperature), 1e-5)
        scaled = np.log(np.maximum(candidate_probs, 1e-12)) / temperature
        weights = np.exp(scaled - scaled.max())
        weights /= weights.sum()

        next_tok = random.choices(
            candidate_tokens,
            weights=weights.tolist(),
            k=1,
        )[0]

        if next_tok == EOS:
            break
        generated.append(next_tok)
        prev = next_tok

        if instr_vec is not None:
            likeness = cosine_similarity(
                instr_vec,
                model.lexical_vectors.get(next_tok, {}),
            )
            curved = sigmoid_curve(likeness, 12.0, 0.3)
            compound_factor = min(
                compound_cap,
                compound_factor * (1.0 + compound_growth * curved),
            )

    return detokenize(generated)


# ============================================================
# 7. CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate text from a .txt dataset using a "
            "CUDA feature-only neural layer."
        )
    )
    p.add_argument(
        "--corpus",
        required=True,
        help="Path to a plain-text dataset file.",
    )
    p.add_argument(
        "--prompt",
        default="",
        help="Seed text. Empty = start from scratch.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=60,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    p.add_argument(
        "--candidate-limit",
        type=int,
        default=60,
    )
    p.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="0 = skip training, use random init only.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--no-instruction-pull",
        action="store_true",
    )
    p.add_argument(
        "--no-consensus",
        action="store_true",
        help="Skip the kernel-consensus scratch-run ensemble.",
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=NUM_GENERATIONS,
        help="Scratch runs for consensus.",
    )
    p.add_argument(
        "--kernel-sharpness",
        type=float,
        default=KERNEL_SHARPNESS,
    )
    p.add_argument(
        "--isomorphism-tau",
        type=float,
        default=ISOMORPHISM_TAU,
    )
    p.add_argument(
        "--isomorphism-sharpness",
        type=float,
        default=ISOMORPHISM_SHARPNESS,
    )
    p.add_argument(
        "--context-reduction-fraction",
        type=float,
        default=CONTEXT_REDUCTION_FRACTION,
    )
    p.add_argument(
        "--context-reduction-max-dims",
        type=int,
        default=CONTEXT_REDUCTION_MAX_DIMS,
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' (default) or 'auto'. CPU not supported in this build.",
    )
    p.add_argument(
        "--single-gpu",
        action="store_true",
        help=(
            "Disable DataParallel; use only one GPU even if "
            "more are visible."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    p.add_argument(
        "--verbose-features",
        action="store_true",
        help="Print CUDA feature extraction information.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    configure_cuda()

    path = Path(args.corpus)
    if not path.exists():
        print(f"ERROR: {path} does not exist.")
        return

    model = CorpusModel(
        isomorphism_tau=args.isomorphism_tau,
        isomorphism_sharpness=args.isomorphism_sharpness,
        context_reduction_fraction=args.context_reduction_fraction,
        context_reduction_max_dims=args.context_reduction_max_dims,
    )

    model.ingest(path.read_text(encoding="utf-8"))
    model.finalize()

    iso_classes = (
        len(set(model.isomorphism_map.values()))
        if model.isomorphism_map
        else 0
    )
    print(
        f"Vocabulary: {len(model.unigram)} tokens | "
        f"bigram contexts: {len(model.bigram)}"
    )
    print(
        f"Isomorphism classes: {iso_classes} "
        f"(from {len(model.isomorphism_map)} vocab tokens)"
    )

    extractor = FeatureExtractor()

    feature_layer = NeuralFeatureLayer(
        seed=args.seed or 0,
        device=args.device,
        use_all_gpus=not args.single_gpu,
        lr=args.lr,
    )

    gpu_note = (
        f"{feature_layer.gpu_count} GPU(s), "
        f"DataParallel={feature_layer.multi_gpu}"
        if feature_layer.device.type == "cuda"
        else "no GPU"
    )
    print(
        f"Feature layer device: {feature_layer.device} ({gpu_note})"
    )

    if args.train_steps > 0:
        print(
            f"Training feature layer on the corpus's own bigram "
            f"statistics ({args.train_steps} steps)..."
        )
        train_feature_layer(
            feature_layer,
            extractor,
            model,
            steps=args.train_steps,
            batch_size=args.batch_size,
            candidate_limit=args.candidate_limit,
            lr=args.lr,
            seed=args.seed or 0,
        )
    else:
        print(
            "Skipping training -- using randomly initialized "
            "feature layer."
        )

    if not args.no_consensus:
        print(
            f"\nRunning {args.num_generations} unbiased scratch "
            f"generations for kernel consensus..."
        )

    print("\nGenerating...\n" + "=" * 70)
    while True:
        text_out = generate(
            model,
            extractor,
            feature_layer,
            prompt=input("USER: "),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            use_instruction_pull=not args.no_instruction_pull,
            seed=args.seed,
            use_consensus=not args.no_consensus,
            num_generations=args.num_generations,
            kernel_sharpness=args.kernel_sharpness,
            verbose_features=args.verbose_features,
        )

        print(text_out)
        print("=" * 70)


if __name__ == "__main__":
    main()
