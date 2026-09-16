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
#      per-token scores, PLUS a compounding instruction-likeness curve
#      (see below) that pulls the final text back toward the original
#      instruction as it gets longer
#
# --- Kernelized adversarial consensus ---
# Each scratch run is first canonicalized through the vocabulary's
# "isomorphism classes": tokens whose bigram-context distributions are
# near-identical (cosine similarity above ISOMORPHISM_TAU, after
# sharpening -- see below) play the same structural role in the grammar,
# so they're collapsed to one canonical representative before anything
# is counted. This means a vote for any member of a class reinforces the
# whole class, both when the modifier is built and when it's looked up
# during final scoring.
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
# --- Context-dimension reduction (feature agglomeration) ---
# Before isomorphism classes are built, the *context* dimensions of each
# token's lexical vector (i.e. which bigram-left-context it followed, and
# how often) are themselves clustered and merged: the most frequent
# CONTEXT_REDUCTION_MAX_DIMS context columns are agglomeratively paired
# up by cosine similarity of the tokens that follow them, until only
# CONTEXT_REDUCTION_FRACTION of them remain, and every token's vector is
# re-expressed in that smaller basis. This is a from-scratch,
# dependency-free stand-in for PCA / feature agglomeration -- it does not
# require numpy/sklearn. Long-tail context dims outside the cap are left
# untouched. Isomorphism classing and influence scoring both then run on
# this reduced space.
#
# Because dimensionality reduction is lossy and tends to make cosine
# similarities cluster upward, the isomorphism threshold is sharpened
# by raising raw cosine similarity to ISOMORPHISM_SHARPNESS before
# comparing to ISOMORPHISM_TAU. Cosine similarities here are always in
# [0, 1] (context vectors are non-negative frequencies): an exponent > 1
# shrinks weak similarities faster than strong ones (more selective),
# while an exponent < 1 pulls similarities upward toward 1, i.e. makes
# the threshold *more permissive* / less selective. Which direction you
# want depends on whether dimension reduction is being used aggressively
# (favor > 1, more selective) or you deliberately want more classes to
# merge (favor < 1).
#
# --- Deep bilinear activation (adhoc, destroys isomorphisms) ---
# Immediately after context-dimension reduction and immediately BEFORE
# isomorphism classing, each token's (reduced) lexical vector can
# optionally be pushed through a small stack of hand-rolled "bilinear"
# layers -- a dependency-free stand-in for a tiny DNN embedding tower.
# Each layer hashes the sparse context-vector into two independent dense
# projections (feature hashing with random signs, keyed by layer index),
# multiplies them elementwise (the "bilinear" interaction -- a product of
# two learned-ish projections of the same input, the way GLU/bilinear
# pooling layers work), scales, and squashes with tanh. The output of
# layer L feeds layer L+1, so it's "deep" in the same stacked sense as a
# DNN, even though there is no training: the projections are fixed,
# content-addressed hashes rather than learned weights.
#
# This is deliberately *destructive* of the isomorphism structure that
# _build_isomorphism_classes() looks for. Two tokens with literally
# identical bigram-context distributions still hash into the same
# buckets, so exact duplicates remain merged -- but any near-miss
# similarity that dimension reduction and natural corpus noise produce
# gets scrambled: small differences in the input get thrown into
# different hash buckets with independent random signs, and the
# elementwise product of two independent projections is far more
# sensitive to those differences than cosine similarity on the smooth,
# additive lexical vectors was. The effect compounds with depth. So
# post-activation cosine similarities collapse toward orthogonality for
# anything that wasn't already an exact match, ISOMORPHISM_TAU is rarely
# cleared, and most classes that survive dimension reduction fall apart
# into singletons. This is intentional: it's an adhoc way to turn the
# smooth "structural role" equivalence back into distinct per-token
# identity before isomorphism classing runs, at the cost of the classing
# step doing almost nothing.
#
# --- Compounding instruction-likeness curve (final generation only) ---
# A single instruction vector is built once from the original prompt's
# tokens (summing their lexical vectors). During the *final* biased
# generation pass -- never the independent scratch runs, which must stay
# diverse and unbiased for the consensus step to mean anything -- every
# candidate token's cosine similarity to that instruction vector is
# pushed through an increasing sigmoid curve and added to its score,
# scaled by a running `compound_factor`. That factor is multiplied up a
# little further after every token actually chosen (proportional to how
# instruction-like it was), so the pull toward the instruction gets
# stronger as the continuation gets longer -- it compounds -- but is
# clamped at INSTRUCTION_COMPOUND_CAP so it cannot spiral into simply
# echoing the instruction back verbatim. This bias stacks additively on
# top of whatever candidate_modifier is already active (the kernel
# consensus modifier, or the ontology's concept-vocabulary modifier in
# list generation), rather than replacing it. The factor resets to 1.0
# at the start of every independent continuation (each generate() call,
# each list item), so it never carries over and saturates across an
# entire multi-item list.
#
# (The original script also computed a "post-generation rebinding"
#  pass and a "second generation" pass, but neither was ever
#  displayed or used -- removed as dead code.)
#
# --- GPU acceleration ---
# The three stages above whose cost scales with vocab_size^2 or with
# context-dimension_count^2 -- context-dimension reduction, deep bilinear
# activation, and the shared influence/isomorphism similarity matrix --
# are implemented as dense torch tensor ops on DEVICE (CUDA if available,
# else CPU) rather than nested Python loops calling the dict-based
# cosine_similarity per pair. Everything per-token/per-step during actual
# generation (sample_next, _score_next_token, etc.) stays on small Python
# dicts: those calls happen one candidate token at a time against a
# handful of candidates, so the GPU dispatch overhead would outweigh any
# benefit there.
# ============================================================

MODEL_PATH = "model.json"

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 120

MIN_COUNT = 1
INFLUENCE_TAU = 0.7

CURVE_K = 18.0
CURVE_MIDPOINT = 0.5

CANDIDATE_LIMIT = 15
LEXICAL_WEIGHT = 0.15
VECTOR_WEIGHT = 0.15

# --- consensus / kernelized adversarial ensemble settings ---
NUM_GENERATIONS = 15        # how many scratch runs to generate per turn
KERNEL_SHARPNESS = 0.8     # adversarial sharpening exponent on kernel agreement
MODIFIER_WEIGHT = 0.1      # how much the consensus modifier biases the final generation
CONSENSUS_BASELINE_SPLIT = 0.1  # 0.0 = pure baseline prob, 1.0 = pure consensus modifier

# --- vocab isomorphism settings ---
ISOMORPHISM_TAU = 0.97     # cosine similarity threshold (post-sharpening) for treating
                           # two tokens as structurally interchangeable in the vocabulary
ISOMORPHISM_SHARPNESS = 0.8  # exponent applied to raw cosine similarity before comparing
                             # to ISOMORPHISM_TAU. Cosine values live in [0,1]; an exponent
                             # < 1 (as set here) pulls similarities upward, making the
                             # threshold more permissive -- more tokens merge into classes.

# --- context-dimension reduction settings (feature agglomeration) ---
CONTEXT_REDUCTION_FRACTION = 0.1   # keep this fraction of context dimensions after merging
CONTEXT_REDUCTION_MAX_DIMS = 200  # only cluster the N most frequent context dims;
                                    # merging is O(n^2) per round, so this caps cost.
                                    # rarer context dims pass through untouched.

# --- deep bilinear activation settings (adhoc, destroys isomorphisms) ---
ENABLE_DEEP_BILINEAR_ACTIVATION = True  # applied after context reduction, before
                                         # isomorphism classing (see header docstring)
BILINEAR_LAYERS = 32        # how many stacked hash-bilinear+tanh layers to apply
BILINEAR_DIM = 128           # hashed dense dimensionality used within each layer
BILINEAR_SCALE = 14.0        # multiplies the elementwise bilinear product before tanh

# --- Markovian transitivity masking settings ---
TRANSITIVITY_DECAY = 0.01   # decay applied per hop when composing A->B->C into A->C
TRANSITIVITY_WEIGHT = 0.71  # how strongly the prompt-pattern mask biases scoring
TRANSITIVITY_MASK_PENALTY = 1.0   # "deficit strength" for masked-out tokens (fed into the exponential, not a raw log-penalty anymore)
TRANSITIVITY_SUPERPOLY_K = 13.0    # exponential growth rate applied to promise strength; higher = more explosive gap between weakly- and strongly-promised tokens

# --- compounding instruction-likeness curve settings (final generation only) ---
INSTRUCTION_COMPOUND_CURVE_K = 2.0        # steepness of the increasing sigmoid applied
                                            # to a candidate token's cosine similarity
                                            # to the instruction vector
INSTRUCTION_COMPOUND_MIDPOINT = 1110.5        # similarity value at which the curve crosses 0.5
INSTRUCTION_COMPOUND_GROWTH = 0.65         # how much each chosen token's instruction-likeness
                                            # grows the running compound_factor
INSTRUCTION_COMPOUND_CAP = 4.0             # hard ceiling on compound_factor, so the pull
                                            # toward the instruction cannot run away into
                                            # degenerate echoing of the prompt
INSTRUCTION_COMPOUND_WEIGHT = 1.4          # base strength of the instruction-likeness bias
                                            # in the score, before compounding is applied


INVERSION_GATE_TOKENS = 600       

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



def cosine_similarity(a, b, eps=1e-12):
    """GPU-compatible cosine similarity for dict or tensor inputs."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a = a / torch.clamp(torch.linalg.vector_norm(a, dim=-1, keepdim=True), min=eps)
        b = b / torch.clamp(torch.linalg.vector_norm(b, dim=-1, keepdim=True), min=eps)
        return (a * b).sum(dim=-1)
    # Fallback to dict version
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
    isomorphism_sharpness: float = ISOMORPHISM_SHARPNESS
    context_reduction_fraction: float = CONTEXT_REDUCTION_FRACTION
    context_reduction_max_dims: int = CONTEXT_REDUCTION_MAX_DIMS
    enable_deep_bilinear_activation: bool = ENABLE_DEEP_BILINEAR_ACTIVATION
    bilinear_layers: int = BILINEAR_LAYERS
    bilinear_dim: int = BILINEAR_DIM
    bilinear_scale: float = BILINEAR_SCALE

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

        # Reduce the dimensionality of the context space (feature
        # agglomeration) before anything downstream -- influence scoring
        # and isomorphism classing -- runs on these vectors.
        self._reduce_context_dimensions()

        # Adhoc deep bilinear activation: deliberately scrambles the
        # smooth near-duplicate structure that dimension reduction
        # leaves behind, so isomorphism classing below has much less to
        # find (see header docstring for the rationale).
        if self.enable_deep_bilinear_activation:
            self._apply_deep_bilinear_activation()

        # Influence scoring and isomorphism classing both need an n x n
        # (vocab-sized) pairwise cosine similarity matrix -- previously
        # computed as two separate O(vocab^2) Python loops, each calling
        # the dict-based cosine_similarity per pair. That's exactly the
        # kind of dense, uniform, embarrassingly parallel math that
        # belongs on the GPU: build the whole vocab as one dense tensor
        # and get every pairwise similarity from a single matmul instead.
        self._build_influence_and_isomorphism()

        self.finalized = True

    def _reduce_context_dimensions(
        self,
        fraction: Optional[float] = None,
        max_dims: Optional[int] = None,
    ) -> None:
        """
        Reduce the dimensionality of lexical_vectors by agglomeratively
        merging the context dimensions (bigram left-contexts) that behave
        most alike across tokens -- clustering the *columns* of the token
        x context matrix, not the tokens themselves. Two context dims
        merge when the tokens that follow them, weighted by frequency,
        look similar; merging sums their weight into one new
        pseudo-dimension.

        This is a from-scratch, dependency-free substitute for PCA/feature
        agglomeration (no numpy/sklearn required) -- "dependency-free"
        meaning no sklearn/PCA library, not no GPU: the per-round search
        for the best pair to merge is an O(m^2) pairwise cosine similarity
        over up to `max_dims` columns, which is a dense matmul plus a
        row-norm, i.e. exactly the kind of math a GPU chews through in one
        call instead of a nested Python loop. `max_dims` still caps which
        dimensions are even considered -- context dims outside the cap are
        long-tail and left untouched, just carried through under their
        original name.
        """
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

        # Build the dense (columns x tokens) matrix once, on-device. Each
        # row is one context dimension's weight across every token --
        # this is the thing whose rows we're agglomeratively merging.
        token_order = list(self.lexical_vectors.keys())
        token_index = {tok: i for i, tok in enumerate(token_order)}
        col_index = {c: i for i, c in enumerate(active)}
        rows = torch.zeros((len(active), len(token_order)), dtype=DTYPE, device=DEVICE)
        for token, vec in self.lexical_vectors.items():
            ti = token_index[token]
            for ctx, w in vec.items():
                ci = col_index.get(ctx)
                if ci is not None:
                    rows[ci, ti] = w

        names: List[str] = list(active)
        # Track which original leaf dims each live row has absorbed so
        # far, so the final remap can flatten a chain of merges back to
        # their original context-key names (mirrors the old `groups`
        # dict, just kept alongside the tensor rows instead of dicts).
        members: List[List[str]] = [[c] for c in active]

        while len(names) > target_count:
            norm = torch.clamp(torch.linalg.vector_norm(rows, dim=1, keepdim=True), min=1e-12)
            normed = rows / norm
            sim = normed @ normed.T
            sim.fill_diagonal_(-1.0)  # never "merge" a row with itself
            n_cur = sim.shape[0]
            flat_idx = int(torch.argmax(sim).item())
            i, j = divmod(flat_idx, n_cur)
            if i > j:
                i, j = j, i

            merged_name = f"{names[i]}+{names[j]}"
            # Same "a minus b" combination as the original implementation
            # (not a typo we're introducing -- preserved as-is so behavior
            # doesn't silently change).
            merged_vec = (rows[i] - rows[j]).unsqueeze(0)
            merged_members = members[i] + members[j]

            keep = [k for k in range(n_cur) if k not in (i, j)]
            keep_idx = torch.tensor(keep, dtype=torch.long, device=DEVICE)
            rows = torch.cat([rows.index_select(0, keep_idx), merged_vec], dim=0)
            names = [names[k] for k in keep] + [merged_name]
            members = [members[k] for k in keep] + [merged_members]

        remap: Dict[str, str] = {
            leaf: gid for gid, leaves in zip(names, members) for leaf in leaves
        }

        new_vectors: Dict[str, Dict[str, float]] = {}
        for token, vec in self.lexical_vectors.items():
            new_vec: Dict[str, float] = defaultdict(float)
            for ctx, w in vec.items():
                new_vec[remap.get(ctx, ctx)] += w
            new_vectors[token] = dict(new_vec)
        self.lexical_vectors = new_vectors

    # ---------------- deep bilinear activation (adhoc, destroys isomorphisms) ----------------

    @staticmethod
    def _hash_feature(key: str, layer: int, projection: int, dim: int) -> Tuple[int, float]:
        """
        Deterministic feature-hashing trick (Weinberger et al. style):
        maps an arbitrary string key to a dense index in [0, dim) plus a
        random +/-1 sign, both derived from a hash so no weight matrix
        needs to be stored or trained. `layer` and `projection` (0 or 1,
        selecting which of the two per-layer bilinear branches this call
        is for) are folded into the hash so every layer/branch gets an
        independent-looking projection of the same input, the way
        independently-initialized DNN weight matrices would.
        """
        h = hash((layer, projection, key))
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        return idx, sign

    def _hash_projection_matrix(
        self, keys: List[str], layer: int, projection: int, dim: int
    ) -> torch.Tensor:
        """
        Materializes `_hash_feature` as an (len(keys) x dim) matrix P
        where P[k, _hash_feature(keys[k], ...).idx] = sign, all zeros
        elsewhere. Summing a token's weighted keys through the hash
        (the old per-token Python loop) is then exactly `x @ P` for a
        whole batch of tokens `x` at once -- one GPU matmul standing in
        for what used to be a hash lookup per (token, key) pair. Built
        once per layer/branch, not once per token.
        """
        mat = torch.zeros((len(keys), dim), dtype=DTYPE, device=DEVICE)
        for k, key in enumerate(keys):
            idx, sign = self._hash_feature(key, layer, projection, dim)
            mat[k, idx] = sign
        return mat

    def _apply_deep_bilinear_activation(self) -> None:
        """
        Adhoc "deep" bilinear tower, run once for the whole vocabulary as
        dense GPU tensor ops instead of per-token Python:

          1. Stack every token's sparse vector into one dense
             (vocab_size x input_dim) matrix `x`.
          2. Per layer: hash-project `x` into two independent dense
             branches `u = x @ P_u`, `v = x @ P_v` (stand-ins for two
             learned Linear layers), take their elementwise product (the
             "bilinear" interaction -- bilinear pooling / GLU-style,
             not a plain linear layer), scale, and squash with tanh.
             All vocab_size rows are transformed in the same matmul.
          3. The layer's dense output (columns named "L{layer}:{i}")
             becomes the next layer's dense input directly -- no
             sparse/dense round-trip between layers -- which is the
             "deep" part: depth composes on-device, only converting
             back to the sparse dict format the rest of the pipeline
             expects after the last layer.

        This produces the same values `_hash_feature`-driven summation
        would produce per token (matmul is just batched, weighted
        summation into hash buckets), just computed for every token in
        one shot rather than with a Python loop per token. Small input
        perturbations still land in different hash buckets with
        independent signs, and the elementwise product of two such
        projections still amplifies rather than smooths those
        differences -- destructive of near-duplicate structure, as
        intended (see header docstring).
        """
        token_order = list(self.lexical_vectors.keys())
        if not token_order:
            return
        input_keys = sorted({k for vec in self.lexical_vectors.values() for k in vec})
        key_index = {k: i for i, k in enumerate(input_keys)}

        x = torch.zeros((len(token_order), len(input_keys)), dtype=DTYPE, device=DEVICE)
        for ti, token in enumerate(token_order):
            for key, w in self.lexical_vectors[token].items():
                x[ti, key_index[key]] = w

        dim = self.bilinear_dim
        current_keys = input_keys
        for layer in range(self.bilinear_layers):
            proj_u = self._hash_projection_matrix(current_keys, layer, 0, dim)
            proj_v = self._hash_projection_matrix(current_keys, layer, 1, dim)
            u = x @ proj_u
            v = x @ proj_v
            x = torch.tanh(self.bilinear_scale * u * v)
            current_keys = [f"L{layer}:{i}" for i in range(dim)]
            if not torch.any(x):
                break  # matches the old early-exit once a token's vector goes fully empty

        x_cpu = x.detach().cpu()
        nz_rows, nz_cols = torch.nonzero(x_cpu, as_tuple=True)
        per_token: Dict[int, Dict[str, float]] = defaultdict(dict)
        for r, c in zip(nz_rows.tolist(), nz_cols.tolist()):
            per_token[r][current_keys[c]] = float(x_cpu[r, c])

        self.lexical_vectors = {
            token: per_token.get(ti, {}) for ti, token in enumerate(token_order)
        }

    def _vocab_similarity_matrix(self) -> Tuple[List[str], torch.Tensor]:
        """
        Builds the full (vocab_size x vocab_size) pairwise cosine
        similarity matrix in one shot on the GPU: stack every vocabulary
        token's (dimension-reduced, possibly bilinear-activated) lexical
        vector into a dense (vocab_size x dim) matrix, L2-normalize each
        row, and matmul against its own transpose. Both influence scoring
        and isomorphism classing used to each run their own O(vocab^2)
        Python loop of dict-based cosine_similarity calls; they now share
        this single dense computation instead.
        """
        vocab = self.vocabulary
        keys = sorted({k for vec in self.lexical_vectors.values() for k in vec})
        key_index = {k: i for i, k in enumerate(keys)}
        mat = torch.zeros((len(vocab), len(keys)), dtype=DTYPE, device=DEVICE)
        for i, token in enumerate(vocab):
            for key, w in self.lexical_vectors.get(token, {}).items():
                j = key_index.get(key)
                if j is not None:
                    mat[i, j] = w
        norm = torch.clamp(torch.linalg.vector_norm(mat, dim=1, keepdim=True), min=1e-12)
        normed = mat / norm
        sim = normed @ normed.T
        return vocab, sim

    def _build_influence_and_isomorphism(self) -> None:
        """
        Computes influence_vectors and isomorphism_map together from one
        shared GPU similarity matrix (see `_vocab_similarity_matrix`),
        replacing what used to be two separate O(vocab^2) Python loops.
        The per-pair *decisions* (threshold, sharpen, greedy canonical
        assignment) are still made on CPU -- they're branchy/sequential
        and vocab-sized, not vocab^2-sized, so there's nothing to gain by
        moving them -- but the O(vocab^2 * dim) similarity computation
        itself, which dominates cost for any real vocabulary, now runs as
        a single dense matmul instead of vocab^2 individual calls.
        """
        vocab, sim = self._vocab_similarity_matrix()
        sim_cpu = sim.detach().cpu()

        # --- influence scoring ---
        self.influence_vectors = {}
        for i, source in enumerate(vocab):
            row = sim_cpu[i]
            scores: Dict[str, float] = {}
            hits = (row >= self.influence_tau).nonzero(as_tuple=True)[0].tolist()
            for j in hits:
                if j == i:
                    continue
                scores[vocab[j]] = float(row[j])
            self.influence_vectors[source] = scores

        self.isomorphism_map = self._isomorphism_map_from_similarity(vocab, sim_cpu)

    def _isomorphism_map_from_similarity(
        self, vocab: List[str], sim_cpu: torch.Tensor
    ) -> Dict[str, str]:
        """
        Given a precomputed (vocab_size x vocab_size) cosine similarity
        matrix, does the actual (still sequential/branchy, so left on
        CPU) greedy isomorphism-class assignment: two tokens are treated
        as isomorphic if their similarity, raised to
        `isomorphism_sharpness`, is >= isomorphism_tau -- i.e. they play
        the same structural role in the grammar even if they're literally
        different words. Each class collapses to a single canonical token
        (the first member encountered, in vocabulary order) so a vote
        cast for any member reinforces the whole class.

        Cosine similarity was guaranteed in [0, 1] back when lexical
        vectors were plain non-negative frequencies, so raising it to a
        fractional isomorphism_sharpness was always safe. Deep bilinear
        activation (see above) can produce vectors with negative
        components (tanh output), so raw similarity can now be negative
        -- and a negative base to a fractional power is a complex number
        in Python, not a real one. Clamp to [0, 1] first: a negative
        similarity just means "not structurally similar," which belongs
        at the low end of the scale, not off the real line.

        When deep bilinear activation is enabled, the vectors feeding
        this similarity matrix are already the scrambled, deep-activated
        ones, so in practice almost nothing clears isomorphism_tau except
        tokens whose inputs were already exactly identical -- this is by
        design, not a bug (see header docstring).
        """
        n = len(vocab)
        clamped = torch.clamp(sim_cpu, min=0.0, max=1.0)
        sharpened = clamped ** self.isomorphism_sharpness

        canonical: Dict[str, str] = {}
        assigned = [False] * n
        for i in range(n):
            if assigned[i]:
                continue
            canonical[vocab[i]] = vocab[i]
            assigned[i] = True
            if not self.lexical_vectors.get(vocab[i]):
                continue
            row = sharpened[i]
            for j in range(i + 1, n):
                if assigned[j] or not self.lexical_vectors.get(vocab[j]):
                    continue
                if float(row[j]) >= self.isomorphism_tau:
                    canonical[vocab[j]] = vocab[i]
                    assigned[j] = True
        return canonical

    def _build_isomorphism_classes(self) -> None:
        """
        Standalone isomorphism-only entry point, kept for the from_dict()
        fallback path (loading an older save that has lexical_vectors but
        no isomorphism_map, and where influence_vectors is loaded from
        the file rather than recomputed -- this must NOT overwrite that).
        Builds the same GPU similarity matrix `_build_influence_and_isomorphism`
        would, but only fills in isomorphism_map, leaving influence_vectors
        untouched.
        """
        vocab, sim = self._vocab_similarity_matrix()
        self.isomorphism_map = self._isomorphism_map_from_similarity(vocab, sim.detach().cpu())

    def _canonical(self, token: str) -> str:
        return self.isomorphism_map.get(token, token)

    def _canonicalize_run(self, run: List[str]) -> Counter:
        return Counter(self._canonical(t) for t in run)

    # ---------------- compounding instruction-likeness curve ----------------

    @staticmethod
    def _sigmoid_curve(x: float, k: float, midpoint: float) -> float:
        """
        Increasing logistic curve: 0 as x -> -inf, 1 as x -> +inf, 0.5 at
        x == midpoint. Used to turn a raw cosine similarity (instruction
        likeness) into a smooth 0..1 bias term. This is the mirror image
        of `_curve_weight`, which is a *decreasing* curve used for EOS
        suppression -- this one increases with similarity instead.
        """
        return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))

    def _instruction_vector(self, prompt: str) -> Dict[str, float]:
        """
        Build a single fixed context-vector for the whole instruction by
        summing the (dimension-reduced) lexical vectors of its tokens.
        This is the target that the compounding curve pulls the final
        generation toward -- it is computed once from the original prompt
        and never updated as generation proceeds.
        """
        tokens = [t for t in tokenize(prompt) if t not in IGNORED_TOKENS]
        agg: Dict[str, float] = defaultdict(float)
        for t in tokens:
            vec = self.lexical_vectors.get(t, {})
            for ctx, w in vec.items():
                agg[ctx] += w
        return dict(agg)

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
        instruction_vector: Optional[Dict[str, float]] = None,
        instruction_weight: float = 0.0,
        compound_factor: float = 1.0,
        invert_instruction: bool = False,
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
                # This term stacks additively on top of candidate_modifier
                # above (the kernel-consensus bias) -- it does not replace
                # it. compound_factor grows across the generation loop
                # (see generate()), so this term's influence increases as
                # the continuation gets longer, "compounding" toward
                # likeness with the instruction -- unless the length gate
                # has fired (invert_instruction=True), in which case the
                # sign is flipped and it pushes away from the instruction
                # instead, with the same growing magnitude.
                likeness = cosine_similarity(
                    instruction_vector, self.lexical_vectors.get(token, {})
                )
                curved_likeness = self._sigmoid_curve(
                    likeness, INSTRUCTION_COMPOUND_CURVE_K, INSTRUCTION_COMPOUND_MIDPOINT
                )
                sign = -1.0 if invert_instruction else 1.0
                score += sign * instruction_weight * compound_factor * curved_likeness
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
        invert_instruction: bool = False,
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
            invert_instruction=invert_instruction,
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
        invert_instruction: bool = False,
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
            invert_instruction=invert_instruction,
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
        reform_window: int = 0,
        reform_passes: int = 1,
    ) -> str:
        """
        reform_window/reform_passes ("reform previous tokens in sync"):
        Off by default (reform_window=0), matching prior behavior exactly.
        When reform_window > 0, after each new token is appended we walk
        back over the last `reform_window` *generated* tokens (never the
        fixed prompt) and re-sample each one, in place, using whatever
        `compound_factor` / instruction pull / modifiers apply *right now*
        -- not the (earlier, less-informed) state that picked it originally.
        This is done `reform_passes` times per step, left-to-right, so a
        reformed token can itself be re-reformed later in the same pass
        using its now-updated left neighbor. It keeps earlier tokens "in
        sync" with the trajectory the sequence has since taken, rather
        than freezing them at generation time.
        """
        generated = tokenize(prompt)
        prompt_len = len(generated)

        # Fixed instruction target, built once from the original prompt
        # (never recomputed from the growing `generated` list). Off by
        # default (instruction_weight=0.0) -- callers that want the
        # compounding pull toward the instruction (generate_consensus's
        # final pass) opt in explicitly.
        instruction_vector = self._instruction_vector(prompt) if instruction_weight else None
        compound_factor = 1.0

        def _resample_at(pos: int, invert: bool) -> str:
            # Re-sample generated[pos] using only the tokens *before* it
            # (the model is left-context only) but with the *current*
            # compound_factor/instruction state, so the choice reflects
            # how far the generation has progressed since pos was first
            # picked.
            left_context = " ".join(generated[:pos])
            return self.sample_next(
                left_context,
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

            if reform_window > 0:
                # Only reform tokens we generated ourselves, never the
                # user's original prompt, and never the token we just
                # picked this step (it's already maximally "in sync").
                window_start = max(prompt_len, len(generated) - 1 - reform_window)
                window_end = len(generated) - 1
                for _ in range(max(reform_passes, 1)):
                    for pos in range(window_start, window_end):
                        generated[pos] = _resample_at(pos, invert)

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

    # ---------------- non-conjoint context matrix / rhombus selection ----------------

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

    # ---------------- Markovian transitivity / prompt-pattern masking ----------------

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
            "isomorphism_sharpness": self.isomorphism_sharpness,
            "context_reduction_fraction": self.context_reduction_fraction,
            "context_reduction_max_dims": self.context_reduction_max_dims,
            "enable_deep_bilinear_activation": self.enable_deep_bilinear_activation,
            "bilinear_layers": self.bilinear_layers,
            "bilinear_dim": self.bilinear_dim,
            "bilinear_scale": self.bilinear_scale,
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
            enable_deep_bilinear_activation=data.get(
                "enable_deep_bilinear_activation", ENABLE_DEEP_BILINEAR_ACTIVATION
            ),
            bilinear_layers=data.get("bilinear_layers", BILINEAR_LAYERS),
            bilinear_dim=data.get("bilinear_dim", BILINEAR_DIM),
            bilinear_scale=data.get("bilinear_scale", BILINEAR_SCALE),
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


def _strip_structural_tokens(tokens: List[str]) -> List[str]:
    """Drop <bos>/<eos>/<unk> before display -- the underlying generate()
    treats them as ordinary vocabulary (see module docstring), so
    generated text needs this cleanup to read as plain text rather than
    leaking markup."""
    return [t for t in tokens if t not in IGNORED_TOKENS]



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


# ============================================================
# Main
# ============================================================

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

        print("\nGenerating final (modifier-biased, compounding toward instruction)...")
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
