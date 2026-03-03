#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic Riddle Generator V9.0 — Complete Implementation
===============================================================================

ARCHITECTURE OVERVIEW:
- Extract N target words from the prompt (e.g., "consider", "nature", "understanding")
- For each target word, generate a multi-line riddle (default 5 lines).
- Each line uses a semantic form ("identity", "possession", "capability", etc.)
- The target word acts as the semantic centroid: it pulls the generated vocabulary
  toward its meaning, but is STRICTLY BANNED from the output.
- Result: The system constructs cryptic, cohesive riddles mathematically bound 
  to the unspoken target word.

CHAIN-OF-THOUGHT REASONING:
Every major function includes reasoning about WHY that function exists and HOW it
integrates with the broader system. This document is self-explaining via docstrings.
===============================================================================
"""

from __future__ import annotations
import re
import math
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import pandas as pd
import gradio as gr
import torch
import torch.nn.functional as F
from datasets import load_dataset

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

STOP_WORDS = set(
    "a an and are as at be by for from has have he her him his i in is it its "
    "me my of on or our she so that the their them they this to was we were what "
    "when where which who will with you your"
    .split()
)

COGNITIVE_TOKENS = {"[PROBLEM]", "[SOLUTION]"}

TOPO_KEYWORDS = {
    "homology", "cohomology", "persistent", "filtration", "barcode", "betti",
    "euler", "simplicial", "homotopy", "manifold", "morse", "sheaf"
}

_VOWELS = set("aeiouy")
_COMMON_BIGRAMS: set = {
    "th", "he", "in", "er", "an", "re", "on", "en", "at", "ou", "ed", "nd",
    "to", "or", "ea", "ti", "es", "st", "ar", "nt", "is", "al", "it", "as",
    "ha", "et", "se", "ng", "le", "of",
}

_LATINATE_PREFIXES = {
    "pre", "post", "anti", "auto", "bio", "geo", "hyper", "hypo", "inter",
    "intra", "micro", "macro", "meta", "mono", "multi", "neo", "non", "over",
    "poly", "pseudo", "semi", "sub", "super", "trans", "ultra", "uni", "dis",
    "mis", "un", "re", "de",
}

_LATINATE_SUFFIXES = {
    "tion", "sion", "ment", "ness", "ity", "ism", "ist", "ize", "ise", "ful",
    "less", "ous", "ious", "eous", "ance", "ence", "able", "ible", "ive",
    "ative", "ology", "ography", "ician", "ation", "ization", "isation",
}

_EARLY_WORDS: Dict[str, float] = {
    "cat": 2.5, "dog": 2.5, "mom": 2.2, "dad": 2.2, "baby": 2.8, "ball": 2.6,
    "cup": 2.7, "eye": 2.4, "ear": 2.5, "nose": 2.6, "hat": 2.8, "shoe": 2.9,
    "bed": 2.7, "hot": 3.0, "cold": 3.1, "big": 3.0, "small": 3.2, "run": 3.1,
    "eat": 2.9, "go": 2.5, "yes": 2.4, "no": 2.3, "hi": 2.2, "bye": 2.3,
    "more": 2.8, "up": 2.6, "down": 2.8, "in": 2.5, "out": 2.7, "on": 2.6,
    "off": 2.8, "want": 2.7, "help": 3.0, "play": 2.9, "walk": 3.0,
    "look": 2.8, "see": 2.5, "hear": 2.8, "think": 3.5, "know": 3.4,
    "hand": 2.9, "foot": 2.9, "head": 2.7, "face": 2.8, "name": 3.2,
    "home": 3.0, "door": 3.1, "car": 2.8, "tree": 3.0, "book": 3.2,
}

COHO_MAX_DIM = 12
COHO_MIN_DIM = 2
LENGTH_CEIL = 14
COHO_FILTRATION_STEPS = 8
SHIFT_MAG_MIN = 0.05
SHIFT_MAG_MAX = 0.35
AGREEMENT_BONUS_MIN = 0.10
AGREEMENT_BONUS_MAX = 0.60

RIDDLE_PREFIXES = {
    "identity": ("i", "am"),
    "possession": ("i", "have"),
    "capability": ("i", "can"),
    "contrast": ("yet", "i"),
    "origin": ("i", "come"),
}

# ────────────────────────────────────────────────────────────────────────────
# RIDDLE FORM CLASS
# ────────────────────────────────────────────────────────────────────────────

class RiddleForm:
    """
    REASONING:
    Forms represent the lines of a riddle. Each line has a specific semantic intent 
    (identity, possession, capability, contrast, origin) and acts as a gravitational 
    center for text generation, pulling vocabulary toward the hidden target word 
    without explicitly naming it.
    """
    
    FORMS = ["identity", "possession", "capability", "contrast", "origin"]

    def __init__(self, target_word: str, form_name: str, riddle_index: int, line_index: int):
        self.target_word = target_word.lower()
        self.form_name = form_name if form_name in self.FORMS else "identity"
        self.riddle_index = riddle_index
        self.line_index = line_index
        
        self.activation_per_line: Dict[int, float] = {}
        self.total_activation: float = 0.0
        self.spreading_context: List[str] = []
        self.value_accumulated: float = 0.0

    def __repr__(self) -> str:
        return f"{self.target_word}[{self.form_name}@rid{self.riddle_index}-L{self.line_index}]"

    def to_string(self) -> str:
        return f"{self.target_word}_{self.form_name}_{self.riddle_index}_{self.line_index}"

    def activate_in_line(self, line_index: int, strength: float = 1.0):
        self.activation_per_line[line_index] = \
            self.activation_per_line.get(line_index, 0.0) + strength
        self.total_activation += strength

    def spread_to_word(self, word: str, strength: float = 0.5):
        if word not in self.spreading_context:
            self.spreading_context.append(word)
        self.value_accumulated += strength

    def get_total_value(self) -> float:
        return self.total_activation + self.value_accumulated


# ────────────────────────────────────────────────────────────────────────────
# RIDDLE GENERATION PLAN
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RiddleGenerationPlan:
    """
    REASONING:
    Orchestrates form creation, assignment, and value reporting for the riddle system. 
    It ensures that every target extracted from the prompt receives a structurally 
    sound riddle composed of several descriptive lines.
    """
    extracted_targets: List[str] = field(default_factory=list)
    riddles: Dict[int, List[RiddleForm]] = field(default_factory=lambda: defaultdict(list))
    line_outputs: Dict[Tuple[int, int], str] = field(default_factory=dict)
    
    def build_forms(self, target_words: List[str], num_riddles: int):
        self.extracted_targets = target_words
        word_index = 0
        for r_idx in range(num_riddles):
            word = target_words[word_index % len(target_words)]
            word_index += 1
            for l_idx, f_name in enumerate(RiddleForm.FORMS):
                form = RiddleForm(word, f_name, r_idx, l_idx)
                self.riddles[r_idx].append(form)

    def record_line_generation(
        self, riddle_index: int, line_index: int, text: str, 
        form_activation: float = 1.0, influenced_words: Optional[List[str]] = None
    ):
        self.line_outputs[(riddle_index, line_index)] = text
        forms = self.riddles.get(riddle_index, [])
        if line_index < len(forms):
            form = forms[line_index]
            form.activate_in_line(line_index, form_activation)
            if influenced_words:
                for w in influenced_words:
                    form.spread_to_word(w, 0.5)

    def generate_report(self) -> str:
        lines = [
            f"{'='*70}",
            f" Riddle Generation Form Report",
            f"{'='*70}",
            f"Target words: {', '.join(self.extracted_targets)}",
            f"Total riddles generated: {len(self.riddles)}",
            f"",
        ]

        all_forms = []
        for r_forms in self.riddles.values():
            all_forms.extend(r_forms)

        sorted_forms = sorted(all_forms, key=lambda f: f.get_total_value(), reverse=True)
        total_value = sum(f.get_total_value() for f in sorted_forms)

        lines.append(f"Total cumulative activation: {total_value:.4f}\n")
        lines.append("Form Rankings (Top 30):")
        lines.append(
            f"{'Rank':<5} {'Riddle':<8} {'Target':<15} {'Line Form':<15} "
            f"{'Value':<10} {'%':<8} {'Influenced':<10}"
        )
        lines.append(f"{'-'*85}")

        for rank, form in enumerate(sorted_forms[:30], 1):
            pct = 100 * form.get_total_value() / max(total_value, 1e-8)
            num_influenced = len(form.spreading_context)
            lines.append(
                f"{rank:<5} {form.riddle_index:<8} {form.target_word:<15} "
                f"{form.form_name:<15} {form.get_total_value():<10.4f} "
                f"{pct:<8.2f} {num_influenced:<10}"
            )

        lines.append("\nTarget-to-Word Influence Map (Top 10 Forms):\n")
        for rank, form in enumerate(sorted_forms[:10], 1):
            if form.spreading_context:
                influenced_str = ", ".join(form.spreading_context[:8])
                if len(form.spreading_context) > 8:
                    influenced_str += f", ... (+{len(form.spreading_context)-8} more)"
                lines.append(
                    f"{rank:2d}. {form.target_word}[{form.form_name}@rid{form.riddle_index}]\n"
                    f" → Influenced: {influenced_str}"
                )

        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# SEMANTIC SIMILARITY & WORD AGE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def semantic_similarity(word1: str, word2: str) -> float:
    w1, w2 = word1.lower(), word2.lower()
    if w1 == w2:
        return 1.0

    lev_dist = edit_distance(w1, w2)
    max_len = max(len(w1), len(w2))
    lev_sim = 1.0 - (lev_dist / max(max_len, 1))

    len_dist = abs(len(w1) - len(w2))
    len_sim = 1.0 - (len_dist / max_len)

    bigrams1 = set(w1[i:i+2] for i in range(len(w1)-1))
    bigrams2 = set(w2[i:i+2] for i in range(len(w2)-1))
    if bigrams1 and bigrams2:
        bigram_sim = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
    else:
        bigram_sim = 0.0

    combined = 0.4 * lev_sim + 0.3 * len_sim + 0.3 * bigram_sim
    return float(combined)

def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ────────────────────────────────────────────────────────────────────────────
# LENGTH-DEPENDENT TOPOLOGY FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def length_alpha(word: str, ceil: int = LENGTH_CEIL) -> float:
    n = len(word.strip())
    mid = ceil / 2.0
    return float(1.0 / (1.0 + math.exp(-0.55 * (n - mid))))

def _build_ngram_simplex(word: str, n: int = 2) -> List[Tuple[str, ...]]:
    w = word.lower()
    ngrams = [w[i:i + n] for i in range(max(1, len(w) - n + 1))]
    vertices: List[Tuple[str, ...]] = [(g,) for g in ngrams]
    edges: List[Tuple[str, ...]] = []
    for i in range(len(ngrams)):
        for j in range(i + 1, len(ngrams)):
            shared = sum(a == b for a, b in zip(ngrams[i], ngrams[j]))
            if shared >= n - 1:
                edges.append((ngrams[i], ngrams[j]))
    return vertices + edges

def _cohomological_betti1(word: str) -> int:
    w = word.lower()
    if len(w) < 2:
        return 0
    simplices = _build_ngram_simplex(w, n=2)
    vertices = [s for s in simplices if len(s) == 1]
    edges = [s for s in simplices if len(s) == 2]

    parent = {v[0]: v[0] for v in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False

    tree_edges = 0
    for e in edges:
        if union(e[0], e[1]):
            tree_edges += 1

    V = len(vertices)
    E = len(edges)
    C = V - tree_edges
    beta1 = max(0, E - V + C)
    return beta1

def cohomological_dim(word: str) -> int:
    beta1 = _cohomological_betti1(word)
    raw = COHO_MIN_DIM + min(beta1, COHO_MAX_DIM - COHO_MIN_DIM)
    return max(COHO_MIN_DIM, int(round(raw / 2) * 2))

def length_dim(word: str) -> int:
    return cohomological_dim(word)

def length_shift_mag(word: str) -> float:
    α = length_alpha(word)
    beta1_norm = min(1.0, _cohomological_betti1(word) / max(1, COHO_MAX_DIM))
    alpha_blend = 0.6 * α + 0.4 * beta1_norm
    return SHIFT_MAG_MIN + alpha_blend * (SHIFT_MAG_MAX - SHIFT_MAG_MIN)

def length_agreement_bonus(word: str) -> float:
    α = length_alpha(word)
    coho_ratio = cohomological_dim(word) / COHO_MAX_DIM
    alpha_blend = 0.5 * α + 0.5 * coho_ratio
    return AGREEMENT_BONUS_MIN + alpha_blend * (AGREEMENT_BONUS_MAX - AGREEMENT_BONUS_MIN)

def length_topo_kernel(word: str) -> float:
    α = length_alpha(word)
    return float(0.05 + 0.95 * (α ** 1.5))


# ────────────────────────────────────────────────────────────────────────────
# AoA DATASET & WORD AGE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

AOA_DATASET_URL = "https://norare.clld.org/contributions/Kuperman-2012-AoA/English-AoA-30K.csv"
AOA_COL_WORD = "Word"
AOA_COL_AOA = "AoA"

def load_aoa_dataset(max_rows: int = 35_000) -> Dict[str, float]:
    try:
        df = pd.read_csv(AOA_DATASET_URL, nrows=max_rows)
        if AOA_COL_WORD not in df.columns or AOA_COL_AOA not in df.columns:
            return {}
        df = df[[AOA_COL_WORD, AOA_COL_AOA]].dropna()
        return {
            str(w).strip().lower(): float(a)
            for w, a in zip(df[AOA_COL_WORD], df[AOA_COL_AOA])
        }
    except Exception:
        return {}

def _count_syllables(word: str) -> int:
    w = word.lower().rstrip("e")
    count = sum(
        1 for i, c in enumerate(w)
        if c in _VOWELS and (i == 0 or w[i - 1] not in _VOWELS)
    )
    return max(1, count)

def _morpheme_complexity(word: str) -> float:
    w = word.lower()
    score = 0.0
    for p in _LATINATE_PREFIXES:
        if w.startswith(p) and len(w) > len(p) + 2:
            score += 0.25
            break
    for s in _LATINATE_SUFFIXES:
        if w.endswith(s) and len(w) > len(s) + 2:
            score += 0.25 * (1 + len(s) / 6)
            break
    return min(1.0, score)

def _bigram_familiarity(word: str) -> float:
    w = word.lower()
    if len(w) < 2:
        return 0.5
    bigrams = [w[i:i + 2] for i in range(len(w) - 1)]
    return sum(1 for b in bigrams if b in _COMMON_BIGRAMS) / len(bigrams)

def _ortho_neighborhood_size(word: str, aoa_dict: Dict[str, float]) -> int:
    w = word.lower()
    n = len(w)
    count = 0
    for cand in aoa_dict:
        if len(cand) == n and cand != w:
            diffs = sum(a != b for a, b in zip(w, cand))
            if diffs == 1:
                count += 1
            if count >= 20:
                break
    return count

def calculate_word_age(
    word: str, aoa: Dict[str, float], corpus_freq: Optional[Dict[str, int]] = None, corpus_total: int = 1,
) -> float:
    w = word.lower().strip()
    if not w or not w[0].isalpha():
        return 10.0
    if w in aoa:
        return aoa[w]
    if w in _EARLY_WORDS:
        return _EARLY_WORDS[w]

    n_chars = len(w)
    n_syl = _count_syllables(w)
    morph = _morpheme_complexity(w)
    bigram_f = _bigram_familiarity(w)
    neigh = _ortho_neighborhood_size(w, aoa)

    if corpus_freq and w in corpus_freq:
        rel_freq = corpus_freq[w] / max(corpus_total, 1)
        log_freq = math.log(1 + rel_freq * 1_000_000)
    else:
        log_freq = 0.0

    intercept, β_len, β_syl, β_morph, β_big, β_freq, β_neigh = 8.5, 0.30, 0.55, 2.80, 1.60, 0.18, 0.40

    estimated = (
        intercept + β_len * (n_chars - 5) + β_syl * (n_syl - 2) + β_morph * morph
        - β_big * bigram_f - β_freq * log_freq - β_neigh * math.log(1 + neigh)
    )

    return float(max(2.0, min(20.0, estimated)))

def word_age(aoa: Dict[str, float], token: str, corpus_freq: Optional[Dict[str, int]] = None, corpus_total: int = 1) -> float:
    return calculate_word_age(token, aoa, corpus_freq, corpus_total)

def age_continuity_boost(age1: float, age2: float, strength: float = 0.12) -> float:
    d = abs(age1 - age2)
    early = min(age1, age2, 8.0) / 8.0
    return float(strength * math.exp(-d / 3.0) * early)


# ────────────────────────────────────────────────────────────────────────────
# TOPOLOGICAL & SEMANTIC SCALARS
# ────────────────────────────────────────────────────────────────────────────

def topo_weight(token: str) -> float:
    tl = token.lower()
    base = min(1.0, sum(0.4 for kw in TOPO_KEYWORDS if kw in tl))
    length_presence = 0.05 * length_alpha(token)
    raw = base + length_presence
    return float(min(1.0, raw * length_topo_kernel(token)))

def semantic_scalar(t1: str, t2: str) -> float:
    n = max(len(t1), len(t2), 1)
    dist = abs(len(t1) - len(t2))
    return float(1.0 - dist / n)

def centroid_boost(
    aoa: Dict[str, float], current: str, candidates: List[str], strength: float = 0.10,
    corpus_freq: Optional[Dict[str, int]] = None, corpus_total: int = 1,
) -> np.ndarray:
    cs_topo = topo_weight(current)
    cs_age = word_age(aoa, current, corpus_freq, corpus_total)
    boosts = np.zeros(len(candidates), dtype=np.float32)

    for i, c in enumerate(candidates):
        sim = semantic_scalar(current, c)
        tw = (topo_weight(c) + cs_topo) * 0.5
        ab = age_continuity_boost(cs_age, word_age(aoa, c, corpus_freq, corpus_total))
        boosts[i] = strength * sim * (1.0 + tw + ab) / 3.0

    return boosts


# ────────────────────────────────────────────────────────────────────────────
# LENGTH-DEPENDENT EMBEDDER
# ────────────────────────────────────────────────────────────────────────────

class LengthDependentEmbedder:
    def embed(self, token: str, dim: Optional[int] = None) -> np.ndarray:
        d = dim if dim is not None else length_dim(token)
        raw_bytes = hashlib.sha256(token.encode("utf-8")).digest()
        repeated = (raw_bytes * ((d // 32) + 2))[:d]
        vec = np.array(list(repeated), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)

    def shift_vector(self, token: str, dim: int, magnitude: float) -> np.ndarray:
        raw_bytes = hashlib.md5(token.encode("utf-8")).digest()
        repeated = (raw_bytes * ((dim // 16) + 2))[:dim]
        vec = np.array(list(repeated), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-8)) * magnitude

    @staticmethod
    def _norm01(arr: np.ndarray) -> np.ndarray:
        mn = float(arr.min())
        mx = float(arr.max())
        return (arr - mn) / (mx - mn + 1e-12)

    def length_dependent_weights(
        self, w1: str, w2: str, candidates: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(candidates)
        pass1_raw = np.zeros(N, dtype=np.float32)
        pass2_raw = np.zeros(N, dtype=np.float32)
        topo_kernels = np.zeros(N, dtype=np.float32)

        anchor_shift_mag = length_shift_mag(w2)
        anchor_agree_bonus = length_agreement_bonus(w2)

        for i, c in enumerate(candidates):
            dim = length_dim(c)
            e_w2 = self.embed(w2, dim=dim)
            e_c = self.embed(c, dim=dim)
            shift = self.shift_vector(w1, dim=dim, magnitude=anchor_shift_mag)

            e_w2_shifted = e_w2 + shift
            norm_s = float(e_w2_shifted.sum())
            e_w2_shifted = e_w2_shifted / (abs(norm_s) + 1e-8)

            pass1_raw[i] = float(np.dot(e_w2, e_c))
            pass2_raw[i] = float(np.dot(e_w2_shifted, e_c))
            topo_kernels[i] = length_topo_kernel(c)

        p1 = self._norm01(pass1_raw)
        p2 = self._norm01(pass2_raw)
        de_score = np.minimum(p1, p2)

        base_combined = 0.5 * (p1 + p2)
        agreement_part = float(anchor_agree_bonus) * de_score
        combined = base_combined + topo_kernels * agreement_part
        return p1, p2, self._norm01(combined)


# ────────────────────────────────────────────────────────────────────────────
# N-GRAM LANGUAGE MODEL
# ────────────────────────────────────────────────────────────────────────────

class NGramLM:
    def __init__(self, add_k: float = 1.5):
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0

    def ingest(self, tokens: List[str]) -> None:
        for t in tokens:
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1

        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i + 1])
            self.bi[k] = self.bi.get(k, 0) + 1

        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.tri[k] = self.tri.get(k, 0) + 1

        self.vocab = list(self.uni.keys())

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        cands: List[str] = []
        for (a, b, c) in self.tri:
            if a == w1 and b == w2:
                cands.append(c)

        if not cands:
            for (a, b) in self.bi:
                if a == w2:
                    cands.append(b)

        if not cands:
            cands = [w for w, _ in sorted(self.uni.items(), key=lambda x: -x[1])[:150]]

        seen, out = set(), []
        for w in cands:
            if w not in seen and w not in COGNITIVE_TOKENS:
                seen.add(w)
                out.append(w)

        cands = out[:400]
        V = len(self.vocab) + 1
        k = self.add_k

        def prob(w3: str) -> float:
            c12 = self.bi.get((w1, w2), 0)
            c123 = self.tri.get((w1, w2, w3), 0)
            if c12 > 0:
                return (c123 + k) / (c12 + k * V)
            return (self.uni.get(w3, 0) + k) / (self.total + k * V)

        probs = torch.tensor([prob(w) for w in cands], dtype=torch.float32)
        return cands, probs / (probs.sum() + 1e-12)


# ────────────────────────────────────────────────────────────────────────────
# TOKENIZER & DETOKENIZER
# ────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\[[A-Z\-]+\]|[A-Za-z][A-Za-z0-9_'-]*|[.,;:!?()]")

def tokenize(text: str) -> List[str]:
    text = text.replace("\\n", " ")
    tokens = _TOKEN_RE.findall(text)
    out: List[str] = []
    for t in tokens:
        if t in COGNITIVE_TOKENS:
            out.append(t)
        elif re.match(r"[A-Za-z]", t):
            out.append(t.lower())
        elif t in ".,;:!?()":
            out.append(t)
    return out

def detokenize(tokens: List[str]) -> str:
    out: List[str] = []
    for t in tokens:
        if t in COGNITIVE_TOKENS:
            continue
        elif t in ".,;:?)":
            if out:
                out[-1] += t
        elif t == "(":
            out.append(t)
        else:
            if out and out[-1].endswith("("):
                out[-1] += t
            else:
                out.append(t)
    s = " ".join(out)
    return re.sub(r"([a-z])", lambda m: m.group(1), s)


# ────────────────────────────────────────────────────────────────────────────
# CORPUS STATE
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusState:
    lm: NGramLM
    embedder: LengthDependentEmbedder
    aoa: Dict[str, float]
    riddle_plan: RiddleGenerationPlan = field(default_factory=RiddleGenerationPlan)
    token_boost: Dict[str, float] = field(default_factory=dict)
    corpus_freq: Dict[str, int] = field(default_factory=dict)
    corpus_total: int = 1


def build_state(text: str, aoa: Dict[str, float], prompt: str = "", num_riddles: int = 5) -> CorpusState:
    tokens = tokenize(text)
    lm = NGramLM(add_k=1.5)
    lm.ingest(tokens)

    embedder = LengthDependentEmbedder()
    total = max(1, sum(lm.uni.values()))

    token_boost: Dict[str, float] = {}
    for tok, freq in lm.uni.items():
        if len(tok) > 3 and tok not in STOP_WORDS and re.match(r"^[a-z]", tok):
            token_boost[tok] = min(0.5, math.log(1 + (freq / total) * 1000.0) * 0.1)

    prompt_tokens = tokenize(prompt)
    alpha_tokens = [
        t for t in prompt_tokens
        if len(t) > 2 and re.match(r"^[a-z]", t) and t not in STOP_WORDS
    ]
    if not alpha_tokens:
        alpha_tokens = ["time", "shadow", "echo", "memory"]

    riddle_plan = RiddleGenerationPlan()
    riddle_plan.build_forms(alpha_tokens, num_riddles)

    return CorpusState(
        lm=lm,
        embedder=embedder,
        aoa=aoa,
        riddle_plan=riddle_plan,
        token_boost=token_boost,
        corpus_freq=lm.uni,
        corpus_total=total,
    )


# ────────────────────────────────────────────────────────────────────────────
# SENTENCE GENERATION: Riddle Lines
# ────────────────────────────────────────────────────────────────────────────

def next_probs(
    state: CorpusState, w1: str, w2: str, current_form: Optional[RiddleForm],
    temp: float = 1.2, de_strength: float = 0.18,
) -> Tuple[List[str], torch.Tensor]:
    
    cands, base_probs = state.lm.next_dist(w1, w2)
    _, _, de_combined = state.embedder.length_dependent_weights(w1=w1, w2=w2, candidates=cands)
    de_t = torch.tensor(de_combined, dtype=torch.float32)

    form_boost = torch.zeros_like(de_t)
    if current_form:
        target = current_form.target_word
        for idx, c in enumerate(cands):
            sim = semantic_similarity(target, c)
            form_boost[idx] = 0.35 * sim

    cb = centroid_boost(state.aoa, w2, cands, strength=0.10, corpus_freq=state.corpus_freq, corpus_total=state.corpus_total)
    cb_t = torch.tensor(cb, dtype=torch.float32)
    tb = torch.tensor([state.token_boost.get(c, 0.0) for c in cands], dtype=torch.float32)

    w2_age = word_age(state.aoa, w2, state.corpus_freq, state.corpus_total)
    age_arr = np.array(
        [age_continuity_boost(w2_age, word_age(state.aoa, c, state.corpus_freq, state.corpus_total)) for c in cands],
        dtype=np.float32,
    )
    age_t = torch.tensor(age_arr, dtype=torch.float32)
    topo_kernels = torch.tensor([length_topo_kernel(c) for c in cands], dtype=torch.float32)
    topo_cb = cb_t * (0.5 + 0.5 * topo_kernels)

    boosts = float(de_strength) * de_t + topo_cb + 0.10 * tb + 0.15 * age_t + form_boost
    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
    
    # REASONING: Riddle Constraint
    # Strictly ban the target word and its sub/super-variants from appearing
    # so the riddle maintains its secrecy.
    if current_form:
        target = current_form.target_word
        ban_mask = torch.ones_like(logits, dtype=torch.bool)
        target_stem = target[:4] if len(target) >= 4 else target
        
        for idx, c in enumerate(cands):
            if c == target:
                ban_mask[idx] = False
            elif len(target) >= 4 and target_stem in c:
                ban_mask[idx] = False
            elif len(c) >= 4 and c in target:
                ban_mask[idx] = False
                
        logits[~ban_mask] = -1e9

    logits = logits / max(float(temp), 1e-6)
    return cands, F.softmax(logits, dim=-1)


def generate_riddles(
    state: CorpusState, prompt: str, seed: int = 42, num_riddles: int = 5,
    tokens_per_line: int = 8, temp: float = 1.2,
) -> str:
    rng = np.random.default_rng(int(seed))
    result_text = []

    for r_idx in range(num_riddles):
        forms = state.riddle_plan.riddles.get(r_idx, [])
        if not forms:
            continue
            
        target_word = forms[0].target_word
        result_text.append(f"### Riddle #{r_idx + 1}")
        
        for l_idx, form in enumerate(forms):
            w1, w2 = RIDDLE_PREFIXES.get(form.form_name, ("i", "am"))
            line_tokens = [w1, w2]
            influenced_words = set()
            
            for _ in range(int(tokens_per_line)):
                cands, probs = next_probs(state, w1, w2, current_form=form, temp=float(temp))
                p = probs.detach().cpu().numpy()
                if p.sum() == 0:
                    break
                p = p / (p.sum() + 1e-12)
                tok = cands[int(rng.choice(len(cands), p=p))]
                line_tokens.append(tok)
                
                if semantic_similarity(target_word, tok) > 0.4:
                    influenced_words.add(tok)
                    
                w1, w2 = w2, tok
                
                line_text = detokenize(line_tokens).strip()
                if line_text[-1] in ".!?":
                    break
            
            result_text.append(line_text)
            state.riddle_plan.record_line_generation(r_idx, l_idx, line_text, 1.0, list(influenced_words))
            
        result_text.append(f"**What am I?** [Target: {target_word.capitalize()}]\n")
        
    return "\n".join(result_text)


# ────────────────────────────────────────────────────────────────────────────
# CORPUS LOADING & UI
# ────────────────────────────────────────────────────────────────────────────

def load_corpus(
    use_hf: bool, hf_dataset: str, hf_split: str, hf_max_rows: int, text_file,
    hf_config: str = "", hf_col: str = "", hf_token: str = "",
) -> str:
    if use_hf:
        load_kwargs = {"split": hf_split}
        if hf_config and hf_config.strip():
            load_kwargs["name"] = hf_config.strip()
        if hf_token and hf_token.strip():
            load_kwargs["token"] = hf_token.strip()
        ds = load_dataset(hf_dataset, **load_kwargs)
        rows = min(int(hf_max_rows) if int(hf_max_rows) > 0 else len(ds), len(ds))
        col = hf_col.strip() if hf_col.strip() and hf_col.strip() in ds.column_names else (
            "text" if "text" in ds.column_names else ds.column_names[0]
        )
        return "\n".join(str(x) for x in ds.select(range(rows))[col])

    if text_file is None:
        raise ValueError("No file provided.")
    path = text_file if isinstance(text_file, str) else (
        text_file.name if hasattr(text_file, "name") else str(text_file.get("path", ""))
    )
    return Path(path).read_text(encoding="utf-8", errors="replace")


def run_session(
    use_hf, hf_dataset, hf_split, hf_max_rows, hf_config, hf_col, hf_token, text_file,
    prompt, seed, num_riddles, tokens_per_line, temp, progress=gr.Progress(),
):
    try:
        progress(0.05, desc="Loading AoA dataset (Kuperman 2012)…")
        aoa = load_aoa_dataset()

        progress(0.15, desc="Loading corpus…")
        text = load_corpus(
            bool(use_hf), str(hf_dataset), str(hf_split), int(hf_max_rows),
            text_file, hf_config=str(hf_config), hf_col=str(hf_col), hf_token=str(hf_token),
        )

        progress(0.30, desc="Building language model and riddle plan…")
        state = build_state(text, aoa, prompt=str(prompt), num_riddles=int(num_riddles))

        progress(0.50, desc="Generating riddles…")
        sentences = generate_riddles(
            state, str(prompt), seed=int(seed), num_riddles=int(num_riddles),
            tokens_per_line=int(tokens_per_line), temp=float(temp),
        )

        progress(0.80, desc="Analyzing target activation…")
        form_report = state.riddle_plan.generate_report()

        return sentences, form_report

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}", ""

def toggle_hf(val):
    return (
        gr.update(visible=val), gr.update(visible=val), gr.update(visible=val),
        gr.update(visible=val), gr.update(visible=val), gr.update(visible=val),
        gr.update(visible=not val),
    )


def build_app():
    with gr.Blocks(title="NeuroSymbolic Riddle Generator V9.0", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# NeuroSymbolic Riddle Generator V9.0")

        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face Dataset", value=True)
                hf_dataset = gr.Textbox(label="HF Dataset", value="AiresPucrs/stanford-encyclopedia-philosophy")
                hf_split = gr.Textbox(label="Split", value="train")
                hf_max_rows = gr.Slider(0, 2000, value=300, step=100, label="Max rows (0 = all)")
                hf_config = gr.Textbox(label="Dataset Config / Subset", value="")
                hf_col = gr.Textbox(label="Text Column Override", value="")
                hf_token = gr.Textbox(label="HF Token", value="", type="password")
                text_file = gr.File(label="Upload .txt/.md", file_types=[".txt", ".md"], visible=False)
                use_hf.change(
                    toggle_hf, [use_hf],
                    [hf_dataset, hf_split, hf_max_rows, hf_config, hf_col, hf_token, text_file]
                )

                seed = gr.Number(value=42, label="Seed")
                num_riddles = gr.Slider(1, 50, value=5, step=1, label="Number of Riddles")
                tokens_per_line = gr.Slider(4, 300, value=80, step=1, label="Tokens per Line")
                temp = gr.Slider(0.8, 2.5, value=1.7, step=0.1, label="Temperature")

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt (Extracts target words for riddles)",
                    value="Consider the nature of understanding",
                    lines=2,
                )
                btn = gr.Button("Construct Riddles", variant="primary", size="lg")

                gr.Markdown("## Generated Riddles")
                output_sentences = gr.Textbox(label="Riddles", lines=40)

                gr.Markdown("## Form Activation Analysis")
                output_report = gr.Textbox(label="Form Report", lines=40)

        btn.click(
            run_session,
            inputs=[
                use_hf, hf_dataset, hf_split, hf_max_rows,
                hf_config, hf_col, hf_token, text_file,
                prompt, seed, num_riddles, tokens_per_line, temp
            ],
            outputs=[output_sentences, output_report],
        )

        gr.Markdown(
            "### Key Features\n"
            "- **Riddle Constraints:** Output lines are structurally seeded (e.g., 'I am', 'Yet I').\n"
            "- **Target Masking:** Target word sets the semantic centroid but is strictly banned from generation.\n"
            "- **Cohomological Topology:** Embedding dim derived from β₁ Betti number of n-gram simplicial complex.\n"
            "- **Age-of-Acquisition (AoA):** Kuperman 2012 dataset + regression model.\n"
            "- **Double-Entendre Embedder:** Two-pass similarity for robustness."
        )

        return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(share=False)
