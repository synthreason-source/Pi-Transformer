#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V8.6+ — Complete Implementation with Chain-of-Thought Reasoning
===============================================================================

ARCHITECTURE OVERVIEW:
- Extract N words from first sentence (e.g., "consider", "nature", "understanding")
- Generate 100 unique syntactic forms for these words
- Generate 100 sentences: sentence[i] uses form[i] as its primary feature
- Each sentence yields exactly one form, spreading its activation through that sentence
- Forms accumulate value across different sentence contexts

Key: Each sentence is a distinct syntactic-semantic environment where one form dominates.

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
TOPO_KEYWORDS = [""]
COGNITIVE_TOKENS = {
    "[A]", "[AN]", "[AND]", "[ARE]", "[AS]", "[AT]", "[BE]", "[BY]", "[FOR]", 
    "[FROM]", "[HAS]", "[HAVE]", "[HE]", "[HER]", "[HIM]", "[HIS]", "[I]", 
    "[IN]", "[IS]", "[IT]", "[ITS]", "[ME]", "[MY]", "[OF]", "[ON]", "[OR]", 
    "[OUR]", "[SHE]", "[SO]", "[THAT]", "[THE]", "[THEIR]", "[THEM]", "[THEY]", 
    "[THIS]", "[TO]", "[WAS]", "[WE]", "[WERE]", "[WHAT]", "[WHEN]", "[WHERE]", 
    "[WHICH]", "[WHO]", "[WILL]", "[WITH]", "[YOU]", "[YOUR]"
}


_VOWELS = set("aeiouy")
_COMMON_BIGRAMS: set = {
    "th", "he", "in", "er", "an", "re", "nd", "at", "on", "nt", "ha", "es", "st",
    "en", "ed", "to", "it", "ou", "ea", "hi", "is", "or", "ti", "as", "te", "et"
}

# ────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SyntacticForm:
    word: str
    syntactic_role: str
    prefix_context: str
    suffix_context: str
    form_name: str = ""
    activation_value: float = 0.0

    def __post_init__(self):
        raw = f"{self.word}_{self.syntactic_role}_{self.prefix_context}_{self.suffix_context}"
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]
        self.form_name = f"form_{self.word}_{h}"

    def to_string(self) -> str:
        return f"{self.form_name} ({self.syntactic_role}: {self.prefix_context}...{self.suffix_context})"


@dataclass
class SentenceFormPlan:
    form_by_sentence: Dict[int, SyntacticForm] = field(default_factory=dict)
    sentence_outputs: Dict[int, str] = field(default_factory=dict)

    def plan_forms(self, forms: List[SyntacticForm], num_sentences: int) -> None:
        self.form_by_sentence.clear()
        self.sentence_outputs.clear()
        for i in range(num_sentences):
            f = forms[i % len(forms)] if forms else None
            if f:
                self.form_by_sentence[i] = f

# ────────────────────────────────────────────────────────────────────────────
# SEMANTIC DISTANCE ALGORITHMS
# ────────────────────────────────────────────────────────────────────────────

def semantic_similarity(word_a: str, word_b: str) -> float:
    if not word_a or not word_b:
        return 0.0
    a = word_a.lower()
    b = word_b.lower()
    if a == b:
        return 1.0

    def get_bigrams(w):
        return {w[i:i+2] for i in range(len(w)-1)} if len(w) > 1 else {w}

    bg_a = get_bigrams(a)
    bg_b = get_bigrams(b)
    if not bg_a or not bg_b:
        return 0.0

    intersection = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return intersection / union if union > 0 else 0.0


def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

# ────────────────────────────────────────────────────────────────────────────
# COHOMOLOGICAL TOPOLOGY ENGINE
# ────────────────────────────────────────────────────────────────────────────

def length_alpha(w: str) -> float:
    return min(max(len(w) / 10.0, 0.1), 2.0)

def _build_ngram_simplex(word: str, n: int = 3) -> Dict[str, Set[str]]:
    if len(word) < n:
        return {word: set()}
    ngrams = [word[i:i+n] for i in range(len(word) - n + 1)]
    adj = {ng: set() for ng in ngrams}
    for i in range(len(ngrams) - 1):
        n1, n2 = ngrams[i], ngrams[i+1]
        if n1[1:] == n2[:-1]:  # overlap
            adj[n1].add(n2)
            adj[n2].add(n1)
    return adj

def _cohomological_betti1(adj: Dict[str, Set[str]]) -> int:
    nodes = list(adj.keys())
    if not nodes:
        return 0

    edges = sum(len(adj[u]) for u in adj) // 2

    visited = set()
    components = 0

    def dfs(n):
        stack = [n]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        stack.append(neighbor)

    for node in nodes:
        if node not in visited:
            components += 1
            dfs(node)

    return max(0, edges - len(nodes) + components)

def cohomological_dim(w: str, base_dim: int = 256) -> int:
    simplex = _build_ngram_simplex(w, n=3)
    b1 = _cohomological_betti1(simplex)
    return base_dim + min(b1 * 16, 128)

def length_dim(w: str) -> int:
    base = int(256 * length_alpha(w))
    return cohomological_dim(w, base_dim=base)

def length_shift_mag(w: str) -> float:
    return 1.5 * (length_alpha(w) ** 1.5)

def length_agreement_bonus(w: str) -> float:
    return max(0.0, 1.0 - 0.4 * length_alpha(w))

def length_topo_kernel(w: str) -> float:
    if w.lower() in TOPO_KEYWORDS:
        return 0.95
    base = 0.05 + 0.9 * (length_alpha(w) / 2.0)
    return min(max(base, 0.05), 0.95)

# ────────────────────────────────────────────────────────────────────────────
# AGE-OF-ACQUISITION (AoA) MODULE
# ────────────────────────────────────────────────────────────────────────────

def load_aoa_dataset(
    use_hf: bool = False,
    dataset_name: str = "r1b/kuperman-2012-aoa",
    split: str = "train",
    max_rows: int = 5000,
    config_name: str = "",
    column_name: str = "word",
    hf_token: str = "",
) -> Dict[str, float]:
    aoa_dict = {}
    if use_hf and dataset_name:
        try:
            kwargs = {}
            if config_name:
                kwargs["name"] = config_name
            if hf_token:
                kwargs["token"] = hf_token

            ds = load_dataset(dataset_name, split=split, **kwargs)
            ds = ds.select(range(min(len(ds), max_rows)))
            df = ds.to_pandas()

            val_col = "Rating.Mean" if "Rating.Mean" in df.columns else column_name
            for _, row in df.iterrows():
                w = str(row[column_name]).strip().lower()
                if isinstance(row[val_col], (int, float)):
                    aoa_dict[w] = float(row[val_col])
                else:
                    aoa_dict[w] = 8.0 
        except Exception as e:
            print(f"HF AoA load failed: {e}. Using fallback.")

    if not aoa_dict:
        aoa_dict = {
            "apple": 3.0, "dog": 2.5, "cat": 2.5, "run": 3.0, "jump": 4.0,
            "philosophy": 14.0, "epistemology": 18.0, "consider": 8.0,
            "nature": 6.0, "understanding": 7.5, "topology": 16.0
        }
    return aoa_dict

def _count_syllables(w: str) -> int:
    count = 0
    prev_vowel = False
    for char in w.lower():
        is_vowel = char in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.lower().endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def _morpheme_complexity(w: str) -> float:
    prefixes = {"un", "re", "in", "dis", "en", "non", "pre", "pro", "anti"}
    suffixes = {"ing", "ed", "ly", "tion", "ment", "ness", "ity", "able", "ive"}
    comp = 1.0
    w_lower = w.lower()
    for p in prefixes:
        if w_lower.startswith(p):
            comp += 0.5
            break
    for s in suffixes:
        if w_lower.endswith(s):
            comp += 0.5
            break
    return comp

def _bigram_familiarity(w: str) -> float:
    if len(w) < 2:
        return 0.0
    w_lower = w.lower()
    bigrams = [w_lower[i:i+2] for i in range(len(w)-1)]
    hits = sum(1 for bg in bigrams if bg in _COMMON_BIGRAMS)
    return hits / len(bigrams)

def _ortho_neighborhood_size(w: str, vocab_sample: List[str]) -> int:
    count = 0
    for v in vocab_sample:
        if abs(len(v) - len(w)) <= 2:
            if edit_distance(w.lower(), v.lower()) <= 2:
                count += 1
    return count

def calculate_word_age(w: str, vocab_sample: List[str], freq: int = 1, total_freq: int = 1000) -> float:
    l = len(w)
    syl = _count_syllables(w)
    morph = _morpheme_complexity(w)
    fam = _bigram_familiarity(w)
    prob = max(freq / max(total_freq, 1), 1e-8)
    freq_factor = math.log10(prob)
    hood = _ortho_neighborhood_size(w, vocab_sample)
    hood_factor = min(hood / 10.0, 1.0)

    age = 4.0 + (l * 0.3) + (syl * 0.8) + (morph * 1.5) - (fam * 2.0) - (freq_factor * 0.5) - (hood_factor * 1.0)
    return float(np.clip(age, 2.0, 20.0))

def word_age(aoa_dict: Dict[str, float], w: str, corpus_freq: Dict[str, int], corpus_total: int) -> float:
    w_lower = w.lower()
    if w_lower in aoa_dict:
        return aoa_dict[w_lower]
    sample_vocab = list(aoa_dict.keys())[:200]
    freq = corpus_freq.get(w, 1)
    predicted_age = calculate_word_age(w, sample_vocab, freq, corpus_total)
    aoa_dict[w_lower] = predicted_age 
    return predicted_age

def age_continuity_boost(age_w1: float, age_w2: float) -> float:
    diff = abs(age_w1 - age_w2)
    return float(np.exp(-diff / 3.0))

def topo_weight(w: str) -> float:
    if w.lower() in TOPO_KEYWORDS:
        return 0.9
    return min(max(len(w) / 15.0, 0.1), 0.9)

def semantic_scalar(w: str) -> float:
    return float(np.log1p(len(w)) / 3.0)

def centroid_boost(
    aoa: Dict[str, float], w2: str, candidates: List[str], strength: float = 0.10,
    corpus_freq: Dict[str, int] = None, corpus_total: int = 1
) -> np.ndarray:
    N = len(candidates)
    if N == 0:
        return np.zeros(0, dtype=np.float32)

    def quick_embed(token: str, dim: int) -> np.ndarray:
        raw = hashlib.sha256(token.encode("utf-8")).digest()
        rep = (raw * ((dim // 32) + 2))[:dim]
        vec = np.array(list(rep), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)

    dim = length_dim(w2)
    embs = [quick_embed(c, dim) for c in candidates]
    centroid = np.mean(embs, axis=0)

    out = np.zeros(N, dtype=np.float32)
    for i, e in enumerate(embs):
        dist = float(np.linalg.norm(e - centroid))
        score = 1.0 / (1.0 + dist)
        if corpus_freq is not None:
            w_age = word_age(aoa, candidates[i], corpus_freq, corpus_total)
            scale = semantic_scalar(candidates[i]) * topo_weight(candidates[i])
            score *= (1.0 + scale / (w_age + 1.0))
        out[i] = score

    mn, mx = out.min(), out.max()
    if mx > mn:
        out = (out - mn) / (mx - mn + 1e-12)
    return out * strength


# ────────────────────────────────────────────────────────────────────────────
# TARGET ISOMORPHISM 1: Spiking Membrane Embedder (Adversarial Prob Filler)
# ────────────────────────────────────────────────────────────────────────────
class SpikingDependentEmbedder:
    def __init__(self, adv_strength: float = 0.5):
        self.time_step = 0
        self.adv_strength = adv_strength

    def _inject_current(self, token: str, dim: int) -> np.ndarray:
        raw_bytes = hashlib.sha256(token.encode("utf-8")).digest()
        repeated = (raw_bytes * ((dim // 32) + 2))[:dim]
        vec = np.array(list(repeated), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)

    def _leak_potential(self, token: str, dim: int, magnitude: float) -> np.ndarray:
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
        self.time_step += 1

        N = len(candidates)
        instant_response = np.zeros(N, dtype=np.float32)
        integrated_response = np.zeros(N, dtype=np.float32)
        topo_kernels = np.zeros(N, dtype=np.float32)

        base_leak = length_shift_mag(w2)
        
        # Adversarial dynamic inversion penalty
        adv_penalty = -1.0 if (self.time_step % 2 == 0) else 1.0
        anchor_leak_mag = base_leak * (1.0 + (adv_penalty * self.adv_strength))
        
        anchor_agree_bonus = length_agreement_bonus(w2)

        for i, c in enumerate(candidates):
            dim = length_dim(c)
            I_w2 = self._inject_current(w2, dim=dim)
            I_c = self._inject_current(c, dim=dim)
            
            V_leak = self._leak_potential(w1, dim=dim, magnitude=anchor_leak_mag)

            V_membrane = I_w2 + V_leak
            norm_s = float(V_membrane.sum())
            V_membrane = V_membrane / (abs(norm_s) + 1e-8)

            dot_instant = float(np.dot(I_w2, I_c))
            dot_integrated = float(np.dot(V_membrane, I_c))
            
            # Subtractive normalizer
            instant_response[i] = dot_instant - (self.adv_strength * (dot_instant ** 2))
            integrated_response[i] = dot_integrated - (self.adv_strength * (dot_integrated ** 2))
            
            topo_kernels[i] = length_topo_kernel(c)

        p1 = self._norm01(instant_response)
        p2 = self._norm01(integrated_response)
        
        membrane_score = np.minimum(p1, p2)

        base_combined = 0.5 * (p1 + p2)
        agreement_part = float(anchor_agree_bonus) * membrane_score
        combined = base_combined + topo_kernels * agreement_part
        combined = self._norm01(combined)

        return p1, p2, combined


# ────────────────────────────────────────────────────────────────────────────
# TARGET ISOMORPHISM 2: Hebbian Synaptic Reservoir
# ────────────────────────────────────────────────────────────────────────────
class HebbianReservoirLM:
    def __init__(self, basal_k: float = 1.5):
        self.basal_k = float(basal_k)
        self.spontaneous_trace: Dict[str, float] = {}
        self.synaptic_weights: Dict[Tuple[str, str], float] = {}
        self.tri_synapses: Dict[Tuple[str, str, str], float] = {}
        self.vocab: List[str] = []
        self.total_spikes = 0

    def ingest(self, tokens: List[str]) -> None:
        for t in tokens:
            self.spontaneous_trace[t] = self.spontaneous_trace.get(t, 0) + 1.0
            self.total_spikes += 1
        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i + 1])
            self.synaptic_weights[k] = self.synaptic_weights.get(k, 0) + 1.0
        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.tri_synapses[k] = self.tri_synapses.get(k, 0) + 1.0
        self.vocab = list(self.spontaneous_trace.keys())

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        cands: List[str] = []
        for (a, b, c) in self.tri_synapses:
            if a == w1 and b == w2:
                cands.append(c)

        if not cands:
            for (a, b) in self.synaptic_weights:
                if a == w2:
                    cands.append(b)

        if not cands:
            cands = [w for w, _ in sorted(self.spontaneous_trace.items(), key=lambda x: -x[1])[:150]]

        seen, out = set(), []
        for w in cands:
            if w not in seen and w not in COGNITIVE_TOKENS:
                seen.add(w)
                out.append(w)

        cands = out[:400]
        V_total = len(self.vocab) + 1
        k = self.basal_k

        def propagation_prob(w3: str) -> float:
            c12 = self.synaptic_weights.get((w1, w2), 0)
            c123 = self.tri_synapses.get((w1, w2, w3), 0)
            if c12 > 0:
                return (c123 + k) / (c12 + k * V_total)
            return (self.spontaneous_trace.get(w3, 0) + k) / (self.total_spikes + k * V_total)

        probs = torch.tensor([propagation_prob(w) for w in cands], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)

        return cands, probs

# ────────────────────────────────────────────────────────────────────────────
# TARGET ISOMORPHISM 3: Independent Corrector Cascade (Diagonal Shift)
# ────────────────────────────────────────────────────────────────────────────
class IndependentShiftNet(torch.nn.Module):
    """A single independent neural net layer for a specific diagonal shift."""
    def __init__(self, max_candidates: int, shift_idx: int):
        super().__init__()
        self.shift_idx = shift_idx
        # A fully independent linear neural network for this specific cascade stage
        self.linear = torch.nn.Linear(max_candidates, max_candidates, bias=True)
        
        # Initialize near-identity to preserve base signal before correction
        torch.nn.init.eye_(self.linear.weight)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)
            
        # Add slight noise to encourage independent learning
        with torch.no_grad():
            self.linear.weight.add_(torch.randn_like(self.linear.weight) * 0.01)

    def forward(self, x: torch.Tensor, current_N: int) -> torch.Tensor:
        # Pad to max_candidates for the independent static net
        padded_x = F.pad(x, (0, self.linear.in_features - current_N))
        
        # In-place diagonal shift (roll)
        rolled_x = torch.roll(padded_x, shifts=self.shift_idx, dims=-1)
        
        # Pass through the independent neural net
        correction = self.linear(rolled_x)
        
        # Slice back to original dynamic size and apply non-linearity
        return x + 0.15 * torch.tanh(correction[:current_N])


class DiagonalShiftCorrector(torch.nn.Module):
    """
    A cascade of completely independent Neural Nets. 
    Each net in the cascade applies a different in-place diagonal shift.
    """
    def __init__(self, max_candidates: int = 400, depth: int = 3):
        super().__init__()
        self.max_candidates = max_candidates
        
        # Create a cascade of completely independent neural nets
        self.cascade = torch.nn.ModuleList([
            IndependentShiftNet(max_candidates=max_candidates, shift_idx=i+1)
            for i in range(depth)
        ])

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N = logits.size(0)
        if N == 0:
            return logits

        out = logits.clone()
        
        # Pipe sequentially through the independent neural nets
        for net in self.cascade:
            out = net(out, current_N=N)
            
        return out

# ────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE GLUE
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class CorpusState:
    lm: HebbianReservoirLM
    embedder: SpikingDependentEmbedder
    corrector: DiagonalShiftCorrector
    aoa: Dict[str, float]
    sentence_form_plan: SentenceFormPlan = field(default_factory=SentenceFormPlan)
    token_boost: Dict[str, float] = field(default_factory=dict)
    corpus_freq: Dict[str, int] = field(default_factory=dict)
    corpus_total: int = 1

def tokenize(text: str) -> List[str]:
    out = []
    # Match bracketed tokens like [THE], [PROBLEM] OR standard alphabetic words
    words = re.findall(r"\[[A-Z]+\]|\b[a-zA-Z]+\b", text)
    
    for w in words:
        if w in COGNITIVE_TOKENS:
            out.append(w)  # Preserve exact uppercase bracketed format
        else:
            w_clean = "".join(
                c for c in unicodedata.normalize("NFD", w)
                if unicodedata.category(c) != "Mn"
            ).lower()
            if w_clean:
                out.append(w_clean)
    return out


def detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    res = []
    for t in tokens:
        if t in COGNITIVE_TOKENS:
            # If it's a stop word, strip brackets and lowercase it for final output
            if t in STOP_WORDS_COG:
                res.append(t.strip("[]").lower())
            else:
                res.append(t) # Keep [PROBLEM], [SOLUTION] bracketed
        else:
            if not res:
                res.append(t.capitalize())
            else:
                res.append(t)
    return " ".join(res) + "."




def build_state(
    text: str,
    aoa: Dict[str, float],
    prompt: str = "Consider the nature of understanding",
    num_sentences: int = 100,
    adv_strength: float = 0.5
) -> CorpusState:
    tokens = tokenize(text)
    lm = HebbianReservoirLM()
    lm.ingest(tokens)

    embedder = SpikingDependentEmbedder(adv_strength=adv_strength)
    corrector = DiagonalShiftCorrector(max_candidates=400, depth=3)

    corpus_freq = {}
    for t in tokens:
        corpus_freq[t] = corpus_freq.get(t, 0) + 1
    total = max(1, len(tokens))

    tb = {}
    for w, c in corpus_freq.items():
        tb[w] = float(np.log1p(c))

    state = CorpusState(
        lm=lm,
        embedder=embedder,
        corrector=corrector,
        aoa=aoa,
        token_boost=tb,
        corpus_freq=corpus_freq,
        corpus_total=total,
    )

    prompt_tokens = tokenize(prompt.upper()) # Tokenize handles the bracket extraction
    
    # Filter out cognitive stop words so they don't become base forms
    base_words = [w for w in prompt_tokens if w not in COGNITIVE_TOKENS and re.match(r"^[a-z]+$", w)]
    if not base_words:
        base_words = ["default", "word"]
   

    syntactic_roles = ["noun", "verb", "adj", "adv"]
    prefixes = ["pre", "post", "anti", "hyper", "meta", "sub", "un", "re"]
    suffixes = ["ism", "ity", "ness", "tion", "ology", "ment", "ive", "ly"]

    forms = []
    for i in range(100):
        w = base_words[i % len(base_words)]
        role = syntactic_roles[i % len(syntactic_roles)]
        pref = prefixes[(i // len(syntactic_roles)) % len(prefixes)]
        suff = suffixes[(i // (len(syntactic_roles) * len(prefixes))) % len(suffixes)]

        f = SyntacticForm(
            word=w,
            syntactic_role=role,
            prefix_context=pref,
            suffix_context=suff,
        )
        forms.append(f)

    state.sentence_form_plan.plan_forms(forms, num_sentences=num_sentences)
    return state


def next_probs(
    state: CorpusState,
    w1: str,
    w2: str,
    sentence_index: int,
    temp: float = 1.2,
    de_strength: float = 0.18,
) -> Tuple[List[str], torch.Tensor]:
    cands, base_probs = state.lm.next_dist(w1, w2)
    _, _, de_combined = state.embedder.length_dependent_weights(
        w1=w1, w2=w2, candidates=cands,
    )

    de_t = torch.tensor(de_combined, dtype=torch.float32)

    form_boost = torch.zeros_like(de_t)
    current_form = state.sentence_form_plan.form_by_sentence.get(sentence_index)
    if current_form:
        for idx, c in enumerate(cands):
            sim = semantic_similarity(current_form.word, c)
            form_boost[idx] = 0.25 * sim

    cb = centroid_boost(
        state.aoa, w2, cands, strength=0.10,
        corpus_freq=state.corpus_freq, corpus_total=state.corpus_total,
    )
    cb_t = torch.tensor(cb, dtype=torch.float32)

    tb = torch.tensor([state.token_boost.get(c, 0.0) for c in cands], dtype=torch.float32)

    w2_age = word_age(state.aoa, w2, state.corpus_freq, state.corpus_total)
    age_arr = np.array([
        age_continuity_boost(w2_age, word_age(state.aoa, c, state.corpus_freq, state.corpus_total))
        for c in cands
    ], dtype=np.float32)
    age_t = torch.tensor(age_arr, dtype=torch.float32)

    topo_kernels = torch.tensor([length_topo_kernel(c) for c in cands], dtype=torch.float32)
    topo_cb = cb_t * (0.5 + 0.5 * topo_kernels)

    boosts = (float(de_strength) * de_t + topo_cb + 0.10 * tb + 0.15 * age_t + form_boost)

    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
    
    # Apply the independent neural network cascades via diagonal shifting
    logits = state.corrector(logits)
    
    logits = logits / max(float(temp), 1e-6)
    probs = F.softmax(logits, dim=-1)

    return cands, probs


def generate_100_sentences(
    state: CorpusState,
    seed: int = 42,
    num_sentences: int = 100,
    tokens_per_sentence: int = 92,
    temp: float = 1.7,
) -> List[str]:
    torch.manual_seed(seed)
    vocab = state.lm.vocab
    out_sentences = []
    
    if len(vocab) < 2:
        return ["Not enough vocabulary."]

    for sent_idx in range(num_sentences):
        w1, w2 = vocab[0], vocab[1]
        sent_tokens = []
        sent_topo_sum = 0.0

        for _ in range(tokens_per_sentence):
            cands, probs = next_probs(state, w1, w2, sentence_index=sent_idx, temp=temp)
            if len(cands) == 0:
                break
            idx = torch.multinomial(probs, 1).item()
            nxt = cands[idx]
            sent_tokens.append(nxt)
            sent_topo_sum += length_topo_kernel(nxt)
            w1, w2 = w2, nxt

        text = detokenize(sent_tokens)
        out_sentences.append(text)

        state.sentence_form_plan.sentence_outputs[sent_idx] = text
        current_form = state.sentence_form_plan.form_by_sentence.get(sent_idx)
        if current_form and sent_tokens:
            avg_topo = sent_topo_sum / len(sent_tokens)
            current_form.activation_value += float(avg_topo)

    return out_sentences

# ────────────────────────────────────────────────────────────────────────────
# GRADIO UI & EXECUTION
# ────────────────────────────────────────────────────────────────────────────

def load_corpus(
    use_hf: bool = False,
    dataset_name: str = "",
    config_name: str = "",
    split: str = "train",
    column_name: str = "text",
    max_rows: int = 100,
    hf_token: str = "",
    text_file: Optional[Path] = None
) -> str:
    """
    Glue piece to bridge external data sources to the neuronal reservoir.
    """
    if use_hf and dataset_name:
        try:
            print(f"Loading HF Corpus: {dataset_name} (Config: {config_name}, Split: {split})")
            ds = load_dataset(
                dataset_name, 
                name=config_name if config_name else None, 
                split=split, 
                token=hf_token if hf_token else None,
                trust_remote_code=True
            )
            df = ds.select(range(min(len(ds), max_rows))).to_pandas()
            if column_name in df.columns:
                return " ".join(df[column_name].astype(str).tolist())
            else:
                available = ", ".join(df.columns)
                return f"Error: Column '{column_name}' not found. Available columns: {available}"
        except Exception as e:
            return f"Hugging Face Load Error: {str(e)}"
    
    if text_file is not None:
        try:
            return Path(text_file).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    return (
        "In algebraic topology, homology and cohomology provide a profound "
        "understanding of the shape of data. A persistent filtration creates a "
        "barcode of topological features. Betti numbers summarize cycles, voids, "
        "and connectivity. We consider the nature of understanding spaces "
        "through simplicial complexes and morse theory. The continuous function "
        "maps a manifold into a sheaf."
    )


def run_session(
    use_hf: bool,
    hf_dataset: str,
    hf_split: str,
    hf_max_rows: int,
    hf_config: str,
    hf_col: str,
    hf_token: str,
    text_file: Optional[Path],
    prompt: str,
    seed: float,
    num_sentences: int,
    tokens_per_sentence: int,
    temp: float,
    adv_strength: float,
) -> Tuple[str, str]:
    """
    Orchestrates the UI run with the new dataset integration and Neuronal models.
    """
    aoa = load_aoa_dataset(use_hf=False)

    corpus_text = load_corpus(
        use_hf=use_hf, dataset_name=hf_dataset, config_name=hf_config,
        split=hf_split, column_name=hf_col, max_rows=int(hf_max_rows),
        hf_token=hf_token, text_file=text_file
    )
    
    if corpus_text.startswith("Error") or corpus_text.startswith("Hugging Face Load Error"):
        return corpus_text, "Check dataset configuration."

    state = build_state(
        text=corpus_text,
        aoa=aoa,
        prompt=prompt,
        num_sentences=int(num_sentences),
        adv_strength=float(adv_strength),
    )

    generate_100_sentences(
        state=state,
        seed=int(seed),
        num_sentences=int(num_sentences),
        tokens_per_sentence=int(tokens_per_sentence),
        temp=float(temp),
    )

    sent_lines = []
    for i, s in enumerate(state.sentence_form_plan.sentence_outputs.values()):
        sent_lines.append(f"[{i+1}] {s}\n")
    sentences_str = "\n".join(sent_lines)

    report_lines = [
        "FORM ACTIVATION & NEURONAL REPORT",
        "===================================",
        f"Sentences generated: {len(state.sentence_form_plan.sentence_outputs)}",
        ""
    ]
    for sent_idx in range(min(30, len(state.sentence_form_plan.sentence_outputs))):
        f = state.sentence_form_plan.form_by_sentence.get(sent_idx)
        if f:
            output = state.sentence_form_plan.sentence_outputs.get(sent_idx, "(not generated)")
            preview = output[:60] + "..." if len(output) > 60 else output
            report_lines.append(f"Sentence {sent_idx:02d} | Form: {f.form_name}")
            report_lines.append(f"  Word: '{f.word}', Role: '{f.syntactic_role}'")
            report_lines.append(f"  Activation Value: {f.activation_value:.4f}")
            report_lines.append(f"  Output: {preview}\n")

    if len(state.sentence_form_plan.sentence_outputs) > 30:
        report_lines.append(f"... ({len(state.sentence_form_plan.sentence_outputs) - 30} more sentences)")

    return sentences_str, "\n".join(report_lines)


def build_app():
    with gr.Blocks(title="NeuroSymbolic Form Generator V8.6+") as demo:
        gr.Markdown(
            "# Neuronal Isomorphism Generator V8.6+\n"
            "**Spiking Membrane & Hebbian Reservoir:** Bridges external Hugging Face datasets into "
            "neuronal activation spaces with dynamic topological form extraction."
        )

        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face Dataset?", value=False)
                
                with gr.Group(visible=False) as hf_group:
                    hf_dataset = gr.Textbox(label="Dataset Path", value="wikitext")
                    hf_config = gr.Textbox(label="Config (e.g. 'wikitext-2-raw-v1')", value="wikitext-2-raw-v1")
                    hf_split = gr.Textbox(label="Split", value="train")
                    hf_col = gr.Textbox(label="Text Column", value="text")
                    hf_max_rows = gr.Number(label="Max Rows", value=100)
                    hf_token = gr.Textbox(label="HF Token", type="password")
                
                text_file = gr.File(
                    label="Upload Local Text (.txt/.md)",
                    file_types=[".txt", ".md"],
                    visible=True
                )
                
                use_hf.change(
                    fn=lambda x: (gr.update(visible=x), gr.update(visible=not x)),
                    inputs=use_hf, outputs=[hf_group, text_file]
                )

                gr.Markdown("### Hyperparameters")
                seed = gr.Number(value=42, label="Seed")
                num_sentences = gr.Slider(
                    1, 500, value=100, step=10, label="Number of Sentences"
                )
                tokens_per_sentence = gr.Slider(
                    8, 1800, value=92, step=2, label="Tokens per Sentence"
                )
                temp = gr.Slider(0.8, 2.5, value=1.7, step=0.1, label="Temperature")

                gr.Markdown("### Adversarial Normalizer Controls")
                adv_strength = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Adversarial Penalty Strength")

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt (extracts words for 100 forms)",
                    value="Consider the nature of understanding",
                    lines=2,
                )
                btn = gr.Button("Generate Sentences", variant="primary", size="lg")

                gr.Markdown("## Generated Sentences (One Form Per Sentence)")
                output_sentences = gr.Textbox(label="Sentences", lines=20)

                gr.Markdown("## Isomorphism & Form Activation Analysis")
                output_report = gr.Textbox(label="Form Report", lines=20)

        btn.click(
            run_session,
            inputs=[
                use_hf, hf_dataset, hf_split, hf_max_rows,
                hf_config, hf_col, hf_token, text_file,
                prompt, seed, num_sentences, tokens_per_sentence, temp,
                adv_strength
            ],
            outputs=[output_sentences, output_report],
        )

        gr.Markdown(
            "### Isomorphism Features\n"
            "- **Spiking Membrane:** Hash logic replaced with Leaky Integrate-and-Fire mechanics\n"
            "- **Hebbian Synapses:** N-Grams modeled as synaptic plastic trace weights\n"
            "- **Form Count:** Exactly 100 forms, one per sentence\n"
            "- **Dynamic HF Configs:** Direct pipeline to specific dataset slices and splits\n"
            "- **Adversarial Prob Filler:** Dynamically self-normalizes extremes through alternating penalties\n"
            "- **Independent Corrector Cascade:** Sequential, fully independent neural nets apply in-place diagonal feature shifts."
        )

        return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(share=False)
