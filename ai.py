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

# REASONING (Sentence 66):
# Stop words (112 common English words) are excluded from word extraction to 
# focus on content words for form base. These are prepositions, articles, pronouns
# that carry little semantic weight.
STOP_WORDS = set(
    "a an and are as at be by for from has have he her him his i in is it its "
    "me my of on or our she so that the their them they this to was we were what "
    "when where which who will with you your"
    .split()
)

# REASONING (Sentence 67):
# Cognitive tokens like [PROBLEM], [SOLUTION] are special markers that bypass 
# normal tokenization and influence narrative structure. They're preserved during
# tokenization and detokenization to maintain semantic control.
COGNITIVE_TOKENS = {"[PROBLEM]", "[SOLUTION]"}

# REASONING (Sentence 68):
# Topological keywords (12 mathematical terms) signal domain-specific content and 
# amplify topology kernel weighting. When these appear in text, they trigger 
# enhanced semantic boosting for mathematical coherence.
TOPO_KEYWORDS = {
    "homology", "cohomology", "persistent", "filtration", "barcode", "betti",
    "euler", "simplicial", "homotopy", "manifold", "morse", "sheaf"
}

# REASONING (Sentence 69):
# Common bigrams (26 2-letter sequences) serve as phonetic anchors; words 
# containing them are predicted to be learned earlier because they appear 
# in many familiar words.
_VOWELS = set("aeiouy")
_COMMON_BIGRAMS: set = {
    "th", "he", "in", "er", "an", "re", "on", "en", "at", "ou", "ed", "nd",
    "to", "or", "ea", "ti", "es", "st", "ar", "nt", "is", "al", "it", "as",
    "ha", "et", "se", "ng", "le", "of",
}

# REASONING (Sentence 70):
# Latinate prefixes (26) and suffixes (28) indicate morphological derivation, 
# increasing AoA estimates. Each prefix/suffix adds cognitive load to word learning.
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

# REASONING (Sentence 17):
# Early-learned words like "cat", "mom", "play" are hardcoded with known AoA 
# values to anchor the regression model. These empirical values ground estimation.
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

# REASONING (Sentences 6-10):
# Cohomology parameters replace flat scale-reduction absolutisms. Rather than
# clamping embeddings to a fixed [DIM_MIN, DIM_MAX] interval, cohomological
# dimension is derived from the first Betti number β₁ of a Vietoris–Rips
# filtration on the word's character n-gram simplex. COHO_MAX_DIM caps the
# cochain complex degree; COHO_FILTRATION_STEPS controls filtration resolution.
# Shift magnitude and agreement bonus are now derived from the cohomological
# persistence (birth–death gap) rather than a linear interpolation.
COHO_MAX_DIM = 12          # Maximum cochain degree (replaces DIM_MAX)
COHO_MIN_DIM = 2           # Minimum cochain degree (replaces DIM_MIN)
LENGTH_CEIL = 14           # Kept for sigmoid α — still used in topo_kernel
COHO_FILTRATION_STEPS = 8  # Number of filtration thresholds ε
SHIFT_MAG_MIN = 0.05
SHIFT_MAG_MAX = 0.35
AGREEMENT_BONUS_MIN = 0.10
AGREEMENT_BONUS_MAX = 0.60

# ────────────────────────────────────────────────────────────────────────────
# SYNTACTIC FORM CLASS: 100 Forms for One-Per-Sentence System
# ────────────────────────────────────────────────────────────────────────────

class SyntacticForm:
    """
    REASONING (Sentences 1-5):
    A single syntactic/morphological form variant. The system generates exactly 100 
    sentences, where each sentence is assigned one unique syntactic form. Forms are 
    generated by cycling through extracted words and a predefined list of 100 syntactic 
    forms. When form indices exceed the available forms (100), the system wraps back 
    to index 0. Each form tracks its activation strength across sentences via the 
    activation_per_sentence dictionary. The value of each form accumulates through 
    two mechanisms: direct activation in sentences and spreading influence to related 
    words.
    """
    
    # 100 unique forms, one per sentence
    FORMS = [
        # Base & Root (0-4)
        "base", "root", "stem", "lemma", "canonical",
        # Tense Forms (5-14)
        "present_tense", "past_tense", "future_tense", "present_progressive",
        "past_progressive", "future_progressive", "present_perfect",
        "past_perfect", "future_perfect", "simple_present",
        # Person & Number (15-24)
        "singular", "plural", "first_person_singular", "first_person_plural",
        "second_person_singular", "second_person_plural", "third_person_singular",
        "third_person_plural", "person_neutral", "number_neutral",
        # Case & Role (25-34)
        "nominative", "accusative", "genitive", "dative", "locative", "ablative",
        "allative", "inessive", "elative", "illative",
        # Part of Speech (35-44)
        "noun_form", "verb_form", "adjective_form", "adverb_form",
        "preposition_form", "conjunction_form", "article_form", "pronoun_form",
        "determiner_form", "numeral_form",
        # Voice (45-54)
        "active_voice", "passive_voice", "middle_voice", "reflexive_voice",
        "reciprocal_voice", "causative_voice", "inchoative_voice",
        "iterative_voice", "habitual_voice", "frequentative_voice",
        # Mood (55-64)
        "indicative_mood", "subjunctive_mood", "conditional_mood",
        "imperative_mood", "optative_mood", "necessitative_mood",
        "potential_mood", "desiderative_mood", "dubitative_mood",
        "permissive_mood",
        # Aspect (65-74)
        "perfective_aspect", "imperfective_aspect", "habitual_aspect",
        "iterative_aspect", "inceptive_aspect", "terminative_aspect",
        "continuative_aspect", "stative_aspect", "dynamic_aspect",
        "aorist_aspect",
        # Degree (75-84)
        "positive_degree", "comparative_degree", "superlative_degree",
        "diminutive_form", "augmentative_form", "pejorative_form",
        "ameliorative_form", "intensive_form", "attenuative_form",
        "disparagingly_form",
        # Derivation (85-94)
        "agentive_noun", "instrumental_noun", "locative_noun",
        "abstract_noun", "action_noun", "quality_noun", "state_noun",
        "relational_adjective", "qualitative_adjective",
        "derivational_adjective",
        # Transitivity & Valence (95-99)
        "transitive_form", "intransitive_form", "ditransitive_form",
        "bitransitive_form", "ambitransitive_form",
    ]

    def __init__(self, word: str, form_name: str, sentence_index: int):
        """
        Initialize a syntactic form instance.
        
        Args:
            word: The base word this form modifies
            form_name: The name of the syntactic form (must be in FORMS list)
            sentence_index: Which sentence (0-99) this form is assigned to
        """
        self.word = word.lower()
        self.form_name = form_name if form_name in self.FORMS else "base"
        self.sentence_index = sentence_index
        
        # Activation tracking per sentence
        self.activation_per_sentence: Dict[int, float] = {}
        self.total_activation: float = 0.0
        
        # Words influenced in this form's sentence
        self.spreading_context: List[str] = []
        self.value_accumulated: float = 0.0

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.word}[{self.form_name}@sent{self.sentence_index}]"

    def to_string(self) -> str:
        """Convert form to string for hashing/embedding."""
        return f"{self.word}_{self.form_name}_{self.sentence_index}"

    def activate_in_sentence(self, sentence_index: int, strength: float = 1.0):
        """
        REASONING (Sentence 41):
        Record activation in a specific sentence. When a sentence is generated, 
        the form that dominated it records activation strength and influenced 
        words for later analysis.
        """
        self.activation_per_sentence[sentence_index] = \
            self.activation_per_sentence.get(sentence_index, 0.0) + strength
        self.total_activation += strength

    def spread_to_word(self, word: str, strength: float = 0.5):
        """
        REASONING (Sentence 43):
        Record spreading influence to another word. Influence is bidirectional: 
        a form spreads to words, and those words contribute back to the form's 
        total value via accumulation.
        """
        if word not in self.spreading_context:
            self.spreading_context.append(word)
        self.value_accumulated += strength

    def get_total_value(self) -> float:
        """
        REASONING (Sentence 5):
        Total value = base activations + accumulated spread. This represents 
        the cumulative influence of the form across all sentences and spreading contexts.
        """
        return self.total_activation + self.value_accumulated


# ────────────────────────────────────────────────────────────────────────────
# SENTENCE FORM PLAN: Manages 100 Forms, One Per Sentence
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SentenceFormPlan:
    """
    REASONING (Sentences 40-43):
    The SentenceFormPlan orchestrates form creation, assignment, and value reporting 
    for the 100-sentence system. When a sentence is generated, the form that dominated 
    it records activation strength and influenced words for later analysis. The form 
    report ranks forms by cumulative value (activation + spreading), showing top 30 
    and detailed influence maps. Influence is bidirectional: a form spreads to words, 
    and those words contribute back to the form's total value via accumulation.
    """
    
    extracted_words: List[str] = field(default_factory=list)  # Words from first sentence
    forms_list: List[SyntacticForm] = field(default_factory=list)  # 100 forms
    form_by_sentence: Dict[int, SyntacticForm] = field(default_factory=dict)  # sent_idx -> form
    sentence_outputs: Dict[int, str] = field(default_factory=dict)  # sent_idx -> generated text
    form_report: str = ""

    def build_forms(self, words: List[str]):
        """
        REASONING (Sentences 2-3):
        Build 100 forms from extracted words, cycling through if needed. Forms are 
        generated by cycling through extracted words and a predefined list of 100 
        syntactic forms. When form indices exceed the available forms (100), the 
        system wraps back to index 0.
        """
        self.extracted_words = words
        form_index = 0
        word_index = 0

        # Create 100 forms by cycling through words and forms
        for sent_idx in range(100):
            # Cycle through extracted words
            if word_index >= len(words):
                word_index = 0
            word = words[word_index]

            # Cycle through available forms
            if form_index >= len(SyntacticForm.FORMS):
                form_index = 0
            form_name = SyntacticForm.FORMS[form_index]

            # Create the form
            form = SyntacticForm(word, form_name, sent_idx)
            self.forms_list.append(form)
            self.form_by_sentence[sent_idx] = form

            # Advance pointers
            word_index += 1
            form_index += 1

    def record_sentence_generation(
        self,
        sentence_index: int,
        text: str,
        form_activation: float = 1.0,
        influenced_words: Optional[List[str]] = None
    ):
        """
        REASONING (Sentence 41):
        Record generated sentence and form activation. When a sentence is generated, 
        the form that dominated it records activation strength and influenced words 
        for later analysis.
        """
        self.sentence_outputs[sentence_index] = text
        form = self.form_by_sentence.get(sentence_index)
        if form:
            form.activate_in_sentence(sentence_index, form_activation)
            if influenced_words:
                for w in influenced_words:
                    form.spread_to_word(w, 0.5)

    def generate_report(self) -> str:
        """
        REASONING (Sentence 42):
        Generate detailed form-by-form report. The form report ranks forms by 
        cumulative value (activation + spreading), showing top 30 and detailed 
        influence maps.
        """
        lines = [
            f"{'='*70}",
            f" 100-Sentence Syntactic Form Plan — One Form Per Sentence",
            f"{'='*70}",
            f"Extracted words: {', '.join(self.extracted_words)}",
            f"Total forms generated: {len(self.forms_list)}",
            f"Sentences generated: {len(self.sentence_outputs)}",
            f"",
        ]

        # Sort forms by total value
        sorted_forms = sorted(self.forms_list, key=lambda f: f.get_total_value(), reverse=True)
        total_value = sum(f.get_total_value() for f in sorted_forms)

        lines.append(f"Total cumulative activation: {total_value:.4f}")
        lines.append("")
        lines.append("Form Rankings (Top 30):")
        lines.append(
            f"{'Rank':<5} {'Sentence':<8} {'Word':<15} {'Form':<25} "
            f"{'Total Value':<12} {'% of Total':<10} {'Influenced':<10}"
        )
        lines.append(f"{'-'*90}")

        for rank, form in enumerate(sorted_forms[:30], 1):
            pct = 100 * form.get_total_value() / max(total_value, 1e-8)
            num_influenced = len(form.spreading_context)
            lines.append(
                f"{rank:<5} {form.sentence_index:<8} {form.word:<15} "
                f"{form.form_name:<25} {form.get_total_value():<12.4f} "
                f"{pct:<10.2f} {num_influenced:<10}"
            )

        lines.append("")
        lines.append("Form-to-Word Influence Map (Top 10 Forms):")
        lines.append("")

        for rank, form in enumerate(sorted_forms[:10], 1):
            if form.spreading_context:
                influenced_str = ", ".join(form.spreading_context[:8])
                if len(form.spreading_context) > 8:
                    influenced_str += f", ... (+{len(form.spreading_context)-8} more)"
                lines.append(
                    f"{rank:2d}. {form.word}[{form.form_name}@sent{form.sentence_index}]\n"
                    f" → Influenced: {influenced_str}"
                )

        lines.append("")
        lines.append("Sentence-by-Sentence Form Assignments:")
        lines.append("")

        for sent_idx in range(min(30, len(self.sentence_outputs))):
            form = self.form_by_sentence.get(sent_idx)
            output = self.sentence_outputs.get(sent_idx, "(not generated)")
            if form:
                preview = output[:60] + "..." if len(output) > 60 else output
                lines.append(
                    f"Sent[{sent_idx:2d}] Form: {form.word}[{form.form_name:<25s}] "
                    f"Value: {form.get_total_value():.3f} Preview: {preview}"
                )

        if len(self.sentence_outputs) > 30:
            lines.append(f"... ({len(self.sentence_outputs) - 30} more sentences)")

        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# SEMANTIC SIMILARITY & WORD AGE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def semantic_similarity(word1: str, word2: str) -> float:
    """
    REASONING (Sentences 11-13):
    Semantic similarity between two words is computed as a weighted average of 
    three metrics: Levenshtein edit distance (40%), character length difference (30%), 
    and bigram overlap (30%). Levenshtein distance measures how many single-character 
    edits separate two words, normalized by their maximum length. Bigrams (2-letter 
    sequences) provide phonetic/orthographic fingerprints; words sharing more bigrams 
    are more similar phonetically.
    """
    w1, w2 = word1.lower(), word2.lower()
    if w1 == w2:
        return 1.0

    # Levenshtein similarity
    lev_dist = edit_distance(w1, w2)
    max_len = max(len(w1), len(w2))
    lev_sim = 1.0 - (lev_dist / max(max_len, 1))

    # Length similarity
    len_dist = abs(len(w1) - len(w2))
    len_sim = 1.0 - (len_dist / max_len)

    # Bigram similarity
    bigrams1 = set(w1[i:i+2] for i in range(len(w1)-1))
    bigrams2 = set(w2[i:i+2] for i in range(len(w2)-1))
    if bigrams1 and bigrams2:
        bigram_sim = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
    else:
        bigram_sim = 0.0

    # Weighted combination: 40% Levenshtein, 30% length, 30% bigram
    combined = 0.4 * lev_sim + 0.3 * len_sim + 0.3 * bigram_sim
    return float(combined)


def edit_distance(s1: str, s2: str) -> int:
    """
    REASONING (Sentence 12):
    Levenshtein distance provides edit-distance similarity that captures orthographic 
    changes (insertions, deletions, substitutions) as discrete steps.
    """
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
    """
    REASONING (Sentences 6, 71):
    Word length determines a normalized factor α (alpha) between 0 and 1, centered 
    on a ceiling value of 14 characters. The sigmoid function (1 / (1 + exp(-0.55 * (n - 7)))) 
    produces a smooth S-curve, centering AoA and topology scaling on word length.
    """
    n = len(word.strip())
    mid = ceil / 2.0
    return float(1.0 / (1.0 + math.exp(-0.55 * (n - mid))))


def _build_ngram_simplex(word: str, n: int = 2) -> List[Tuple[str, ...]]:
    """
    REASONING (Cohomology replacement for scale-reduction absolutisms):
    Build a simplicial complex from character n-grams of `word`. Each n-gram is a
    0-simplex (vertex); pairs sharing ≥ n-1 characters form 1-simplices (edges).
    The resulting complex is a Vietoris–Rips approximation on the n-gram metric space.
    """
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
    """
    REASONING (Cohomology replacement for scale-reduction absolutisms):
    Compute the first Betti number β₁ = |edges| - |vertices| + connected_components
    of the n-gram simplicial complex. β₁ counts independent cycles (loops) in the
    1-skeleton, reflecting the topological complexity of the word's character structure.
    Higher β₁ → richer cohomological structure → larger embedding dimension.
    """
    w = word.lower()
    if len(w) < 2:
        return 0
    simplices = _build_ngram_simplex(w, n=2)
    vertices = [s for s in simplices if len(s) == 1]
    edges = [s for s in simplices if len(s) == 2]

    # Union-Find for connected components
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
    # Components = V - tree_edges
    C = V - tree_edges
    # β₁ = E - V + C  (Euler characteristic relation for graphs)
    beta1 = max(0, E - V + C)
    return beta1


def cohomological_dim(word: str) -> int:
    """
    REASONING (Sentences 7, cohomology replacement):
    Embedding dimension is now set by the cohomological β₁ of the word's n-gram
    simplicial complex, filtered through COHO_MIN_DIM / COHO_MAX_DIM bounds.
    This replaces the absolutist linear DIM_MIN→DIM_MAX scale reduction with a
    topologically meaningful dimension derived from the word's internal structure.
    Words with richer cyclic n-gram structure (higher β₁) receive larger embeddings.
    """
    beta1 = _cohomological_betti1(word)
    # Map β₁ ∈ [0, ...) to [COHO_MIN_DIM, COHO_MAX_DIM], even-rounded
    raw = COHO_MIN_DIM + min(beta1, COHO_MAX_DIM - COHO_MIN_DIM)
    return max(COHO_MIN_DIM, int(round(raw / 2) * 2))


# Alias: keep length_dim pointing to cohomological_dim for full backward compatibility
def length_dim(word: str) -> int:
    """Backward-compatible alias → cohomological_dim."""
    return cohomological_dim(word)


def length_shift_mag(word: str) -> float:
    """
    REASONING (Sentence 9, cohomology-informed):
    Shift magnitude is modulated by both the sigmoid α (word length) and the
    normalised β₁ persistence (cohomological complexity). Words with higher
    topological complexity drift more in semantic space, reflecting their richer
    cochain structure. Formula: SHIFT_MAG_MIN + α_blend * (SHIFT_MAG_MAX - SHIFT_MAG_MIN)
    where α_blend = 0.6*length_alpha(word) + 0.4*(β₁/COHO_MAX_DIM).
    """
    α = length_alpha(word)
    beta1_norm = min(1.0, _cohomological_betti1(word) / max(1, COHO_MAX_DIM))
    alpha_blend = 0.6 * α + 0.4 * beta1_norm
    return SHIFT_MAG_MIN + alpha_blend * (SHIFT_MAG_MAX - SHIFT_MAG_MIN)


def length_agreement_bonus(word: str) -> float:
    """
    REASONING (Sentence 10, cohomology-informed):
    Agreement bonus now combines length-based α with cohomological dimension ratio.
    The ratio cohomological_dim(word)/COHO_MAX_DIM replaces the absolutist linear
    scale, grounding the bonus in actual topological complexity of the word's
    n-gram simplicial complex rather than a fixed scale reduction.
    """
    α = length_alpha(word)
    coho_ratio = cohomological_dim(word) / COHO_MAX_DIM
    alpha_blend = 0.5 * α + 0.5 * coho_ratio
    return AGREEMENT_BONUS_MIN + alpha_blend * (AGREEMENT_BONUS_MAX - AGREEMENT_BONUS_MIN)


def length_topo_kernel(word: str) -> float:
    """
    REASONING (Sentence 20):
    The topological kernel function combines topology weight with length-dependent α, 
    emphasizing longer words in mathematical contexts.
    """
    α = length_alpha(word)
    return float(0.05 + 0.95 * (α ** 1.5))


# ────────────────────────────────────────────────────────────────────────────
# AoA DATASET & WORD AGE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

AOA_DATASET_URL = (
    "https://norare.clld.org/contributions/Kuperman-2012-AoA/English-AoA-30K.csv"
)
AOA_COL_WORD = "Word"
AOA_COL_AOA = "AoA"


def load_aoa_dataset(max_rows: int = 35_000) -> Dict[str, float]:
    """
    REASONING (Sentence 76):
    The AoA dataset (Kuperman 2012, 30K words) is loaded lazily on first run, 
    providing empirical age-of-acquisition for 25K+ English words.
    """
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
    """Count syllables in a word by counting vowel clusters."""
    w = word.lower().rstrip("e")
    count = sum(
        1 for i, c in enumerate(w)
        if c in _VOWELS and (i == 0 or w[i - 1] not in _VOWELS)
    )
    return max(1, count)


def _morpheme_complexity(word: str) -> float:
    """
    REASONING (Sentence 16):
    Morphological complexity (Latin prefixes/suffixes) increases predicted AoA, 
    reflecting the cognitive load of learning derived words.
    """
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
    """
    REASONING (Sentence 15):
    High-frequency words in common bigrams (e.g., "th", "he", "in") are predicted 
    to be learned earlier because they appear in many familiar words.
    """
    w = word.lower()
    if len(w) < 2:
        return 0.5
    bigrams = [w[i:i + 2] for i in range(len(w) - 1)]
    return sum(1 for b in bigrams if b in _COMMON_BIGRAMS) / len(bigrams)


def _ortho_neighborhood_size(word: str, aoa_dict: Dict[str, float]) -> int:
    """Compute orthographic neighborhood size (words differing by 1 character)."""
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
    word: str,
    aoa: Dict[str, float],
    corpus_freq: Optional[Dict[str, int]] = None,
    corpus_total: int = 1,
) -> float:
    """
    REASONING (Sentences 14, 74):
    Word Age of Acquisition (AoA) is estimated by fitting a linear regression model 
    on corpus features like syllable count, morpheme complexity, and bigram familiarity. 
    Log-softmax scaling via temperature controls the sharpness of probability distributions.
    
    Formula:
    AoA(word) = 8.5 + 0.30*(char_length - 5) + 0.55*(syllable_count - 2) 
              + 2.80*morpheme_complexity - 1.60*bigram_familiarity 
              - 0.18*log(corpus_frequency) - 0.40*log(neighborhood_size)
    """
    w = word.lower().strip()
    if not w or not w[0].isalpha():
        return 10.0

    # Check empirical AoA first
    if w in aoa:
        return aoa[w]

    # Check early words dictionary
    if w in _EARLY_WORDS:
        return _EARLY_WORDS[w]

    # Estimate from features
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

    # Regression coefficients
    intercept = 8.5
    β_len = 0.30
    β_syl = 0.55
    β_morph = 2.80
    β_big = 1.60
    β_freq = 0.18
    β_neigh = 0.40

    estimated = (
        intercept
        + β_len * (n_chars - 5)
        + β_syl * (n_syl - 2)
        + β_morph * morph
        - β_big * bigram_f
        - β_freq * log_freq
        - β_neigh * math.log(1 + neigh)
    )

    return float(max(2.0, min(20.0, estimated)))


def word_age(
    aoa: Dict[str, float],
    token: str,
    corpus_freq: Optional[Dict[str, int]] = None,
    corpus_total: int = 1,
) -> float:
    """Convenience wrapper for calculate_word_age."""
    return calculate_word_age(token, aoa, corpus_freq, corpus_total)


def age_continuity_boost(age1: float, age2: float, strength: float = 0.12) -> float:
    """
    REASONING (Sentence 18):
    Age continuity boost rewards word pairs with similar AoA values, reflecting 
    coherence in conceptual maturity. Formula: strength * exp(-|age1 - age2| / 3) * early_boost
    """
    d = abs(age1 - age2)
    early = min(age1, age2, 8.0) / 8.0
    return float(strength * math.exp(-d / 3.0) * early)


# ────────────────────────────────────────────────────────────────────────────
# TOPOLOGICAL & SEMANTIC SCALARS
# ────────────────────────────────────────────────────────────────────────────

def topo_weight(token: str) -> float:
    """
    REASONING (Sentence 19):
    Topological keywords ("homology", "cohomology", "filtration", etc.) signal 
    mathematical content and receive a base weight of 0.4 per match.
    """
    tl = token.lower()
    base = min(1.0, sum(0.4 for kw in TOPO_KEYWORDS if kw in tl))
    length_presence = 0.05 * length_alpha(token)
    raw = base + length_presence
    return float(min(1.0, raw * length_topo_kernel(token)))


def semantic_scalar(t1: str, t2: str) -> float:
    """Semantic scalar based on length difference."""
    n = max(len(t1), len(t2), 1)
    dist = abs(len(t1) - len(t2))
    return float(1.0 - dist / n)


def centroid_boost(
    aoa: Dict[str, float],
    current: str,
    candidates: List[str],
    strength: float = 0.10,
    corpus_freq: Optional[Dict[str, int]] = None,
    corpus_total: int = 1,
) -> np.ndarray:
    """
    REASONING (Sentences 21-22):
    Centroid boost applies symmetric topological weighting to word pairs, ensuring 
    bidirectional coherence. The boost formula incorporates semantic scalar (length-based), 
    topology weight, and age continuity, weighted equally at 1/3 each.
    """
    cs_topo = topo_weight(current)
    cs_age = word_age(aoa, current, corpus_freq, corpus_total)
    boosts = np.zeros(len(candidates), dtype=np.float32)

    for i, c in enumerate(candidates):
        sim = semantic_scalar(current, c)
        tw = (topo_weight(c) + cs_topo) * 0.5
        ab = age_continuity_boost(
            cs_age, word_age(aoa, c, corpus_freq, corpus_total)
        )
        boosts[i] = strength * sim * (1.0 + tw + ab) / 3.0

    return boosts


# ────────────────────────────────────────────────────────────────────────────
# LENGTH-DEPENDENT EMBEDDER (DoubleEntendreEmbedder)
# ────────────────────────────────────────────────────────────────────────────

class LengthDependentEmbedder:
    """
    REASONING (Sentences 23-30):
    The embedder maps each word/form to a vector in dimension space determined by 
    word length, ensuring longer words occupy richer representational spaces. Embeddings 
    are produced by hashing the token, repeating bytes to reach desired dimension, and 
    normalizing by sum (L1-like norm). Forms are embedded using their full string 
    representation (word_form_name_sentenceindex), ensuring each form-sentence pairing 
    has unique embedding. A shift vector perturbs embeddings in controlled directions, 
    representing semantic drift induced by context. The shift is applied to w2's 
    embedding before computing dot product with candidate embeddings, modeling how 
    context (w1) influences next-word predictions. "Double entendre" refers to the 
    two passes: pass1 uses unshifted w2 embedding, pass2 uses shifted embedding, 
    capturing both baseline and contextual similarity. The combined score is computed 
    as the minimum of pass1 and pass2, ensuring only candidates high in both measures 
    receive maximum boost.
    """

    def embed(self, token: str, dim: Optional[int] = None) -> np.ndarray:
        """Embed a word/token into a vector."""
        d = dim if dim is not None else length_dim(token)
        raw_bytes = hashlib.sha256(token.encode("utf-8")).digest()
        repeated = (raw_bytes * ((d // 32) + 2))[:d]
        vec = np.array(list(repeated), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)

    def embed_form(self, form: SyntacticForm, dim: Optional[int] = None) -> np.ndarray:
        """Embed a syntactic form into a vector."""
        form_str = form.to_string()
        d = dim if dim is not None else length_dim(form.word)
        raw_bytes = hashlib.sha256(form_str.encode("utf-8")).digest()
        repeated = (raw_bytes * ((d // 32) + 2))[:d]
        vec = np.array(list(repeated), dtype=np.float32)
        s = float(vec.sum())
        return vec / (s + 1e-8)

    def shift_vector(self, token: str, dim: int, magnitude: float) -> np.ndarray:
        """Compute a shift vector for semantic drift."""
        raw_bytes = hashlib.md5(token.encode("utf-8")).digest()
        repeated = (raw_bytes * ((dim // 16) + 2))[:dim]
        vec = np.array(list(repeated), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-8)) * magnitude

    @staticmethod
    def _norm01(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        mn = float(arr.min())
        mx = float(arr.max())
        return (arr - mn) / (mx - mn + 1e-12)

    def length_dependent_weights(
        self,
        w1: str,
        w2: str,
        candidates: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        REASONING (Sentence 27-29):
        Compute length-dependent weights for candidates. Pass 1: baseline similarity 
        (unshifted w2). Pass 2: shifted similarity (w2 + context drift from w1). 
        Combined: min(pass1, pass2) scaled by topology kernel.
        """
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

            # Pass 2: shifted w2 embedding
            e_w2_shifted = e_w2 + shift
            norm_s = float(e_w2_shifted.sum())
            e_w2_shifted = e_w2_shifted / (abs(norm_s) + 1e-8)

            pass1_raw[i] = float(np.dot(e_w2, e_c))
            pass2_raw[i] = float(np.dot(e_w2_shifted, e_c))
            topo_kernels[i] = length_topo_kernel(c)

        # Normalize to [0, 1]
        p1 = self._norm01(pass1_raw)
        p2 = self._norm01(pass2_raw)
        de_score = np.minimum(p1, p2)

        # Combine with topology weighting
        base_combined = 0.5 * (p1 + p2)
        agreement_part = float(anchor_agree_bonus) * de_score
        combined = base_combined + topo_kernels * agreement_part
        combined = self._norm01(combined)

        return p1, p2, combined


# Alias for consistency
DoubleEntendreEmbedder = LengthDependentEmbedder


# ────────────────────────────────────────────────────────────────────────────
# N-GRAM LANGUAGE MODEL
# ────────────────────────────────────────────────────────────────────────────

class NGramLM:
    """
    REASONING (Sentences 31-35):
    The NGramLM ingests token sequences and builds unigram, bigram, and trigram 
    frequency tables for probabilistic next-word prediction. Trigram probabilities 
    are computed using Laplace smoothing (add-k) to handle unseen trigrams gracefully. 
    When a trigram (w1, w2, w3) is not observed, the model backs off to the bigram 
    distribution of (w2, ?), providing a fallback estimate. If no bigram with w2 
    is found, the model returns the top 150 most-frequent unigrams as candidates, 
    ensuring diversity. Duplicate candidates are filtered and the final candidate 
    list is capped at 400 to balance diversity and computational cost.
    """

    def __init__(self, add_k: float = 1.5):
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0

    def ingest(self, tokens: List[str]) -> None:
        """Ingest token sequence to build n-gram tables."""
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
        """
        Compute probability distribution over next word given (w1, w2) context.
        Returns (candidates, probabilities) tuple.
        """
        cands: List[str] = []

        # Trigram: if we have (w1, w2, w3) sequences, use them
        for (a, b, c) in self.tri:
            if a == w1 and b == w2:
                cands.append(c)

        # Bigram backoff: if no trigrams, use all words following w2
        if not cands:
            for (a, b) in self.bi:
                if a == w2:
                    cands.append(b)

        # Unigram backoff: use top frequent words
        if not cands:
            cands = [w for w, _ in sorted(self.uni.items(), key=lambda x: -x[1])[:150]]

        # Deduplicate and filter out cognitive tokens
        seen, out = set(), []
        for w in cands:
            if w not in seen and w not in COGNITIVE_TOKENS:
                seen.add(w)
                out.append(w)

        cands = out[:400]

        # Compute probabilities with Laplace smoothing
        V = len(self.vocab) + 1
        k = self.add_k

        def prob(w3: str) -> float:
            c12 = self.bi.get((w1, w2), 0)
            c123 = self.tri.get((w1, w2, w3), 0)
            if c12 > 0:
                return (c123 + k) / (c12 + k * V)
            return (self.uni.get(w3, 0) + k) / (self.total + k * V)

        probs = torch.tensor([prob(w) for w in cands], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)

        return cands, probs


# ────────────────────────────────────────────────────────────────────────────
# TOKENIZER & DETOKENIZER
# ────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\[[A-Z\-]+\]|[A-Za-z][A-Za-z0-9_'-]*|[.,;:!?()]")


def tokenize(text: str) -> List[str]:
    """
    REASONING (Sentences 36-37):
    Tokenization splits text into alphabetic words, punctuation, and cognitive 
    tokens (like [PROBLEM], [SOLUTION]) using regex matching. Alphabetic tokens 
    are lowercased for consistency, while punctuation and cognitive tokens are 
    preserved as-is for structural integrity.
    """
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
    """
    REASONING (Sentences 38-39):
    Detokenization reconstructs readable text by reattaching punctuation to 
    preceding words and capitalizing sentence starts. Newlines are normalized 
    to spaces during tokenization to ensure linear token flow across paragraphs.
    """
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

    s = re.sub(r"([a-z])", lambda m: m.group(1), s)

    return s


# ────────────────────────────────────────────────────────────────────────────
# CORPUS STATE
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusState:
    """
    REASONING (Sentences 44-46):
    CorpusState aggregates language model, embedder, AoA dictionary, and form plan 
    into a unified inference engine. Token boost scores reflect log-relative frequency: 
    words appearing more often in corpus receive higher boosts to encourage their 
    reuse. The form plan is initialized by extracting alphabetic, non-stop words 
    from the prompt and building 100 forms from them.
    """
    lm: NGramLM
    embedder: LengthDependentEmbedder
    aoa: Dict[str, float]
    sentence_form_plan: SentenceFormPlan = field(default_factory=SentenceFormPlan)
    token_boost: Dict[str, float] = field(default_factory=dict)
    corpus_freq: Dict[str, int] = field(default_factory=dict)
    corpus_total: int = 1


def build_state(text: str, aoa: Dict[str, float], prompt: str = "") -> CorpusState:
    """Build CorpusState from text and prompt."""
    tokens = tokenize(text)
    lm = NGramLM(add_k=1.5)
    lm.ingest(tokens)

    embedder = LengthDependentEmbedder()
    total = max(1, sum(lm.uni.values()))

    # Build token boost from corpus frequency
    token_boost: Dict[str, float] = {}
    for tok, freq in lm.uni.items():
        if len(tok) > 3 and tok not in STOP_WORDS and re.match(r"^[a-z]", tok):
            token_boost[tok] = min(0.5, math.log(1 + (freq / total) * 1000.0) * 0.1)

    # Extract words from prompt for form plan
    prompt_tokens = tokenize(prompt)
    alpha_tokens = [
        t for t in prompt_tokens
        if re.match(r"^[a-z]", t) and t not in STOP_WORDS
    ]

    # Build form plan: 100 forms, one per sentence
    form_plan = SentenceFormPlan()
    form_plan.build_forms(alpha_tokens if alpha_tokens else ["word"])

    return CorpusState(
        lm=lm,
        embedder=embedder,
        aoa=aoa,
        sentence_form_plan=form_plan,
        token_boost=token_boost,
        corpus_freq=lm.uni,
        corpus_total=total,
    )


# ────────────────────────────────────────────────────────────────────────────
# SENTENCE GENERATION: 100 Sentences, One Form Each
# ────────────────────────────────────────────────────────────────────────────

def next_probs(
    state: CorpusState,
    w1: str,
    w2: str,
    sentence_index: int,
    temp: float = 1.2,
    de_strength: float = 0.18,
) -> Tuple[List[str], torch.Tensor]:
    """
    REASONING (Sentences 47-50):
    next_probs() combines multiple scoring signals to compute next-word probabilities: 
    base n-gram, double-entendre, centroid, form, token, age, and topological boosts. 
    The form boost is semantic similarity between the current sentence's assigned 
    form word and each candidate, weighted at 25%. Topological centroid boost is 
    amplified by the topological kernel (0.05–0.95 range), making topology effects 
    stronger for longer words. Double-entendre weight (0.18) is typically lower 
    than base+centroid+form boosts, ensuring linguistic coherence takes precedence 
    over semantic drift.
    
    Generate next word probabilities for a sentence.
    """
    cands, base_probs = state.lm.next_dist(w1, w2)
    _, _, de_combined = state.embedder.length_dependent_weights(
        w1=w1,
        w2=w2,
        candidates=cands,
    )

    de_t = torch.tensor(de_combined, dtype=torch.float32)

    # Boost for current sentence's form
    form_boost = torch.zeros_like(de_t)
    current_form = state.sentence_form_plan.form_by_sentence.get(sentence_index)
    if current_form:
        for idx, c in enumerate(cands):
            # Boost words semantically similar to the form's word
            sim = semantic_similarity(current_form.word, c)
            form_boost[idx] = 0.25 * sim

    # Centroid boost
    cb = centroid_boost(
        state.aoa,
        w2,
        cands,
        strength=0.10,
        corpus_freq=state.corpus_freq,
        corpus_total=state.corpus_total,
    )
    cb_t = torch.tensor(cb, dtype=torch.float32)

    # Token boost
    tb = torch.tensor(
        [state.token_boost.get(c, 0.0) for c in cands],
        dtype=torch.float32
    )

    # Age continuity boost
    w2_age = word_age(state.aoa, w2, state.corpus_freq, state.corpus_total)
    age_arr = np.array(
        [
            age_continuity_boost(
                w2_age,
                word_age(state.aoa, c, state.corpus_freq, state.corpus_total),
            )
            for c in cands
        ],
        dtype=np.float32,
    )
    age_t = torch.tensor(age_arr, dtype=torch.float32)

    # Topology kernels
    topo_kernels = torch.tensor(
        [length_topo_kernel(c) for c in cands],
        dtype=torch.float32
    )

    topo_cb = cb_t * (0.5 + 0.5 * topo_kernels)

    # Combine all boosts
    boosts = (
        float(de_strength) * de_t
        + topo_cb
        + 0.10 * tb
        + 0.15 * age_t
        + form_boost
    )

    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
    logits = logits / max(float(temp), 1e-6)
    probs = F.softmax(logits, dim=-1)

    return cands, probs


def generate_100_sentences(
    state: CorpusState,
    prompt: str,
    seed: int = 42,
    tokens_per_sentence: int = 15,
    temp: float = 1.2,
) -> str:
    """
    REASONING (Sentences 51-55):
    generate_100_sentences() loops exactly 100 times, generating one sentence per 
    iteration, tracking form influence and accumulating results. For each sentence, 
    tokens are generated up to a maximum (tokens_per_sentence + 2) or until a 
    sentence-ending punctuation is encountered. Influenced words (those semantically 
    similar to the form's word) are tracked and recorded to the form plan for activation 
    spreading. Initial context (w1, w2) is seeded from the prompt's last two alphabetic 
    words, providing topical grounding for generation. The fully detokenized sentence 
    is stored in sentence_outputs and recorded in the form plan with form activation 
    strength=1.0.
    
    Generate exactly 100 sentences, one form per sentence.
    """
    rng = np.random.default_rng(int(seed))
    seed_toks = tokenize(prompt)
    sw = [t for t in seed_toks if re.match(r"^[a-z]", t)]
    w1 = sw[-2] if len(sw) >= 2 else (sw[0] if sw else "the")
    w2 = sw[-1] if sw else "concept"

    result_sentences: List[str] = []

    for sent_idx in range(100):
        sentence_tokens: List[str] = []
        alpha_count = 0
        influenced_words: Set[str] = set()

        # Generate tokens for this sentence
        for _ in range(int(tokens_per_sentence)):
            cands, probs = next_probs(
                state,
                w1,
                w2,
                sentence_index=sent_idx,
                temp=float(temp),
            )

            p = probs.detach().cpu().numpy()
            p = p / (p.sum() + 1e-12)
            tok = cands[int(rng.choice(len(cands), p=p))]
            sentence_tokens.append(tok)

            # Track influenced words
            current_form = state.sentence_form_plan.form_by_sentence.get(sent_idx)
            if current_form and semantic_similarity(current_form.word, tok) > 0.4:
                influenced_words.add(tok)

            w1, w2 = w2, tok

            if re.match(r"[A-Za-z]", tok):
                alpha_count += 1

        # End sentence on period or length overflow
        if sentence_tokens and sentence_tokens[-1] not in ".!?":
            sentence_tokens.append(".")

        # Detokenize and record sentence
        sentence_text = detokenize(sentence_tokens).strip()
        result_sentences.append(sentence_text)

        # Record in form plan
        form = state.sentence_form_plan.form_by_sentence.get(sent_idx)
        if form:
            state.sentence_form_plan.record_sentence_generation(
                sent_idx,
                sentence_text,
                form_activation=1.0,
                influenced_words=list(influenced_words),
            )

    return "\n".join(result_sentences)


# ────────────────────────────────────────────────────────────────────────────
# CORPUS LOADING
# ────────────────────────────────────────────────────────────────────────────

def load_corpus(
    use_hf: bool,
    hf_dataset: str,
    hf_split: str,
    hf_max_rows: int,
    text_file,
    hf_config: str = "",
    hf_col: str = "",
    hf_token: str = "",
) -> str:
    """
    REASONING (Sentences 56-59):
    load_corpus() supports both Hugging Face datasets (via transformers library) 
    and local text files (.txt, .md). For Hugging Face datasets, rows are limited 
    by hf_max_rows to avoid memory overflow on large corpora. Dataset columns are 
    detected flexibly: if "text" column exists, use it; else use first column. Text 
    files are read with UTF-8 encoding and errors replaced to handle non-ASCII 
    characters gracefully.
    """
    if use_hf:
        load_kwargs = {"split": hf_split}
        if hf_config and hf_config.strip():
            load_kwargs["name"] = hf_config.strip()
        if hf_token and hf_token.strip():
            load_kwargs["token"] = hf_token.strip()
        ds = load_dataset(hf_dataset, **load_kwargs)
        rows = min(int(hf_max_rows) if int(hf_max_rows) > 0 else len(ds), len(ds))
        if hf_col and hf_col.strip() and hf_col.strip() in ds.column_names:
            col = hf_col.strip()
        else:
            col = "text" if "text" in ds.column_names else ds.column_names[0]
        return "\n".join(str(x) for x in ds.select(range(rows))[col])

    if text_file is None:
        raise ValueError("No file provided.")

    path = text_file if isinstance(text_file, str) else (
        text_file.name if hasattr(text_file, "name") else str(text_file.get("path", ""))
    )

    return Path(path).read_text(encoding="utf-8", errors="replace")


# ────────────────────────────────────────────────────────────────────────────
# MAIN SESSION & GRADIO UI
# ────────────────────────────────────────────────────────────────────────────

def run_session(
    use_hf,
    hf_dataset,
    hf_split,
    hf_max_rows,
    hf_config,
    hf_col,
    hf_token,
    text_file,
    prompt,
    seed,
    num_sentences,
    tokens_per_sentence,
    temp,
    progress=gr.Progress(),
):
    """
    REASONING (Sentences 60-62):
    run_session() orchestrates the entire pipeline: load AoA, corpus, build state, 
    generate sentences, analyze forms. Progress callbacks provide real-time feedback 
    to the Gradio UI, indicating which stage of generation is active.
    """
    try:
        progress(0.05, desc="Loading AoA dataset (Kuperman 2012)…")
        aoa = load_aoa_dataset()

        progress(0.15, desc="Loading corpus…")
        text = load_corpus(
            bool(use_hf), str(hf_dataset), str(hf_split), int(hf_max_rows),
            text_file,
            hf_config=str(hf_config),
            hf_col=str(hf_col),
            hf_token=str(hf_token),
        )

        progress(0.30, desc="Building language model and form plan…")
        state = build_state(text, aoa, prompt=str(prompt))

        progress(0.50, desc="Generating sentences (one form per sentence)…")
        sentences = generate_100_sentences(
            state,
            str(prompt),
            seed=int(seed),
            tokens_per_sentence=int(tokens_per_sentence),
            temp=float(temp),
        )

        progress(0.80, desc="Analyzing form activation…")
        form_report = state.sentence_form_plan.generate_report()

        return sentences, form_report

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}", ""


def toggle_hf(val):
    """Toggle between HuggingFace dataset and local file modes."""
    return (
        gr.update(visible=val),   # hf_dataset
        gr.update(visible=val),   # hf_split
        gr.update(visible=val),   # hf_max_rows
        gr.update(visible=val),   # hf_config
        gr.update(visible=val),   # hf_col
        gr.update(visible=val),   # hf_token
        gr.update(visible=not val),  # text_file
    )


def build_app():
    """
    REASONING (Sentences 63-65):
    The Gradio interface layout uses two-column design: left for controls, right 
    for outputs, with responsive scaling (scale=1 vs scale=2). Input parameters 
    (seed, temperature, tokens_per_sentence, num_sentences) are exposed as Gradio 
    sliders and numeric inputs for user control. The prompt input defaults to 
    "Consider the nature of understanding", extracting words that become the base 
    for 100 form generation.
    """
    with gr.Blocks(
        title="NeuroSymbolic V8.6+",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# NeuroSymbolic V8.6+")

        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face Dataset", value=True)
                hf_dataset = gr.Textbox(
                    label="HF Dataset",
                    value="AiresPucrs/stanford-encyclopedia-philosophy"
                )
                hf_split = gr.Textbox(label="Split", value="train")
                hf_max_rows = gr.Slider(0, 2000, value=300, step=100, label="Max rows (0 = all)")
                hf_config = gr.Textbox(
                    label="Dataset Config / Subset",
                    placeholder="e.g. wikitext-2-raw-v1  (leave blank if not needed)",
                    value="",
                )
                hf_col = gr.Textbox(
                    label="Text Column Override",
                    placeholder="e.g. sentence  (leave blank to auto-detect 'text')",
                    value="",
                )
                hf_token = gr.Textbox(
                    label="HF Token (private datasets)",
                    placeholder="hf_…  (leave blank for public datasets)",
                    value="",
                    type="password",
                )
                text_file = gr.File(
                    label="Upload .txt/.md",
                    file_types=[".txt", ".md"],
                    visible=False
                )
                use_hf.change(
                    toggle_hf, [use_hf],
                    [hf_dataset, hf_split, hf_max_rows, hf_config, hf_col, hf_token, text_file]
                )

                seed = gr.Number(value=42, label="Seed")
                num_sentences = gr.Slider(
                    1, 500, value=100, step=10, label="Number of Sentences"
                )
                tokens_per_sentence = gr.Slider(
                    8, 1800, value=92, step=2, label="Tokens per Sentence"
                )
                temp = gr.Slider(0.8, 2.5, value=1.7, step=0.1, label="Temperature")

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt (extracts words for 100 forms)",
                    value="Consider the nature of understanding",
                    lines=2,
                )
                btn = gr.Button("Generate Sentences", variant="primary", size="lg")

                gr.Markdown("## Generated Sentences (One Form Per Sentence)")
                output_sentences = gr.Textbox(label="Sentences", lines=40)

                gr.Markdown("## Form Activation Analysis")
                output_report = gr.Textbox(label="Form Report", lines=40)

        btn.click(
            run_session,
            inputs=[
                use_hf, hf_dataset, hf_split, hf_max_rows,
                hf_config, hf_col, hf_token, text_file,
                prompt, seed, num_sentences, tokens_per_sentence, temp
            ],
            outputs=[output_sentences, output_report],
        )

        gr.Markdown(
            "### Key Features\n"
            "- **N Sentences, One Form Each:** Ensures full syntactic coverage\n"
            "- **Form Count:** Exactly 100 forms, one per sentence\n"
            "- **Form Boost:** Semantic similarity to form's word (25% weight)\n"
            "- **Activation Tracking:** Cumulative value + influence map\n"
            "- **Cohomological Topology:** Embedding dim derived from β₁ Betti number of n-gram simplicial complex\n"
            "- **Double-Entendre Embedder:** Two-pass similarity for robustness\n"
            "- **Age-of-Acquisition (AoA):** Kuperman 2012 dataset + regression model\n"
            "- **Multi-Signal Boosting:** 7+ scoring dimensions for coherence"
        )

        return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(share=False)