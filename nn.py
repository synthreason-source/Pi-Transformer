#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V9.2.0 — Dual-Pass Mirror Inversion × TetraGrid Shift
===============================================================================

CHANGE vs V9.1.5:
- Pass 1: Generate normally (trigrams + TetraGrid forward).
- build_mirror_set(): collect ALL generated tokens across all sentences,
  invert their frequency distribution
  (high-freq generated → low mirror weight; rare → high mirror weight).
- TetraGridIsomorphism.mirror_forward(): replaces the random Bernoulli
  inversification with deterministic mirror weights, injecting them into
  the sparse TetraGrid lattice positions.  Mirror-weighted candidates
  (those under-explored in pass 1) receive boosted G[0,0] / G[1,1]
  injection, shifting the 2x2 readout toward novel vocabulary.
- next_probs_mirrored(): calls mirror_forward instead of forward, recomputes
  base trigram probs + mirror-shifted TetraGrid logits, renormalises.
- generate_second_pass(): runs a full generation loop using
  next_probs_mirrored, seeded from the same trigram heads.
- run_session() now executes both passes and returns four outputs.
===============================================================================
"""

from __future__ import annotations
import re
import math
import random
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd
import gradio as gr
import torch
import torch.nn.functional as F
from datasets import load_dataset

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

STOP_WORDS_COG = set(
    "a an and are as at be by for from has have he her him his i in is it its "
    "me my of on or our she so that the their them they this to was we were what "
    "when where which who will with you your if because while"
    .split()
)
COGNITIVE_TOKENS = {f"[{w.upper()}]" for w in STOP_WORDS_COG}
PUNCT_TOKENS = {",", ".", "!", "?", ";", ":"}

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
# TARGET ISOMORPHISM 1: The TetraGrid 2x2 Lattice Layer
# ────────────────────────────────────────────────────────────────────────────

class TetraGridIsomorphism(torch.nn.Module):
    def __init__(self, token_to_idx: Dict[str, int],
                 adv_strength: float = 0.5,
                 densify_mag: float = 0.08,
                 embed_dim: int = 256):
        super().__init__()
        self.adv_strength = adv_strength
        self.densify_mag  = densify_mag
        self.embed_dim    = embed_dim
        self.token_to_idx = token_to_idx
        vocab_size = len(token_to_idx) + 1
        self.unk_idx = vocab_size - 1
        self.E_embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.L_embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.shift_matrix = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.shift_matrix.weight.copy_(torch.tensor([[0.85, 0.15], [0.15, 0.85]]))
            self.shift_matrix.bias.fill_(0.0)

    def _get_E(self, token: str) -> torch.Tensor:
        idx = self.token_to_idx.get(token, self.unk_idx)
        vec = torch.abs(self.E_embed(torch.tensor(idx)))
        return vec / (vec.sum() + 1e-8)

    def _get_L(self, token: str, magnitude: float) -> torch.Tensor:
        idx = self.token_to_idx.get(token, self.unk_idx)
        vec = torch.abs(self.L_embed(torch.tensor(idx)))
        norm = torch.norm(vec)
        return (vec / (norm + 1e-8)) * magnitude

    def _build_G(self, anchor_word: str, candidates: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared lattice construction used by both forward paths."""
        anchor_leak_mag = 0.05 if anchor_word in PUNCT_TOKENS \
            else min(max(len(anchor_word) / 10.0, 0.1), 2.0)

        E_A = self._get_E(anchor_word).unsqueeze(0)
        L_A = self._get_L(anchor_word, anchor_leak_mag).unsqueeze(0)

        def c_mag(c):
            return 0.05 if c in PUNCT_TOKENS else min(max(len(c) / 10.0, 0.1), 2.0)

        E_C = torch.stack([self._get_E(c) for c in candidates])
        L_C = torch.stack([self._get_L(c, c_mag(c)) for c in candidates])

        N_00 = torch.matmul(E_C, E_A.T).squeeze(-1)
        N_01 = torch.matmul(L_C, E_A.T).squeeze(-1)
        N_10 = torch.matmul(E_C, L_A.T).squeeze(-1)
        N_11 = torch.matmul(L_C, L_A.T).squeeze(-1)

        G = torch.stack([
            torch.stack([N_00, N_01], dim=-1),
            torch.stack([N_10, N_11], dim=-1)
        ], dim=-2)

        G = G - self.adv_strength * (G ** 2)
        G_shifted = self.shift_matrix(G)
        G = G + 0.15 * torch.tanh(G_shifted)

        det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
        return G, det

    def _readout_and_normalise(self, G: torch.Tensor) -> torch.Tensor:
        readout = (G[:, 0, 0] + G[:, 1, 1]) + 0.5 * (
            torch.abs(G[:, 0, 1]) + torch.abs(G[:, 1, 0]))
        mn, mx = readout.min(), readout.max()
        if mx > mn:
            readout = (readout - mn) / (mx - mn + 1e-12)
        return readout

    # ── Pass-1 forward (original Bernoulli inversification) ──────────────
    def forward(self, anchor_word: str, candidates: List[str]) -> torch.Tensor:
        if not candidates:
            return torch.zeros(0)
        G, det = self._build_G(anchor_word, candidates)
        threshold  = det.median() if det.numel() > 0 else 0.0
        sparse_mask = (det < threshold).float()
        bernoulli_mask = torch.bernoulli(torch.full((len(candidates),), 0.5))
        inversified  = 1.0 - bernoulli_mask
        injection    = sparse_mask * inversified * self.densify_mag
        G[:, 0, 0] += injection
        G[:, 1, 1] += injection
        return self._readout_and_normalise(G)

    # ── Pass-2 mirror_forward (deterministic mirror-weight inversification) ─
    def mirror_forward(self,
                       anchor_word: str,
                       candidates: List[str],
                       mirror_weights: Dict[str, float],
                       mirror_strength: float = 0.35) -> torch.Tensor:
        """
        Replace the random Bernoulli mask with normalised mirror weights.

        Mirror weights encode INVERTED generation frequency from pass 1:
          - tokens rarely generated  → high mirror weight → maximum injection
          - tokens over-generated    → low  mirror weight → suppressed injection

        This shifts the 2×2 TetraGrid lattice toward novel vocabulary,
        biasing pass-2 probs away from pass-1's most-used words.
        """
        if not candidates:
            return torch.zeros(0)

        G, det = self._build_G(anchor_word, candidates)
        threshold   = det.median() if det.numel() > 0 else 0.0
        sparse_mask = (det < threshold).float()

        # Build per-candidate mirror weight tensor
        m_raw = torch.tensor(
            [mirror_weights.get(c, 0.0) for c in candidates],
            dtype=torch.float32
        )
        mn_m, mx_m = m_raw.min(), m_raw.max()
        if mx_m > mn_m:
            m_norm = (m_raw - mn_m) / (mx_m - mn_m + 1e-12)
        else:
            m_norm = torch.zeros_like(m_raw)

        # Mirror injection: sparse_mask × mirror_norm × boosted densify_mag
        injection = sparse_mask * m_norm * self.densify_mag * (1.0 + mirror_strength)
        G[:, 0, 0] += injection
        G[:, 1, 1] += injection

        return self._readout_and_normalise(G)


# ────────────────────────────────────────────────────────────────────────────
# TARGET ISOMORPHISM 2: Hebbian Synaptic Reservoir — TRIGRAMS ONLY
# ────────────────────────────────────────────────────────────────────────────

class HebbianReservoirLM:
    def __init__(self, basal_k: float = 1.5):
        self.basal_k = float(basal_k)
        self.spontaneous_trace: Dict[str, float] = {}
        self.tri_synapses: Dict[Tuple[str, str, str], float] = {}
        self.trigram_heads: Dict[Tuple[str, str], List[str]] = {}
        self._global_tri_continuations: List[str] = []
        self._global_tri_weights: List[float] = []
        self.vocab: List[str] = []
        self.token_to_idx: Dict[str, int] = {}
        self.total_spikes = 0

    def ingest(self, tokens: List[str]) -> None:
        for t in tokens:
            self.spontaneous_trace[t] = self.spontaneous_trace.get(t, 0) + 1.0
            self.total_spikes += 1
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            k = (w1, w2, w3)
            self.tri_synapses[k] = self.tri_synapses.get(k, 0) + 1.0
            head = (w1, w2)
            if head not in self.trigram_heads:
                self.trigram_heads[head] = []
            if w3 not in self.trigram_heads[head]:
                self.trigram_heads[head].append(w3)
        for token in self.spontaneous_trace:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.token_to_idx)
        cont_counts: Dict[str, float] = {}
        for (_, _, w3), cnt in self.tri_synapses.items():
            cont_counts[w3] = cont_counts.get(w3, 0) + cnt
        self._global_tri_continuations = list(cont_counts.keys())
        self._global_tri_weights = [cont_counts[w] for w in self._global_tri_continuations]
        raw_vocab = list(self.spontaneous_trace.keys())
        self.vocab = [v for v in raw_vocab if v not in PUNCT_TOKENS and v not in COGNITIVE_TOKENS]

    def next_dist(self, w1: str, w2: str) -> Tuple[List[str], torch.Tensor]:
        head = (w1, w2)
        if head in self.trigram_heads:
            cands_raw = self.trigram_heads[head]
            weights_raw = [self.tri_synapses.get((w1, w2, w3), 0) for w3 in cands_raw]
        else:
            cands_raw  = list(self._global_tri_continuations)
            weights_raw = list(self._global_tri_weights)
        seen: Dict[str, float] = {}
        for w, wt in zip(cands_raw, weights_raw):
            seen[w] = seen.get(w, 0) + wt
        cands   = list(seen.keys())[:400]
        weights = [seen[w] for w in cands]
        k = self.basal_k
        V_total = len(self.vocab) + 1
        if head in self.trigram_heads:
            c12_total = sum(weights)
            probs_raw = [(wt + k) / (c12_total + k * V_total) for wt in weights]
        else:
            total_w = sum(weights)
            probs_raw = [(wt + k) / (total_w + k * V_total) for wt in weights]
        probs = torch.tensor(probs_raw, dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cands, probs


# ────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE GLUE & STATE
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusState:
    lm: HebbianReservoirLM
    tetra_grid: TetraGridIsomorphism
    sentence_form_plan: SentenceFormPlan = field(default_factory=SentenceFormPlan)
    time_step: int = 0

def tokenize(text: str) -> List[str]:
    out = []
    words = re.findall(r"\[[A-Z]+\]|\b[a-zA-Z]+\b|[.,!?;:]", text)
    for w in words:
        if w in COGNITIVE_TOKENS or w in PUNCT_TOKENS:
            out.append(w)
        else:
            w_clean = "".join(
                c for c in unicodedata.normalize("NFD", w)
                if unicodedata.category(c) != "Mn"
            ).lower()
            if w_clean:
                if w_clean in STOP_WORDS_COG:
                    out.append(f"[{w_clean.upper()}]")
                else:
                    out.append(w_clean)
    return out

def detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    res = []
    for t in tokens:
        if t in PUNCT_TOKENS:
            if res and len(res[-1]) > 0 and res[-1][-1] not in PUNCT_TOKENS:
                res[-1] += t
            continue
        if t in COGNITIVE_TOKENS:
            raw = t.strip("[]").lower()
            if raw in STOP_WORDS_COG:
                word = raw.capitalize() if not res or res[-1].endswith(('.', '!', '?')) else raw
                res.append(word)
            else:
                res.append(t)
        else:
            word = t.capitalize() if not res or res[-1].endswith(('.', '!', '?')) else t
            res.append(word)
    out = " ".join(res).strip()
    if out and out[-1] not in PUNCT_TOKENS:
        out += "."
    return out

def build_state(
    text: str,
    prompt: str = "Consider the nature of understanding",
    num_sentences: int = 100,
    adv_strength: float = 0.5,
    densify_mag: float = 0.08
) -> CorpusState:
    tokens = tokenize(text)
    lm = HebbianReservoirLM()
    lm.ingest(tokens)
    tetra_grid = TetraGridIsomorphism(
        token_to_idx=lm.token_to_idx,
        adv_strength=adv_strength,
        densify_mag=densify_mag,
        embed_dim=256
    )
    state = CorpusState(lm=lm, tetra_grid=tetra_grid, time_step=0)
    prompt_tokens = tokenize(prompt.upper())
    base_words = [w for w in prompt_tokens
                  if w not in COGNITIVE_TOKENS and w not in PUNCT_TOKENS and re.match(r"^[a-z]+$", w)]
    if not base_words:
        base_words = ["default", "word"]
    syntactic_roles = ["noun", "verb", "adj", "adv"]
    prefixes = ["pre", "post", "anti", "hyper", "meta", "sub", "un", "re"]
    suffixes = ["ism", "ity", "ness", "tion", "ology", "ment", "ive", "ly"]
    forms = []
    for i in range(100):
        w    = base_words[i % len(base_words)]
        role = syntactic_roles[i % len(syntactic_roles)]
        pref = prefixes[(i // len(syntactic_roles)) % len(prefixes)]
        suff = suffixes[(i // (len(syntactic_roles) * len(prefixes))) % len(suffixes)]
        forms.append(SyntacticForm(word=w, syntactic_role=role, prefix_context=pref, suffix_context=suff))
    state.sentence_form_plan.plan_forms(forms, num_sentences=num_sentences)
    return state


# ────────────────────────────────────────────────────────────────────────────
# PROBABILITY FUNCTIONS  (pass-1 and pass-2)
# ────────────────────────────────────────────────────────────────────────────

def next_probs(
    state: CorpusState, w1: str, w2: str, sentence_index: int,
    temp: float = 1.2, de_strength: float = 0.18,
) -> Tuple[List[str], torch.Tensor]:
    state.time_step += 1
    cands, base_probs = state.lm.next_dist(w1, w2)
    if len(cands) == 0:
        return cands, base_probs
    grid_output = state.tetra_grid(anchor_word=w2, candidates=cands)
    punct_bias    = torch.zeros_like(grid_output)
    punct_penalty = torch.zeros_like(grid_output)
    for idx, c in enumerate(cands):
        if c in PUNCT_TOKENS:
            punct_bias[idx] = -3.5
            if w2 in PUNCT_TOKENS:
                punct_penalty[idx] = -10000.0
    logits = (torch.log(base_probs.clamp_min(1e-12))
              + float(de_strength) * grid_output
              + punct_bias + punct_penalty)
    logits = logits / max(float(temp), 1e-6)
    return cands, F.softmax(logits, dim=-1)


def next_probs_mirrored(
    state: CorpusState, w1: str, w2: str, sentence_index: int,
    mirror_weights: Dict[str, float],
    mirror_strength: float = 0.35,
    temp: float = 1.2,
    de_strength: float = 0.18,
) -> Tuple[List[str], torch.Tensor]:
    """
    Pass-2 probability function.
    Calls mirror_forward() instead of forward() so the TetraGrid injection
    is driven by inverted generation frequencies rather than Bernoulli noise.
    """
    state.time_step += 1
    cands, base_probs = state.lm.next_dist(w1, w2)
    if len(cands) == 0:
        return cands, base_probs
    grid_output = state.tetra_grid.mirror_forward(
        anchor_word=w2, candidates=cands,
        mirror_weights=mirror_weights, mirror_strength=mirror_strength
    )
    punct_bias    = torch.zeros_like(grid_output)
    punct_penalty = torch.zeros_like(grid_output)
    for idx, c in enumerate(cands):
        if c in PUNCT_TOKENS:
            punct_bias[idx] = -3.5
            if w2 in PUNCT_TOKENS:
                punct_penalty[idx] = -10000.0
    logits = (torch.log(base_probs.clamp_min(1e-12))
              + float(de_strength) * grid_output
              + punct_bias + punct_penalty)
    logits = logits / max(float(temp), 1e-6)
    return cands, F.softmax(logits, dim=-1)


# ────────────────────────────────────────────────────────────────────────────
# MIRROR SET CONSTRUCTION  (bridge between pass-1 and pass-2)
# ────────────────────────────────────────────────────────────────────────────

def build_mirror_set(all_generated_token_lists: List[List[str]]) -> Dict[str, float]:
    """
    1. Count every token produced across all pass-1 sentences.
    2. Invert frequencies:  mirror_weight = (max_freq - freq + 1)
       → over-generated tokens approach zero weight
       → under-explored tokens approach max weight
    3. Normalise to sum = 1.0

    This inverted distribution is used by mirror_forward() to steer
    TetraGrid injection in pass-2 toward vocabulary not yet exploited.
    """
    freq: Dict[str, float] = {}
    for token_list in all_generated_token_lists:
        for t in token_list:
            if t not in PUNCT_TOKENS and t not in COGNITIVE_TOKENS:
                freq[t] = freq.get(t, 0) + 1.0
    if not freq:
        return {}
    max_f  = max(freq.values())
    mirror = {w: (max_f - f + 1.0) for w, f in freq.items()}
    total  = sum(mirror.values())
    return {w: v / total for w, v in mirror.items()} if total > 0 else mirror


# ────────────────────────────────────────────────────────────────────────────
# GENERATION — pass-1 (tracks raw token lists) and pass-2 (mirror-shifted)
# ────────────────────────────────────────────────────────────────────────────

def _generation_loop(
    state: CorpusState,
    rng_seed: int,
    num_sentences: int,
    tokens_per_sentence: int,
    temp: float,
    prob_fn,          # callable(state, w1, w2, sent_idx) → (cands, probs)
) -> Tuple[List[str], List[List[str]]]:
    """
    Shared generation loop used by both passes.
    Returns (sentence_strings, raw_token_lists).
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)

    trigram_head_list = list(state.lm.trigram_heads.keys())
    random.shuffle(trigram_head_list)
    if not trigram_head_list:
        return ["Not enough trigram data."], []

    MIN_TOKENS_BEFORE_PUNCT = max(3, int(tokens_per_sentence * 0.15))
    MIN_TOKENS_BEFORE_END   = max(4, int(tokens_per_sentence * 0.85))
    WORDS_SINCE_PUNCT_MIN   = 3
    END_PUNCT = {".", "?", "!"}

    def best_non_punct(cands_list, p_tensor):
        best_i, best_p = None, -1.0
        for i, (c, p) in enumerate(zip(cands_list, p_tensor.tolist())):
            if c not in PUNCT_TOKENS and p > best_p:
                best_i, best_p = i, p
        return cands_list[best_i] if best_i is not None else "the"

    out_sentences  = []
    all_token_lists = []

    for sent_idx in range(num_sentences):
        sent_tokens   = []
        words_since_punct = 999
        w1, w2 = random.choice(trigram_head_list)

        for _ in range(tokens_per_sentence):
            cands, probs = prob_fn(state, w1, w2, sent_idx)
            if len(cands) == 0:
                break
            idx = torch.multinomial(probs, 1).item()
            nxt = cands[idx]

            if nxt in PUNCT_TOKENS:
                too_early = (len(sent_tokens) < MIN_TOKENS_BEFORE_PUNCT) or \
                            (words_since_punct < WORDS_SINCE_PUNCT_MIN)
                too_early_end = (nxt in END_PUNCT) and (len(sent_tokens) < MIN_TOKENS_BEFORE_END)
                if too_early or too_early_end:
                    nxt = best_non_punct(cands, probs)
                else:
                    words_since_punct = 0
            else:
                words_since_punct += 1

            sent_tokens.append(nxt)
            w1, w2 = w2, nxt

            if nxt in END_PUNCT and len(sent_tokens) >= MIN_TOKENS_BEFORE_END:
                break

        text = detokenize(sent_tokens)
        out_sentences.append(text)
        all_token_lists.append(sent_tokens)
        state.sentence_form_plan.sentence_outputs[sent_idx] = text
        form = state.sentence_form_plan.form_by_sentence.get(sent_idx)
        if form and sent_tokens:
            form.activation_value += 1.0

    return out_sentences, all_token_lists


def generate_100_sentences(
    state: CorpusState,
    seed: int = 42,
    num_sentences: int = 100,
    tokens_per_sentence: int = 92,
    temp: float = 1.7,
) -> Tuple[List[str], List[List[str]]]:
    """Pass-1 generation. Returns (sentences, token_lists)."""
    def prob_fn(s, w1, w2, si):
        return next_probs(s, w1, w2, si, temp=temp)
    return _generation_loop(state, seed, num_sentences, tokens_per_sentence, temp, prob_fn)


def generate_second_pass(
    state: CorpusState,
    mirror_weights: Dict[str, float],
    seed: int = 43,
    num_sentences: int = 100,
    tokens_per_sentence: int = 92,
    temp: float = 1.7,
    mirror_strength: float = 0.35,
) -> Tuple[List[str], List[List[str]]]:
    """
    Pass-2: re-run generation with mirror-shifted TetraGrid probabilities.

    The mirror_weights (inverted pass-1 frequencies) replace Bernoulli
    noise in the sparse lattice injection, steering the 2×2 TetraGrid
    readout toward vocabulary under-explored in pass-1.
    """
    # Reset form plan outputs so pass-2 fills them fresh
    state.sentence_form_plan.sentence_outputs.clear()
    for f in state.sentence_form_plan.form_by_sentence.values():
        f.activation_value = 0.0

    def prob_fn(s, w1, w2, si):
        return next_probs_mirrored(
            s, w1, w2, si,
            mirror_weights=mirror_weights,
            mirror_strength=mirror_strength,
            temp=temp,
        )
    return _generation_loop(state, seed, num_sentences, tokens_per_sentence, temp, prob_fn)


# ────────────────────────────────────────────────────────────────────────────
# CORPUS LOADING
# ────────────────────────────────────────────────────────────────────────────

def load_corpus(
    use_hf=False, dataset_name="", config_name="", split="train",
    column_name="text", max_rows=100, hf_token="", text_file=None
):
    if use_hf and dataset_name:
        try:
            ds = load_dataset(
                dataset_name, name=config_name if config_name else None,
                split=split, token=hf_token if hf_token else None, trust_remote_code=True,
            )
            df = ds.select(range(min(len(ds), max_rows))).to_pandas()
            if column_name in df.columns:
                return " ".join(df[column_name].astype(str).tolist())
            return f"Error: Column '{column_name}' not found."
        except Exception as e:
            return f"Hugging Face Load Error: {str(e)}"
    if text_file is not None:
        try:
            file_path = text_file.name if hasattr(text_file, "name") else str(text_file)
            return Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"
    return (
        "In algebraic topology, homology and cohomology provide a profound "
        "understanding of the shape of data. A persistent filtration creates a "
        "barcode of topological features. Betti numbers summarize cycles, voids, "
        "and connectivity. We consider the nature of understanding spaces "
        "through simplicial complexes and morse theory."
    )


# ────────────────────────────────────────────────────────────────────────────
# SESSION RUNNER  (both passes)
# ────────────────────────────────────────────────────────────────────────────

def _format_sentences(sentence_outputs: Dict[int, str]) -> str:
    return "\n".join(f"[{i+1}] {s}" for i, s in sentence_outputs.items())

def _format_report(state: CorpusState, pass_label: str,
                   mirror_weights: Optional[Dict[str, float]] = None) -> str:
    lines = [
        f"FORM ACTIVATION & GRID REPORT  —  {pass_label}",
        "=" * 50,
        f"Sentences generated: {len(state.sentence_form_plan.sentence_outputs)}",
    ]
    if mirror_weights is not None:
        top_mirror = sorted(mirror_weights.items(), key=lambda x: -x[1])[:10]
        lines.append("\nTop-10 Mirror-Boosted Tokens (under-explored in pass-1):")
        for w, wt in top_mirror:
            lines.append(f"  {w:<20s}  mirror_weight={wt:.5f}")
        lines.append("")
    for sent_idx in range(min(30, len(state.sentence_form_plan.sentence_outputs))):
        f = state.sentence_form_plan.form_by_sentence.get(sent_idx)
        if f:
            output = state.sentence_form_plan.sentence_outputs.get(sent_idx, "")
            lines += [
                f"Sentence {sent_idx:02d} | Form: {f.form_name}",
                f"  Word: '{f.word}', Role: '{f.syntactic_role}'",
                f"  Activation Value: {f.activation_value:.4f}",
                f"  Output: {output[:70]}...",
                "",
            ]
    return "\n".join(lines)

def run_session(
    use_hf, hf_dataset, hf_split, hf_max_rows, hf_config, hf_col, hf_token,
    text_file, prompt, seed, num_sentences, tokens_per_sentence,
    temp, adv_strength, densify_mag, mirror_strength,
):
    corpus_text = load_corpus(
        use_hf=use_hf, dataset_name=hf_dataset, config_name=hf_config, split=hf_split,
        column_name=hf_col, max_rows=int(hf_max_rows), hf_token=hf_token, text_file=text_file,
    )
    if corpus_text.startswith("Error") or corpus_text.startswith("Hugging Face"):
        return corpus_text, "", "", "Check dataset configuration."

    state = build_state(
        text=corpus_text, prompt=prompt,
        num_sentences=int(num_sentences),
        adv_strength=float(adv_strength),
        densify_mag=float(densify_mag),
    )

    # ── Pass 1 ──────────────────────────────────────────────────────────────
    p1_sentences, p1_token_lists = generate_100_sentences(
        state=state, seed=int(seed),
        num_sentences=int(num_sentences),
        tokens_per_sentence=int(tokens_per_sentence),
        temp=float(temp),
    )
    p1_text   = _format_sentences(state.sentence_form_plan.sentence_outputs)
    p1_report = _format_report(state, "PASS 1 — TetraGrid Bernoulli")

    # ── Mirror set construction ──────────────────────────────────────────────
    mirror_weights = build_mirror_set(p1_token_lists)

    # ── Pass 2 ──────────────────────────────────────────────────────────────
    p2_sentences, _ = generate_second_pass(
        state=state,
        mirror_weights=mirror_weights,
        seed=int(seed) + 1,
        num_sentences=int(num_sentences),
        tokens_per_sentence=int(tokens_per_sentence),
        temp=float(temp),
        mirror_strength=float(mirror_strength),
    )
    p2_text   = _format_sentences(state.sentence_form_plan.sentence_outputs)
    p2_report = _format_report(state, "PASS 2 — Mirror-Shifted TetraGrid", mirror_weights)

    return p1_text, p1_report, p2_text, p2_report


# ────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ────────────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="NeuroSymbolic V9.2.0 — Dual-Pass Mirror TetraGrid") as demo:
        gr.Markdown(
            "# NeuroSymbolic V9.2.0: Dual-Pass Mirror Inversion × TetraGrid Shift\n"
        )

        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face Dataset?", value=False)
                hf_dataset  = gr.Textbox(label="Dataset Path",  value="AiresPucrs/stanford-encyclopedia-philosophy", visible=False)
                hf_config   = gr.Textbox(label="Config",        value="",      visible=False)
                hf_split    = gr.Textbox(label="Split",         value="train", visible=False)
                hf_col      = gr.Textbox(label="Text Column",   value="text",  visible=False)
                hf_max_rows = gr.Number( label="Max Rows",      value=100,     visible=False)
                hf_token    = gr.Textbox(label="HF Token",      type="password", visible=False)
                text_file   = gr.File(label="Upload Local Text (.txt / .md)",
                                      file_types=[".txt", ".md"], visible=True)

                def _toggle(use_hf_val):
                    hv = gr.update(visible=use_hf_val)
                    fv = gr.update(visible=not use_hf_val)
                    return hv, hv, hv, hv, hv, hv, fv
                use_hf.change(fn=_toggle, inputs=use_hf,
                              outputs=[hf_dataset, hf_config, hf_split, hf_col,
                                       hf_max_rows, hf_token, text_file])

                gr.Markdown("### Core Hyperparameters")
                seed                = gr.Number(value=42,  label="Seed")
                num_sentences       = gr.Slider(1,   200, value=100, step=10,  label="Sentences")
                tokens_per_sentence = gr.Slider(8,   200, value=92,  step=2,   label="Tokens / Sentence")
                temp                = gr.Slider(0.8, 2.5, value=1.7, step=0.1, label="Temperature")

                gr.Markdown("### TetraGrid Lattice Controls")
                adv_strength  = gr.Slider(0.0, 1.0, value=0.5,  step=0.05, label="Grid Adversarial Penalty")
                densify_mag   = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Bernoulli Trace Inversification (pass 1)")
                mirror_strength = gr.Slider(0.0, 2.0, value=0.35, step=0.05,
                                            label="Mirror Shift Strength (pass 2)")

            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt (extracts words for 100 SyntacticForms)",
                    value="Consider the nature of understanding", lines=2
                )
                btn = gr.Button("Run Dual-Pass TetraGrid ×  Mirror Inversion",
                                variant="primary", size="lg")

                with gr.Tab("Output"):
                    p2_sentences_out = gr.Textbox(label="Output", lines=20)

        btn.click(
            run_session,
            inputs=[
                use_hf, hf_dataset, hf_split, hf_max_rows, hf_config, hf_col, hf_token,
                text_file, prompt, seed, num_sentences, tokens_per_sentence,
                temp, adv_strength, densify_mag, mirror_strength,
            ],
            outputs=[p2_sentences_out],
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(share=False)
