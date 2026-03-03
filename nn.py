#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSymbolic V9.0.1 — TetraGrid 2x2 Lattice Architecture
===============================================================================

ARCHITECTURE OVERVIEW:
We have encapsulated the disparate dot products, leak potentials, and anti-sparsification 
logic into a unified 2x2 Neural Network Grid: The TetraGrid Isomorphism.

For any Anchor Word (A) and Candidate Word (C), we project their Embeddings (E) 
and Leak Potentials (L) into a 2x2 matrix of interacting neurons:

    G(A, C) = [ N_00(A,C)  N_01(A,C) ]  =  [ E(A) • E(C)    E(A) • L(C) ]
              [ N_10(A,C)  N_11(A,C) ]     [ L(A) • E(C)    L(A) • L(C) ]

- N_00: Pure Instantaneous Semantic alignment
- N_01: Forward Leak Interference
- N_10: Backward Leak Interference
- N_11: Pure Topological Homology

All corrections (Diagonal shifts, Adversarial Normalization, and Bernoulli 
Inversification) operate explicitly via matrix math (Determinant, Trace, linear 
transformations) on this 2x2 grid layer.
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
# TARGET ISOMORPHISM 1: The TetraGrid 2x2 Lattice Layer
# ────────────────────────────────────────────────────────────────────────────
class TetraGridIsomorphism(torch.nn.Module):
    """
    Invents a 2x2 matrix network for interaction logic, uniting Membrane Spikes,
    Diagonal Shifts, and Bernoulli Anti-Sparsification.
    """
    def __init__(self, adv_strength: float = 0.5, densify_mag: float = 0.08, embed_dim: int = 256):
        super().__init__()
        self.adv_strength = adv_strength
        self.densify_mag = densify_mag
        self.embed_dim = embed_dim
        
        # Diagonal Shift cascade via learned 2x2 interaction
        # We initialize it slightly off-diagonal to mix pure matches with cross-interferences
        self.shift_matrix = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.shift_matrix.weight.copy_(torch.tensor([[0.85, 0.15], [0.15, 0.85]]))
            self.shift_matrix.bias.fill_(0.0)

    def _get_E(self, token: str) -> torch.Tensor:
        """E(X) -> Extracts Injection Current feature vector."""
        raw = hashlib.sha256(token.encode("utf-8")).digest()
        repeated = (raw * ((self.embed_dim // 32) + 2))[:self.embed_dim]
        vec = torch.tensor(list(repeated), dtype=torch.float32)
        return vec / (vec.sum() + 1e-8)

    def _get_L(self, token: str, magnitude: float) -> torch.Tensor:
        """L(X) -> Extracts base Leak Potential feature vector."""
        raw = hashlib.md5(token.encode("utf-8")).digest()
        repeated = (raw * ((self.embed_dim // 16) + 2))[:self.embed_dim]
        vec = torch.tensor(list(repeated), dtype=torch.float32)
        norm = torch.norm(vec)
        return (vec / (norm + 1e-8)) * magnitude

    def forward(self, anchor_word: str, anchor_leak_mag: float, candidates: List[str], p_fields: torch.Tensor) -> torch.Tensor:
        # Generate Tensors (1, D) and (N, D)
        E_A = self._get_E(anchor_word).unsqueeze(0)
        L_A = self._get_L(anchor_word, anchor_leak_mag).unsqueeze(0)
        
        if not candidates:
            return torch.zeros(0)

        E_C = torch.stack([self._get_E(c) for c in candidates])
        L_C = torch.stack([self._get_L(c, length_shift_mag(c)) for c in candidates])

        # Form the 2x2 interacting lattice for N candidates simultaneously
        # Matrix Math: G_00 = E_C @ E_A.T
        N_00 = torch.matmul(E_C, E_A.T).squeeze(-1)  # (N,)
        N_01 = torch.matmul(L_C, E_A.T).squeeze(-1)
        N_10 = torch.matmul(E_C, L_A.T).squeeze(-1)
        N_11 = torch.matmul(L_C, L_A.T).squeeze(-1)

        # G shape: (N, 2, 2) Grid
        G = torch.stack([
            torch.stack([N_00, N_01], dim=-1),
            torch.stack([N_10, N_11], dim=-1)
        ], dim=-2)

        # MATH 1: Adversarial Probability Filler (Element-wise inversion mapping)
        # G' = G - \alpha * G^2
        G = G - self.adv_strength * (G ** 2)

        # MATH 2: Independent Diagonal Shift Corrector
        # G'' = G' + tanh( W_shift * G' )
        G_shifted = self.shift_matrix(G)
        G = G + 0.15 * torch.tanh(G_shifted)

        # MATH 3: Bernoulli Anti-Sparsification
        # det(G) = (N_00 * N_11) - (N_01 * N_10)
        det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
        threshold = det.median() if det.numel() > 0 else 0.0
        sparse_mask = (det < threshold).float()

        bernoulli_mask = torch.bernoulli(p_fields)
        inversified = 1.0 - bernoulli_mask

        injection = sparse_mask * inversified * self.densify_mag
        
        # Inject mass into the Trace of the Grid in-place
        G[:, 0, 0] = G[:, 0, 0] + injection
        G[:, 1, 1] = G[:, 1, 1] + injection

        # READOUT: Trace + Cross-talk absolute magnitude
        readout = (G[:, 0, 0] + G[:, 1, 1]) + 0.5 * (torch.abs(G[:, 0, 1]) + torch.abs(G[:, 1, 0]))
        
        # Normalize strictly to [0, 1]
        mn, mx = readout.min(), readout.max()
        if mx > mn:
            readout = (readout - mn) / (mx - mn + 1e-12)
            
        return readout


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
# ARCHITECTURE GLUE & STATE
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class CorpusState:
    lm: HebbianReservoirLM
    tetra_grid: TetraGridIsomorphism
    sentence_form_plan: SentenceFormPlan = field(default_factory=SentenceFormPlan)
    token_boost: Dict[str, float] = field(default_factory=dict)
    corpus_freq: Dict[str, int] = field(default_factory=dict)
    corpus_total: int = 1
    time_step: int = 0

def tokenize(text: str) -> List[str]:
    out = []
    words = re.findall(r"\[[A-Z]+\]|\b[a-zA-Z]+\b", text)
    for w in words:
        if w in COGNITIVE_TOKENS:
            out.append(w)
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
            if t in STOP_WORDS_COG:
                res.append(t.strip("[]").lower())
            else:
                res.append(t)
        else:
            if not res:
                res.append(t.capitalize())
            else:
                res.append(t)
    return " ".join(res) + "."

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
        adv_strength=adv_strength, 
        densify_mag=densify_mag,
        embed_dim=256
    )

    corpus_freq = {}
    for t in tokens:
        corpus_freq[t] = corpus_freq.get(t, 0) + 1
    total = max(1, len(tokens))

    tb = {w: float(np.log1p(c)) for w, c in corpus_freq.items()}

    state = CorpusState(
        lm=lm,
        tetra_grid=tetra_grid,
        token_boost=tb,
        corpus_freq=corpus_freq,
        corpus_total=total,
        time_step=0
    )

    prompt_tokens = tokenize(prompt.upper())
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

        forms.append(SyntacticForm(word=w, syntactic_role=role, prefix_context=pref, suffix_context=suff))

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
    
    state.time_step += 1
    cands, base_probs = state.lm.next_dist(w1, w2)
    
    if len(cands) == 0:
        return cands, base_probs

    # Dynamically compute dataset Bernoulli fields
    freqs = torch.tensor([state.corpus_freq.get(c, 1) for c in cands], dtype=torch.float32)
    mean_freq = freqs.mean() + 1e-8
    p_fields = torch.clamp(mean_freq / (freqs + mean_freq), min=0.1, max=0.9)

    # Alternate Leak magnitude contextually
    adv_penalty = -1.0 if (state.time_step % 2 == 0) else 1.0
    anchor_leak_mag = length_shift_mag(w2) * (1.0 + (adv_penalty * state.tetra_grid.adv_strength))

    # Calculate probabilities explicitly through the TetraGrid 2x2 layer
    grid_output = state.tetra_grid(
        anchor_word=w2, 
        anchor_leak_mag=anchor_leak_mag, 
        candidates=cands, 
        p_fields=p_fields
    )

    # Agreement boosts & Semantic mapping
    anchor_agree_bonus = length_agreement_bonus(w2)
    topo_kernels = torch.tensor([length_topo_kernel(c) for c in cands], dtype=torch.float32)
    
    de_t = grid_output + topo_kernels * anchor_agree_bonus * grid_output

    form_boost = torch.zeros_like(de_t)
    current_form = state.sentence_form_plan.form_by_sentence.get(sentence_index)
    if current_form:
        for idx, c in enumerate(cands):
            sim = semantic_similarity(current_form.word, c)
            form_boost[idx] = 0.25 * sim

    tb = torch.tensor([state.token_boost.get(c, 0.0) for c in cands], dtype=torch.float32)

    boosts = (float(de_strength) * de_t + 0.10 * tb + form_boost)

    logits = torch.log(base_probs.clamp_min(1e-12)) + boosts
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
# GRADIO UI & DATASET LOADING
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
    densify_mag: float
) -> Tuple[str, str]:

    corpus_text = load_corpus(
        use_hf=use_hf, dataset_name=hf_dataset, config_name=hf_config,
        split=hf_split, column_name=hf_col, max_rows=int(hf_max_rows),
        hf_token=hf_token, text_file=text_file
    )

    if corpus_text.startswith("Error") or corpus_text.startswith("Hugging Face Load Error"):
        return corpus_text, "Check dataset configuration."

    state = build_state(
        text=corpus_text,
        prompt=prompt,
        num_sentences=int(num_sentences),
        adv_strength=float(adv_strength),
        densify_mag=float(densify_mag),
    )

    generate_100_sentences(
        state=state,
        seed=int(seed),
        num_sentences=int(num_sentences),
        tokens_per_sentence=int(tokens_per_sentence),
        temp=float(temp),
    )

    sent_lines = [f"[{i+1}] {s}\n" for i, s in state.sentence_form_plan.sentence_outputs.items()]
    
    report_lines = [
        "FORM ACTIVATION & NEURONAL REPORT",
        "===================================",
        f"Sentences generated: {len(state.sentence_form_plan.sentence_outputs)}\n"
    ]
    for sent_idx in range(min(30, len(state.sentence_form_plan.sentence_outputs))):
        f = state.sentence_form_plan.form_by_sentence.get(sent_idx)
        if f:
            output = state.sentence_form_plan.sentence_outputs.get(sent_idx, "")
            report_lines.append(f"Sentence {sent_idx:02d} | Form: {f.form_name}")
            report_lines.append(f"  Word: '{f.word}', Role: '{f.syntactic_role}'")
            report_lines.append(f"  Activation Value: {f.activation_value:.4f}")
            report_lines.append(f"  Output: {output[:60]}...\n")

    return "\n".join(sent_lines), "\n".join(report_lines)

def build_app():
    with gr.Blocks(title="NeuroSymbolic Form Generator V9.0.1") as demo:
        gr.Markdown(
            "# Neuronal Isomorphism Generator V9.0.1: TetraGrid Lattice\n"
            "**The TetraGrid Isomorphism:** An elegant 2x2 grid encapsulating dot products, "
            "leak potentials, diagonal shifts, and Bernoulli Anti-Sparsification directly via Matrix math.\n\n"
            "**Math Encapsulation:**\n"
            "> `G = [[ E(Anchor)•E(Cand), E(Anchor)•L(Cand) ], [ L(Anchor)•E(Cand), L(Anchor)•L(Cand) ]]`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                use_hf = gr.Checkbox(label="Use Hugging Face Dataset?", value=False)
                
                with gr.Group(visible=False) as hf_group:
                    hf_dataset = gr.Textbox(label="Dataset Path", value="AiresPucrs/stanford-encyclopedia-philosophy")
                    hf_config = gr.Textbox(label="Config (e.g. '')", value="")
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
                num_sentences = gr.Slider(1, 200, value=100, step=10, label="Sentences")
                tokens_per_sentence = gr.Slider(8, 200, value=92, step=2, label="Tokens")
                temp = gr.Slider(0.8, 2.5, value=1.7, step=0.1, label="Temperature")

                gr.Markdown("### TetraGrid Lattice Controls")
                adv_strength = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Grid Adversarial Penalty")
                densify_mag = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Bernoulli Trace Inversification")

            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt (extracts words for 100 forms)", value="Consider the nature of understanding", lines=2)
                btn = gr.Button("Engage TetraGrid Neural Layer", variant="primary", size="lg")

                gr.Markdown("## Output Stream")
                output_sentences = gr.Textbox(label="Sentences", lines=20)
                output_report = gr.Textbox(label="Grid Analysis Report", lines=20)

        btn.click(
            run_session, 
            inputs=[
                use_hf, hf_dataset, hf_split, hf_max_rows,
                hf_config, hf_col, hf_token, text_file,
                prompt, seed, num_sentences, tokens_per_sentence, 
                temp, adv_strength, densify_mag
            ], 
            outputs=[output_sentences, output_report]
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(share=False)
