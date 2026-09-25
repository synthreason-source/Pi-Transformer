"""
HECM TOY LANGUAGE MODEL: Mechanisms & Mathematics (Optimized Edition)
===================================================================
A generalized context-state Markov chain with sparse diffusion and modulation.

Efficiency Updates:
  - Uses scipy.sparse.csr_matrix for O(E) memory scaling instead of dense O(N^2).
  - Uses scipy.sparse.linalg.expm_multiply for fast diffusion scaling.
  - Pre-allocates and vectorizes scoring loops for minimal CPU overhead.
"""

from __future__ import annotations
import math
import re
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


# ----------------------------------------------------------------------
# 1. DATASET LOADER
# ----------------------------------------------------------------------
class DatasetLoader:
    BOS = "<BOS>"
    UNK = "<UNK>"

    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.tokens: list[str] = self._tokenize(raw_text)
        self.vocab: list[str] = [self.UNK, self.BOS] + sorted(set(self.tokens))
        self.word_to_idx: dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+|[.,!?;]", text.lower())

    def idx(self, word: str) -> int:
        return self.word_to_idx.get(word, self.word_to_idx[self.UNK])


# ----------------------------------------------------------------------
# 2. N-GRAM CONTEXT-STATE CONSTRUCTION (SPARSE)
# ----------------------------------------------------------------------
class ContextStateBuilder:
    """
    Builds context-states using sparse coordinate lists to bypass 
    dense memory allocation limits.
    """

    def __init__(self, tokens: list[str], order: int = 3):
        self.order = order
        self.context_length = order - 1
        self.tokens = tokens

        windows = [
            tuple(tokens[i:i + self.context_length])
            for i in range(len(tokens) - self.context_length + 1)
        ]
        self.contexts: list[tuple] = sorted(set(windows))
        self.context_to_idx: dict[tuple, int] = {c: i for i, c in enumerate(self.contexts)}
        n = len(self.contexts)

        # Build sparse edge transitions efficiently
        edges = {}
        for i in range(len(windows) - 1):
            a = self.context_to_idx[windows[i]]
            b = self.context_to_idx[windows[i + 1]]
            edges[(a, b)] = edges.get((a, b), 0.0) + 1.0

        rows = [k[0] for k in edges.keys()]
        cols = [k[1] for k in edges.keys()]
        data = list(edges.values())

        self.M = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        self.counts = np.asarray(self.M.sum(axis=1)).ravel()

    def log_sort_contexts(self):
        order = np.argsort(-self.counts)
        log_weights = -np.log(self.counts[order] + 1.0)
        return order, log_weights

    @staticmethod
    def permute_matrix(M: sp.csr_matrix, order: np.ndarray) -> sp.csr_matrix:
        """Permutes rows and columns of a sparse CSR matrix efficiently."""
        return M[order, :][:, order].tocsr()


# ----------------------------------------------------------------------
# 3. SPARSE NORMALIZATION
# ----------------------------------------------------------------------
def row_normalize_sparse(M: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalizes a sparse CSR transition matrix safely."""
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv_sums = sp.diags(1.0 / row_sums)
    P = inv_sums @ M
    return P.tocsr()


# ----------------------------------------------------------------------
# 4. SPARSE MATRIX-EXPONENTIAL DIFFUSION WEIGHTING
# ----------------------------------------------------------------------
def sparse_diffusion_weighting(P_raw: sp.csr_matrix, beta: float = 0.15,
                               remove_self_loops: bool = True) -> sp.csr_matrix:
    """
    Computes diffusion using expm_multiply to avoid O(N^3) dense calculations.
    """
    n = P_raw.shape[0]
    I = sp.eye(n, format='csr')
    L = P_raw - I

    # Compute matrix exponential action directly via Krylov subspace approximation
    D = expm_multiply(beta * L, I)
    if not sp.issparse(D):
        D = sp.csr_matrix(D)

    if remove_self_loops:
        D.setdiag(0.0)
        D.eliminate_zeros()

    return row_normalize_sparse(D)


# ----------------------------------------------------------------------
# 5. SINUSOIDAL TEMPERATURE MODULATION
# ----------------------------------------------------------------------
class TemperatureSchedule:
    def __init__(self, base_temp=0.9, amplitude=0.5, omega=0.3, phase=0.0, floor=0.15):
        self.base_temp = base_temp
        self.amplitude = amplitude
        self.omega = omega
        self.phase = phase
        self.floor = floor

    def __call__(self, step: int) -> float:
        t = self.base_temp + self.amplitude * math.sin(self.omega * step + self.phase)
        return max(t, self.floor)


# ----------------------------------------------------------------------
# 7. PAYLOAD CUSTOMIZER
# ----------------------------------------------------------------------
class PayloadCustomizer:
    def __init__(self, spec: str | None, word_to_idx: dict[str, int]):
        self.weights: dict[int, float] = {}
        if not spec:
            return
        spec = spec.strip()
        if spec.startswith("--boost"):
            spec = spec[len("--boost"):].strip()
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            word, w = pair.rsplit(":", 1)
            word = word.strip().lower()
            if word in word_to_idx:
                self.weights[word_to_idx[word]] = float(w)


# ----------------------------------------------------------------------
# 6. GENERATION & SAMPLING
# ----------------------------------------------------------------------
def choose_from_logits(logits: np.ndarray, top_k: int, rng: np.random.Generator) -> int:
    top_k = min(top_k, len(logits))
    top_idx = np.argpartition(-logits, top_k - 1)[:top_k]
    top_logits = logits[top_idx]
    top_logits -= top_logits.max()
    probs = np.exp(top_logits)
    probs /= probs.sum()
    return int(rng.choice(top_idx, p=probs))


class HECMToyLM:
    def __init__(self, raw_text: str, order: int = 3, beta: float = 0.15, alpha: float = 0.5,
                 temp_schedule: TemperatureSchedule | None = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.loader = DatasetLoader(raw_text)
        self.builder = ContextStateBuilder(self.loader.tokens, order=order)
        
        order_perm, log_weights = self.builder.log_sort_contexts()
        self.sorted_order = order_perm
        self.log_weights = log_weights
        
        M_sorted = self.builder.permute_matrix(self.builder.M, order_perm)
        self.pos_in_sorted = np.empty_like(order_perm)
        self.pos_in_sorted[order_perm] = np.arange(len(order_perm))

        self.P_raw_sorted = row_normalize_sparse(M_sorted)
        D_sorted = sparse_diffusion_weighting(self.P_raw_sorted, beta=beta, remove_self_loops=True)
        
        # Sparse blend
        self.P_final_sorted = alpha * self.P_raw_sorted + (1 - alpha) * D_sorted
        self.P_final_sorted = row_normalize_sparse(self.P_final_sorted)

        self.temp_schedule = temp_schedule or TemperatureSchedule()
        self.order = order
        self.context_length = order - 1

    def _context_row(self, context: tuple) -> np.ndarray:
        idx = self.builder.context_to_idx.get(context)
        n = len(self.builder.contexts)
        if idx is None:
            return np.full(n, 1.0 / n)
        row_csr = self.P_final_sorted[self.pos_in_sorted[idx]]
        return row_csr.toarray().ravel()

    def generate(self, prompt: str, steps: int = 30, top_k: int = 8,
                 payload_spec: str | None = None, bias_strength: float = 1.5) -> str:
        payload = PayloadCustomizer(payload_spec, self.loader.word_to_idx)
        prompt_tokens = DatasetLoader._tokenize(prompt)
        pad = [self.loader.BOS] * max(0, self.context_length - len(prompt_tokens))
        context = tuple((pad + prompt_tokens)[-self.context_length:])

        output = list(prompt_tokens)
        contexts_by_idx = self.builder.contexts

        for step in range(steps):
            probs = self._context_row(context)
            T = self.temp_schedule(step)
            logits = np.log(probs + 1e-12) / T

            if payload.weights:
                boosted = logits.copy()
                for j, cand_context in enumerate(contexts_by_idx):
                    last_word_idx = self.loader.idx(cand_context[-1])
                    if last_word_idx in payload.weights:
                        boosted[j] += bias_strength * payload.weights[last_word_idx]
                logits = boosted

            next_ctx_local_idx = choose_from_logits(logits, top_k, self.rng)
            next_context = contexts_by_idx[next_ctx_local_idx]
            next_word = next_context[-1]

            if next_word == self.loader.BOS:
                break
            output.append(next_word)
            context = next_context

        text = " ".join(output)
        text = re.sub(r"\s+([.,!?;])", r"\1", text)
        return text


# ----------------------------------------------------------------------
# DEMO RUN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    with open(input("Filename: "), "r", encoding="utf-8") as file:
        corpus = file.read()

    temp_sched = TemperatureSchedule(base_temp=0.9, amplitude=0.6, omega=0.35, phase=0.0)
    model = HECMToyLM(corpus, order=5, beta=0.15, alpha=0.5, temp_schedule=temp_sched, seed=7)

    print(f"Vocabulary size: {len(model.loader.vocab)}")
    print(f"Context-states (order={model.order}): {len(model.builder.contexts)}")
    print("\n=== Generation WITH sparse optimization & payload customizer ===")
    while True:
        print(model.generate(input("USER: "), steps=200, top_k=5, payload_spec="--boost forest:2.0,dog:1.0", bias_strength=2.0))
