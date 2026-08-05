import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import difflib
import tempfile
import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ==========================================================
# 1. Dataset Loading & Context Analysis
# ==========================================================

def load_and_analyze_dataset(filename, context_window=5):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    tokens = raw_text.lower().split()
    if len(tokens) < 3:
        raise ValueError("Dataset must contain at least 3 tokens.")

    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    unique_trigrams = sorted(set(trigrams))
    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    if unk_trigram not in unique_trigrams:
        unique_trigrams.append(unk_trigram)

    vocab_size = len(unique_trigrams)
    word_to_idx = {t: i for i, t in enumerate(unique_trigrams)}
    idx_to_word = {i: t for i, t in enumerate(unique_trigrams)}

    context_to_row = {}
    row_count = 0
    trigram_counts = defaultdict(Counter)

    for i, t in enumerate(trigrams):
        context = tuple(tokens[max(0, i - context_window):i])
        if len(context) == 0:
            continue
        if context not in context_to_row:
            context_to_row[context] = row_count
            row_count += 1
        trigram_counts[context][word_to_idx.get(t, word_to_idx[unk_trigram])] += 1

    # Build a SPARSE frequency matrix instead of a dense one. trigram_counts
    # is already sparse (most context/target pairs never co-occur), so a
    # dense [row_count, vocab_size] float32 tensor wastes enormous amounts of
    # memory on zeros -- e.g. 5,000 contexts x 20,000 vocab trigrams is
    # 100M floats (~400MB) for a matrix that might only have a few thousand
    # nonzero entries. Only the row/col indices + nonzero values are stored.
    rows, cols, vals = [], [], []
    for context, targets in trigram_counts.items():
        r_idx = context_to_row[context]
        for target_idx, count in targets.items():
            rows.append(r_idx)
            cols.append(target_idx)
            vals.append(float(count))

    if vals:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        with torch.sparse.check_sparse_tensor_invariants(False):
            freq_matrix = torch.sparse_coo_tensor(
                indices, values, size=(max(row_count, 1), vocab_size)
            ).coalesce()
        col_sums = torch.sparse.sum(freq_matrix, dim=0).to_dense()
    else:
        col_sums = torch.zeros(vocab_size, dtype=torch.float32)

    max_freq = col_sums.max().clamp_min(1e-5)
    normalized_inv_freq = 1.0 - (col_sums / max_freq)
    exp_boost = torch.exp(normalized_inv_freq * 2.0)

    top_indices = torch.argsort(col_sums, descending=True)
    frequent_trigrams = [idx_to_word[i.item()] for i in top_indices if idx_to_word[i.item()] != unk_trigram]
    frequent_words = list(dict.fromkeys([w for tg in frequent_trigrams for w in tg]))

    return (
        word_to_idx,
        idx_to_word,
        vocab_size,
        exp_boost,
        frequent_trigrams,
        unique_trigrams,
        frequent_words,
        context_to_row,
    )


# ==========================================================
# 2. Logit Transform Zoo: Triangular Transpose + 3D CP-Violation Transpose
# ==========================================================

def apply_triangular_logit_transpose(logits, mode="standard"):
    """
    Applies a geometric triangular transpose on predicted logits.
    Logits are reshaped into local triangular triples (v1, v2, v3) across channels/vocab,
    and transposed.

    Modes:
      - 'standard': Reverses vertex orientation (v1 <-> v3 transpose).
      - 'anti': Flips along secondary axis (v1 <-> v2 anti-transpose).
      - 'full_transpose': Performs 2D matrix transpose on packed triangle logit grid.
    """
    if mode == "none" or logits is None:
        return logits

    B, V = logits.shape

    # Pad vocabulary dimension to nearest multiple of 3 to form 3-vertex triangles
    pad_len = (3 - (V % 3)) % 3
    if pad_len > 0:
        padded_logits = F.pad(logits, (0, pad_len), value=-1e9)
    else:
        padded_logits = logits

    # Reshape into [B, N, 3] triangular vertex triples
    triangles = padded_logits.view(B, -1, 3)

    # Full 2D triangular matrix transposition
    triangles_transposed = triangles.transpose(1, 2).contiguous()
    triangles_transposed = triangles_transposed.view(B, -1, 3)

    # Flatten back to original logit vocabulary shape
    transposed_logits = triangles_transposed.view(B, -1)[:, :V]
    return transposed_logits


def apply_cp_violation_transpose_3d(
    logits,
    mode="standard",
    violation_strength=0.15,
    compute_gradient=False,
):
    """
    A CP-violation-style transform on flat TRIANGLES (3-vertex groups),
    the same packing the plain triangular transpose uses -- not 3x3x3 cubes.
    An earlier version of this function packed logits into cubes, which is
    the wrong shape for this operator: cube packing requires the padded
    vocab length to be an exact multiple of a *cube* (e.g. 3^3=27), and any
    code that tries to size that cube dynamically from the vocab (rather
    than using the fixed 3) can easily end up requesting a `.view()` whose
    dimensions don't multiply out to the actual padded length -- exactly
    the "shape '[1,1,13,13,13]' is invalid for input of size 51" kind of
    RuntimeError. Triangles avoid this entirely: padding to a multiple of 3
    is always exact and never depends on a derived side length.

    Per triangle (v1, v2, v3), three physics-flavored operators are applied:
      - P (parity):   reverses vertex ordering (v1,v2,v3) -> (v3,v2,v1)
                       (`torch.flip`), analogous to a spatial mirror.
      - C (charge):   negates the resulting values, analogous to swapping
                       particle <-> antiparticle amplitude sign.
      - CP-transformed triangle: C applied on top of P, i.e. -(reversed triangle).

    If C and P were perfect symmetries, blending the CP-transformed triangle
    back in would do nothing (transformed == original). `violation_strength`
    is what breaks that symmetry: the final logits are an asymmetric mix of
    the original triangle and its CP-transformed counterpart, so
    `violation_strength=0` reduces to the identity and `violation_strength=1`
    reduces to a pure CP transform.

    Args:
        logits: [B, V] logits tensor.
        mode: 'none' returns logits unchanged. Any other string enables the
              transform (kept for interface parity with the plain triangular version).
        violation_strength: float in [0, 1], how strongly the CP-transformed
              triangle is blended back into the original ("degree of violation").
        compute_gradient: if True, also returns d(transformed)/d(logits)
              (summed grad_outputs of ones), useful for inspecting how
              sensitive the transform is to the raw logits. Requires
              `logits.requires_grad_(True)` to be set by the caller.

    Returns:
        transformed_logits [B, V], or (transformed_logits, gradient) if
        compute_gradient=True. `gradient` is None if logits didn't require grad.
    """
    if mode == "none" or logits is None:
        return (logits, None) if compute_gradient else logits

    B, V = logits.shape
    triangle_size = 3

    # NOTE: padding must be neutral (0.0), not a large negative sentinel like
    # some other transforms use. This transform includes a charge
    # conjugation (negation) step, so a -1e9 pad would flip to +1e9 and,
    # once shuffled by the flip, could land on a *real* vocab slot in the
    # last (partially-padded) triangle -- silently creating one
    # astronomically large logit that dominates every sampling step. 0.0 is
    # safe under negation and dilutes harmlessly when blended back in.
    pad_len = (triangle_size - (V % triangle_size)) % triangle_size
    padded = F.pad(logits, (0, pad_len), value=0.0) if pad_len > 0 else logits

    triangles = padded.view(B, -1, triangle_size)

    # --- P: parity inversion, reverse vertex order within each triangle ---
    parity_triangles = torch.flip(triangles, dims=[2])

    # --- C: charge conjugation, negate the amplitude ---
    cp_transposed = -parity_triangles

    # --- CP violation: asymmetric blend between original and CP-transformed triangle ---
    v = max(0.0, min(1.0, violation_strength))
    violated_triangles = (1.0 - v) * triangles + v * cp_transposed

    transformed_logits = violated_triangles.reshape(B, -1)[:, :V]

    if not compute_gradient:
        return transformed_logits

    gradient = None
    if logits.requires_grad:
        grad_outputs = torch.ones_like(transformed_logits)
        gradient = torch.autograd.grad(
            outputs=transformed_logits,
            inputs=logits,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]

    return transformed_logits, gradient


def apply_cp_violation_transpose_square(logits, mode="standard", violation_strength=0.15):
    """
    A CP-violation transform packed into 2x2 SQUARES (4 values each) instead
    of flat triangles (3) or the old, buggy cube packing (27). Squares are
    the only one of these shapes where "transpose" is literally the
    linear-algebra operation -- swapping rows and columns of a square
    matrix -- rather than a stand-in (an axis permutation for cubes, or a
    meaningless op on a 1D triangle).

    Per 2x2 square [[a, b], [c, d]]:
      - P (parity):  flip row order        -> [[c, d], [a, b]]
      - C (charge):  negate                -> -parity
      - transpose:   swap rows and columns of the charge-conjugated square
        (a genuine matrix transpose, `.transpose(-1, -2)`)
      - violated = (1-v)*square + v*transposed

    Deliberately GRADIENT-FREE: unlike `apply_cp_violation_transpose_3d`,
    this function has no `compute_gradient` option and never touches
    autograd. It's meant as a structural alternative to gradient descent --
    see `MarkovSeedingLayer.inspect_cp_violation_square_shift`, which uses
    this transform's own algebraic before/after difference as a sensitivity
    signal instead of backpropagating one.
    """
    if mode == "none" or logits is None:
        return logits

    B, V = logits.shape
    square_size = 4  # 2 x 2

    pad_len = (square_size - (V % square_size)) % square_size
    padded = F.pad(logits, (0, pad_len), value=0.0) if pad_len > 0 else logits

    squares = padded.view(B, -1, 2, 2)

    parity_squares = torch.flip(squares, dims=[2])       # P: flip row order
    charge_conjugate_squares = -parity_squares            # C: negate
    cp_transposed = charge_conjugate_squares.transpose(-1, -2)  # genuine matrix transpose

    v = max(0.0, min(1.0, violation_strength))
    violated_squares = (1.0 - v) * squares + v * cp_transposed

    return violated_squares.reshape(B, -1)[:, :V]


def apply_logit_transform(logits, mode="none", violation_strength=0.15, compute_gradient=False):
    """
    Dispatch helper: routes to the flat triangular transpose, the triangular
    CP-violation transpose, or the square CP-violation transpose depending
    on `mode`.

    Modes:
      - 'none': identity
      - 'standard' / 'anti' / 'full_transpose': flat triangular transpose (2D)
      - 'cp_violation_3d': triangular CP-violation transpose (+ optional gradient)
      - 'cp_violation_square': square CP-violation transpose (gradient-free by design)
    """
    if mode == "cp_violation_3d":
        return apply_cp_violation_transpose_3d(
            logits, mode=mode, violation_strength=violation_strength, compute_gradient=compute_gradient
        )
    if mode == "cp_violation_square":
        result = apply_cp_violation_transpose_square(logits, mode=mode, violation_strength=violation_strength)
        return (result, None) if compute_gradient else result
    result = apply_triangular_logit_transpose(logits, mode=mode)
    return (result, None) if compute_gradient else result


# ==========================================================
# 2c. Classical Math Kernel -- old-fashioned matrices, no tensor ops
# ==========================================================
# Hand-computed, nested-loop matrix math (no torch/numpy), used to
# independently cross-check the tensor kernel above. Two differently-written
# implementations of the same math agreeing is stronger evidence than one.

def classical_matmul(A, B):
    """Nested-loop matrix multiplication."""
    ra, ca, rb, cb = len(A), len(A[0]), len(B), len(B[0])
    if ca != rb:
        raise ValueError(f"Cannot multiply {ra}x{ca} by {rb}x{cb}")
    return [[sum(A[i][k] * B[k][j] for k in range(ca)) for j in range(cb)] for i in range(ra)]


def classical_transpose(A):
    """Nested-loop 2D transpose: A[i][j] <-> A[j][i]."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def classical_identity(n):
    """The n x n identity matrix, built by hand."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def classical_cp_violation_transpose_triangle(triangle, violation_strength=0.15):
    """
    Same op as `apply_cp_violation_transpose_3d`, for one flat triangle
    (3-element list) of plain floats, via explicit index arithmetic instead
    of torch.flip:
      P (parity):  reverse the triangle    -> triangle[2-i]
      C (charge):  negate                  -> -parity
      violated = (1-v)*triangle + v*(C applied to P)
    """
    v = max(0.0, min(1.0, violation_strength))
    return [
        (1.0 - v) * triangle[i] + v * (-triangle[2 - i])
        for i in range(3)
    ]


def verify_cp_violation_kernel_agreement(violation_strength=0.15, atol=1e-5):
    """Cross-checks the tensor kernel against the classical kernel on a
    random flat triangle. Returns (agrees: bool, max_abs_diff: float)."""
    triangle = [random.uniform(-3.0, 3.0) for _ in range(3)]
    logits = torch.tensor([triangle], dtype=torch.float32)
    tensor_flat = apply_cp_violation_transpose_3d(logits, violation_strength=violation_strength)[0].tolist()

    classical_flat = classical_cp_violation_transpose_triangle(triangle, violation_strength=violation_strength)

    max_diff = max(abs(a - b) for a, b in zip(tensor_flat, classical_flat))
    return max_diff < atol, max_diff


def classical_cp_violation_transpose_square(square, violation_strength=0.15):
    """
    Same op as `apply_cp_violation_transpose_square`, for one 2x2 square
    (list[list[float]]) of plain floats, via explicit index arithmetic
    instead of torch.flip/.transpose:
      P (parity):  flip row order        -> [square[1], square[0]]
      C (charge):  negate                -> -parity
      transpose:   swap rows and columns -> charge[j][i]
      violated = (1-v)*square + v*transposed
    """
    v = max(0.0, min(1.0, violation_strength))
    parity = [square[1], square[0]]
    charge = [[-parity[i][j] for j in range(2)] for i in range(2)]
    transposed = [[charge[j][i] for j in range(2)] for i in range(2)]
    return [[(1.0 - v) * square[i][j] + v * transposed[i][j] for j in range(2)] for i in range(2)]


def verify_cp_violation_square_kernel_agreement(violation_strength=0.15, atol=1e-5):
    """Cross-checks the tensor square kernel against the classical square
    kernel on a random 2x2 square. Returns (agrees: bool, max_abs_diff: float)."""
    flat = [random.uniform(-3.0, 3.0) for _ in range(4)]
    logits = torch.tensor([flat], dtype=torch.float32)
    tensor_flat = apply_cp_violation_transpose_square(logits, violation_strength=violation_strength)[0].tolist()

    square = [[flat[0], flat[1]], [flat[2], flat[3]]]
    classical_square = classical_cp_violation_transpose_square(square, violation_strength=violation_strength)
    classical_flat = [classical_square[0][0], classical_square[0][1], classical_square[1][0], classical_square[1][1]]

    max_diff = max(abs(a - b) for a, b in zip(tensor_flat, classical_flat))
    return max_diff < atol, max_diff


# ==========================================================
# 2d. Standard-Library Function Ontology (double purport)
# ==========================================================
# Each function's literal documented purpose ("standard_purpose") next to
# the second, local role it actually plays in this module -- the same call
# site is honest under both readings at once, not a hidden meaning.

STANDARD_LIBRARY_FUNCTION_ONTOLOGY = {
    "math.log": ("natural logarithm", "turns a probability into additive lookahead evidence, and the value curvature is computed on"),
    "torch.exp": ("e^x, elementwise over a tensor", "turns inverse trigram frequency into a sampling boost (exp_boost)"),
    "math.isclose": ("approximate float equality within a tolerance", "flags near-zero curvature steps where the confidence trajectory is locally straight"),
    "difflib.get_close_matches": ("fuzzy string matching", "snaps typed words onto the nearest known vocab word"),
    "collections.Counter": ("multiset / frequency dict", "the sparse basis freq_matrix and exp_boost are built from"),
    "torch.topk": ("k largest values + indices", "both filters sampling AND supplies a step's reported 'premise'"),
    "torch.argmax": ("index of the max value", "the deduction operator -- also what makes it seed-independent"),
    "torch.multinomial": ("categorical sampling", "the sampling operator; source of the arbitrary chain's divergence"),
    "torch.flip": ("reverse a tensor axis", "implements the 'parity' (P) operator in the CP-violation transpose"),
    "torch.Tensor.transpose": ("swap two tensor axes", "the genuine matrix transpose in the square kernel, and the axis swap in the triangular one"),
}


def print_function_ontology():
    """Prints the double-purport table: standard meaning vs. local role."""
    for name, (standard, local) in STANDARD_LIBRARY_FUNCTION_ONTOLOGY.items():
        print(f"{name}\n  standard:  {standard}\n  this file: {local}")


class OntologyUsageTracker:
    """
    Context manager that monkey-patches every function referenced by
    STANDARD_LIBRARY_FUNCTION_ONTOLOGY -- not just math.log/math.isclose --
    counts real invocations during the `with` block, and restores the
    originals on exit. This is how every entry in the table gets proven to
    describe code that actually ran, the same way the curvature functions
    were verified by spying on them earlier.
    """
    TRACKED_TORCH_FUNCS = ("topk", "argmax", "multinomial", "flip", "exp")

    def __init__(self):
        self.counts = {name: 0 for name in STANDARD_LIBRARY_FUNCTION_ONTOLOGY}
        self._originals = {}

    def _spy(self, key, fn):
        def wrapped(*args, **kwargs):
            self.counts[key] += 1
            return fn(*args, **kwargs)
        return wrapped

    def __enter__(self):
        g = globals()  # this module's global namespace (where Counter is bound)

        self._originals["math.log"] = math.log
        math.log = self._spy("math.log", math.log)

        self._originals["math.isclose"] = math.isclose
        math.isclose = self._spy("math.isclose", math.isclose)

        for name in self.TRACKED_TORCH_FUNCS:
            key = f"torch.{name}"
            orig = getattr(torch, name)
            self._originals[key] = orig
            setattr(torch, name, self._spy(key, orig))

        self._originals["torch.Tensor.transpose"] = torch.Tensor.transpose
        orig_transpose = torch.Tensor.transpose
        def spy_transpose(tensor_self, *a, **kw):
            self.counts["torch.Tensor.transpose"] += 1
            return orig_transpose(tensor_self, *a, **kw)
        torch.Tensor.transpose = spy_transpose

        self._originals["difflib.get_close_matches"] = difflib.get_close_matches
        difflib.get_close_matches = self._spy("difflib.get_close_matches", difflib.get_close_matches)

        # Counter was imported with `from collections import Counter`, so it's
        # bound directly into THIS module's globals -- patch that name, not
        # the collections module (which nothing here looks up dynamically).
        self._originals["collections.Counter"] = g["Counter"]
        orig_counter = g["Counter"]
        def spy_counter(*a, **kw):
            self.counts["collections.Counter"] += 1
            return orig_counter(*a, **kw)
        g["Counter"] = spy_counter

        return self

    def __exit__(self, *exc_info):
        g = globals()
        math.log = self._originals["math.log"]
        math.isclose = self._originals["math.isclose"]
        for name in self.TRACKED_TORCH_FUNCS:
            setattr(torch, name, self._originals[f"torch.{name}"])
        torch.Tensor.transpose = self._originals["torch.Tensor.transpose"]
        difflib.get_close_matches = self._originals["difflib.get_close_matches"]
        g["Counter"] = self._originals["collections.Counter"]
        return False


def run_full_ontology_audit(sample_text=None):
    """
    Exercises every function in STANDARD_LIBRARY_FUNCTION_ONTOLOGY in one
    self-contained run: dataset loading (collections.Counter), a forward
    pass + population sampling under the square kernel (torch.topk,
    torch.multinomial, torch.flip, torch.Tensor.transpose, torch.exp via
    exp_boost), a deductive+lookahead chain (torch.argmax), curvature
    (math.log, math.isclose), and a spell-correction call
    (difflib.get_close_matches) -- while an OntologyUsageTracker counts
    real invocations. Returns the call-count dict.
    """
    if sample_text is None:
        sample_text = "the quick brown fox jumps over the lazy dog " * 20

    # Use the OS's real temp directory (tempfile.gettempdir()) rather than a
    # hardcoded Unix path like "/tmp" -- "/tmp" doesn't exist on Windows,
    # which is exactly what caused FileNotFoundError here. mkstemp also
    # avoids collisions if this ever runs concurrently.
    fd, tmp_path = tempfile.mkstemp(prefix="_ontology_audit_corpus_", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(sample_text)

        with OntologyUsageTracker() as tracker:
            (word_to_idx, idx_to_word, vocab_size, exp_boost, frequent_trigrams,
             corpus_trigrams, frequent_words, context_to_row) = load_and_analyze_dataset(tmp_path, context_window=5)

            unk_id = word_to_idx[("<UNK>", "<UNK>", "<UNK>")]
            layer = MarkovSeedingLayer(vocab_size=vocab_size, embed_dim=8, context_size=5)
            seed = torch.tensor([[unk_id] * 5], dtype=torch.long)

            layer.generate_population_seeds(
                seed, sequence_length=3, pop_size=1, amp_boost=exp_boost,
                idx_to_word=idx_to_word, unk_id=unk_id,
                transpose_mode="cp_violation_square", top_k=5,
            )
            chain = layer.generate_deductive_reasoning_chain(
                seed, num_reasoning_steps=5, amp_boost=exp_boost,
                idx_to_word=idx_to_word, unk_id=unk_id,
                transpose_mode="cp_violation_3d", top_k=5,
                lookahead_width=2, lookahead_depth=1, verbose=False,
            )
            attach_curvature_to_chain(chain)

            known_words = sorted({w for tg in word_to_idx.keys() for w in tg})
            difflib.get_close_matches("teh", known_words, n=1, cutoff=0.0)

        return tracker.counts
    finally:
        # Clean up the temp file regardless of success or failure above.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def audit_ontology_function_usage(call_counts):
    """
    Prints each ontology entry's role next to how many times it was
    actually invoked this run, flagging anything the table claims but the
    run never reached. Returns True iff every entry was called at least once.
    """
    print("=== ONTOLOGY USAGE AUDIT ===")
    all_used = True
    for name, (standard, local) in STANDARD_LIBRARY_FUNCTION_ONTOLOGY.items():
        count = call_counts.get(name, 0)
        if count == 0:
            all_used = False
            flag = "  [WARNING: never called this run]"
        else:
            flag = ""
        print(f"  [{name}] called {count}x | {local}{flag}")
    return all_used


# ==========================================================
# 2b. Chain-of-Reasoning Record
# ==========================================================

@dataclass
class ReasoningStep:
    """
    One link in the generation's chain of reasoning: everything that was
    actually computed to decide a single generated token. Nothing here is
    invented after the fact -- each field is read directly off the tensors
    used during that step's forward pass and sampling, so the chain is a
    faithful, inspectable trace of the model's decision process rather than
    a post-hoc explanation.
    """
    step_index: int
    context_tokens: List[Tuple[str, str, str]]
    transform_mode: str
    top_candidates: List[Tuple[Tuple[str, str, str], float]]
    chosen_word: Tuple[str, str, str]
    chosen_prob: float
    selection_method: str = "sampled"  # "sampled" (stochastic) or "deductive" (argmax)
    runner_up_gap: Optional[float] = None  # margin between top-1 and top-2 confidence, deductive mode only
    lookahead_depth: int = 0  # how many steps ahead were simulated before committing to this choice
    lookahead_scores: Optional[List[Tuple[Tuple[str, str, str], float]]] = None  # (candidate, cumulative log-prob incl. rollout)
    overrode_greedy: bool = False  # True if lookahead picked something other than the naive immediate-best candidate
    curvature: Optional[float] = None  # second finite difference of log(chosen_prob); None for edge steps (no two neighbors)
    curvature_is_flat: Optional[bool] = None  # True if curvature ~ 0.0, per math.isclose
    curvature_ratio: Optional[float] = None  # exp(curvature): the same bend, viewed multiplicatively
    close_to_previous: Optional[bool] = None  # per difflib.get_close_matches against the prior step's phrase

    def as_text(self) -> str:
        context_str = ' '.join(w for tg in self.context_tokens for w in tg)
        candidates_str = ", ".join(
            f"{'/'.join(word)}={prob:.3f}" for word, prob in self.top_candidates
        )

        if self.selection_method == "deductive":
            gap_str = f", margin over runner-up: {self.runner_up_gap:.3f}" if self.runner_up_gap is not None else ""

            if self.lookahead_depth > 0 and self.lookahead_scores:
                lookahead_str = ", ".join(
                    f"{'/'.join(word)}(cum_logp={score:.3f})" for word, score in self.lookahead_scores
                )
                override_str = (
                    " [overrides naive immediate-best choice -- a shallower lookahead "
                    "would have picked differently]" if self.overrode_greedy else ""
                )
                return (
                    f"[Step {self.step_index + 1}] "
                    f"PREMISE: given context \"{context_str}\" under transform {self.transform_mode}, "
                    f"immediate candidates ranked {candidates_str}. "
                    f"LOOKING AHEAD {self.lookahead_depth} step(s), simulated continuations score: {lookahead_str}. "
                    f"THEREFORE: the candidate with the best cumulative outcome follows -> "
                    f"\"{'/'.join(self.chosen_word)}\" (p={self.chosen_prob:.3f}{gap_str}){override_str}{self._curvature_suffix()}"
                )

            gap_str = f", margin over runner-up: {self.runner_up_gap:.3f}" if self.runner_up_gap is not None else ""
            return (
                f"[Step {self.step_index + 1}] "
                f"PREMISE: given context \"{context_str}\" under transform {self.transform_mode}, "
                f"candidates ranked {candidates_str}. "
                f"THEREFORE: the highest-confidence candidate follows deductively -> "
                f"\"{'/'.join(self.chosen_word)}\" (p={self.chosen_prob:.3f}{gap_str}){self._curvature_suffix()}"
            )

        return (
            f"[Step {self.step_index + 1}] "
            f"context: \"{context_str}\" "
            f"| transform: {self.transform_mode} "
            f"| considered: {candidates_str} "
            f"| chose: \"{'/'.join(self.chosen_word)}\" (p={self.chosen_prob:.3f}){self._curvature_suffix()}"
        )

    def _curvature_suffix(self) -> str:
        if self.curvature is None:
            return ""
        flat_str = " [flat]" if self.curvature_is_flat else ""
        close_str = " [echoes previous]" if self.close_to_previous else ""
        ratio_str = f", ratio={self.curvature_ratio:.3f}" if self.curvature_ratio is not None else ""
        return f" | curvature={self.curvature:+.4f}{ratio_str}{flat_str}{close_str}"


# ==========================================================
# 2e. Reasoning Curvature -- built on ALL TEN entries of
#     STANDARD_LIBRARY_FUNCTION_ONTOLOGY, not just documented by two of them
# ==========================================================
#
# "Curvature" of a reasoning chain: how sharply the confidence trajectory
# bends step to step (second finite difference of log(chosen_prob)), plus a
# chain-wide analysis of that bend. Every one of the ten ontology functions
# is genuinely called somewhere in this pipeline -- the table below is a
# map from function to the exact role it plays here, not aspirational
# documentation:
#
#   math.log               per-step log-probability the curvature is built from
#   math.isclose            flags near-zero (flat) curvature
#   torch.exp                curvature_ratio: the same bend viewed multiplicatively
#   difflib.get_close_matches  flags steps whose phrase echoes the previous one
#   collections.Counter      tallies how many steps bent up / down / flat
#   torch.topk                the k sharpest-bending steps in the chain
#   torch.argmax               the single sharpest-bending step
#   torch.multinomial          a magnitude-weighted random "representative" bend
#   torch.flip                  reversal-symmetry self-check on the curvature formula
#   torch.Tensor.transpose    packs (step_index, curvature) into a real table

CURVATURE_ONTOLOGY_FUNCTIONS = tuple(STANDARD_LIBRARY_FUNCTION_ONTOLOGY.keys())


def compute_curvature(chosen_probs, flat_tolerance=1e-3):
    """
    Given a sequence of per-step probabilities, returns a list of
    (curvature, is_flat, ratio) for each interior step (index 0 of the
    output corresponds to original index 1 -- edge steps have no two
    neighbors and are excluded here; callers map back onto the full chain).

    curvature[i] = log(p[i+1]) - 2*log(p[i]) + log(p[i-1])   -- via math.log
    is_flat[i]   = math.isclose(curvature[i], 0.0, abs_tol=flat_tolerance)
    ratio[i]     = exp(curvature[i])                          -- via torch.exp
    """
    log_probs = [math.log(max(p, 1e-12)) for p in chosen_probs]  # ontology: math.log
    results = []
    for i in range(1, len(log_probs) - 1):
        c = log_probs[i + 1] - 2.0 * log_probs[i] + log_probs[i - 1]
        is_flat = math.isclose(c, 0.0, abs_tol=flat_tolerance)  # ontology: math.isclose
        ratio = torch.exp(torch.tensor(c)).item()  # ontology: torch.exp
        results.append((c, is_flat, ratio))
    return results


def attach_curvature_to_chain(chain, flat_tolerance=1e-3):
    """
    Computes curvature over `chain`'s chosen_prob sequence via
    `compute_curvature` and writes the results back onto each interior
    ReasoningStep's curvature / curvature_is_flat / curvature_ratio fields.
    Also flags, for every step but the first, whether its chosen phrase is a
    close textual match to the previous step's -- via
    `difflib.get_close_matches` -- since a chain that keeps echoing itself
    is a distinct (and detectable) kind of "flatness" from zero curvature.
    The first and last steps keep curvature=None (no two neighbors to bend
    between). Mutates and returns `chain`.
    """
    probs = [s.chosen_prob for s in chain]
    curvatures = compute_curvature(probs, flat_tolerance=flat_tolerance)
    for offset, (c, is_flat, ratio) in enumerate(curvatures):
        step = chain[offset + 1]
        step.curvature = c
        step.curvature_is_flat = is_flat
        step.curvature_ratio = ratio

    for i in range(1, len(chain)):
        prev_phrase = " ".join(chain[i - 1].chosen_word)
        this_phrase = " ".join(chain[i].chosen_word)
        matches = difflib.get_close_matches(this_phrase, [prev_phrase], n=1, cutoff=0.6)  # ontology: difflib.get_close_matches
        chain[i].close_to_previous = bool(matches)

    return chain


def analyze_chain_curvature(chain, top_k=3):
    """
    Chain-wide curvature analysis, using the five ontology functions that
    only make sense over a whole series rather than a single step:

      torch.topk        -> the `top_k` sharpest-bending steps (by |curvature|)
      torch.argmax        -> the single sharpest-bending step
      torch.multinomial     -> a magnitude-weighted random "representative" step
      torch.flip              -> reversal-symmetry self-check on the curvature formula
      torch.Tensor.transpose -> packs (step_index, curvature) into a real [N, 2] table
      collections.Counter     -> tallies steps as up-bending / down-bending / flat

    Returns a dict of results; returns None if the chain has no interior
    (curvature-bearing) steps to analyze.
    """
    interior = [s for s in chain if s.curvature is not None]
    if not interior:
        return None

    indices = [s.step_index for s in interior]
    curvature_values = [s.curvature for s in interior]
    curvature_tensor = torch.tensor(curvature_values)
    magnitude_tensor = curvature_tensor.abs()

    k = min(top_k, len(curvature_values))
    top_magnitudes, top_positions = torch.topk(magnitude_tensor, k=k)  # ontology: torch.topk
    top_steps = [(indices[p.item()], curvature_tensor[p.item()].item()) for p in top_positions]

    peak_position = torch.argmax(magnitude_tensor).item()  # ontology: torch.argmax
    peak_step = (indices[peak_position], curvature_values[peak_position])

    sample_weights = magnitude_tensor + 1e-6  # multinomial requires positive weights
    sampled_position = torch.multinomial(sample_weights, num_samples=1).item()  # ontology: torch.multinomial
    sampled_step = (indices[sampled_position], curvature_values[sampled_position])

    # Reversal-symmetry self-check: the second finite difference is invariant
    # under reversing the whole sequence (up to reindexing), so recomputing
    # curvature on the flipped log-prob series and flipping the result back
    # should reproduce the original curvature values -- a genuine correctness
    # check, not a decorative use of torch.flip.
    log_probs_tensor = torch.log(torch.tensor([s.chosen_prob for s in chain]).clamp_min(1e-12))
    flipped_log_probs = torch.flip(log_probs_tensor, dims=[0])  # ontology: torch.flip
    flipped_curvature = [
        (flipped_log_probs[i + 1] - 2.0 * flipped_log_probs[i] + flipped_log_probs[i - 1]).item()
        for i in range(1, len(flipped_log_probs) - 1)
    ]
    reversal_asymmetry = max(
        abs(a - b) for a, b in zip(curvature_values, list(reversed(flipped_curvature)))
    )

    # Pack (step_index, curvature) as a real [2, N] tensor and transpose it
    # into the [N, 2] table shape a report would actually want to iterate.
    stacked = torch.tensor([indices, curvature_values], dtype=torch.float32)
    table = stacked.transpose(0, 1)  # ontology: torch.Tensor.transpose

    sign_labels = [
        "flat" if s.curvature_is_flat else ("up" if s.curvature > 0 else "down")
        for s in interior
    ]
    sign_counts = Counter(sign_labels)  # ontology: collections.Counter

    return {
        "top_steps": top_steps,
        "peak_step": peak_step,
        "sampled_step": sampled_step,
        "reversal_asymmetry": reversal_asymmetry,
        "table": table,
        "sign_counts": sign_counts,
    }


def print_curvature_report(chain, top_k=3):
    """
    Prints the ontology's standard/role explanation for every function this
    pipeline uses -- all ten entries, not a subset -- then each step's
    curvature, then the chain-wide analysis from `analyze_chain_curvature`.
    """
    print("=== CURVATURE REPORT ===")
    for name in CURVATURE_ONTOLOGY_FUNCTIONS:
        standard, local = STANDARD_LIBRARY_FUNCTION_ONTOLOGY[name]
        print(f"  [{name}] standard: {standard} | here: {local}")
    print()

    for step in chain:
        if step.curvature is None:
            print(f"[Step {step.step_index + 1}] curvature: n/a (edge step) | chose: \"{'/'.join(step.chosen_word)}\"")
        else:
            print(f"[Step {step.step_index + 1}] chose: \"{'/'.join(step.chosen_word)}\"{step._curvature_suffix()}")

    analysis = analyze_chain_curvature(chain, top_k=top_k)
    if analysis is None:
        print("\n(chain too short for chain-wide curvature analysis)")
        return

    print("\n--- chain-wide analysis ---")
    top_str = ", ".join(f"step {i + 1} ({c:+.4f})" for i, c in analysis["top_steps"])
    print(f"top-{top_k} sharpest bends (torch.topk): {top_str}")
    peak_i, peak_c = analysis["peak_step"]
    print(f"single sharpest bend (torch.argmax):    step {peak_i + 1} ({peak_c:+.4f})")
    samp_i, samp_c = analysis["sampled_step"]
    print(f"magnitude-weighted random pick (torch.multinomial): step {samp_i + 1} ({samp_c:+.4f})")
    print(f"reversal-symmetry check (torch.flip), max discrepancy: {analysis['reversal_asymmetry']:.2e}")
    print(f"sign tally (collections.Counter): {dict(analysis['sign_counts'])}")
    print(f"(step_index, curvature) table shape after torch.Tensor.transpose: {tuple(analysis['table'].shape)}")


# ==========================================================
# 3. Contextual Markov Neural Generator with Logit Transpose
# ==========================================================

class MarkovSeedingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, context_size=5):
        super().__init__()
        self.context_size = context_size
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_size, embed_dim)
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.lm_head = nn.Sequential(
            nn.Linear(embed_dim * context_size + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(self, idx, transpose_mode="none", violation_strength=0.15, compute_gradient=False):
        B, T = idx.shape
        T_use = min(T, self.context_size)
        idx = idx[:, -T_use:]

        w_emb = self.word_embedding(idx)
        positions = torch.arange(0, T_use, dtype=torch.long, device=idx.device)
        p_emb = self.pos_embedding(positions)

        x = w_emb + p_emb
        x_flat = x.reshape(B, -1)

        context_summary = x.mean(dim=1)
        context_summary = self.context_proj(context_summary)

        logits = self.lm_head(torch.cat([x_flat, context_summary], dim=-1))

        if compute_gradient and not logits.requires_grad:
            logits.requires_grad_(True)
            logits.retain_grad()

        # Apply the selected logit transform (flat triangular or 3D CP-violation)
        if compute_gradient:
            logits, gradient = apply_logit_transform(
                logits,
                mode=transpose_mode,
                violation_strength=violation_strength,
                compute_gradient=True,
            )
            return logits, gradient

        logits = apply_logit_transform(
            logits,
            mode=transpose_mode,
            violation_strength=violation_strength,
            compute_gradient=False,
        )
        return logits

    # ==========================================================
    # Fundamentals: every generation mode below is one call into this
    # canonical rollout engine, parameterized by a handful of orthogonal
    # choices (selection rule, lookahead, trace recording). There is exactly
    # one place that knows how to pad a context, run the forward pass, mask
    # <UNK>, apply top_k, and turn logits into a distribution -- previously
    # this logic was duplicated (with small drifts) across three separate
    # generation methods.
    # ==========================================================

    def _pad_context(self, idx):
        """Right-aligns `idx` to exactly `context_size` tokens, left-padding
        by repeating the first available token if the sequence is shorter."""
        cond = idx[:, -self.context_size:]
        if cond.shape[1] < self.context_size:
            pad = cond[:, :1].repeat(1, self.context_size - cond.shape[1])
            cond = torch.cat([pad, cond], dim=1)
        return cond

    def _next_token_distribution(self, idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, temperature=1.0):
        """The single fundamental step: context -> padded context -> forward
        pass (+ logit transform) -> <UNK> mask -> top_k filter -> softmax.
        Returns (probs, padded_context) since callers that record a trace
        need the context tokens too.
        """
        cond = self._pad_context(idx)

        logits = self(
            cond,
            transpose_mode=transpose_mode,
            violation_strength=violation_strength,
            compute_gradient=False,
        ) + amp_boost
        logits[:, unk_id] = -float('inf')

        if top_k is not None and top_k < logits.shape[-1]:
            topv, topi = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, topi, topv)
            logits = filtered

        probs = F.softmax(logits / temperature, dim=-1)
        return probs, cond

    def _rollout_log_prob(self, start_idx, num_steps, amp_boost, unk_id, transpose_mode, violation_strength, top_k):
        """Simulates `num_steps` of greedy (argmax) continuation from
        `start_idx` and returns the summed log-probability of that path.
        This is the primitive lookahead is built on: "if I commit to this
        candidate, how well-supported is where the model deterministically
        goes from here?"
        """
        cur = start_idx
        cumulative_log_prob = 0.0
        for _ in range(num_steps):
            probs, _ = self._next_token_distribution(cur, amp_boost, unk_id, transpose_mode, violation_strength, top_k)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            p = probs[0, next_id.item()].item()
            cumulative_log_prob += math.log(max(p, 1e-12))
            cur = torch.cat([cur, next_id], dim=1)
        return cumulative_log_prob

    def _select_with_lookahead(self, probs, shadow_idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, width, depth):
        """Scores the top-`width` immediate candidates by cumulative
        log-probability (their own + `depth` steps of simulated greedy
        future) and returns the best one. Returns
        (next_id, [(candidate_id, cumulative_score), ...], overrode_greedy).
        """
        naive_best_id = torch.argmax(probs, dim=-1).item()

        w = min(width, probs.shape[-1])
        cand_probs, cand_idx = torch.topk(probs, k=w, dim=-1)

        scored = []
        best_id, best_score = None, -float('inf')
        for ci, cp in zip(cand_idx[0], cand_probs[0]):
            immediate_log_prob = math.log(max(cp.item(), 1e-12))
            hypothetical_idx = torch.cat([shadow_idx, ci.view(1, 1)], dim=1)
            future_log_prob = (
                self._rollout_log_prob(hypothetical_idx, depth, amp_boost, unk_id, transpose_mode, violation_strength, top_k)
                if depth > 0 else 0.0
            )
            cumulative = immediate_log_prob + future_log_prob
            scored.append((ci.item(), cumulative))
            if cumulative > best_score:
                best_score, best_id = cumulative, ci.view(1, 1)

        return best_id, scored, (best_id.item() != naive_best_id)

    @torch.no_grad()
    def _generate(
        self,
        seed_context_idx,
        num_steps,
        amp_boost,
        idx_to_word,
        unk_id,
        transpose_mode,
        violation_strength,
        selection="sample",       # "sample" (torch.multinomial) or "argmax" (deductive)
        temperature=1.0,          # only meaningful when selection="sample"
        top_k=25,
        lookahead_width=0,
        lookahead_depth=0,        # > 0 overrides token choice with simulated-future scoring
        top_candidates_shown=5,
        record_trace=False,
        verbose=False,
    ):
        """
        THE fundamental rollout engine. Every public generation method on
        this class is a thin, named wrapper around this one loop:
          - generate_population_seeds        -> selection="sample",  record_trace=False
          - generate_arbitrary_reasoning_chain -> selection="sample",  record_trace=True
          - generate_deductive_reasoning_chain -> selection="argmax", record_trace=True, lookahead optional

        Always runs on an independent clone of `seed_context_idx` -- callers
        never share trajectory state with each other.

        Returns (words, reasoning_chain) where reasoning_chain is None
        unless record_trace=True.
        """
        shadow_idx = seed_context_idx.clone()
        amp_boost = amp_boost.to(seed_context_idx.device)

        words = []
        chain: Optional[List[ReasoningStep]] = [] if record_trace else None

        for step in range(num_steps):
            temp = temperature if selection == "sample" else 1.0
            probs, cond = self._next_token_distribution(
                shadow_idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, temperature=temp
            )

            top_candidates, runner_up_gap = None, None
            if record_trace:
                k_show = min(top_candidates_shown, probs.shape[-1])
                top_probs, top_idx = torch.topk(probs, k=k_show, dim=-1)
                top_candidates = [
                    (idx_to_word[i.item()], p.item()) for i, p in zip(top_idx[0], top_probs[0])
                ]
                if len(top_candidates) >= 2:
                    runner_up_gap = top_candidates[0][1] - top_candidates[1][1]

            lookahead_scores, overrode_greedy = None, False

            if lookahead_depth > 0:
                next_id, scored_ids, overrode_greedy = self._select_with_lookahead(
                    probs, shadow_idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k,
                    lookahead_width, lookahead_depth,
                )
                if record_trace:
                    lookahead_scores = [(idx_to_word[i], s) for i, s in scored_ids]
            elif selection == "argmax":
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            chosen_word = idx_to_word[next_id.item()]
            chosen_prob = probs[0, next_id.item()].item()
            words.append(chosen_word)

            if record_trace:
                method_label = "deductive" if (selection == "argmax" or lookahead_depth > 0) else "sampled"
                step_record = ReasoningStep(
                    step_index=step,
                    context_tokens=[idx_to_word[i.item()] for i in cond[0]],
                    transform_mode=transpose_mode,
                    top_candidates=top_candidates,
                    chosen_word=chosen_word,
                    chosen_prob=chosen_prob,
                    selection_method=method_label,
                    runner_up_gap=runner_up_gap,
                    lookahead_depth=lookahead_depth,
                    lookahead_scores=lookahead_scores,
                    overrode_greedy=overrode_greedy,
                )
                chain.append(step_record)
                if verbose:
                    print(step_record.as_text())

            shadow_idx = torch.cat([shadow_idx, next_id], dim=1)

        return words, chain

    # ---------- named, backward-compatible wrappers around _generate ----------

    def generate_population_seeds(
        self, seed_context_idx, sequence_length, pop_size, amp_boost, idx_to_word, unk_id,
        transpose_mode="standard", violation_strength=0.15, temperature=1.2, top_k=25,
    ):
        """Produces `pop_size` independent stochastic rollouts of the final output text."""
        seeds = []
        for _ in range(pop_size):
            words, _ = self._generate(
                seed_context_idx, sequence_length, amp_boost, idx_to_word, unk_id,
                transpose_mode, violation_strength,
                selection="sample", temperature=temperature, top_k=top_k, record_trace=False,
            )
            seeds.append(words)
        return seeds

    def generate_arbitrary_reasoning_chain(
        self, seed_context_idx, num_reasoning_steps, amp_boost, idx_to_word, unk_id,
        transpose_mode="cp_violation_3d", violation_strength=0.15, temperature=0.8, top_k=25,
        top_candidates_shown=5, verbose=True,
    ):
        """
        A chain-of-reasoning that is ARBITRARY relative to the final output
        text: an independent stochastic shadow rollout, cloned from the same
        seed context but never shared with (or fed back into) whatever
        trajectory ultimately becomes the final generated text.
        `num_reasoning_steps` is unrelated to the final text's length.
        """
        _, chain = self._generate(
            seed_context_idx, num_reasoning_steps, amp_boost, idx_to_word, unk_id,
            transpose_mode, violation_strength,
            selection="sample", temperature=temperature, top_k=top_k,
            top_candidates_shown=top_candidates_shown, record_trace=True, verbose=verbose,
        )
        return chain

    def generate_deductive_reasoning_chain(
        self, seed_context_idx, num_reasoning_steps, amp_boost, idx_to_word, unk_id,
        transpose_mode="cp_violation_3d", violation_strength=0.15, top_k=25,
        top_candidates_shown=5, lookahead_width=3, lookahead_depth=2, verbose=True,
    ):
        """
        A DEDUCTIVE shadow rollout: no temperature, no sampling -- every
        choice is argmax, so re-running on the same context and weights
        always reproduces the same chain regardless of RNG state.

        With `lookahead_depth > 0`, a step doesn't just take the immediate
        best candidate: it simulates `lookahead_depth` steps of greedy
        continuation for the top `lookahead_width` candidates and picks
        whichever has the best cumulative log-probability. This can and
        does override the naive immediate-best pick
        (see `ReasoningStep.overrode_greedy`). Set `lookahead_depth=0` to
        disable and fall back to pure immediate-argmax deduction.
        """
        _, chain = self._generate(
            seed_context_idx, num_reasoning_steps, amp_boost, idx_to_word, unk_id,
            transpose_mode, violation_strength,
            selection="argmax", top_k=top_k, top_candidates_shown=top_candidates_shown,
            lookahead_width=lookahead_width, lookahead_depth=lookahead_depth,
            record_trace=True, verbose=verbose,
        )
        return chain

    def inspect_cp_violation_gradient(self, idx, violation_strength=0.15):
        """
        Convenience helper: runs a forward pass with transpose_mode='cp_violation_3d'
        and compute_gradient=True, returning (logits, gradient) so the caller can
        inspect how sensitive the CP-violation transform is to the raw logits at
        this context. Not used inside no_grad generation loops.
        """
        logits, gradient = self.forward(
            idx,
            transpose_mode="cp_violation_3d",
            violation_strength=violation_strength,
            compute_gradient=True,
        )
        return logits, gradient

    @torch.no_grad()
    def inspect_cp_violation_square_shift(self, idx, violation_strength=0.15):
        """
        A GRADIENT-FREE alternative to `inspect_cp_violation_gradient`. Where
        that method asks "how sensitive is the transform, via backprop /
        autograd", this one never touches calculus at all: it takes the raw
        (untransformed) logits, applies the square-shaped CP-violation
        transform to them directly, and returns the transform's own
        before/after difference as the sensitivity signal. The square shape
        is what makes this meaningful without a derivative -- because the
        transpose step is a literal matrix transpose (swap rows/columns of
        a real 2x2 square), the shift already tells you exactly which
        logits traded places and by how much, algebraically, with no need
        to differentiate anything.

        Returns:
            (transformed_logits, shift) where shift = transformed_logits -
            raw_logits, same shape as the model's output.
        """
        raw_logits = self.forward(idx, transpose_mode="none")
        transformed_logits = apply_cp_violation_transpose_square(
            raw_logits, violation_strength=violation_strength
        )
        shift = transformed_logits - raw_logits
        return transformed_logits, shift


# ==========================================================
# 4. Pipeline Execution Loop
# ==========================================================

if __name__ == "__main__":
    tri_agrees, tri_max_diff = verify_cp_violation_kernel_agreement(violation_strength=0.15)
    sq_agrees, sq_max_diff = verify_cp_violation_square_kernel_agreement(violation_strength=0.15)
    print(f"[self-test] triangle kernel agreement: {'PASS' if tri_agrees else 'FAIL'} (max_diff={tri_max_diff:.2e})")
    print(f"[self-test] square kernel agreement:   {'PASS' if sq_agrees else 'FAIL'} (max_diff={sq_max_diff:.2e})")

    ontology_counts = run_full_ontology_audit()
    ontology_all_used = audit_ontology_function_usage(ontology_counts)
    print(f"[self-test] ontology fully exercised: {'PASS' if ontology_all_used else 'FAIL'}")
    print()

    filename = input("Filename: ").strip()
    (
        word_to_idx,
        idx_to_word,
        vocab_size,
        amp_boost,
        frequent_trigrams,
        corpus_trigrams,
        frequent_words,
        context_to_row,
    ) = load_and_analyze_dataset(filename, context_window=5)

    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    unk_id = word_to_idx[unk_trigram]

    markov_seed_layer = MarkovSeedingLayer(vocab_size=vocab_size, embed_dim=64, context_size=5)
    known_words = sorted(set(frequent_words) | {w for tg in word_to_idx.keys() if tg != unk_trigram for w in tg})

    print("\nEnter text to guide generation. Type 'quit' to exit.")
    print("(Generation uses the 3D CP-violation transpose: transpose_mode='cp_violation_3d')")
    while True:
        raw_input_text = input("\nUSER: ").strip()
        if raw_input_text.lower() in {"quit", "exit", "stop"}:
            break

        tokens = raw_input_text.lower().split() if raw_input_text else []
        corrected_tokens = []
        for w in tokens:
            matches = difflib.get_close_matches(w, known_words, n=1, cutoff=0.0)
            corrected_tokens.append(matches[0] if matches else w)

        seed_context_words = corrected_tokens[-markov_seed_layer.context_size:]
        if len(seed_context_words) < markov_seed_layer.context_size:
            fallback_words = frequent_words[: markov_seed_layer.context_size - len(seed_context_words)]
            seed_context_words = fallback_words + seed_context_words

        while len(seed_context_words) < markov_seed_layer.context_size:
            seed_context_words.insert(0, frequent_words[0] if frequent_words else "<unk>")

        seed_context = [(seed_context_words[i], seed_context_words[i + 1], seed_context_words[i + 2])
                        for i in range(max(1, len(seed_context_words) - 2))]
        seed_context = seed_context[-1] if seed_context else frequent_trigrams[0]

        context_tensor = torch.tensor(
            [[word_to_idx.get(seed_context[0], unk_id),
              word_to_idx.get(seed_context[1], unk_id),
              word_to_idx.get(seed_context[2], unk_id),
              word_to_idx.get(seed_context[0], unk_id),
              word_to_idx.get(seed_context[1], unk_id)]],
            dtype=torch.long
        )

        # Gradient-free: inspect the square-shaped CP-violation shift at this
        # context, instead of computing an autograd gradient for it.
        square_logits, square_shift = markov_seed_layer.inspect_cp_violation_square_shift(
            context_tensor, violation_strength=0.15
        )
        print(f"[cp_violation_square] shift norm at this context: {square_shift.norm().item():.4f}")

        print("\n=== CHAIN OF REASONING (deductive with lookahead, independent of final text) ===")
        reasoning_chain = markov_seed_layer.generate_deductive_reasoning_chain(
            seed_context_idx=context_tensor,
            num_reasoning_steps=8,  # unrelated to the final text's length below
            amp_boost=amp_boost,
            idx_to_word=idx_to_word,
            unk_id=unk_id,
            transpose_mode="cp_violation_3d",
            violation_strength=0.15,
            top_k=25,
            top_candidates_shown=5,
            lookahead_width=3,   # how many immediate candidates get simulated forward
            lookahead_depth=2,   # how many steps ahead each candidate is rolled out
            verbose=True,  # prints one ReasoningStep line per step as it happens
        )

        attach_curvature_to_chain(reasoning_chain)
        print()
        print_curvature_report(reasoning_chain)

        print("\n=== GENERATING FINAL TEXT WITH 3D CP-VIOLATION TRANSPOSE ===")
        seeds = markov_seed_layer.generate_population_seeds(
            seed_context_idx=context_tensor,
            sequence_length=150,
            pop_size=1,
            amp_boost=amp_boost,
            idx_to_word=idx_to_word,
            unk_id=unk_id,
            transpose_mode="cp_violation_3d",
            violation_strength=0.15,
            temperature=0.8,
            top_k=25,
        )

        flattened_words = [word for trigram in seeds[0] for word in trigram]
        output_text = ' '.join(flattened_words)
        print("\n=== FINAL TEXT ===")
        print(f"{output_text}")

        print("-" * 65)
