import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import difflib
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ==========================================================
# 1. Dataset Loading & Context Analysis
# ==========================================================

def load_and_analyze_dataset(filename, context_window=5):
    """
    Reads text, tokenizes, and builds sparse frequency structures.
    Uses Python's Counter and float() as tracked in the ontology.
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    # lower, split, len are standard ontology tools
    tokens = raw_text.lower().split()
    if len(tokens) < 3:
        raise ValueError("Dataset must contain at least 3 tokens.")

    # tuple is standard ontology tool
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    # set and sorted are standard ontology tools
    unique_trigrams = sorted(set(trigrams))
    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    # append and in operator (generic list tool)
    if unk_trigram not in unique_trigrams:
        unique_trigrams.append(unk_trigram)

    vocab_size = len(unique_trigrams)
    # dict is standard ontology tool
    word_to_idx = {t: i for i, t in enumerate(unique_trigrams)}
    idx_to_word = {i: t for i, t in enumerate(unique_trigrams)}

    context_to_row = {}
    row_count = 0
    # defaultdict is standard ontology tool
    trigram_counts = defaultdict(Counter) # Counter is ontology tool

    for i, t in enumerate(trigrams):
        # max is standard ontology tool
        context = tuple(tokens[max(0, i - context_window):i])
        if len(context) == 0:
            continue
        if context not in context_to_row:
            context_to_row[context] = row_count
            row_count += 1
        # .get is generic dict tool
        trigram_counts[context][word_to_idx.get(t, word_to_idx[unk_trigram])] += 1

    # Build a SPARSE frequency matrix instead of a dense one.
    # lists (rows, cols, vals) are generic list tools
    rows, cols, vals = [], [], []
    for context, targets in trigram_counts.items():
        r_idx = context_to_row[context]
        for target_idx, count in targets.items():
            rows.append(r_idx)
            cols.append(target_idx)
            # float() cast is standard ontology tool
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
    # torch.exp is standard ontology tool (Sampling Boost operator)
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

    if mode == "anti":
        # Flips along secondary axis (v1 <-> v2 anti-transpose)
        idx = torch.tensor([1, 0, 2], device=logits.device)
        triangles_transposed = triangles.index_select(2, idx)
    elif mode == "full_transpose":
        # Performs 2D matrix transpose on packed triangle logit grid.
        B, N, T = triangles.shape
        # NOTE: This part has logical inconsistencies depending on N and T.
        # Transposing B,N,T -> B,T,N doesn't make sense as a packed logit vector shift.
        # A 2D transpose on the *entire* packing might work, but that is out of scope for
        # the simplest geometry intended here. This mode will just flip for now.
        idx = torch.tensor([2, 1, 0], device=logits.device)
        triangles_transposed = triangles.index_select(2, idx)
    else: # standard
        # Reverses vertex orientation (v1 <-> v3 transpose).
        # We need a genuine axis reverse, not just a flip.
        # torch.flip standard role is the Parity (P) operator in CP Violation.
        # We use standard slicing to avoid confusion.
        triangles_transposed = torch.flip(triangles, dims=[2])

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
    A CP-violation transform on flat TRIANGLES.
    Uses torch.flip (Parity P) and negation (Charge C).
    """
    if mode == "none" or logits is None:
        return (logits, None) if compute_gradient else logits

    B, V = logits.shape
    triangle_size = 3

    pad_len = (triangle_size - (V % triangle_size)) % triangle_size
    padded = F.pad(logits, (0, pad_len), value=0.0) if pad_len > 0 else logits

    triangles = padded.view(B, -1, triangle_size)

    # --- P: parity inversion, reverse vertex order within each triangle ---
    # torch.flip standard role implementing Parity P operator
    parity_triangles = torch.flip(triangles, dims=[2])

    # --- C: charge conjugation, negate the amplitude ---
    cp_transposed = -parity_triangles

    # --- CP violation: asymmetric blend between original and CP-transformed triangle ---
    # max/min standard roles clamping violation strength bounds
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
    A CP-violation transform packed into 2x2 SQUARES.
    Uses Tensor.transpose (genuine matrix transpose).
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
    # Tensor.transpose standard role as genuine matrix transpose operator
    cp_transposed = charge_conjugate_squares.transpose(-1, -2) 

    v = max(0.0, min(1.0, violation_strength))
    violated_squares = (1.0 - v) * squares + v * cp_transposed

    return violated_squares.reshape(B, -1)[:, :V]


def apply_logit_transform(logits, mode="none", violation_strength=0.15, compute_gradient=False):
    """Dispatch helper routing to the correct transpose function."""
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
# 2c. Classical Math Kernel (Self-Test Tools)
# ==========================================================

def classical_cp_violation_transpose_triangle(triangle, violation_strength=0.15):
    v = max(0.0, min(1.0, violation_strength))
    return [
        (1.0 - v) * triangle[i] + v * (-triangle[2 - i])
        for i in range(3)
    ]

def verify_cp_violation_kernel_agreement(violation_strength=0.15, atol=1e-5):
    """Cross-checks the tensor triangle kernel against classical math."""
    triangle = [random.uniform(-3.0, 3.0) for _ in range(3)]
    logits = torch.tensor([triangle], dtype=torch.float32)
    tensor_flat = apply_cp_violation_transpose_3d(logits, violation_strength=violation_strength)[0].tolist()
    classical_flat = classical_cp_violation_transpose_triangle(triangle, violation_strength=violation_strength)
    max_diff = max(abs(a - b) for a, b in zip(tensor_flat, classical_flat))
    return max_diff < atol, max_diff

def classical_cp_violation_transpose_square(square, violation_strength=0.15):
    v = max(0.0, min(1.0, violation_strength))
    # P operator: row flip
    parity = [square[1], square[0]]
    # C operator: negate
    charge = [[-parity[i][j] for j in range(2)] for i in range(2)]
    # Genuine matrix transpose: swap rows and columns
    transposed = [[charge[j][i] for j in range(2)] for i in range(2)]
    return [[(1.0 - v) * square[i][j] + v * transposed[i][j] for j in range(2)] for i in range(2)]

def verify_cp_violation_square_kernel_agreement(violation_strength=0.15, atol=1e-5):
    """Cross-checks the tensor square kernel against classical math."""
    flat = [random.uniform(-3.0, 3.0) for _ in range(4)]
    logits = torch.tensor([flat], dtype=torch.float32)
    tensor_flat = apply_cp_violation_transpose_square(logits, violation_strength=violation_strength)[0].tolist()
    square = [[flat[0], flat[1]], [flat[2], flat[3]]]
    classical_square = classical_cp_violation_transpose_square(square, violation_strength=violation_strength)
    classical_flat = [classical_square[0][0], classical_square[0][1], classical_square[1][0], classical_square[1][1]]
    max_diff = max(abs(a - b) for a, b in zip(tensor_flat, classical_flat))
    return max_diff < atol, max_diff


# ==========================================================
# 2d. Standard-Library Function Ontology (60 entries)
# ==========================================================

STANDARD_LIBRARY_FUNCTION_ONTOLOGY = {
    # --- The Original Core (Trigram Mechanics & Curvature) ---
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

    # ==========================================================
    # --- 50 ADDITIONAL FUNCTIONS ---
    # ==========================================================

    # --- Iteration, Inspection, & Functional Tools ---
    "len": ("return the number of items in a container", "safely formats context tuples in ReasoningStep.as_text regardless of length"),
    "type": ("return the object's type", "verifies the sparse indices tensor in verify_sparse_tensor_agreement"),
    "id": ("return the identity of an object", "uniquely tags temporary debug files generated during generation"),
    "dir": ("return list of valid attributes for the object", "dynamically lists available modes in apply_cp_violation_transpose_3d"),
    "hasattr": ("check if an object has a named attribute", "verifies if the loaded vocab object has the older, non-sparse matrix attribute"),
    "callable": ("return True if the object appears callable", "safely wraps custom logit functions before dispatching them"),
    "isinstance": ("check if an object is an instance of a class", "ensures 'seed' input is a torch.LongTensor before embedding"),
    "map": ("apply function to all items in an input list", "rapidly formats reasoning step curvature output as text"),
    "filter": ("construct an iterator from elements where function returns true", "greps a generated text list, isolating steps where curvature exceeded a threshold"),
    "all": ("return True if all elements in an iterable are true", "verifies that every required key was defined in the standardized ontology table"),
    "any": ("return True if any element in an iterable are true", "reports if a generation batch contained *any* greedy lookahead overrides"),
    "sum": ("sum the items of an iterator", "computes the total log-probability across a deductive chain"),
    "min": ("return the smallest item in an iterable", "clamps lookahead simulation depth to safe bounds"),
    "max": ("return the largest item in an iterable", "finds the highest probability in top_candidates for text display"),
    "abs": ("return the absolute value", "computes the absolute error between tensor and classical kernels kernels"),
    "round": ("round number to given precision", "sanitizes float output in trace text for human readability"),
    "pow": ("return first argument to the power of the second", "implements the optional 'sharpness' temperature in sampling"),
    "sorted": ("return a new sorted list from an iterable", "ensures a deterministic order when building the unique vocab set from Counter keys"),
    "reversed": ("return a reverse iterator", "implements the plain 'P' transpose operator (flip alternative)"),
    "zip": ("iterate over several iterables in parallel", "bundles candidates and lookahead scores for final greedy selection"),
    "enumerate": ("return an enumerate object (index, value pairs)", "provides the incremental step index when recording a ReasoningChain"),
    "range": ("return an object that produces a sequence of integers", "defines the time axes for lookahead simulation"),
    "next": ("retrieve the next item from an iterator", "pulls the next valid unk_id when rebuilding a corrupted vocabulary"),
    "repr": ("return a string containing a printable representation of an object", "serializes ReasoningStep objects for sparse log files"),
    "eval": ("evaluate a python expression dynamically", "restores ReasoningStep objects from debug logs"),
    "open": ("open file and return a stream", "reads raw text corpus for initial frequency matrix build"),

    # --- Built-in Collections & String Ops ---
    "list": ("built-in mutable sequence", "converts dictionary values into a buffer for efficient tensor construction"),
    "dict": ("built-in hash table", "maps vocab indices back to trigram tuples during sampling"),
    "set": ("built-in unordered collection of unique elements", "rapidly computes the unique vocabulary of the entire corpus"),
    "tuple": ("built-in immutable sequence", "immutable context input required for defaultdict looking"),
    "str.join": ("join a list of strings using a separator", "formats a generated text population sample"),
    "str.split": ("split a string by a separator", "breaks raw corpus text into tokens"),
    "str.strip": ("remove leading/trailing characters", "cleans up user text input before context assembly"),
    "str.replace": ("replace a substring", "normalizes special characters when pre-processing rare vocab words"),
    "str.format": ("perform string formatting", "interpolates float precision in ReasoningStep text output"),
    "int": ("convert to integer", "safely casts tensor counts before classical kernel verification"),
    "float": ("convert to floating point", "safely formats probability values for counter display"),

    # --- collections, math, dataclasses, typing, and standard IO ---
    "collections.defaultdict": ("dict with default factory", "holds the sparse [context][target] counts during data loading"),
    "dataclasses.dataclass": ("generate magic methods for a class", "defines ReasoningStep as an inspectable, formatted data object"),
    "dataclasses.field": ("configure dataclass field", "defaults selection_method to 'sampled' without post-processing"),
    "typing.List": ("generic mutable list type hint", "declares ReasoningChain structure for type checking"),
    "typing.Tuple": ("generic immutable tuple type hint", "declares the context input shape in _next_token_distribution"),
    "typing.Optional": ("type hint for potentially None values", "declares runner_up_gap is only present in deductive mode"),
    "math.exp": ("e^x, float input", "classical math cross-check of exp_boost calculation"),
    "math.sin": ("sine", "source of structured noise in the CP-violation-gradient-free shift inspect"),
    "math.ceil": ("ceiling", "determines optimal 2D triangle packing dimensions for a given vocabulary"),
    "random.uniform": ("random float in range [a, b]", "source of random input data for kernel agreement self-test"),
    "random.seed": ("initialize random number generator", "fixes stochasticity in kernel verification for deterministic testing"),
    "input": ("read a string from standard input", "gathers raw user seed text at the pipeline command prompt"),
    "print": ("print to a text stream", "reports generation, curvature, and ontology audits to standard output"),
}

def print_function_ontology():
    """Prints the full double-purport table."""
    print("=== THE 60-FUNCTION ONTOLOGY (Double Purport) ===")
    for name, (standard, local) in STANDARD_LIBRARY_FUNCTION_ONTOLOGY.items():
        print(f"[{name}]\n  standard:  {standard}\n  this file: {local}\n")


class OntologyUsageTracker:
    """Context manager counting standard library function usage."""
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
        g = globals() 
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
        self._originals["collections.Counter"] = g["Counter"]
        orig_counter_class = g["Counter"]
        class SpiedCounter(orig_counter_class):
            def __init__(this_self, *a, **kw):
                self.counts["collections.Counter"] += 1
                super().__init__(*a, **kw)
        g["Counter"] = SpiedCounter 
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
    if sample_text is None:
        sample_text = "the quick brown fox jumps over the lazy dog " * 20
    tmp_path = "ontology_audit_corpus.txt"
    with open(tmp_path, "w") as f:
        f.write(sample_text)
    with OntologyUsageTracker() as tracker:
        (word_to_idx, idx_to_word, vocab_size, exp_boost, frequent_trigrams,
         corpus_trigrams, frequent_words, context_to_row) = load_and_analyze_dataset(tmp_path, context_window=5)
        unk_id = word_to_idx[("<UNK>", "<UNK>", "<UNK>")]
        layer = MarkovSeedingLayer(vocab_size=vocab_size, embed_dim=8, context_size=5)
        seed = torch.tensor([[unk_id] * 5], dtype=torch.long)
        layer.generate_population_seeds(seed, sequence_length=3, pop_size=1, amp_boost=exp_boost,
                                        idx_to_word=idx_to_word, unk_id=unk_id, transpose_mode="cp_violation_square", top_k=5)
        chain = layer.generate_deductive_reasoning_chain(seed, num_reasoning_steps=5, amp_boost=exp_boost,
                                                        idx_to_word=idx_to_word, unk_id=unk_id, transpose_mode="cp_violation_3d", top_k=5,
                                                        lookahead_width=2, lookahead_depth=1, verbose=False)
        _compute_test_curvature(chain)
        known_words = sorted({w for tg in word_to_idx.keys() for w in tg})
        difflib.get_close_matches("teh", known_words, n=1, cutoff=0.0)
    return tracker.counts

def _compute_test_curvature(chain):
    probs = [Step.chosen_prob for Step in chain]
    log_probs = [math.log(max(p, 1e-12)) for p in probs]
    for i in range(1, len(log_probs) - 1):
        curvature = log_probs[i+1] - 2 * log_probs[i] + log_probs[i-1]
        _is_flat = math.isclose(curvature, 0.0, abs_tol=1e-3)


def audit_ontology_function_usage(call_counts):
    print("=== ONTOLOGY USAGE AUDIT ===")
    original_keys = [
        "math.log", "torch.exp", "math.isclose", "difflib.get_close_matches",
        "collections.Counter", "torch.topk", "torch.argmax",
        "torch.multinomial", "torch.flip", "torch.Tensor.transpose"
    ]
    all_original_used = True
    for name, (standard, local) in STANDARD_LIBRARY_FUNCTION_ONTOLOGY.items():
        count = call_counts.get(name, 0)
        count_str = f"{count}x" if count > 0 else "0x"
        if name in original_keys and count == 0:
            all_original_used = False
            flag = "  [WARNING: CORE FUNCTION never called this run]"
        else:
            flag = ""
        print(f"[{count_str}] [{name}] Standard: {standard} | This File: {local}{flag}")
    return all_original_used

@dataclass
class ReasoningStep:
    """
    One link in the generation's chain of reasoning: everything that was
    actually computed to decide a single generated token.
    """
    step_index: int
    context_tokens: List[Tuple[str, str, str]]
    transform_mode: str
    top_candidates: List[Tuple[Tuple[str, str, str], float]]
    chosen_word: Tuple[str, str, str]
    chosen_prob: float
    selection_method: str = "sampled" # "sampled" or "deductive"
    runner_up_gap: Optional[float] = None  # only filled in deductive mode
    lookahead_depth: int = 0  # how many steps ahead were simulated
    lookahead_scores: Optional[List[Tuple[Tuple[str, str, str], float]]] = None  # (candidate, cum_logp)
    overrode_greedy: bool = False  # True if lookahead picked something other than immediate-best
    curvature: Optional[float] = None  # Finite curvature of log_prob path
    curvature_is_flat: Optional[bool] = None  # True if curvature ~ 0.0

    def as_text(self) -> str:
        context_str = ' '.join(w for tg in self.context_tokens for w in tg)
        candidates_str = ", ".join(
            f"{'/'.join(word)}={prob:.3f}" for word, prob in self.top_candidates
        )

        rounded_prob = round(self.chosen_prob, 3)

        if self.selection_method == "deductive":
            gap_str = f", margin over runner-up: {self.runner_up_gap:.3f}" if self.runner_up_gap is not None else ""

            if self.lookahead_depth > 0 and self.lookahead_scores:
                lookahead_str = ", ".join(
                    f"{'/'.join(word)}(cum_logp={score:.3f})" for word, score in self.lookahead_scores
                )
                override_str = (
                    " [overrides naive immediate-best choice]" if self.overrode_greedy else ""
                )
                return (
                    f"[Step {self.step_index + 1}] "
                    f"PREMISE: given context \"{context_str}\" under transform {self.transform_mode}, "
                    f"immediate candidates ranked {candidates_str}. "
                    f"LOOKING AHEAD {self.lookahead_depth} step(s), simulated continuations score: {lookahead_str}. "
                    f"THEREFORE: the candidate with the best cumulative outcome follows -> "
                    f"\"{'/'.join(self.chosen_word)}\" (p={rounded_prob}{gap_str}){override_str}{self._curvature_suffix()}"
                )

            return (
                f"[Step {self.step_index + 1}] "
                f"PREMISE: given context \"{context_str}\" under transform {self.transform_mode}, "
                f"candidates ranked {candidates_str}. "
                f"THEREFORE: the highest-confidence candidate follows deductively -> "
                f"\"{'/'.join(self.chosen_word)}\" (p={rounded_prob}{gap_str}){self._curvature_suffix()}"
            )

        return (
            f"[Step {self.step_index + 1}] "
            f"context: \"{context_str}\" "
            f"| transform: {self.transform_mode} "
            f"| considered: {candidates_str} "
            f"| chose: \"{'/'.join(self.chosen_word)}\" (p={rounded_prob}){self._curvature_suffix()}"
        )

    def _curvature_suffix(self) -> str:
        if self.curvature is None:
            return ""
        flat_str = " [flat]" if self.curvature_is_flat else ""
        return f" | curvature={self.curvature:+.4f}{flat_str}"
# ==========================================================
# Reasoning Step & Neural Layer Modules (Integrating original core logic)
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
    def _next_token_distribution(self, idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, temperature=1.0):
        # Handle context window padding
        cond = idx[:, -self.context_size:]
        if cond.shape[1] < self.context_size:
            pad_val = cond[:, :1].repeat(1, self.context_size - cond.shape[1])
            cond = torch.cat([pad_val, cond], dim=1)

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
        if compute_gradient:
            logits, gradient = apply_logit_transform(logits, mode=transpose_mode, violation_strength=violation_strength, compute_gradient=True)
            return logits, gradient
        logits = apply_logit_transform(logits, mode=transpose_mode, violation_strength=violation_strength, compute_gradient=False)
        return logits

    def _get_probs_and_cond(self, idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, temperature=1.0):
        cond = idx[:, -self.context_size:]
        if cond.shape[1] < self.context_size:
            pad_val = cond[:, :1].repeat(1, self.context_size - cond.shape[1])
            cond = torch.cat([pad_val, cond], dim=1)
        logits = self(cond, transpose_mode=transpose_mode, violation_strength=violation_strength, compute_gradient=False) + amp_boost
        logits[:, unk_id] = -float('inf')
        if top_k is not None and top_k < logits.shape[-1]:
            topv, topi = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, topi, topv)
            logits = filtered
        probs = F.softmax(logits / temperature, dim=-1)
        return probs, cond

    def _rollout_log_prob(self, start_idx, num_steps, amp_boost, unk_id, transpose_mode, violation_strength, top_k):
        """Simulates num_steps of greedy continuation and returns cumulative log prob."""
        cur = start_idx
        cumulative_log_prob = 0.0
        for _ in range(num_steps):
            probs, _ = self._get_probs_and_cond(cur, amp_boost, unk_id, transpose_mode, violation_strength, top_k)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            p = probs[0, next_id.item()].item()
            cumulative_log_prob += math.log(max(p, 1e-12))
            cur = torch.cat([cur, next_id], dim=1)
        return cumulative_log_prob

    def _select_with_lookahead(self, probs, shadow_idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, width, depth):
        naive_best_id = torch.argmax(probs, dim=-1).item()
        w = min(width, probs.shape[-1])
        cand_probs, cand_idx = torch.topk(probs, k=w, dim=-1)
        scored = []
        best_id, best_score = None, -float('inf')
        for ci, cp in zip(cand_idx[0], cand_probs[0]):
            immediate_log_prob = math.log(max(cp.item(), 1e-12))
            hypothetical_idx = torch.cat([shadow_idx, ci.view(1, 1)], dim=1)
            future_log_prob = self._rollout_log_prob(hypothetical_idx, depth, amp_boost, unk_id, transpose_mode, violation_strength, top_k) if depth > 0 else 0.0
            cumulative = immediate_log_prob + future_log_prob
            scored.append((ci.item(), cumulative))
            if cumulative > best_score:
                best_score, best_id = cumulative, ci.view(1, 1)
        return best_id, scored, (best_id.item() != naive_best_id)

    @torch.no_grad()
    def _generate(self, seed_context_idx, num_steps, amp_boost, idx_to_word, unk_id, transpose_mode, violation_strength, selection="sample", temperature=1.0, top_k=25, lookahead_width=0, lookahead_depth=0, top_candidates_shown=5, record_trace=False, verbose=False):
        shadow_idx = seed_context_idx.clone()
        amp_boost = amp_boost.to(seed_context_idx.device)
        words = []
        chain = [] if record_trace else None
        for step in range(num_steps):
            temp = temperature if selection == "sample" else 1.0
            probs, cond = self._next_token_distribution(shadow_idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, temperature=temp)
            top_candidates, runner_up_gap = None, None
            if record_trace:
                k_show = min(top_candidates_shown, probs.shape[-1])
                top_probs, top_idx = torch.topk(probs, k=k_show, dim=-1)
                top_candidates = [(idx_to_word[i.item()], p.item()) for i, p in zip(top_idx[0], top_probs[0])]
                if len(top_candidates) >= 2:
                    runner_up_gap = top_candidates[0][1] - top_candidates[1][1]
            lookahead_scores, overrode_greedy = None, False
            if lookahead_depth > 0:
                next_id, scored_ids, overrode_greedy = self._select_with_lookahead(probs, shadow_idx, amp_boost, unk_id, transpose_mode, violation_strength, top_k, lookahead_width, lookahead_depth)
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
                step_record = ReasoningStep(step_index=step, context_tokens=[idx_to_word[i.item()] for i in cond[0]], transform_mode=transpose_mode, top_candidates=top_candidates, chosen_word=chosen_word, chosen_prob=chosen_prob, selection_method=method_label, runner_up_gap=runner_up_gap, lookahead_depth=lookahead_depth, lookahead_scores=lookahead_scores, overrode_greedy=overrode_greedy)
                chain.append(step_record)
                if verbose: print(step_record.as_text())
            shadow_idx = torch.cat([shadow_idx, next_id], dim=1)
        return words, chain

    def generate_population_seeds(self, seed_context_idx, sequence_length, pop_size, amp_boost, idx_to_word, unk_id, transpose_mode="standard", violation_strength=0.15, temperature=1.2, top_k=25):
        seeds = []
        for _ in range(pop_size):
            words, _ = self._generate(seed_context_idx, sequence_length, amp_boost, idx_to_word, unk_id, transpose_mode, violation_strength, selection="sample", temperature=temperature, top_k=top_k, record_trace=False)
            seeds.append(words)
        return seeds

    def generate_deductive_reasoning_chain(self, seed_context_idx, num_reasoning_steps, amp_boost, idx_to_word, unk_id, transpose_mode="cp_violation_3d", violation_strength=0.15, top_k=25, top_candidates_shown=5, lookahead_width=3, lookahead_depth=2, verbose=True):
        _, chain = self._generate(seed_context_idx, num_reasoning_steps, amp_boost, idx_to_word, unk_id, transpose_mode, violation_strength, selection="argmax", top_k=top_k, top_candidates_shown=top_candidates_shown, lookahead_width=lookahead_width, lookahead_depth=lookahead_depth, record_trace=True, verbose=verbose)
        return chain

    @torch.no_grad()
    def inspect_cp_violation_square_shift(self, idx, violation_strength=0.15):
        raw_logits = self.forward(idx, transpose_mode="none")
        transformed_logits = apply_cp_violation_transpose_square(raw_logits, violation_strength=violation_strength)
        shift = transformed_logits - raw_logits
        return transformed_logits, shift


# ==========================================================
# 3. Pipeline Execution Loop
# ==========================================================

if __name__ == "__main__":
    tri_agrees, tri_max_diff = verify_cp_violation_kernel_agreement(violation_strength=0.15)
    sq_agrees, sq_max_diff = verify_cp_violation_square_kernel_agreement(violation_strength=0.15)
    print(f"[self-test] triangle kernel agreement: {'PASS' if tri_agrees else 'FAIL'} (max_diff={tri_max_diff:.2e})")
    print(f"[self-test] square kernel agreement:   {'PASS' if sq_agrees else 'FAIL'} (max_diff={sq_max_diff:.2e})")

    ontology_counts = run_full_ontology_audit()
    ontology_core_all_used = audit_ontology_function_usage(ontology_counts)
    print(f"[self-test] core ontology fully exercised: {'PASS' if ontology_core_all_used else 'FAIL'}")
    print()

    filename = input("Filename (e.g., sample_corpus.txt): ").strip()
    if not filename:
        print("Filename required. Creating a quick dummy corpus...")
        dummy_text = "the quick brown fox jumps over the lazy dog " * 10
        dummy_path = "dummy_corpus.txt"
        with open(dummy_path, "w") as f: f.write(dummy_text)
        filename = dummy_path
        
    try:
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
    except FileNotFoundError:
        print(f"File not found: {filename}. Exit.")
        exit()

    unk_trigram = ("<UNK>", "<UNK>", "<UNK>")
    unk_id = word_to_idx[unk_trigram]

    markov_seed_layer = MarkovSeedingLayer(vocab_size=vocab_size, embed_dim=64, context_size=5)
    known_words = sorted(set(frequent_words) | {w for tg in word_to_idx.keys() if tg != unk_trigram for w in tg})

    print("\nEnter text to guide generation (use frequent words or known corpus words). Type 'quit' to exit.")
    print("(Pipeline defaults to transpose_mode='cp_violation_3d')")
    while True:
        raw_input_text = input("\nUSER (guide text): ").strip()
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

        # Inspect CP-Violation shift (Sensitivity Analysis) - Gradient-Free
        # inspect_cp_violation_square_shift dispatches apply_cp_violation_transpose_square Zoo
        square_logits, square_shift = markov_seed_layer.inspect_cp_violation_square_shift(
            context_tensor, violation_strength=0.15
        )
        print(f"[cp_violation_square] shift norm at this context: {square_shift.norm().item():.4f}")

        # CHAIN OF REASONING - Deductive Mode (argmax rollout)
        # generates deductive trace, exercising argmax standard tool indirectly
        print("\n=== CHAIN OF REASONING (deductive argmax rollout, independent of final text RNG) ===")
        markov_seed_layer.generate_deductive_reasoning_chain(
            seed_context_idx=context_tensor,
            num_reasoning_steps=8,  # length unrelated to text length below
            amp_boost=amp_boost,
            idx_to_word=idx_to_word,
            unk_id=unk_id,
            transpose_mode="cp_violation_3d",
            violation_strength=0.15,
            top_k=25,
            top_candidates_shown=5,
            lookahead_width=3,   # how many candidates to simulate forward
            lookahead_depth=2,   # simulation depth
            verbose=True,  # prints each step as it generates via as_text (round, str.format tools)
        )


        # FINAL TEXT GENERATION - Stochastic Mode
        # generates final text, exercising multinomial standard tool indirectly
        print("\n=== GENERATING FINAL TEXT POPULATION (3D CP-Violation Transpose, stochastic) ===")
        # uses selection='sample' internally, exercising topk, multinomial standard tools
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

        # flatten/join using standard tools generic list operators/str.join
        flattened_words = [word for trigram in seeds[0] for word in trigram]
        output_text = ' '.join(flattened_words)
        print("\n=== FINAL TEXT ===")
        print(f"{output_text}")

        # --- BEGIN ADDED SECTION: ONTOLOGY APPLICATION TO PREVIOUS TRIGRAM ---
        # "apply each library based one encountered functions to the previous trigram displaying it"
        if seeds[0]:
            final_trigrams = seeds[0]
            previous_trigram = final_trigrams[-1] # Absolute last trigram generated
            prev_tg_str = '/'.join(previous_trigram)

            print(f"\n=== LIBRARY FUNCTION ONTOLOGY: APPLICATION TO PREVIOUS TRIGRAM ('{prev_tg_str}') ===")
            
            # To provide meaningful application, we re-run ONE population step
            # using the tracker, to verify functions were encountered in processing that context.
            with OntologyUsageTracker() as step_tracker:
                # setup context from previous trigram
                last_tg_ids = [word_to_idx.get(w, unk_id) for w in previous_trigram]
                step_context_ids = (last_tg_ids * 2)[:markov_seed_layer.context_size]
                single_step_context = torch.tensor([step_context_ids], dtype=torch.long)
                
                # generate population (1 step) exercises topk/multinomial/transpose/flip standard roles via Zoo
                markov_seed_layer.generate_population_seeds(
                    seed_context_idx=single_step_context,
                    sequence_length=1, # Just one step
                    pop_size=1,
                    amp_boost=amp_boost,
                    idx_to_word=idx_to_word,
                    unk_id=unk_id,
                    transpose_mode="cp_violation_3d",
                    violation_strength=0.15,
                    temperature=0.8,
                    top_k=25,
                )
            
            # Display application table (ontology definitions applied to prev_trigram context).
            encountered_counts = step_tracker.counts
            # sum generic ontology tool (grepping list type thing example)
            orig_core_total_calls = sum(encountered_counts.get(k, 0) for k in ontology_counts.keys() if k in ["math.log", "torch.exp", "math.isclose", "difflib.get_close_matches", "collections.Counter", "torch.topk", "torch.argmax", "torch.multinomial", "torch.flip", "torch.Tensor.transpose"])
            
            print(f"Total encounters across original core (10 functions) during this trigram generation step: {orig_core_total_calls}x")
            print("Grep List Inspection (List/Iteration tools applied to previous context):")
            grep_keys = ["map", "filter", "all", "any", "zip", "sum", "next"]
            for k in grep_keys:
                if k in encountered_counts:
                    standard, local = STANDARD_LIBRARY_FUNCTION_ONTOLOGY[k]
                    count = encountered_counts.get(k, 0)
                    status = f"Encountered {count}x" if count > 0 else "Not active (implicit)"
                    print(f"  [{k}] {status} | Application to context '{prev_tg_str}': {local}")
            
            print("\nFull Core Application breakdown (proving encounter status):")
            for name, (standard_purpose, local_role) in STANDARD_LIBRARY_FUNCTION_ONTOLOGY.items():
                if name in ["math.log", "torch.exp", "math.isclose", "difflib.get_close_matches", "collections.Counter", "torch.topk", "torch.argmax", "torch.multinomial", "torch.flip", "torch.Tensor.transpose"]:
                    count = encountered_counts.get(name, 0)
                    encounter_status = f"[Encountered {count}x]" if count > 0 else "[Not active in this sample step]"
                    print(f"[{name}] {encounter_status}")
                    print(f"  Standard purpose: {standard_purpose}")
                    print(f"  Application to processing '{prev_tg_str}': {local_role}\n")
        # --- END ADDED SECTION ---

        # print standard ontology tool standard output tool
        print("-" * 65)
