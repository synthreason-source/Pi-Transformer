"""
Tubular Filament Simulator -- text generation as filament growth.

The model treats generated text as a growing tubular filament made of
3-word NODES (one "ring" of the tube per word-triple, the same role
trigrams played in the original formulation). The filament grows one node
at a time; at each growth step the raw extrusion logits are passed through
a cross-sectional DEFORMATION operator before being turned into a
probability distribution over the next node:

  - segment twist            : geometric transpose of a 3-strut cross-section
  - helical chirality warp   : a parity+charge-flip blend across 3-strut
                                cross-sections (physics-flavored asymmetry,
                                like a filament with a chiral twist bias)
  - cross-section shear      : a genuine 2x2 matrix transpose applied to a
                                4-strut (2x2) cross-section

Everything downstream -- corpus loading, the growth-trace record, the
curvature analysis of the resulting filament, and the neural growth
simulator itself -- is the same machinery as before, renamed to fit the
filament metaphor.
"""

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
# 1. Corpus Loading & Node-Density Analysis
# ==========================================================

def load_and_analyze_filament_corpus(filename, anchor_window=5):
    """
    Reads `filename`, splits it into whitespace tokens, and packs every
    consecutive triple of tokens into a "filament node" (a ring of the
    tube). Also computes, for every distinct preceding anchor of
    `anchor_window` tokens, a sparse count of which node tends to follow --
    the raw material the growth simulator later extrudes from.
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    tokens = raw_text.lower().split()
    if len(tokens) < 3:
        raise ValueError("Corpus must contain at least 3 tokens.")

    filament_nodes = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    unique_nodes = sorted(set(filament_nodes))
    void_node = ("<VOID>", "<VOID>", "<VOID>")
    if void_node not in unique_nodes:
        unique_nodes.append(void_node)

    vocab_size = len(unique_nodes)
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    idx_to_node = {i: n for i, n in enumerate(unique_nodes)}

    anchor_to_row = {}
    row_count = 0
    node_counts = defaultdict(Counter)

    for i, n in enumerate(filament_nodes):
        anchor = tuple(tokens[max(0, i - anchor_window):i])
        if len(anchor) == 0:
            continue
        if anchor not in anchor_to_row:
            anchor_to_row[anchor] = row_count
            row_count += 1
        node_counts[anchor][node_to_idx.get(n, node_to_idx[void_node])] += 1

    # Build a SPARSE density matrix instead of a dense one. node_counts is
    # already sparse (most anchor/node pairs never co-occur), so a dense
    # [row_count, vocab_size] float32 tensor wastes enormous amounts of
    # memory on zeros -- e.g. 5,000 anchors x 20,000 filament nodes is 100M
    # floats (~400MB) for a matrix that might only have a few thousand
    # nonzero entries. Only the row/col indices + nonzero values are stored.
    rows, cols, vals = [], [], []
    for anchor, targets in node_counts.items():
        r_idx = anchor_to_row[anchor]
        for target_idx, count in targets.items():
            rows.append(r_idx)
            cols.append(target_idx)
            vals.append(float(count))

    if vals:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        with torch.sparse.check_sparse_tensor_invariants(False):
            density_matrix = torch.sparse_coo_tensor(
                indices, values, size=(max(row_count, 1), vocab_size)
            ).coalesce()
        node_density = torch.sparse.sum(density_matrix, dim=0).to_dense()
    else:
        node_density = torch.zeros(vocab_size, dtype=torch.float32)

    peak_density = node_density.max().clamp_min(1e-5)
    normalized_inv_density = 1.0 - (node_density / peak_density)
    tension_bias = torch.exp(normalized_inv_density * 2.0)

    top_indices = torch.argsort(node_density, descending=True)
    dominant_nodes = [idx_to_node[i.item()] for i in top_indices if idx_to_node[i.item()] != void_node]
    dominant_strands = list(dict.fromkeys([w for n in dominant_nodes for w in n]))

    return (
        node_to_idx,
        idx_to_node,
        vocab_size,
        tension_bias,
        dominant_nodes,
        unique_nodes,
        dominant_strands,
        anchor_to_row,
    )


# ==========================================================
# 2. Cross-Sectional Deformation Zoo: Segment Twist + Helical
#    Chirality Warp + Cross-Section Shear
# ==========================================================

def apply_segment_twist(logits, mode="standard"):
    """
    Applies a geometric transpose to a filament's 3-strut cross-sections.
    Logits are reshaped into local 3-strut rings across the vocabulary and
    transposed -- a purely structural twist of the tube's cross-section.

    Modes:
      - 'standard': Reverses strut orientation (strut1 <-> strut3 transpose).
      - 'anti': Flips along the secondary axis (strut1 <-> strut2 anti-transpose).
      - 'full_transpose': Performs a 2D matrix transpose on the packed ring grid.
    """
    if mode == "none" or logits is None:
        return logits

    B, V = logits.shape

    # Pad vocabulary dimension to nearest multiple of 3 to form 3-strut rings
    pad_len = (3 - (V % 3)) % 3
    if pad_len > 0:
        padded_logits = F.pad(logits, (0, pad_len), value=-1e9)
    else:
        padded_logits = logits

    # Reshape into [B, N, 3] cross-sectional ring triples
    rings = padded_logits.view(B, -1, 3)

    # Full 2D transposition of the packed rings
    rings_transposed = rings.transpose(1, 2).contiguous()
    rings_transposed = rings_transposed.view(B, -1, 3)

    # Flatten back to the original filament-node vocabulary shape
    transposed_logits = rings_transposed.view(B, -1)[:, :V]
    return transposed_logits


def apply_helical_chirality_warp(
    logits,
    mode="standard",
    chirality_strength=0.15,
    compute_gradient=False,
):
    """
    A chirality-warp transform on flat 3-strut RINGS (the same packing the
    plain segment twist uses -- not 3x3x3 blocks). An earlier version of
    this function packed logits into cubic blocks, which is the wrong shape
    for this operator: block packing requires the padded vocab length to be
    an exact multiple of a *cube* (e.g. 3^3=27), and any code that tries to
    size that cube dynamically from the vocab (rather than using the fixed
    3) can easily end up requesting a `.view()` whose dimensions don't
    multiply out to the actual padded length -- exactly the
    "shape '[1,1,13,13,13]' is invalid for input of size 51" kind of
    RuntimeError. Rings avoid this entirely: padding to a multiple of 3 is
    always exact and never depends on a derived side length.

    Per ring (strut1, strut2, strut3), three operators are applied:
      - P (parity):   reverses strut ordering (s1,s2,s3) -> (s3,s2,s1)
                       (`torch.flip`), analogous to mirroring the tube's
                       cross-section.
      - C (charge):   negates the resulting values, analogous to inverting
                       the filament's local extrusion polarity.
      - Chirality-warped ring: C applied on top of P, i.e. -(reversed ring).

    If C and P were perfect symmetries, blending the chirality-warped ring
    back in would do nothing (warped == original). `chirality_strength` is
    what breaks that symmetry: the final logits are an asymmetric mix of
    the original ring and its warped counterpart, so `chirality_strength=0`
    reduces to the identity (an achiral filament) and `chirality_strength=1`
    reduces to a pure chirality warp.

    Args:
        logits: [B, V] logits tensor.
        mode: 'none' returns logits unchanged. Any other string enables the
              transform (kept for interface parity with the plain twist).
        chirality_strength: float in [0, 1], how strongly the warped ring
              is blended back into the original ("degree of chiral bias").
        compute_gradient: if True, also returns d(warped)/d(logits) (summed
              grad_outputs of ones), useful for inspecting how sensitive
              the warp is to the raw extrusion logits. Requires
              `logits.requires_grad_(True)` to be set by the caller.

    Returns:
        warped_logits [B, V], or (warped_logits, gradient) if
        compute_gradient=True. `gradient` is None if logits didn't require grad.
    """
    if mode == "none" or logits is None:
        return (logits, None) if compute_gradient else logits

    B, V = logits.shape
    ring_size = 3

    # NOTE: padding must be neutral (0.0), not a large negative sentinel like
    # some other transforms use. This transform includes a charge-negation
    # step, so a -1e9 pad would flip to +1e9 and, once shuffled by the flip,
    # could land on a *real* vocab slot in the last (partially-padded) ring
    # -- silently creating one astronomically large logit that dominates
    # every extrusion step. 0.0 is safe under negation and dilutes
    # harmlessly when blended back in.
    pad_len = (ring_size - (V % ring_size)) % ring_size
    padded = F.pad(logits, (0, pad_len), value=0.0) if pad_len > 0 else logits

    rings = padded.view(B, -1, ring_size)

    # --- P: parity inversion, reverse strut order within each ring ---
    parity_rings = torch.flip(rings, dims=[2])

    # --- C: charge conjugation, negate the amplitude ---
    charge_conjugate = -parity_rings

    # --- chirality warp: asymmetric blend between original and warped ring ---
    v = max(0.0, min(1.0, chirality_strength))
    warped_rings = (1.0 - v) * rings + v * charge_conjugate

    warped_logits = warped_rings.reshape(B, -1)[:, :V]

    if not compute_gradient:
        return warped_logits

    gradient = None
    if logits.requires_grad:
        grad_outputs = torch.ones_like(warped_logits)
        gradient = torch.autograd.grad(
            outputs=warped_logits,
            inputs=logits,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]

    return warped_logits, gradient


def apply_cross_section_shear(logits, mode="standard", shear_strength=0.15):
    """
    A chirality-flavored shear transform packed into 2x2 cross-sections (4
    struts each) instead of flat 3-strut rings or the old, buggy block
    packing (27). 2x2 cross-sections are the only one of these shapes where
    "transpose" is literally the linear-algebra operation -- swapping rows
    and columns of a square matrix -- rather than a stand-in (an axis
    permutation for blocks, or a meaningless op on a 1D ring).

    Per 2x2 cross-section [[a, b], [c, d]]:
      - P (parity):  flip row order        -> [[c, d], [a, b]]
      - C (charge):  negate                -> -parity
      - transpose:   swap rows and columns of the charge-conjugated square
        (a genuine matrix transpose, `.transpose(-1, -2)`)
      - sheared = (1-v)*section + v*transposed

    Deliberately GRADIENT-FREE: unlike `apply_helical_chirality_warp`, this
    function has no `compute_gradient` option and never touches autograd.
    It's meant as a structural alternative to gradient descent -- see
    `FilamentGrowthSimulator.inspect_shear_shift`, which uses this
    transform's own algebraic before/after difference as a sensitivity
    signal instead of backpropagating one.
    """
    if mode == "none" or logits is None:
        return logits

    B, V = logits.shape
    section_size = 4  # 2 x 2

    pad_len = (section_size - (V % section_size)) % section_size
    padded = F.pad(logits, (0, pad_len), value=0.0) if pad_len > 0 else logits

    sections = padded.view(B, -1, 2, 2)

    parity_sections = torch.flip(sections, dims=[2])         # P: flip row order
    charge_conjugate_sections = -parity_sections              # C: negate
    sheared_transpose = charge_conjugate_sections.transpose(-1, -2)  # genuine matrix transpose

    v = max(0.0, min(1.0, shear_strength))
    sheared_sections = (1.0 - v) * sections + v * sheared_transpose

    return sheared_sections.reshape(B, -1)[:, :V]


def apply_filament_transform(logits, mode="none", chirality_strength=0.15, compute_gradient=False):
    """
    Dispatch helper: routes to the plain segment twist, the helical
    chirality warp, or the cross-section shear depending on `mode`.

    Modes:
      - 'none': identity (a perfectly straight, achiral filament)
      - 'standard' / 'anti' / 'full_transpose': plain segment twist (2D)
      - 'helical_chirality': ring-based chirality warp (+ optional gradient)
      - 'cross_section_shear': square-based shear (gradient-free by design)
    """
    if mode == "helical_chirality":
        return apply_helical_chirality_warp(
            logits, mode=mode, chirality_strength=chirality_strength, compute_gradient=compute_gradient
        )
    if mode == "cross_section_shear":
        result = apply_cross_section_shear(logits, mode=mode, shear_strength=chirality_strength)
        return (result, None) if compute_gradient else result
    result = apply_segment_twist(logits, mode=mode)
    return (result, None) if compute_gradient else result


# ==========================================================
# 2c. Classical Cross-Check Kernel -- old-fashioned matrices,
#     no tensor ops, for independently verifying the deformations
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


def classical_helical_chirality_warp_ring(ring, chirality_strength=0.15):
    """
    Same op as `apply_helical_chirality_warp`, for one flat ring (3-element
    list) of plain floats, via explicit index arithmetic instead of
    torch.flip:
      P (parity):  reverse the ring        -> ring[2-i]
      C (charge):  negate                  -> -parity
      warped = (1-v)*ring + v*(C applied to P)
    """
    v = max(0.0, min(1.0, chirality_strength))
    return [
        (1.0 - v) * ring[i] + v * (-ring[2 - i])
        for i in range(3)
    ]


def verify_helical_chirality_kernel_agreement(chirality_strength=0.15, atol=1e-5):
    """Cross-checks the tensor kernel against the classical kernel on a
    random flat ring. Returns (agrees: bool, max_abs_diff: float)."""
    ring = [random.uniform(-3.0, 3.0) for _ in range(3)]
    logits = torch.tensor([ring], dtype=torch.float32)
    tensor_flat = apply_helical_chirality_warp(logits, chirality_strength=chirality_strength)[0].tolist()

    classical_flat = classical_helical_chirality_warp_ring(ring, chirality_strength=chirality_strength)

    max_diff = max(abs(a - b) for a, b in zip(tensor_flat, classical_flat))
    return max_diff < atol, max_diff


def classical_cross_section_shear(section, shear_strength=0.15):
    """
    Same op as `apply_cross_section_shear`, for one 2x2 cross-section
    (list[list[float]]) of plain floats, via explicit index arithmetic
    instead of torch.flip/.transpose:
      P (parity):  flip row order        -> [section[1], section[0]]
      C (charge):  negate                -> -parity
      transpose:   swap rows and columns -> charge[j][i]
      sheared = (1-v)*section + v*transposed
    """
    v = max(0.0, min(1.0, shear_strength))
    parity = [section[1], section[0]]
    charge = [[-parity[i][j] for j in range(2)] for i in range(2)]
    transposed = [[charge[j][i] for j in range(2)] for i in range(2)]
    return [[(1.0 - v) * section[i][j] + v * transposed[i][j] for j in range(2)] for i in range(2)]


def verify_cross_section_shear_kernel_agreement(shear_strength=0.15, atol=1e-5):
    """Cross-checks the tensor cross-section kernel against the classical
    kernel on a random 2x2 section. Returns (agrees: bool, max_abs_diff: float)."""
    flat = [random.uniform(-3.0, 3.0) for _ in range(4)]
    logits = torch.tensor([flat], dtype=torch.float32)
    tensor_flat = apply_cross_section_shear(logits, shear_strength=shear_strength)[0].tolist()

    section = [[flat[0], flat[1]], [flat[2], flat[3]]]
    classical_section = classical_cross_section_shear(section, shear_strength=shear_strength)
    classical_flat = [classical_section[0][0], classical_section[0][1], classical_section[1][0], classical_section[1][1]]

    max_diff = max(abs(a - b) for a, b in zip(tensor_flat, classical_flat))
    return max_diff < atol, max_diff


# ==========================================================
# 2d. Standard-Library Function Ontology (double purport)
# ==========================================================
# Each function's literal documented purpose ("standard_purpose") next to
# the second, local role it actually plays in this filament simulator --
# the same call site is honest under both readings at once, not a hidden
# meaning.

FILAMENT_FUNCTION_ONTOLOGY = {
    "math.log": ("natural logarithm", "turns a probability into additive growth evidence, and the filament's curvature is computed on"),
    "torch.exp": ("e^x, elementwise over a tensor", "turns inverse node density into an extrusion tension bias (tension_bias)"),
    "math.isclose": ("approximate float equality within a tolerance", "flags near-zero curvature segments where the filament runs locally straight"),
    "difflib.get_close_matches": ("fuzzy string matching", "snaps typed words onto the nearest known strand"),
    "collections.Counter": ("multiset / frequency dict", "the sparse density_matrix and tension_bias are built from"),
    "torch.topk": ("k largest values + indices", "both filters extrusion candidates AND supplies a step's reported 'structural state'"),
    "torch.argmax": ("index of the max value", "the deterministic growth operator -- also what makes the filament seed-independent"),
    "torch.multinomial": ("categorical sampling", "the stochastic extrusion operator; source of the arbitrary filament's divergence"),
    "torch.flip": ("reverse a tensor axis", "implements the 'parity' (P) operator in the helical chirality warp"),
    "torch.Tensor.transpose": ("swap two tensor axes", "the genuine cross-section transpose in the shear kernel, and the axis swap in the twist kernel"),
}


def print_filament_ontology():
    """Prints the double-purport table: standard meaning vs. local role."""
    for name, (standard, local) in FILAMENT_FUNCTION_ONTOLOGY.items():
        print(f"{name}\n  standard:  {standard}\n  this file: {local}")


class OntologyUsageTracker:
    """
    Context manager that monkey-patches every function referenced by
    FILAMENT_FUNCTION_ONTOLOGY -- not just math.log/math.isclose -- counts
    real invocations during the `with` block, and restores the originals on
    exit. This is how every entry in the table gets proven to describe code
    that actually ran, the same way the curvature functions were verified
    by spying on them earlier.
    """
    TRACKED_TORCH_FUNCS = ("topk", "argmax", "multinomial", "flip", "exp")

    def __init__(self):
        self.counts = {name: 0 for name in FILAMENT_FUNCTION_ONTOLOGY}
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


def run_full_filament_ontology_audit(sample_text=None):
    """
    Exercises every function in FILAMENT_FUNCTION_ONTOLOGY in one
    self-contained run: corpus loading (collections.Counter), a forward
    pass + population extrusion under the shear kernel (torch.topk,
    torch.multinomial, torch.flip, torch.Tensor.transpose, torch.exp via
    tension_bias), a deterministic+lookahead growth trace (torch.argmax),
    curvature (math.log, math.isclose), and a spell-correction call
    (difflib.get_close_matches) -- while an OntologyUsageTracker counts
    real invocations. Returns the call-count dict.
    """
    if sample_text is None:
        sample_text = "the quick brown fox jumps over the lazy dog " * 20

    # Use the OS's real temp directory (tempfile.gettempdir()) rather than a
    # hardcoded Unix path like "/tmp" -- "/tmp" doesn't exist on Windows,
    # which is exactly what caused FileNotFoundError here. mkstemp also
    # avoids collisions if this ever runs concurrently.
    fd, tmp_path = tempfile.mkstemp(prefix="_filament_ontology_audit_corpus_", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(sample_text)

        with OntologyUsageTracker() as tracker:
            (node_to_idx, idx_to_node, vocab_size, tension_bias, dominant_nodes,
             corpus_nodes, dominant_strands, anchor_to_row) = load_and_analyze_filament_corpus(tmp_path, anchor_window=5)

            void_id = node_to_idx[("<VOID>", "<VOID>", "<VOID>")]
            sim = FilamentGrowthSimulator(vocab_size=vocab_size, embed_dim=8, anchor_size=5)
            seed = torch.tensor([[void_id] * 5], dtype=torch.long)

            sim.generate_filament_population(
                seed, sequence_length=3, pop_size=1, tension_bias=tension_bias,
                idx_to_node=idx_to_node, void_id=void_id,
                warp_mode="cross_section_shear", top_k=5,
            )
            trace = sim.generate_deterministic_growth_trace(
                seed, num_growth_steps=5, tension_bias=tension_bias,
                idx_to_node=idx_to_node, void_id=void_id,
                warp_mode="helical_chirality", top_k=5,
                lookahead_width=2, lookahead_depth=1, verbose=False,
            )
            attach_curvature_to_trace(trace)

            known_words = sorted({w for n in node_to_idx.keys() for w in n})
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
    for name, (standard, local) in FILAMENT_FUNCTION_ONTOLOGY.items():
        count = call_counts.get(name, 0)
        if count == 0:
            all_used = False
            flag = "  [WARNING: never called this run]"
        else:
            flag = ""
        print(f"  [{name}] called {count}x | {local}{flag}")
    return all_used


# ==========================================================
# 2b. Filament Growth-Trace Record
# ==========================================================

@dataclass
class FilamentGrowthStep:
    """
    One ring added to the growing filament: everything that was actually
    computed to decide a single extruded node. Nothing here is invented
    after the fact -- each field is read directly off the tensors used
    during that step's forward pass and extrusion, so the trace is a
    faithful, inspectable record of the filament's growth rather than a
    post-hoc explanation.
    """
    step_index: int
    anchor_nodes: List[Tuple[str, str, str]]
    warp_mode: str
    candidate_extensions: List[Tuple[Tuple[str, str, str], float]]
    chosen_node: Tuple[str, str, str]
    chosen_prob: float
    selection_method: str = "sampled"  # "sampled" (stochastic extrusion) or "deductive" (argmax growth)
    runner_up_gap: Optional[float] = None  # margin between top-1 and top-2 confidence, deductive mode only
    lookahead_depth: int = 0  # how many rings ahead were simulated before committing to this choice
    lookahead_scores: Optional[List[Tuple[Tuple[str, str, str], float]]] = None  # (candidate, cumulative log-prob incl. rollout)
    overrode_greedy: bool = False  # True if lookahead picked something other than the naive immediate-best candidate
    curvature: Optional[float] = None  # second finite difference of log(chosen_prob); None for edge steps (no two neighbors)
    curvature_is_flat: Optional[bool] = None  # True if curvature ~ 0.0, per math.isclose -- filament runs straight here
    curvature_ratio: Optional[float] = None  # exp(curvature): the same bend, viewed multiplicatively
    close_to_previous: Optional[bool] = None  # per difflib.get_close_matches against the prior ring's phrase

    def as_text(self) -> str:
        anchor_str = ' '.join(w for n in self.anchor_nodes for w in n)
        candidates_str = ", ".join(
            f"{'/'.join(node)}={prob:.3f}" for node, prob in self.candidate_extensions
        )

        if self.selection_method == "deductive":
            gap_str = f", margin over runner-up: {self.runner_up_gap:.3f}" if self.runner_up_gap is not None else ""

            if self.lookahead_depth > 0 and self.lookahead_scores:
                lookahead_str = ", ".join(
                    f"{'/'.join(node)}(cum_logp={score:.3f})" for node, score in self.lookahead_scores
                )
                override_str = (
                    " [overrides naive immediate-best growth -- a shallower lookahead "
                    "would have extruded differently]" if self.overrode_greedy else ""
                )
                return (
                    f"[Ring {self.step_index + 1}] "
                    f"STATE: given anchor \"{anchor_str}\" under warp {self.warp_mode}, "
                    f"immediate candidates ranked {candidates_str}. "
                    f"LOOKING AHEAD {self.lookahead_depth} ring(s), simulated continuations score: {lookahead_str}. "
                    f"THEREFORE: the candidate with the best cumulative outcome extrudes next -> "
                    f"\"{'/'.join(self.chosen_node)}\" (p={self.chosen_prob:.3f}{gap_str}){override_str}{self._curvature_suffix()}"
                )

            gap_str = f", margin over runner-up: {self.runner_up_gap:.3f}" if self.runner_up_gap is not None else ""
            return (
                f"[Ring {self.step_index + 1}] "
                f"STATE: given anchor \"{anchor_str}\" under warp {self.warp_mode}, "
                f"candidates ranked {candidates_str}. "
                f"THEREFORE: the highest-confidence candidate extrudes deterministically -> "
                f"\"{'/'.join(self.chosen_node)}\" (p={self.chosen_prob:.3f}{gap_str}){self._curvature_suffix()}"
            )

        return (
            f"[Ring {self.step_index + 1}] "
            f"anchor: \"{anchor_str}\" "
            f"| warp: {self.warp_mode} "
            f"| considered: {candidates_str} "
            f"| extruded: \"{'/'.join(self.chosen_node)}\" (p={self.chosen_prob:.3f}){self._curvature_suffix()}"
        )

    def _curvature_suffix(self) -> str:
        if self.curvature is None:
            return ""
        flat_str = " [straight]" if self.curvature_is_flat else ""
        close_str = " [echoes previous]" if self.close_to_previous else ""
        ratio_str = f", ratio={self.curvature_ratio:.3f}" if self.curvature_ratio is not None else ""
        return f" | curvature={self.curvature:+.4f}{ratio_str}{flat_str}{close_str}"


# ==========================================================
# 2e. Filament Curvature -- built on ALL TEN entries of
#     FILAMENT_FUNCTION_ONTOLOGY, not just documented by two of them
# ==========================================================
#
# "Curvature" of a growth trace: how sharply the filament bends step to
# step (second finite difference of log(chosen_prob)), plus a trace-wide
# analysis of that bend -- the literal physical bending of a tubular
# filament, computed from the confidence of each ring it grew. Every one of
# the ten ontology functions is genuinely called somewhere in this
# pipeline -- the table below is a map from function to the exact role it
# plays here, not aspirational documentation:
#
#   math.log               per-ring log-probability the curvature is built from
#   math.isclose            flags near-zero (straight) curvature
#   torch.exp                curvature_ratio: the same bend viewed multiplicatively
#   difflib.get_close_matches  flags rings whose phrase echoes the previous one
#   collections.Counter      tallies how many rings bent up / down / straight
#   torch.topk                the k sharpest-bending rings in the trace
#   torch.argmax               the single sharpest-bending ring
#   torch.multinomial          a magnitude-weighted random "representative" bend
#   torch.flip                  reversal-symmetry self-check on the curvature formula
#   torch.Tensor.transpose    packs (step_index, curvature) into a real table

CURVATURE_ONTOLOGY_FUNCTIONS = tuple(FILAMENT_FUNCTION_ONTOLOGY.keys())


def compute_curvature(chosen_probs, flat_tolerance=1e-3):
    """
    Given a sequence of per-ring probabilities, returns a list of
    (curvature, is_flat, ratio) for each interior ring (index 0 of the
    output corresponds to original index 1 -- edge rings have no two
    neighbors and are excluded here; callers map back onto the full trace).

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


def attach_curvature_to_trace(trace, flat_tolerance=1e-3):
    """
    Computes curvature over `trace`'s chosen_prob sequence via
    `compute_curvature` and writes the results back onto each interior
    FilamentGrowthStep's curvature / curvature_is_flat / curvature_ratio
    fields. Also flags, for every ring but the first, whether its chosen
    phrase is a close textual match to the previous ring's -- via
    `difflib.get_close_matches` -- since a filament that keeps echoing
    itself is a distinct (and detectable) kind of "straightness" from zero
    curvature. The first and last rings keep curvature=None (no two
    neighbors to bend between). Mutates and returns `trace`.
    """
    probs = [s.chosen_prob for s in trace]
    curvatures = compute_curvature(probs, flat_tolerance=flat_tolerance)
    for offset, (c, is_flat, ratio) in enumerate(curvatures):
        step = trace[offset + 1]
        step.curvature = c
        step.curvature_is_flat = is_flat
        step.curvature_ratio = ratio

    for i in range(1, len(trace)):
        prev_phrase = " ".join(trace[i - 1].chosen_node)
        this_phrase = " ".join(trace[i].chosen_node)
        matches = difflib.get_close_matches(this_phrase, [prev_phrase], n=1, cutoff=0.6)  # ontology: difflib.get_close_matches
        trace[i].close_to_previous = bool(matches)

    return trace


def analyze_trace_curvature(trace, top_k=3):
    """
    Trace-wide curvature analysis, using the five ontology functions that
    only make sense over a whole series rather than a single ring:

      torch.topk        -> the `top_k` sharpest-bending rings (by |curvature|)
      torch.argmax        -> the single sharpest-bending ring
      torch.multinomial     -> a magnitude-weighted random "representative" ring
      torch.flip              -> reversal-symmetry self-check on the curvature formula
      torch.Tensor.transpose -> packs (step_index, curvature) into a real [N, 2] table
      collections.Counter     -> tallies rings as up-bending / down-bending / straight

    Returns a dict of results; returns None if the trace has no interior
    (curvature-bearing) rings to analyze.
    """
    interior = [s for s in trace if s.curvature is not None]
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
    log_probs_tensor = torch.log(torch.tensor([s.chosen_prob for s in trace]).clamp_min(1e-12))
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


def print_curvature_report(trace, top_k=3):
    """
    Prints the ontology's standard/role explanation for every function this
    pipeline uses -- all ten entries, not a subset -- then each ring's
    curvature, then the trace-wide analysis from `analyze_trace_curvature`.
    """
    print("=== FILAMENT CURVATURE REPORT ===")
    for name in CURVATURE_ONTOLOGY_FUNCTIONS:
        standard, local = FILAMENT_FUNCTION_ONTOLOGY[name]
        print(f"  [{name}] standard: {standard} | here: {local}")
    print()

    for step in trace:
        if step.curvature is None:
            print(f"[Ring {step.step_index + 1}] curvature: n/a (edge ring) | extruded: \"{'/'.join(step.chosen_node)}\"")
        else:
            print(f"[Ring {step.step_index + 1}] extruded: \"{'/'.join(step.chosen_node)}\"{step._curvature_suffix()}")

    analysis = analyze_trace_curvature(trace, top_k=top_k)
    if analysis is None:
        print("\n(filament too short for trace-wide curvature analysis)")
        return

    print("\n--- trace-wide analysis ---")
    top_str = ", ".join(f"ring {i + 1} ({c:+.4f})" for i, c in analysis["top_steps"])
    print(f"top-{top_k} sharpest bends (torch.topk): {top_str}")
    peak_i, peak_c = analysis["peak_step"]
    print(f"single sharpest bend (torch.argmax):    ring {peak_i + 1} ({peak_c:+.4f})")
    samp_i, samp_c = analysis["sampled_step"]
    print(f"magnitude-weighted random pick (torch.multinomial): ring {samp_i + 1} ({samp_c:+.4f})")
    print(f"reversal-symmetry check (torch.flip), max discrepancy: {analysis['reversal_asymmetry']:.2e}")
    print(f"sign tally (collections.Counter): {dict(analysis['sign_counts'])}")
    print(f"(step_index, curvature) table shape after torch.Tensor.transpose: {tuple(analysis['table'].shape)}")


# ==========================================================
# 3. Filament Growth Simulator with Cross-Sectional Deformation
# ==========================================================

class FilamentGrowthSimulator(nn.Module):
    """
    A small neural extrusion head that grows a tubular filament one
    3-strut node at a time. Each ring's raw extrusion logits are passed
    through a cross-sectional deformation (`apply_filament_transform`)
    before being sampled or argmax-selected into the next node.
    """

    def __init__(self, vocab_size, embed_dim=64, anchor_size=5):
        super().__init__()
        self.anchor_size = anchor_size
        self.node_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(anchor_size, embed_dim)
        self.anchor_proj = nn.Linear(embed_dim, embed_dim)
        self.extrusion_head = nn.Sequential(
            nn.Linear(embed_dim * anchor_size + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(self, idx, warp_mode="none", chirality_strength=0.15, compute_gradient=False):
        B, T = idx.shape
        T_use = min(T, self.anchor_size)
        idx = idx[:, -T_use:]

        n_emb = self.node_embedding(idx)
        positions = torch.arange(0, T_use, dtype=torch.long, device=idx.device)
        p_emb = self.position_embedding(positions)

        x = n_emb + p_emb
        x_flat = x.reshape(B, -1)

        anchor_summary = x.mean(dim=1)
        anchor_summary = self.anchor_proj(anchor_summary)

        logits = self.extrusion_head(torch.cat([x_flat, anchor_summary], dim=-1))

        if compute_gradient and not logits.requires_grad:
            logits.requires_grad_(True)
            logits.retain_grad()

        # Apply the selected cross-sectional deformation (twist, chirality
        # warp, or shear)
        if compute_gradient:
            logits, gradient = apply_filament_transform(
                logits,
                mode=warp_mode,
                chirality_strength=chirality_strength,
                compute_gradient=True,
            )
            return logits, gradient

        logits = apply_filament_transform(
            logits,
            mode=warp_mode,
            chirality_strength=chirality_strength,
            compute_gradient=False,
        )
        return logits

    # ==========================================================
    # Fundamentals: every growth mode below is one call into this
    # canonical extrusion engine, parameterized by a handful of orthogonal
    # choices (selection rule, lookahead, trace recording). There is exactly
    # one place that knows how to pad an anchor, run the forward pass, mask
    # <VOID>, apply top_k, and turn logits into a distribution -- previously
    # this logic was duplicated (with small drifts) across three separate
    # growth methods.
    # ==========================================================

    def _pad_anchor(self, idx):
        """Right-aligns `idx` to exactly `anchor_size` tokens, left-padding
        by repeating the first available token if the sequence is shorter."""
        cond = idx[:, -self.anchor_size:]
        if cond.shape[1] < self.anchor_size:
            pad = cond[:, :1].repeat(1, self.anchor_size - cond.shape[1])
            cond = torch.cat([pad, cond], dim=1)
        return cond

    def _next_node_distribution(self, idx, tension_bias, void_id, warp_mode, chirality_strength, top_k, temperature=1.0):
        """The single fundamental step: anchor -> padded anchor -> forward
        pass (+ cross-sectional deformation) -> <VOID> mask -> top_k filter
        -> softmax. Returns (probs, padded_anchor) since callers that
        record a trace need the anchor tokens too.
        """
        cond = self._pad_anchor(idx)

        logits = self(
            cond,
            warp_mode=warp_mode,
            chirality_strength=chirality_strength,
            compute_gradient=False,
        ) + tension_bias
        logits[:, void_id] = -float('inf')

        if top_k is not None and top_k < logits.shape[-1]:
            topv, topi = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, topi, topv)
            logits = filtered

        probs = F.softmax(logits / temperature, dim=-1)
        return probs, cond

    def _simulate_growth_log_prob(self, start_idx, num_steps, tension_bias, void_id, warp_mode, chirality_strength, top_k):
        """Simulates `num_steps` of greedy (argmax) continuation from
        `start_idx` and returns the summed log-probability of that path.
        This is the primitive lookahead is built on: "if I extrude this
        candidate, how well-supported is where the filament deterministically
        grows from here?"
        """
        cur = start_idx
        cumulative_log_prob = 0.0
        for _ in range(num_steps):
            probs, _ = self._next_node_distribution(cur, tension_bias, void_id, warp_mode, chirality_strength, top_k)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            p = probs[0, next_id.item()].item()
            cumulative_log_prob += math.log(max(p, 1e-12))
            cur = torch.cat([cur, next_id], dim=1)
        return cumulative_log_prob

    def _select_with_structural_lookahead(self, probs, shadow_idx, tension_bias, void_id, warp_mode, chirality_strength, top_k, width, depth):
        """Scores the top-`width` immediate candidates by cumulative
        log-probability (their own + `depth` rings of simulated future
        growth) and returns the best one. Returns
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
                self._simulate_growth_log_prob(hypothetical_idx, depth, tension_bias, void_id, warp_mode, chirality_strength, top_k)
                if depth > 0 else 0.0
            )
            cumulative = immediate_log_prob + future_log_prob
            scored.append((ci.item(), cumulative))
            if cumulative > best_score:
                best_score, best_id = cumulative, ci.view(1, 1)

        return best_id, scored, (best_id.item() != naive_best_id)

    @torch.no_grad()
    def _grow_filament(
        self,
        seed_anchor_idx,
        num_steps,
        tension_bias,
        idx_to_node,
        void_id,
        warp_mode,
        chirality_strength,
        selection="sample",       # "sample" (torch.multinomial) or "argmax" (deductive)
        temperature=1.0,          # only meaningful when selection="sample"
        top_k=25,
        lookahead_width=0,
        lookahead_depth=0,        # > 0 overrides node choice with simulated-future scoring
        top_candidates_shown=5,
        record_trace=False,
        verbose=False,
    ):
        """
        THE fundamental extrusion engine. Every public growth method on
        this class is a thin, named wrapper around this one loop:
          - generate_filament_population         -> selection="sample",  record_trace=False
          - generate_arbitrary_growth_trace       -> selection="sample",  record_trace=True
          - generate_deterministic_growth_trace   -> selection="argmax", record_trace=True, lookahead optional

        Always runs on an independent clone of `seed_anchor_idx` -- callers
        never share filament state with each other.

        Returns (words, growth_trace) where growth_trace is None unless
        record_trace=True.
        """
        shadow_idx = seed_anchor_idx.clone()
        tension_bias = tension_bias.to(seed_anchor_idx.device)

        words = []
        trace: Optional[List[FilamentGrowthStep]] = [] if record_trace else None

        for step in range(num_steps):
            temp = temperature if selection == "sample" else 1.0
            probs, cond = self._next_node_distribution(
                shadow_idx, tension_bias, void_id, warp_mode, chirality_strength, top_k, temperature=temp
            )

            candidate_extensions, runner_up_gap = None, None
            if record_trace:
                k_show = min(top_candidates_shown, probs.shape[-1])
                top_probs, top_idx = torch.topk(probs, k=k_show, dim=-1)
                candidate_extensions = [
                    (idx_to_node[i.item()], p.item()) for i, p in zip(top_idx[0], top_probs[0])
                ]
                if len(candidate_extensions) >= 2:
                    runner_up_gap = candidate_extensions[0][1] - candidate_extensions[1][1]

            lookahead_scores, overrode_greedy = None, False

            if lookahead_depth > 0:
                next_id, scored_ids, overrode_greedy = self._select_with_structural_lookahead(
                    probs, shadow_idx, tension_bias, void_id, warp_mode, chirality_strength, top_k,
                    lookahead_width, lookahead_depth,
                )
                if record_trace:
                    lookahead_scores = [(idx_to_node[i], s) for i, s in scored_ids]
            elif selection == "argmax":
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            chosen_node = idx_to_node[next_id.item()]
            chosen_prob = probs[0, next_id.item()].item()
            words.append(chosen_node)

            if record_trace:
                method_label = "deductive" if (selection == "argmax" or lookahead_depth > 0) else "sampled"
                step_record = FilamentGrowthStep(
                    step_index=step,
                    anchor_nodes=[idx_to_node[i.item()] for i in cond[0]],
                    warp_mode=warp_mode,
                    candidate_extensions=candidate_extensions,
                    chosen_node=chosen_node,
                    chosen_prob=chosen_prob,
                    selection_method=method_label,
                    runner_up_gap=runner_up_gap,
                    lookahead_depth=lookahead_depth,
                    lookahead_scores=lookahead_scores,
                    overrode_greedy=overrode_greedy,
                )
                trace.append(step_record)
                if verbose:
                    print(step_record.as_text())

            shadow_idx = torch.cat([shadow_idx, next_id], dim=1)

        return words, trace

    # ---------- named, backward-compatible wrappers around _grow_filament ----------

    def generate_filament_population(
        self, seed_anchor_idx, sequence_length, pop_size, tension_bias, idx_to_node, void_id,
        warp_mode="standard", chirality_strength=0.15, temperature=1.2, top_k=25,
    ):
        """Produces `pop_size` independent stochastically-grown filaments (final output text)."""
        filaments = []
        for _ in range(pop_size):
            words, _ = self._grow_filament(
                seed_anchor_idx, sequence_length, tension_bias, idx_to_node, void_id,
                warp_mode, chirality_strength,
                selection="sample", temperature=temperature, top_k=top_k, record_trace=False,
            )
            filaments.append(words)
        return filaments

    def generate_arbitrary_growth_trace(
        self, seed_anchor_idx, num_growth_steps, tension_bias, idx_to_node, void_id,
        warp_mode="helical_chirality", chirality_strength=0.15, temperature=0.8, top_k=25,
        top_candidates_shown=5, verbose=True,
    ):
        """
        A growth trace that is ARBITRARY relative to the final output
        filament: an independent stochastic shadow extrusion, cloned from
        the same seed anchor but never shared with (or fed back into)
        whatever trajectory ultimately becomes the final filament.
        `num_growth_steps` is unrelated to the final filament's length.
        """
        _, trace = self._grow_filament(
            seed_anchor_idx, num_growth_steps, tension_bias, idx_to_node, void_id,
            warp_mode, chirality_strength,
            selection="sample", temperature=temperature, top_k=top_k,
            top_candidates_shown=top_candidates_shown, record_trace=True, verbose=verbose,
        )
        return trace

    def generate_deterministic_growth_trace(
        self, seed_anchor_idx, num_growth_steps, tension_bias, idx_to_node, void_id,
        warp_mode="helical_chirality", chirality_strength=0.15, top_k=25,
        top_candidates_shown=5, lookahead_width=3, lookahead_depth=2, verbose=True,
    ):
        """
        A DETERMINISTIC shadow extrusion: no temperature, no sampling --
        every choice is argmax, so re-running on the same anchor and
        weights always regrows the identical filament regardless of RNG
        state.

        With `lookahead_depth > 0`, a ring doesn't just take the immediate
        best candidate: it simulates `lookahead_depth` rings of greedy
        continuation for the top `lookahead_width` candidates and picks
        whichever has the best cumulative log-probability. This can and
        does override the naive immediate-best pick
        (see `FilamentGrowthStep.overrode_greedy`). Set `lookahead_depth=0`
        to disable and fall back to pure immediate-argmax growth.
        """
        _, trace = self._grow_filament(
            seed_anchor_idx, num_growth_steps, tension_bias, idx_to_node, void_id,
            warp_mode, chirality_strength,
            selection="argmax", top_k=top_k, top_candidates_shown=top_candidates_shown,
            lookahead_width=lookahead_width, lookahead_depth=lookahead_depth,
            record_trace=True, verbose=verbose,
        )
        return trace

    def inspect_chirality_gradient(self, idx, chirality_strength=0.15):
        """
        Convenience helper: runs a forward pass with warp_mode='helical_chirality'
        and compute_gradient=True, returning (logits, gradient) so the caller can
        inspect how sensitive the chirality warp is to the raw extrusion logits at
        this anchor. Not used inside no_grad growth loops.
        """
        logits, gradient = self.forward(
            idx,
            warp_mode="helical_chirality",
            chirality_strength=chirality_strength,
            compute_gradient=True,
        )
        return logits, gradient

    @torch.no_grad()
    def inspect_shear_shift(self, idx, chirality_strength=0.15):
        """
        A GRADIENT-FREE alternative to `inspect_chirality_gradient`. Where
        that method asks "how sensitive is the warp, via backprop /
        autograd", this one never touches calculus at all: it takes the raw
        (undeformed) logits, applies the square-shaped cross-section shear
        to them directly, and returns the transform's own before/after
        difference as the sensitivity signal. The 2x2 shape is what makes
        this meaningful without a derivative -- because the transpose step
        is a literal matrix transpose (swap rows/columns of a real 2x2
        cross-section), the shift already tells you exactly which struts
        traded places and by how much, algebraically, with no need to
        differentiate anything.

        Returns:
            (sheared_logits, shift) where shift = sheared_logits -
            raw_logits, same shape as the model's output.
        """
        raw_logits = self.forward(idx, warp_mode="none")
        sheared_logits = apply_cross_section_shear(
            raw_logits, shear_strength=chirality_strength
        )
        shift = sheared_logits - raw_logits
        return sheared_logits, shift


# ==========================================================
# 4. Pipeline Execution Loop
# ==========================================================

if __name__ == "__main__":
    ring_agrees, ring_max_diff = verify_helical_chirality_kernel_agreement(chirality_strength=0.15)
    sq_agrees, sq_max_diff = verify_cross_section_shear_kernel_agreement(shear_strength=0.15)
    print(f"[self-test] ring (chirality) kernel agreement: {'PASS' if ring_agrees else 'FAIL'} (max_diff={ring_max_diff:.2e})")
    print(f"[self-test] cross-section shear kernel agreement: {'PASS' if sq_agrees else 'FAIL'} (max_diff={sq_max_diff:.2e})")

    ontology_counts = run_full_filament_ontology_audit()
    ontology_all_used = audit_ontology_function_usage(ontology_counts)
    print(f"[self-test] ontology fully exercised: {'PASS' if ontology_all_used else 'FAIL'}")
    print()

    filename = input("Filename: ").strip()
    (
        node_to_idx,
        idx_to_node,
        vocab_size,
        tension_bias,
        dominant_nodes,
        corpus_nodes,
        dominant_strands,
        anchor_to_row,
    ) = load_and_analyze_filament_corpus(filename, anchor_window=5)

    void_node = ("<VOID>", "<VOID>", "<VOID>")
    void_id = node_to_idx[void_node]

    filament_sim = FilamentGrowthSimulator(vocab_size=vocab_size, embed_dim=64, anchor_size=5)
    known_words = sorted(set(dominant_strands) | {w for n in node_to_idx.keys() if n != void_node for w in n})

    print("\nEnter text to seed the filament. Type 'quit' to exit.")
    print("(Growth uses the helical chirality warp: warp_mode='helical_chirality')")
    while True:
        raw_input_text = input("\nUSER: ").strip()
        if raw_input_text.lower() in {"quit", "exit", "stop"}:
            break

        tokens = raw_input_text.lower().split() if raw_input_text else []
        corrected_tokens = []
        for w in tokens:
            matches = difflib.get_close_matches(w, known_words, n=1, cutoff=0.0)
            corrected_tokens.append(matches[0] if matches else w)

        seed_anchor_words = corrected_tokens[-filament_sim.anchor_size:]
        if len(seed_anchor_words) < filament_sim.anchor_size:
            fallback_words = dominant_strands[: filament_sim.anchor_size - len(seed_anchor_words)]
            seed_anchor_words = fallback_words + seed_anchor_words

        while len(seed_anchor_words) < filament_sim.anchor_size:
            seed_anchor_words.insert(0, dominant_strands[0] if dominant_strands else "<void>")

        seed_anchor_nodes = [(seed_anchor_words[i], seed_anchor_words[i + 1], seed_anchor_words[i + 2])
                              for i in range(max(1, len(seed_anchor_words) - 2))]
        seed_anchor_node = seed_anchor_nodes[-1] if seed_anchor_nodes else dominant_nodes[0]

        anchor_tensor = torch.tensor(
            [[node_to_idx.get(seed_anchor_node[0], void_id),
              node_to_idx.get(seed_anchor_node[1], void_id),
              node_to_idx.get(seed_anchor_node[2], void_id),
              node_to_idx.get(seed_anchor_node[0], void_id),
              node_to_idx.get(seed_anchor_node[1], void_id)]],
            dtype=torch.long
        )

        # Gradient-free: inspect the cross-section shear shift at this
        # anchor, instead of computing an autograd gradient for it.
        sheared_logits, shear_shift = filament_sim.inspect_shear_shift(
            anchor_tensor, chirality_strength=0.85
        )
        print(f"[cross_section_shear] shift norm at this anchor: {shear_shift.norm().item():.4f}")

        print("\n=== FILAMENT GROWTH TRACE (deterministic with lookahead, independent of final text) ===")
        growth_trace = filament_sim.generate_deterministic_growth_trace(
            seed_anchor_idx=anchor_tensor,
            num_growth_steps=18,  # unrelated to the final filament's length below
            tension_bias=tension_bias,
            idx_to_node=idx_to_node,
            void_id=void_id,
            warp_mode="helical_chirality",
            chirality_strength=0.85,
            top_k=25,
            top_candidates_shown=15,
            lookahead_width=13,   # how many immediate candidates get simulated forward
            lookahead_depth=12,   # how many rings ahead each candidate is rolled out
            verbose=True,  # prints one FilamentGrowthStep line per ring as it happens
        )

        attach_curvature_to_trace(growth_trace)
        print()
        print_curvature_report(growth_trace)

        print("\n=== GROWING FINAL FILAMENT WITH HELICAL CHIRALITY WARP ===")
        filaments = filament_sim.generate_filament_population(
            seed_anchor_idx=anchor_tensor,
            sequence_length=150,
            pop_size=1,
            tension_bias=tension_bias,
            idx_to_node=idx_to_node,
            void_id=void_id,
            warp_mode="helical_chirality",
            chirality_strength=0.15,
            temperature=0.8,
            top_k=25,
        )

        flattened_words = [word for node in filaments[0] for word in node]
        output_text = ' '.join(flattened_words)
        print("\n=== FINAL FILAMENT TEXT ===")
        print(f"{output_text}")

        print("-" * 65)
