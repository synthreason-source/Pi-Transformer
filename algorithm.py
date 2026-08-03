import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import difflib
from collections import defaultdict, Counter

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

def apply_triangular_logit_transpose(logits):
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
    Extends the triangular transpose into a full 3D "CP-violation" style transform.

    Logits are packed into 3x3x3 cubes (27 values per cube) instead of flat
    triangles of 3. Three physics-flavored operators are then applied:

      - P (parity):   reverses vertex ordering along one cube axis
                       (`torch.flip`), analogous to a spatial mirror.
      - C (charge):   negates the resulting values, analogous to swapping
                       particle <-> antiparticle amplitude sign.
      - CP transpose: a genuine 3D transpose (axis permutation) is applied
                       on top of the C and P operators, giving the full
                       "CP-transformed" cube.

    If C and P were perfect symmetries, blending the CP-transformed cube
    back in would do nothing (transformed == original). The
    `violation_strength` term is what breaks that symmetry: the final
    logits are an asymmetric mix of the original cube and its CP-transformed
    counterpart, so `violation_strength=0` reduces to the identity and
    `violation_strength=1` reduces to a pure CP transform.

    Args:
        logits: [B, V] logits tensor.
        mode: 'none' returns logits unchanged. Any other string enables the
              transform (kept for interface parity with the triangular version).
        violation_strength: float in [0, 1], how strongly the CP-transformed
              cube is blended back into the original ("degree of violation").
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
    cube_size = 5  # 3 x 3 x 3

    # NOTE: padding must be neutral (0.0), not a large negative sentinel like
    # the flat triangular transpose uses. This transform includes a charge
    # conjugation (negation) step, so a -1e9 pad would flip to +1e9 and, once
    # shuffled by the 3D axis permutation, could land on a *real* vocab slot
    # in the last (partially-padded) cube -- silently creating one
    # astronomically large logit that dominates every sampling step. 0.0 is
    # safe under negation and dilutes harmlessly when blended back in.
    pad_len = (cube_size - (V % cube_size)) % cube_size
    if pad_len > 0:
        padded = F.pad(logits, (0, pad_len), value=0.0)
    else:
        padded = logits

    n_cubes = padded.shape[1] // cube_size
    cubes = padded.view(B, n_cubes, 3, 3, 3)

    # --- P: parity inversion, reverse ordering along the first spatial axis ---
    parity_cubes = torch.flip(cubes, dims=[2])

    # --- C: charge conjugation, negate the amplitude ---
    charge_conjugate_cubes = -parity_cubes

    # --- Full 3D transpose: permute the three spatial axes of the cube ---
    cp_transposed = charge_conjugate_cubes.permute(0, 1, 4, 3, 2).contiguous()

    # --- CP violation: asymmetric blend between original and CP-transformed cube ---
    v = max(0.0, min(1.0, violation_strength))
    violated_cubes = (1.0 - v) * cubes + v * cp_transposed

    transformed_logits = violated_cubes.reshape(B, -1)[:, :V]

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


def apply_logit_transform(logits, mode="none", violation_strength=0.15, compute_gradient=False):
    """
    Dispatch helper: routes to the flat triangular transpose or the new 3D
    CP-violation transpose depending on `mode`.

    Modes:
      - 'none': identity
      - 'standard' / 'anti' / 'full_transpose': flat triangular transpose (2D)
      - 'cp_violation_3d': the new 3D CP-violation transpose (+ optional gradient)
    """
    return apply_cp_violation_transpose_3d(
        logits, mode=mode, violation_strength=violation_strength, compute_gradient=compute_gradient
    )
    result = apply_triangular_logit_transpose(logits, mode=mode)
    return (result, None) if compute_gradient else result


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

    @torch.no_grad()
    def generate_population_seeds(
        self,
        seed_context_idx,
        sequence_length,
        pop_size,
        amp_boost,
        idx_to_word,
        unk_id,
        violation_strength=0.15,
        temperature=1.2,
        top_k=25,
    ):
        seeds = []
        amp_boost = amp_boost.to(seed_context_idx.device)

        for _ in range(pop_size):
            curr_idx = seed_context_idx.clone()
            candidate_sequence = []

            for _ in range(sequence_length):
                cond = curr_idx[:, -self.context_size:]
                if cond.shape[1] < self.context_size:
                    pad = cond[:, :1].repeat(1, self.context_size - cond.shape[1])
                    cond = torch.cat([pad, cond], dim=1)

                # Forward pass + logit transform (triangular or 3D CP-violation)
                logits = self(
                    cond,
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
                next_id = torch.multinomial(probs, num_samples=1)

                curr_idx = torch.cat([curr_idx, next_id], dim=1)
                candidate_sequence.append(idx_to_word[next_id.item()])

            seeds.append(candidate_sequence)

        return seeds

    def inspect_cp_violation_gradient(self, idx, violation_strength=0.15):
        """
        Convenience helper: runs a forward pass with transpose_mode='cp_violation_3d'
        and compute_gradient=True, returning (logits, gradient) so the caller can
        inspect how sensitive the CP-violation transform is to the raw logits at
        this context. Not used inside no_grad generation loops.
        """
        logits, gradient = self.forward(
            idx,
            violation_strength=violation_strength,
            compute_gradient=True,
        )
        return logits, gradient


# ==========================================================
# 4. Pipeline Execution Loop
# ==========================================================

if __name__ == "__main__":
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

        # Optional: inspect the CP-violation gradient at this context before generating
        grad_logits, grad_tensor = markov_seed_layer.inspect_cp_violation_gradient(
            context_tensor, violation_strength=0.15
        )
        if grad_tensor is not None:
            print(f"[cp_violation_3d] gradient norm at this context: {grad_tensor.norm().item():.4f}")

        print("\n=== GENERATING SAMPLES WITH 3D CP-VIOLATION TRANSPOSE ===")
        seeds = markov_seed_layer.generate_population_seeds(
            seed_context_idx=context_tensor,
            sequence_length=150,
            pop_size=1,
            amp_boost=amp_boost,
            idx_to_word=idx_to_word,
            unk_id=unk_id,
            violation_strength=0.15,
            temperature=0.8,
            top_k=25,
        )

        flattened_words = [word for trigram in seeds[0] for word in trigram]
        output_text = ' '.join(flattened_words)
        print(f"{output_text}")

        print("-" * 65)
