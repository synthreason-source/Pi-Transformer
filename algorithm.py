import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------
# 1. Pieces reused directly from the uploaded optical-simulation script.
#    These are the ONLY functions allowed to touch the matrix multiply
#    that turns a context vector into a next-token distribution.
# ----------------------------------------------------------------------

def encode_same_plane(values, filters):
    if values.shape != filters.shape:
        raise ValueError("values and filters must have identical shapes")
    if np.any(filters < 0.0) or np.any(filters > 1.0):
        raise ValueError("filters must be within [0, 1]")
    return values * filters


def optical_mac_reference(A, B):
    """Shared-index optical MAC: sum_j A[:, j] (x) B[j, :].
    This is the ONLY place a token index is derived from in this file."""
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("inner dimensions must match")
    channels = np.empty((p, m, n), dtype=float)
    for j in range(p):
        channels[j] = A[:, j:j + 1] * B[j:j + 1, :]
    return channels.sum(axis=0), channels


# ----------------------------------------------------------------------
# 2. Sample corpus + trigram statistics (plain language modeling, no
#    optics involved yet -- this just builds the digital transition
#    table that will later be pushed through the optical MAC).
# ----------------------------------------------------------------------

with open("science_corpus.txt", 'r', encoding='utf-8') as f:
    CORPUS = f.read()

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def build_trigram_tables(tokens):
    vocab = sorted(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    contexts = []
    context_to_idx = {}
    counts = {}  # context_idx -> {word_idx: count}

    for i in range(len(tokens) - 2):
        ctx = (tokens[i], tokens[i + 1])
        nxt = tokens[i + 2]
        if ctx not in context_to_idx:
            context_to_idx[ctx] = len(contexts)
            contexts.append(ctx)
        cidx = context_to_idx[ctx]
        widx = word_to_idx[nxt]
        counts.setdefault(cidx, {})
        counts[cidx][widx] = counts[cidx].get(widx, 0) + 1

    num_contexts = len(contexts)
    vocab_size = len(vocab)

    # Transition matrix B: rows = context, cols = next-word probability
    B = np.zeros((num_contexts, vocab_size), dtype=float)
    for cidx, wc in counts.items():
        total = sum(wc.values())
        for widx, c in wc.items():
            B[cidx, widx] = c / total

    return vocab, word_to_idx, contexts, context_to_idx, B


# ----------------------------------------------------------------------
# 3. Bridge: turn "current context" into a one-hot row vector A, run it
#    through encode_same_plane (trivial pass-through filter, values
#    already in [0,1] as probabilities) and then optical_mac_reference,
#    and read the next token index off the OPTICAL output only.
# ----------------------------------------------------------------------

def context_to_onehot(context, context_to_idx):
    num_contexts = len(context_to_idx)
    A = np.zeros((1, num_contexts), dtype=float)
    if context in context_to_idx:
        A[0, context_to_idx[context]] = 1.0
        return A, True
    return A, False  # unseen context -> zero vector, caller handles fallback


def next_token_via_optical_mac(context, context_to_idx, B, rng):
    A, seen = context_to_onehot(context, context_to_idx)
    if not seen:
        return None, None, None

    # trivial same-plane encoding: filters = 1 everywhere (pass-through),
    # kept here only so the optical pipeline stages from the source
    # script (encode -> MAC) are both actually exercised.
    filters_A = np.ones_like(A)
    filters_B = np.ones_like(B)
    A_enc = encode_same_plane(A, filters_A)
    B_enc = encode_same_plane(B, filters_B)

    optical_C, channels = optical_mac_reference(A_enc, B_enc)  # (1, vocab_size)

    dist = optical_C[0]
    total = dist.sum()
    if total <= 0:
        return None, optical_C, channels

    probs = dist / total
    token_idx = int(rng.choice(len(probs), p=probs))  # sampled from OPTICAL MAC output
    return token_idx, optical_C, channels


# ----------------------------------------------------------------------
# 4. Generation loop
# ----------------------------------------------------------------------

def generate(seed_context, n_tokens, vocab, context_to_idx, B, seed=0):
    rng = np.random.default_rng(seed)
    tokens_out = list(seed_context)
    context = seed_context

    for _ in range(n_tokens):
        token_idx, optical_C, channels = next_token_via_optical_mac(
            context, context_to_idx, B, rng
        )
        if token_idx is None:
            break  # unseen context / dead end, stop generation
        word = vocab[token_idx]
        tokens_out.append(word)
        context = (context[1], word)

    return tokens_out


def run():
    tokens = tokenize(CORPUS)
    vocab, word_to_idx, contexts, context_to_idx, B = build_trigram_tables(tokens)

    print(f"Vocab size: {len(vocab)}")
    print(f"Distinct trigram contexts: {len(contexts)}")
    print(f"Transition matrix B shape (contexts x vocab): {B.shape}")

    seed_context = ("it", "is")
    generated = generate(seed_context, n_tokens=200, vocab=vocab,
                          context_to_idx=context_to_idx, B=B, seed=42)

    print("\nSeed context:", seed_context)
    print("Generated (token indices derived ONLY from optical_mac_reference output):")
    print(" ".join(generated))

    # Show one worked example: distribution + optical MAC channel stack
    A, seen = context_to_onehot(seed_context, context_to_idx)
    filters_A = np.ones_like(A)
    filters_B = np.ones_like(B)
    A_enc = encode_same_plane(A, filters_A)
    B_enc = encode_same_plane(B, filters_B)
    optical_C, channels = optical_mac_reference(A_enc, B_enc)
    top_idx = int(np.argmax(optical_C[0]))
    print(f"\nFor context {seed_context}, optical MAC argmax token index = {top_idx} "
          f"-> word '{vocab[top_idx]}'")

    # ------------------------------------------------------------
    # Diagnostic figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(B, aspect="auto", cmap="viridis")
    axes[0].set_title("Digital trigram transition matrix B\n(context rows x vocab cols)")
    axes[0].set_xlabel("vocab index")
    axes[0].set_ylabel("context index")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(optical_C, aspect="auto", cmap="inferno")
    axes[1].set_title(f"Optical MAC output for context {seed_context}\n(1 x vocab)")
    axes[1].set_xlabel("vocab index")
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Trigram LLM: token index derived only from optical_mac_reference()")
    fig.savefig("trigram_optical_mac.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved figure to trigram_optical_mac.png")

    return {
        "vocab": vocab,
        "contexts": contexts,
        "B": B,
        "generated": generated,
    }


if __name__ == "__main__":
    run()
