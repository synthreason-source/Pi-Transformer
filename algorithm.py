import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------
# Pipeline shape (latest SynthReason - 2026 diagram):
#
#   Dataset --> Prompt Isolate <-- Reasoning Prompt Markov Subset
#                                          |
#                                    (its own output)
#                                          v
#                                     New Dataset --> Contextual Prompt
#                                                      Markov Subset
#                                                            |
#                                                      (its own output)
#                                                            v
#                                                       New Dataset --> Prompt
#                                                                       Markov
#                                                                       Subset
#                                                                            |
#                                                                            v
#                                                                     Generate Out
#
# Key change from the previous version: each subset no longer just
# hands the next subset a two-word seed context inside a SHARED
# transition table. Instead, each subset's generated text becomes an
# entirely fresh corpus -- a "New Dataset" -- that the next subset
# builds its OWN trigram table / semantic linked list from scratch.
# "Prompt Isolate" is a single reusable OPERATION (not a one-time
# setup step): it's called every time we need to turn a block of text
# into a valid (seed_context, vector) pair against whatever vocab is
# currently in play. The diagram draws it once because it's the same
# function reused at each loop-back point.
#
# UNCHANGED constraint preserved throughout: optical_mac_reference is
# still the only place a token index is derived from.
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


def tokenize(text):
    return text.lower().split()


# ----------------------------------------------------------------------
# Cybernetic primitives (kept from the previous version)
# ----------------------------------------------------------------------

def shannon_entropy(dist, eps=1e-12):
    p = np.clip(dist, eps, None)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def kl_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    return float((p * np.log2(p / q)).sum())


class MarkovSignal:
    __slots__ = ("name", "dist", "entropy", "divergence_from_input")

    def __init__(self, name, dist, input_dist=None):
        self.name = name
        self.dist = dist
        self.entropy = shannon_entropy(dist)
        self.divergence_from_input = (
            kl_divergence(dist, input_dist) if input_dist is not None else 0.0
        )


class HomeostatController:
    def __init__(self, target_entropy, kp=0.08, alpha_bounds=(0.05, 0.85), alpha_init=0.35):
        self.target_entropy = target_entropy
        self.kp = kp
        self.lo, self.hi = alpha_bounds
        self.alpha = alpha_init
        self.error_trace = []
        self.alpha_trace = []

    def update(self, observed_entropy):
        error = observed_entropy - self.target_entropy
        self.alpha = float(np.clip(self.alpha + self.kp * error, self.lo, self.hi))
        self.error_trace.append(error)
        self.alpha_trace.append(self.alpha)
        return self.alpha, error


# ----------------------------------------------------------------------
# Dataset bundle: everything derived from one corpus of tokens. Every
# "New Dataset" triangle in the diagram is one of these, built fresh.
# ----------------------------------------------------------------------

@dataclass
class DatasetBundle:
    tokens: list
    vocab: list
    word_to_idx: dict
    contexts: list
    context_to_idx: dict
    B: np.ndarray
    vectors: np.ndarray
    nodes: list
    vocab_size: int


class ContextNode:
    __slots__ = ("context", "idx", "vector", "next", "prev")

    def __init__(self, context, idx, vector):
        self.context = context
        self.idx = idx
        self.vector = vector
        self.next = None
        self.prev = None


def build_trigram_tables(tokens):
    vocab = sorted(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    contexts = []
    context_to_idx = {}
    counts = {}

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

    B = np.zeros((num_contexts, vocab_size), dtype=float)
    for cidx, wc in counts.items():
        total = sum(wc.values())
        for widx, c in wc.items():
            B[cidx, widx] = c / total

    return vocab, word_to_idx, contexts, context_to_idx, B


def build_context_vectors(tokens, contexts, context_to_idx, vocab_size, word_to_idx, window=3):
    vectors = np.zeros((len(contexts), vocab_size), dtype=float)
    for i in range(len(tokens) - 2):
        ctx = (tokens[i], tokens[i + 1])
        cidx = context_to_idx.get(ctx)
        if cidx is None:
            continue
        lo = max(0, i - window)
        hi = min(len(tokens), i + 2 + window)
        for t in tokens[lo:hi]:
            vectors[cidx, word_to_idx[t]] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_semantic_linked_list(contexts, vectors):
    n = len(contexts)
    nodes = [ContextNode(contexts[i], i, vectors[i]) for i in range(n)]
    if n == 0:
        return nodes, None
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        sims = vectors @ vectors[current]
        sims[visited] = -np.inf
        nxt = int(np.argmax(sims))
        nodes[current].next = nodes[nxt]
        nodes[nxt].prev = nodes[current]
        visited[nxt] = True
        order.append(nxt)
        current = nxt
    return nodes, nodes[order[0]]


def build_dataset_bundle(tokens):
    """Turns raw tokens into a full DatasetBundle. This IS the 'New
    Dataset' triangle: called fresh at every stage boundary on
    whatever text the previous subset produced."""
    vocab, word_to_idx, contexts, context_to_idx, B = build_trigram_tables(tokens)
    vocab_size = len(vocab)
    vectors = build_context_vectors(tokens, contexts, context_to_idx, vocab_size, word_to_idx)
    nodes, _ = build_semantic_linked_list(contexts, vectors)
    return DatasetBundle(tokens, vocab, word_to_idx, contexts, context_to_idx, B, vectors, nodes,
                          vocab_size)


# ----------------------------------------------------------------------
# STAGE: Dataset In (reads the file that seeds the very first bundle)
# ----------------------------------------------------------------------

def stage_dataset_in(path):
    with open(path, "r", encoding="utf-8") as f:
        corpus = f.read()
    return tokenize(corpus)


# ----------------------------------------------------------------------
# STAGE: Prompt Isolate -- a reusable operation, not a one-time step.
# Called at every loop-back point to turn a block of text into a valid
# (seed_context, vector) pair against a given bundle's vocab.
# ----------------------------------------------------------------------

def isolate_seed(text, bundle):
    prompt_tokens = tokenize(text)
    if len(prompt_tokens) < 2:
        raise ValueError("Need at least two tokens to isolate a seed context")
    seed_context = (prompt_tokens[-2], prompt_tokens[-1])

    vector = np.zeros(bundle.vocab_size, dtype=float)
    for t in prompt_tokens:
        if t in bundle.word_to_idx:
            vector[bundle.word_to_idx[t]] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return seed_context, vector


# ----------------------------------------------------------------------
# Per-token signal builders, parameterized by a DatasetBundle
# ----------------------------------------------------------------------

def nearest_known_context(query_vector, vectors):
    sims = vectors @ query_vector
    return int(np.argmax(sims)), float(np.max(sims))


def vectorize_query_context(context, word_to_idx, vocab_size):
    v = np.zeros(vocab_size, dtype=float)
    for w in context:
        if w in word_to_idx:
            v[word_to_idx[w]] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def reasoning_signal_for(context, bundle, hops):
    uniform = np.ones(bundle.vocab_size) / bundle.vocab_size

    if context in bundle.context_to_idx:
        start_idx = bundle.context_to_idx[context]
    else:
        qv = vectorize_query_context(context, bundle.word_to_idx, bundle.vocab_size)
        if np.linalg.norm(qv) == 0 or len(bundle.vectors) == 0:
            return MarkovSignal("Reasoning", uniform, input_dist=uniform)
        start_idx, _ = nearest_known_context(qv, bundle.vectors)

    rows = [bundle.B[start_idx]]
    node = bundle.nodes[start_idx]
    for _ in range(hops):
        if node.next is None:
            break
        node = node.next
        rows.append(bundle.B[node.idx])

    prior = np.mean(rows, axis=0)
    total = prior.sum()
    dist = prior / total if total > 0 else uniform
    return MarkovSignal("Reasoning", dist, input_dist=uniform)


def resolve_context_row(context, bundle, min_sim=0.05):
    if context in bundle.context_to_idx:
        return bundle.context_to_idx[context], True
    qv = vectorize_query_context(context, bundle.word_to_idx, bundle.vocab_size)
    if np.linalg.norm(qv) == 0 or len(bundle.vectors) == 0:
        return None, False
    nn_idx, sim = nearest_known_context(qv, bundle.vectors)
    if sim < min_sim:
        return None, False
    return nn_idx, False


def contextual_signal_for(reasoning_signal, context, bundle):
    row_idx, exact = resolve_context_row(context, bundle)
    if row_idx is None:
        return MarkovSignal("Contextual", reasoning_signal.dist, input_dist=reasoning_signal.dist), False
    contextual = reasoning_signal.dist * bundle.B[row_idx]
    total = contextual.sum()
    dist = contextual / total if total > 0 else reasoning_signal.dist
    return MarkovSignal("Contextual", dist, input_dist=reasoning_signal.dist), exact


def prompt_signal_for(contextual_signal, prompt_vector, alpha):
    if np.linalg.norm(prompt_vector) == 0 or alpha <= 0:
        return MarkovSignal("Prompt", contextual_signal.dist, input_dist=contextual_signal.dist)
    prompt_weight = prompt_vector.copy()
    if prompt_weight.sum() > 0:
        prompt_weight = prompt_weight / prompt_weight.sum()
    blended = (1 - alpha) * contextual_signal.dist + alpha * prompt_weight
    total = blended.sum()
    dist = blended / total if total > 0 else contextual_signal.dist
    return MarkovSignal("Prompt", dist, input_dist=contextual_signal.dist)


def sample_from_signal(signal, rng):
    """Still the only path to a token index: optical_mac_reference."""
    vocab_size = signal.dist.shape[0]
    A = np.ones((1, 1), dtype=float)
    B_row = signal.dist.reshape(1, vocab_size)
    A_enc = encode_same_plane(A, np.ones_like(A))
    B_enc = encode_same_plane(B_row, np.ones_like(B_row))
    optical_C, channels = optical_mac_reference(A_enc, B_enc)
    dist = optical_C[0]
    total = dist.sum()
    if total <= 0:
        return None
    probs = dist / total
    return int(rng.choice(len(probs), p=probs))


def score_tokens(tokens_out, seed_len):
    body = tokens_out[seed_len:]
    if not body:
        return -1e9
    unique_ratio = len(set(body)) / len(body)
    return unique_ratio * len(body)


# ----------------------------------------------------------------------
# STAGE: Reasoning Prompt Markov Subset
# (runs on whatever bundle it's handed -- here, the ORIGINAL dataset,
# per the diagram's "Dataset" triangle feeding this side of the graph)
# ----------------------------------------------------------------------

def stage_reasoning_prompt_markov_subset(seed_context, bundle, n_tokens, seed_range, hops=3):
    best_tokens, best_score, best_log = None, -1e18, None
    for s in seed_range:
        rng = np.random.default_rng(s)
        tokens_out = list(seed_context)
        context = seed_context
        entropy_log = []
        for _ in range(n_tokens):
            signal = reasoning_signal_for(context, bundle, hops)
            idx = sample_from_signal(signal, rng)
            if idx is None:
                break
            entropy_log.append(signal.entropy)
            word = bundle.vocab[idx]
            tokens_out.append(word)
            context = (context[1], word)
        sc = score_tokens(tokens_out, len(seed_context))
        if sc > best_score:
            best_score, best_tokens, best_log = sc, tokens_out, entropy_log
    return best_tokens, best_log


# ----------------------------------------------------------------------
# STAGE: Contextual Prompt Markov Subset
# (runs on the New Dataset built from the Reasoning subset's own text)
# ----------------------------------------------------------------------

def stage_contextual_prompt_markov_subset(seed_context, bundle, n_tokens, seed_range, hops=3):
    best_tokens, best_score, best_log = None, -1e18, None
    for s in seed_range:
        rng = np.random.default_rng(s)
        tokens_out = list(seed_context)
        context = seed_context
        entropy_log = []
        for _ in range(n_tokens):
            reasoning_signal = reasoning_signal_for(context, bundle, hops)
            contextual_signal, _ = contextual_signal_for(reasoning_signal, context, bundle)
            idx = sample_from_signal(contextual_signal, rng)
            if idx is None:
                break
            entropy_log.append(contextual_signal.entropy)
            word = bundle.vocab[idx]
            tokens_out.append(word)
            context = (context[1], word)
        sc = score_tokens(tokens_out, len(seed_context))
        if sc > best_score:
            best_score, best_tokens, best_log = sc, tokens_out, entropy_log
    return best_tokens, best_log


# ----------------------------------------------------------------------
# STAGE: Prompt Markov Subset (final; runs on the New Dataset built
# from the Contextual subset's own text; re-anchored to the ORIGINAL
# prompt, re-vectorized against THIS bundle's vocab, via the homeostat
# controller)
# ----------------------------------------------------------------------

def stage_prompt_markov_subset(seed_context, prompt_vector, bundle, n_tokens, seed_range, hops=3,
                                target_entropy_frac=0.55, kp=0.08):
    max_entropy = np.log2(bundle.vocab_size) if bundle.vocab_size > 1 else 1.0
    best_tokens, best_score, best_log = None, -1e18, None

    for s in seed_range:
        rng = np.random.default_rng(s)
        tokens_out = list(seed_context)
        context = seed_context
        controller = HomeostatController(target_entropy=target_entropy_frac * max_entropy, kp=kp,
                                          alpha_init=0.35)
        log = {"reasoning_H": [], "contextual_H": [], "prompt_H": [], "alpha": [], "error": []}

        for _ in range(n_tokens):
            reasoning_signal = reasoning_signal_for(context, bundle, hops)
            contextual_signal, _ = contextual_signal_for(reasoning_signal, context, bundle)
            final_signal = prompt_signal_for(contextual_signal, prompt_vector, controller.alpha)

            idx = sample_from_signal(final_signal, rng)
            if idx is None:
                break

            _, error = controller.update(final_signal.entropy)
            log["reasoning_H"].append(reasoning_signal.entropy)
            log["contextual_H"].append(contextual_signal.entropy)
            log["prompt_H"].append(final_signal.entropy)
            log["alpha"].append(controller.alpha)
            log["error"].append(error)

            word = bundle.vocab[idx]
            tokens_out.append(word)
            context = (context[1], word)

        avg_abs_error = np.mean(np.abs(log["error"])) if log["error"] else 0.0
        sc = score_tokens(tokens_out, len(seed_context)) - 0.15 * avg_abs_error
        if sc > best_score:
            best_score, best_tokens, best_log = sc, tokens_out, log

    return best_tokens, best_log


# ----------------------------------------------------------------------
# STAGE: Generate Out -- orchestrates the whole chain end to end
# ----------------------------------------------------------------------

def stage_generate_out(dataset_path, prompt_text, reasoning_tokens=400, contextual_tokens=400,
                        final_tokens=20, seed_range=range(20), hops=3):
    # Dataset (original) --------------------------------------------------
    tokens0 = stage_dataset_in(dataset_path)
    bundle0 = build_dataset_bundle(tokens0)

    # Prompt Isolate (first use, against the original dataset) -----------
    seed_context0, _ = isolate_seed(prompt_text, bundle0)

    # Reasoning Prompt Markov Subset --------------------------------------
    reasoning_output, r_log = stage_reasoning_prompt_markov_subset(
        seed_context0, bundle0, reasoning_tokens, seed_range, hops
    )
    reasoning_text = " ".join(reasoning_output)

    # loop-back arrow: Reasoning -> Prompt Isolate (reused operation,
    # against the ORIGINAL vocab since every word Reasoning produced
    # necessarily came from bundle0's vocab)
    seed_context1, _ = isolate_seed(reasoning_text, bundle0)

    # New Dataset, built fresh from the Reasoning subset's own output ----
    bundle1 = build_dataset_bundle(tokenize(reasoning_text))

    # Contextual Prompt Markov Subset (runs on New Dataset bundle1) -------
    contextual_output, c_log = stage_contextual_prompt_markov_subset(
        seed_context1, bundle1, contextual_tokens, seed_range, hops
    )
    contextual_text = " ".join(contextual_output)

    # New Dataset, built fresh from the Contextual subset's own output ---
    bundle2 = build_dataset_bundle(tokenize(contextual_text))
    seed_context2, _ = isolate_seed(contextual_text, bundle2)

    # Original prompt re-anchored into bundle2's vocab space for the
    # final stage's homeostat-controlled blending
    _, prompt_vector_final = isolate_seed(prompt_text, bundle2)

    # Prompt Markov Subset (final, runs on New Dataset bundle2) ----------
    final_output, p_log = stage_prompt_markov_subset(
        seed_context2, prompt_vector_final, bundle2, final_tokens, seed_range, hops
    )

    return {
        "reasoning_output": reasoning_output,
        "contextual_output": contextual_output,
        "final_output": final_output,
        "final_log": p_log,
        "bundle_sizes": (bundle0.vocab_size, bundle1.vocab_size, bundle2.vocab_size),
    }


def run():
    dataset_path = input("Dataset filename: ")
    prompt_text = input("Prompt: ")

    result = stage_generate_out(dataset_path, prompt_text)

    v0, v1, v2 = result["bundle_sizes"]
    print(f"[Dataset -> Prompt Isolate] original vocab={v0}")
    print(f"\n[Reasoning Prompt Markov Subset output]")
    print(" ".join(result["reasoning_output"]))
    print(f"\n[New Dataset from Reasoning output] vocab={v1}")
    print(f"\n[Contextual Prompt Markov Subset output]")
    print(" ".join(result["contextual_output"]))
    print(f"\n[New Dataset from Contextual output] vocab={v2}")
    print(f"\n[Generate Out] (Prompt Markov Subset, anchored to original prompt)")
    print(" ".join(result["final_output"]))

    log = result["final_log"]
    print(f"\nFinal-pass mean entropy: reasoning={np.mean(log['reasoning_H']):.3f}  "
          f"contextual={np.mean(log['contextual_H']):.3f}  prompt={np.mean(log['prompt_H']):.3f} bits")
    print(f"Final-pass mean |feedback error|={np.mean(np.abs(log['error'])):.3f} bits")

    steps = list(range(len(log["reasoning_H"])))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(steps, log["reasoning_H"], label="Reasoning H")
    axes[0].plot(steps, log["contextual_H"], label="Contextual H")
    axes[0].plot(steps, log["prompt_H"], label="Prompt (final) H")
    axes[0].set_ylabel("entropy (bits)")
    axes[0].set_title("Final pass: per-stage entropy (on 2nd-generation New Dataset)")
    axes[0].legend()

    axes[1].plot(steps, log["alpha"], label="alpha (control gain)", color="tab:green")
    axes[1].plot(steps, log["error"], label="feedback error (bits)", color="tab:orange")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_xlabel("generation step")
    axes[1].set_ylabel("controller state")
    axes[1].set_title("Homeostat controller (final pass)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("new_dataset_pipeline_diagnostics.png", dpi=150)

    return result


if __name__ == "__main__":
    run()
