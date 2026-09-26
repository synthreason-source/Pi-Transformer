
import argparse
import glob
import math
import re
import sys
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

UNK = "<unk>"
BOS = "<bos>"


# ---------------------------------------------------------------------------
# 1. Dataset loading (unchanged — no construct in the paper maps to this)
# ---------------------------------------------------------------------------

def load_corpus(paths=None, raw_text=None, lowercase=True, strip_punct=False):
    text_chunks = []
    if paths:
        found_any = False
        for pattern in paths:
            for fp in sorted(glob.glob(pattern)):
                found_any = True
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    text_chunks.append(f.read())
        if not found_any:
            print(f"[warn] no files matched {paths}", file=sys.stderr)

    if raw_text:
        text_chunks.append(raw_text)
    if not text_chunks:
        text_chunks.append(_DEMO_CORPUS)

    full_text = "\n".join(text_chunks)
    if lowercase:
        full_text = full_text.lower()

    tokens = re.findall(r"[a-z0-9']+|[.,!?;:]", full_text)
    if strip_punct:
        tokens = [t for t in tokens if re.match(r"[a-z0-9']+", t)]
    return tokens


with open(input("Filename: "), "r", encoding="utf-8") as file:
    _DEMO_CORPUS = file.read()


def build_vocab(tokens, min_count=1):
    counts = Counter(tokens)
    kept = sorted([w for w, c in counts.items() if c >= min_count])
    vocab = [UNK, BOS] + kept
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_idx, counts


# ---------------------------------------------------------------------------
# 2. The membrane: context-state construction
#
#    Exchange: "topological energy membrane separating internal state space
#    from environment" -> the boundary crossed when the Markov chain moves
#    from one context-state to the next. d-Omega is not a spatial surface;
#    it is the transition itself. Quanta N are tokens: one per crossing,
#    genuinely indivisible (the tokenizer's vocabulary IS a discrete set),
#    which is the one place in the whole exercise where "quanta" is not a
#    metaphor borrowed from QM — it is just an accurate word for "discrete
#    countable unit," which is what a token already was.
# ---------------------------------------------------------------------------

def build_membrane_transition_graph(tokens, word_to_idx, order=3):
    """Build the square context -> context transition count matrix. This
    IS the membrane: d-Omega = the set of (context_state -> context_state)
    edges; a single generation step = one crossing of d-Omega; the emitted
    word at that step = one quantum.

    Renamed from build_ngram_contexts(). Same function, same sparse-matrix
    justification: a dense (num_contexts x num_contexts) "membrane" at
    real corpus scale is gigabytes to hundreds of gigabytes, which is
    exactly the mass-density blowup discussed in build_mass_density_term()
    below — the membrane and the mass term are two names for adjacent
    facts about the same object, not two separate mechanisms.
    """
    ctx_len = order - 1
    if ctx_len < 1:
        raise ValueError("order must be >= 2")

    bos_idx = word_to_idx[BOS]
    unk_idx = word_to_idx[UNK]
    ids = [word_to_idx.get(t, unk_idx) for t in tokens]
    padded = [bos_idx] * ctx_len + ids

    context_to_idx = {}
    contexts = []
    context_counts = Counter()
    edges = Counter()

    def get_ctx_idx(ctx_tuple):
        if ctx_tuple not in context_to_idx:
            context_to_idx[ctx_tuple] = len(contexts)
            contexts.append(ctx_tuple)
        return context_to_idx[ctx_tuple]

    n = len(padded)
    for i in range(n - ctx_len):
        ctx = tuple(padded[i:i + ctx_len])
        next_ctx = tuple(padded[i + 1:i + 1 + ctx_len])
        ci = get_ctx_idx(ctx)
        cj = get_ctx_idx(next_ctx)
        context_counts[ctx] += 1
        edges[(ci, cj)] += 1.0

    m = len(contexts)
    if edges:
        rows, cols, data = zip(*[(r, c, v) for (r, c), v in edges.items()])
    else:
        rows, cols, data = (), (), ()
    membrane = sp.csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float64)

    return contexts, context_to_idx, membrane, context_counts


def log_sort_contexts(contexts, context_counts):
    def key(ctx):
        c = context_counts.get(ctx, 0)
        return -math.log(c + 1.0)
    return sorted(range(len(contexts)), key=lambda i: key(contexts[i]))


def permute_matrix(M, order):
    order = np.asarray(order)
    if sp.issparse(M):
        return M[order, :][:, order].tocsr()
    return M[np.ix_(order, order)]


def membrane_flux_normalize(M, eps=1e-9):
    """Row-normalize the membrane's transition weights into a proper
    conservation law: total probability flux leaving any context-state
    across d-Omega sums to 1. This is the literal, enforced version of the
    paper's heuristic surface integral N = oint_dOmega sigma(x) dx — not
    an analogy to it, an instance of it, since a row-stochastic matrix row
    IS a discrete measure over the boundary that sums to a fixed quantum
    budget per step.

    Renamed from row_normalize(). Behavior unchanged.
    """
    if sp.issparse(M):
        row_sums = np.asarray(M.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        inv = sp.diags(1.0 / row_sums)
        return (inv @ M).tocsr()
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return M / (row_sums + eps)


# ---------------------------------------------------------------------------
# 3. Mass-density coupling term
#
#    Exchange: rho_eff (effective mass-density, coupled to retrospective
#    orientation) -> the memory footprint and O(n^2)-to-O(n^3) compute cost
#    of diffusing probability mass across the membrane. This is the ONE
#    construct in the original paper that upgrades from metaphor to fact
#    when ported to code: "retrospection carries more computational mass"
#    stops being poetic and becomes a literal, measurable claim about
#    bytes and FLOPs, which is exactly what the original file's own
#    docstring already said before any relabeling ("137,702 contexts ->
#    ~141 GiB").
# ---------------------------------------------------------------------------

def build_mass_density_term(P, beta=0.15, remove_self_loops=True):
    """expm(beta * (P - I)): the diffusion/propagation step. In the paper's
    vocabulary this is where mass-density is "felt" — it is the most
    expensive operation in the pipeline (O(n^3) for a state space of size
    n), and its cost is precisely what rho_eff = M(t)/|Omega_active(t)|
    measures. A system leaning harder on this term (more diffusion, less
    raw/verbatim retrospective lookup) really does carry more
    computational mass: more memory, more latency, more bandwidth.

    Renamed from diffusion_weighting(). Behavior unchanged, including the
    self-loop caveat: exp(-beta) concentrates on the diagonal for small
    beta, which would make generation stick to one context-state forever
    (i.e., infinite effective mass, zero motion) unless removed.
    """
    m = P.shape[0]
    L = P - np.eye(m)
    D = expm(beta * L)
    D = np.clip(D, 0, None)

    if remove_self_loops:
        np.fill_diagonal(D, 0.0)
        row_sums = D.sum(axis=1)
        dead_rows = row_sums <= 1e-12
        if np.any(dead_rows):
            fallback = P.copy()
            np.fill_diagonal(fallback, 0.0)
            fb_sums = fallback.sum(axis=1, keepdims=True)
            fb_sums[fb_sums == 0] = 1.0
            fallback = fallback / fb_sums
            D[dead_rows] = fallback[dead_rows]

    D = membrane_flux_normalize(D)
    return D


def blend_temporal_orientation(P_retrospective, P_diffused, alpha=0.5,
                                remove_self_loops=True):
    """Blend the purely retrospective (raw n-gram, tau = -1) distribution
    with the mass-density-diffused one, controlled by alpha. alpha is the
    closest thing in this codebase to a dial on tau: alpha=0 is maximally
    retrospective (pure lookup), alpha=1 is maximally diffused (mass-heavy,
    still not "anticipatory" in any real sense — see the note in
    compute_temporal_index() below about why diffusion isn't lookahead).

    Renamed from blend(). Behavior unchanged.
    """
    P = (1 - alpha) * P_retrospective * alpha * P_diffused
    if remove_self_loops:
        np.fill_diagonal(P, 0.0)
        P = membrane_flux_normalize(P)
    return P


def compute_mass_density(num_states, dtype_bytes=8):
    """rho_eff = M(t) / |Omega_active(t)|, made concrete: bytes required
    for a DENSE membrane of this many context-states, divided by the
    number of active states. This is the real, computable version of the
    coupling constant lambda from the original speculative paper — no
    free parameter, just arithmetic.
    """
    dense_bytes = (num_states ** 2) * dtype_bytes
    rho_eff = dense_bytes / max(num_states, 1)
    return dense_bytes, rho_eff


# ---------------------------------------------------------------------------
# 4. Temporal index computation
#
#    Exchange: tau (subjective temporal orientation) -> fraction of
#    per-step decision compute spent on the payload lookahead-rerank
#    (prospective: scoring a word BEFORE it's committed) versus the raw
#    context lookup (retrospective: reading what was already observed).
# ---------------------------------------------------------------------------

def compute_temporal_index(boost_strength, lookahead_ops=1, retrospective_ops=1):
    """tau = (C_lookahead - C_retrospective) / (C_lookahead + C_retrospective),
    gated by whether the payload customizer is active at all.

    With boost_strength == 0, the payload rerank never executes (see
    sample_at_boundary below), so tau = -1 exactly: this generator is
    purely retrospective, same as any causal-masked autoregressive model.

    With boost_strength > 0, the payload dot-product IS a one-step
    lookahead — it scores the word that would be newly introduced by a
    candidate transition before that transition is taken — so tau ticks
    positive by an amount proportional to how much compute that rerank
    consumes relative to the base lookup. This is a real, bounded,
    non-metaphorical number, not a stand-in for felt anticipation.

    Note on the diffusion term: build_mass_density_term() does NOT count
    as lookahead, even though it's a "future-looking" propagator in the
    continuous-time-Markov sense. It diffuses probability mass over
    ALREADY-OBSERVED transitions; it doesn't evaluate not-yet-taken
    candidate words the way the payload rerank does. Mass-density and
    temporal index are independent axes here, exactly as the original
    paper treated rho_eff and tau as coupled-but-distinct variables.
    """
    if boost_strength == 0.0:
        return -1.0
    total = lookahead_ops + retrospective_ops
    return (boost_strength * lookahead_ops - retrospective_ops) / (
        boost_strength * lookahead_ops + retrospective_ops
    )


# ---------------------------------------------------------------------------
# 5. Sinusoidal temperature — kept under its plain name
#
#    This is the one function this file will NOT rename into the paper's
#    "harmonic resonance" vocabulary. See module docstring.
# ---------------------------------------------------------------------------

def sinusoidal_temperature(step, base_temp=0.8, amplitude=0.4, omega=0.5, phase=0.0, floor=0.05):
    t = base_temp + amplitude * math.sin(omega * step + phase)
    return max(floor, t)


# ---------------------------------------------------------------------------
# 6. Sampling: where the decision operator D acts on the rule R
#
#    Exchange: "decisions precede the rule-set" -> D narrows/reranks the
#    ALREADY-LAWFUL candidate set (nonzero-probability transitions); it
#    can never make a zero-probability transition eligible. R (the
#    observed n-gram law, P_final) is never edited by D. This is the
#    weak, defensible reading of the inversion, not the strong claim that
#    decisions rewrite physics.
# ---------------------------------------------------------------------------

def sample_at_boundary(probs, temperature=1.0, top_k=10, rng=None,
                        lookahead_bias=None, boost_strength=0.0):
    """Cross d-Omega once: pick the next context-state (i.e. emit one
    quantum). `lookahead_bias` is the decision operator D's input;
    `probs` (P_final's row) is the rule R. D can only rerank within
    nonzero_idx — the domain R already permits — never expand it.

    Renamed from sample_next(). Behavior unchanged.
    """
    rng = rng or np.random.default_rng()
    probs = np.asarray(probs, dtype=np.float64)

    nonzero_idx = np.nonzero(probs > 0)[0]
    if len(nonzero_idx) == 0:
        return None

    if top_k is not None and 0 < top_k < len(nonzero_idx):
        sub = np.argpartition(probs[nonzero_idx], -top_k)[-top_k:]
        top_idx = nonzero_idx[sub]
    else:
        top_idx = nonzero_idx

    top_probs = probs[top_idx]
    logits = np.log(top_probs) / max(temperature, 1e-6)
    if lookahead_bias is not None and boost_strength != 0.0:
        logits = logits + boost_strength * np.asarray(lookahead_bias)[top_idx]
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    choice = rng.choice(len(top_idx), p=weights)
    return top_idx[choice]


def parse_payload(payload_str, word_to_idx):
    vocab_size = len(word_to_idx)
    payload_vector = np.zeros(vocab_size, dtype=np.float64)
    if not payload_str:
        return payload_vector
    for item in payload_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            word, weight_str = item.rsplit(":", 1)
            try:
                weight = float(weight_str)
            except ValueError:
                print(f"[warn] payload entry '{item}' has a non-numeric weight; skipping", file=sys.stderr)
                continue
        else:
            word, weight = item, 1.0
        word = word.strip().lower()
        idx = word_to_idx.get(word)
        if idx is None:
            print(f"[warn] payload word '{word}' is not in the vocabulary; skipping", file=sys.stderr)
            continue
        payload_vector[idx] = weight
    return payload_vector


class HECMModel:
    """Same model as before. Internally, the membrane, mass-density term,
    and temporal index are now named as such and exposed as attributes
    (self.rho_eff, self.tau) so they can be inspected directly rather than
    inferred from comments."""

    def __init__(self, tokens, order=3, min_count=1, beta=0.0, alpha=0.0,
                 max_diffusion_states=1500, payload=None, boost_strength=0.0):
        self.order = order
        vocab, word_to_idx, counts = build_vocab(tokens, min_count=min_count)
        self.vocab = vocab
        self.word_to_idx = word_to_idx

        contexts, context_to_idx, membrane, context_counts = build_membrane_transition_graph(
            tokens, word_to_idx, order=order
        )

        perm = log_sort_contexts(contexts, context_counts)
        contexts = [contexts[i] for i in perm]
        membrane = permute_matrix(membrane, perm)
        context_to_idx = {ctx: i for i, ctx in enumerate(contexts)}

        self.contexts = contexts
        self.context_to_idx = context_to_idx
        self.context_counts = context_counts

        self.P_retrospective = membrane_flux_normalize(membrane)

        num_states = len(contexts)
        self.dense_bytes, self.rho_eff = compute_mass_density(num_states)

        self.P_diffused = self.P_retrospective
        self.P_final = self.P_retrospective
        

        self.boost_strength = boost_strength
        self.tau = compute_temporal_index(boost_strength)

        if payload is None:
            payload = np.zeros(len(vocab), dtype=np.float64)
        self.payload = payload
        self.context_word_score = np.array(
            [sum(payload[w] for w in ctx) for ctx in contexts], dtype=np.float64
        )
        self.introduced_word_score = np.array(
            [payload[ctx[-1]] for ctx in contexts], dtype=np.float64
        )
        self.context_total_score = self.context_word_score + 0.0

    def vocab_size(self):
        return len(self.vocab)

    def num_states(self):
        return len(self.contexts)

    def _get_row(self, ctx_idx):
        row = self.P_final[ctx_idx]
        if sp.issparse(self.P_final):
            return np.asarray(row.todense()).ravel()
        return np.asarray(row)

    def _tokenize_prompt(self, prompt):
        toks = re.findall(r"[a-z0-9']+|[.,!?;:]", prompt.lower())
        unk = self.word_to_idx[UNK]
        return [self.word_to_idx.get(t, unk) for t in toks]

    def _weighted_pick(self, candidates, boost_strength, rng):
        if boost_strength == 0.0 or len(candidates) == 1:
            return candidates[rng.integers(len(candidates))] if len(candidates) > 1 else candidates[0]
        scores = boost_strength * self.context_total_score[candidates]
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        return candidates[rng.choice(len(candidates), p=weights)]

    def _initial_context(self, prompt_ids, rng, boost_strength=0.0):
        ctx_len = self.order - 1
        bos = self.word_to_idx[BOS]
        padded = [bos] * ctx_len + prompt_ids
        desired = tuple(padded[-ctx_len:])

        if desired in self.context_to_idx:
            return self.context_to_idx[desired]

        for k in range(ctx_len - 1, 0, -1):
            suffix = desired[-k:]
            candidates = [i for i, ctx in enumerate(self.contexts) if ctx[-k:] == suffix]
            if candidates:
                return self._weighted_pick(candidates, boost_strength, rng)

        counts_arr = np.array([self.context_counts.get(ctx, 0) for ctx in self.contexts])
        top_candidates = list(np.argsort(counts_arr)[-10:])
        return self._weighted_pick(top_candidates, boost_strength, rng)

    def generate(self, prompt, max_new_tokens=40, top_k=10, base_temp=0.8,
                 amplitude=0.4, omega=0.5, phase=0.0, seed=None,
                 boost_strength=0.0):
        rng = np.random.default_rng(seed)
        prompt_ids = self._tokenize_prompt(prompt)

        ctx_idx = self._initial_context(prompt_ids, rng, boost_strength=boost_strength)
        out_ids = list(prompt_ids)
        quanta_crossed = 0  # N: count of boundary crossings this run

        for step in range(max_new_tokens):
            temp = sinusoidal_temperature(step, base_temp=base_temp, amplitude=amplitude,
                                           omega=omega, phase=phase)
            probs = self._get_row(ctx_idx)
            next_ctx_idx = sample_at_boundary(
                probs, temperature=temp, top_k=top_k, rng=rng,
                lookahead_bias=self.introduced_word_score, boost_strength=boost_strength,
            )

            if next_ctx_idx is None:
                ctx_idx = self._initial_context(out_ids, rng, boost_strength=boost_strength)
                continue

            next_word_id = self.contexts[next_ctx_idx][-1]
            out_ids.append(next_word_id)
            ctx_idx = next_ctx_idx
            quanta_crossed += 1

        self.last_quanta_crossed = quanta_crossed
        words = [self.vocab[i] for i in out_ids]
        return _detokenize(words)


def _detokenize(tokens):
    out = []
    for tok in tokens:
        if tok in (BOS,):
            continue
        if out and re.match(r"[.,!?;:]", tok):
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    text = " ".join(out)
    if text:
        text = text[0].upper() + text[1:]
    return text


def main():
    ap = argparse.ArgumentParser(description="HECM model, relabeled to the temporal-index paper's vocabulary")
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.3, help="mass-density diffusion strength")
    ap.add_argument("--alpha", type=float, default=0.1, help="temporal-orientation blend factor")
    ap.add_argument("--max-diffusion-states", type=int, default=5)
    ap.add_argument("--prompt", type=str, default="the fox and")
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--base-temp", type=float, default=0.8)
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--boost", type=str, default=None)
    ap.add_argument("--boost-strength", type=float, default=1.0)
    args = ap.parse_args()

    tokens = load_corpus(paths=args.data)
    print(f"[info] loaded {len(tokens)} tokens", file=sys.stderr)

    boost_strength = args.boost_strength if args.boost else 0.0

    model = HECMModel(
        tokens, order=args.order, min_count=args.min_count,
        beta=args.beta, alpha=args.alpha,
        max_diffusion_states=args.max_diffusion_states,
        boost_strength=boost_strength,
    )

    print(
        f"[info] order={args.order} | vocab={model.vocab_size()} | "
        f"membrane states={model.num_states()} | "
        f"rho_eff={model.rho_eff:.1f} bytes/state | "
        f"tau={model.tau:.3f}",
        file=sys.stderr,
    )

    if args.boost:
        payload = parse_payload(args.boost, model.word_to_idx)
        model.payload = payload
        model.context_word_score = np.array(
            [sum(payload[w] for w in ctx) for ctx in model.contexts], dtype=np.float64
        )
        model.introduced_word_score = np.array(
            [payload[ctx[-1]] for ctx in model.contexts], dtype=np.float64
        )
        model.context_total_score = model.context_word_score

    def run_once(prompt):
        result = model.generate(
            prompt, max_new_tokens=args.length, top_k=args.top_k,
            base_temp=args.base_temp, amplitude=args.amplitude,
            omega=args.omega, phase=args.phase, seed=args.seed,
            boost_strength=boost_strength,
        )
        print(result)
        print(f"[info] quanta crossed this run (N): {model.last_quanta_crossed}", file=sys.stderr)

    print("[info] interactive mode -- type a prompt, or 'quit' to exit", file=sys.stderr)
    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            break
        if prompt.strip().lower() in ("quit", "exit"):
            break
        run_once(prompt)


if __name__ == "__main__":
    main()
