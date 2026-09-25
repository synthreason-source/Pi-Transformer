#!/usr/bin/env python3
"""
HECM Toy Language Model (n-gram, default: trigram)
====================================================

A word-level model inspired by the "Harmonic-Exponential Cartesian Mapping"
(HECM) sketch, generalized to arbitrary n-gram order (default: order=3,
i.e. trigrams). See README.md for the full mapping between the original
sketch and what's implemented here.

Key idea for going beyond bigrams (order > 2)
----------------------------------------------
A trigram's natural table is CONTEXT (two words) -> NEXT WORD, which is
rectangular (num_contexts x vocab_size), not square. The matrix-exponential
step needs a SQUARE matrix to mean anything (it's a propagator: it has to
map the state space back onto itself). The standard fix for higher-order
Markov chains is to treat each context -- e.g. the pair (w1, w2) -- as a
single STATE, and build a square STATE -> STATE transition matrix: moving
from context (w1, w2) to context (w2, w3) whenever the trigram (w1, w2, w3)
was observed. This is square by construction, so log-sorting, matrix-
exponential diffusion, and sinusoidal-temperature sampling all carry over
completely unchanged from the bigram case. order=2 (context length 1)
reduces exactly to the original bigram model.

Pipeline:
  1. Dataset loader -> tokenize text, build a vocabulary.
  2. Cartesian product of CONTEXTS x CONTEXTS -> a square count matrix over
     (order-1)-word context states. This generalizes the original A x B
     word-pair matrix.
  3. Log-sorting -> contexts reordered by descending log-frequency.
  4. Matrix-exponential weighting -> expm() on the row-normalized context
     transition matrix (a real diffusion/propagation technique), blended
     with the raw n-gram probabilities.
  5. Sinusoidal modulation -> a sine wave over generation step index biases
     sampling temperature (replaces the ill-defined "curve intersection").
  6. Text prompting -> CLI / function interface, prompt in, continuation out.
  7. Payload customizer -> an optional user-supplied word:weight "payload"
     vector. At each step, every candidate trigram is scored by the dot
     product of the payload against that trigram's word composition, and
     this score reranks (never expands) the set of verbatim candidates --
     so generation can be steered toward chosen themes/words without ever
     breaking the "every trigram is real" guarantee.

Still fundamentally an n-gram frequency model, not a neural network.

Narrative framing (descriptive only -- nothing below is executed)
-------------------------------------------------------------------
Two pieces of the original HECM sketch never got turned into code, because
there was no real mechanism to attach them to (see the README's mapping
table for the parts that *did* get an honest reinterpretation). They're
described here in the same narrative register the sketch used, purely to
preserve the intent -- not to imply the code performs them:

  "Harmonic resonance field" / sinusoidal curve intersection -- the sketch
  pictures each chunk of the transition surface as a string with its own
  natural frequency, and generation as a receiver tuning in to the chunks
  that ring together, letting noisy transitions fall silent while the
  strongly-linked phrases carry through. It's an evocative way to talk
  about surfacing stable patterns in a sequence. Nothing in this file tunes
  into anything, though -- the closest real mechanism is the sinusoidal
  *temperature* schedule in Phase 5, and what it actually does is far
  plainer: it nudges sampling between more cautious and more adventurous
  on a fixed clock, with no sense of which words "belong together."

  "Phase-locked, rhythmically stable feature space" -- the sketch describes
  generation settling into reinforced pathways the way a pendulum settles
  into a stable orbit, so that only "mathematically reinforced harmonic
  nodes" get visited. That's a nice image for "the output gets more
  predictable the longer it runs," but there's no dynamical system here
  with fixed points or phase-locking to settle into. If what you actually
  want is generation that leans toward safer, more repeated phrasing over a
  long run, `--base-temp` and a small `--amplitude` get you there for real
  -- that's an honest, working knob, just not the mechanism the sketch
  describes.
"""

import argparse
import glob
import math
import re
import sys
from collections import Counter

import numpy as np
from scipy.linalg import expm

UNK = "<unk>"
BOS = "<bos>"


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------

def load_corpus(paths=None, raw_text=None, lowercase=True, strip_punct=False):
    """Load one or more text files (glob patterns supported) and/or a raw
    string, and tokenize into a flat list of word tokens."""
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


# ---------------------------------------------------------------------------
# 2. Vocabulary and n-gram context-state construction
# ---------------------------------------------------------------------------

def build_vocab(tokens, min_count=1):
    counts = Counter(tokens)
    kept = sorted([w for w, c in counts.items() if c >= min_count])
    vocab = [UNK, BOS] + kept
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_idx, counts


def build_ngram_contexts(tokens, word_to_idx, order=3):
    """Build the square CONTEXT -> CONTEXT transition count matrix for an
    n-gram model of the given order (order=3 -> trigram, context length 2).

    Returns:
        contexts: list of context tuples (word indices), index-aligned with M.
        context_to_idx: dict mapping context tuple -> row/col index.
        M: (num_contexts x num_contexts) count matrix.
        context_counts: Counter of how often each context tuple occurred.
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
    edges = Counter()  # (context_idx, next_context_idx) -> count

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
    M = np.zeros((m, m), dtype=np.float64)
    for (ci, cj), c in edges.items():
        M[ci, cj] = c

    return contexts, context_to_idx, M, context_counts


def log_sort_contexts(contexts, context_counts):
    """Reorder contexts by descending log-frequency. Returns new order
    (list of old indices) to be applied to both `contexts` and the matrix."""
    def key(ctx):
        c = context_counts.get(ctx, 0)
        return -math.log(c + 1.0)

    order = sorted(range(len(contexts)), key=lambda i: key(contexts[i]))
    return order


def permute_matrix(M, order):
    return M[np.ix_(order, order)]


# ---------------------------------------------------------------------------
# 3 & 4. Normalization and matrix-exponential diffusion weighting
# ---------------------------------------------------------------------------

def row_normalize(M, eps=1e-9):
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return M / (row_sums + eps)


def diffusion_weighting(P, beta=0.15, remove_self_loops=True):
    """expm(beta * (P - I)) -- continuous-time Markov propagator, blended
    later with the raw n-gram probabilities.

    Caveat (important): expm(beta*(P - I)) == exp(-beta) * expm(beta*P).
    The exp(-beta) factor is the continuous-time "probability no jump has
    happened yet," which lands almost entirely on the diagonal (self-loops)
    for small beta. That's a sensible reading of "elapsed time beta" in a
    continuous-time chain, but it's meaningless for discrete, one-word-per-
    step text generation -- every generation step must move to a *different*
    context. Left unchecked, it makes the generator repeat the same word/
    context over and over (e.g. "the the the the").

    remove_self_loops=True (default) strips that diagonal out after the
    exponential is computed and renormalizes each row over its remaining
    (genuinely different) target states, which is what you want for
    generation. Set it False only if you specifically want to inspect the
    raw continuous-time kernel.
    """
    m = P.shape[0]
    L = P - np.eye(m)
    D = expm(beta * L)
    D = np.clip(D, 0, None)

    if remove_self_loops:
        np.fill_diagonal(D, 0.0)
        row_sums = D.sum(axis=1)
        # Rows that lost all their mass (rare: a state whose only diffused
        # probability was the self-loop) fall back to the raw row's
        # off-diagonal distribution, so generation never gets stuck.
        dead_rows = row_sums <= 1e-12
        if np.any(dead_rows):
            fallback = P.copy()
            np.fill_diagonal(fallback, 0.0)
            fb_sums = fallback.sum(axis=1, keepdims=True)
            fb_sums[fb_sums == 0] = 1.0
            fallback = fallback / fb_sums
            D[dead_rows] = fallback[dead_rows]

    D = row_normalize(D)
    return D


def blend(P_raw, P_diffused, alpha=0.5, remove_self_loops=True):
    P = (1 - alpha) * P_raw + alpha * P_diffused
    if remove_self_loops:
        np.fill_diagonal(P, 0.0)
        P = row_normalize(P)
    return P


# ---------------------------------------------------------------------------
# 5. Sinusoidal temperature modulation
# ---------------------------------------------------------------------------

def sinusoidal_temperature(step, base_temp=0.8, amplitude=0.4, omega=0.5, phase=0.0, floor=0.05):
    t = base_temp + amplitude * math.sin(omega * step + phase)
    return max(floor, t)


# ---------------------------------------------------------------------------
# 6. Sampling and generation
# ---------------------------------------------------------------------------

def sample_next(probs, temperature=1.0, top_k=10, rng=None, bias=None, bias_strength=0.0):
    """Sample strictly among states with nonzero probability -- i.e. only
    transitions that were genuinely observed in training data. Zero-
    probability entries are never eligible, however small temperature or
    top_k padding might otherwise make them. Returns None if there is no
    eligible (nonzero-probability) state at all (a true dead end).

    `bias` (optional): a full-length array of per-state scores (e.g. a
    payload dot-product score). When given, it RERANKS the already-eligible
    (nonzero-probability) candidates via bias_strength * bias[state] added
    to their log-probability -- it can never make a zero-probability state
    eligible, so the verbatim guarantee is preserved regardless of
    bias_strength.
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
    if bias is not None and bias_strength != 0.0:
        logits = logits + bias_strength * np.asarray(bias)[top_idx]
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    choice = rng.choice(len(top_idx), p=weights)
    return top_idx[choice]


def parse_payload(payload_str, word_to_idx):
    """Parse a "word:weight,word:weight,..." string into a dense payload
    vector over the vocabulary (unknown words are dropped with a warning).
    Weights can be negative to *penalize* a word instead of boosting it."""
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
                print(f"[warn] payload entry '{item}' has a non-numeric "
                      f"weight; skipping", file=sys.stderr)
                continue
        else:
            word, weight = item, 1.0

        word = word.strip().lower()
        idx = word_to_idx.get(word)
        if idx is None:
            print(f"[warn] payload word '{word}' is not in the vocabulary; "
                  f"skipping", file=sys.stderr)
            continue
        payload_vector[idx] = weight

    return payload_vector


class HECMModel:
    def __init__(self, tokens, order=3, min_count=1, beta=0.0, alpha=0.0,
                 max_diffusion_states=1500, payload=None):
        self.order = order
        vocab, word_to_idx, counts = build_vocab(tokens, min_count=min_count)
        self.vocab = vocab
        self.word_to_idx = word_to_idx

        contexts, context_to_idx, M, context_counts = build_ngram_contexts(
            tokens, word_to_idx, order=order
        )

        perm = log_sort_contexts(contexts, context_counts)
        contexts = [contexts[i] for i in perm]
        M = permute_matrix(M, perm)
        context_to_idx = {ctx: i for i, ctx in enumerate(contexts)}

        self.contexts = contexts
        self.context_to_idx = context_to_idx
        self.context_counts = context_counts

        self.P_raw = row_normalize(M)

        num_states = len(contexts)
        if alpha <= 0.0 or beta <= 0.0:
            # Verbatim mode: skip the (expensive, and by default unused)
            # diffusion step entirely and sample straight from the raw,
            # actually-observed n-gram probabilities.
            self.P_diffused = self.P_raw
            self.P_final = self.P_raw
        elif num_states > max_diffusion_states:
            print(
                f"[warn] {num_states} context-states exceeds "
                f"max_diffusion_states={max_diffusion_states}; skipping "
                f"matrix-exponential diffusion (O(n^3) cost) and using raw "
                f"n-gram probabilities only. Raise --max-diffusion-states "
                f"to override.",
                file=sys.stderr,
            )
            self.P_diffused = self.P_raw
            self.P_final = self.P_raw
        else:
            self.P_diffused = diffusion_weighting(self.P_raw, beta=beta)
            self.P_final = blend(self.P_raw, self.P_diffused, alpha=alpha)

        # --- Payload customizer -------------------------------------------
        # payload: dense vocab-length vector of user weights (from
        # parse_payload). We precompute, per context-state j:
        #   context_word_score[j]  = sum of payload weights over j's own
        #                            (order-1) words -- the "sub-trigram"
        #                            (context-only) part of the dot product.
        #   introduced_word_score[j] = payload weight of j's last word --
        #                            the word that gets newly emitted when
        #                            transitioning INTO state j.
        # The full "trigram payload dot product" for a transition
        # ctx_idx -> j is context_word_score[ctx_idx] + introduced_word_score[j].
        # The first term is constant across all candidates j reachable from
        # a given ctx_idx, so it cancels out of *that* step's softmax -- it
        # only matters when comparing candidates that have DIFFERENT
        # preceding context (i.e. when picking among fallback/restart
        # states), where it correctly favors thematically-relevant contexts.
        if payload is None:
            payload = np.zeros(len(vocab), dtype=np.float64)
        self.payload = payload
        self.context_word_score = np.array(
            [sum(payload[w] for w in ctx) for ctx in contexts], dtype=np.float64
        )
        self.introduced_word_score = np.array(
            [payload[ctx[-1]] for ctx in contexts], dtype=np.float64
        )
        self.context_total_score = self.context_word_score + 0.0  # alias for clarity

    def vocab_size(self):
        return len(self.vocab)

    def num_states(self):
        return len(self.contexts)

    def _tokenize_prompt(self, prompt):
        toks = re.findall(r"[a-z0-9']+|[.,!?;:]", prompt.lower())
        unk = self.word_to_idx[UNK]
        return [self.word_to_idx.get(t, unk) for t in toks]

    def _weighted_pick(self, candidates, boost_strength, rng):
        """Pick among a list of candidate state indices, softmax-weighted
        by boost_strength * (their own context-word payload score), falling
        back to uniform when boost_strength is 0 or all scores tie."""
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

        # Fallback 1: find any registered context sharing the longest
        # possible suffix with `desired`, ranked by payload score.
        for k in range(ctx_len - 1, 0, -1):
            suffix = desired[-k:]
            candidates = [
                i for i, ctx in enumerate(self.contexts) if ctx[-k:] == suffix
            ]
            if candidates:
                return self._weighted_pick(candidates, boost_strength, rng)

        # Fallback 2: pick among the most frequent contexts, weighted by payload.
        counts_arr = np.array(
            [self.context_counts.get(ctx, 0) for ctx in self.contexts]
        )
        top_candidates = list(np.argsort(counts_arr)[-10:])
        return self._weighted_pick(top_candidates, boost_strength, rng)

    def generate(self, prompt, max_new_tokens=40, top_k=10, base_temp=0.8,
                 amplitude=0.4, omega=0.5, phase=0.0, seed=None,
                 boost_strength=0.0):
        rng = np.random.default_rng(seed)
        prompt_ids = self._tokenize_prompt(prompt)

        ctx_idx = self._initial_context(prompt_ids, rng, boost_strength=boost_strength)
        out_ids = list(prompt_ids)

        for step in range(max_new_tokens):
            temp = sinusoidal_temperature(
                step, base_temp=base_temp, amplitude=amplitude,
                omega=omega, phase=phase,
            )
            probs = self.P_final[ctx_idx]
            next_ctx_idx = sample_next(
                probs, temperature=temp, top_k=top_k, rng=rng,
                bias=self.introduced_word_score, bias_strength=boost_strength,
            )

            if next_ctx_idx is None:
                # True dead end: this exact context never had any observed
                # continuation in training data. Restart from a fresh real
                # context (payload-weighted) rather than fabricate one.
                ctx_idx = self._initial_context(out_ids, rng, boost_strength=boost_strength)
                continue

            next_word_id = self.contexts[next_ctx_idx][-1]  # newly introduced word
            out_ids.append(next_word_id)
            ctx_idx = next_ctx_idx

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="HECM toy n-gram language model")
    ap.add_argument("--data", nargs="*", default=None,
                     help="Text file path(s) or glob pattern(s). Omit for a "
                          "built-in demo corpus.")
    ap.add_argument("--order", type=int, default=3,
                     help="n-gram order: 2=bigram, 3=trigram (default), "
                          "4=4-gram, etc. Context length = order - 1.")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.0,
                     help="Matrix-exponential diffusion strength. 0 (default) "
                          "= verbatim mode: sample strictly from actually-"
                          "observed n-gram transitions, no synthesis.")
    ap.add_argument("--alpha", type=float, default=0.0,
                     help="Blend factor: 0 (default)=raw/verbatim n-gram "
                          "probs only, 1=fully diffused probs.")
    ap.add_argument("--max-diffusion-states", type=int, default=1500,
                     help="Skip the O(n^3) diffusion step above this many "
                          "context-states (trigrams+ can have many contexts).")
    ap.add_argument("--prompt", type=str, default="the fox and")
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--base-temp", type=float, default=0.8)
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--boost", type=str, default=None,
                     help="Payload customizer: comma-separated 'word:weight' "
                          "pairs (weight optional, defaults to 1.0; use "
                          "negative weights to penalize a word), e.g. "
                          "'forest:2.0,dog:1.0,fire:-1.5'. At each step, "
                          "candidate trigrams are reranked by the dot "
                          "product of this payload against the trigram's "
                          "words -- never adds a transition that wasn't "
                          "already observed, so verbatim mode still holds.")
    ap.add_argument("--boost-strength", type=float, default=1.0,
                     help="Scales the payload dot-product bias before it's "
                          "added to sampling log-probabilities. 0 disables "
                          "the customizer even if --boost is set.")
    args = ap.parse_args()

    tokens = load_corpus(paths=args.data)
    print(f"[info] loaded {len(tokens)} tokens", file=sys.stderr)

    model = HECMModel(
        tokens, order=args.order, min_count=args.min_count,
        beta=args.beta, alpha=args.alpha,
        max_diffusion_states=args.max_diffusion_states,
    )
    print(f"[info] order={args.order} (context length {args.order-1}) | "
          f"vocab size: {model.vocab_size()} | context-states: {model.num_states()}",
          file=sys.stderr)

    # Build the payload after the model exists, so we can validate words
    # against its vocabulary and populate the model's payload-derived scores.
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
        nonzero = np.count_nonzero(payload)
        print(f"[info] payload customizer active: {nonzero} word(s) weighted, "
              f"boost-strength={args.boost_strength}", file=sys.stderr)
        boost_strength = args.boost_strength
    else:
        boost_strength = 0.0

    def run_once(prompt):
        result = model.generate(
            prompt,
            max_new_tokens=args.length,
            top_k=args.top_k,
            base_temp=args.base_temp,
            amplitude=args.amplitude,
            omega=args.omega,
            phase=args.phase,
            seed=args.seed,
            boost_strength=boost_strength,
        )
        print(result)

    if args.interactive:
        print("[info] interactive mode -- type a prompt, or 'quit' to exit", file=sys.stderr)
        while True:
            try:
                prompt = input("prompt> ")
            except EOFError:
                break
            if prompt.strip().lower() in ("quit", "exit"):
                break
            run_once(prompt)
    else:
        run_once(args.prompt)


if __name__ == "__main__":
    main()
