#!/usr/bin/env python3
"""Markov text model with prompt completion, using the generation
techniques distilled from the neural-circuit simulation.

  python markov_gen.py data.txt --prompt "once upon a" --n 3
  python markov_gen.py data.txt --prompt "once upon a" --target-entropy 2.0 --anchor 2.0
  python markov_gen.py data.txt --prompt "once upon a" --beam 8
  python markov_gen.py data.txt --calibrate --diagnose --prompt "once upon a"
  python markov_gen.py data.txt --interactive

Techniques (each is off by default, so the plain sampler still works):
  --target-entropy H   hold per-step entropy near H by solving for temperature
                       (avoids die-out/repetition vs runaway incoherence)
  --anchor A           leaky logit bias on the prompt's content words, decaying
                       every step and re-pulsed every --anchor-interval steps
  --beam W             beam search: heapq pruning, linked-list paths, exact
                       Markov-state dedup, length-normalised final ranking
  --calibrate          find the smallest beam width matching a wide-beam reference
  --diagnose           measure whether the anchor really raises prompt-word recurrence
  --base-clip P        clip the unigram backoff at the P-th percentile of counts
Always on: generation stops at end-of-text, and max_len is raised to at least
the shortest observed path to end-of-text (+ --slack).
"""
import argparse
import heapq
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace

import numpy as np

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", ""
TOKEN_RE = re.compile(r"[a-z0-9']+|[.,!?;:]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORD_RE = re.compile(r"[a-z0-9']")


def detok(tokens):
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))


def load_sentences(path):
    text = open(path, encoding="utf-8").read().lower()
    sents = [TOKEN_RE.findall(c) for c in SENT_SPLIT.split(text)]
    return [s for s in sents if len(s) >= 2]


class Vocab:
    def __init__(self, sents, min_count=1):
        counts = Counter(t for s in sents for t in s)
        kept = sorted((t for t, c in counts.items() if c >= min_count),
                      key=lambda t: (-counts[t], t))
        self.itos = [PAD, UNK, BOS, EOS] + kept
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.V = len(self.itos)
        self.pad, self.unk, self.bos, self.eos = range(4)
        self.counts = np.zeros(self.V)
        for t in kept:
            self.counts[self.stoi[t]] = counts[t]

    def encode(self, toks):
        return [self.stoi.get(t, self.unk) for t in toks]


class NGram:
    """Interpolated n-gram; order = number of previous tokens used."""

    def __init__(self, vocab, order=2, lam=0.7, k=0.01, base_clip=100.0):
        self.v, self.order, self.lam, self.k = vocab, order, lam, k
        self.base_clip = base_clip
        self.tables = [defaultdict(Counter) for _ in range(order + 1)]

    def fit(self, seqs):
        for seq in seqs:
            for t in range(1, len(seq)):
                for n in range(min(self.order, t) + 1):
                    self.tables[n][tuple(seq[t - n:t])][seq[t]] += 1
        base = self.v.counts.copy()
        base[self.v.eos] = len(seqs)
        if self.base_clip < 100:  # robust normalisation: stop a few giant counts dominating
            pos = base[base > 0]
            base = np.minimum(base, np.percentile(pos, self.base_clip))
        base += self.k
        base[[self.v.pad, self.v.bos, self.v.unk]] = 0.0
        self.base = base / base.sum()

    def dist(self, ctx):
        p = self.base
        ctx = tuple(ctx[-self.order:]) if self.order else ()
        for n in range(len(ctx) + 1):  # shortest -> longest context
            c = self.tables[n].get(ctx[len(ctx) - n:] if n else ())
            if c:
                vec = np.zeros(self.v.V)
                for tok, cnt in c.items():
                    vec[tok] = cnt
                p = self.lam * vec / vec.sum() + (1 - self.lam) * p
        return p

    def logits(self, ctx):
        """Log-probabilities as logits, specials masked. Fresh array: safe to edit."""
        lg = np.log(np.maximum(self.dist(ctx), 1e-300))
        lg[[self.v.pad, self.v.bos, self.v.unk]] = -np.inf
        return lg


# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
@dataclass
class Gen:
    max_len: int = 40
    temp: float = 0.9
    top_k: int = 40
    target_entropy: float = 0.0   # >0 -> entropy-targeted temperature (overrides temp)
    beam: int = 0                 # >0 -> beam search instead of sampling
    min_len: int = 0
    length_alpha: float = 0.7     # beam final ranking: score / len**alpha
    anchor_amp: float = 0.0       # >0 -> leaky anchor on prompt content words
    anchor_decay: float = 0.9
    anchor_interval: int = 16
    slack: int = 20


# ---------------------------------------------------------------------------
# Criticality control: pick the temperature that yields a target entropy
# ---------------------------------------------------------------------------
def entropy(logits, temp):
    z = logits / temp
    p = np.exp(z - z.max())
    p /= p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def temperature_for_entropy(logits, target, lo=0.05, hi=3.0, iters=24):
    """Bisection; entropy is monotone increasing in temperature."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if entropy(logits, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def top_k_mask(lg, k):
    if 0 < k < len(lg):
        cut = np.partition(lg, -k)[-k]
        lg = np.where(lg < cut, -np.inf, lg)
    return lg


def pick(lg, gen):
    lg = top_k_mask(lg, gen.top_k)
    t = (temperature_for_entropy(lg, gen.target_entropy)
         if gen.target_entropy > 0 else max(gen.temp, 1e-3))
    z = lg / t
    p = np.exp(z - z.max())
    p /= p.sum()
    return int(np.random.choice(len(p), p=p)), t


# ---------------------------------------------------------------------------
# Leaky anchor with periodic re-injection (precomputed so beams share it)
# ---------------------------------------------------------------------------
class AnchorSchedule:
    def __init__(self, amp, decay, interval, n):
        level, self.levels = amp, []
        for t in range(n):
            self.levels.append(level)
            level *= decay                       # leak
            if (t + 1) % interval == 0:           # periodic pulse
                level += amp

    def level(self, t):
        return self.levels[min(t, len(self.levels) - 1)]


def anchor_ids(vocab, toks, skip_top=20):
    """Prompt content words: alphanumeric, in vocab, not among the most frequent tokens."""
    common = set(np.argsort(-vocab.counts)[:skip_top].tolist())
    return sorted({vocab.stoi[t] for t in toks
                   if t in vocab.stoi and WORD_RE.match(t) and vocab.stoi[t] not in common})


# ---------------------------------------------------------------------------
# Beam search: heapq pruning + linked-list paths + exact state dedup
# ---------------------------------------------------------------------------
def materialize(node):
    out = []
    while node is not None:
        tok, node = node
        out.append(tok)
    return out[::-1]


def log_softmax(lg):
    m = lg.max()
    return lg - (m + np.log(np.exp(lg - m).sum()))


def beam_search(step_fn, ctx0, eos, max_new, width, top_k, alpha, order):
    path0 = None
    for t in ctx0:
        path0 = (t, path0)
    state_len = max(order, 1)  # a Markov model's future depends only on this tail
    beam, done = [(0.0, path0)], []
    for step in range(max_new):
        cands = {}
        for score, path in beam:
            seq = materialize(path)
            logp = log_softmax(step_fn(seq))
            k = min(top_k if top_k > 0 else 16, len(logp) - 1)
            for tok in np.argpartition(-logp, k)[:k]:
                if not np.isfinite(logp[tok]):
                    continue
                tok = int(tok)
                s = score + float(logp[tok])
                newp = (tok, path)
                if tok == eos:
                    done.append((s / (step + 1) ** alpha, newp))
                    continue
                tail = tuple(seq[-(state_len - 1):] + [tok]) if state_len > 1 else (tok,)
                if tail not in cands or s > cands[tail][0]:
                    cands[tail] = (s, newp)
        if not cands or len(done) >= width:
            break
        beam = heapq.nlargest(width, cands.values(), key=lambda c: c[0])
    pool = done or [(s / max_new ** alpha, p) for s, p in beam]
    out = materialize(max(pool, key=lambda c: c[0])[1])[len(ctx0):]
    return out[:-1] if out and out[-1] == eos else out


def calibrate_beam(model, vocab, sents, widths=(1, 2, 4, 8, 16, 32), ref=64,
                   n_prompts=6, max_len=25, top_k=20):
    """Smallest width whose outputs match a wide-beam reference on sentence starts."""
    rng = random.Random(0)
    starts = [s[:2] for s in rng.sample(sents, min(n_prompts, len(sents)))]

    def run(w):
        g = Gen(max_len=max_len, top_k=top_k, beam=w)
        return [tuple(generate(model, vocab, s, g)[0]) for s in starts]

    reference = run(ref)
    for w in widths:
        if run(w) == reference:
            return w
    return ref


# ---------------------------------------------------------------------------
# Feasibility: shortest observed path from the prompt's last token to end-of-text
# ---------------------------------------------------------------------------
def min_steps_to_eos(model, vocab, start):
    if model.order < 1:
        return None
    graph = model.tables[1]
    seen, q = {start}, deque([(start, 0)])
    while q:
        s, d = q.popleft()
        nxt = graph.get((s,))
        if not nxt:
            continue
        if vocab.eos in nxt:
            return d + 1
        for n in nxt:
            if n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(model, vocab, toks, gen):
    """Continue token list `toks`. Returns (generated ids, per-step temperatures)."""
    ctx0 = [vocab.bos] + vocab.encode(toks)
    anchors = anchor_ids(vocab, toks) if gen.anchor_amp > 0 else []
    sched = (AnchorSchedule(gen.anchor_amp, gen.anchor_decay,
                            gen.anchor_interval, gen.max_len + 1) if anchors else None)

    def step(seq):
        t = len(seq) - len(ctx0)
        lg = model.logits(seq)
        if t < gen.min_len:
            lg[vocab.eos] = -np.inf
        if sched:
            lg[anchors] += sched.level(t)
        return lg

    if gen.beam > 0:
        return beam_search(step, ctx0, vocab.eos, gen.max_len, gen.beam,
                           gen.top_k, gen.length_alpha, model.order), []
    ctx, out, temps = list(ctx0), [], []
    for _ in range(gen.max_len):
        nxt, t = pick(step(ctx), gen)
        if nxt == vocab.eos:
            break
        out.append(nxt)
        ctx.append(nxt)
        temps.append(t)
    return out, temps


def complete(model, vocab, prompt, gen):
    """Continue a prompt. Returns (text, oov_words, note)."""
    toks = TOKEN_RE.findall(prompt.lower())
    if not toks:
        raise ValueError("prompt has no usable tokens")
    ids = vocab.encode(toks)
    oov = [t for t, i in zip(toks, ids) if i == vocab.unk]
    d = min_steps_to_eos(model, vocab, ids[-1])
    note = None
    if d is None:
        note = "no observed path to end-of-text from the last prompt token"
    else:
        gen = replace(gen, max_len=max(gen.max_len, d + gen.slack))
    out, _ = generate(model, vocab, toks, gen)
    return detok(toks + [vocab.itos[i] for i in out]), oov, note


def diagnose(model, vocab, prompt, gen, n=30):
    """Is the anchor doing anything real? Compare prompt-word recurrence on vs off."""
    toks = TOKEN_RE.findall(prompt.lower())
    anchors = set(anchor_ids(vocab, toks))
    if not anchors:
        print("  diagnose: prompt has no content words to anchor on")
        return
    base = replace(gen, anchor_amp=0.0, beam=0)
    on = replace(gen, anchor_amp=gen.anchor_amp or 2.0, beam=0)
    rates = {}
    for label, g in (("anchor off", base), ("anchor on ", on)):
        hits = total = 0
        temps = []
        for _ in range(n):
            out, t = generate(model, vocab, toks, g)
            hits += sum(o in anchors for o in out)
            total += len(out)
            temps += t
        rates[label] = hits / max(total, 1)
        mt = f"{np.mean(temps):.2f}" if temps else "n/a"
        print(f"  {label}: anchor-word rate {rates[label]:.4f}  "
              f"mean len {total / n:.1f}  mean T {mt}")
    if rates["anchor on "] <= rates["anchor off"]:
        print("  warning: anchor did not raise recurrence; the signal is decorative here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--n", type=int, default=1, help="completions per prompt")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--min-len", type=int, default=0)
    ap.add_argument("--target-entropy", type=float, default=0.0)
    ap.add_argument("--anchor", type=float, default=0.0, help="anchor amplitude")
    ap.add_argument("--anchor-decay", type=float, default=0.9)
    ap.add_argument("--anchor-interval", type=int, default=16)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--length-alpha", type=float, default=0.7)
    ap.add_argument("--slack", type=int, default=20)
    ap.add_argument("--base-clip", type=float, default=100.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sents = load_sentences(args.path)
    vocab = Vocab(sents, args.min_count)
    model = NGram(vocab, args.order, base_clip=args.base_clip)
    model.fit([[vocab.bos] + vocab.encode(s) + [vocab.eos] for s in sents])

    gen = Gen(max_len=args.max_len, temp=args.temp, top_k=args.top_k,
              target_entropy=args.target_entropy, beam=args.beam,
              min_len=args.min_len, length_alpha=args.length_alpha,
              anchor_amp=args.anchor, anchor_decay=args.anchor_decay,
              anchor_interval=args.anchor_interval, slack=args.slack)

    if args.calibrate:
        w = calibrate_beam(model, vocab, sents)
        print(f"calibrated beam width: {w} (matches a width-64 reference); use --beam {w}")

    def run(prompt):
        oov, note = set(), None
        for _ in range(args.n):
            text, o, note = complete(model, vocab, prompt, gen)
            oov.update(o)
            print(" ", text)
        if note:
            print(f"  ({note})")
        if oov:
            print(f"  (not in vocab, treated as <unk>: {sorted(oov)})")
        if args.diagnose:
            diagnose(model, vocab, prompt, gen)

    if args.prompt:
        run(args.prompt)
    if args.interactive or not (args.prompt or args.calibrate):
        print("Type a prompt (empty line to quit).")
        while True:
            try:
                p = input("> ").strip()
            except EOFError:
                break
            if not p:
                break
            try:
                run(p)
            except ValueError as e:
                print(" ", e)


if __name__ == "__main__":
    main()
