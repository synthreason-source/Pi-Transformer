"""
Cyclic Semantic Evidence Register (CSER) — Free-Manipulation Variant
=====================================================================

R(phi_i) = E_{i mod n}

Instead of a static bijection between hypothesis-slot i and evidence-slot
(i mod n), this implementation treats the binding as LIVE: every cycle,
hypotheses and their evidence may be SORTED (ranked by confidence),
EXCHANGED (a falsified slot is handed to an untested hypothesis), and
merged under EQUIVALENCE (duplicates collapse, shrinking n).

The cyclic index itself never breaks — n is just re-read every cycle,
so wrap-around (i mod n) stays correct even as n shrinks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import itertools


# ----------------------------------------------------------------------
# Evidence states
# ----------------------------------------------------------------------

class Evidence(Enum):
    TRUE = "⊤"       # Supported
    FALSE = "⊥"      # Contradicted
    UNKNOWN = "?"     # Insufficient

    def __repr__(self):
        return self.value


def reconcile(prev: Evidence, new: Evidence) -> Evidence:
    """Combine old and new evidence for the same slot.
    A confirmed TRUE or FALSE is 'sticky' unless directly contradicted;
    UNKNOWN never overrides a settled state."""
    if new is Evidence.UNKNOWN:
        return prev
    if prev is Evidence.UNKNOWN:
        return new
    if prev != new:
        # direct contradiction: most recent observation wins,
        # but this is where you'd log/flag it
        return new
    return new


def merge(a: Evidence, b: Evidence) -> Evidence:
    """⊕ operator for collapsing equivalent hypotheses' evidence.
    TRUE dominates FALSE dominates UNKNOWN (most-informative wins,
    with TRUE prioritized as the stronger claim)."""
    order = {Evidence.TRUE: 2, Evidence.FALSE: 1, Evidence.UNKNOWN: 0}
    return a if order[a] >= order[b] else b


def confidence(e: Evidence) -> int:
    """Ranking key for SORT — higher is 'more resolved / more supported'."""
    return {Evidence.TRUE: 2, Evidence.UNKNOWN: 1, Evidence.FALSE: 0}[e]


# ----------------------------------------------------------------------
# Hypothesis
# ----------------------------------------------------------------------

@dataclass
class Hypothesis:
    name: str
    payload: object = None  # arbitrary domain data (a claim, a dream-element, etc.)

    def __repr__(self):
        return f"φ({self.name})"


# ----------------------------------------------------------------------
# The register
# ----------------------------------------------------------------------

@dataclass
class CSER:
    hypotheses: list[Hypothesis]
    evidence: list[Evidence] = field(default_factory=list)
    evaluate: Callable[[Hypothesis, object], Evidence] = None
    equivalent: Callable[[Hypothesis, Hypothesis], bool] = None
    collapse: Callable[[Hypothesis, Hypothesis], Hypothesis] = None
    log: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.evidence:
            self.evidence = [Evidence.UNKNOWN] * len(self.hypotheses)
        assert len(self.hypotheses) == len(self.evidence)

    @property
    def n(self) -> int:
        return len(self.hypotheses)

    # -- the three manipulation primitives -----------------------------

    def sort(self):
        """Rank hypotheses/evidence pairs by confidence, most-resolved first."""
        order = sorted(range(self.n), key=lambda idx: -confidence(self.evidence[idx]))
        self.hypotheses = [self.hypotheses[i] for i in order]
        self.evidence = [self.evidence[i] for i in order]

    def exchange(self, k: int):
        """If slot k was just falsified, hand its evidence-slot to an
        untested hypothesis so testing capacity isn't wasted on dead ends."""
        if self.evidence[k] is not Evidence.FALSE:
            return
        untested = [j for j in range(self.n) if j != k and self.evidence[j] is Evidence.UNKNOWN]
        if not untested:
            return
        j = untested[0]
        self.hypotheses[k], self.hypotheses[j] = self.hypotheses[j], self.hypotheses[k]
        self.evidence[k], self.evidence[j] = self.evidence[j], self.evidence[k]
        self.log.append(f"EXCHANGE: swapped slot {k} <-> {j}")

    def collapse_equivalents(self, k: int):
        """Merge any hypothesis equivalent to Φ[k] into it, shrinking n."""
        if self.equivalent is None:
            return
        i = 0
        while i < self.n:
            if i != k and i < self.n and k < self.n and \
               self.equivalent(self.hypotheses[k], self.hypotheses[i]):
                merged = self.collapse(self.hypotheses[k], self.hypotheses[i]) \
                    if self.collapse else self.hypotheses[k]
                merged_evidence = merge(self.evidence[k], self.evidence[i])
                self.log.append(
                    f"EQUIVALENCE: collapsed {self.hypotheses[i]} into {self.hypotheses[k]}"
                )
                # remove i, keep k updated
                del self.hypotheses[i]
                del self.evidence[i]
                if i < k:
                    k -= 1
                self.hypotheses[k] = merged
                self.evidence[k] = merged_evidence
            else:
                i += 1

    # -- main loop -------------------------------------------------------

    def run(self, observations: list[object], max_cycles: Optional[int] = None):
        obs_cycle = itertools.cycle(observations) if observations else itertools.repeat(None)
        max_cycles = max_cycles or (len(observations) * 3 if observations else self.n * 3)

        i = 0
        stale_streak = 0
        while i < max_cycles and Evidence.UNKNOWN in self.evidence:
            k = i % self.n                      # cyclic index, n re-read each pass
            phi = self.hypotheses[k]
            e_prev = self.evidence[k]

            o = next(obs_cycle)
            e_new = self.evaluate(phi, o) if self.evaluate else Evidence.UNKNOWN

            self.evidence[k] = reconcile(e_prev, e_new)
            changed = self.evidence[k] != e_prev
            self.log.append(f"[cycle {i}] slot {k} {phi}: {e_prev.value} -> {self.evidence[k].value}")

            # --- free manipulation, applied with full freedom ---
            self.sort()
            k = self.hypotheses.index(phi)      # re-locate phi after sort
            self.exchange(k)
            self.collapse_equivalents(k)

            # convergence check: stop once a full pass over all n slots
            # produces no change (remaining '?' are simply unresolvable
            # given the observations available)
            stale_streak = 0 if changed else stale_streak + 1
            if stale_streak >= self.n:
                self.log.append(f"[converged] no changes for a full pass ({self.n} slots); stopping early")
                break

            i += 1

        return self.hypotheses, self.evidence

    def report(self):
        lines = ["Cyclic Semantic Evidence Register — final state", "-" * 48]
        for phi, e in zip(self.hypotheses, self.evidence):
            lines.append(f"  {phi!r:20} : {e.value}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Semantic evaluator (local embedding model)
# ----------------------------------------------------------------------

class SemanticEvaluator:
    """
    Judges support/contradiction using local sentence embeddings instead of
    literal substring matching, so paraphrased evidence ("I lost my teeth")
    can still confirm a hypothesis named "teeth loss".

    Uses sentence-transformers (all-MiniLM-L6-v2 by default). The model
    weights download once from Hugging Face on first use, then run fully
    offline/locally afterwards.

        pip install sentence-transformers --break-system-packages

    support_threshold: cosine similarity above which an observation is
    considered to be "about" the hypothesis at all (below it -> UNKNOWN,
    i.e. irrelevant/no signal). Tune per corpus; 0.35-0.5 is a reasonable
    starting range for MiniLM.
    """

    NEGATORS = ("not ", "n't", "never", " no ", "without", "isn't",
                "wasn't", "didn't", "hasn't", "haven't")

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 support_threshold: float = 0.10, model=None):
        from sentence_transformers import SentenceTransformer  # local import: optional dep
        self.model = model or SentenceTransformer(model_name)
        self.support_threshold = support_threshold
        self._hyp_cache: dict[str, object] = {}
        self._obs_cache: dict[str, object] = {}

    def _embed(self, text: str, cache: dict):
        if text not in cache:
            cache[text] = self.model.encode(text, normalize_embeddings=True)
        return cache[text]

    def __call__(self, phi: "Hypothesis", obs: Optional[str]) -> Evidence:
        if not obs:
            return Evidence.UNKNOWN

        # Embed the hypothesis as a short natural-language claim, not just
        # the bare keyword -- embeddings work far better on full phrases.
        hyp_text = phi.payload if isinstance(phi.payload, str) else f"I dreamed about {phi.name}"
        hyp_vec = self._embed(hyp_text, self._hyp_cache)
        obs_vec = self._embed(obs.strip(), self._obs_cache)

        similarity = float(hyp_vec @ obs_vec)  # cosine sim (vectors are normalized)

        if similarity < self.support_threshold:
            return Evidence.UNKNOWN  # not topically related enough to count as evidence

        negated = any(neg in f" {obs.strip().lower()} " for neg in self.NEGATORS)
        return Evidence.FALSE if negated else Evidence.TRUE


def keyword_evaluate(phi: "Hypothesis", obs: Optional[str]) -> Evidence:
    """Fallback evaluator: literal substring + localized negation check.
    Used automatically if sentence-transformers isn't installed."""
    NEGATORS = ("not ", "n't", "never", "no ", "without")
    if not obs:
        return Evidence.UNKNOWN
    text = obs.strip().lower()
    key = phi.name.strip().lower()
    if key not in text:
        return Evidence.UNKNOWN
    idx = text.find(key)
    window = text[max(0, idx - 20):idx]
    negated = any(neg in window for neg in NEGATORS)
    return Evidence.FALSE if negated else Evidence.TRUE


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    filename = input("Filename (blank = built-in demo text): ").strip()
    if filename:
        with open(filename, "r", encoding="utf-8") as f:
            observations = [s.strip() for s in f.read().split(".") if s.strip()]
    else:
        observations = [
            "I dreamed of falling last night",
            "I did not fly today",
            "I lost several teeth in the dream",
            "something was chasing me through a hallway",
        ]

    hyps = [
        Hypothesis("falling"),
        Hypothesis("flying"),
        Hypothesis("teeth loss"),
        Hypothesis("being chased"),
    ]
    while True:
        custom = input("Custom hypothesis (leave blank to skip): ").strip()
        if custom:
            hyps.append(Hypothesis(custom))
        if not custom:
            break
            
    def equivalent(a: Hypothesis, b: Hypothesis) -> bool:
        key_a = a.payload if a.payload else a.name
        key_b = b.payload if b.payload else b.name
        return a is not b and key_a == key_b

    def collapse(a: Hypothesis, b: Hypothesis) -> Hypothesis:
        return a  # keep the earlier one, arbitrarily

    try:
        evaluate = SemanticEvaluator()
        print("Using local sentence-embedding model for evaluation.\n")
    except ImportError:
        evaluate = keyword_evaluate
        print("sentence-transformers not installed -- falling back to keyword matching.")
        print("Install with: pip install sentence-transformers --break-system-packages\n")

    cser = CSER(hypotheses=hyps, evaluate=evaluate, equivalent=equivalent, collapse=collapse)
    cser.run(observations, max_cycles=min(20000, len(observations) * 50))

    print("\n".join(cser.log))
    print()
    print(cser.report())
