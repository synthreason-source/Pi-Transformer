"""
Psychology-concept transition beam search.

Same engineering pattern as the numeric equation-driven beam search this
was adapted from:

  - PATH REPRESENTATION: paths are an immutable linked list ("cons cell":
    (state, parent_node)). Extending a path is O(1) -- "stay" reuses the
    same node reference, "move" allocates exactly one new node. The full
    path is only materialized at the end, by walking the parent chain.

  - BEAM WIDTH CAP: width at step k follows the same log-space formula as
    before, hard-capped at max_beam_width regardless of what the formula
    would theoretically allow -- the actual, enforceable memory bound.

  - heapq.nsmallest instead of a full sort, O(m log width) instead of
    O(m log m), to keep only the width-best candidates each step.

This applies the same circular adjacency rule to psychology concepts:
neighboring concepts are disallowed, while non-neighboring concepts remain valid.
The rule is a structural constraint for your model rather than a claim that those
psychological concepts cannot coexist in real life; adjacent emotions in Plutchik’s
original model are actually commonly treated as blendable dyads.



"""

import math
import heapq

# ---------------------------------------------------------------------
# 1. THE CONCEPTS -- Plutchik's wheel of emotions (eight primary
#    emotions), arranged in wheel order. Emotions adjacent on the wheel
#    are considered psychologically "close" -- a plausible direct
#    transition. This is a real, textbook model (Plutchik, 1980), not an
#    invented one.
# ---------------------------------------------------------------------
WHEEL = [
    "Emotion",
    "Feeling",
    "Mood",
    "Affect",
    "Attention",
    "Perception",
    "Memory",
    "Learning",
    "Reasoning",
    "Motivation",
    "Behavior",
    "Identity",
    "Empathy",
    "Attachment",
    "Stress",
    "Anxiety",
    "Trauma",
    "Resilience",
]
N_STATES = len(WHEEL)
INDEX = {name: i for i, name in enumerate(WHEEL)}

# ---------------------------------------------------------------------
# 2. KNOWN IMPOSSIBLE TRANSITIONS
#    Plutchik's model pairs each emotion with a polar opposite directly
#    across the wheel (4 positions away, since there are 8 states):
#    Joy <-> Sadness, Trust <-> Disgust, Fear <-> Anger,
#    Anticipation <-> Surprise. The theory holds that you do not swing
#    straight from one to its opposite in a single step -- something
#    intermediate has to happen first. Encoded as a hard-forbidden edge
#    set, checked before a candidate is added to the beam.
# ---------------------------------------------------------------------
OPPOSITE_OFFSET = N_STATES // 2  # 4 positions apart = opposite on the wheel
IMPOSSIBLE = set()
for i, name in enumerate(WHEEL):
    opposite = WHEEL[(i + OPPOSITE_OFFSET) % N_STATES]
    IMPOSSIBLE.add((name, opposite))
    IMPOSSIBLE.add((opposite, name))


def neighbors(state):
    """Adjacent wheel states plus 'stay' -- the plausible one-step moves."""
    i = INDEX[state]
    return [state, WHEEL[(i - 1) % N_STATES], WHEEL[(i + 1) % N_STATES]]


def is_impossible(a, b):
    """O(1) check against the known-impossible transition table."""
    return (a, b) in IMPOSSIBLE


def _materialize(node):
    """Walk an immutable path linked-list back to a plain Python list."""
    out = []
    while node is not None:
        state, node = node
        out.append(state)
    out.reverse()
    return out


def transition_beam_search(start, target, max_steps, r, c=2,
                            max_beam_width=200, n_results=5, paths_per_state=3):
    """
    Beam search over psychological-state transitions from `start` to
    `target`, avoiding known-impossible direct transitions.

    beam: list of (state, path_node)  where path_node is None or
          (state, parent) -- the immutable cons-list.

    Unlike the numeric solver (which dedupes to exactly one path per
    partial sum), this keeps up to `paths_per_state` distinct paths per
    state, since the point here is to surface MULTIPLE plausible
    transition sequences, not just the first one found.

    Returns (list_of_paths, steps_used).
    """
    theoretical_cap = N_STATES ** c  # kept only for reporting; not the real cap
    cap = min(theoretical_cap, max_beam_width)

    beam = [(start, (start, None))]  # root cons-cell encodes the start state itself
    completed = []

    for k in range(1, max_steps + 1):
        # seen: state -> list of distinct path_nodes reaching that state
        seen = {}
        for state, node in beam:
            for nxt in neighbors(state):
                if nxt != state and is_impossible(state, nxt):
                    continue  # <-- stepwise filter: drop known-impossible jumps
                new_node = node if nxt == state else (nxt, node)
                bucket = seen.setdefault(nxt, [])
                if len(bucket) < paths_per_state and new_node not in bucket:
                    bucket.append(new_node)

        if target in seen:
            for node in seen[target]:
                path = _materialize(node)
                if path not in completed:
                    completed.append(path)
            if len(completed) >= n_results:
                return completed[:n_results], k

        # ---- same log-space width formula, same hard cap ----
        log_width = N_STATES * math.log(2) + k * math.log(1 - r)
        if log_width > math.log(cap):
            width = cap
        else:
            width = max(1, min(cap, math.ceil(math.exp(log_width))))

        # flatten (state, node) pairs for selection, then pick width-best
        # by wheel-distance to target using heapq.nsmallest -- O(m log width)
        flat = [(s, node) for s, nodes in seen.items() for node in nodes]
        if len(flat) <= width:
            beam = flat
        else:
            target_idx = INDEX[target]

            def wheel_distance(kv):
                d = abs(INDEX[kv[0]] - target_idx)
                return min(d, N_STATES - d)

            beam = heapq.nsmallest(width, flat, key=wheel_distance)

    return completed, max_steps


def run(label, start, target, max_steps=1000000, r=0.15, c=2,
        max_beam_width=200, n_results=5):
    print(f"=== {label}  (start={start}, target={target}, r={r}, "
          f"c={c}, max_beam_width={max_beam_width}) ===")
    paths, steps_used = transition_beam_search(
        start, target, max_steps, r, c, max_beam_width, n_results
    )
    if paths:
        print(f"found {len(paths)} distinct transition path(s) in <= {steps_used} steps:")
        for p in paths:
            print("  " + " -> ".join(p))
    else:
        print(f"FAILED to find a path within {max_steps} steps")
    print()


if __name__ == "__main__":
    # Sadness -> Joy is exactly an opposite pair, so the direct edge is
    # blocked by IMPOSSIBLE. The search has to route through intermediate
    # states (e.g. via Fear/Trust or Disgust/Anger) instead.


    # Anticipation -> Surprise is also an opposite pair.
    run("Route",
        "Reasoning", "Trauma")
