import re
import random
from functools import reduce

from typing import List, Tuple

def dedupe_by_start_end(sentences: List[str]) -> List[str]:
    """
    Keeps only sentences whose (first_token, last_token) pair is unique.
    If duplicates exist, later ones are discarded ("popped").
    """
    seen = set()
    result = []

    for s in sentences:
        tokens = tokenize(s)
        if not tokens:
            continue

        key = (tokens[0], tokens[-1])

        if key in seen:
            # "pop" behavior: skip duplicate sentence
            continue

        seen.add(key)
        result.append(s)

    return result
    
# =========================
# 🧩 PURE LAMBDA UTILITIES
# =========================

def tokenize(text):
    """Pure function: text → tokens"""
    return text.lower().split()


def World(world_id, text):
    """A Kripke world is a pure closure (no mutation)."""
    tokens = tokenize(text)
    weight = 1.0

    return lambda: {
        "id": world_id,
        "text": text,
        "tokens": tokens,
        "weight": weight
    }


# =========================
# 🌐 KRIPKE ACCESSIBILITY
# =========================

def accessible(world_a):
    """Kripke relation R(w1, w2) as lambda predicate"""
    ta = set(world_a()["tokens"])

    return lambda world_b: bool(
        ta & set(world_b()["tokens"])
    )


def neighborhood(world, worlds):
    """◇ operator: all accessible worlds"""
    R = accessible(world)
    return list(filter(R, worlds))


# =========================
# 🧠 PURE MARKOV MODEL
# =========================

def markov_from_worlds(worlds):
    """
    Builds a pure function:
    (w_i, w_{i+1}) → [possible w_{i+2}]
    """

    table = {}

    for w in worlds:
        tokens = w()["tokens"]

        for i in range(len(tokens) - 2):
            key = (tokens[i], tokens[i + 1])
            nxt = tokens[i + 2]

            if key not in table:
                table[key] = []
            table[key].append(nxt)

    return lambda pair: table.get(pair, [])


# =========================
# 🔁 Y-COMBINATOR (FIXPOINT)
# =========================

def Y(f):
    """Fixpoint combinator (enables recursion without mutation)"""
    return (lambda x: f(lambda *args: x(x)(*args)))(
        lambda x: f(lambda *args: x(x)(*args))
    )


# =========================
# 🔥 GENERATOR (PURE RECURSION)
# =========================

def generator(markov, max_words):

    def step(gen):

        def run(state, out):

            if len(out) >= max_words:
                return out

            a, b = state
            options = markov((a, b))

            if not options:
                return out

            nxt = random.choice(options)

            return gen((b, nxt), out + [nxt])

        return run

    return Y(step)


# =========================
# 🌍 ENTRY SELECTION (KRIPKE EVAL)
# =========================

def find_entry(prompt_tokens, worlds):
    best = worlds[0]
    best_score = -1

    pt = set(prompt_tokens)

    for w in worlds:
        score = len(pt & set(w()["tokens"]))
        if score > best_score:
            best = w
            best_score = score

    return best


# =========================
# 🧬 SYSTEM (PURE KRIPKE ENGINE)
# =========================

class LambdaKripkeSystem:

    def __init__(self, texts):
        self.worlds = [
            World(i, t) for i, t in enumerate(texts)
        ]

    # -------------------------
    # GENERATION PIPELINE
    # -------------------------
    def generate(self, prompt, max_words=50):

        ptoks = tokenize(prompt)

        start_world = find_entry(ptoks, self.worlds)

        # Kripke neighborhood (◇W)
        neighbors = neighborhood(start_world, self.worlds)

        if not neighbors:
            neighbors = self.worlds

        markov = markov_from_worlds(neighbors)
        gen = generator(markov, max_words)

        tokens = start_world()["tokens"]

        if len(tokens) < 2:
            return start_world()["text"]

        seed = (tokens[0], tokens[1])

        output_tokens = gen(seed, list(seed))

        return " ".join(output_tokens)


# =========================
# 🧪 DEMO EXECUTION
# =========================
def load_txt_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    sentences = [ln.strip() for ln in raw.split(".") if ln.strip()]
    return dedupe_by_start_end(sentences)

if __name__ == "__main__":

    corpus = load_txt_file(input("Filename: "))

    system = LambdaKripkeSystem(corpus)
    while True:
        prompt = input("USER: ")
        print("\nGENERATED:\n")
        print(system.generate(prompt, max_words=4000))
