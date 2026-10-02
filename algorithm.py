import math
import random
import re
import textwrap
from collections import Counter
from itertools import zip_longest

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# gensim provides the GloVe vectors. If it is missing, the program falls
# back to plain prompt-word biasing on the full graph.
try:
    from gensim.models import KeyedVectors
    import gensim.downloader as gensim_api
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


# ----------------------------------------------------------------------
# ORIGINAL SOLVER KEPT EXACTLY THE SAME
# ----------------------------------------------------------------------

class PCMMaxCut:
    def __init__(self, graph, noise=0.02, drift_nu=0.08):
        self.graph = graph
        self.n = len(graph)
        self.noise = noise
        self.drift_nu = drift_nu
        self.spins = [random.choice([-1, 1]) for _ in range(self.n)]
        self.age = 1

    def local_field(self, i):
        return sum(
            weight * self.spins[j]
            for j, weight in self.graph[i].items()
        )

    def delta_energy(self, i):
        return -2 * self.spins[i] * self.local_field(i)

    def energy(self):
        value = 0.0
        for i in range(self.n):
            for j, weight in self.graph[i].items():
                if j > i:
                    value += weight * self.spins[i] * self.spins[j]
        return value

    def cut_size(self):
        total = 0
        for i in range(self.n):
            for j, weight in self.graph[i].items():
                if j > i and self.spins[i] != self.spins[j]:
                    total += weight
        return total

    def pcm_flip_attempt(self, temperature):
        i = random.randrange(self.n)
        delta = self.delta_energy(i)

        drift = self.age ** (-self.drift_nu)
        thermal_noise = random.gauss(
            0.0,
            self.noise
        ) * max(1.0, abs(self.local_field(i)))

        effective_delta = delta + thermal_noise * drift

        if effective_delta <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(
                -effective_delta / max(temperature, 1e-9)
            )

        if accept:
            self.spins[i] *= -1
            return i, True, delta

        return i, False, delta

    def run(self, iterations=1000, temperature=1.0, cooling=0.995):
        history = []

        for _ in range(iterations):
            _, accepted, _ = self.pcm_flip_attempt(temperature)

            temperature *= cooling
            self.age += 1

            history.append({
                "energy": self.energy(),
                "cut": self.cut_size(),
                "accepted": accepted,
                "temperature": temperature,
                "spins": self.spins[:],
            })

        return history


# ----------------------------------------------------------------------
# TEXT HELPERS
# ----------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z']+")


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def clean_word(word):
    return re.sub(r"[^a-z']", "", word.lower())


def content_tokens(words):
    """Cleaned, non-stopword tokens."""
    cleaned = (clean_word(w) for w in words)
    return [w for w in cleaned if w and w not in ENGLISH_STOP_WORDS]


# ----------------------------------------------------------------------
# GLOVE PROMPT SLICER (probes the graph directly, no corpus segments)
# ----------------------------------------------------------------------

class GloveSlicer:
    """
    Slices a prompt into word windows, embeds each slice with GloVe, and
    measures the Manhattan (L1) distance from the slice to every bigram
    node of the dataset graph. The nearest nodes are the peaks, which seed
    generation. Each slice's vocabulary is expanded with GloVe neighbours
    that exist in the dataset.
    """

    def __init__(self, glove_name="glove-wiki-gigaword-100", glove_path=None):
        if glove_path:
            print(f"Loading GloVe from {glove_path}")
            self.kv = KeyedVectors.load_word2vec_format(
                glove_path, binary=False, no_header=True
            )
        else:
            print(f"Loading GloVe: {glove_name} (cached after first download)")
            self.kv = gensim_api.load(glove_name)

        self.dim = self.kv.vector_size
        self.node_matrix = None
        self.node_valid = None

    # -- embedding -----------------------------------------------------

    def embed(self, tokens):
        vecs = [self.kv[t] for t in tokens if t in self.kv]
        if not vecs:
            return None
        v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(v)
        return (v / norm).astype(np.float32) if norm > 0 else None

    def index_nodes(self, id_to_node):
        """Embed every bigram node (content words only)."""
        n = len(id_to_node)
        self.node_matrix = np.zeros((n, self.dim), dtype=np.float32)
        self.node_valid = np.zeros(n, dtype=bool)

        for index, pair in id_to_node.items():
            vec = self.embed(content_tokens(pair))
            if vec is not None:
                self.node_matrix[index] = vec
                self.node_valid[index] = True

        print(
            f"Embedded {int(self.node_valid.sum())}/{n} graph nodes with GloVe."
        )

    # -- slicing -------------------------------------------------------

    def slice_prompt(self, prompt, width=2, stride=1):
        tokens = [t for t in tokenize(prompt) if t not in ENGLISH_STOP_WORDS]
        if not tokens:
            tokens = tokenize(prompt)

        if len(tokens) <= width:
            windows = [tokens]
        else:
            windows = [
                tokens[i:i + width]
                for i in range(0, len(tokens) - width + 1, stride)
            ]

        slices = []
        for window in windows:
            vec = self.embed(window)
            if vec is not None:
                slices.append((window, vec))
        return slices

    # -- Manhattan distance and peaks ----------------------------------

    @staticmethod
    def manhattan(a, b):
        return float(np.abs(a - b).sum())

    def node_distances(self, slice_vec):
        """Manhattan distance from one slice to every node (inf if invalid)."""
        if self.node_matrix is None:
            raise RuntimeError("Call index_nodes() first.")

        distances = np.abs(self.node_matrix - slice_vec).sum(axis=1)
        distances[~self.node_valid] = np.inf
        return distances

    @staticmethod
    def find_peaks(distances, top_k=5, prominence=1.0):
        """
        Peaks = the nodes whose distance sits at least `prominence`
        standard deviations below the mean, capped at top_k. Always
        returns at least the single nearest node.
        """
        finite = np.isfinite(distances)
        if not finite.any():
            return []

        mean = distances[finite].mean()
        std = distances[finite].std()

        k = min(top_k, int(finite.sum()))
        nearest = np.argpartition(np.where(finite, distances, np.inf), k - 1)[:k]
        nearest = nearest[np.argsort(distances[nearest])]

        if std == 0:
            return [int(nearest[0])]

        peaks = [
            int(i) for i in nearest
            if (mean - distances[i]) / std >= prominence
        ]

        return peaks or [int(nearest[0])]

    # -- accommodating vocabulary --------------------------------------

    def expand_vocab(self, slice_tokens, slice_vec, dataset_vocab, topn=80, keep=8):
        """
        Slice words plus nearest GloVe neighbours that also occur in the
        dataset graph, so the Markov walk can actually reach them.
        """
        vocab = set(slice_tokens)
        added = 0
        for word, _ in self.kv.similar_by_vector(slice_vec, topn=topn):
            if word in dataset_vocab and word not in ENGLISH_STOP_WORDS:
                vocab.add(word)
                added += 1
                if added >= keep:
                    break
        return vocab


def dataset_vocab_from_nodes(id_to_node):
    """
    Graph nodes keep raw whitespace tokens ("dog.", "fox,"), so map each
    cleaned word to every raw form present in the graph.
    """
    clean_to_raw = {}
    for pair in id_to_node.values():
        for raw in pair:
            clean_to_raw.setdefault(clean_word(raw), set()).add(raw.lower())
    return clean_to_raw


def to_raw_vocab(clean_vocab, clean_to_raw):
    raw = set()
    for word in clean_vocab:
        raw.update(clean_to_raw.get(word, {word}))
    return raw


def print_peak_table(slices, per_slice_distances, per_slice_peaks, id_to_node, max_rows=12):
    """Peak nodes down the side, one distance column per slice."""
    best = {}
    for peaks, distances in zip(per_slice_peaks, per_slice_distances):
        for node in peaks:
            best[node] = min(best.get(node, np.inf), distances[node])

    rows = sorted(best, key=lambda node: best[node])[:max_rows]

    labels = [" ".join(tokens)[:14] for tokens, _ in slices]
    col = 16
    print("\nManhattan distance (peak node x slice)   * = peak for that slice")
    print("node".ljust(24) + "".join(label.ljust(col) for label in labels))

    for node in rows:
        name = " ".join(id_to_node[node])[:22]
        line = name.ljust(24)
        for peaks, distances in zip(per_slice_peaks, per_slice_distances):
            cell = f"{distances[node]:.3f}"
            if node in peaks:
                cell += " *"
            line += cell.ljust(col)
        print(line)


def render_side_by_side(columns, total_width=118, gap=3):
    """columns: list of (header, text). Prints them as parallel columns."""
    n = len(columns)
    width = max(18, (total_width - gap * (n - 1)) // n)
    wrapped = [textwrap.wrap(text, width) for _, text in columns]
    sep = " " * gap

    lines = [sep.join(header[:width].ljust(width) for header, _ in columns)]
    lines.append(sep.join("-" * width for _ in columns))
    for parts in zip_longest(*wrapped, fillvalue=""):
        lines.append(sep.join(part.ljust(width) for part in parts))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# GRAPH CREATION
# ----------------------------------------------------------------------

def process_text_to_trigram_graph(text, max_nodes=200):
    """
    Builds a graph where nodes are bigrams and edges represent trigrams.
    """
    text = text.lower()
    words = text.split()

    if len(words) < 3:
        return {}, {}, {}

    all_bigrams = [
        (words[i], words[i + 1])
        for i in range(len(words) - 1)
    ]

    top_bigrams = [
        bigram
        for bigram, _ in Counter(all_bigrams).most_common(max_nodes)
    ]

    bigram_set = set(top_bigrams)

    node_to_id = {
        bigram: index
        for index, bigram in enumerate(top_bigrams)
    }

    id_to_node = {
        index: bigram
        for index, bigram in enumerate(top_bigrams)
    }

    graph = {
        index: {}
        for index in range(len(top_bigrams))
    }

    for i in range(len(words) - 2):
        b1 = (words[i], words[i + 1])
        b2 = (words[i + 1], words[i + 2])

        if b1 in bigram_set and b2 in bigram_set:
            id1 = node_to_id[b1]
            id2 = node_to_id[b2]

            if id1 != id2:
                graph[id1][id2] = graph[id1].get(id2, 0) + 1
                graph[id2][id1] = graph[id2].get(id1, 0) + 1

    return graph, node_to_id, id_to_node


# ----------------------------------------------------------------------
# PROMPT-AWARE MARKOV GENERATOR
# ----------------------------------------------------------------------

def generate_markovian_prompt_text(
    graph,
    id_to_node,
    spins,
    prompt_vocab,
    length=100,
):
    n = len(graph)

    if n == 0:
        return ""

    prompt_vocab = {
        word.lower()
        for word in prompt_vocab
        if word.strip()
    }

    valid_starts = [
        i
        for i in range(n)
        if any(word in prompt_vocab for word in id_to_node[i])
    ]

    current_node = (
        random.choice(valid_starts)
        if valid_starts
        else random.randrange(n)
    )

    output = list(id_to_node[current_node])

    for _ in range(max(0, length - 2)):
        current_spin = spins[current_node]
        neighbors = graph[current_node]
        current_word_2 = id_to_node[current_node][1]

        valid_neighbors = [
            neighbor
            for neighbor in neighbors
            if id_to_node[neighbor][0] == current_word_2
        ]

        if not valid_neighbors:
            fallback_nodes = [
                i
                for i, spin in enumerate(spins)
                if (
                    spin != current_spin
                    and id_to_node[i][0] == current_word_2
                )
            ]

            if fallback_nodes:
                next_node = random.choice(fallback_nodes)
            else:
                next_node = random.randrange(n)
        else:
            probabilities = []

            for neighbor in valid_neighbors:
                base_weight = graph[current_node][neighbor]

                cut_multiplier = (
                    5.0
                    if spins[neighbor] != current_spin
                    else 1.0
                )

                neighbor_words = id_to_node[neighbor]

                prompt_multiplier = (
                    50.0
                    if any(word in prompt_vocab for word in neighbor_words)
                    else 1.0
                )

                score = (
                    base_weight
                    * cut_multiplier
                    * prompt_multiplier
                )

                probabilities.append(score)

            next_node = random.choices(
                valid_neighbors,
                weights=probabilities,
                k=1,
            )[0]

        output.append(id_to_node[next_node][1])
        current_node = next_node

    return " ".join(output)


# ----------------------------------------------------------------------
# GLOVE PROBE + SIDE-BY-SIDE GENERATION
# ----------------------------------------------------------------------

def glove_probe_and_generate(
    prompt,
    slicer,
    graph,
    id_to_node,
    spins,
    clean_to_raw,
    width=2,
    top_k=5,
    length=60,
    max_columns=4,
):
    slices = slicer.slice_prompt(prompt, width=width)

    if not slices:
        print("GloVe knows none of the prompt words; no slices to probe.")
        return False

    # Keep the output readable: cap the number of side-by-side columns.
    if len(slices) > max_columns:
        step = len(slices) / max_columns
        slices = [slices[int(i * step)] for i in range(max_columns)]

    per_slice_distances = [slicer.node_distances(vec) for _, vec in slices]
    per_slice_peaks = [
        slicer.find_peaks(d, top_k=top_k) for d in per_slice_distances
    ]

    if not any(per_slice_peaks):
        print("No usable peaks found.")
        return False

    print_peak_table(
        slices, per_slice_distances, per_slice_peaks, id_to_node
    )

    dataset_vocab = set(clean_to_raw)

    columns = []
    for (tokens, slice_vec), peaks in zip(slices, per_slice_peaks):
        clean_vocab = slicer.expand_vocab(tokens, slice_vec, dataset_vocab)
        raw_vocab = to_raw_vocab(clean_vocab, clean_to_raw)

        # The peak nodes' own words also seed the walk.
        for node in peaks:
            raw_vocab.update(id_to_node[node])

        text = generate_markovian_prompt_text(
            graph,
            id_to_node,
            spins,
            raw_vocab,
            length=length,
        )

        added = len(clean_vocab) - len(tokens)
        header = f"[{' '.join(tokens)}] +{max(added, 0)} words"
        columns.append((header, text.capitalize() + "."))

    print("\nGenerated Output (one column per prompt slice):\n")
    print(render_side_by_side(columns))
    print()
    return True


# ----------------------------------------------------------------------
# PLAIN FALLBACK (no GloVe)
# ----------------------------------------------------------------------

def plain_generate(prompt, graph, id_to_node, spins):
    final_text = generate_markovian_prompt_text(
        graph,
        id_to_node,
        spins,
        prompt.split(),
        length=400,
    )

    print("\nGenerated Output:")
    print(f"> {final_text.capitalize()}.\n")


# ----------------------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------------------

def load_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {filepath} not found. Using fallback text.")

        return (
            "The quick brown fox jumps over the lazy dog. "
            "The lazy dog barks at the brown fox. "
            "The quick brown fox runs away quickly into the deep "
            "dark brown forest. The dark forest is full of mystery. "
            "A dog barks loudly in the dark."
        )


def load_slicer(id_to_node):
    if not GENSIM_AVAILABLE:
        print("gensim not installed (pip install gensim). Using plain generation.")
        return None

    glove_path = input(
        "Local GloVe .txt path (blank = download glove-wiki-gigaword-100): "
    ).strip()

    try:
        slicer = GloveSlicer(glove_path=glove_path or None)
        slicer.index_nodes(id_to_node)
        return slicer
    except Exception as error:
        print(f"Could not load GloVe ({error}). Using plain generation.")
        return None


def main():
    dataset_file = input("Dataset filename: ").strip()

    dataset_text = clean_text(load_text_file(dataset_file))

    print("Building graph from the complete dataset...")
    graph, node_to_id, id_to_node = process_text_to_trigram_graph(
        dataset_text,
        max_nodes=150000,
    )

    if not graph:
        print("Graph is empty. Check the dataset.")
        return

    print(f"Graph built with {len(graph)} unique nodes.")

    print("Running PCMMaxCut solver for structural baseline...")

    solver = PCMMaxCut(
        graph,
        noise=0.00001,
        drift_nu=0.00000001,
    )

    history = solver.run(
        iterations=5000,
        temperature=0.0000005,
        cooling=0.0000000199,
    )

    best_cut = max(
        history,
        key=lambda item: item["cut"],
    )

    print(
        f"Base partition created. "
        f"Max Cut Size: {best_cut['cut']}\n"
    )

    spins = best_cut["spins"]
    clean_to_raw = dataset_vocab_from_nodes(id_to_node)
    slicer = load_slicer(id_to_node)

    while True:
        try:
            user_prompt = input("USER: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_prompt:
            continue

        if user_prompt.lower() in {"exit", "quit"}:
            break

        done = False

        if slicer is not None:
            done = glove_probe_and_generate(
                user_prompt,
                slicer,
                graph,
                id_to_node,
                spins,
                clean_to_raw,
                width=2,
                top_k=5,
                length=60,
                max_columns=4,
            )

        # Fall back to plain prompt-word biasing if GloVe is unavailable
        # or knows none of the prompt words.
        if not done:
            plain_generate(user_prompt, graph, id_to_node, spins)


if __name__ == "__main__":
    main()
