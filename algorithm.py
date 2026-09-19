from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import gradio as gr


MODEL_PATH = "model.json"
DEFAULT_CORPUS_FILE = "corpus.txt"

MAX_NEW_TOKENS = 800
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 20

MIN_COUNT = 1
SMOOTHING_ALPHA = 0.05

IMAGE_BIAS_WEIGHT = 1.2

SIGN_INVARIANT_SIMILARITY = True

COT_STEPS = 4
COT_STATE_DECAY = 0.5
COT_MIN_RELEVANCE = 0.05
COT_REDUNDANCY_LIMIT = 0.9
COT_TOP_N = 3

COT_SEED_WORDS = 12
COT_BIAS_WEIGHT = 0.6
GENERATE_REFRESH_EVERY = 8

USE_HEMICUBE = True
HC_RES = 6
HC_MAX_DEPTH = 4
HC_MIN_MEMBERS = 3
HC_HOMOGENEITY_STOP = 0.6
HC_FIELD_WEIGHT = 0.8
HC_IMAGE_SIZE = 640

IMAGE_DESCRIPTOR_WORDS: Dict[str, List[str]] = {
    "bright": ["bright", "light", "sunny", "white", "day"],
    "dark": ["dark", "night", "shadow", "black", "dim"],
    "warm": ["warm", "red", "orange", "fire", "hot"],
    "cool": ["cool", "blue", "cold", "ice", "water"],
    "green": ["green", "grass", "forest", "leaf", "plant"],
}

RANDOM_SEED = None
random.seed(RANDOM_SEED)

IGNORED_TOKENS = {
    "<bos>",
    "<eos>",
    "<unk>",
}


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def split_sentences(text: str) -> List[str]:
    parts = text.split(".")
    return [part.strip() for part in parts if part.strip()]


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(token for token in tokens if token not in IGNORED_TOKENS)


def cosine_similarity(
    a: Dict[str, float],
    b: Dict[str, float],
    sign_invariant: bool = SIGN_INVARIANT_SIMILARITY,
) -> float:
    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)

    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    value = dot / (norm_a * norm_b)

    return abs(value) if sign_invariant else value


def align_sign(
    a: Dict[str, float], b: Dict[str, float]
) -> Tuple[Dict[str, float], float]:
    dot = sum(a[key] * b[key] for key in set(a) & set(b))
    sign = 1.0 if dot >= 0 else -1.0

    return {key: sign * value for key, value in b.items()}, sign


def lexical_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a) - IGNORED_TOKENS
    set_b = set(b) - IGNORED_TOKENS

    if not set_a or not set_b:
        return 0.0

    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def vdot(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b[key] for key, value in a.items() if key in b)


def vnorm(a: Dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in a.values()))


def vaxpy(a: Dict[str, float], b: Dict[str, float], scale: float) -> Dict[str, float]:
    out = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, 0.0) + scale * value
    return out


def vunit(a: Dict[str, float]) -> Dict[str, float]:
    norm = vnorm(a)
    if norm <= 0.0:
        return {}
    return {key: value / norm for key, value in a.items()}


@dataclass
class CorpusReference:
    sentence: str
    tokens: List[str]
    vector: Dict[str, float]
    frequency: int = 1


@dataclass
class Candidate:
    sentence: str
    symbolic_overlap: float
    vector_similarity: float
    frequency: int
    score: float
    rank: int = 0


class CorpusSearch:
    def __init__(
        self,
        lexical_weight: float = 0.5,
        vector_weight: float = 0.5,
        sign_invariant: bool = SIGN_INVARIANT_SIMILARITY,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.sign_invariant = sign_invariant
        self.references: List[CorpusReference] = []
        self.doc_freq: Counter = Counter()
        self.weighted: List[Dict[str, float]] = []

    def idf(self, token: str) -> float:
        n = len(self.references)
        return math.log((1 + n) / (1 + self.doc_freq.get(token, 0)))

    def weight_vector(self, vector: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for token, weight in vector.items():
            value = weight * self.idf(token)
            if value > 0.0:
                out[token] = value
        return out

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(sentence.lower() for sentence in sentences)

        self.references = []
        self.doc_freq = Counter()
        self.weighted = []

        for sentence in sentences:
            tokens = tokenize(sentence)

            if not tokens:
                continue

            bow = bag_of_words(tokens)
            self.doc_freq.update(bow.keys())

            self.references.append(
                CorpusReference(
                    sentence=sentence,
                    tokens=tokens,
                    vector={token: float(count) for token, count in bow.items()},
                    frequency=counts[sentence.lower()],
                )
            )

        self.weighted = [self.weight_vector(ref.vector) for ref in self.references]

    def analyze(self, prompt: str, limit: int = 5) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)
        prompt_vector = {
            token: float(count) for token, count in bag_of_words(prompt_tokens).items()
        }

        candidates: List[Candidate] = []

        for reference in self.references:
            symbolic = lexical_overlap(prompt_tokens, reference.tokens)
            vector_similarity = cosine_similarity(
                prompt_vector, reference.vector, sign_invariant=self.sign_invariant
            )

            score = (
                self.lexical_weight * symbolic + self.vector_weight * vector_similarity
            )

            candidates.append(
                Candidate(
                    sentence=reference.sentence,
                    symbolic_overlap=symbolic,
                    vector_similarity=vector_similarity,
                    frequency=reference.frequency,
                    score=score,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        candidates = candidates[:limit]

        for index, candidate in enumerate(candidates, start=1):
            candidate.rank = index

        return candidates


@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT
    smoothing_alpha: float = SMOOTHING_ALPHA

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)

            if not words:
                continue

            sequence = ["<bos>", "<bos>", *words, self.eos_token]
            self._add_sequence(sequence)

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return

        for token in sequence:
            self.unigram[token] += 1

        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1

        for first, second, third in zip(sequence, sequence[1:], sequence[2:]):
            key = f"{first}\t{second}"
            self.trigram[key][third] += 1

        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(
            token for token, count in self.unigram.items() if count >= self.min_count
        )

        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        self.finalized = True

    def _smoothed_distribution(self, counts: Counter) -> Dict[str, float]:
        if not counts:
            return {}

        alpha = self.smoothing_alpha
        total = sum(counts.values()) + alpha * len(counts)

        return {token: (count + alpha) / total for token, count in counts.items()}

    def backoff_distribution(
        self, previous: str, previous_previous: Optional[str]
    ) -> Dict[str, float]:
        if previous_previous is not None:
            key = f"{previous_previous}\t{previous}"
            counts = self.trigram.get(key)

            if counts:
                return self._smoothed_distribution(counts)

        counts = self.bigram.get(previous)

        if counts:
            return self._smoothed_distribution(counts)

        return self._smoothed_distribution(self.unigram)

    def cooccurrence_bias(self, seed_weights: Dict[str, float]) -> Dict[str, float]:
        bias: Dict[str, float] = defaultdict(float)

        for word, weight in seed_weights.items():
            if weight <= 0:
                continue

            followers = self.bigram.get(word)

            if not followers:
                continue

            total = sum(followers.values())

            for token, count in followers.items():
                bias[token] += weight * (count / total)

        return dict(bias)

    def resolve_context(self, tokens: List[str]) -> Tuple[str, Optional[str]]:
        if not tokens:
            return "<bos>", "<bos>"

        previous = tokens[-1]
        previous_previous = tokens[-2] if len(tokens) >= 2 else "<bos>"

        return previous, previous_previous

    def sample_next(
        self,
        tokens: List[str],
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        bias: Optional[Dict[str, float]] = None,
        bias_weight: float = 0.0,
    ) -> str:
        if not self.finalized:
            self.finalize()

        previous, previous_previous = self.resolve_context(tokens)
        distribution = self.backoff_distribution(previous, previous_previous)

        if not distribution:
            return self.eos_token

        temperature = max(temperature, 1e-5)

        adjusted: Dict[str, float] = {}

        for token, probability in distribution.items():
            log_prob = math.log(max(probability, 1e-12))

            if bias and bias_weight:
                log_prob += bias_weight * bias.get(token, 0.0)

            adjusted[token] = log_prob / temperature

        top_tokens = sorted(adjusted, key=adjusted.get, reverse=True)[:top_k]

        maximum = max(adjusted[token] for token in top_tokens)
        weights = [math.exp(adjusted[token] - maximum) for token in top_tokens]

        return random.choices(top_tokens, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        bias: Optional[Dict[str, float]] = None,
        bias_weight: float = 0.0,
        bias_provider: Optional[
            Callable[[List[str]], Tuple[Optional[Dict[str, float]], float]]
        ] = None,
        refresh_every: int = GENERATE_REFRESH_EVERY,
    ) -> str:
        prompt_tokens = tokenize(prompt)
        generated = list(prompt_tokens)

        for step in range(max_new_tokens):
            if bias_provider is not None and step % max(1, refresh_every) == 0:
                bias, bias_weight = bias_provider(generated)

            token = self.sample_next(
                generated,
                temperature=temperature,
                top_k=top_k,
                bias=bias,
                bias_weight=bias_weight,
            )

            if token == self.eos_token:
                break

            generated.append(token)

        continuation = generated[len(prompt_tokens):]
        visible_tokens = [token for token in continuation if token not in IGNORED_TOKENS]

        return " ".join(visible_tokens)

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "smoothing_alpha": self.smoothing_alpha,
            "unigram": dict(self.unigram),
            "bigram": {key: dict(value) for key, value in self.bigram.items()},
            "trigram": {key: dict(value) for key, value in self.trigram.items()},
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data.get("eos_token", "<eos>"),
            unk_token=data.get("unk_token", "<unk>"),
            min_count=data.get("min_count", MIN_COUNT),
            smoothing_alpha=data.get("smoothing_alpha", SMOOTHING_ALPHA),
        )

        model.unigram = Counter(data.get("unigram", {}))

        model.bigram = defaultdict(
            Counter,
            {key: Counter(value) for key, value in data.get("bigram", {}).items()},
        )

        model.trigram = defaultdict(
            Counter,
            {key: Counter(value) for key, value in data.get("trigram", {}).items()},
        )

        model.vocabulary = data.get("vocabulary", [])
        model.finalized = data.get("finalized", False)

        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class HemiCell:
    face: str
    row: int
    col: int
    members: List[int]
    weight: float
    centroid: Dict[str, float]
    homogeneity: float
    children: Optional["HemiCube"] = None


def hemicube_cell(dx: float, dy: float, dz: float, res: int) -> Tuple[str, int, int, float]:
    half = max(1, res // 2)
    ax, ay = abs(dx), abs(dy)

    def bucket(value: float, n: int) -> int:
        return max(0, min(n - 1, int(value * n)))

    if dz > 0.0 and dz >= max(ax, ay):
        px, py = dx / dz, dy / dz
        col = bucket((px + 1.0) / 2.0, res)
        row = bucket((py + 1.0) / 2.0, res)
        ff = 1.0 / (math.pi * (px * px + py * py + 1.0) ** 2)
        return "top", row, col, ff

    if ax >= ay:
        face = "+x" if dx >= 0 else "-x"
        s, h = dy / ax, dz / ax
    else:
        face = "+y" if dy >= 0 else "-y"
        s, h = dx / ay, dz / ay

    col = bucket((s + 1.0) / 2.0, res)
    row = bucket(h, half)
    ff = h / (math.pi * (s * s + h * h + 1.0) ** 2)

    return face, row, col, ff


class HemiCube:
    def __init__(
        self,
        vectors: List[Dict[str, float]],
        indices: List[int],
        normal: Dict[str, float],
        depth: int = 0,
        res: int = HC_RES,
        max_depth: int = HC_MAX_DEPTH,
        min_members: int = HC_MIN_MEMBERS,
        stop: float = HC_HOMOGENEITY_STOP,
        sign_invariant: bool = SIGN_INVARIANT_SIMILARITY,
    ) -> None:
        self.depth = depth
        self.res = max(2, res)
        self.half = max(1, self.res // 2)
        self.max_depth = max_depth
        self.min_members = max(2, min_members)
        self.stop = stop
        self.sign_invariant = sign_invariant
        self.normal: Dict[str, float] = {}
        self.cells: Dict[Tuple[str, int, int], HemiCell] = {}

        self._build(vectors, indices, normal)

    def _build(
        self, vectors: List[Dict[str, float]], indices: List[int], normal: Dict[str, float]
    ) -> None:
        members = [i for i in indices if vectors[i]]

        if not members:
            return

        n = vunit(normal)

        if not n:
            total: Dict[str, float] = {}
            for i in members:
                for key, value in vunit(vectors[i]).items():
                    total[key] = total.get(key, 0.0) + value
            n = vunit(total)

        if not n:
            return

        self.normal = n
        u, v = self._tangents(vectors, members, n)

        buckets: Dict[Tuple[str, int, int], List[Tuple[int, float, float]]] = defaultdict(list)

        for i in members:
            vec = vectors[i]
            mag = vnorm(vec)

            if mag <= 0.0:
                continue

            a = vdot(vec, n)
            x = vdot(vec, u) if u else 0.0
            y = vdot(vec, v) if v else 0.0

            if a < 0.0:
                if not self.sign_invariant:
                    continue
                a, x, y = -a, -x, -y

            t = math.sqrt(a * a + x * x + y * y)

            if t <= 1e-12:
                continue

            face, row, col, ff = hemicube_cell(x / t, y / t, a / t, self.res)
            captured = min(1.0, t / mag)
            buckets[(face, row, col)].append((i, ff, captured))

        for key, items in buckets.items():
            ids = [i for i, _, _ in items]
            weight = sum(ff * captured for _, ff, captured in items)
            centroid, homogeneity = self._centroid(vectors, ids)

            cell = HemiCell(
                face=key[0],
                row=key[1],
                col=key[2],
                members=ids,
                weight=weight,
                centroid=centroid,
                homogeneity=homogeneity,
            )

            if (
                homogeneity < self.stop
                and len(ids) >= self.min_members
                and self.depth < self.max_depth
                and centroid
            ):
                child = HemiCube(
                    vectors,
                    ids,
                    centroid,
                    depth=self.depth + 1,
                    res=self.res,
                    max_depth=self.max_depth,
                    min_members=self.min_members,
                    stop=self.stop,
                    sign_invariant=self.sign_invariant,
                )
                if child.cells:
                    cell.children = child

            self.cells[key] = cell

    def _tangents(
        self,
        vectors: List[Dict[str, float]],
        members: List[int],
        n: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        residuals: List[Dict[str, float]] = []
        for i in members:
            vec = vectors[i]
            residuals.append(vaxpy(vec, n, -vdot(vec, n)))

        best, best_norm = {}, 0.0
        for r in residuals:
            rn = vnorm(r)
            if rn > best_norm:
                best, best_norm = r, rn

        u = {k: value / best_norm for k, value in best.items()} if best_norm > 1e-12 else {}

        best, best_norm = {}, 0.0
        for r in residuals:
            if u:
                r = vaxpy(r, u, -vdot(r, u))
            rn = vnorm(r)
            if rn > best_norm:
                best, best_norm = r, rn

        v = {k: value / best_norm for k, value in best.items()} if best_norm > 1e-12 else {}

        return u, v

    def _centroid(
        self, vectors: List[Dict[str, float]], ids: List[int]
    ) -> Tuple[Dict[str, float], float]:
        acc: Dict[str, float] = {}
        count = 0

        for i in ids:
            unit = vunit(vectors[i])

            if not unit:
                continue

            if self.sign_invariant and acc:
                unit, _ = align_sign(acc, unit)

            for key, value in unit.items():
                acc[key] = acc.get(key, 0.0) + value

            count += 1

        if not count:
            return {}, 0.0

        return vunit(acc), min(1.0, vnorm(acc) / count)

    def level_counts(self) -> List[int]:
        counts: List[int] = []
        carried = 0
        frontier: List[HemiCube] = [self]

        while frontier:
            cells = [c for cube in frontier for c in cube.cells.values()]
            counts.append(len(cells) + carried)
            carried += sum(1 for c in cells if c.children is None)
            frontier = [c.children for c in cells if c.children is not None]

        return counts

    def fractal_dimension(self) -> float:
        counts = [c for c in self.level_counts() if c > 0]

        if len(counts) < 2:
            return 0.0

        xs = [level * math.log(self.res) for level in range(len(counts))]
        ys = [math.log(c) for c in counts]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)

        if denom == 0.0:
            return 0.0

        return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom

    def depth_reached(self) -> int:
        return len(self.level_counts()) - 1

    def describe(self, sentences: List[str], limit: int = 6, indent: int = 0) -> List[str]:
        lines: List[str] = []
        pad = "  " * indent
        ranked = sorted(self.cells.values(), key=lambda c: c.weight, reverse=True)
        shown = max(2, limit - indent)

        for cell in ranked[:shown]:
            tag = (
                f"{pad}{cell.face}({cell.row},{cell.col}) n={len(cell.members)} "
                f"weight={cell.weight:.3f} homogeneity={cell.homogeneity:.2f}"
            )

            if cell.children is None:
                preview = sentences[cell.members[0]]
                if len(preview) > 70:
                    preview = preview[:67] + "..."
                lines.append(f"{tag} | {preview}")
            else:
                lines.append(f"{tag} -> re-project")
                lines.extend(cell.children.describe(sentences, limit, indent + 1))

        if len(ranked) > shown:
            lines.append(f"{pad}... {len(ranked) - shown} more cells")

        return lines

    def _net_position(self, cell: HemiCell) -> Tuple[int, int]:
        res, half = self.res, self.half

        if cell.face == "top":
            return half + cell.row, half + cell.col
        if cell.face == "+x":
            return half + cell.col, half + res + (half - 1 - cell.row)
        if cell.face == "-x":
            return half + cell.col, cell.row
        if cell.face == "+y":
            return half + res + (half - 1 - cell.row), half + cell.col
        return cell.row, half + cell.col

    @staticmethod
    def _cell_color(cell: HemiCell, peak: float, depth: int) -> np.ndarray:
        h = cell.homogeneity
        energy = 0.35 + 0.65 * math.sqrt(min(1.0, cell.weight / peak)) if peak > 0 else 0.5
        rgb = np.array([255.0 * (1.0 - h), 200.0 * h, 90.0 + 40.0 * min(depth, 4)])
        return np.clip(rgb * energy, 0, 255)

    def render(self, size: int = HC_IMAGE_SIZE) -> np.ndarray:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:, :] = (12, 12, 18)
        self._paint(canvas, 0.0, 0.0, float(size))
        return canvas

    def _paint(self, canvas: np.ndarray, x0: float, y0: float, size: float) -> None:
        net = self.res + 2 * self.half
        unit = size / net
        peak = max((c.weight for c in self.cells.values()), default=0.0)

        for cell in self.cells.values():
            r, c = self._net_position(cell)
            xs, xe = int(round(x0 + c * unit)), int(round(x0 + (c + 1) * unit))
            ys, ye = int(round(y0 + r * unit)), int(round(y0 + (r + 1) * unit))

            if xe <= xs or ye <= ys:
                continue

            color = self._cell_color(cell, peak, self.depth)

            if cell.children is not None and unit >= 4.0:
                canvas[ys:ye, xs:xe] = (color * 0.3).astype(np.uint8)
                cell.children._paint(canvas, x0 + c * unit, y0 + r * unit, unit)
            else:
                canvas[ys:ye, xs:xe] = color.astype(np.uint8)
                if ye - ys >= 4 and xe - xs >= 4:
                    edge = (color * 0.5).astype(np.uint8)
                    canvas[ys, xs:xe] = edge
                    canvas[ye - 1, xs:xe] = edge
                    canvas[ys:ye, xs] = edge
                    canvas[ys:ye, xe - 1] = edge


def build_semantic_tree(
    search: CorpusSearch, state: Dict[str, float], **kwargs: Any
) -> HemiCube:
    return HemiCube(
        search.weighted,
        list(range(len(search.references))),
        search.weight_vector(state),
        sign_invariant=search.sign_invariant,
        **kwargs,
    )


def describe_tree(cube: HemiCube, search: CorpusSearch) -> str:
    if not cube.cells:
        return "Hemicube is empty (no usable sentences)."

    counts = cube.level_counts()
    header = [
        f"Depth reached: {cube.depth_reached()} (max {cube.max_depth})",
        f"Occupied cells per level: {counts}",
        f"Box-counting dimension: {cube.fractal_dimension():.3f}",
        "",
    ]
    sentences = [ref.sentence for ref in search.references]

    return "\n".join(header + cube.describe(sentences))


@dataclass
class ThoughtStep:
    index: int
    sentence: str
    relevance: float
    novelty: float
    path: str = ""
    homogeneity: float = 0.0


class ChainOfThought:
    def __init__(
        self,
        search: CorpusSearch,
        decay: float = COT_STATE_DECAY,
        min_relevance: float = COT_MIN_RELEVANCE,
        redundancy_limit: float = COT_REDUNDANCY_LIMIT,
        top_n: int = COT_TOP_N,
    ) -> None:
        self.search = search
        self.decay = decay
        self.min_relevance = min_relevance
        self.redundancy_limit = redundancy_limit
        self.top_n = max(1, top_n)
        self.last_state: Dict[str, float] = {}
        self.last_field: Dict[str, float] = {}

    def _cos(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return cosine_similarity(a, b, sign_invariant=self.search.sign_invariant)

    def _begin(self) -> None:
        self.last_field = {}

    def _score_candidates(
        self,
        state: Dict[str, float],
        chosen_vectors: List[Dict[str, float]],
        used: set,
    ) -> List[Tuple[float, int, float, float]]:
        scored: List[Tuple[float, int, float, float]] = []

        for idx, reference in enumerate(self.search.references):
            if idx in used:
                continue

            relevance = self._cos(state, reference.vector)

            if relevance < self.min_relevance:
                continue

            overlap = max(
                (self._cos(reference.vector, v) for v in chosen_vectors),
                default=0.0,
            )

            if overlap >= self.redundancy_limit:
                continue

            novelty = 1.0 - overlap
            scored.append((relevance * novelty, idx, relevance, novelty))

        return scored

    def _pick(
        self,
        state: Dict[str, float],
        chosen_vectors: List[Dict[str, float]],
        used: set,
    ) -> Optional[Tuple[int, float, float, str, float]]:
        scored = self._score_candidates(state, chosen_vectors, used)

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        pool = scored[: self.top_n]

        _, idx, relevance, novelty = random.choices(
            pool, weights=[item[0] for item in pool], k=1
        )[0]

        return idx, relevance, novelty, "", 0.0

    def run(self, prompt: str, steps: int = COT_STEPS) -> List[ThoughtStep]:
        state: Dict[str, float] = {
            token: float(count)
            for token, count in bag_of_words(tokenize(prompt)).items()
        }
        chosen_vectors: List[Dict[str, float]] = []
        used: set = set()
        chain: List[ThoughtStep] = []

        self._begin()

        for _ in range(steps):
            pick = self._pick(state, chosen_vectors, used)

            if pick is None:
                break

            idx, relevance, novelty, path, homogeneity = pick

            reference = self.search.references[idx]
            used.add(idx)
            chosen_vectors.append(reference.vector)

            chain.append(
                ThoughtStep(
                    index=len(chain) + 1,
                    sentence=reference.sentence,
                    relevance=relevance,
                    novelty=novelty,
                    path=path,
                    homogeneity=homogeneity,
                )
            )

            state = {token: self.decay * weight for token, weight in state.items()}
            for token, weight in reference.vector.items():
                state[token] = state.get(token, 0.0) + weight

        self.last_state = state

        return chain


class HemicubeChainOfThought(ChainOfThought):
    def __init__(
        self,
        search: CorpusSearch,
        res: int = HC_RES,
        max_depth: int = HC_MAX_DEPTH,
        min_members: int = HC_MIN_MEMBERS,
        stop: float = HC_HOMOGENEITY_STOP,
        **kwargs: Any,
    ) -> None:
        super().__init__(search, **kwargs)
        self.res = res
        self.max_depth = max_depth
        self.min_members = min_members
        self.stop = stop
        self.last_cube: Optional[HemiCube] = None

    @staticmethod
    def _energy(cell: HemiCell) -> float:
        return cell.weight * (0.25 + 0.75 * cell.homogeneity)

    def _pick(
        self,
        state: Dict[str, float],
        chosen_vectors: List[Dict[str, float]],
        used: set,
    ) -> Optional[Tuple[int, float, float, str, float]]:
        scored = self._score_candidates(state, chosen_vectors, used)

        if not scored:
            return None

        info = {idx: (relevance, novelty, score) for score, idx, relevance, novelty in scored}

        cube = HemiCube(
            self.search.weighted,
            list(info),
            self.search.weight_vector(state),
            res=self.res,
            max_depth=self.max_depth,
            min_members=self.min_members,
            stop=self.stop,
            sign_invariant=self.search.sign_invariant,
        )
        self.last_cube = cube

        if not cube.cells:
            return ChainOfThought._pick(self, state, chosen_vectors, used)

        path: List[str] = []
        current = cube
        cell: Optional[HemiCell] = None

        while current is not None and current.cells:
            ranked = sorted(current.cells.values(), key=self._energy, reverse=True)
            pool = ranked[: self.top_n]
            cell = random.choices(
                pool, weights=[max(self._energy(c), 1e-9) for c in pool], k=1
            )[0]
            path.append(f"{cell.face}({cell.row},{cell.col})")
            current = cell.children

        assert cell is not None

        members = [i for i in cell.members if i in info]

        if not members:
            return ChainOfThought._pick(self, state, chosen_vectors, used)

        members.sort(key=lambda i: info[i][2], reverse=True)
        pool_ids = members[: self.top_n]
        idx = random.choices(
            pool_ids, weights=[max(info[i][2], 1e-9) for i in pool_ids], k=1
        )[0]

        self.last_field = vaxpy(
            self.last_field, cell.centroid, 0.5 + 0.5 * cell.homogeneity
        )

        relevance, novelty, _ = info[idx]

        return idx, relevance, novelty, " > ".join(path), cell.homogeneity


def make_chain(search: CorpusSearch, use_hemicube: bool = USE_HEMICUBE) -> ChainOfThought:
    return HemicubeChainOfThought(search) if use_hemicube else ChainOfThought(search)


def format_chain(chain: List[ThoughtStep]) -> str:
    if not chain:
        return "No reasoning steps found for that prompt."

    lines = []
    for step in chain:
        line = (
            f"Step {step.index}: {step.sentence}.  "
            f"(relevance={step.relevance:.3f}, novelty={step.novelty:.3f}"
        )
        if step.path:
            line += f", homogeneity={step.homogeneity:.2f}, path={step.path}"
        lines.append(line + ")")
    lines.append(f"Conclusion: {chain[-1].sentence}.")

    return "\n".join(lines)


def merge_seed_weights(
    base: Dict[str, float], extra: Dict[str, float], extra_weight: float
) -> Dict[str, float]:
    merged = dict(base)

    for word, weight in extra.items():
        merged[word] = merged.get(word, 0.0) + extra_weight * weight

    return merged


class ReasoningSubstrate:
    def __init__(
        self,
        model: NGramModel,
        search: CorpusSearch,
        initial_state: Dict[str, float],
        prompt_length: int = 0,
        decay: float = COT_STATE_DECAY,
        top_words: int = COT_SEED_WORDS,
        field: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model = model
        self.search = search
        self.anchor = dict(initial_state)
        self.state = dict(initial_state)
        self.decay = decay
        self.top_words = top_words
        self.field = dict(field) if field else {}
        self._seen = prompt_length

    def seed_weights(self) -> Dict[str, float]:
        weighted = {
            token: weight * self.search.idf(token)
            for token, weight in self.state.items()
            if token in self.model.unigram and token not in IGNORED_TOKENS
        }
        weighted = {token: w for token, w in weighted.items() if w > 0}

        if self.field:
            scale = max(weighted.values(), default=0.0) or 1.0
            field_peak = max(self.field.values(), default=0.0)

            if field_peak > 0:
                for token, w in self.field.items():
                    if w > 0 and token in self.model.unigram and token not in IGNORED_TOKENS:
                        weighted[token] = weighted.get(token, 0.0) + (
                            HC_FIELD_WEIGHT * scale * w / field_peak
                        )

        top = sorted(weighted.items(), key=lambda item: item[1], reverse=True)
        top = top[: self.top_words]

        if not top:
            return {}

        peak = top[0][1]

        return {token: weight / peak for token, weight in top}

    def alignment(self) -> float:
        return cosine_similarity(
            self.state, self.anchor, sign_invariant=self.search.sign_invariant
        )

    def update(self, new_tokens: List[str]) -> None:
        if not new_tokens:
            return

        self.state = {token: self.decay * w for token, w in self.state.items()}

        for token, count in bag_of_words(new_tokens).items():
            self.state[token] = self.state.get(token, 0.0) + float(count)

    def provider(
        self, base_seeds: Dict[str, float], base_weight: float
    ) -> Callable[[List[str]], Tuple[Optional[Dict[str, float]], float]]:
        def refresh(generated: List[str]) -> Tuple[Optional[Dict[str, float]], float]:
            new_tokens = generated[self._seen:]
            self.update(new_tokens)
            self._seen = len(generated)

            seeds = merge_seed_weights(base_seeds, self.seed_weights(), COT_BIAS_WEIGHT)
            scale = 1.0 + (1.0 - self.alignment())

            return self.model.cooccurrence_bias(seeds), base_weight * scale

        return refresh


def extract_image_descriptors(image: Any) -> Dict[str, float]:
    if image is None:
        return {}

    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    image = np.asarray(image).astype(np.float32)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    r = float(image[:, :, 0].mean()) / 255.0
    g = float(image[:, :, 1].mean()) / 255.0
    b = float(image[:, :, 2].mean()) / 255.0

    brightness = (r + g + b) / 3.0
    warmth = r - b
    greenness = g - (r + b) / 2.0

    return {
        "bright": max(0.0, (brightness - 0.5) * 2.0),
        "dark": max(0.0, (0.5 - brightness) * 2.0),
        "warm": max(0.0, warmth),
        "cool": max(0.0, -warmth),
        "green": max(0.0, greenness),
    }


def image_seed_words(model: NGramModel, image: Any) -> Dict[str, float]:
    descriptors = extract_image_descriptors(image)
    seed_weights: Dict[str, float] = {}

    for descriptor_key, intensity in descriptors.items():
        if intensity <= 0:
            continue

        for word in IMAGE_DESCRIPTOR_WORDS.get(descriptor_key, []):
            if word in model.unigram:
                seed_weights[word] = seed_weights.get(word, 0.0) + intensity

    return seed_weights


TEXT_MODEL: NGramModel | None = None
CORPUS_SEARCH: CorpusSearch | None = None
CORPUS_TEXT_CACHE: str | None = None


def load_text_model() -> NGramModel:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Upload a corpus and click 'Train model' first."
        )
    model = NGramModel.load_json(model_path)
    if not model.finalized:
        model.finalize()
    return model


def load_corpus_search(corpus_text: str) -> CorpusSearch:
    search = CorpusSearch()
    search.build_index(corpus_text)
    return search


def reload_globals():
    global TEXT_MODEL, CORPUS_SEARCH, CORPUS_TEXT_CACHE

    TEXT_MODEL = load_text_model()

    if CORPUS_TEXT_CACHE:
        corpus_text = CORPUS_TEXT_CACHE
    else:
        default_path = Path(DEFAULT_CORPUS_FILE)
        if default_path.exists():
            corpus_text = default_path.read_text(encoding="utf-8", errors="replace")
            CORPUS_TEXT_CACHE = corpus_text
        else:
            corpus_text = ""

    CORPUS_SEARCH = load_corpus_search(corpus_text)


def format_matches(prompt: str, limit: int = 5) -> str:
    if CORPUS_SEARCH is None or not CORPUS_SEARCH.references:
        return "No corpus file loaded."

    candidates = CORPUS_SEARCH.analyze(prompt, limit=limit)
    if not candidates:
        return "No corpus matches found."

    lines = []
    for candidate in candidates:
        lines.append(
            f"{candidate.rank}. "
            f"{candidate.sentence} "
            f"(score={candidate.score:.3f}, "
            f"overlap={candidate.symbolic_overlap:.3f}, "
            f"vector={candidate.vector_similarity:.3f})"
        )
    return "\n".join(lines)


def train_model_from_file(corpus_file):
    global CORPUS_TEXT_CACHE

    if corpus_file is None:
        return "No corpus file uploaded.", "", ""

    if hasattr(corpus_file, "name"):
        path = Path(corpus_file.name)
    else:
        path = Path(corpus_file)

    if not path.exists():
        return f"Corpus file not found: {path}", "", ""

    corpus_text = path.read_text(encoding="utf-8", errors="replace")
    CORPUS_TEXT_CACHE = corpus_text

    model = NGramModel()
    model.ingest_text(corpus_text)
    model.finalize()

    model_path = Path(MODEL_PATH)
    model.save_json(model_path)

    reload_globals()

    sample_lines = "\n".join(corpus_text.splitlines()[:5])

    summary = (
        f"Vocabulary: {len(model.vocabulary)}\n"
        f"Unigrams: {len(model.unigram)}\n"
        f"Bigram contexts: {len(model.bigram)}\n"
        f"Trigram contexts: {len(model.trigram)}"
    )

    return f"Model trained and saved to {MODEL_PATH}.", sample_lines, summary


def process_camera_image(image, user_prompt: str):
    if TEXT_MODEL is None or CORPUS_SEARCH is None:
        return (
            image,
            "Model not loaded. Upload a corpus and click 'Train model' first.",
            "",
            "",
        )

    seed_weights = image_seed_words(TEXT_MODEL, image)

    prompt = user_prompt.strip() if user_prompt and user_prompt.strip() else ""

    chain: List[ThoughtStep] = []
    substrate: Optional[ReasoningSubstrate] = None

    if prompt and CORPUS_SEARCH.references:
        cot = make_chain(CORPUS_SEARCH)
        chain = cot.run(prompt, steps=COT_STEPS)
        substrate = ReasoningSubstrate(
            TEXT_MODEL,
            CORPUS_SEARCH,
            cot.last_state,
            prompt_length=len(tokenize(prompt)),
            field=cot.last_field,
        )

    reasoning_seeds = substrate.seed_weights() if substrate else {}
    combined_seeds = merge_seed_weights(seed_weights, reasoning_seeds, COT_BIAS_WEIGHT)
    bias = TEXT_MODEL.cooccurrence_bias(combined_seeds)

    generated = TEXT_MODEL.generate(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        bias=bias,
        bias_weight=IMAGE_BIAS_WEIGHT,
        bias_provider=(
            substrate.provider(seed_weights, IMAGE_BIAS_WEIGHT) if substrate else None
        ),
    )

    if prompt:
        corpus_matches = format_matches(prompt, limit=5)
        if chain:
            corpus_matches += "\n\nReasoning chain used for modifiers:\n" + format_chain(chain)
    else:
        corpus_matches = "No prompt provided for corpus search."

    bias_lines = []
    if seed_weights:
        bias_lines.append(
            "Image-derived seed words (found in corpus): "
            + ", ".join(f"{word} ({weight:.2f})" for word, weight in seed_weights.items())
        )
    else:
        bias_lines.append("No image-derived seed words matched the corpus vocabulary.")

    if reasoning_seeds:
        bias_lines.append(
            "Reasoning seed words (refreshed during generation): "
            + ", ".join(f"{word} ({weight:.2f})" for word, weight in reasoning_seeds.items())
        )

    bias_summary = "\n".join(bias_lines)

    return image, bias_summary, corpus_matches, generated


def run_chain_of_thought(prompt: str, n_steps: int, use_hemicube: bool = USE_HEMICUBE):
    if CORPUS_SEARCH is None or not CORPUS_SEARCH.references:
        return "No corpus loaded. Upload a corpus and click 'Train model' first.", None, ""

    if not prompt or not prompt.strip():
        return "Enter a prompt first.", None, ""

    cot = make_chain(CORPUS_SEARCH, use_hemicube=bool(use_hemicube))
    chain = cot.run(prompt, steps=int(n_steps))

    image = None
    tree_text = ""

    if use_hemicube and cot.last_state:
        tree = build_semantic_tree(CORPUS_SEARCH, cot.last_state)
        if tree.cells:
            image = tree.render()
            tree_text = describe_tree(tree, CORPUS_SEARCH)

    return format_chain(chain), image, tree_text


with gr.Blocks(title="Stochastic N-Gram Model") as demo:
    gr.Markdown(
        """
# Stochastic N-Gram Model

1. Upload any text file as corpus.
2. Click **Train model**.
3. Optionally use the webcam/upload -- simple image stats (brightness /
   warmth / greenness) bias generation toward words that actually follow
   related seed words in your corpus.

Every probability used for generation is computed directly from n-gram
counts observed in the corpus; text is sampled stochastically from that
distribution, so re-running the same prompt can give different results.

The generation process uses a **hemicube recursion** over the corpus:
sentences are projected onto a half-cube oriented on the current reasoning
state. Cells that are not homogeneous are re-projected on their own centroid,
producing a self-similar tree; the chain of thought descends that tree to
select steps. The resulting state then biases token sampling via real bigram
statistics from the corpus.
"""
    )

    gr.Markdown("## 1. Corpus & Training")

    with gr.Row():
        with gr.Column(scale=1):
            corpus_file_input = gr.File(
                label="Corpus file (any type)",
                file_types=["file"],
            )
            train_button = gr.Button("Train model", variant="primary")

        with gr.Column(scale=2):
            train_status = gr.Textbox(label="Training status", lines=2)
            corpus_sample = gr.Textbox(label="Corpus sample (first 5 lines)", lines=5)
            model_summary = gr.Textbox(label="Model summary", lines=6)

    train_button.click(
        fn=train_model_from_file,
        inputs=[corpus_file_input],
        outputs=[train_status, corpus_sample, model_summary],
    )

    gr.Markdown("## 2. Camera + Stochastic Generation")

    with gr.Row():
        with gr.Column(scale=1):
            camera = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Camera image",
                webcam_options=gr.WebcamOptions(mirror=False),
            )
            user_prompt = gr.Textbox(label="Optional text prompt", lines=3)
            recognize_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            processed_image = gr.Image(label="Processed image", type="numpy")
            feature_tokens_output = gr.Textbox(
                label="Bias words (image + reasoning)", lines=4
            )
            corpus_output = gr.Textbox(label="Corpus matches", lines=6)
            generated_output = gr.Textbox(label="Generated text", lines=12)

    recognize_button.click(
        fn=process_camera_image,
        inputs=[camera, user_prompt],
        outputs=[
            processed_image,
            feature_tokens_output,
            corpus_output,
            generated_output,
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    try:
        reload_globals()
    except FileNotFoundError:
        pass

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
