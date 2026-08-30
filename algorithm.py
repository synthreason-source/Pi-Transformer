from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\[\]{}\-]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def cosine_similarity(
    a: Dict[str, float],
    b: Dict[str, float],
) -> float:
    if not a or not b:
        return 0.0

    common = set(a).intersection(b)
    dot = sum(a[k] * b[k] for k in common)

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


@dataclass
class SparseSymbolManifest:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = 1
    influence_tau: float = 0.5
    curve_name: str = "sigmoid"
    curve_k: float = 8.0
    curve_midpoint: float = 0.5

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)

    finalized: bool = False

    def ingest_text(self, text: str) -> None:
        for sentence in split_sentences(text):
            words = tokenize(sentence)
            if not words:
                continue

            sequence = ["<bos>", "<bos>"] + words + [self.eos_token]
            self.add_sequence_words(sequence)

    def ingest_dataset_words(self, sequences: Iterable[Iterable[str]]) -> None:
        for words in sequences:
            sequence = ["<bos>", "<bos>"] + list(words) + [self.eos_token]
            self.add_sequence_words(sequence)

    def add_sequence_words(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return

        for token in sequence:
            self.unigram[token] += 1

        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1

        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1

        self.finalized = False

    def register_vocabulary(self) -> None:
        self.vocabulary = sorted(
            token
            for token, count in self.unigram.items()
            if count >= self.min_count
        )

        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

    def update_intrinsics(self) -> None:
        self.register_vocabulary()

        token_contexts: Dict[str, Counter] = defaultdict(Counter)

        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}

        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1

            self.lexical_vectors[token] = {
                context: count / total
                for context, count in counts.items()
            }

    def build_influence_vectors(self) -> None:
        self.influence_vectors = {}

        for source in self.vocabulary:
            source_vector = self.lexical_vectors.get(source, {})
            scores: Dict[str, float] = {}

            for target in self.vocabulary:
                if source == target:
                    continue

                target_vector = self.lexical_vectors.get(target, {})
                similarity = cosine_similarity(source_vector, target_vector)

                if similarity >= self.influence_tau:
                    scores[target] = similarity

            self.influence_vectors[source] = scores

    def finalize(self) -> None:
        self.update_intrinsics()
        self.build_influence_vectors()
        self.finalized = True

    def backoff_distributions(
        self,
        previous_token: str,
        previous_previous_token: str | None = None,
    ) -> Dict[str, float]:
        if previous_previous_token is not None:
            key = f"{previous_previous_token}\t{previous_token}"
            counts = self.trigram.get(key)

            if counts:
                return self._normalize_counts(counts)

        counts = self.bigram.get(previous_token)

        if counts:
            return self._normalize_counts(counts)

        return self._normalize_counts(self.unigram)

    @staticmethod
    def _normalize_counts(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())

        if total == 0:
            return {}

        return {
            token: count / total
            for token, count in counts.items()
        }

    def prompt_indices(self, prompt: str) -> List[str]:
        return tokenize(prompt)

    def resolve_prompt_context(self, prompt: str) -> Tuple[str, str | None]:
        tokens = self.prompt_indices(prompt)

        if not tokens:
            return "<bos>", None

        previous = tokens[-1]
        previous_previous = tokens[-2] if len(tokens) >= 2 else None

        return previous, previous_previous

    def prompt_curve_sequence(self, prompt: str) -> List[float]:
        tokens = self.prompt_indices(prompt)

        if not tokens:
            return [0.0]

        eos_position = None

        for index, token in enumerate(tokens):
            if token == self.eos_token:
                eos_position = index
                break

        if eos_position is None:
            eos_position = len(tokens)

        denominator = max(len(tokens), 1)

        return [
            min(1.0, max(0.0, index / denominator))
            for index in range(eos_position + 1)
        ]

    def curve_weight(self, p_eos: float) -> float:
        p_eos = min(1.0, max(0.0, p_eos))

        if self.curve_name == "linear":
            return 1.0 - p_eos

        if self.curve_name == "inverse":
            return 1.0 / (1.0 + p_eos)

        if self.curve_name == "sigmoid":
            z = self.curve_k * (p_eos - self.curve_midpoint)
            return 1.0 / (1.0 + math.exp(z))

        raise ValueError(f"Unknown curve: {self.curve_name}")

    def similarity_breakdown(
        self,
        context_token: str,
        candidates: Iterable[str],
    ) -> Dict[str, float]:
        source = self.lexical_vectors.get(context_token, {})
        scores = {}

        for candidate in candidates:
            target = self.lexical_vectors.get(candidate, {})
            scores[candidate] = cosine_similarity(source, target)

        return scores

    def influence_score(
        self,
        context_token: str,
        candidate: str,
    ) -> float:
        direct = self.influence_vectors.get(context_token, {})
        return direct.get(candidate, 0.0)

    def influence_vector(
        self,
        context_token: str,
        candidates: Iterable[str],
    ) -> Dict[str, float]:
        return {
            candidate: self.influence_score(context_token, candidate)
            for candidate in candidates
        }

    def score_next_token(
        self,
        prompt: str,
        candidate_limit: int = 64,
    ) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        previous, previous_previous = self.resolve_prompt_context(prompt)

        base = self.backoff_distributions(
            previous,
            previous_previous,
        )

        if not base:
            return {}

        candidates = sorted(
            base,
            key=base.get,
            reverse=True,
        )[:candidate_limit]

        similarities = self.similarity_breakdown(previous, candidates)
        influences = self.influence_vector(previous, candidates)

        p_eos = base.get(self.eos_token, 0.0)
        curve_weight = self.curve_weight(p_eos)

        scores = {}

        for token in candidates:
            log_probability = safe_log(base[token])

            similarity_term = similarities.get(token, 0.0)
            influence_term = influences.get(token, 0.0)

            score = (
                log_probability
                + curve_weight * 0.35 * similarity_term
                + curve_weight * 0.65 * influence_term
            )

            scores[token] = score

        return scores

    def probabilities(
        self,
        prompt: str,
        temperature: float = 1.0,
        candidate_limit: int = 64,
    ) -> Dict[str, float]:
        scores = self.score_next_token(prompt, candidate_limit)

        if not scores:
            return {}

        temperature = max(temperature, 1e-5)

        scaled = {
            token: score / temperature
            for token, score in scores.items()
        }

        maximum = max(scaled.values())
        exponentials = {
            token: math.exp(score - maximum)
            for token, score in scaled.items()
        }

        total = sum(exponentials.values())

        return {
            token: value / total
            for token, value in exponentials.items()
        }

    def sample_next(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> str:
        probabilities = self.probabilities(
            prompt,
            temperature=temperature,
            candidate_limit=max(top_k, 1),
        )

        if not probabilities:
            return self.eos_token

        items = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        tokens = [token for token, _ in items]
        weights = [weight for _, weight in items]

        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> str:
        generated = tokenize(prompt)

        for _ in range(max_new_tokens):
            current_prompt = " ".join(generated)
            token = self.sample_next(
                current_prompt,
                temperature=temperature,
                top_k=top_k,
            )

            if token == self.eos_token:
                break

            generated.append(token)

        return self.detokenize(generated)

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:)\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        return text

    def to_dict(self) -> dict:
        return {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "min_count": self.min_count,
            "influence_tau": self.influence_tau,
            "curve_name": self.curve_name,
            "curve_k": self.curve_k,
            "curve_midpoint": self.curve_midpoint,
            "unigram": dict(self.unigram),
            "bigram": {
                key: dict(value)
                for key, value in self.bigram.items()
            },
            "trigram": {
                key: dict(value)
                for key, value in self.trigram.items()
            },
            "lexical_vectors": self.lexical_vectors,
            "influence_vectors": self.influence_vectors,
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SparseSymbolManifest":
        model = cls(
            eos_token=data["eos_token"],
            unk_token=data["unk_token"],
            min_count=data["min_count"],
            influence_tau=data["influence_tau"],
            curve_name=data["curve_name"],
            curve_k=data["curve_k"],
            curve_midpoint=data["curve_midpoint"],
        )

        model.unigram = Counter(data["unigram"])
        model.bigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value in data["bigram"].items()
            },
        )
        model.trigram = defaultdict(
            Counter,
            {
                key: Counter(value)
                for key, value in data["trigram"].items()
            },
        )
        model.lexical_vectors = data["lexical_vectors"]
        model.influence_vectors = data["influence_vectors"]
        model.vocabulary = data["vocabulary"]
        model.finalized = data["finalized"]

        return model

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SparseSymbolManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def build_model(
    input_path: str,
    output_path: str,
    curve: str,
    min_count: int,
    influence_tau: float,
) -> None:
    text = Path(input_path).read_text(encoding="utf-8")

    model = SparseSymbolManifest(
        curve_name=curve,
        min_count=min_count,
        influence_tau=influence_tau,
    )

    model.ingest_text(text)
    model.finalize()
    model.save_json(output_path)

    print(f"Saved model to {output_path}")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    print(f"Unigram types: {len(model.unigram)}")
    print(f"Bigram contexts: {len(model.bigram)}")
    print(f"Trigram contexts: {len(model.trigram)}")


def run_generation(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> None:
    model = SparseSymbolManifest.load_json(model_path)

    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse Symbol Manifest language model"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("input")
    train_parser.add_argument("output")
    train_parser.add_argument(
        "--curve",
        choices=["linear", "sigmoid", "inverse"],
        default="sigmoid",
    )
    train_parser.add_argument("--min-count", type=int, default=1)
    train_parser.add_argument(
        "--influence-tau",
        type=float,
        default=0.5,
    )

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("model")
    generate_parser.add_argument("prompt")
    generate_parser.add_argument("--max-new-tokens", type=int, default=50)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    if args.command == "train":
        build_model(
            input_path=args.input,
            output_path=args.output,
            curve=args.curve,
            min_count=args.min_count,
            influence_tau=args.influence_tau,
        )

    elif args.command == "generate":
        run_generation(
            model_path=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
