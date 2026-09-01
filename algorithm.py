from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional imports for GPT-2 integration
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_GPT2 = True
except ImportError:
    HAS_GPT2 = False

# ============================================================
# Small n-gram language model + corpus similarity search.
# ============================================================

MODEL_PATH = "model.json"
USE_GPT2 = False  # Set to True to use pretrained GPT-2 instead of n-gram
GPT2_MODEL_NAME = "gpt2" # Can be "gpt2", "gpt2-medium", etc.

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 20

MIN_COUNT = 1
INFLUENCE_TAU = 0.5

CURVE_K = 8.0
CURVE_MIDPOINT = 0.5

CANDIDATE_LIMIT = 15
LEXICAL_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[.,!?;:()\[\]{}\-]")
IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}


# ============================================================
# Text utilities
# ============================================================

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def bag_of_words(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t not in IGNORED_TOKENS)


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a) - IGNORED_TOKENS
    sb = set(b) - IGNORED_TOKENS
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


# ============================================================
# Corpus similarity search (used for the "candidates" display)
# ============================================================

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
    """Finds corpus sentences closest to a prompt (lexical + cosine)."""

    def __init__(
        self,
        lexical_weight: float = LEXICAL_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.references: List[CorpusReference] = []

    def build_index(self, corpus_text: str) -> None:
        sentences = split_sentences(corpus_text)
        counts = Counter(s.lower() for s in sentences)

        self.references = []
        for sentence in sentences:
            tokens = tokenize(sentence)
            if not tokens:
                continue
            bow = bag_of_words(tokens)
            self.references.append(
                CorpusReference(
                    sentence=sentence,
                    tokens=tokens,
                    vector={t: float(c) for t, c in bow.items()},
                    frequency=counts[sentence.lower()],
                )
            )

    def analyze(self, prompt: str, limit: int = 5) -> List[Candidate]:
        prompt_tokens = tokenize(prompt)
        prompt_vector = {t: float(c) for t, c in bag_of_words(prompt_tokens).items()}

        candidates = []
        for ref in self.references:
            symbolic = lexical_overlap(prompt_tokens, ref.tokens)
            vector_sim = cosine_similarity(prompt_vector, ref.vector)
            score = self.lexical_weight * symbolic + self.vector_weight * vector_sim
            candidates.append(
                Candidate(
                    sentence=ref.sentence,
                    symbolic_overlap=symbolic,
                    vector_similarity=vector_sim,
                    frequency=ref.frequency,
                    score=score,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:limit]
        for i, c in enumerate(candidates, start=1):
            c.rank = i
        return candidates


# ============================================================
# N-gram language model
# ============================================================

@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT
    influence_tau: float = INFLUENCE_TAU
    curve_k: float = CURVE_K
    curve_midpoint: float = CURVE_MIDPOINT

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
            self._add_sequence(sequence)

    def _add_sequence(self, sequence: List[str]) -> None:
        if len(sequence) < 3:
            return
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}
        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {ctx: c / total for ctx, c in counts.items()}

        self.influence_vectors = {}
        for source in self.vocabulary:
            source_vec = self.lexical_vectors.get(source, {})
            scores = {}
            for target in self.vocabulary:
                if source == target:
                    continue
                sim = cosine_similarity(source_vec, self.lexical_vectors.get(target, {}))
                if sim >= self.influence_tau:
                    scores[target] = sim
            self.influence_vectors[source] = scores

        self.finalized = True

    def _backoff_distribution(self, prev: str, prev_prev: Optional[str]) -> Dict[str, float]:
        if prev_prev is not None:
            counts = self.trigram.get(f"{prev_prev}\t{prev}")
            if counts:
                return self._normalize(counts)
        counts = self.bigram.get(prev)
        if counts:
            return self._normalize(counts)
        return self._normalize(self.unigram)

    @staticmethod
    def _normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()} if total else {}

    def _curve_weight(self, p_eos: float) -> float:
        p_eos = min(1.0, max(0.0, p_eos))
        z = self.curve_k * (p_eos - self.curve_midpoint)
        return 1.0 / (1.0 + math.exp(z))

    def _resolve_context(self, prompt: str) -> Tuple[str, Optional[str]]:
        tokens = tokenize(prompt)
        if not tokens:
            return "<bos>", None
        prev = tokens[-1]
        prev_prev = tokens[-2] if len(tokens) >= 2 else None
        return prev, prev_prev

    def _score_next_token(self, prompt: str, candidate_limit: int = 64) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()

        prev, prev_prev = self._resolve_context(prompt)
        base = self._backoff_distribution(prev, prev_prev)
        if not base:
            return {}

        candidates = sorted(base, key=base.get, reverse=True)[:candidate_limit]
        source_vec = self.lexical_vectors.get(prev, {})
        influences = self.influence_vectors.get(prev, {})
        curve = self._curve_weight(base.get(self.eos_token, 0.0))

        scores = {}
        for token in candidates:
            similarity = cosine_similarity(source_vec, self.lexical_vectors.get(token, {}))
            influence = influences.get(token, 0.0)
            scores[token] = (
                safe_log(base[token])
                + curve * 0.35 * similarity
                + curve * 0.65 * influence
            )
        return scores

    def _probabilities(self, prompt: str, temperature: float, candidate_limit: int) -> Dict[str, float]:
        scores = self._score_next_token(prompt, candidate_limit)
        if not scores:
            return {}
        temperature = max(temperature, 1e-5)
        scaled = {t: s / temperature for t, s in scores.items()}
        maximum = max(scaled.values())
        exps = {t: math.exp(s - maximum) for t, s in scaled.items()}
        total = sum(exps.values())
        return {t: v / total for t, v in exps.items()} if total else {}

    def sample_next(self, prompt: str, temperature: float = 0.8, top_k: int = 20) -> str:
        probs = self._probabilities(prompt, temperature, max(top_k, 1))
        if not probs:
            return self.eos_token
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 20) -> str:
        generated = tokenize(prompt)
        for _ in range(max_new_tokens):
            token = self.sample_next(" ".join(generated), temperature, top_k)
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
            "curve_k": self.curve_k,
            "curve_midpoint": self.curve_midpoint,
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {k: dict(v) for k, v in self.trigram.items()},
            "lexical_vectors": self.lexical_vectors,
            "influence_vectors": self.influence_vectors,
            "vocabulary": self.vocabulary,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(
            eos_token=data["eos_token"],
            unk_token=data["unk_token"],
            min_count=data["min_count"],
            influence_tau=data["influence_tau"],
            curve_k=data["curve_k"],
            curve_midpoint=data["curve_midpoint"],
        )
        model.unigram = Counter(data["unigram"])
        model.bigram = defaultdict(Counter, {k: Counter(v) for k, v in data["bigram"].items()})
        model.trigram = defaultdict(Counter, {k: Counter(v) for k, v in data["trigram"].items()})
        model.lexical_vectors = data["lexical_vectors"]
        model.influence_vectors = data["influence_vectors"]
        model.vocabulary = data["vocabulary"]
        model.finalized = data["finalized"]
        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NGramModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ============================================================
# GPT-2 Language Model Wrapper
# ============================================================

class GPT2Generator:
    """Hugging Face GPT-2 wrapper matching the generation interface."""

    def __init__(self, model_name: str = GPT2_MODEL_NAME) -> None:
        if not HAS_GPT2:
            raise ImportError("Please install PyTorch and Transformers to use GPT-2: pip install torch transformers")
        
        print(f"\nLoading Hugging Face GPT-2 model ({model_name})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Dummy attributes to mimic stats printouts safely
        self.vocabulary = list(self.tokenizer.get_vocab().keys())
        self.unigram = Counter()
        self.bigram = {}
        self.trigram = {}

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 20) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Handle generation settings safely
        do_sample = temperature > 0.0
        
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text


# ============================================================
# Display
# ============================================================

def display_candidates(candidates: List[Candidate]) -> None:
    print()
    print("=" * 70)
    print("REASONING (Corpus Context Match)")
    print("=" * 70)
    if not candidates:
        print("No candidates found.")
        return
    model_gpt = GPT2Generator(GPT2_MODEL_NAME)
    generated_string = []
    for c in candidates:
        # Added a leading space so words don't collide ("sentence this is" instead of "sentencethis is")
        prompt_text = c.sentence.strip() + " this is"
        
        generated = model_gpt.generate(  # Change to model_gpt if that's your variable name
            prompt_text,
            max_new_tokens=440,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )       
        
        try:
            parts = [p.strip() for p in generated.split('.') if p.strip()]
            
            if len(parts) >= 2:
                print(f'"""{parts[0]}.' + '"""')
                print(f"\n{parts[1]}.]\n\n")
            elif len(parts) == 1:
                print(f'"""{parts[0]}.' + '"""\n')
            else:
                print(f'"""{generated}"""\n')
                
        except (IndexError, ValueError):
            # Fallback if splitting fails unexpectedly
            print(f'"""{generated}"""\n')

def display_generation(generated: str) -> None:
    print()
    print("=" * 70)
    print("GENERATED")
    print("=" * 70)
    print()
    print(generated)


# ============================================================
# Main
# ============================================================

def main() -> None:
    corpus_path = Path(input("Filename: "))
    if not corpus_path.exists():
        print(f"\nERROR: {corpus_path} does not exist.")
        return

    corpus_text = corpus_path.read_text(encoding="utf-8")

    # Initialize model depending on flag

    if Path(MODEL_PATH).exists():
        print("\nLoading existing n-gram model...")
        model = NGramModel.load_json(MODEL_PATH)
    else:
        print("\nTraining n-gram model...")
        model = NGramModel(min_count=MIN_COUNT, influence_tau=INFLUENCE_TAU)
        model.ingest_text(corpus_text)
        model.finalize()
        model.save_json(MODEL_PATH)

    print(f"\nVocabulary: {len(model.vocabulary)}")
    print(f"Unigrams: {len(model.unigram)}")
    print(f"Bigram contexts: {len(model.bigram)}")
    print(f"Trigram contexts: {len(model.trigram)}")

    search = CorpusSearch(lexical_weight=LEXICAL_WEIGHT, vector_weight=VECTOR_WEIGHT)
    search.build_index(corpus_text)  # built once, not on every turn

    while True:
        prompt = input("\nUSER: ").strip()
        if not prompt:
            print("Empty prompt.")
            continue

        candidates = search.analyze(prompt, limit=CANDIDATE_LIMIT)
        display_candidates(candidates)


if __name__ == "__main__":
    main()
