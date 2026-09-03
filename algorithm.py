from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

MODEL_PATH = "model.json"
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
        sentences = split_sentences(text)
        for sentence in tqdm(sentences, desc="Ingesting sentences"):
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
        for token in tqdm(self.vocabulary, desc="Building lexical vectors"):
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {ctx: c / total for ctx, c in counts.items()}

        self.influence_vectors = {}
        for source in tqdm(self.vocabulary, desc="Computing influence vectors"):
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

    def score_next_token_decomposed(self, prompt: str, candidate_limit: int = 64):
        """Same math as _score_next_token, but returns the base (explained)
        and boost (unexplained) contributions separately per candidate."""
        if not self.finalized:
            self.finalize()
        prev, prev_prev = self._resolve_context(prompt)
        base = self._backoff_distribution(prev, prev_prev)
        if not base:
            return {}, 0.0
        candidates = sorted(base, key=base.get, reverse=True)[:candidate_limit]
        source_vec = self.lexical_vectors.get(prev, {})
        influences = self.influence_vectors.get(prev, {})
        p_eos = base.get(self.eos_token, 0.0)
        curve = self._curve_weight(p_eos)

        decomposed = {}
        for token in candidates:
            explained = safe_log(base[token])
            similarity = cosine_similarity(source_vec, self.lexical_vectors.get(token, {}))
            influence = influences.get(token, 0.0)
            boost = curve * 0.35 * similarity + curve * 0.65 * influence
            decomposed[token] = {
                "base_prob": base[token],
                "explained_score": explained,
                "boost_score": boost,
                "total_score": explained + boost,
                "similarity": similarity,
                "influence": influence,
            }
        return decomposed, curve

    def _score_next_token(self, prompt: str, candidate_limit: int = 64) -> Dict[str, float]:
        decomposed, _ = self.score_next_token_decomposed(prompt, candidate_limit)
        return {t: d["total_score"] for t, d in decomposed.items()}

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

    def _sample_with_prob(self, prompt: str, temperature: float, top_k: int) -> Tuple[str, float]:
        """Like sample_next, but also returns the probability mass the
        model itself assigned to the token it picked -- this is the
        confidence signal used to drive automodification."""
        probs = self._probabilities(prompt, temperature, max(top_k, 1))
        if not probs:
            return self.eos_token, 0.0
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        token = random.choices(tokens, weights=weights, k=1)[0]
        return token, probs[token]

    def generate_automodifying(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        min_temp: float = 0.2,
        max_temp: float = 2.0,
        adapt_rate: float = 0.4,
        trace: bool = False,
    ):
        """
        Sec. 4: S_{t+1} = F(S_t, I_t); pi_{t+1} = G(pi_t, S_t, I_t) -- run
        as an actual loop inside generation, using a feedback signal that
        is guaranteed to be informative every step: the model's own
        confidence (probability mass) in the token it just emitted.

          confidence high  -> temperature tightens  (commit further)
          confidence low   -> temperature loosens   (explore more)

        This changes `temp`, which is then used to sample the NEXT token,
        so the effect is not cosmetic -- it changes what actually gets
        generated. `trace=True` returns the per-step (token, confidence,
        temp) log so the effect can be inspected directly.
        """
        generated = tokenize(prompt)
        temp = temperature
        log = []

        for step in range(max_new_tokens):
            context = " ".join(generated)
            token, confidence = self._sample_with_prob(context, temp, top_k)
            #if token == self.eos_token:
                #break

            temp_before = temp
            # pi_{t+1} = G(pi_t, S_t, I_t): temperature IS part of the
            # generation policy here, and it is being rewritten based on
            # what the system just observed about its own output.
            temp = temp * (1 + adapt_rate * (0.5 - confidence))
            temp = max(min_temp, min(max_temp, temp))
            if token not in IGNORED_TOKENS and token not in generated:
                generated.append(token)
            if trace:
                log.append((step, token, confidence, temp_before, temp))

            # S_{t+1} = F(S_t, I_t): feed the emitted token back into the
            # model's own counts before the next step is scored.
            window = generated[-3:] if len(generated) >= 3 else ["<bos>"] * (3 - len(generated)) + generated
            self.unigram[token] += 1
            if len(window) >= 2:
                self.bigram[window[-2]][window[-1]] += 1
            if len(window) >= 3:
                self.trigram[f"{window[-3]}\t{window[-2]}"][window[-1]] += 1

        text = self.detokenize(generated)
        return (text, log) if trace else text

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:)\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        return text

with open(input("Filename: "), "r", encoding="utf-8") as file:
    CORPUS = file.read()


model = NGramModel(influence_tau=0.15)  # lower threshold so the boost term
model.ingest_text(CORPUS)               # actually engages on this small corpus
model.finalize()
    
while True:
    text = model.generate_automodifying(prompt=input("USER: "), max_new_tokens=650)
    print()
    print("GENERATED (automodifying):", text)
