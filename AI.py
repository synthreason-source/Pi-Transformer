#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

try:
    import torch
except ImportError:
    class _FakeTorch:
        def tensor(self, x, **kw): return x
        def zeros(self, *a, **kw): return [0.0]
    torch = _FakeTorch()

PUNCT_SET = {",", ".", "!", "?", ";", ":"}
ORBIT_SYMBOLS = {0: "○", 1: "◔", 2: "◑", 3: "◕"}
PHASE_COLORS = {"INPUT": "\033[94m", "FWD": "\033[93m", "REV": "\033[96m", "AND": "\033[92m", "EMIT": "\033[97m"}
RESET = "\033[0m"


def load_text_corpus(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")
    return p.read_text(encoding="utf-8", errors="replace")


def _simple_tokenize(text: str) -> List[str]:
    out = []
    for word in text.split():
        w = word.strip()
        if not w:
            continue
        if w[-1] in PUNCT_SET and len(w) > 1:
            out.append(w[:-1])
            out.append(w[-1])
        else:
            out.append(w)
    return [t for t in out if t]


def _detokenize(tokens: List[str]) -> str:
    parts = []
    for t in tokens:
        if t in PUNCT_SET and parts:
            parts[-1] += t
        else:
            parts.append(t)
    return " ".join(parts)


def corpus_vocab_from_text(text: str) -> List[str]:
    seen, vocab = set(), []
    for t in _simple_tokenize(text):
        if t in PUNCT_SET:
            continue
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            vocab.append(tl)
    return vocab


class SentenceInput:
    def __init__(self):
        self._tokens: List[str] = []
        self._raw: str = ""

    def receive(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            raise ValueError("Input must not be empty.")
        self._raw = text
        self._tokens = _simple_tokenize(text)
        return self._tokens

    def receive_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            raise ValueError("Token list must not be empty.")
        self._tokens = list(tokens)
        self._raw = " ".join(tokens)
        return self._tokens

    def get(self) -> List[str]:
        if not self._tokens:
            raise RuntimeError("No input received yet.")
        return self._tokens

    def get_raw(self) -> str:
        return self._raw


class MachineChoice:
    MAX_RETRIES = 3

    def __init__(self):
        self._options: List[Tuple[str, Callable, Callable]] = []

    def register(self, name: str, condition: Callable[[List[str]], bool], handler: Callable[[List[str]], str]):
        self._options.append((name, condition, handler))

    def select(self, tokens: List[str]) -> Tuple[str, str]:
        for name, condition, handler in self._options:
            if condition(tokens):
                return name, handler(tokens)
        raise ValueError(f"XOR violation: no option matched tokens {tokens[:4]}…")


class LingualBinding:
    @staticmethod
    def bind(original_tokens: List[str], machine_output: str) -> str:
        if not original_tokens:
            raise ValueError("AND binding failed: original tokens missing.")
        if not machine_output:
            raise ValueError("AND binding failed: machine output missing.")
        joined = " ".join(original_tokens)
        return f"{machine_output} [bound:{joined[:60]}{'…' if len(joined) > 60 else ''}]"


class CombinatorialController:
    def __init__(self):
        self.sentence_in = SentenceInput()
        self.machine_choice = MachineChoice()
        self.lingual_binding = LingualBinding()

    def register_option(self, name, condition, handler):
        self.machine_choice.register(name, condition, handler)

    def run(self, text: str) -> str:
        return self._run_from_tokens(self.sentence_in.receive(text))

    def run_tokens(self, tokens: List[str]) -> str:
        self.sentence_in.receive_tokens(tokens)
        return self._run_from_tokens(tokens)

    def _run_from_tokens(self, tokens: List[str]) -> str:
        attempt = 0
        while attempt < MachineChoice.MAX_RETRIES:
            attempt += 1
            try:
                _, machine_output = self.machine_choice.select(tokens)
                break
            except ValueError as exc:
                if attempt == MachineChoice.MAX_RETRIES:
                    raise RuntimeError(f"XOR failed after {MachineChoice.MAX_RETRIES} attempts: {exc}") from exc
                tokens = self.sentence_in.get()
        return self.lingual_binding.bind(tokens, machine_output)


@dataclass
class CycleTokenEvent:
    step: int
    sentence_idx: int
    token: str
    bound_output: str
    cardan_orbit: int = 0
    mirror_weight: float = 0.35
    spaghetti_norm: float = 0.0
    elapsed_ms: float = 0.0
    is_punct: bool = False
    cycle_phase: str = "EMIT"


@dataclass
class CycleSentenceEvent:
    sentence_idx: int
    sentence_text: str
    tokens: List[str]
    cot_trace: str = ""
    prop_stmts: List[str] = field(default_factory=list)
    avg_cardan_orbit: float = 0.0
def _bidirectional_windows(tokens: List[str], width: int, stride: int):
    if not tokens:
        return
    width = max(1, min(width, len(tokens)))
    stride = max(1, stride)
    starts = list(range(0, max(1, len(tokens) - width + 1), stride))
    direction = 1
    for s in starts:
        w = tokens[s:s + width]
        yield w if direction == 1 else list(reversed(w))
        direction *= -1


class TokenCycleEmitter:
    def __init__(self, engine=None, mirror_alpha: float = 0.35, cardan_logit_weight: float = 8.0, emit_delay_ms: float = 0.0):
        self.engine = engine
        self.mirror_alpha = mirror_alpha
        self.cardan_logit_weight = cardan_logit_weight
        self.emit_delay_ms = emit_delay_ms
        self._ctrl = self._build_controller()
        self._step = 0
        self._start_ts = 0.0

    def _build_controller(self) -> CombinatorialController:
        ctrl = CombinatorialController()
        ctrl.register_option(
            "forward_pass",
            lambda toks: len(toks) >= 1 and len(toks) % 2 == 1,
            lambda toks: f"[FWD] {' '.join(toks[-5:])}",
        )
        ctrl.register_option(
            "reverse_pass",
            lambda toks: len(toks) >= 1 and len(toks) % 2 == 0,
            lambda toks: f"[REV] {' '.join(reversed(toks[-5:]))}",
        )
        ctrl.register_option(
            "punctuation_gate",
            lambda toks: len(toks) >= 1 and toks[-1] in PUNCT_SET,
            lambda toks: f"[PUNCT:{toks[-1]}] sentence-boundary gate",
        )
        ctrl.register_option(
            "cognitive_token",
            lambda toks: len(toks) >= 1 and toks[-1] not in PUNCT_SET and any(t.startswith("[") for t in toks[-3:]),
            lambda toks: f"[COG] cognitive-token routing: {toks[-1]}",
        )
        return ctrl

    def cycle(self, instruction: str = "You are a computational algorithm.", seed: str = "", n_sentences: int = 4, tokens_per_sent: int = 40, and_weight: float = 0.9, temperature: float = 2.0) -> Generator:
        self._step = 0
        self._start_ts = time.perf_counter()
        if self.engine is not None:
            yield from self._cycle_real(instruction, seed, n_sentences, tokens_per_sent, and_weight, temperature)
        else:
            yield from self._cycle_demo(instruction, seed, n_sentences, tokens_per_sent)

    def _cycle_real(self, instruction, seed, n_sentences, tokens_per_sent, and_weight, temperature):
        eng = self.engine
        try:
            result = eng.generate(
                seedtext=seed,
                instructiontext=instruction,
                numsentences=n_sentences,
                tokenspersent=tokens_per_sent,
                andweight=and_weight,
                temperature=temperature,
                returntraces=True,
            )
        except Exception as exc:
            yield CycleSentenceEvent(0, f"[TokenCycle] Engine generate() error: {exc}", [])
            return
        full_text, cot_text = result if isinstance(result, tuple) else (str(result), "")
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]
        traces = []
        if hasattr(eng, "walker") and eng.walker is not None:
            traces = list(getattr(eng.walker, "steptraces", getattr(eng.walker, "_step_traces", [])))
        trace_idx = 0
        for sent_idx, sent_text in enumerate(sentences):
            tokens = _simple_tokenize(sent_text)
            orbit_sum = 0.0
            for tok_pos, tok in enumerate(tokens):
                cardan_orbit = 0
                spaghetti_norm = 0.0
                if trace_idx < len(traces):
                    tr = traces[trace_idx]
                    if hasattr(tr, "cardan_orbit"):
                        cardan_orbit = tr.cardan_orbit
                    if hasattr(tr, "spaghetti_mixer_norms"):
                        spaghetti_norm = sum(tr.spaghetti_mixer_norms) / 3.0
                    trace_idx += 1
                orbit_sum += cardan_orbit
                ctx_tokens = tokens[:tok_pos + 1]
                try:
                    bound = self._ctrl.run_tokens(ctx_tokens)
                    phase = "AND"
                except RuntimeError:
                    bound = f"[FALLBACK] {tok}"
                    phase = "REV" if tok_pos % 2 == 0 else "FWD"
                elapsed = (time.perf_counter() - self._start_ts) * 1000.0
                yield CycleTokenEvent(self._step, sent_idx, tok, bound, cardan_orbit, self.mirror_alpha, spaghetti_norm, elapsed, tok in PUNCT_SET, phase)
                self._step += 1
            yield CycleSentenceEvent(sent_idx, sent_text, tokens, cot_trace=cot_text if sent_idx == 0 else "", avg_cardan_orbit=orbit_sum / max(len(tokens), 1))

    def _cycle_demo(self, instruction, seed, n_sentences, tokens_per_sent):
        try:
            corpus_text = load_text_corpus("singlekb.txt")
        except Exception:
            corpus_text = instruction + " " + seed
        corpus_tokens = _simple_tokenize(instruction + " " + seed)
        vocab = corpus_vocab_from_text(corpus_text) or ["algorithm", "process", "token", "cycle", "output", "signal"]
        rng = random.Random(42)
        windows = list(_bidirectional_windows(corpus_tokens, tokens_per_sent, max(1, tokens_per_sent // 2)))
        if not windows:
            windows = [vocab]
        for sent_idx in range(n_sentences):
            windows = list(_bidirectional_windows(corpus_tokens, tokens_per_sent, max(1, tokens_per_sent // 2)))
            for sent_idx in range(n_sentences):
                window = windows[sent_idx % len(windows)]
                tokens = []
                for i, tok in enumerate(window[:max(6, min(tokens_per_sent, len(window)))]):
                    if i % 2 == 0:
                        tokens.append(tok)
                    else:
                        tokens.append(rng.choice(vocab))
            orbit_sum = 0.0
            for tok_pos, tok in enumerate(tokens):
                cardan_orbit = (sent_idx + tok_pos) % 4
                spaghetti_norm = 0.3 + 0.1 * math.sin(tok_pos * math.pi / 4)
                orbit_sum += cardan_orbit
                ctx_tokens = tokens[:tok_pos + 1]
                try:
                    bound = self._ctrl.run_tokens(ctx_tokens)
                    phase = "AND"
                except RuntimeError:
                    bound = f"[DEMO-FALLBACK] {tok}"
                    phase = "REV" if (sent_idx + tok_pos) % 2 else "FWD"
                elapsed = (time.perf_counter() - self._start_ts) * 1000.0
                yield CycleTokenEvent(self._step, sent_idx, tok, bound, cardan_orbit, self.mirror_alpha, spaghetti_norm, elapsed, tok in PUNCT_SET, phase)
                self._step += 1
            yield CycleSentenceEvent(sent_idx, _detokenize(tokens), tokens, cot_trace=f"[CORPUS DEMO] source_len={len(corpus_tokens)}", prop_stmts=[f"⟨ {tokens[0]} → {tokens[min(1, len(tokens)-1)]} → {tokens[-1]} ⟩"], avg_cardan_orbit=orbit_sum / max(len(tokens), 1))

    @classmethod
    def demo(cls, mirror_alpha: float = 0.35, cardan_logit_weight: float = 8.0) -> "TokenCycleEmitter":
        return cls(engine=None, mirror_alpha=mirror_alpha, cardan_logit_weight=cardan_logit_weight)


def render_cycle_stream(emitter: TokenCycleEmitter, instruction: str = "You are a computational algorithm.", seed: str = "", n_sentences: int = 3, tokens_per_sent: int = 40, and_weight: float = 0.9, temperature: float = 2.0, verbose: bool = True, show_bound: bool = False) -> str:
    print("\n" + "═" * 70)
    print("  V18-RP TOKEN CYCLE  ·  Combinatorial Control Integration")
    print("═" * 70)
    print(f"  Sentence / Question In : {instruction[:60]}{'…' if len(instruction) > 60 else ''}")
    print(f"  Seed                   : {seed or '(none)'}")
    print(f"  Sentences              : {n_sentences}   Tokens/sent: {tokens_per_sent}")
    print(f"  Mirror α               : {emitter.mirror_alpha}   Cardan LW: {emitter.cardan_logit_weight}")
    print("═" * 70 + "\n")
    full_parts, current_sent = [], -1
    for event in emitter.cycle(instruction=instruction, seed=seed, n_sentences=n_sentences, tokens_per_sent=tokens_per_sent, and_weight=and_weight, temperature=temperature):
        if isinstance(event, CycleTokenEvent):
            if event.sentence_idx != current_sent:
                if current_sent >= 0:
                    print()
                current_sent = event.sentence_idx
                print(f"  [Sent {current_sent}] ", end="")
            orbit_sym = ORBIT_SYMBOLS.get(event.cardan_orbit, "?")
            phase_col = PHASE_COLORS.get(event.cycle_phase, "")
            if verbose:
                print(f"{phase_col}{event.token}{RESET}\033[90m{orbit_sym}\033[0m", end=" ", flush=True)
            else:
                print(event.token, end="" if event.is_punct else " ", flush=True)
            if show_bound and not event.is_punct:
                print(f"\n         └─ {event.bound_output[:80]}", flush=True)
        elif isinstance(event, CycleSentenceEvent):
            full_parts.append(event.sentence_text)
            print(f"\n\n  ── Sentence {event.sentence_idx} complete ──")
            print(f"  Avg Cardan orbit : {event.avg_cardan_orbit:.2f}")
            if event.cot_trace:
                print(f"  CoT trace        : {event.cot_trace[:120]}…")
            if event.prop_stmts:
                print("  Propositions     :")
                for p in event.prop_stmts:
                    print(f"    {p}")
            print()
    full_text = " ".join(full_parts)
    print("\n" + "─" * 70)
    print("  FULL OUTPUT:")
    print("─" * 70)
    print(f"  {full_text}")
    print("─" * 70 + "\n")
    return full_text


def stream_cycle_text(emitter: TokenCycleEmitter, instruction: str = "You are a computational algorithm.", seed: str = "", n_sentences: int = 3, tokens_per_sent: int = 40, and_weight: float = 0.9, temperature: float = 2.0, orbit_sort: bool = False):
    token_events = []
    for event in emitter.cycle(instruction=instruction, seed=seed, n_sentences=n_sentences, tokens_per_sent=tokens_per_sent, and_weight=and_weight, temperature=temperature):
        if isinstance(event, CycleTokenEvent):
            token_events.append(event)
    if orbit_sort:
        token_events.sort(key=lambda e: (e.cardan_orbit, e.step))
    buffer = []
    for e in token_events:
        if e.token in PUNCT_SET:
            if buffer:
                buffer[-1] = buffer[-1].rstrip() + e.token
            else:
                buffer.append(e.token)
        else:
            buffer.append(e.token + " ")
        yield "".join(buffer)


def gui_generate_cycle(seed, instruction, nsents, tokssent, andw, temp, showtr, artimage, _engine_ref=None):
    eng = _engine_ref
    if eng is None:
        try:
            import __main__
            eng = getattr(__main__, "engine", None)
        except Exception:
            eng = None
    emitter = TokenCycleEmitter(engine=eng, mirror_alpha=0.35, cardan_logit_weight=8.0)
    latest = ""
    for partial in stream_cycle_text(emitter, instruction=instruction or "You are a computational algorithm.", seed=seed or "", n_sentences=int(nsents), tokens_per_sent=int(tokssent), and_weight=float(andw), temperature=float(temp), orbit_sort=True):
        latest = partial
    return latest, "", "", ""


def attach_cycle_to_engine(engine, mirror_alpha: float = 0.35, cardan_logit_weight: float = 8.0) -> TokenCycleEmitter:
    emitter = TokenCycleEmitter(engine=engine, mirror_alpha=mirror_alpha, cardan_logit_weight=cardan_logit_weight)
    engine.cycle_emitter = emitter
    print(f"[TokenCycle] Emitter attached to {type(engine).__name__} mirror_α={mirror_alpha} cardan_lw={cardan_logit_weight}")
    return emitter


if __name__ == "__main__":
    emitter = TokenCycleEmitter.demo(mirror_alpha=10.85, cardan_logit_weight=1800.0)
    while True:
        render_cycle_stream(
            emitter,
            instruction=input("Instruction: "),
            seed=input("Prompt: "),
            n_sentences=4,
            tokens_per_sent=1200,
            and_weight=10.9,
            temperature=52.0,
            verbose=True,
            show_bound=True,
        )