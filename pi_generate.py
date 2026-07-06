import re
import argparse
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import random

class MemoryStore:
    def __init__(self):
        self.values = []

    def add(self, text):
        text = text.strip()
        if text:
            self.values.append(text)

    def retrieve(self, prompt):
        if not self.values:
            return ""
        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        best = self.values[0]
        best_score = -1
        for item in self.values:
            item_words = set(re.findall(r"\w+", item.lower()))
            score = len(prompt_words.intersection(item_words))
            if score > best_score:
                best_score = score
                best = item
        return best

class TrigramWordEngine:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.vocab = []
        self.labels = []
        self.label_counts = Counter()
        self.label_bigrams = defaultdict(Counter)
        self.label_trigrams = defaultdict(Counter)
        self.lru = OrderedDict()
        self.bitshift_clusters = defaultdict(set)
        self.latent_curve = []
        self.global_shift = 0
        self.seed_history = []
        self.remap_memory = defaultdict(Counter)
        self.rule_memory = MemoryStore()

        # --- backward-direction indices, used only for the certainty duplex ---
        self.word_trigrams_rev = defaultdict(Counter)   # (w2, w3) -> Counter(w1)
        self.label_trigrams_rev = defaultdict(Counter)  # (l2, l3) -> Counter(l1)

    def _build_clusters(self, contexts):
        self.bitshift_clusters.clear()
        total_occurrences = max(1, sum(sum(ctx.values()) for ctx in contexts.values()))
        curve_exponent = 1.5
        for word, ctx in contexts.items():
            local_freq = sum(ctx.values())
            prob = (local_freq / total_occurrences) ** (1 / curve_exponent)
            bucket = int(prob * 1024)
            self.bitshift_clusters[bucket << 2].add(word)

        sorted_keys = sorted(self.bitshift_clusters.keys())
        self.latent_curve = [self.bitshift_clusters[k] for k in sorted_keys]

    def _remap_vocab(self):
        if not self.latent_curve:
            return set()

        self.global_shift = (self.global_shift + 1) % len(self.latent_curve)
        return self.latent_curve[self.global_shift]

    def remap_token(self, token, current_bucket):
        if token in self.remap_memory and self.remap_memory[token]:
            return self._sample(self.remap_memory[token], temperature=0.7)

        if token in current_bucket:
            return token

        return random.choice(list(current_bucket))

    def cluster_fallback(self, word):
        if self.latent_curve:
            idx = self.global_shift % len(self.latent_curve)
            words = self.latent_curve[idx]
            if words:
                return random.choice(list(words))

        if self.vocab:
            return random.choice(self.vocab)

        return None

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def _is_word(self, tok):
        return re.fullmatch(r"\w+", tok) is not None

    def _is_label(self, tok):
        return tok.startswith("label_") or tok.startswith("tag_")

    def _make_label(self, tok):
        if not self._is_word(tok):
            return None
        if len(tok) <= 3:
            return None
        if tok.isdigit():
            return None
        return tok

    def train(self, texts):
        contexts = defaultdict(Counter)
        all_tokens = []
        label_tokens = []

        for text in texts:
            tokens = self.tokenize(text)
            if not tokens:
                continue

            all_tokens.extend(tokens)

            for tok in tokens:
                lab = self._make_label(tok)
                if lab:
                    self.label_counts[lab] += 1
                    label_tokens.append(lab)

            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                self.bigrams[w1][w2] += 1
                self.lru[w1] = True
                self.lru.move_to_end(w1)

            for i in range(len(tokens) - 2):
                self.trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
                # backward index: given (w2, w3), what was w1?
                self.word_trigrams_rev[(tokens[i + 1], tokens[i + 2])][tokens[i]] += 1

            label_seq = [self._make_label(t) for t in tokens]
            label_seq = [t for t in label_seq if t]
            for i in range(len(label_seq) - 1):
                self.label_bigrams[label_seq[i]][label_seq[i + 1]] += 1
            for i in range(len(label_seq) - 2):
                self.label_trigrams[(label_seq[i], label_seq[i + 1])][label_seq[i + 2]] += 1
                # backward index: given (l2, l3), what was l1?
                self.label_trigrams_rev[(label_seq[i + 1], label_seq[i + 2])][label_seq[i]] += 1

        self.vocab = sorted(set(all_tokens))
        self.labels = sorted(set(label_tokens))
        self._build_clusters(contexts)

        while len(self.bigrams) > self.capacity:
            oldest, _ = self.lru.popitem(last=False)
            if oldest in self.bigrams:
                del self.bigrams[oldest]

    def _sample(self, counter, temperature=0.8):
        if not counter:
            return None

        words = list(counter.keys())
        counts = np.sqrt(np.array(list(counter.values()), dtype=np.float64))
        logits = np.log(counts + 1e-12) / max(float(temperature), 1e-6)
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return np.random.choice(words, p=probs)

    def next_word(self, w1, w2, temp):
        if (w1, w2) in self.trigrams:
            nxt = self._sample(self.trigrams[(w1, w2)], temp)
            if nxt:
                return nxt

        if w2 in self.bigrams:
            nxt = self._sample(self.bigrams[w2], temp)
            if nxt:
                return nxt

        return self.cluster_fallback(w2)

    def manifold_weight(self, t, d, a=6.0, b=1.0, c=0.1):
        return (1 - t) * a + t * b + ((4 ** d)/(t+1)) * c

    def label_weight(self, t, label_strength=0.5, a=2.5, b=0.8):
        return (1 - t) * a * label_strength + t ** b

    def _label_pool_from_prompt(self, prompt_words):
        pool = []
        for w in prompt_words:
            lab = self._make_label(w)
            if lab:
                pool.append(lab)
        return pool

    def _label_candidates(self, label_w1, label_w2):
        candidates = Counter()
        if (label_w1, label_w2) in self.label_trigrams:
            candidates.update(self.label_trigrams[(label_w1, label_w2)])
        if label_w2 in self.label_bigrams:
            candidates.update(self.label_bigrams[label_w2])
        if not candidates:
            if self.labels:
                candidates.update(Counter({lab: self.label_counts.get(lab, 1) for lab in self.labels}))
        return candidates

    # ------------------------------------------------------------------
    # Certainty duplex: a real, measurable answer to "which structure is
    # more predictable in this dataset" instead of an asserted claim.
    #
    # For any distribution P over "what comes next", Shannon entropy
    #   H(P) = -sum(p * log2(p))
    # is a standard, well-defined measure of uncertainty: H = 0 means the
    # outcome is always the same (fully certain); higher H means the
    # outcome is spread across more possibilities (less certain).
    #
    # "Duplex" here means we measure it in both directions:
    #   forward:  given (t1, t2), how uncertain is t3?
    #   backward: given (t2, t3), how uncertain is t1?
    # for both the word-level trigrams and the label-level trigrams, so we
    # can compare word-structure vs label-structure certainty head to head.
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(counter):
        if not counter:
            return None
        counts = np.array(list(counter.values()), dtype=np.float64)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _weighted_avg_entropy(table):
        """table: dict of context -> Counter(next_token -> count)."""
        total_weight = 0.0
        weighted_sum = 0.0
        per_bucket = {}
        for ctx, counter in table.items():
            h = TrigramWordEngine._entropy(counter)
            if h is None:
                continue
            weight = sum(counter.values())
            per_bucket[ctx] = h
            weighted_sum += h * weight
            total_weight += weight
        avg = (weighted_sum / total_weight) if total_weight > 0 else None
        return avg, per_bucket, total_weight

    def measure_certainty(self):
        """
        Computes forward/backward entropy for word-trigrams and
        label-trigrams across the whole trained dataset, and returns a
        report dict. Lower entropy = more certain / more predictable.
        """
        word_fwd_avg, word_fwd_buckets, word_fwd_n = self._weighted_avg_entropy(self.trigrams)
        word_bwd_avg, word_bwd_buckets, word_bwd_n = self._weighted_avg_entropy(self.word_trigrams_rev)
        label_fwd_avg, label_fwd_buckets, label_fwd_n = self._weighted_avg_entropy(self.label_trigrams)
        label_bwd_avg, label_bwd_buckets, label_bwd_n = self._weighted_avg_entropy(self.label_trigrams_rev)

        def safe(x):
            return None if x is None else round(x, 4)

        report = {
            "word": {
                "forward_entropy_bits": safe(word_fwd_avg),
                "backward_entropy_bits": safe(word_bwd_avg),
                "num_forward_observations": word_fwd_n,
                "num_backward_observations": word_bwd_n,
                "num_forward_contexts": len(word_fwd_buckets),
                "num_backward_contexts": len(word_bwd_buckets),
            },
            "label": {
                "forward_entropy_bits": safe(label_fwd_avg),
                "backward_entropy_bits": safe(label_bwd_avg),
                "num_forward_observations": label_fwd_n,
                "num_backward_observations": label_bwd_n,
                "num_forward_contexts": len(label_fwd_buckets),
                "num_backward_contexts": len(label_bwd_buckets),
            },
        }

        # Head-to-head verdict, based only on the numbers above.
        comparable = (
            report["word"]["forward_entropy_bits"] is not None
            and report["label"]["forward_entropy_bits"] is not None
        )
        if comparable:
            w = report["word"]["forward_entropy_bits"]
            l = report["label"]["forward_entropy_bits"]
            if abs(w - l) < 1e-6:
                verdict = "word-level and label-level structure are equally certain (tie) in this dataset"
            elif w < l:
                verdict = "word-level trigram structure is MORE certain (lower entropy) than label-level structure in this dataset"
            else:
                verdict = "label-level trigram structure is MORE certain (lower entropy) than word-level structure in this dataset"
        else:
            verdict = "not enough label data in this dataset to compare (no label trigrams observed)"

        report["verdict"] = verdict
        return report

    def _mix_word_and_label_candidates(self, word_candidates, label_candidates, t, prompt_set):
        mixed = Counter()

        total_word = sum(word_candidates.values()) if word_candidates else 0.0
        total_label = sum(label_candidates.values()) if label_candidates else 0.0

        if word_candidates:
            for tok, val in word_candidates.items():
                d = 0.0 if tok in prompt_set else 1.0
                mixed[tok] += val * self.manifold_weight(t, d)

        if label_candidates:
            for tok, val in label_candidates.items():
                label_strength = float(self.label_counts.get(tok, 1))
                mixed[tok] += val * self.label_weight(t, label_strength=max(label_strength, 1.0))

        label_mix = 0.5
        if total_word > 0 and total_label > 0:
            label_mix = 0.35 + 0.25 * np.sin(2 * np.pi * t)
            word_mix = 1.0 - label_mix
            for tok in list(mixed.keys()):
                if self._is_label(tok):
                    mixed[tok] *= label_mix
                else:
                    mixed[tok] *= word_mix

        total = sum(mixed.values())
        if total > 0:
            for tok in list(mixed.keys()):
                mixed[tok] /= total

        return mixed

    def generate_response(self, prompt, length=30, temperature=0.8):
        prompt_tokens = self.tokenize(prompt)
        prompt_words = [t for t in prompt_tokens if re.fullmatch(r"\w+", t)]

        if len(prompt_words) >= 2:
            w1, w2 = prompt_words[-2], prompt_words[-1]
        elif len(prompt_words) == 1:
            w1 = w2 = prompt_words[0]
        else:
            w1 = w2 = random.choice(self.vocab) if self.vocab else ""

        label_prompt = self._label_pool_from_prompt(prompt_words)
        if len(label_prompt) >= 2:
            lw1, lw2 = label_prompt[-2], label_prompt[-1]
        elif len(label_prompt) == 1:
            lw1 = lw2 = label_prompt[0]
        else:
            lw1 = lw2 = random.choice(self.labels) if self.labels else ""

        output = [w1, w2]
        prompt_set = set(prompt_words)

        for i in range(length):
            word_candidates = Counter()
            label_candidates = Counter()

            if (w1, w2) in self.trigrams:
                word_candidates.update(self.trigrams[(w1, w2)])
            if w2 in self.bigrams:
                word_candidates.update(self.bigrams[w2])

            if self.labels:
                label_candidates = self._label_candidates(lw1, lw2)

            if word_candidates or label_candidates:
                t = i / max(length - 1, 1)
                candidates = self._mix_word_and_label_candidates(
                    word_candidates,
                    label_candidates,
                    t,
                    prompt_set
                )
                nxt = self._sample(candidates, temperature)
            else:
                nxt = self.cluster_fallback(w2)

            if not nxt:
                break

            output.append(nxt)

            if self._is_label(nxt):
                lw1, lw2 = lw2, nxt
            else:
                w1, w2 = w2, nxt
                lab = self._make_label(nxt)
                if lab and lab in self.labels:
                    lw1, lw2 = lw2, lab

        return self.detokenize(output)

    def detokenize(self, tokens):
        out = []
        for tok in tokens:
            if not out:
                out.append(tok)
            elif tok.split():
                out[-1] +=  " " + tok
            else:
                out.append(" " + tok)
        return " ".join(out).strip()

def load_text_file(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        return [s.strip() for s in re.split(r"[.!?\n]+", f.read()) if s.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--certainty", action="store_true",
                         help="Print the forward/backward entropy certainty duplex report and exit.")
    args = parser.parse_args()

    engine = TrigramWordEngine()
    engine.train(load_text_file(args.training_file))

    if args.certainty:
        import json
        print(json.dumps(engine.measure_certainty(), indent=2))
        return

    while True:
        user = input("USER: ").strip()
        if not user:
            break
        print("AI:", engine.generate_response(user, args.length, args.temperature))

if __name__ == "__main__":
    main()
