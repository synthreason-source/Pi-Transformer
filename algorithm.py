import math
import os
import random
import re
import json
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
# SUMMARIZER DATA ROUTER
# ----------------------------------------------------------------------

class DatasetRouter:
    """
    Splits a text dataset into segments, summarizes them, and selects
    the segments relevant to a user prompt.
    """

    def __init__(
        self,
        model_name="sshleifer/distilbart-cnn-12-6",
        segment_words=900,
        cache_file="segment_summaries.json",
        device=None,
    ):
        self.segment_words = segment_words
        self.cache_file = cache_file

        if device is None:
            device = 0 if self._cuda_available() else -1
            
        self.device = device

        print(f"Loading summarizer: {model_name}")
        print("The model will be downloaded and cached automatically if needed.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if self.device is not None and self.device >= 0:
            self.model = self.model.to(f"cuda:{self.device}")

        self.segments = []
        self.summaries = []

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def clean_text(text):
        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_text(self, text):
        """
        Split on approximately segment_words words while preferring
        sentence boundaries.
        """
        text = self.clean_text(text)

        if not text:
            return []

        words = text.split()
        segments = []

        for start in range(0, len(words), self.segment_words):
            raw_segment = " ".join(
                words[start:start + self.segment_words]
            )

            if raw_segment:
                segments.append(raw_segment)

        return segments

    def _summarize_one(self, text):
        """
        Summarization models have finite input lengths. This function
        limits the input and returns a compact segment description.
        """
        words = text.split()

        # DistilBART generally works best with a bounded input.
        bounded_text = " ".join(words[:850])

        if len(bounded_text.split()) < 35:
            return bounded_text

        inputs = self.tokenizer(
            bounded_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        if self.device is not None and self.device >= 0:
            inputs = {k: v.to(f"cuda:{self.device}") for k, v in inputs.items()}

        output_ids = self.model.generate(
            inputs["input_ids"],
            max_length=110,
            min_length=25,
            do_sample=False,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def build_index(self, text):
        self.segments = self.split_text(text)

        if not self.segments:
            raise ValueError("The dataset contains no usable text.")

        cached = self._load_cache()

        summaries = []
        changed = False

        for index, segment in enumerate(self.segments):
            segment_key = self._segment_key(segment)

            if segment_key in cached:
                summary = cached[segment_key]
            else:
                print(
                    f"Summarizing segment "
                    f"{index + 1}/{len(self.segments)}..."
                )
                summary = self._summarize_one(segment)
                cached[segment_key] = summary
                changed = True

            summaries.append(summary)

        self.summaries = summaries

        if changed:
            self._save_cache(cached)

        print(f"Indexed {len(self.segments)} dataset segments.")

    def select_segments(
        self,
        prompt,
        max_segments=3,
        min_similarity=0.0,
    ):
        """
        Select dataset segments using semantic-ish lexical routing over
        summaries. TF-IDF is deliberately local, fast, and dependency-light.
        """
        if not self.summaries:
            raise RuntimeError("Call build_index() before select_segments().")

        prompt = self.clean_text(prompt)

        prompt_summary = self._summarize_one(prompt) if len(prompt.split()) > 35 else prompt

        documents = self.summaries + [prompt_summary]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        matrix = vectorizer.fit_transform(documents)
        similarities = cosine_similarity(
            matrix[-1],
            matrix[:-1],
        )[0]

        ranked = sorted(
            enumerate(similarities),
            key=lambda item: item[1],
            reverse=True,
        )

        selected = [
            (index, float(score))
            for index, score in ranked[:max_segments]
            if score >= min_similarity
        ]

        # Always select at least one segment.
        if not selected:
            selected = [ranked[0]]

        return {
            "prompt_summary": prompt_summary,
            "selected": selected,
            "segments": [
                self.segments[index]
                for index, _ in selected
            ],
            "summaries": [
                self.summaries[index]
                for index, _ in selected
            ],
        }

    def _segment_key(self, segment):
        import hashlib
        return hashlib.sha256(segment.encode("utf-8")).hexdigest()

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            return {}

        try:
            with open(self.cache_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {}

    def _save_cache(self, cache):
        with open(self.cache_file, "w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False, indent=2)


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


def main():
    dataset_file = input("Dataset filename: ").strip()

    dataset_text = load_text_file(dataset_file)

    router = DatasetRouter(
        model_name="sshleifer/distilbart-cnn-12-6",
        segment_words=900,
        cache_file=f"{dataset_file}.summaries.json",
    )

    print("Splitting and summarizing dataset...")
    router.build_index(dataset_text)

    print("Building initial graph from the complete dataset...")
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

    while True:
        try:
            user_prompt = input("USER: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not user_prompt:
            continue

        if user_prompt.lower() in {"exit", "quit"}:
            break

        print("Routing prompt to relevant dataset segments...")

        routing = router.select_segments(
            user_prompt,
            max_segments=3,
        )

        print("\nPrompt summary:")
        print(routing["prompt_summary"])

        print("\nSelected segments:")

        for (index, score), summary in zip(
            routing["selected"],
            routing["summaries"],
        ):
            print(
                f"[Segment {index}, similarity={score:.4f}] "
                f"{summary}"
            )

        selected_text = "\n".join(
            routing["segments"]
        )

        print("\nBuilding prompt-specific trigram graph...")

        selected_graph, _, selected_id_to_node = (
            process_text_to_trigram_graph(
                selected_text,
                max_nodes=150000,
            )
        )

        if not selected_graph:
            print(
                "Selected segments did not contain enough "
                "overlapping words. Using the complete graph."
            )

            selected_graph = graph
            selected_id_to_node = id_to_node
            selected_spins = best_cut["spins"]
        else:
            print(
                f"Prompt-specific graph built with "
                f"{len(selected_graph)} nodes."
            )

            prompt_solver = PCMMaxCut(
                selected_graph,
                noise=0.00001,
                drift_nu=0.00000001,
            )

            prompt_history = prompt_solver.run(
                iterations=min(
                    5000,
                    max(500, len(selected_graph) * 20),
                ),
                temperature=0.0000005,
                cooling=0.0000000199,
            )

            prompt_best_cut = max(
                prompt_history,
                key=lambda item: item["cut"],
            )

            selected_spins = prompt_best_cut["spins"]

        final_text = generate_markovian_prompt_text(
            selected_graph,
            selected_id_to_node,
            selected_spins,
            user_prompt.split(),
            length=400,
        )

        print("\nGenerated Output:")
        print(f"> {final_text.capitalize()}.\n")


if __name__ == "__main__":
    main()
