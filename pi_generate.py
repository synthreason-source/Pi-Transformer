from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Iterable, Any
from collections import defaultdict, Counter
import os
import math
import random

WorldId = str


def _tok(text: str) -> List[str]:
    """Cleans and tokenizes text into lowercase words."""
    t = text.lower()
    words = [w for w in t.split() if w]
    out = []
    for w in words:
        w = "".join(ch for ch in w if ch.isalnum() or ch in {"-", "'"})
        if w:
            out.append(w)
    return out


def _cap(text: str) -> str:
    text = text
    if not text:
        return text
    return text[0] + text[1:]


def load_txt_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return [ln for ln in raw.split(".")]


@dataclass
class MemoryNode:
    id: WorldId
    text: str
    tokens: List[str]
    token_set: Set[str]
    weight: float = 1.0


class GenerativeKripkeSystem:
    def __init__(self):
        self.nodes: Dict[WorldId, MemoryNode] = {}
        self.edges: Dict[WorldId, List[WorldId]] = defaultdict(list)
        self.rng = random.Random()
        self.stop_words = {"the", "a", "an", "and", "or", "but", "of", "to", "in", "is", "it", "that", "was", "for", "on", "as", "with", "this"}

    def add_node(self, node_id: WorldId, text: str, metrics: Tuple[float, float, float]):
        toks = _tok(text)
        weight = math.log(1.0 + metrics[0] + 0.5 * metrics[1] + 0.25 * metrics[2]) + 1.0
        self.nodes[node_id] = MemoryNode(id=node_id, text=text, tokens=toks, token_set=set(toks), weight=weight)

    def build_dense_matrix(self):
        """Builds a semantic web. If sentences share ANY non-stop-word, they link."""
        node_ids = list(self.nodes.keys())
        for i, id1 in enumerate(node_ids):
            n1 = self.nodes[id1]
            content1 = n1.token_set - self.stop_words
            
            # Every node can always access itself
            self.edges[id1].append(id1)
            
            for id2 in node_ids[i+1:]:
                n2 = self.nodes[id2]
                content2 = n2.token_set - self.stop_words
                
                if content1 & content2:
                    self.edges[id1].append(id2)
                    self.edges[id2].append(id1)

    def find_entry(self, prompt: str) -> WorldId:
        """Finds the most relevant cluster starting point."""
        ptoks = set(_tok(prompt))
        if not ptoks:
            return next(iter(self.nodes.keys())) if self.nodes else ""
            
        best_id = list(self.nodes.keys())[0]
        best_score = -1
        
        for nid, node in self.nodes.items():
            score = len(ptoks & node.token_set)
            if score > best_score:
                best_score = score
                best_id = nid
        return best_id

    def generate(self, prompt: str, max_words: int = 150) -> str:
        start_id = self.find_entry(prompt)
        if not start_id or start_id not in self.edges:
            return "The system network is uninitialized or isolated."

        # Grab all mathematically accessible nodes in the neighborhood
        neighbors = self.edges[start_id]
        weighted_neighbors = [(nid, self.nodes[nid].weight) for nid in neighbors]
        weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        top_nodes = [self.nodes[nid] for nid, _ in weighted_neighbors]
        if not top_nodes:
            return "No accessible transitions calculated."

        # --- MARKOV TRANSITION ENGINE ---
        # Build lookahead probability distributions from token sets in the neighborhood
        markov_table: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        starting_pairs: List[Tuple[str, str]] = []

        for node in top_nodes:
            words = node.tokens
            if len(words) < 3:
                continue
            # Keep track of natural sentence entry states
            starting_pairs.append((words[0], words[1]))
            for i in range(len(words) - 2):
                key = (words[i], words[i+1])
                markov_table[key].append(words[i+2])

        if not markov_table:
            # Absolute fallback if states don't meet minimum length
            return _cap(top_nodes[0].text)

        # Initialize the state walk sequence
        current_pair = self.rng.choice(starting_pairs) if starting_pairs else list(markov_table.keys())[0]
        generated_tokens = [current_pair[0], current_pair[1]]

        # Traverse state space until token limits or dead ends are arrived at
        for _ in range(max_words - 2):
            if current_pair in markov_table:
                possibilities = markov_table[current_pair]
                next_word = self.rng.choice(possibilities)
                generated_tokens.append(next_word)
                current_pair = (current_pair[1], next_word)
            else:
                # Stochastic loop break if we run off the text margin
                break

        base_prose = " ".join(generated_tokens) + "."
        
        # Pull thematic indicators from the graph cluster
        keywords = Counter()
        for node in top_nodes:
            keywords.update([w for w in node.tokens if w not in self.stop_words])
        
        modal_terms = [k for k, _ in keywords.most_common(5)]
        meta_clause = f" [Vector trace patterns: {', '.join(modal_terms)}]" if modal_terms else ""

        output = _cap(base_prose)
        
        # Enforce hard bounds limits
        res_words = output.split()
        if len(res_words) > max_words:
            output = " ".join(res_words[:max_words])
        return output


def build_system_from_texts(texts: List[str]) -> GenerativeKripkeSystem:
    sys = GenerativeKripkeSystem()
    for i, txt in enumerate(texts):
        toks = _tok(txt)
        if len(toks) < 3:
            continue
        desire = min(1.0, sum(1 for t in toks if t in {"want", "desire", "need", "seek"}) / max(1, len(toks)))
        emotion = min(1.0, sum(1 for t in toks if t in {"love", "fear", "dark", "joy", "sad"}) / max(1, len(toks)))
        salience = min(1.0, len(set(toks)) / max(1, len(toks)))
        
        sys.add_node(f"w{i}", txt, (desire, emotion, salience))
        
    sys.build_dense_matrix()
    return sys


if __name__ == "__main__":
    if os.path.exists("x.txt"):
        print("Initializing Markov-Kripke synthesis framework from xaa.txt...")
        lines = load_txt_file("x.txt")
        system = build_system_from_texts(lines)
        print(f"Graph locked with {len(system.nodes)} responsive logic nodes.")
        while True:
            try:
                prompt = input("USER: ")
                if not prompt:
                    continue
                print("Generated text:", system.generate(prompt))
                print("-" * 50)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print("Mock execution (xaa.txt not found):")
        mock_data = [
            "We seek heavy data streams and dark structures in the system framework.",
            "The heavy code architecture holds things we desire to parse.",
            "Dark frameworks reveal processing patterns within the storage system."
        ]
        system = build_system_from_texts(mock_data)
        print("Output:", system.generate("heavy dark frameworks"))