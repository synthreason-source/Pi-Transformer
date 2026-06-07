"""
Generative Kripke System: A Markov-style text generator grounded in a semantic "world graph".

This module implements a lightweight generative language system inspired by:
- Kripke semantics (modal logic): worlds = nodes, accessibility = edges
- Markov chains: next word predicted from (word_i, word_{i+1}) pairs
- Semantic clustering: texts sharing tokens are linked into a "dense matrix"

The system:
1. Tokenizes input texts into nodes (called "worlds")
2. Builds a semantic graph: nodes overlap if they share non-stop-word tokens
3. For a given prompt:
   - Finds the most relevant starting world (node)
   - Collects its neighborhood (accessible worlds)
   - Builds a 2nd-order Markov model from token sequences in that neighborhood
   - Generates text by walking the Markov chain until token limit or dead end

Use case:
- Prototype for generative knowledge bases
- Stylized text generation from domain-specific corpora
- Exploratory "memory" system where each node is a memory fragment

Author: George W (AI developer, Melbourne)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Iterable, Any
from collections import defaultdict, Counter
import os
import math
import random

# Alias for world identifiers in the Kripke-style graph
WorldId = str


def _tok(text: str) -> List[str]:
    """
    Cleans and tokenizes text into lowercase words.
    
    Steps:
    1. Lowercase the input
    2. Split on whitespace
    3. Strip non-alphanumeric characters except '-' and "'"
    4. Drop empty tokens
    
    Returns a list of cleaned tokens.
    """
    t = text.lower()
    words = [w for w in t.split() if w]
    out = []
    for w in words:
        w = "".join(ch for ch in w if ch.isalnum() or ch in {"-", "'"})
        if w:
            out.append(w)
    return out


def _cap(text: str) -> str:
    """
    Capitalizes the first character of a string, leaving the rest unchanged.
    
    Used to ensure generated prose starts with a capital letter.
    """
    text = text
    if not text:
        return text
    return text[0] + text[1:]


def load_txt_file(path: str) -> List[str]:
    """
    Loads a text file and splits it into "sentences" using '.' as delimiter.
    
    Note:
    - This is a simple splitter; it does not handle abbreviations, quotes, etc.
    - Each element in the returned list is one "segment" from the original text.
    
    Parameters:
        path: Path to the .txt file
    
    Returns:
        List of text segments (roughly "sentences")
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return [ln for ln in raw.split(".")]


@dataclass
class MemoryNode:
    """
    Represents a single "world" in the Kripke-style semantic graph.
    
    Each node is a memory fragment / text snippet with:
    - id: unique world identifier
    - text: original raw text
    - tokens: cleaned token list
    - token_set: set of tokens for fast overlap checks
    - weight: semantic importance score derived from metrics
    """
    id: WorldId
    text: str
    tokens: List[str]
    token_set: Set[str]
    weight: float = 1.0


class GenerativeKripkeSystem:
    """
    A generative text system built on a semantic graph of "worlds".
    
    Conceptual model:
    - Worlds: MemoryNode instances, each representing a text fragment
    - Accessibility relation: edges between worlds that share content
    - Generation: Markov chain walk over tokens in the accessible neighborhood
    
    Key methods:
    - add_node: insert a new world with text and metrics
    - build_dense_matrix: connect worlds that share any non-stop-word token
    - find_entry: choose the best starting world for a given prompt
    - generate: produce new text using a 2nd-order Markov model from the neighborhood
    """

    def __init__(self):
        """
        Initialize the system with:
        - Empty node graph (worlds)
        - Empty edge graph (accessibility relation)
        - Random number generator for stochastic generation
        - Stop-word set to filter out common, low-information tokens
        """
        self.nodes: Dict[WorldId, MemoryNode] = {}
        self.edges: Dict[WorldId, List[WorldId]] = defaultdict(list)
        self.rng = random.Random()
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "of", "to", "in",
            "is", "it", "that", "was", "for", "on", "as", "with", "this"
        }

    def add_node(self, node_id: WorldId, text: str, metrics: Tuple[float, float, float]):
        """
        Add a new world (node) to the system.
        
        Parameters:
            node_id: Unique identifier for this world (e.g. "w0", "w1")
            text: Raw text content of this memory fragment
            metrics: Tuple (desire, emotion, salience) used to compute weight
            
        Weight computation:
            weight = log(1 + desire + 0.5*emotion + 0.25*salience) + 1
        This gives higher weight to fragments with stronger desire/emotion/salience.
        """
        toks = _tok(text)
        weight = math.log(1.0 + metrics[0] + 0.5 * metrics[1] + 0.25 * metrics[2]) + 1.0
        self.nodes[node_id] = MemoryNode(
            id=node_id,
            text=text,
            tokens=toks,
            token_set=set(toks),
            weight=weight
        )

    def build_dense_matrix(self):
        """
        Build the semantic web (accessibility relation) between worlds.
        
        Rule:
        - Two worlds are linked if they share ANY non-stop-word token.
        - Each world is also linked to itself (reflexive accessibility).
        
        This creates a "dense" graph where semantically related fragments
        form clusters, enabling localized generation.
        """
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
        """
        Find the most relevant starting world for a given prompt.
        
        Strategy:
        - Tokenize the prompt
        - For each world, count how many prompt tokens appear in its token set
        - Return the world with the highest overlap score
        
        If no tokens or no nodes, return the first node (or empty string).
        """
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
        """
        Generate new text based on a prompt using a Markov-Kripke approach.
        
        Pipeline:
        1. Find the best starting world (node) for the prompt.
        2. Gather all accessible worlds (neighbors in the graph).
        3. Build a 2nd-order Markov table from token sequences in those worlds:
           - Key: (word_i, word_{i+1})
           - Value: list of possible word_{i+2}
        4. Initialize with a random starting pair from the neighborhood.
        5. Walk the Markov chain until:
           - max_words reached, or
           - no continuation for current pair (dead end)
        6. Append a metadata clause with top keywords from the cluster.
        7. Truncate to max_words if needed.
        
        Parameters:
            prompt: User input to guide generation
            max_words: Maximum number of words in generated output
        
        Returns:
            Generated prose string, capitalized and punctuated.
        """
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
    """
    Build a GenerativeKripkeSystem from a list of text fragments.
    
    For each text:
    - Tokenize
    - Skip if too short (< 3 tokens)
    - Compute metrics:
        desire: frequency of desire-related words (want, desire, need, seek)
        emotion: frequency of emotion-related words (love, fear, dark, joy, sad)
        salience: ratio of unique tokens to total tokens (diversity)
    - Add as a node with id "w{i}"
    
    Then build the dense semantic matrix (connect overlapping worlds).
    
    Parameters:
        texts: List of text fragments (e.g. sentences, paragraphs)
    
    Returns:
        A fully built GenerativeKripkeSystem
    """
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
    """
    Interactive CLI demo for the Generative Kripke System.
    
    Behavior:
    - If 'singlekb.txt' exists:
        - Load its contents as text fragments
        - Build the system
        - Enter an interactive loop:
            - Ask USER for a prompt
            - Generate and print text
    - If 'singlekb.txt' does not exist:
        - Run in mock mode with a small example corpus
        - Print one generated example
    
    Example usage:
        $ python your_script.py
        GPU/Kripke: Initializing Markov-Kripke synthesis framework from xaa.txt...
        Graph locked with 42 responsive logic nodes.
        USER: heavy dark frameworks
        Generated text: Dark frameworks reveal heavy data streams and dark structures...
    """
    if os.path.exists("singlekb.txt"):
        print("Initializing Markov-Kripke synthesis framework from xaa.txt...")
        lines = load_txt_file("singlekb.txt")
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
