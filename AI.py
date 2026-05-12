#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
π → BASE-26 → PYTORCH GEOMETRIC TRIGRAM GENERATOR
════════════════════════════════════════════════

ENHANCED WITH:
- PyTorch EnhancedTextProcessor integration
- TF-IDF + Geometric embeddings  
- Compass-based vertex processing
- Dynamic geometric term boosting
- Mohr-Mascheroni construction awareness
"""

import sys
import re
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque, Counter
from difflib import SequenceMatcher
from mpmath import mp, pi as mpi

import nltk
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.probability import (
    ConditionalFreqDist,
    ConditionalProbDist,
    LidstoneProbDist,
)
from nltk.corpus import words as nltk_words


# ============================================================
# CONFIG
# ============================================================

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(300_000)

PI_PREC = 15000
PI_STREAM_LEN = 12000
DIGITS_PER_SAMPLE = 3
NGRAM_N = 3
CONTEXT_WINDOW = NGRAM_N - 1
LIDSTONE_GAMMA = 0.1
GEN_WORDS = 160
WORD_FIND_MIN = 2


# ============================================================
# PYTORCH GEOMETRIC PROCESSOR (INTEGRATED)
# ============================================================

def custom_sigmoid(x):
    x_safe = torch.where(torch.abs(x) > torch.tensor(0.5), x, torch.exp(x) * 1.5)
    return torch.sigmoid(-5.0 / x_safe)

class MathProcessor(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.compass_radius_scale = 1.0
        
    def circle_circle_intersection(self, center1, radius1, center2, radius2):
        d = torch.norm(center2 - center1)
        intersect_condition = torch.logical_or(d <= (radius1 + radius2), d >= torch.abs(radius1 - radius2))
        if not intersect_condition.any():
            return torch.zeros(2, 2, device=self.device), torch.tensor(False, device=self.device)
        return torch.zeros(2, 2, device=self.device), torch.tensor(True, device=self.device)
        
    def compass_only_midpoint(self, point1, point2):
        center_dist = torch.norm(point2 - point1)
        radius = center_dist * self.compass_radius_scale
        intersections, valid = self.circle_circle_intersection(point1, radius, point2, radius)
        if valid:
            midpoint = (intersections[0] + intersections[1]) / 2
            return midpoint
        else:
            return (point1 + point2) / 2

class EnhancedTextProcessor(nn.Module):
    def __init__(self, num_neurons=256, device='cpu', vocab_limit=5000, max_features=1000):
        super().__init__()
        self.num_neurons = num_neurons
        self.device = device
        self.vocab_limit = vocab_limit
        self.word_to_idx = {}
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.ngram_cache = {}
        self.math_processor = MathProcessor(device=device)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z0-9]+\b',
            stop_words=None
        )
        
        self.tfidf_scaler = StandardScaler()
        self.is_vectorizer_fitted = False
        
        self.tfidf_projection = nn.Sequential(
            nn.Linear(max_features, num_neurons // 4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(num_neurons // 4, num_neurons // 4),
            nn.LayerNorm(num_neurons // 4)
        )
        
        self.word_embeddings = nn.Embedding(vocab_limit + 1, num_neurons // 4)
        self.position_embeddings = nn.Embedding(1000, num_neurons // 4)
        self.geometric_embeddings = nn.Embedding(100, num_neurons // 4)
        
        self.question_processor = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.LayerNorm(num_neurons)
        )
        
        self.compass_feature_processor = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons)
        )
        
        self.register_parameter('geometric_sigmoid_scale', nn.Parameter(torch.tensor(1.2)))
        self.register_parameter('tfidf_sigmoid_scale', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('question_weight', nn.Parameter(torch.tensor(0.3)))
        
        self.geometric_terms = {
            'compass': 0, 'circle': 1, 'intersection': 2, 'construction': 3,
            'midpoint': 4, 'perpendicular': 5, 'radius': 6, 'center': 7,
            'arc': 8, 'point': 9, 'line': 10, 'geometry': 11,
            'mohr': 12, 'theorem': 13, 'euclidean': 14,
            'straightedge': 15, 'triangle': 16, 'square': 17, 'polygon': 18,
            'angle': 19, 'bisector': 20, 'chord': 21, 'diameter': 22,
            'tangent': 23, 'secant': 24, 'vertex': 25, 'edge': 26
        }
        
        self.question_patterns = {
            'what': 0, 'how': 1, 'why': 2, 'when': 3, 'where': 4,
            'which': 5, 'who': 6, 'explain': 7, 'describe': 8,
            'define': 9, 'prove': 10, 'solve': 11, 'calculate': 12
        }
        
    def fit_vectorizer(self, documents):
        processed_docs = [' '.join(doc) if isinstance(doc, list) else doc for doc in documents]
        if not processed_docs: return
            
        self.vectorizer.fit(processed_docs)
        tfidf_matrix = self.vectorizer.transform(processed_docs)
        self.tfidf_scaler.fit(tfidf_matrix.toarray())
        self.is_vectorizer_fitted = True
        
        feature_names = self.vectorizer.get_feature_names_out()
        self.bigram_counts = Counter(name for name in feature_names if len(name.split()) == 2)
        self.trigram_counts = Counter(name for name in feature_names if len(name.split()) == 3)
    
    def get_geometric_boost(self, word):
        if word in self.geometric_terms:
            return torch.tensor(1.8, device=self.device)
        return torch.tensor(1.0, device=self.device)
    
    def process_text_features(self, text):
        if not self.is_vectorizer_fitted:
            return torch.zeros(self.num_neurons, device=self.device)
        
        tfidf_vec = self.vectorizer.transform([text]).toarray()[0]
        tfidf_scaled = torch.tensor(self.tfidf_scaler.transform([tfidf_vec])[0], 
                                  dtype=torch.float32, device=self.device)
        features = self.tfidf_projection(tfidf_scaled)
        return features


# ============================================================
# EMBEDDED CORPUS & TOKEN HELPERS (unchanged)
# ============================================================

def embedded_corpus():
    return """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it.

So she was considering in her own mind whether the pleasure of making a
daisy chain would be worth the trouble of getting up and picking the daisies.

Suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that, nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear!
I shall be late!"

When the Rabbit actually took a watch out of its waistcoat pocket and looked
at it and hurried on, Alice started to her feet.

The rabbit hole went straight on like a tunnel for some way and then dipped
suddenly down.

Either the well was very deep or she fell very slowly, for she had plenty of
time as she went down to look about her and wonder what was going to happen next.
"""

def tokenise_alpha(text):
    tokenizer = RegexpTokenizer(r"[a-z]+")
    return tokenizer.tokenize(text.lower())

def extract_word_pairs(prompt):
    words = [w.lower() for w in word_tokenize(prompt) if w.isalpha()]
    return list(ngrams(words, 2))

def capitalise_text(words):
    if not words: return ""
    txt = " ".join(words)
    chars = list(txt)
    if chars: chars[0] = chars[0].upper()
    for i in range(len(chars) - 2):
        if chars[i] == "." and chars[i + 1] == " ":
            chars[i + 2] = chars[i + 2].upper()
    txt = "".join(chars)
    txt = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', txt)
    return txt

def build_model(corpus):
    tokens = tokenise_alpha(corpus)
    padded = ["<s>"] * (NGRAM_N - 1) + tokens + ["</s>"]
    trigrams_ = list(ngrams(padded, NGRAM_N))
    cfd = ConditionalFreqDist((tuple(tg[:-1]), tg[-1]) for tg in trigrams_)
    vocab = set(tokens) | {"</s>"}
    for ctx in list(cfd.conditions()):
        if len(cfd[ctx]) == 0: cfd[ctx]["</s>"] += 1
    cpd = ConditionalProbDist(
        cfd,
        lambda fd: LidstoneProbDist(fd, gamma=LIDSTONE_GAMMA, bins=max(1, len(vocab)))
    )
    return cpd, vocab

def load_dictionary(vocab):
    try:
        words = {w.lower() for w in nltk_words.words() if w.isalpha()}
        return words | set(vocab)
    except Exception:
        return set(vocab)


# ============================================================
# ENHANCED π SAMPLER WITH PYTORCH
# ============================================================

class GeometricPiSampler:
    def __init__(self, stream, text_processor, temperature=2.5, top_k=100, top_p=1.0, repetition_penalty=1.08):
        self.stream = stream
        self.text_processor = text_processor
        self.pos = 0
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.history = Counter()
        self.device = text_processor.device
        
    def seek(self, pos):
        self.pos = pos % len(self.stream)
        self.history.clear()
    
    def next_unit(self):
        val = 0
        base = 26 ** DIGITS_PER_SAMPLE
        for _ in range(DIGITS_PER_SAMPLE):
            val = val * 26 + self.stream[self.pos % len(self.stream)]
            self.pos += 1
        return val / base
    
    def get_geometric_bias(self, word):
        """PyTorch geometric term boosting"""
        boost = self.text_processor.get_geometric_boost(word)
        return boost.item()
    
    def sample(self, dist, context_text=""):
        samples = list(dist.samples())
        if not samples: return "</s>"

        # Base scoring with repetition penalty + GEOMETRIC BOOSTING
        base_scored = []
        for s in samples:
            p = max(1e-12, float(dist.prob(s)))
            count = self.history[s]
            if count > 0: p /= (self.repetition_penalty ** count)
            
            # 🔥 PYTORCH GEOMETRIC BOOST
            geometric_boost = self.get_geometric_bias(s)
            p *= geometric_boost
            
            base_scored.append((s, p))

        # Temperature scaling
        scored = [(s, p ** (1.0 / self.temperature)) for s, p in base_scored]
        total = sum(p for _, p in scored)
        scored = [(s, p / total) for s, p in scored]

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:self.top_k]

        # Top-p filtering
        kept = []
        accum = 0.0
        for s, p in scored:
            kept.append((s, p))
            accum += p
            if accum >= self.top_p: break

        # 3 orthogonal π streams for XOR
        u_a = self.next_unit()
        u_b = self.next_unit() 
        u_c = self.next_unit()

        # XOR fusion with geometric context
        xor_scores = []
        for rank, (word, base_p) in enumerate(kept):
            idx = rank / max(1, len(kept) - 1)
            region_a = (1.0 - abs(idx - u_a)) * (1.0 - u_b) * (1.0 - u_c)
            region_b = u_b * (1.0 - abs(idx - u_a)) * (1.0 - u_c)
            region_c = u_c * (1.0 - u_a) * (1.0 - u_b)
            xor_blend = max(region_a, region_b, region_c)
            
            # Geometric orthogonality boost
            geo_boost = self.get_geometric_bias(word)
            final_p = base_p * xor_blend * geo_boost
            
            xor_scores.append((word, final_p))

        # Final selection
        xor_total = sum(p for _, p in xor_scores)
        if xor_total <= 0:
            chosen = kept[-1][0] if kept else "</s>"
        else:
            xor_scored = [(w, p / xor_total) for w, p in xor_scores]
            xor_draw = (u_a * (1-u_b) * (1-u_c) + u_b * (1-u_a) * (1-u_c) + u_c * (1-u_a) * (1-u_b)) / 1.5
            
            cumulative = 0.0
            chosen = xor_scored[-1][0]
            for word, p in xor_scored:
                cumulative += p
                if xor_draw < cumulative:
                    chosen = word
                    break

        self.history[chosen] += 1
        return chosen


# ============================================================
# TRIANGLE & GENERATION (ENHANCED)
# ============================================================

class Triangle:
    def __init__(self, stream_len, offset_extra=0, bend_degrees=13.0):
        base = offset_extra % stream_len
        bend_shift = int(round((bend_degrees / 360.0) * stream_len))
        self.A = base % stream_len
        self.B = (base + stream_len // 3 + bend_shift) % stream_len
        self.C = (base + 2 * stream_len // 3 + bend_shift) % stream_len
        self.vertices = {"A": self.A, "B": self.B, "C": self.C}

def generate_text(cpd, sampler, text_processor, prompt="", n_words=GEN_WORDS):
    seed_words = tokenise_alpha(prompt)
    if len(seed_words) >= CONTEXT_WINDOW:
        init = seed_words[-CONTEXT_WINDOW:]
    else:
        init = ["<s>"] * (CONTEXT_WINDOW - len(seed_words)) + seed_words

    context = deque(init, maxlen=CONTEXT_WINDOW)
    words = list(seed_words)

    context_text = " ".join(words[-10:])  # Recent context for geometric processing

    for _ in range(n_words):
        ctx = tuple(context)
        try:
            dist = cpd[ctx]
            samples = list(dist.samples())
        except Exception:
            samples = []

        if not samples:
            context.clear()
            context.extend(["<s>"] * CONTEXT_WINDOW)
            continue

        word = sampler.sample(dist, context_text)
        if word in ("</s>", ""):
            context.clear()
            context.extend(["<s>"] * CONTEXT_WINDOW)
            continue

        words.append(word)
        context.append(word)
        context_text = " ".join(words[-10:])

    return capitalise_text(words)


# ============================================================
# PROMPT SEARCH WITH PYTORCH (ENHANCED)
# ============================================================

def all_pairs_match(pairs, text, fuzzy_threshold=0.72):
    lower_text = text.lower()
    for pair in pairs:
        pair_str = " ".join(pair)
        if pair_str in lower_text: continue
        score = SequenceMatcher(None, pair_str, text).quick_ratio()
        if score < fuzzy_threshold: return False, pair
    return True, None

def brute_force_prompt_search(prompt, cpd, stream, text_processor, vertex="A", max_solutions=10):
    pairs = extract_word_pairs(prompt)
    if not pairs:
        print("No valid word pairs extracted.")
        return []

    print("\n🔥 Searching π-space with PyTorch Geometric Enhancement...")
    print(f"\nPrompt:\n{prompt}\n")

    found = []
    for bend_x10 in range(0, 451, 5):
        bend = bend_x10 / 10.0
        print(f"bend = {bend:.1f}", end=" ")

        for offset in range(0, PI_STREAM_LEN, 5):
            triangle = Triangle(PI_STREAM_LEN, offset_extra=offset, bend_degrees=bend)
            start = triangle.vertices[vertex]

            sampler = GeometricPiSampler(stream, text_processor)
            sampler.seek(start)

            text = generate_text(cpd, sampler, text_processor, prompt=prompt)

            matches_all, failed_pair = all_pairs_match(pairs, text)
            if matches_all:
                found.append({
                    "prompt": prompt, "bend": bend, "offset": offset,
                    "vertex": vertex, "text": text
                })
                print("\n🎉 GEOMETRIC MATCH FOUND!")
                print(f"bend={bend:.1f} offset={offset}")
                print("\n" + text + "\n")
                
                if len(found) >= max_solutions: return found
        print()
    return found


# ============================================================
# MAIN WITH PYTORCH INITIALIZATION
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Initialize PyTorch Geometric Processor
    text_processor = EnhancedTextProcessor(device=device).to(device)
    
    # Fit with Alice corpus + geometric training data
    training_docs = [embedded_corpus()] + [
        "geometry compass circle construction triangle vertex",
        "mohr mascheroni theorem compass only euclidean construction",
        "circle intersection midpoint perpendicular bisector radius",
        "straightedge triangle square polygon angle theorem"
    ]
    text_processor.fit_vectorizer(training_docs)
    
    print("✅ PyTorch Geometric Processor ready!")
    print(f"📊 Geometric terms: {len(text_processor.geometric_terms)}")
    print(f"📊 Bigram counts: {len(text_processor.bigram_counts)}")

    filename = input("\nCorpus filename (ENTER for embedded): ").strip()
    if filename:
        with open(filename, "r", encoding="utf-8") as f:
            corpus = f.read()
    else:
        corpus = embedded_corpus()

    print("\nBuilding trigram model...")
    cpd, vocab = build_model(corpus)
    dictionary = load_dictionary(vocab)

    print("Building π stream...")
    stream = build_pi_stream()

    while True:
        print("\n" + "="*60)
        print("🌐 PYTORCH π-GEOMETRIC PROMPT SEARCH")
        print("="*60)

        prompt = input("\nEnter prompt:\n> ").strip()
        if not prompt: continue

        results = brute_force_prompt_search(
            prompt, cpd, stream, text_processor, vertex="A", max_solutions=5
        )
        
        if not results:
            print("\n❌ No geometric matches found.")
        else:
            print(f"\n✅ Found {len(results)} PyTorch-enhanced solutions!")


def build_pi_stream(decimals=PI_PREC, length=PI_STREAM_LEN):
    mp.dps = decimals + 50
    D = 10 ** decimals
    frac = int(mp.floor(mpi * D)) - 3 * D
    stream = []
    for _ in range(length):
        frac *= 26
        stream.append(frac // D)
        frac %= D
    return stream


if __name__ == "__main__":
    main()