#!/usr/bin/env python3
"""
cube_world_hash_resolver.py

Takes a word, hashes it with SHA-256, slices the digest into eight
4-byte chunks, and maps each chunk (mod len(WORDS)) to a word in a
fixed list. Deterministic: same input always resolves to the same
output words.


If indeed our 2d world is actually cube world then the transitive
product of everyday predictions could be represented in such a way...
4D time slice with an unresolved sum or hotswap on either end, a catch 22.

Usage:
    python3 cube_world_hash_resolver.py "catch-22"
    python3 cube_world_hash_resolver.py          # prompts interactively
"""

import hashlib
import sys

with open(input("Filename: "), "r", encoding="utf-8") as file:
    WORDS = file.read().split()


def sha256_hex(word: str) -> str:
    return hashlib.sha256(word.encode("utf-8")).hexdigest()


def resolve(word: str):
    """Returns (hex_digest, list of (slice, int_value, index, resolved_word))."""
    hex_digest = sha256_hex(word)

    # 64 hex chars total -> 8 slices of 8 hex chars (4 bytes) each
    slices = [hex_digest[i:i + 8] for i in range(0, 64, 8)]

    results = []
    for s in slices:
        n = int(s, 16)
        idx = n % len(WORDS)
        results.append((s, n, idx, WORDS[idx]))

    return hex_digest, results


def report(word: str):
    hex_digest, results = resolve(word)

    unresolved_sum = results[0][3]          # slice 0
    hotswap = results[1][3]                 # slice 1
    transitive_product = " · ".join(r[3] for r in results[2:6])  # slices 2-5

    print(f"\nword:  {word}")
    print(f"sha256: {hex_digest}\n")

    print(f"{'unresolved sum':20} {unresolved_sum}")
    print()

    print(f"{'byte slice':10} {'index':>6}   word")
    print("-" * 34)
    i = 0
    for s, n, idx, w in results:

        if i == 3:
            print(f"{s:10} {idx:>6}   {w}")
        i += 1 
    print()


if __name__ == "__main__":
    
    while True:
        if len(sys.argv) > 1:
            input_word = " ".join(sys.argv[1:])
        else:
            input_word = input("drop a word into the cube: ").strip() or "catch-22"

        report(input_word)
        print()
