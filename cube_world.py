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

Completely deterministic, set since the beginning of the document...
Somehow manifest in everyday life? a one sided proof?
"""

import hashlib
import sys
import itertools

with open(input("Filename: "), "r", encoding="utf-8") as file:
    WORDS = file.read().split()


def sha256_hex(word: str) -> str:
    return hashlib.sha256(word.encode("utf-8")).hexdigest()


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for d in range(3, int(n ** 0.5) + 1, 2):
        if n % d == 0:
            return False
    return True


def primes():
    """Infinite generator of prime numbers: 2, 3, 5, 7, 11, ..."""
    n = 1
    while True:
        if is_prime(n) or n == 1:
            yield n
        n += 1


def resolve(word: str):
    """Returns (hex_digest, list of (slice, int_value, index, resolved_word))."""
    hex_digest = sha256_hex(word)

    # 64 hex chars total -> 8 slices of 8 hex chars (4 bytes) each
    slices = [hex_digest[i:i + 8] for i in range(0, 64, 8)]

    results = []
    # Outer loop: every prime F < 1000 (2, 3, 5, 7, ..., 997)
    # Inner loop: all 8 byte-slices, re-resolved against that prime
    for F in itertools.takewhile(lambda p: p < 1000, primes()):
        for s in slices:
            n = int(s, 16)
            divisor = max(1, len(WORDS) // F)  # guard against len(WORDS) < F
            idx = n % divisor
            results.append((s, n, idx, WORDS[idx]))
            break
    return hex_digest, results


def report(word: str):
    hex_digest, results = resolve(word)

    # Each prime F contributes a block of 8 rows (one per byte-slice).
    block = 8
    first_block = results[:block]

    unresolved_sum = first_block[0][3]                     # slice 0, F=2
    hotswap = first_block[1][3]                            # slice 1, F=2
    transitive_product = " · ".join(r[3] for r in first_block[2:6])  # slices 2-5, F=2

    print(f"\nword:  {word}")
    print(f"sha256: {hex_digest}\n")

    print(f"{'unresolved sum':20} {unresolved_sum}")
    print(f"{'transitive product':20} {transitive_product}")
    print()

    print(f"{'F':>6}   {'byte slice':10} {'index':>6}   word")
    print("-" * 44)
    for offset, F_prime in zip(range(0, len(results), block),
                                itertools.takewhile(lambda p: p < 1000, primes())):
        
        if offset + 3 < len(results):
            s, n, idx, w = results[offset + 3]  # the 4th slice (index 3) within this F-block
            print(f"{F_prime:>6}   {s:10} {idx:>6}   {w}")
    print()


if __name__ == "__main__":

    while True:
        if len(sys.argv) > 1:
            input_word = " ".join(sys.argv[1:])
        else:
            input_word = input("drop a word into the cube: ").strip() or "catch-22"

        report(input_word)
        print()
