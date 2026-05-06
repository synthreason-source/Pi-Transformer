# --------------------------------------------------------------
#  π → Base‑26 → Letters → Real‑time NLTK word finder (fixed)
# --------------------------------------------------------------
import sys
from mpmath import mp, pi
from nltk.corpus import words   # NLTK supplies the word list

# Ensure NLTK data is available (run once: nltk.download('words'))
try:
    DICTIONARY = set(w.lower() for w in words.words() if len(w) >= 3)
except LookupError:
    import nltk
    nltk.download('words')
    DICTIONARY = set(w.lower() for w in words.words() if len(w) >= 3)

print(f"📖  Loaded {len(DICTIONARY):,} words from NLTK.\n")

# ---------- 1. Compute π in base‑26 ----------
PRECISION_DIGITS = 5000          # decimal digits of π
BASE26_LENGTH    = 3000          # how many base‑26 chars to generate
mp.dps = PRECISION_DIGITS + 50   # guard digits

pi_str = mp.nstr(pi, PRECISION_DIGITS, strip_zeros=False).replace(".", "")
n_frac = len(pi_str) - 1
D      = 10 ** n_frac
frac   = int(pi_str[1:])          # fractional part as integer over D

INT_CHAR = chr(ord('A') + 3)      # 3 → 'D'
base26_chars = []
for _ in range(BASE26_LENGTH):
    frac *= 26
    digit = frac // D
    frac  = frac % D
    base26_chars.append(chr(ord('A') + digit))

pi_letters = INT_CHAR + "".join(base26_chars)   # flat string for searching
flat       = pi_letters.lower()

# ---------- 2. Scan and report matches in real time ----------
found = {}   # word → first position (we keep only the earliest hit)
print("🔍  Scanning for English words (press Ctrl‑C to stop)…\n")
try:
    for start in range(len(flat)):
        # limit word length to a reasonable max (e.g., 15)
        for length in range(3, min(16, len(flat) - start + 1)):
            candidate = flat[start:start + length]
            if candidate in DICTIONARY and candidate not in found:
                found[candidate] = start
                # Show a short context snippet with the word highlighted
                lo = max(0, start - 4)
                hi = min(len(pi_letters), start + length + 4)
                ctx = pi_letters[lo:hi].upper()
                rel = start - lo
                highlighted = ctx[:rel] + f"[{ctx[rel:rel+len(candidate)]}]" + ctx[rel+len(candidate):]
                print(f"  {candidate:<12} @ {start:>5} …{highlighted}…")
except KeyboardInterrupt:
    print("\n✅  Scan stopped by user.")

# ---------- 3. Summary ----------
print("\n=== SUMMARY ===")
print(f"Unique words found : {len(found)}")
if found:
    longest = max(found, key=len)
    print(f"Longest word      : {longest} (length {len(longest)})")
    print(f"First occurrence  : {found[longest]}")
else:
    print("No words were found in the scanned range.")
