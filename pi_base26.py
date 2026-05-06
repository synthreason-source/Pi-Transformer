# --------------------------------------------------------------
#  π → Base‑26 → Letters → Real‑time NLTK word finder (streaming)
# --------------------------------------------------------------
import sys
from mpmath import mp, pi as mpi
from nltk.corpus import words   # NLTK supplies the word list
import nltk
from collections import deque

# ------------------------------------------------------------------
# 1. Load dictionary (NLTK words) – done once at start
# ------------------------------------------------------------------
try:
    DICTIONARY = set(w.lower() for w in words.words() if len(w) >= 3)
except LookupError:                     # first‑time run: download the corpus
    nltk.download('words')
    DICTIONARY = set(w.lower() for w in words.words() if len(w) >= 3)

print(f"📖  Loaded {len(DICTIONARY):,} words from NLTK.\n")

# ------------------------------------------------------------------
# 2. Streaming π decimal digits (chunk‑by‑chunk)
# ------------------------------------------------------------------
def pi_decimal_digits(chunk_size: int = 1000):
    """
    Yield decimal digits of π one‑by‑one (as ints) after the leading '3'.
    Uses mpmath to extend precision in chunks.
    """
    offset = 0                     # how many fractional digits already yielded
    while True:
        mp.dps = offset + chunk_size + 5   # compute a little extra for safety
        s = mp.nstr(mpi, mp.dps, strip_zeros=False).replace('.', '')
        # s[0] is the integer part '3'; fractional part starts at s[1]
        frac = s[1 + offset : 1 + offset + chunk_size]
        if not frac:               # no more digits (should never happen with mpmath)
            return
        for ch in frac:
            yield int(ch)
        offset += chunk_size

# ------------------------------------------------------------------
# 3. Convert streaming decimal digits → base‑26 letters on the fly
# ------------------------------------------------------------------
def pi_base26_letters():
    """
    Generator that yields base‑26 letters from the fractional part of π.
    It keeps the fractional value as a big integer numerator/denominator
    and refreshes it when more decimal digits are needed.
    """
    dec_gen = pi_decimal_digits()
    # State for the exact rational value of the fractional part seen so far
    num = 0          # numerator
    den = 1          # denominator

    for dec_digit in dec_gen:
        # Incorporate the new decimal digit into the fraction:
        #   new_value = old_value * 10 + dec_digit
        num = num * 10 + dec_digit
        den *= 10

        # Extract one base‑26 digit (multiply by 26, take integer part)
        num *= 26
        digit = num // den
        num = num % den          # keep remainder for next step
        yield chr(ord('A') + digit)

# ------------------------------------------------------------------
# 4. Real‑time word search over the streaming letter sequence
# ------------------------------------------------------------------
def stream_and_find_words():
    buffer = deque()          # holds recent letters (lower‑case)
    found  = set()            # (word, position) pairs already reported
    pos    = 0                # 0‑based index in the infinite base‑26 stream
    MIN_LEN, MAX_LEN = 3, 15  # reasonable word length bounds

    for letter in pi_base26_letters():
        buffer.append(letter.lower())
        pos += 1

        # Keep buffer from growing unbounded (we only need to look back MAX_LEN)
        if len(buffer) > MAX_LEN + 20:
            buffer.popleft()

        buf_str = ''.join(buffer)
        buf_len = len(buf_str)

        # Only substrings that END at the newest character can be new discoveries
        for length in range(MIN_LEN, min(MAX_LEN, buf_len) + 1):
            start_idx = buf_len - length
            word = buf_str[start_idx:]
            if word in DICTIONARY and len(word) > 7:
                global_start = pos - length          # position in the infinite stream
                key = (word, global_start)
                if key in found:
                    continue
                found.add(key)

                # Build a short context snippet with the word highlighted
                ctx_lo = max(0, start_idx - 4)
                ctx_hi = min(buf_len, buf_len + 4)
                ctx = buf_str[ctx_lo:ctx_hi].upper()
                rel = start_idx - ctx_lo
                highlighted = (
                    ctx[:rel] +
                    f"[{ctx[rel:rel+len(word)]}]" +
                    ctx[rel+len(word):]
                )
                print(f"{word:<12} @ {global_start:>8} …{highlighted}…", flush=True)

# ------------------------------------------------------------------
# 5. Main entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        stream_and_find_words()
    except KeyboardInterrupt:
        print("\n✅  Scan stopped by user.")
        print("\n=== SUMMARY ===")
        print(f"Unique words found : {len(found)}")
        if found:
            longest = max(found, key=lambda kv: len(kv[0]))[0]
            print(f"Longest word      : {longest} (length {len(longest)})")
            # Show the earliest occurrence of that longest word
            earliest_pos = min(pos for w, pos in found if w == longest)
            print(f"First occurrence  : {earliest_pos}")
        else:
            print("No words were found in the scanned range.")
