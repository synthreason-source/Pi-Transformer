"""
Pi → Base-26 → Letters → Dictionary Word Finder
Maps each base-26 digit of π to a letter (0=A … 25=Z),
then hunts for real English words in the resulting string.
"""

from mpmath import mp, pi
from collections import defaultdict
import re, sys

# set_int_max_str_digits only exists in Python 3.11+ (older versions have no limit)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100_000)

# ── 1. Load English dictionary (tries multiple sources) ───────────────────────

def load_from_hunspell(path):
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)  # skip word-count header line
        for line in f:
            word = line.split("/")[0].strip().lower()
            if word.isalpha() and len(word) >= 3:
                words.add(word)
    return words

def load_from_plaintext(path):
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip().lower()
            if word.isalpha() and len(word) >= 3:
                words.add(word)
    return words

def load_dictionary():
    # 1) Common hunspell / aspell locations
    hunspell_paths = [
        "/usr/share/hunspell/en_US.dic",
        "/usr/share/myspell/en_US.dic",
        "/Library/Spelling/en_US.dic",
        "C:/Program Files/LibreOffice/share/extensions/dict-en/en_US.dic",
    ]
    for p in hunspell_paths:
        try:
            words = load_from_hunspell(p)
            print(f"    Loaded hunspell dict from {p}")
            return words
        except FileNotFoundError:
            pass

    # 2) Plain-text word lists (e.g. /usr/share/dict/words on macOS/Linux)
    plain_paths = [
        "/usr/share/dict/words",
        "/usr/share/dict/american-english",
        "/usr/share/dict/british-english",
        "/usr/dict/words",
    ]
    for p in plain_paths:
        try:
            words = load_from_plaintext(p)
            print(f"    Loaded plain word list from {p}")
            return words
        except FileNotFoundError:
            pass

    # 3) Try downloading a word list
    import urllib.request
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    try:
        print("    Downloading word list...", flush=True)
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode()
        words = {w.strip().lower() for w in text.splitlines()
                 if w.strip().isalpha() and len(w.strip()) >= 3}
        print(f"    Downloaded {len(words):,} words from GitHub")
        return words
    except Exception as e:
        print(f"    Download failed: {e}")

    # 4) Last resort — use Python's own keyword list + a small built-in corpus
    import keyword
    words = {w.lower() for w in keyword.kwlist if len(w) >= 3}
    # Add a compact but solid English word set
    fallback = """
    the and for are but not you all can her was one our out day get has him
    his how its new now old see two way who boy did its let put say she too
    use able back ball band bank base bath bear beat been bird blow blue boat
    body bone book born both burn call came card care case city come cool
    cost dark dead deal dear deep door down draw drop dust each east easy
    edge else even face fact fall farm fast felt find fine fire fish five flat
    flew food foot form four free from full game gave give glad gold gone
    good grow hair half hand hard have head hear heat held help here hide
    high hill hold hole home hope horn huge hunt idea join jump just keep
    kept kind knew know land last late lead leaf left less life lift like
    line list live load lock long look lord lose love made main make many
    mark mass mean meet milk mind mine miss more most move much must name
    near need next nice nine none noon more note once only open over paid
    pain pair park part pass past path pick plan play poor pull push race
    rain rang read real rest rice rich ride ring rise road rock roll room
    root rope rose rule rule safe said sail salt same sand save seed sell
    sent sets shed ship shoe shop show shut side sign silk sing sink size
    skin slip slow snow soft soil sold some song soon sort soul spin star
    stay stem step stop such suit sure swim tail take talk tall tape task
    test than them then they thin this thus tide tied till time tiny tire
    told tone took tool tops torn tour town tree trim trip true tube tune
    turn upon used very view vote wait wake walk wall want warm wash wave
    weak wear week well went were west what when whom wide wife wild will
    wind wine wing wish with wolf wood word wore work worn wrap year your
    zero zone able acid aged also arch area baby back bail bake bare bark
    barn base bath bead beam bean beat been bell belt best bite blow blur
    bond bony boom boot bore both bowl brow buck bulb bulk bull bump bush
    busy buzz cake calm camp cane cape cart cash cast cave cent chin chip
    chop clam clap clay clip club coal coat coil cord core cork corn coup
    cove crab crop crow cube cure curl curt cute dale dame damp dare dart
    dash dawn daze deem dent desk dial dice dike dill dime dine dip dire
    disc dish disk dive dock does dome dote dour dove drab drag dram dray
    drip drop drug drum dual dull dump dune dusk duty dyer earl earn east
    eave edit else emit epic even ever ewer exam exit fad faint fame fang
    fare fawn faze fern fife fill film find fits fizz flag flak flap flat
    flaw flea fled flew flex flip flit flog flop flow foam fold fond font
    ford fore fork form fort foul fowl fray frog from fund funk furl fury
    fuse fuss gate gaze gear gild gilt gird girl gist glad glee glen glob
    glue glum gnat gnaw goat gobs gong gore gout gown grad gram grim grip
    grit grog grub gust hack hail hair hale halt hank hare harp hash haste
    hate haze heed heel helm hemp herb herd hiss hive hoax hock hone hood
    hoof hook hoop hose host hour howl hull hump hung hymn icon idle inch
    inky isle itch item jack jade jake jamb jape jeer jest jiffy joke jolt
    junk jury keen kegs kelp kiln kink knit knob knot lack lame lamp lane
    lard lark lash lass last lath laud lava lawn laze lazy lean leek leer
    lens lent levy lime limp lino lira lobe loft loin lore lorn lout luge
    lull lump lung lure lurk lust mace maid male mall malt mane mast mate
    maze mead meal meek meet melt memo mend mesh mild mile mime mint mire
    mist mitt moat moil mold molt mope moss mote moth mould mound mount
    mourn mow muck mugs mule murk musk mutt nave navy nigh node noir nook
    norm nose noun nude null numb oafs oath oboe odds omen once ooze opus
    orca orgy otic oven owed owes owns pace pack pact page palm pane pant
    pare park pave pawn peat peel peer pelf pelt peon perk pert pest pile
    pill pine ping pint pipe pity pixel pixy plod plot plow plug plum plop
    plus poem poet pole poll polo pond pore pork port pose post pout prey
    prig prim prod prop pros prow puck puff puke pulp pun punt pure purr
    pylon quay quit quiz raga rage raid rail rain ramp rang rank rant rapt
    rash rasp rave razz ream reap reed reef reel rein rely rend rent resin
    retch rife riff rift rime rind rink riot ripe roam roar robe rook rope
    rout rove ruff rugs ruin rump ruse rush rust rut sack sage saga sago
    sap sash sate saul scam scar scud seam sear sect seem seer self serf
    shed shin shy sift silk sink sire site size skew skim skin skip skull
    slab slap slat slaw sled slid slim slip slit slob slop slot slow slug
    slum slur slut smew snag snap snob snoop snot snub soak soap sobs sock
    soma some soot sore sort soup sour sown span spar spat spec sped spell
    spent spew spin spiv spot spry spur stab stag stew stir stub stud stun
    sty such suck sulk sump sung sunk sure surf sumo sway swam swan swap
    swat swig swam swum sync tabs tack tack tads tame tang tare teak teal
    team tear teem tell temp tend tent term tern test tidy tied tier tilt
    toad toil toll tong toot tops tore tors toss tote tray trot tube tuck
    tuft turd turf twig twit tyke udder ugly ulna undo unit unto urge used
    vain vale van vane vat veer vein very vest veto view vim vise void volt
    volt waft wail waif wait wane ward ware wary watt wavy weld welt went
    whet whim whip whir whit wick wide wigs wilt wimp wind wine wink wisp
    woe woke wont woof worm wren writ yak yam yap yarn yawl yore yuan yore
    """
    for w in fallback.split():
        w = w.strip().lower()
        if w.isalpha() and len(w) >= 3:
            words.add(w)
    print(f"    Using built-in fallback word list ({len(words):,} words)")
    return words

print("📖  Loading dictionary...", flush=True)
DICTIONARY = load_dictionary()
print(f"    {len(DICTIONARY):,} words loaded.\n")

# ── 2. Compute π in base 26 ───────────────────────────────────────────────────

PRECISION_DIGITS = 5000   # decimal digits of precision
BASE26_LENGTH    = 3000   # how many base-26 chars to generate

mp.dps = PRECISION_DIGITS + 50   # extra guard digits

print(f"🔢  Computing π to {PRECISION_DIGITS} decimal digits...", flush=True)

# Work with the fractional part using exact big-integer arithmetic.
# π as a rational approximation: grab enough decimal digits, treat as integer / 10^n
pi_str = mp.nstr(pi, PRECISION_DIGITS, strip_zeros=False).replace(".", "")
# pi_str[0] = '3', rest is fractional digits
n_frac = len(pi_str) - 1
D   = 10 ** n_frac                # denominator
frac = int(pi_str[1:])            # numerator of fractional part  (= frac_value * D)

# Integer part: 3  →  'D'
INT_CHAR = chr(ord('A') + 3)

# Extract base-26 digits from the fractional part
base26_chars = []
for _ in range(BASE26_LENGTH):
    frac *= 26
    digit = frac // D
    frac  = frac %  D
    base26_chars.append(chr(ord('A') + digit))

pi_b26 = INT_CHAR + "." + "".join(base26_chars)
pi_letters = INT_CHAR + "".join(base26_chars)   # flat string for searching

print(f"    First 80 chars: {pi_b26[:80]}\n")

# ── 3. Find dictionary words ──────────────────────────────────────────────────

print("🔍  Scanning for English words...\n", flush=True)

found = defaultdict(list)   # word → list of positions

flat = pi_letters.lower()
n    = len(flat)

for start in range(n):
    for length in range(3, min(16, n - start + 1)):
        candidate = flat[start:start + length]
        if candidate in DICTIONARY:
            found[candidate].append(start)

# Deduplicate: keep only the longest match at each position to avoid noise
# (so "the" isn't reported if "there" is found at the same spot)
filtered = {}
for word, positions in found.items():
    for pos in positions:
        # Check no longer word at this position already captured
        dominated = any(
            other != word and other_pos == pos and len(other) > len(word)
            for other, other_positions in found.items()
            for other_pos in other_positions
        )
        if not dominated:
            if word not in filtered:
                filtered[word] = []
            filtered[word].append(pos)

# ── 4. Print results ──────────────────────────────────────────────────────────

print("=" * 70)
print("  π  IN BASE 26  (A=0 … Z=25)")
print("=" * 70)

# Pretty-print in rows of 60 chars
chunk = 60
header = f"  {INT_CHAR}.  "
flat_frac = "".join(base26_chars)
for i in range(0, min(600, len(flat_frac)), chunk):
    label = f"{i:>6}" if i > 0 else "    0 "
    print(f"  [{label}]  {flat_frac[i:i+chunk]}")

print()
print("=" * 70)
print(f"  WORDS FOUND IN π (base-26)  — {len(filtered)} unique words")
print("=" * 70)

# Sort by word length desc, then alphabetically
sorted_words = sorted(filtered.items(), key=lambda kv: (-len(kv[0]), kv[0]))

by_length = defaultdict(list)
for word, positions in sorted_words:
    by_length[len(word)].append((word, positions))

for length in sorted(by_length.keys(), reverse=True):
    entries = by_length[length]
    print(f"\n  ── {length}-letter words ({len(entries)}) ──")
    for word, positions in sorted(entries):
        pos_str = ", ".join(str(p) for p in positions[:5])
        if len(positions) > 5:
            pos_str += f" … (+{len(positions)-5} more)"
        # Show context snippet
        p0 = positions[0]
        lo = max(0, p0 - 4)
        hi = min(len(pi_letters), p0 + len(word) + 4)
        ctx = pi_letters[lo:hi].upper()
        # Highlight the word inside context
        rel = p0 - lo
        ctx_hi = ctx[:rel] + f"[{ctx[rel:rel+len(word)]}]" + ctx[rel+len(word):]
        print(f"    {word:<18}  pos {pos_str:<30}  …{ctx_hi}…")

print()
print("=" * 70)

# ── 5. Stats ──────────────────────────────────────────────────────────────────
total_chars = len(pi_letters)
covered     = sum(len(w) * len(ps) for w, ps in filtered.items())
longest     = sorted_words[0][0] if sorted_words else "—"

print(f"""
  STATS
  ─────────────────────────────────────────
  Base-26 chars generated : {total_chars:>10,}
  Unique words found       : {len(filtered):>10,}
  Longest word             : {longest!r:>10}
  Word occurrences total   : {sum(len(ps) for ps in filtered.values()):>10,}
""")
print("=" * 70)
