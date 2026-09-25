
import os
import json
import argparse
import random
import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter


# ============================================================
# DATASET
# ============================================================

DATASET_FILE = "singlekb.txt"

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset file not found: {DATASET_FILE}"
    )

with open(
    DATASET_FILE,
    "r",
    encoding="utf-8",
    errors="replace",
) as file:
    DEFAULT_DATASET = file.read()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(
    text: str,
    unit: str = "word",
) -> List[str]:

    if unit == "char":
        return list(text)

    return text.strip().split()


def detokenize(
    tokens: List[str],
    unit: str = "word",
) -> str:

    if unit == "char":
        return "".join(tokens)

    return " ".join(tokens)


# ============================================================
# HELPERS
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# ============================================================
# AUTOMATIC SCALING
# ============================================================

def calculate_trigram_scaling(
    total_tokens: int,
    vocabulary_size: int,
    unique_trigrams: int,
) -> Dict[str, Any]:

    """
    Automatic scaling based on:

        N = dataset token count
        V = vocabulary size
        T = unique trigram contexts

    This scaling never introduces bigram or unigram
    generation.

    The actual model remains:

        P(c | a,b)

        (a,b) -> c
    """

    N = max(
        1,
        int(total_tokens),
    )

    V = max(
        1,
        int(vocabulary_size),
    )

    T = max(
        1,
        int(unique_trigrams),
    )

    # --------------------------------------------------------
    # TOP-K
    #
    #     top_k = clamp(
    #         round(4 * sqrt(V)),
    #         8,
    #         256
    #     )
    # --------------------------------------------------------

    top_k = int(
        round(
            clamp(
                4.0 * math.sqrt(V),
                8,
                min(256, V),
            )
        )
    )

    top_k = max(
        1,
        min(
            V,
            top_k,
        ),
    )

    # --------------------------------------------------------
    # PIECES
    #
    #     pieces =
    #         clamp(
    #             round(2*log2(V+1)),
    #             4,
    #             32
    #         )
    # --------------------------------------------------------

    base_pieces = int(
        round(
            clamp(
                2.0 * math.log2(V + 1),
                4,
                32,
            )
        )
    )

    base_pieces = max(
        1,
        min(
            base_pieces,
            V,
        ),
    )

    # --------------------------------------------------------
    # DATASET DENSITY
    #
    #     density = N / V
    # --------------------------------------------------------

    density = (
        N / float(V)
    )

    # --------------------------------------------------------
    # SMOOTHING
    #
    #     smoothing =
    #         clamp(
    #             0.20 / sqrt(N/V),
    #             0.005,
    #             0.20
    #         )
    #
    # IMPORTANT:
    # This is only applied to already observed third-token
    # candidates of the current trigram.
    # --------------------------------------------------------

    smoothing = clamp(
        0.20
        / math.sqrt(
            max(
                density,
                1.0,
            )
        ),
        0.005,
        0.20,
    )

    # --------------------------------------------------------
    # REPETITION PENALTY
    #
    #     penalty =
    #         1 + 0.35/log2(V+1)
    # --------------------------------------------------------

    repetition_penalty = (
        1.0
        +
        0.35
        /
        max(
            1.0,
            math.log2(V + 1),
        )
    )

    # --------------------------------------------------------
    # REPETITION WINDOW
    # --------------------------------------------------------

    repetition_window = int(
        round(
            clamp(
                4.0 * math.log2(V + 1),
                8,
                64,
            )
        )
    )

    # --------------------------------------------------------
    # TRIGRAM COVERAGE
    # --------------------------------------------------------

    possible_positions = max(
        1,
        N - 2,
    )

    coverage = clamp(
        T / float(
            possible_positions
        ),
        0.0,
        1.0,
    )

    return {
        "total_tokens": N,
        "vocabulary_size": V,
        "unique_trigram_contexts": T,

        "density_N_over_V": density,

        "trigram_coverage": coverage,

        "top_k_formula":
            "clamp(round(4 * sqrt(V)), 8, 256)",

        "top_k": top_k,

        "base_pieces_formula":
            "clamp(round(2 * log2(V + 1)), 4, 32)",

        "base_pieces": base_pieces,

        "smoothing_formula":
            "clamp(0.20 / sqrt(max(N/V, 1)), 0.005, 0.20)",

        "smoothing": smoothing,

        "repetition_penalty_formula":
            "1 + 0.35 / log2(V + 1)",

        "repetition_penalty":
            repetition_penalty,

        "repetition_window":
            repetition_window,

        "generation_distribution":
            "P(token[t] | token[t-2], token[t-1])",

        "generation_transition":
            "(token[t-2], token[t-1]) -> token[t]",

        "backoff": "NONE",

        "bigram_model": False,

        "unigram_model": False,

        "interpolation": False,
    }


# ============================================================
# CONFIG
# ============================================================

class Config:

    def __init__(
        self,
        unit: str = "word",
        seed: int = 42,
        output_dir: str = "output",

        temperature: float = 0.85,

        # ----------------------------------------------------
        # These were missing from the previous Config.
        # ----------------------------------------------------

        sine_amplitude: float = 0.20,
        sine_frequency: float = 1.0,
        sine_phase: float = 0.0,

        plasticity_rate: float = 0.005,
        plasticity_strength: float = 0.02,

        piece_strength: float = 0.025,

        punctuation_penalty: float = 0.50,

        prevent_immediate_repeat: bool = True,

    ):

        self.unit = unit
        self.seed = seed
        self.output_dir = output_dir

        self.temperature = temperature

        # ----------------------------------------------------
        # Curve parameters
        # ----------------------------------------------------

        self.sine_amplitude = (
            sine_amplitude
        )

        self.sine_frequency = (
            sine_frequency
        )

        self.sine_phase = (
            sine_phase
        )

        # ----------------------------------------------------
        # Plasticity
        # ----------------------------------------------------

        self.plasticity_rate = (
            plasticity_rate
        )

        self.plasticity_strength = (
            plasticity_strength
        )

        # ----------------------------------------------------
        # Piecewise modulation
        # ----------------------------------------------------

        self.piece_strength = (
            piece_strength
        )

        # ----------------------------------------------------
        # Repetition
        # ----------------------------------------------------

        self.punctuation_penalty = (
            punctuation_penalty
        )

        self.prevent_immediate_repeat = (
            prevent_immediate_repeat
        )


# ============================================================
# TRUE TRIGRAM MARKOV GENERATOR
# ============================================================

class TrueTrigramMarkovGenerator:

    """
    PURE TRUE-TRIGRAM MODEL.

    The only probability relationship used for generation is:

        P(c | a,b)

    where:

        a = token[t-2]
        b = token[t-1]
        c = token[t]

    Therefore every generated transition is:

        (a,b) -> c

    There is NO:

        bigram fallback
        unigram fallback
        vocabulary-wide random fallback
        n-gram interpolation
    """

    def __init__(
        self,
        dataset_text: str,
        config: Config,
    ):

        self.config = config

        random.seed(
            config.seed
        )

        self.tokens = tokenize(
            dataset_text,
            config.unit,
        )

        if len(
            self.tokens
        ) < 3:

            raise ValueError(
                "A true trigram model requires "
                "at least three dataset tokens."
            )

        # ----------------------------------------------------
        # VOCABULARY
        # ----------------------------------------------------

        self.vocab = sorted(
            set(self.tokens)
        )

        self.token_to_id = {
            token: index
            for index, token
            in enumerate(self.vocab)
        }

        self.id_to_token = {
            index: token
            for index, token
            in enumerate(self.vocab)
        }

        self.vocab_size = len(
            self.vocab
        )

        self.ids = [
            self.token_to_id[token]
            for token in self.tokens
        ]

        # ----------------------------------------------------
        # TRUE TRIGRAM COUNTS
        #
        #     (a,b) -> {c: count}
        # ----------------------------------------------------

        self.trigram_counts = (
            defaultdict(Counter)
        )

        for i in range(
            len(self.ids) - 2
        ):

            a = self.ids[i]
            b = self.ids[i + 1]
            c = self.ids[i + 2]

            self.trigram_counts[
                (a, b)
            ][c] += 1

        if not self.trigram_counts:

            raise ValueError(
                "No true trigram contexts "
                "were constructed."
            )

        # ----------------------------------------------------
        # AUTOMATIC PARAMETERS
        # ----------------------------------------------------

        self.scaled = (
            calculate_trigram_scaling(
                total_tokens=len(
                    self.tokens
                ),
                vocabulary_size=(
                    self.vocab_size
                ),
                unique_trigrams=len(
                    self.trigram_counts
                ),
            )
        )

        self.top_k = (
            self.scaled["top_k"]
        )

        self.base_pieces = (
            self.scaled["base_pieces"]
        )

        self.smoothing = (
            self.scaled["smoothing"]
        )

        self.repetition_penalty = (
            self.scaled[
                "repetition_penalty"
            ]
        )

        self.repetition_window = (
            self.scaled[
                "repetition_window"
            ]
        )

        self.history = []

    # ========================================================
    # TRUE TRIGRAM CONTEXT
    # ========================================================

    def find_initial_context(
        self,
        prompt_tokens: List[str],
    ) -> Tuple[
        List[int],
        bool,
    ]:

        """
        Find an observed two-token trigram context.

        The prompt itself is preserved separately.

        If the final two known prompt tokens form an observed
        trigram context, that pair is used.

        Otherwise an observed dataset trigram context is used.

        No lower-order model is created.
        """

        prompt_ids = [
            self.token_to_id.get(
                token,
                -1,
            )
            for token in prompt_tokens
        ]

        # ----------------------------------------------------
        # Prefer the latest observed pair in the prompt.
        # ----------------------------------------------------

        for i in range(
            len(prompt_ids) - 2,
            -1,
            -1,
        ):

            a = prompt_ids[i]
            b = prompt_ids[i + 1]

            if a < 0 or b < 0:
                continue

            if (
                a,
                b
            ) in self.trigram_counts:

                return (
                    [
                        a,
                        b,
                    ],
                    True,
                )

        # ----------------------------------------------------
        # Prompt does not contain a usable trigram context.
        #
        # Select an actual observed trigram context.
        # ----------------------------------------------------

        initial = next(
            iter(
                self.trigram_counts
            )
        )

        return (
            list(initial),
            False,
        )

    # ========================================================
    # TRUE TRIGRAM DISTRIBUTION
    # ========================================================

    def trigram_distribution(
        self,
        context_ids: List[int],
    ) -> Tuple[
        List[int],
        List[float],
    ]:

        """
        Construct only:

            P(c | a,b)

        No other probability model exists here.
        """

        if len(
            context_ids
        ) != 2:

            raise RuntimeError(
                "True trigram generation requires "
                "exactly two context IDs."
            )

        a = context_ids[0]
        b = context_ids[1]

        counts = (
            self.trigram_counts.get(
                (a, b)
            )
        )

        

        candidate_ids = list(
            counts.keys()
        )

        raw_counts = [
            float(
                counts[token_id]
            )
            for token_id
            in candidate_ids
        ]

        total = sum(
            raw_counts
        )

        if total <= 0:

            raise RuntimeError(
                "Invalid trigram counts."
            )

        probabilities = [
            count / total
            for count
            in raw_counts
        ]

        # ----------------------------------------------------
        # Smoothing remains strictly inside the active
        # trigram continuation set.
        #
        # If the trigram has:
        #
        #     (a,b) -> c1
        #     (a,b) -> c2
        #
        # smoothing only affects c1 and c2.
        #
        # It cannot introduce a token never observed after
        # this exact (a,b) context.
        # ----------------------------------------------------

        if len(
            probabilities
        ) > 1:

            candidate_count = (
                len(probabilities)
            )

            probabilities = [
                (
                    (1.0 - self.smoothing)
                    * probability
                    +
                    (
                        self.smoothing
                        /
                        candidate_count
                    )
                )
                for probability
                in probabilities
            ]

        total = sum(
            probabilities
        )

        probabilities = [
            probability / total
            for probability
            in probabilities
        ]

        return (
            candidate_ids,
            probabilities,
        )

    # ========================================================
    # SINE MODULATION
    # ========================================================

    def sine_factor(
        self,
        step_index: int,
    ) -> float:

        return (
            1.0
            +
            self.config.sine_amplitude
            *
            math.sin(
                self.config.sine_frequency
                *
                step_index
                +
                self.config.sine_phase
            )
        )

    # ========================================================
    # PIECEWISE MODULATION
    # ========================================================

    def piecewise_modulate(
        self,
        probabilities: List[float],
        step_index: int,
    ) -> List[float]:

        """
        Piecewise modulation is applied to the candidates of
        the active true trigram only.
        """

        if len(
            probabilities
        ) <= 1:

            return probabilities

        result = list(
            probabilities
        )

        ranking = sorted(
            range(
                len(result)
            ),
            key=lambda index:
                result[index],
        )

        pieces = min(
            self.base_pieces,
            len(ranking),
        )

        # ----------------------------------------------------
        # Dataset-scaled sine factor.
        # ----------------------------------------------------

        sine = self.sine_factor(
            step_index
        )

        # Convert sine value to a bounded modulation strength.
        sine_offset = (
            sine - 1.0
        )

        for piece_index in range(
            pieces
        ):

            start = (
                piece_index
                *
                len(ranking)
                //
                pieces
            )

            end = (
                (piece_index + 1)
                *
                len(ranking)
                //
                pieces
            )

            if end <= start:
                continue

            if pieces == 1:

                position = 0.5

            else:

                position = (
                    piece_index
                    /
                    float(
                        pieces - 1
                    )
                )

            rank_center = (
                position - 0.5
            )

            scale = (
                1.0
                +
                self.config.piece_strength
                *
                rank_center
                *
                2.0
                *
                sine
            )

            # Small global sine contribution while preserving
            # token-specific rank differences.
            scale += (
                0.05
                *
                sine_offset
                *
                rank_center
            )

            scale = max(
                0.001,
                scale,
            )

            for index in ranking[
                start:end
            ]:

                result[index] *= (
                    scale
                )

        total = sum(
            result
        )

        if total <= 0:

            return probabilities

        return [
            probability / total
            for probability
            in result
        ]

    # ========================================================
    # PLASTICITY
    # ========================================================

    def apply_plasticity(
        self,
        probabilities: List[float],
        step_index: int,
    ) -> List[float]:

        """
        Apply a small rank-dependent plasticity term.

        Still only modifies probabilities belonging to the
        active true-trigram continuation set.
        """

        if len(
            probabilities
        ) <= 1:

            return probabilities

        result = list(
            probabilities
        )

        phase = (
            step_index
            *
            self.config.plasticity_rate
        )

        plasticity = math.sin(
            phase
        )

        ranking = sorted(
            range(
                len(result)
            ),
            key=lambda index:
                result[index],
            reverse=True,
        )

        size = len(
            ranking
        )

        for rank, index in enumerate(
            ranking
        ):

            normalized_rank = (
                rank
                /
                max(
                    1,
                    size - 1,
                )
            )

            centered = (
                0.5
                -
                normalized_rank
            )

            multiplier = (
                1.0
                +
                self.config.plasticity_strength
                *
                plasticity
                *
                centered
            )

            result[index] *= (
                max(
                    0.001,
                    multiplier,
                )
            )

        total = sum(
            result
        )

        if total <= 0:

            return probabilities

        return [
            probability / total
            for probability
            in result
        ]

    # ========================================================
    # REPETITION
    # ========================================================

    def repetition_control(
        self,
        candidate_ids: List[int],
        probabilities: List[float],
        generated_ids: List[int],
    ) -> List[float]:

        if not generated_ids:

            return probabilities

        result = list(
            probabilities
        )

        recent = (
            generated_ids[
                -self.repetition_window:
            ]
        )

        recent_counts = Counter(
            recent
        )

        for index, token_id in enumerate(
            candidate_ids
        ):

            count = (
                recent_counts.get(
                    token_id,
                    0,
                )
            )

            if count:

                result[index] /= (
                    self.repetition_penalty
                    **
                    count
                )

        # ----------------------------------------------------
        # Immediate repetition reduction.
        #
        # Only applied if the trigram has another valid
        # continuation.
        # ----------------------------------------------------

        if (
            len(candidate_ids) > 1
            and generated_ids
            and self.config.prevent_immediate_repeat
        ):

            previous = (
                generated_ids[-1]
            )

            for index, token_id in enumerate(
                candidate_ids
            ):

                if token_id == previous:

                    result[index] *= 0.05

        total = sum(
            result
        )

        if total <= 0:

            return probabilities

        return [
            probability / total
            for probability
            in result
        ]

    # ========================================================
    # TEMPERATURE
    # ========================================================

    def apply_temperature(
        self,
        probabilities: List[float],
    ) -> List[float]:

        temperature = max(
            self.config.temperature,
            1e-12,
        )

        logits = [
            math.log(
                max(
                    probability,
                    1e-12,
                )
            )
            /
            temperature
            for probability
            in probabilities
        ]

        maximum = max(
            logits
        )

        values = [
            math.exp(
                value - maximum
            )
            for value in logits
        ]

        total = sum(
            values
        )

        return [
            value / total
            for value in values
        ]

    # ========================================================
    # TOP-K
    # ========================================================

    def apply_top_k(
        self,
        candidate_ids: List[int],
        probabilities: List[float],
    ) -> Tuple[
        List[int],
        List[float],
    ]:

        k = min(
            self.top_k,
            len(candidate_ids),
        )

        if k >= len(
            candidate_ids
        ):

            return (
                candidate_ids,
                probabilities,
            )

        ranking = sorted(
            range(
                len(candidate_ids)
            ),
            key=lambda index:
                probabilities[index],
            reverse=True,
        )

        selected = ranking[:k]

        ids = [
            candidate_ids[index]
            for index in selected
        ]

        values = [
            probabilities[index]
            for index in selected
        ]

        total = sum(
            values
        )

        values = [
            value / total
            for value in values
        ]

        return (
            ids,
            values,
        )

    # ========================================================
    # SINGLE TRUE-TRIGRAM STEP
    # ========================================================

    def step(
        self,
        context_ids: List[int],
        generated_ids: List[int],
        step_index: int,
    ) -> Tuple[
        int,
        Dict[str, Any],
    ]:

        # ----------------------------------------------------
        # HARD GUARANTEE
        # ----------------------------------------------------

        if len(
            context_ids
        ) != 2:

            raise RuntimeError(
                "Pure true-trigram generation "
                "requires exactly two context tokens."
            )

        a = context_ids[0]
        b = context_ids[1]

        # ----------------------------------------------------
        # THE ONLY MODEL LOOKUP
        #
        #         (a,b) -> c
        # ----------------------------------------------------

        candidate_ids, probabilities = (
            self.trigram_distribution(
                context_ids
            )
        )

        # ----------------------------------------------------
        # Modulation
        # ----------------------------------------------------

        probabilities = (
            self.piecewise_modulate(
                probabilities,
                step_index,
            )
        )

        probabilities = (
            self.apply_plasticity(
                probabilities,
                step_index,
            )
        )

        probabilities = (
            self.repetition_control(
                candidate_ids,
                probabilities,
                generated_ids,
            )
        )

        probabilities = (
            self.apply_temperature(
                probabilities
            )
        )

        candidate_ids, probabilities = (
            self.apply_top_k(
                candidate_ids,
                probabilities,
            )
        )

        # ----------------------------------------------------
        # SAMPLE THIRD TOKEN
        # ----------------------------------------------------

        target_id = random.choices(
            candidate_ids,
            weights=probabilities,
            k=1,
        )[0]

        target_token = (
            self.id_to_token[
                target_id
            ]
        )

        token_a = (
            self.id_to_token[a]
        )

        token_b = (
            self.id_to_token[b]
        )

        # ----------------------------------------------------
        # Explicit trigram metadata.
        # ----------------------------------------------------

        info = {
            "type": (
                "generated_trigram"
            ),

            "model": (
                "PURE_TRUE_TRIGRAM"
            ),

            "step_index": (
                step_index
            ),

            "equation": (
                "P(c | a,b)"
            ),

            "context_ids": [
                int(a),
                int(b),
            ],

            "context": [
                token_a,
                token_b,
            ],

            "selected_id": (
                int(target_id)
            ),

            "selected_token": (
                target_token
            ),

            "trigram": [
                token_a,
                token_b,
                target_token,
            ],

            "transition": (
                f"({token_a}, "
                f"{token_b}) -> "
                f"{target_token}"
            ),

            "candidate_count": (
                len(candidate_ids)
            ),

            "top_k": (
                self.top_k
            ),

            "mode": "trigram",

            "backoff": None,

            "bigram_used": False,

            "unigram_used": False,
        }

        return (
            target_id,
            info,
        )

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
    ) -> str:

        prompt_tokens = tokenize(
            prompt,
            self.config.unit,
        )

        # ----------------------------------------------------
        # Establish an actual observed trigram context.
        # ----------------------------------------------------

        context_ids, prompt_context_used = (
            self.find_initial_context(
                prompt_tokens
            )
        )

        if len(
            context_ids
        ) != 2:

            raise RuntimeError(
                "Could not establish a valid "
                "true-trigram context."
            )

        generated_ids: List[int] = []

        generated_tokens: List[str] = []

        self.history = []

        # ----------------------------------------------------
        # Record prompt.
        # ----------------------------------------------------

        for position, token in enumerate(
            prompt_tokens
        ):

            self.history.append(
                {
                    "type": "prompt",
                    "position": position,
                    "token": token,
                    "known_to_dataset": (
                        token
                        in self.token_to_id
                    ),
                }
            )

        # ----------------------------------------------------
        # Record initialization.
        # ----------------------------------------------------

        self.history.append(
            {
                "type": (
                    "trigram_initialization"
                ),

                "prompt_context_used": (
                    prompt_context_used
                ),

                "context_ids": [
                    int(x)
                    for x in context_ids
                ],

                "context": [
                    self.id_to_token[x]
                    for x in context_ids
                ],

                "equation": (
                    "P(c | a,b)"
                ),
            }
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        for step_index in range(
            max(
                0,
                int(max_new_tokens),
            )
        ):

            # ------------------------------------------------
            # Hard invariant:
            #
            # exactly two tokens enter the model.
            # ------------------------------------------------

            if len(
                context_ids
            ) != 2:

                raise RuntimeError(
                    "Internal trigram context "
                    "invariant violated."
                )

            target_id, info = (
                self.step(
                    context_ids,
                    generated_ids,
                    step_index,
                )
            )

            self.history.append(
                info
            )

            generated_ids.append(
                target_id
            )

            target_token = (
                self.id_to_token[
                    target_id
                ]
            )

            generated_tokens.append(
                target_token
            )

            # ------------------------------------------------
            # TRIGRAM SHIFT
            #
            # Previous:
            #
            #     (a,b) -> c
            #
            # Next:
            #
            #     (b,c) -> d
            # ------------------------------------------------

            context_ids = [
                context_ids[1],
                target_id,
            ]

        # ----------------------------------------------------
        # Continuation.
        # ----------------------------------------------------

        continuation = detokenize(
            generated_tokens,
            self.config.unit,
        )

        # ----------------------------------------------------
        # Preserve the prompt.
        # ----------------------------------------------------

        if not prompt:

            return continuation

        if not continuation:

            return prompt

        if self.config.unit == "char":

            return (
                prompt
                +
                continuation
            )

        return (
            prompt
            +
            " "
            +
            continuation
        )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    def save_history(
        self,
        filepath: str,
    ):

        directory = os.path.dirname(
            filepath
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.history,
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ========================================================
    # SAVE SCALING
    # ========================================================

    def save_scaling(
        self,
        filepath: str,
    ):

        directory = os.path.dirname(
            filepath
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.scaled,
                f,
                indent=2,
            )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Pure True-Trigram Markov "
            "Generator"
        )
    )

   
    parser.add_argument(
        "--tokens",
        type=int,
        default=800,
        help=(
            "Number of new tokens."
        ),
    )

    parser.add_argument(
        "--unit",
        type=str,
        default="word",
        choices=[
            "word",
            "char",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--sine-amplitude",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--sine-frequency",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--sine-phase",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--plasticity-rate",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--plasticity-strength",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--piece-strength",
        type=float,
        default=0.025,
    )

    parser.add_argument(
        "--seed-output",
        type=str,
        default="output",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    config = Config(
        unit=args.unit,
        seed=args.seed,
        output_dir=args.seed_output,

        temperature=args.temperature,

        # ----------------------------------------------------
        # These are now explicitly present in Config.
        # ----------------------------------------------------

        sine_amplitude=(
            args.sine_amplitude
        ),

        sine_frequency=(
            args.sine_frequency
        ),

        sine_phase=(
            args.sine_phase
        ),

        plasticity_rate=(
            args.plasticity_rate
        ),

        plasticity_strength=(
            args.plasticity_strength
        ),

        piece_strength=(
            args.piece_strength
        ),
    )

    print(
        "Initializing "
        "PURE TRUE-TRIGRAM generator..."
    )

    generator = (
        TrueTrigramMarkovGenerator(
            dataset_text=(
                DEFAULT_DATASET
            ),
            config=config,
        )
    )

    # ========================================================
    # PARAMETERS
    # ========================================================

    print()
    print(
        "=" * 76
    )
    print(
        "TRUE TRIGRAM MODEL"
    )
    print(
        "=" * 76
    )

    print(
        f"Dataset tokens:          "
        f"{len(generator.tokens):,}"
    )

    print(
        f"Vocabulary:              "
        f"{generator.vocab_size:,}"
    )

    print(
        f"Unique trigram contexts: "
        f"{len(generator.trigram_counts):,}"
    )

    print()
    print(
        "=" * 76
    )
    print(
        "AUTOMATIC SCALING"
    )
    print(
        "=" * 76
    )

    print(
        f"top_k:                   "
        f"{generator.top_k}"
    )

    print(
        f"base_pieces:             "
        f"{generator.base_pieces}"
    )

    print(
        f"smoothing:               "
        f"{generator.smoothing:.8f}"
    )

    print(
        f"repetition_penalty:      "
        f"{generator.repetition_penalty:.8f}"
    )

    print(
        f"repetition_window:       "
        f"{generator.repetition_window}"
    )

    print()
    print(
        "=" * 76
    )
    print(
        "MODEL STRUCTURE"
    )
    print(
        "=" * 76
    )

    print(
        "Probability model:"
    )

    print(
        "  P(token[t] | "
        "token[t-2], token[t-1])"
    )

    print(
        "Transition:"
    )

    print(
        "  (token[t-2], token[t-1]) "
        "-> token[t]"
    )

    print(
        "Bigram fallback:         NO"
    )

    print(
        "Unigram fallback:        NO"
    )

    print(
        "Vocabulary fallback:     NO"
    )

    print(
        "N-gram interpolation:     NO"
    )

    print()
    print(
        "=" * 76
    )

    while True:
        output = generator.generate(
            prompt=input("USER: "),
            max_new_tokens=args.tokens,
        )

        print()
        print(
            "Generated Output:"
        )
        print(
            output
        )
