#!/usr/bin/env python3
"""
Toy q-based human/machine CAPTCHA with Complexity Ratio Evaluation and Padding Questions.

The challenge shows two noisy symbols per trial, with boundary padding questions.
User choices:
    s = same
    d = different
    q = quit/skip

Evaluation Dimensions:
    1. Disagreement / error rate (q)
    2. Response timing log-normal likelihood
    3. Answer alternation detection (anti-spam)
    4. Ideal Complexity Ratio (CR = C / H) matched against history
    5. Historical footprint analysis (captcha_history.jsonl)
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import math
import secrets
import statistics
import time
import tkinter as tk
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import messagebox
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageTk


# ----------------------------- Configuration -----------------------------

WIDTH = 300
HEIGHT = 150
TRIAL_COUNT = 12
MIN_RESPONSE_SECONDS = 0.20
MAX_RESPONSE_SECONDS = 15.0

# Evaluation thresholds
MAX_WRONG_ANSWERS = 3
MIN_HUMAN_LIKELIHOOD = -32.0
MAX_ALTERNATION_RATE = 0.80  # Reject if >80% of answers alternate (s-d-s-d)
MAX_COMPLEXITY_RATIO = 1.45  # Upper bound for challenge complexity vs human capacity

# Secret used to sign challenge metadata
SERVER_SECRET = secrets.token_bytes(32)

# File used to persist local session history
HISTORY_FILE = Path("captcha_history.jsonl")


# ------------------------------- Data model -------------------------------

@dataclass
class Trial:
    index: int
    left_seed: int
    right_seed: int
    expected: str
    difficulty: float
    is_padding: bool = False
    padding_prompt: str = ""


@dataclass
class Challenge:
    challenge_id: str
    created_at: float
    trials: list[Trial]
    signature: str


@dataclass
class TrialResult:
    index: int
    answer: str
    correct: bool
    response_time: float


@dataclass
class VerificationReport:
    accepted: bool
    reason: str
    score: float
    estimated_q: float
    disagreements: int
    total_trials: int
    mean_response_time: float
    complexity_ratio: float
    results: list[TrialResult]


# ------------------------------ History I/O -------------------------------

def load_history() -> list[dict]:
    """Load historical attempts to inform the current verification test."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []


def append_history(report: VerificationReport) -> None:
    """Save the outcome to history so future tests can evaluate patterns."""
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(report)) + "\n")
    except Exception:
        pass


# ------------------------------ Math helpers ------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def log_normal_pdf(x: float, mean: float, standard_deviation: float) -> float:
    """Log probability density of a normal distribution."""
    sd = max(standard_deviation, 1e-9)
    z = (x - mean) / sd
    return -math.log(sd) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z * z


def estimate_q(results: list[TrialResult]) -> float:
    if not results:
        return 0.0
    disagreements = sum(not result.correct for result in results)
    return disagreements / len(results)


def human_likelihood(results: list[TrialResult]) -> float:
    if not results:
        return float("-inf")

    score = 0.0
    for result in results:
        response_time = clamp(result.response_time, 0.01, 60.0)
        score += log_normal_pdf(
            math.log(response_time),
            mean=math.log(1.5),
            standard_deviation=0.75,
        )
        score += math.log(0.85 if result.correct else 0.15)
        if result.response_time < MIN_RESPONSE_SECONDS:
            score -= 8.0

    return score


def check_response_alternation(results: list[TrialResult], threshold: float = MAX_ALTERNATION_RATE) -> bool:
    """Detects whether user responses show an unnatural alternating pattern (e.g., s-d-s-d)."""
    valid_answers = [r.answer for r in results if r.answer in {"s", "d"}]
    if len(valid_answers) < 6:
        return False

    transitions = len(valid_answers) - 1
    alternations = sum(1 for i in range(transitions) if valid_answers[i] != valid_answers[i + 1])
    alternation_rate = alternations / transitions

    return alternation_rate >= threshold


# -------------------------- Complexity Ratio Engine ------------------------

def calculate_historical_capacity(history: list[dict], default_capacity: float = 0.85) -> float:
    """Calculates historical human capacity H based on recent successful history records."""
    valid_sessions = [r for r in history[-10:] if r.get("accepted")]
    if not valid_sessions:
        return default_capacity

    q_hist = statistics.mean(s.get("estimated_q", 0.15) for s in valid_sessions)
    t_hist = statistics.mean(s.get("mean_response_time", 1.5) for s in valid_sessions)
    t_ref = 1.5

    capacity = (1.0 - q_hist) * (t_hist / t_ref)
    return max(0.2, capacity)


def calculate_challenge_complexity(difficulty: float, trial_count: int) -> float:
    """Calculates challenge complexity C based on visual parameters and sequence length."""
    mark_count = 15 + difficulty * 75
    return difficulty * (1.0 + mark_count / 50.0) * math.log2(trial_count)


def compute_complexity_ratio(challenge: Challenge, history: list[dict]) -> float:
    """Computes the Complexity Ratio (CR = C / H)."""
    visual_trials = [t for t in challenge.trials if not t.is_padding]
    trial_count = len(visual_trials)
    difficulty = visual_trials[0].difficulty if trial_count > 0 else 0.55

    complexity = calculate_challenge_complexity(difficulty, trial_count)
    capacity = calculate_historical_capacity(history)

    return complexity / capacity


# ---------------------------- Challenge creation --------------------------

def create_challenge(
    trial_count: int = TRIAL_COUNT,
    difficulty: float = 0.55,
) -> Challenge:
    difficulty = clamp(difficulty, 0.0, 1.0)
    challenge_id = secrets.token_urlsafe(18)
    created_at = time.time()
    trials: list[Trial] = []

    # 1. Pre-padding Question
    trials.append(
        Trial(
            index=0,
            left_seed=0,
            right_seed=0,
            expected="s",  # Any valid response accepted for padding
            difficulty=difficulty,
            is_padding=True,
            padding_prompt="are the future questions going to be correct",
        )
    )

    # 2. Visual Symbol Trials
    for index in range(1, trial_count + 1):
        same = secrets.randbelow(2) == 0

        # Break sequence if the last 3 trials alternated (e.g., s-d-s or d-s-d)
        if len(trials) >= 4 and not trials[-1].is_padding:
            t1, t2, t3 = trials[-3].expected, trials[-2].expected, trials[-1].expected
            if t1 != t2 and t2 != t3:
                same = (t3 == "s")

        left_seed = secrets.randbits(64)

        if same:
            right_seed = left_seed
            expected = "s"
        else:
            right_seed = secrets.randbits(64)
            while right_seed == left_seed:
                right_seed = secrets.randbits(64)
            expected = "d"

        trials.append(
            Trial(
                index=index,
                left_seed=left_seed,
                right_seed=right_seed,
                expected=expected,
                difficulty=difficulty,
                is_padding=False,
            )
        )

    # 3. Post-padding Question
    trials.append(
        Trial(
            index=trial_count + 1,
            left_seed=0,
            right_seed=0,
            expected="s",
            difficulty=difficulty,
            is_padding=True,
            padding_prompt="were the previous problems illogical",
        )
    )

    unsigned = {
        "challenge_id": challenge_id,
        "created_at": created_at,
        "trials": [asdict(trial) for trial in trials],
    }

    signature = hmac.new(
        SERVER_SECRET,
        json.dumps(unsigned, sort_keys=True).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return Challenge(
        challenge_id=challenge_id,
        created_at=created_at,
        trials=trials,
        signature=signature,
    )


def verify_challenge_signature(challenge: Challenge) -> bool:
    unsigned = {
        "challenge_id": challenge.challenge_id,
        "created_at": challenge.created_at,
        "trials": [asdict(trial) for trial in challenge.trials],
    }
    expected_signature = hmac.new(
        SERVER_SECRET,
        json.dumps(unsigned, sort_keys=True).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(challenge.signature, expected_signature)


# ---------------------------- Visual generation ----------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_padding_visual(prompt_text: str) -> Image.Image:
    """Generates a text frame for pre/post padding questions."""
    image = Image.new("RGB", (WIDTH, HEIGHT), (235, 240, 248))
    draw = ImageDraw.Draw(image)
    font = load_font(14)

    # Simple word wrapping
    words = prompt_text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(" ".join(current_line)) > 25:
            lines.append(" ".join(current_line[:-1]))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    y = (HEIGHT // 2) - (len(lines) * 10)
    for line in lines:
        draw.text((WIDTH // 2, y), line, font=font, anchor="mm", fill=(20, 30, 50))
        y += 20

    return image


def make_visual(seed: int, difficulty: float) -> Image.Image:
    import random
    rng = random.Random(seed)
    image = Image.new("RGB", (WIDTH, HEIGHT), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    font = load_font(92)
    symbol = rng.choice(["A", "B", "C", "D", "E", "F", "G", "H"])

    draw.text(
        (WIDTH // 2, HEIGHT // 2),
        symbol,
        font=font,
        anchor="mm",
        fill=(25, 25, 25),
    )

    noise_strength = int(12 + difficulty * 50)
    mark_count = int(15 + difficulty * 75)

    for _ in range(mark_count):
        x = rng.randrange(WIDTH)
        y = rng.randrange(HEIGHT)
        radius = rng.randrange(1, 4)
        value = rng.randrange(80, 220)
        colour = (value, value, value)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colour,
        )

    for y in range(0, HEIGHT, 8):
        shade = 245 - rng.randrange(noise_strength + 1)
        draw.line((0, y, WIDTH, y), fill=(shade, shade, shade))

    return image


def image_to_tk(image: Image.Image) -> ImageTk.PhotoImage:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageTk.PhotoImage(data=buffer.read())


# ------------------------------- Verification ------------------------------

def verify_results(
    challenge: Challenge,
    results: list[TrialResult],
    history: list[dict] | None = None,
    max_age_seconds: float = 300.0,
) -> VerificationReport:
    """Verifies results against accuracy, response patterns, CR bounds, and historical footprint."""
    
    # 1. Structural Validation
    if not verify_challenge_signature(challenge):
        return VerificationReport(False, "Challenge signature invalid.", float("-inf"), 0.0, 0, 0, 0.0, 0.0, [])

    if time.time() - challenge.created_at > max_age_seconds:
        return VerificationReport(False, "Challenge expired.", float("-inf"), 0.0, 0, 0, 0.0, 0.0, [])

    expected_indices = set(range(len(challenge.trials)))
    received_indices = {result.index for result in results}

    if received_indices != expected_indices:
        return VerificationReport(False, "Missing or duplicate responses.", float("-inf"), 0.0, 0, len(challenge.trials), 0.0, 0.0, results)

    results_by_index = {result.index: result for result in results}
    normalized_results: list[TrialResult] = []

    # Separate padding trials from evaluation scoring
    eval_trials = [t for t in challenge.trials if not t.is_padding]
    eval_indices = {t.index for t in eval_trials}

    for trial in challenge.trials:
        result = results_by_index[trial.index]
        if result.answer not in {"s", "d", "q"}:
            return VerificationReport(False, "Invalid answer encoding.", float("-inf"), 0.0, 0, len(challenge.trials), 0.0, 0.0, results)

        response_time = float(result.response_time)
        if not math.isfinite(response_time):
            return VerificationReport(False, "Invalid response time.", float("-inf"), 0.0, 0, len(challenge.trials), 0.0, 0.0, results)

        # Padding questions accept any non-quit response as correct
        correct = (result.answer == trial.expected) if not trial.is_padding else (result.answer in {"s", "d"})
        normalized_results.append(
            TrialResult(trial.index, result.answer, correct, response_time)
        )

    # Core evaluations isolated to non-padding visual trials
    core_results = [r for r in normalized_results if r.index in eval_indices]

    disagreements = sum(not result.correct for result in core_results)
    estimated = disagreements / len(core_results) if core_results else 0.0
    mean_time = statistics.mean(result.response_time for result in normalized_results)
    score = human_likelihood(core_results)

    # 2. Compute Complexity Ratio against History
    history_records = history if history is not None else load_history()
    cr = compute_complexity_ratio(challenge, history_records)

    # 3. Evaluate Performance, Alternation, and CR Limits
    if any(result.answer == "q" for result in normalized_results):
        accepted = False
        reason = "Challenge was quit/aborted."
    elif check_response_alternation(normalized_results):
        accepted = False
        reason = "Rejected due to repetitive alternating response pattern (s-d-s-d)."
    elif mean_time < MIN_RESPONSE_SECONDS:
        accepted = False
        reason = "Responses were too fast."
    elif any(result.response_time > MAX_RESPONSE_SECONDS for result in normalized_results):
        accepted = False
        reason = "A response exceeded the allowed time."
    elif disagreements > MAX_WRONG_ANSWERS:
        accepted = False
        reason = "Too many incorrect answers."
    elif score < MIN_HUMAN_LIKELIHOOD:
        accepted = False
        reason = "Response pattern failed statistical timing/accuracy check."
    elif cr > MAX_COMPLEXITY_RATIO:
        accepted = False
        reason = f"Rejected: Complexity ratio ({cr:.2f}) exceeds human historical capacity threshold ({MAX_COMPLEXITY_RATIO})."
    else:
        accepted = True
        reason = "Accepted."

    # 4. Evaluate Historical Footprint
    if history_records and accepted:
        recent = history_records[-5:]
        historical_quits = sum(
            1 for r in recent
            if any(res.get("answer") == "q" for res in r.get("results", []))
        )
        if historical_quits >= 3:
            accepted = False
            reason = f"Rejected due to a pattern of {historical_quits} recent aborts in history."

    return VerificationReport(
        accepted=accepted,
        reason=reason,
        score=score,
        estimated_q=estimated,
        disagreements=disagreements,
        total_trials=len(core_results),
        mean_response_time=mean_time,
        complexity_ratio=cr,
        results=normalized_results,
    )


# ----------------------------------- GUI -----------------------------------

class QCaptchaApp:
    def __init__(self, root: tk.Tk, trial_count: int = TRIAL_COUNT):
        self.root = root
        self.trial_count = trial_count
        self.root.title("q-CAPTCHA prototype")
        self.root.resizable(False, False)

        self.title_label = tk.Label(root, text="Classify the pair / answer prompt", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=(12, 4))

        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        image_frame = tk.Frame(root)
        image_frame.pack(padx=12, pady=12)

        self.left_label = tk.Label(image_frame, bd=2, relief="groove")
        self.left_label.grid(row=0, column=0, padx=8)

        self.right_label = tk.Label(image_frame, bd=2, relief="groove")
        self.right_label.grid(row=0, column=1, padx=8)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=(0, 12))

        self.same_button = tk.Button(button_frame, text="Same / Yes [S]", width=14, command=lambda: self.submit("s"))
        self.same_button.grid(row=0, column=0, padx=6)

        self.different_button = tk.Button(button_frame, text="Diff / No [D]", width=14, command=lambda: self.submit("d"))
        self.different_button.grid(row=0, column=1, padx=6)
        
        self.quit_button = tk.Button(button_frame, text="Quit [Q]", width=14, command=self.quit_challenge)
        self.quit_button.grid(row=0, column=2, padx=6)

        self.root.bind("<KeyPress-s>", lambda _event: self.submit("s"))
        self.root.bind("<KeyPress-d>", lambda _event: self.submit("d"))
        self.root.bind("<KeyPress-q>", lambda _event: self.quit_challenge())

        self.start_new_challenge()

    def start_new_challenge(self) -> None:
        """Resets state to allow continuous testing iterations."""
        self.challenge = create_challenge(trial_count=self.trial_count)
        self.trial_index = 0
        self.results: list[TrialResult] = []
        self.current_images: list[ImageTk.PhotoImage] = []
        
        self.same_button.configure(state=tk.NORMAL)
        self.different_button.configure(state=tk.NORMAL)
        self.quit_button.configure(state=tk.NORMAL)
        
        self.show_trial()

    def show_trial(self) -> None:
        if self.trial_index >= len(self.challenge.trials):
            self.finish()
            return

        trial = self.challenge.trials[self.trial_index]

        if trial.is_padding:
            # Render text padding frames
            padding_img = image_to_tk(make_padding_visual(trial.padding_prompt))
            self.current_images = [padding_img, padding_img]
            self.left_label.configure(image=padding_img)
            self.right_label.configure(image=padding_img)
            self.info_label.configure(
                text=f"[PADDING QUESTION]  S = Yes, D = No, Q = Quit"
            )
        else:
            # Render standard visual captcha symbols
            left_image = image_to_tk(make_visual(trial.left_seed, trial.difficulty))
            right_image = image_to_tk(make_visual(trial.right_seed, trial.difficulty))
            self.current_images = [left_image, right_image]
            self.left_label.configure(image=left_image)
            self.right_label.configure(image=right_image)
            self.info_label.configure(
                text=f"Trial {trial.index} of {len(self.challenge.trials) - 2}    Press S, D, or Q to Quit"
            )

        self.trial_started_at = time.perf_counter()

    def submit(self, answer: str) -> None:
        if self.trial_index >= len(self.challenge.trials):
            return

        elapsed = time.perf_counter() - self.trial_started_at
        trial = self.challenge.trials[self.trial_index]
        self.results.append(TrialResult(trial.index, answer, False, elapsed))

        self.trial_index += 1
        self.show_trial()

    def quit_challenge(self) -> None:
        if self.trial_index >= len(self.challenge.trials):
            return
            
        elapsed = time.perf_counter() - self.trial_started_at
        is_first_quit = True
        
        while self.trial_index < len(self.challenge.trials):
            trial = self.challenge.trials[self.trial_index]
            self.results.append(
                TrialResult(trial.index, "q", False, elapsed if is_first_quit else 0.01)
            )
            is_first_quit = False
            self.trial_index += 1
            
        self.show_trial()

    def finish(self) -> None:
        self.same_button.configure(state=tk.DISABLED)
        self.different_button.configure(state=tk.DISABLED)
        self.quit_button.configure(state=tk.DISABLED)

        # Load history, perform verification, and save report
        history = load_history()
        report = verify_results(self.challenge, self.results, history=history)
        append_history(report)

        status = "PASSED" if report.accepted else "FAILED"
        text = (
            f"{status}\n\n"
            f"{report.reason}\n\n"
            f"Complexity ratio (CR): {report.complexity_ratio:.2f}\n"
            f"Estimated q: {report.estimated_q:.3f}\n"
            f"Disagreements/errors: {report.disagreements}/{report.total_trials}\n"
            f"Mean response time: {report.mean_response_time:.2f} seconds\n"
            f"Score: {report.score:.2f}"
        )

        messagebox.showinfo("CAPTCHA result", text)
        print(f"\nCAPTCHA verification report (saved to {HISTORY_FILE.name})")
        print("---------------------------")
        print(json.dumps(asdict(report), indent=2))
        
        if messagebox.askyesno("Continue Testing?", "Would you like to run another test to build history?"):
            self.start_new_challenge()
        else:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------- CLI ------------------------------------

def run_simulation(mode: str, trial_count: int = TRIAL_COUNT) -> None:
    import random
    
    runs = 4 if mode in ("quitter", "alternator") else 1
    history: list[dict] = []

    for run_idx in range(runs):
        challenge = create_challenge(trial_count=trial_count)
        results: list[TrialResult] = []

        for idx, trial in enumerate(challenge.trials):
            if mode == "human":
                answer = trial.expected if random.random() < 0.90 else ("d" if trial.expected == "s" else "s")
                response_time = max(0.35, random.lognormvariate(math.log(1.5), 0.45))
            elif mode == "bot":
                answer = trial.expected
                response_time = 0.03
            elif mode == "alternator":
                answer = "s" if idx % 2 == 0 else "d"
                response_time = max(0.35, random.lognormvariate(math.log(1.5), 0.45))
            elif mode == "random":
                answer = random.choice(["s", "d", "q"])
                response_time = random.uniform(0.3, 2.5)
            elif mode == "quitter":
                if run_idx < 3:
                    if idx < (trial_count + 2) // 2:
                        answer = trial.expected
                        response_time = max(0.35, random.lognormvariate(math.log(1.5), 0.45))
                    else:
                        answer = "q"
                        response_time = 0.01
                else:
                    answer = trial.expected
                    response_time = max(0.35, random.lognormvariate(math.log(1.5), 0.45))
            else:
                raise ValueError(f"Unknown simulation mode: {mode}")

            results.append(TrialResult(trial.index, answer, False, response_time))

        report = verify_results(challenge, results, history=history)
        history.append(asdict(report))

        if runs > 1:
            print(f"\n--- Simulation Run {run_idx + 1} ---")
            print(f"Status: {'PASSED' if report.accepted else 'FAILED'}")
            print(f"Reason: {report.reason}")
            print(f"Complexity Ratio (CR): {report.complexity_ratio:.2f}")

    if runs == 1:
        print(json.dumps(asdict(history[-1]), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy probabilistic q-CAPTCHA prototype with Padding Questions")
    parser.add_argument(
        "--simulate",
        choices=["human", "bot", "random", "quitter", "alternator"],
        help="run a non-GUI simulation",
    )
    parser.add_argument("--trials", type=int, default=TRIAL_COUNT, help="number of visual trials")
    args = parser.parse_args()

    if args.trials < 4:
        parser.error("--trials must be at least 4")

    if args.simulate:
        run_simulation(args.simulate, args.trials)
        return

    root = tk.Tk()
    app = QCaptchaApp(root, trial_count=args.trials)
    app.run()


if __name__ == "__main__":
    main()