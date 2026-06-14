"""
fork_gradio.py
──────────────
Refactored Gradio interface using Hugging Face Transformers.
Implements the fork-weight redistribution mechanism as a custom LogitsProcessor.

Run:
    python fork_gradio.py
"""

import math
import random
import torch
import gradio as gr
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LogitsProcessor, 
    LogitsProcessorList
)

# ── Custom Logits Processor for Fork-Weight Redistribution ────────────────────

class ForkWeightLogitsProcessor(LogitsProcessor):
    """
    Detects high-probability tokens matching specific string criteria and 
    redistributes a portion of their probability mass to their immediate neighbors
    in the vocabulary distribution.
    """
    def __init__(self, tokenizer, min_prob: float = 0.15, min_len: int = 6, alpha: float = 0.5, enabled: bool = True):
        self.tokenizer = tokenizer
        self.min_prob = min_prob
        self.min_len = min_len
        self.alpha = alpha
        self.enabled = enabled
        # Cache token string lengths to avoid redundant decoding during loops
        self.vocab_size = len(tokenizer)
        self.token_lens = [len(tokenizer.decode([i]).strip()) for i in range(self.vocab_size)]
        self.fork_events = [] # Storage log for UI tracking (wiped per generation run)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.enabled:
            return scores

        # Process batch items individually (assuming batch_size=1 for generation UI)
        for batch_idx in range(scores.shape[0]):
            logits = scores[batch_idx]
            probs = torch.softmax(logits, dim=-1)
            
            # Find tokens meeting criteria
            order = torch.argsort(probs, descending=True)
            forked_indices = []
            
            for idx in order.tolist():
                if probs[idx].item() < self.min_prob:
                    break
                if self.token_lens[idx] >= self.min_len:
                    forked_indices.append(idx)
                    if len(forked_indices) >= 8: # Keep safety ceiling from original script
                        break
                        
            if not forked_indices:
                continue

            # Log events for UI display
            current_step = input_ids.shape[1]
            forked_words = [self.tokenizer.decode([i]).strip() for i in forked_indices]
            self.fork_events.append((current_step, forked_words))

            # Mutate distribution mass
            new_probs = probs.clone()
            for fi in forked_indices:
                mass = probs[fi].item()
                keep = mass * self.alpha
                spread = mass * (1.0 - self.alpha)
                
                new_probs[fi] = keep
                
                # Redistribute to index neighbors in the vocabulary space
                left = fi - 1 if fi > 0 else None
                right = fi + 1 if fi < self.vocab_size - 1 else None
                
                if left is not None and right is not None:
                    new_probs[left] += spread * 0.5
                    new_probs[right] += spread * 0.5
                elif left is not None:
                    new_probs[left] += spread
                elif right is not None:
                    new_probs[right] += spread

            # Safeguard log operations and re-convert probabilities back to logits
            new_probs = torch.clamp(new_probs, min=1e-12)
            scores[batch_idx] = torch.log(new_probs)
            
        return scores


# ── Global Model Setup ────────────────────────────────────────────────────────

MODEL_NAME = "gpt2" # Lightweight base model for testing. Can switch to a fine-tuned checkpoint.
print(f"Loading {MODEL_NAME} model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set padding token to avoid errors during generation configuration
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── Generation Core ───────────────────────────────────────────────────────────

def generate_text(
    prompt: str,
    n_words: int,
    seed: int,
    temperature: float,
    rep_penalty: float,
    fork_enabled: bool,
    fork_min_prob: float,
    fork_min_len: int,
    fork_alpha: float,
) -> tuple:
    # Set seed reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # Initialize processor
    fork_processor = ForkWeightLogitsProcessor(
        tokenizer=tokenizer,
        min_prob=fork_min_prob,
        min_len=fork_min_len,
        alpha=fork_alpha,
        enabled=fork_enabled
    )
    
    processors = LogitsProcessorList([fork_processor])

    # Run Transformer generation loop
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(n_words),
            do_sample=True,
            temperature=float(temperature),
            repetition_penalty=float(rep_penalty),
            logits_processor=processors,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    fork_events = fork_processor.fork_events

    # Stats preparation
    tokens_produced = output_ids.shape[1] - input_len
    stats = f"Generated: {tokens_produced} tokens · Fork events: {len(fork_events)}"

    # Parse fork log for output textbox
    if fork_events:
        lines = [f"  token {step:>3}  ←  {', '.join(words)}" for step, words in fork_events[:12]]
        fork_log = "Mass spread from:\n" + "\n".join(lines)
    else:
        fork_log = "No fork events (try lowering min prob or enabling forks)."

    return generated_text, stats, fork_log


# ── Gradio Callback ───────────────────────────────────────────────────────────

def run(
    corpus_file,  # Preserved interface element for optional fine-tuning extensions
    corpus_text: str,
    prompt: str,
    n_words: int,
    seed: int,
    ngram_n: int, # Hidden design choice note: context handles automatically now
    temperature: float,
    rep_penalty: float,
    fork_enabled: bool,
    fork_min_prob: float,
    fork_min_len: int,
    fork_alpha: float,
):
    # Note: Traditional N-gram corpus parsing is swapped for pretrained LLM context.
    # To use a raw file on-the-fly, you would perform a localized LoRA or fine-tuning run.
    # Instead, we evaluate generations directly using the Prompt base inputs.
    
    if not prompt:
        prompt = "The quick fox jumps over"

    try:
        text, stats, fork_log = generate_text(
            prompt=prompt,
            n_words=n_words,
            seed=seed,
            temperature=temperature,
            rep_penalty=rep_penalty,
            fork_enabled=fork_enabled,
            fork_min_prob=fork_min_prob,
            fork_min_len=fork_min_len,
            fork_alpha=fork_alpha
        )
        return text, stats, fork_log
    except Exception as e:
        return f"⚠️ Generation error: {e}", "—", "—"


# ── UI Layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Neural Fork-Weight Generator") as demo:

    gr.Markdown(
        """
# 🌿 Neural Fork-Weight Text Generator
Probability forks are detected at each sampling step — high-mass long tokens
spread half their weight to architectural neighbors in the transformer logit matrix.
        """
    )

    with gr.Row():
        # ── left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Fine-Tuning Context (Optional/Placeholders)")
            corpus_file = gr.File(label="Upload Training Corpus (.txt)", file_types=[".txt"], file_count="single")
            corpus_box = gr.Textbox(value="", label="Or paste plain text directly", lines=1, placeholder="Ignored in zero-shot inference...")

            gr.Markdown("### Prompt & Length")
            prompt_box = gr.Textbox(value="The quick brown fox jumps over the", label="Seed prompt", lines=2)
            run_btn = gr.Button("▶  Generate", variant="primary", size="lg")

            with gr.Row():
                n_words_sl = gr.Slider(10, 150, value=40, step=5, label="Tokens to generate")
                seed_num   = gr.Number(value=42, label="Seed", precision=0)

            gr.Markdown("### Model Configuration")
            with gr.Row():
                ngram_sl = gr.Slider(2, 5, value=3, step=1, label="N-gram constraint (Legacy Display Only)", visible=False)
                temp_sl  = gr.Slider(0.1, 2.5, value=0.8, step=0.1, label="Temperature")
                rep_sl   = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition penalty")

            gr.Markdown("### Fork Settings")
            fork_chk = gr.Checkbox(value=True, label="Enable fork weight redistribution")
            with gr.Row():
                fp_sl  = gr.Slider(0.01, 0.30, value=0.08, step=0.01, label="Min prob to fork")
                fl_sl  = gr.Slider(2, 10,    value=5,    step=1,    label="Min word length")
                fa_sl  = gr.Slider(0.1, 0.9, value=0.6,  step=0.05, label="Alpha (mass kept)")

        # ── right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            out_text = gr.Textbox(label="Generated text", lines=10, interactive=False)
            stats_box = gr.Textbox(label="Stats", lines=1, interactive=False)
            fork_log_box = gr.Textbox(label="Fork log", lines=10, interactive=False)

    run_btn.click(
        fn=run,
        inputs=[
            corpus_file, corpus_box, prompt_box, n_words_sl, seed_num,
            ngram_sl, temp_sl, rep_sl,
            fork_chk, fp_sl, fl_sl, fa_sl,
        ],
        outputs=[out_text, stats_box, fork_log_box],
    )

if __name__ == "__main__":
    demo.launch(share=False)
