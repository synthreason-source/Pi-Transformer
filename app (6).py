"""
app.py
======
Gradio front-end.  Imports everything from pi_transformer_core.
Run with:  python app.py
"""
from __future__ import annotations
import json, traceback
from typing import Optional

import gradio as gr

from pi_transformer_core import (
    AutomorphismTrainer,
    IsomorphismPipeline,
    LockedIsomorphismPipeline,
    build_pipeline,
    build_pipeline_from_preset,
    build_cpd,
    build_context_index,
    HF_DATASET_PRESETS,
)

import gradio as gr

# ── lazy-import the pipeline module ──────────────────────────────────────────
# Assumes pi_automorphism_net.py is in the same directory (or on PYTHONPATH).

# ═════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG  (all keys mirror build_pipeline / AutomorphismTrainer kwargs)
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG: dict = {
    # ── dataset ──────────────────────────────────────────────────────────────
    "preset":           "imdb",        # one of HF_DATASET_PRESETS keys or ""
    "dataset_name":     "",            # used when preset == ""
    "config_name":      "",            # HF sub-config, e.g. "wikitext-2-raw-v1"
    "text_fields":      [],            # list of field names; [] = auto-detect
    # ── preprocessor ─────────────────────────────────────────────────────────
    "max_per_split":    1000,
    "boundaryquota":    8,
    "minlen":           3,
    "streaming":        False,
    # ── pipeline ─────────────────────────────────────────────────────────────
    "locked":           True,
    "ngram_n":          3,
    "lidstone_gamma":   0.1,
    "temperature":      4.3,
    "top_k":            100,
    "top_p":            1.0,
    "rep_penalty":      1.13,
    "insight_penalty":  3.95,
    "l12_blend_alpha":  0.5,
    "l13_sigma":        0.30,
    "l13_floor":        0.04,
    # ── trainer ──────────────────────────────────────────────────────────────
    "n_cands":          100,
    "lr":               5e-4,
    "hidden":           128,
    "weight_decay":     1e-4,
    # ── warmup ───────────────────────────────────────────────────────────────
    "warmup_steps":     50,
    "warmup_batch":     8,
    "warmup_log_every": 25,
    # ── training run ─────────────────────────────────────────────────────────
    "train_steps":      40,
    "train_patience":   8,
    "train_log_every":  10,
    # ── generation ───────────────────────────────────────────────────────────
    "n_words":          120,
    "seed":             42,
}

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_pipeline_cache: dict = {}   # key → (pipe, pre)

def _merge_config(json_file_path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if json_file_path:
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            cfg.update(overrides)
        except Exception as e:
            raise ValueError(f"Could not parse JSON config: {e}")
    return cfg


def _corpus_from_file(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def _build(cfg: dict, corpus_text: Optional[str] = None):
    """Build (or retrieve cached) pipeline from config + optional corpus."""
    cache_key = json.dumps(cfg, sort_keys=True, default=str) + str(bool(corpus_text))
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    pipe_kw = {k: cfg[k] for k in (
        "temperature", "top_k", "top_p", "rep_penalty",
        "insight_penalty", "l12_blend_alpha", "l13_sigma", "l13_floor",
    )}
    pre_kw = {k: cfg[k] for k in (
        "max_per_split", "boundaryquota", "minlen", "streaming",
    )}
    if cfg["max_per_split"]:
        pre_kw["max_per_split"] = int(cfg["max_per_split"])

    if corpus_text:
        # ── file-upload mode: skip HF download, build CPD directly ───────────
        cpd, vocab, tokens = build_cpd(
            corpus_text, cfg["ngram_n"], cfg["lidstone_gamma"]
        )
        ctx_idx = build_context_index(vocab, cpd, tokens)

        # Minimal stub so the pipeline has a .preprocessor attribute
        class _FakePre:
            sentences = [tokens]
            this_tokens = tokens
            def tocorpus(self): return " ".join(self.this_tokens)

        pre = _FakePre()
        pre.this_tokens = tokens
        cls = LockedIsomorphismPipeline if cfg["locked"] else IsomorphismPipeline
        pipe = cls(cpd, ctx_idx, vocab, ngram_n=cfg["ngram_n"], **pipe_kw)
        pipe.preprocessor = pre
    else:
        # ── HF dataset mode ──────────────────────────────────────────────────
        preset = cfg.get("preset", "").strip()
        if preset and preset in HF_DATASET_PRESETS:
            pipe, pre = build_pipeline_from_preset(
                preset,
                locked=cfg["locked"],
                ngram_n=cfg["ngram_n"],
                lidstone_gamma=cfg["lidstone_gamma"],
                preprocessor_kw=pre_kw,
                pipeline_kw=pipe_kw,
            )
        else:
            ds_name = cfg.get("dataset_name", "").strip()
            if not ds_name:
                raise ValueError("Provide a HF dataset name or upload a text file.")
            tf = cfg.get("text_fields") or None
            pipe, pre = build_pipeline(
                ds_name,
                config_name=cfg.get("config_name") or None,
                text_fields=tf,
                locked=cfg["locked"],
                ngram_n=cfg["ngram_n"],
                lidstone_gamma=cfg["lidstone_gamma"],
                preprocessor_kw=pre_kw,
                pipeline_kw=pipe_kw,
            )

    _pipeline_cache[cache_key] = (pipe, pre)
    return pipe, pre


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

def run_generate(
    # ── source ────────────────────────────────────────────────────────────────
    preset_choice: str,
    custom_dataset: str,
    hf_config_name: str,
    corpus_file,          # gr.File → filepath string or None
    json_config_file,     # gr.File → filepath string or None
    # ── pipeline params ──────────────────────────────────────────────────────
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    insight_penalty: float,
    l12_blend_alpha: float,
    locked: bool,
    ngram_n: int,
    max_per_split: int,
    # ── generation ───────────────────────────────────────────────────────────
    prompt: str,
    n_words: int,
    seed: int,
):
    try:
        cfg = _merge_config(json_config_file)

        # UI values override config-file values
        if preset_choice and preset_choice != "(custom)":
            cfg["preset"] = preset_choice
            cfg["dataset_name"] = ""
        else:
            cfg["preset"] = ""
            cfg["dataset_name"] = custom_dataset.strip()

        cfg.update({
            "config_name":     hf_config_name.strip(),
            "temperature":     temperature,
            "top_k":           int(top_k),
            "top_p":           top_p,
            "rep_penalty":     rep_penalty,
            "insight_penalty": insight_penalty,
            "l12_blend_alpha": l12_blend_alpha,
            "locked":          locked,
            "ngram_n":         int(ngram_n),
            "max_per_split":   int(max_per_split),
            "n_words":         int(n_words),
            "seed":            int(seed),
        })

        corpus_text = _corpus_from_file(corpus_file) if corpus_file else None
        pipe, _ = _build(cfg, corpus_text)

        text = pipe.generate_text(
            prompt, cfg["n_words"], seed=cfg["seed"], capitalise=True
        )
        return text, "✅ Done"
    except Exception:
        return "", f"❌ Error{traceback.format_exc()}"


def run_train_and_generate(
    preset_choice, custom_dataset, hf_config_name,
    corpus_file, json_config_file,
    temperature, top_k, top_p, rep_penalty, insight_penalty,
    l12_blend_alpha, locked, ngram_n, max_per_split,
    prompt, n_words, seed,
    warmup_steps, train_steps, lr,
):
    try:
        cfg = _merge_config(json_config_file)
        if preset_choice and preset_choice != "(custom)":
            cfg["preset"] = preset_choice; cfg["dataset_name"] = ""
        else:
            cfg["preset"] = ""; cfg["dataset_name"] = custom_dataset.strip()
        cfg.update({
            "config_name": hf_config_name.strip(),
            "temperature": temperature, "top_k": int(top_k), "top_p": top_p,
            "rep_penalty": rep_penalty, "insight_penalty": insight_penalty,
            "l12_blend_alpha": l12_blend_alpha, "locked": locked,
            "ngram_n": int(ngram_n), "max_per_split": int(max_per_split),
            "n_words": int(n_words), "seed": int(seed),
            "warmup_steps": int(warmup_steps), "train_steps": int(train_steps),
            "lr": float(lr),
        })

        corpus_text = _corpus_from_file(corpus_file) if corpus_file else None
        pipe, _ = _build(cfg, corpus_text)

        trainer = AutomorphismTrainer(
            pipe,
            n_cands=cfg["n_cands"],
            lr=cfg["lr"],
            hidden=cfg["hidden"],
            weight_decay=cfg["weight_decay"],
        )
        log_lines: list[str] = []
        def _pr(*a): log_lines.append(" ".join(str(x) for x in a))

        import builtins
        _orig_print = builtins.print
        builtins.print = _pr          # capture trainer output

        trainer.warmup(
            steps=cfg["warmup_steps"],
            batch=cfg["warmup_batch"],
            log_every=cfg["warmup_log_every"],
        )
        trainer.run(
            n_steps=cfg["train_steps"],
            prompt=prompt, n_words=int(n_words), seed=int(seed),
            log_every=cfg["train_log_every"],
            patience=cfg["train_patience"],
        )
        trainer.report()
        builtins.print = _orig_print  # restore

        text = pipe.generate_text(prompt, int(n_words), seed=int(seed), capitalise=True)
        train_log = "\n".join(log_lines)
        return text, train_log, "✅ Training complete"
    except Exception:
        import builtins; builtins.print = __builtins__["print"] if isinstance(__builtins__, dict) else __import__("builtins").print
        return "", "", f"❌ Error\n{traceback.format_exc()}"


def load_json_preview(json_file):
    if not json_file:
        return json.dumps(DEFAULT_CONFIG, indent=2)
    try:
        with open(json_file, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

PRESETS = ["(custom)"] + sorted(HF_DATASET_PRESETS.keys())

EXAMPLE_CONFIG = json.dumps({
    "preset":        "imdb",
    "max_per_split": 500,
    "ngram_n":       3,
    "temperature":   5.0,
    "top_k":         80,
    "top_p":         0.95,
    "locked":        True,
    "n_words":       80,
    "seed":          7,
}, indent=2)

with gr.Blocks(title="π-Automorphism Net", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🌀 π-Automorphism Text Generator\n"
        "Isomorphism pipeline with learnable layers, rule assertions, and an "
        "automorphism trainer.  Configure via the UI, a **JSON config file**, "
        "or a **plain-text corpus upload**."
    )

    # ── TOP ROW: source + config upload ──────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📂 Data Source")
            preset_dd = gr.Dropdown(
                choices=PRESETS, value="imdb", label="HuggingFace Preset",
            )
            custom_ds = gr.Textbox(
                label="Custom HF dataset name",
                placeholder="e.g.  wikitext  (ignored when preset ≠ custom)",
            )
            hf_cfg_name = gr.Textbox(
                label="HF config / sub-name",
                placeholder="e.g.  wikitext-2-raw-v1",
            )
            corpus_upload = gr.File(
                label="📄 Upload plain-text corpus  (.txt)  — overrides HF dataset",
                file_types=[".txt"],
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Config File")
            json_upload = gr.File(
                label="📋 Upload JSON config  (optional)",
                file_types=[".json"],
            )
            json_preview = gr.Code(
                value=json.dumps(DEFAULT_CONFIG, indent=2),
                language="json",
                label="Active config preview",
                lines=18,
            )
            json_upload.change(load_json_preview, json_upload, json_preview)

    # ── PIPELINE PARAMS ───────────────────────────────────────────────────────
    with gr.Accordion("🔧 Pipeline parameters", open=False):
        with gr.Row():
            temperature   = gr.Slider(0.1, 20.0, value=4.3,  step=0.1, label="Temperature")
            top_k         = gr.Slider(1,   500,  value=100,  step=1,   label="Top-K")
            top_p         = gr.Slider(0.0, 1.0,  value=1.0,  step=0.01,label="Top-P")
        with gr.Row():
            rep_penalty   = gr.Slider(1.0, 5.0,  value=1.13, step=0.01,label="Repetition penalty")
            insight_pen   = gr.Slider(0.0, 10.0, value=3.95, step=0.05,label="Insight penalty")
            l12_alpha     = gr.Slider(0.0, 1.0,  value=0.5,  step=0.01,label="L12 blend α")
        with gr.Row():
            locked_chk    = gr.Checkbox(value=True, label="Locked pipeline (L14)")
            ngram_n       = gr.Slider(2, 5, value=3, step=1, label="N-gram order")
            max_per_split = gr.Number(value=1000, label="Max examples per split", precision=0)

    # ── GENERATION TAB ────────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.TabItem("✍️ Generate"):
            with gr.Row():
                prompt_box = gr.Textbox(
                    value="tell me about yourself",
                    label="Prompt", lines=2, scale=3,
                )
                with gr.Column(scale=1):
                    n_words_sl = gr.Slider(10, 500, value=120, step=10, label="Words to generate")
                    seed_num   = gr.Number(value=42, label="Seed", precision=0)
            gen_btn    = gr.Button("🚀 Generate", variant="primary")
            gen_out    = gr.Textbox(label="Generated text", lines=10, interactive=False)
            gen_status = gr.Textbox(label="Status", lines=2, interactive=False)

            gen_btn.click(
                fn=run_generate,
                inputs=[
                    preset_dd, custom_ds, hf_cfg_name,
                    corpus_upload, json_upload,
                    temperature, top_k, top_p,
                    rep_penalty, insight_pen, l12_alpha,
                    locked_chk, ngram_n, max_per_split,
                    prompt_box, n_words_sl, seed_num,
                ],
                outputs=[gen_out, gen_status],
            )

        with gr.TabItem("🏋️ Train then Generate"):
            with gr.Row():
                warmup_steps_sl = gr.Slider(0, 500, value=50, step=10, label="Warmup steps")
                train_steps_sl  = gr.Slider(0, 500, value=40, step=10, label="Train steps")
                lr_num          = gr.Number(value=5e-4, label="Learning rate")
            train_btn    = gr.Button("🏋️ Train & Generate", variant="primary")
            train_out    = gr.Textbox(label="Generated text (post-training)", lines=8, interactive=False)
            train_log    = gr.Textbox(label="Training log", lines=14, interactive=False)
            train_status = gr.Textbox(label="Status", lines=2, interactive=False)

            train_btn.click(
                fn=run_train_and_generate,
                inputs=[
                    preset_dd, custom_ds, hf_cfg_name,
                    corpus_upload, json_upload,
                    temperature, top_k, top_p,
                    rep_penalty, insight_pen, l12_alpha,
                    locked_chk, ngram_n, max_per_split,
                    prompt_box, n_words_sl, seed_num,
                    warmup_steps_sl, train_steps_sl, lr_num,
                ],
                outputs=[train_out, train_log, train_status],
            )

        with gr.TabItem("📖 Config template"):
            gr.Markdown(
                "Copy this template, fill in your values, save as `config.json`, "
                "and upload it in the **Config File** panel."
            )
            gr.Code(value=EXAMPLE_CONFIG, language="json", label="config.json template")

    gr.Markdown(
        "---\n"
        "**Tips**  ·  "
        "Upload a `.txt` file to skip the HuggingFace download and use your own corpus.  "
        "A `.json` config overrides every parameter shown here.  "
        "Cached pipelines are reused within a session to avoid re-downloading."
    )

if __name__ == "__main__":
    demo.launch(share=False)
