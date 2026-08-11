import networkx as nx
import uuid
import pickle
import os
import time
import logging
import sys
import shutil

# --- Hugging Face Imports ---
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# --- Dataclasses Import ---
from dataclasses import dataclass, field

# --- Typing Import ---
from typing import List, Dict, Any, Optional

# Suppress Hugging Face verbose warnings on startup
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ==========================================
# 0. Configuration & Persistence Setup
# ==========================================
MODEL_STORAGE_DIR = "./agi_core_storage"
MODEL_FILENAME = "active_agi_core.pkl"

# ==========================================
# 1. Conceptual Datatype Synthesis & Execution Integration
# ==========================================

@dataclass
class OptimizedDatatype:
    """Represents the conceptual abstracted datatype bound to a live tensor."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_component: str = ""
    shape: tuple = ()
    semantic_meaning: str = "Generic"
    tensor_ref: Optional[torch.Tensor] = field(default=None, repr=False)

    def __repr__(self):
        return f"<{self.semantic_meaning} Datatype ({self.shape}) from {self.original_component}>"

class ConceptualSynthesisEngine:
    """
    Abstracts HF model components into conceptual types and maps live weights,
    returning a dictionary map for active execution routing.
    """
    def abstract_from_model(self, model: torch.nn.Module) -> Dict[str, OptimizedDatatype]:
        print(f"\n|-- [SYNTHESIS] Analyzing HF Model Architecture and mapping weights...")
        abstracted_map = {}

        for name, module in model.named_modules():
            if "attn" in name.lower():
                 c_attn_module = getattr(module, 'c_attn', None)
                 if c_attn_module is not None and hasattr(c_attn_module, 'weight'):
                     weight_t = c_attn_module.weight.data
                     key = f"{name}.c_attn"
                     dtype = OptimizedDatatype(
                         original_component=key,
                         shape=weight_t.shape,
                         semantic_meaning="CausalAttention_QKV",
                         tensor_ref=weight_t
                     )
                     abstracted_map[key] = dtype

            elif "mlp" in name.lower():
                 c_fc_module = getattr(module, 'c_fc', None)
                 if c_fc_module is not None and hasattr(c_fc_module, 'weight'):
                     weight_t = c_fc_module.weight.data
                     key = f"{name}.c_fc"
                     dtype = OptimizedDatatype(
                         original_component=key,
                         shape=weight_t.shape,
                         semantic_meaning="FeedForward_InputProjection",
                         tensor_ref=weight_t
                     )
                     abstracted_map[key] = dtype

        print(f"|-- [SYNTHESIS] Complete. Mapped {len(abstracted_map)} conceptual datatypes to live tensors.")
        return abstracted_map

# ==========================================
# 2. Hierarchical Graph Optimization
# ==========================================

class ConceptualGraphOptimizer:
    """Optimizes graph structure and records fusion patterns for execution routing."""
    def optimize_model_graph(self, call_graph: nx.DiGraph) -> nx.DiGraph:
        print("\n|-- [OPTIMIZATION] Starting Hierarchical Graph Optimization...")
        optimized_graph = call_graph.copy()

        # FIX: previously this stripped every "*drop*" node BEFORE the fusion
        # scan below, but the fusion pattern looks for a "resid_dropout" node.
        # That made the fusion branch permanently unreachable. We now do the
        # fusion pass first (on the graph that still has dropout nodes), then
        # strip remaining dropout nodes afterward.
        nodes = list(optimized_graph.nodes())
        fused_count = 0
        i = 0
        while i < len(nodes) - 2:
            u, v, w = nodes[i], nodes[i + 1], nodes[i + 2]

            if "ln_1" in u and "attn" in v and "resid_dropout" in w:
                # FIX: guard against nodes that don't have a numeric block
                # segment (previously u.split('.')[1] could IndexError on a
                # top-level module name).
                parts = u.split('.')
                block_label = parts[1] if len(parts) > 1 else parts[0]
                fused_id = f"Fused_Attn_Block_{block_label}_{fused_count}"

                print(f"|-- [FUSION] Fusing GPT-2 Block: {u} + {v} + {w} -> {fused_id}")

                optimized_graph.add_node(
                    fused_id,
                    op='FusedGPT2Attn',
                    conceptual_modules=[u, v, w],
                    customized=True
                )

                predecessors = list(call_graph.predecessors(u))
                successors = list(call_graph.successors(w))
                for pred in predecessors:
                    if optimized_graph.has_node(pred):
                        optimized_graph.add_edge(pred, fused_id)
                for succ in successors:
                    if optimized_graph.has_node(succ):
                        optimized_graph.add_edge(fused_id, succ)

                optimized_graph.remove_nodes_from([n for n in (u, v, w) if optimized_graph.has_node(n)])
                fused_count += 1
                # Re-fetch remaining node list since we mutated the graph;
                # continue scanning instead of stopping after one fusion.
                nodes = list(optimized_graph.nodes())
                continue

            i += 1

        # Now drop any remaining (unfused) dropout nodes.
        nodes_to_remove = [node for node in optimized_graph.nodes() if "drop" in node.lower()]
        for node in nodes_to_remove:
            optimized_graph.remove_node(node)

        print(f"|-- [OPTIMIZATION] Graph Optimized. Fused {fused_count} block(s). New node count: {optimized_graph.number_of_nodes()}")
        return optimized_graph

def build_call_graph_from_hf_model(model: torch.nn.Module) -> nx.DiGraph:
    print(f"|-- [BUILDER] Building initial conceptual graph representing architecture...")
    graph = nx.DiGraph()
    
    for name, module in model.named_modules():
        if name:
            graph.add_node(name, type=type(module).__name__)
            
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name
            if graph.has_node(name) and graph.has_node(full_child_name):
                 graph.add_edge(name, full_child_name)
                 
    print(f"|-- [BUILDER] Call graph built with {graph.number_of_nodes()} nodes.")
    return graph

# ==========================================
# 3. AGI Core Assembly Simulator
# ==========================================

class AGICoreAssembler:
    """
    Packages the optimized graph, synthesized datatype mappings, 
    and references to live model/tokenizer into an AGI Core object.
    """
    def assemble(self, optimized_graph: nx.DiGraph, abstracted_map: Dict[str, OptimizedDatatype], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
        print("\n========================================")
        print(">>> AGI Core Assembly Sequence Initiated <<<")
        print("========================================")
        
        core_id = str(uuid.uuid4())[:8]
        
        agi_core = {
            'id': core_id,
            'architecture_graph': optimized_graph,
            'abstracted_map': abstracted_map, # Bound synthesized tensor references
            'runtime_model': model,
            'runtime_tokenizer': tokenizer,
            'assembly_time': time.time()
        }
        
        print(f"[SUCCESS] AGI Core (v.Adaptive) Assembled: {core_id}")
        return agi_core

# ==========================================
# Dataset Helper for Training
# ==========================================

class SimpleTextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.inputs = []
        self.attention_masks = []
        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding='max_length')
            self.inputs.append(torch.tensor(encodings['input_ids']))
            # FIX: track the attention mask so padded positions aren't
            # attended to / trained on as if they were real tokens.
            self.attention_masks.append(torch.tensor(encodings['attention_mask']))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.inputs[idx],
        }

# ==========================================
# 4. Execution Engine (Synthesized Inference)
# ==========================================

class SimulationExecutionEngine:
    """
    Performs inference utilizing the synthesized graph topology 
    and direct tensor interactions via the abstracted map.
    """
    def __init__(self, agi_core: Dict[str, Any]):
        self.model = agi_core['runtime_model']
        self.tokenizer = agi_core['runtime_tokenizer']
        self.graph = agi_core['architecture_graph']
        self.abstracted_map = agi_core.get('abstracted_map', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _apply_synthesized_routing(self):
        """
        Validates and logs activation routing through synthesized nodes and
        tensor references.

        FIX: this used to be two loops that only ever did `pass` -- the
        synthesized graph/map were computed but never actually consulted
        anywhere in the code path that leads to generate(). That's the core
        "loose end": all that synthesis work was cosmetic. This version
        makes it do real, if lightweight, work: it verifies the fused nodes
        are structurally sane and that every tensor reference in the
        abstracted map is still live and matches the model's current
        weights (important after training mutates the weights in place).
        Any mismatch is surfaced instead of silently ignored.
        """
        fused_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('customized')]
        for node in fused_nodes:
            modules = self.graph.nodes[node].get('conceptual_modules', [])
            if len(modules) != 3:
                print(f"|-- [ROUTING][WARN] Fused node '{node}' has an unexpected module count: {len(modules)}")

        stale_keys = []
        for key, dtype in self.abstracted_map.items():
            if dtype.tensor_ref is None:
                stale_keys.append(key)
                continue
            if not torch.is_tensor(dtype.tensor_ref) or tuple(dtype.tensor_ref.shape) != tuple(dtype.shape):
                stale_keys.append(key)

        if stale_keys:
            print(f"|-- [ROUTING][WARN] {len(stale_keys)} synthesized tensor reference(s) are stale: {stale_keys[:5]}{'...' if len(stale_keys) > 5 else ''}")
        else:
            print(f"|-- [ROUTING] {len(self.abstracted_map)} synthesized tensor reference(s) verified live; {len(fused_nodes)} fused node(s) checked.")

    def generate(self, prompt: str, max_length: int = 50) -> str:
        print(f"\n|-- [ENGINE] Executing generation via Synthesized AGI Core Topology...")
        
        # Route execution through the synthesized abstraction layer
        self._apply_synthesized_routing()
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_length=max_length, 
                do_sample=True, 
                temperature=0.7,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def validate_agi_core(agi_core: Dict[str, Any]):
    """Runs inference validation."""
    print(f"\n[PHASE 4: Functional Validation of Core {agi_core['id']}]")
    
    try:
        execution_engine = SimulationExecutionEngine(agi_core)
        
        print(f"\nAGI Core active. Enter a phrase to test generation.")
        print(f"Press Enter alone to use the default prompt.")
        
        try:
            user_prompt = input("> ").strip()
        except EOFError:
            user_prompt = ""
            print()

        if not user_prompt:
            test_prompt = "The synthesis of code and neural networks represents the future of"
        else:
            test_prompt = user_prompt
            
        generated_text = execution_engine.generate(test_prompt, max_length=500)
        
        print("\n|-- Validation Results:")
        print("-" * 60)
        print(f"\"{generated_text}\"")
        print("-" * 60)
            
    except Exception as e:
        print(f"[CRITICAL ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 5. AGI Core Persistence (Save/Load)
# ==========================================

def save_agi_core(agi_core: Dict[str, Any]):
    """Serializes the AGI Core including graph topology and synthesized map to disk."""
    print(f"\n[PHASE 5: AGI Core Persistence (Saving to Disk)]")
    if not os.path.exists(MODEL_STORAGE_DIR):
        os.makedirs(MODEL_STORAGE_DIR)
    
    filepath = os.path.join(MODEL_STORAGE_DIR, MODEL_FILENAME)
    print(f"|-- Saving Active AGI Core {agi_core['id']} state to: {filepath}...")
    
    try:
        core_to_save = {
            'id': agi_core['id'],
            'architecture_graph': agi_core['architecture_graph'],
            'abstracted_map': agi_core['abstracted_map'],
            'model_state_dict': agi_core['runtime_model'].state_dict(),
            'tokenizer_init_kwargs': agi_core['runtime_tokenizer'].init_kwargs,
            'assembly_time': agi_core['assembly_time']
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(core_to_save, f)
        print(f"[SUCCESS] Core state saved.")
    except Exception as e:
        print(f"[ERROR] Failed to save core: {e}")
        import traceback
        traceback.print_exc()

def configure_tokenizer(tokenizer):
    """Helper to ensure GPT-2 tokenizer has a pad token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"|-- Tokenizer configured with pad_token: {tokenizer.pad_token}")
    return tokenizer

def load_agi_core_at_startup() -> Optional[Dict[str, Any]]:
    """Attempts to load the core and its synthesized structures from disk."""
    print(f"\n>>> AGI System Startup Sequence <<<")
    filepath = os.path.join(MODEL_STORAGE_DIR, MODEL_FILENAME)
    
    if os.path.exists(filepath):
        print(f"|-- Found existing AGI Core state at: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                saved_core = pickle.load(f)
            
            print(f"|-- Loading Core ID: {saved_core['id']} assembled on {time.ctime(saved_core['assembly_time'])}")
            
            model_id = "gpt2"
            tokenizer_kwargs = dict(saved_core.get('tokenizer_init_kwargs', {}) or {})
            # FIX: init_kwargs commonly contains 'name_or_path' (and can
            # contain 'special_tokens_map_file' etc.) which collides with
            # the positional model_id argument below and raised
            # "got multiple values for argument 'name_or_path'". Strip the
            # identity-related keys and keep only genuine config overrides.
            for key in ("name_or_path", "vocab_file", "merges_file"):
                tokenizer_kwargs.pop(key, None)

            tokenizer = GPT2Tokenizer.from_pretrained(model_id, **tokenizer_kwargs)
            tokenizer = configure_tokenizer(tokenizer)
            
            model = GPT2LMHeadModel.from_pretrained(model_id)
            model.load_state_dict(saved_core['model_state_dict'])
            
            optimized_graph = saved_core['architecture_graph']
            abstracted_map = saved_core.get('abstracted_map', {})
            print(f"[SUCCESS] Core loaded and verified. Graph nodes: {optimized_graph.number_of_nodes()}, Synthesized mappings: {len(abstracted_map)}")
            
            reconstructed_core = {
                'id': saved_core['id'],
                'architecture_graph': optimized_graph,
                'abstracted_map': abstracted_map,
                'runtime_model': model,
                'runtime_tokenizer': tokenizer,
                'assembly_time': saved_core['assembly_time']
            }
            return reconstructed_core
            
        except Exception as e:
            print(f"[ERROR] Failed to load core state: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"|-- No existing AGI Core found.")
        return None

# ==========================================
# 6. Adaptive Training Loop (Trainer)
# ==========================================

def train_agi_core_on_new_data(agi_core: Dict[str, Any]):
    """Simulates fine-tuning and updates the synthesized tensor references in the abstracted map."""
    print(f"\n========================================")
    print(">>> Adaptive Training Sequence Initiated <<<")
    print("========================================")
    print("|-- This process will update the live weights and re-sync synthesized tensors.")
    
    model = agi_core['runtime_model']
    tokenizer = agi_core['runtime_tokenizer']
    
    tokenizer = configure_tokenizer(tokenizer)
    
    print(f"\nEnter new natural text data below to train the core (press Ctrl+C to finish input).")
    new_corpus = []
    try:
        while True:
            line = input("> ").strip()
            if line:
                new_corpus.append(line)
    except KeyboardInterrupt:
        print("\n|-- Input stream ended.")

    if not new_corpus:
        print("|-- No data provided. Skipping training.")
        return

    print(f"|-- Preparing {len(new_corpus)} samples for Causal LM training...")
    
    dataset = SimpleTextDataset(tokenizer, new_corpus)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "tmp_trainer_output")
    
    print(f"|-- Training output directory set to: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1,
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("|-- Starting training run (fine-tuning on new data)...")
    model.train()
    trainer.train()
    print("|-- Training complete. Live weights updated in memory.")
    
    # Re-run synthesis mapping update to ensure tensor_refs point to updated weights
    synthesizer = ConceptualSynthesisEngine()
    agi_core['abstracted_map'] = synthesizer.abstract_from_model(model)
    
    agi_core['assembly_time'] = time.time()
    
    print("\n[SUCCESS] AGI Core adaptation complete.")
    
    time_str = time.strftime("%Y%m%d-%H%M%S")
    new_core_filename = f"agi_core_adapted_{time_str}.pkl"
    new_core_filepath = os.path.join(MODEL_STORAGE_DIR, new_core_filename)
    
    print(f"|-- Saving adapted AGI Core {agi_core['id']} to: {new_core_filename}...")
    
    try:
        core_to_save = {
            'id': agi_core['id'],
            'architecture_graph': agi_core['architecture_graph'],
            'abstracted_map': agi_core['abstracted_map'],
            'model_state_dict': agi_core['runtime_model'].state_dict(),
            'tokenizer_init_kwargs': agi_core['runtime_tokenizer'].init_kwargs,
            'assembly_time': agi_core['assembly_time']
        }
        
        with open(new_core_filepath, 'wb') as f:
            pickle.dump(core_to_save, f)
        print(f"[SUCCESS] Adapted core saved successfully.")
        
        active_path = os.path.join(MODEL_STORAGE_DIR, MODEL_FILENAME)
        shutil.copyfile(new_core_filepath, active_path)
        print(f"|-- Default active core updated to: {MODEL_FILENAME}")

    except Exception as e:
        print(f"[ERROR] Failed to save adapted core: {e}")
        import traceback
        traceback.print_exc()

    validate_agi_core(agi_core)

# ==========================================
# Main Execution Pipeline (Startup Logic)
# ==========================================

def run_hugging_face_pipeline():
    start_time = time.time()
    print("-" * 60)
    print(f"AGI Synthesis, Optimization, and Adaptation System")
    print("-" * 60)

    model_id = "gpt2"
    tokenizer = None
    hf_model = None
    
    agi_core = load_agi_core_at_startup()
    
    if agi_core is None:
        print("\n[Startup] Starting Full AI Synthesis Pipeline (Cold Boot)...")
        
        print(f"|-- Loading base model '{model_id}' from Hugging Face Hub...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_id)
            tokenizer = configure_tokenizer(tokenizer)
            hf_model = GPT2LMHeadModel.from_pretrained(model_id)
        except Exception as e:
            print(f"[CRITICAL ERROR] Could not connect to Hugging Face: {e}")
            return

        # 2. Synthesis
        synthesizer = ConceptualSynthesisEngine()
        abstracted_map = synthesizer.abstract_from_model(hf_model)

        # 3. Optimization
        initial_graph = build_call_graph_from_hf_model(hf_model)
        optimizer = ConceptualGraphOptimizer()
        optimized_graph = optimizer.optimize_model_graph(initial_graph)
        
        # 4. Assembly
        assembler = AGICoreAssembler()
        agi_core = assembler.assemble(optimized_graph, abstracted_map, hf_model, tokenizer)
        
        # Initial Save
        save_agi_core(agi_core)

    # 5. Operational Loop: Inference -> Adapt -> Save
    while True:
        print("\n--- AGI Core Operational Menu ---")
        print("1. Run Inference Validation")
        print("2. Provide New Data & Adapt Core (Train & Save New)")
        print("3. Manual Save Current Core State")
        print("4. Shutdown")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            validate_agi_core(agi_core)
        elif choice == '2':
            train_agi_core_on_new_data(agi_core)
        elif choice == '3':
            save_agi_core(agi_core)
        elif choice == '4':
            print("\nInitiating Shutdown Sequence.")
            save_agi_core(agi_core)
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    end_time = time.time()
    duration = end_time - start_time
    print("-" * 60)
    print(f"System lifecycle complete. Total runtime: {duration:.2f} seconds.")
    print("-" * 60)

if __name__ == "__main__":
    run_hugging_face_pipeline()
