import networkx as nx
import uuid
import pickle
import os
import time
import logging
import sys
import shutil # Added for directory cleanup

# --- Hugging Face Imports ---
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, Trainer, TrainingArguments, DataCollatorForLanguageModeling
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
# 1. Conceptual Datatype Synthesis (Abstraction)
# ==========================================

@dataclass
class OptimizedDatatype:
    """Represents the conceptual abstracted datatype."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_component: str = ""
    shape: tuple = ()
    semantic_meaning: str = "Generic"
    # Hold a reference to the actual tensor weight for training
    tensor_ref: Optional[torch.Tensor] = field(default=None, repr=False)

    def __repr__(self):
        return f"<{self.semantic_meaning} Datatype ({self.shape}) from {self.original_component}>"

class ConceptualSynthesisEngine:
    """
    Abstracts HF model components into conceptual types and maps live weights.
    """
    def abstract_from_model(self, model: torch.nn.Module) -> List[OptimizedDatatype]:
        print(f"\n|-- [SYNTHESIS] Analyzing HF Model Architecture and mapping weights...")
        abstracted_types = []

        # Iterate through all modules to find optimization targets
        for name, module in model.named_modules():
            # Handle GPT-2 Attention weight structure.
            if "attn" in name.lower():
                 c_attn_module = getattr(module, 'c_attn', None)
                 if c_attn_module is not None and hasattr(c_attn_module, 'weight'):
                     weight_t = c_attn_module.weight.data
                     dtype = OptimizedDatatype(
                            original_component=f"{name}.c_attn",
                            shape=weight_t.shape,
                            semantic_meaning="CausalAttention_QKV",
                            tensor_ref=weight_t
                        )
                     abstracted_types.append(dtype)
                 else:
                     print(f"|-- [WARN] Skipped weight mapping for Attention layer '{name}': Unexpected structure.")

            # Identify MLP layers (Feed Forward)
            elif "mlp" in name.lower():
                 c_fc_module = getattr(module, 'c_fc', None)
                 if c_fc_module is not None and hasattr(c_fc_module, 'weight'):
                     weight_t = c_fc_module.weight.data
                     dtype = OptimizedDatatype(
                            original_component=f"{name}.c_fc",
                            shape=weight_t.shape,
                            semantic_meaning="FeedForward_InputProjection",
                            tensor_ref=weight_t
                        )
                     abstracted_types.append(dtype)

        print(f"|-- [SYNTHESIS] Complete. Mapped {len(abstracted_types)} conceptual datatypes to live tensors.")
        return abstracted_types

# ==========================================
# 2. Hierarchical Graph Optimization
# ==========================================

class ConceptualGraphOptimizer:
    """
    Optimizes graph; ensures tensor references are preserved during node fusion.
    """
    def optimize_model_graph(self, call_graph: nx.DiGraph) -> nx.DiGraph:
        print("\n|-- [OPTIMIZATION] Starting Hierarchical Graph Optimization...")
        optimized_graph = call_graph.copy()
        
        # Pruning: Remove redundant Dropout layers often found in training graphs
        nodes_to_remove = [node for node in optimized_graph.nodes() if "drop" in node.lower()]
        for node in nodes_to_remove:
            optimized_graph.remove_node(node)

        # Fusion: Look for LayerNorm -> Attention -> Add pattern and fuse it
        nodes = list(optimized_graph.nodes())
        for i in range(len(nodes) - 2):
            u = nodes[i]
            v = nodes[i+1]
            w = nodes[i+2]

            # GPT-2 block structure: ln_1 -> attn -> resid_dropout
            if "ln_1" in u and "attn" in v and "resid_dropout" in w:
                fused_id = f"Fused_Attn_Block_{u.split('.')[1]}"
                print(f"|-- [FUSION] Fusing GPT-2 Block: {u} + {v} + {w} -> {fused_id}")
                
                optimized_graph.add_node(fused_id, 
                                       op='FusedGPT2Attn', 
                                       conceptual_modules=[u, v, w], 
                                       customized=True
                                       )
                
                predecessors = list(call_graph.predecessors(u))
                successors = list(call_graph.successors(w))
                for pred in predecessors: optimized_graph.add_edge(pred, fused_id)
                for succ in successors: optimized_graph.add_edge(fused_id, succ)
                
                optimized_graph.remove_nodes_from([u, v, w])
                break

        print(f"|-- [OPTIMIZATION] Graph Optimized. New node count: {optimized_graph.number_of_nodes()}")
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
    Packages the graph and references to live model/tokenizer into an AGI Core object.
    """
    def assemble(self, optimized_graph: nx.DiGraph, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
        print("\n========================================")
        print(">>> AGI Core Assembly Sequence Initiated <<<")
        print("========================================")
        
        core_id = str(uuid.uuid4())[:8]
        
        agi_core = {
            'id': core_id,
            'architecture_graph': optimized_graph,
            'runtime_model': model, # Live PyTorch model
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
        for text in texts:
            # Tokenize immediately
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding='max_length')
            self.inputs.append(torch.tensor(encodings['input_ids']))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # For CLM, labels are usually the same as inputs (shifted internally by Trainer)
        return {'input_ids': self.inputs[idx], 'labels': self.inputs[idx]}

# ==========================================
# 4. Execution Engine (Inference)
# ==========================================

class SimulationExecutionEngine:
    """
    Performs inference using the live model contained within the AGI Core.
    """
    def __init__(self, agi_core: Dict[str, Any]):
        self.model = agi_core['runtime_model']
        self.tokenizer = agi_core['runtime_tokenizer']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, max_length: int = 50) -> str:
        print(f"\n|-- [ENGINE] Executing generation using Active AGI Core...")
        
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
            
        generated_text = execution_engine.generate(test_prompt, max_length=50)
        
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
    """
    Serializes the AGI Core to disk.
    Saves the live PyTorch state_dict() embedded in the core.
    """
    print(f"\n[PHASE 5: AGI Core Persistence (Saving to Disk)]")
    if not os.path.exists(MODEL_STORAGE_DIR):
        os.makedirs(MODEL_STORAGE_DIR)
    
    filepath = os.path.join(MODEL_STORAGE_DIR, MODEL_FILENAME)
    print(f"|-- Saving Active AGI Core {agi_core['id']} state to: {filepath}...")
    
    try:
        core_to_save = {
            'id': agi_core['id'],
            'architecture_graph': agi_core['architecture_graph'],
            'model_state_dict': agi_core['runtime_model'].state_dict(),
            # Tokenizer config is saved to ensure consistency on reload
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
    """
    Attempts to load the core from disk.
    Reconstructs the live PyTorch model from the saved state_dict.
    """
    print(f"\n>>> AGI System Startup Sequence <<<")
    filepath = os.path.join(MODEL_STORAGE_DIR, MODEL_FILENAME)
    
    if os.path.exists(filepath):
        print(f"|-- Found existing AGI Core state at: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                saved_core = pickle.load(f)
            
            print(f"|-- Loading Core ID: {saved_core['id']} assembled on {time.ctime(saved_core['assembly_time'])}")
            
            # Reconstruct Runtime Objects
            model_id = "gpt2"
            # Pass saved config if available
            tokenizer_kwargs = saved_core.get('tokenizer_init_kwargs', {})
            tokenizer = GPT2Tokenizer.from_pretrained(model_id, **tokenizer_kwargs)
            
            # CRITICAL FIX: Ensure pad token is set immediately after loading tokenizer
            tokenizer = configure_tokenizer(tokenizer)
            
            model = GPT2LMHeadModel.from_pretrained(model_id)
            
            # Load the saved weights into the fresh model
            model.load_state_dict(saved_core['model_state_dict'])
            
            optimized_graph = saved_core['architecture_graph']
            print(f"[SUCCESS] Core loaded and verified. Graph nodes: {optimized_graph.number_of_nodes()}")
            
            reconstructed_core = {
                'id': saved_core['id'],
                'architecture_graph': optimized_graph,
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
    """
    Simulates fine-tuning the AGI Core's live model on new data provided by the user.
    Updated for TrainingArguments compatibility.
    """
    print(f"\n========================================")
    print(">>> Adaptive Training Sequence Initiated <<<")
    print("========================================")
    print("|-- This process will update the live weights of the AGI Core.")
    
    model = agi_core['runtime_model']
    tokenizer = agi_core['runtime_tokenizer']
    
    # Ensure tokenizer is ready for padding
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
    
    # Create Dataset
    dataset = SimpleTextDataset(tokenizer, new_corpus)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # ----------------------------------------------------------------
    # FIX: Updated TrainingArguments for broader library compatibility.
    # Removed 'logging_dir' which caused the TypeError in newer 'transformers' versions.
    # ----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./tmp_trainer_output",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1,
        report_to="none" # Disable W&B etc.
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("|-- Starting training run (fine-tuning on new data)...")
    # Unfreeze model weights for training
    model.train()
    trainer.train()
    print("|-- Training complete. Live weights updated in memory.")
    
    # Update the assembly time to reflect the new training state
    agi_core['assembly_time'] = time.time()
    
    # Cleanup temp directory
    if os.path.exists("./tmp_trainer_output"):
        shutil.rmtree("./tmp_trainer_output")
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")
        
    print("\n[SUCCESS] AGI Core adaptation complete.")
    
    # Save the adapted core to a NEW file to preserve the new state/data.
    print(f"|-- System Policy: Saving adapted core state due to weight updates.")
    
    # Create a unique filename based on timestamp for the new core
    time_str = time.strftime("%Y%m%d-%H%M%S")
    new_core_filename = f"agi_core_adapted_{time_str}.pkl"
    new_core_filepath = os.path.join(MODEL_STORAGE_DIR, new_core_filename)
    
    print(f"|-- Saving adapted AGI Core {agi_core['id']} to: {new_core_filename}...")
    
    try:
        # Prepare state dictionary for serialization
        core_to_save = {
            'id': agi_core['id'], # Keep same ID, update state
            'architecture_graph': agi_core['architecture_graph'],
            'model_state_dict': agi_core['runtime_model'].state_dict(),
            'tokenizer_init_kwargs': agi_core['runtime_tokenizer'].init_kwargs,
            'assembly_time': agi_core['assembly_time']
        }
        
        with open(new_core_filepath, 'wb') as f:
            pickle.dump(core_to_save, f)
        print(f"[SUCCESS] Adapted core saved successfully.")
        
        # Update the default 'active_agi_core.pkl' copy
        # so it loads this new version next time.
        active_path = os.path.join(MODEL_STORAGE_DIR, MODEL_FILENAME)
        shutil.copyfile(new_core_filepath, active_path)
        print(f"|-- Default active core updated to: {MODEL_FILENAME}")

    except Exception as e:
        print(f"[ERROR] Failed to save adapted core: {e}")
        import traceback
        traceback.print_exc()

    # Run validation immediately after training to see effect
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
    
    # 1. Startup: Try to load existing core
    agi_core = load_agi_core_at_startup()
    
    if agi_core is None:
        print("\n[Startup] Starting Full AI Synthesis Pipeline (Cold Boot)...")
        
        print(f"|-- Loading base model '{model_id}' from Hugging Face Hub...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_id)
            # Ensure pad token is set
            tokenizer = configure_tokenizer(tokenizer)
            hf_model = GPT2LMHeadModel.from_pretrained(model_id)
        except Exception as e:
            print(f"[CRITICAL ERROR] Could not connect to Hugging Face: {e}")
            return

        # 2. Synthesis
        synthesizer = ConceptualSynthesisEngine()
        _ = synthesizer.abstract_from_model(hf_model)

        # 3. Optimization
        initial_graph = build_call_graph_from_hf_model(hf_model)
        optimizer = ConceptualGraphOptimizer()
        optimized_graph = optimizer.optimize_model_graph(initial_graph)
        
        # 4. Assembly
        assembler = AGICoreAssembler()
        agi_core = assembler.assemble(optimized_graph, hf_model, tokenizer)
        
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
            # Auto-save on shutdown
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
    # To run this simulation, ensure libraries installed:
    # pip install transformers torch networkx
    try:
        run_hugging_face_pipeline()
    except ImportError:
        print("\n[Error] This simulation requires the 'transformers', 'torch', and 'networkx' libraries.")
        print("Please install them using: pip install transformers torch networkx")
    except KeyboardInterrupt:
        print("\n[System] Pipeline interrupted by user.")

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
    
    # 1. Startup: Try to load existing core
    agi_core = load_agi_core_at_startup()
    
    if agi_core is None:
        print("\n[Startup] Starting Full AI Synthesis Pipeline (Cold Boot)...")
        
        print(f"|-- Loading base model '{model_id}' from Hugging Face Hub...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_id)
            
            # ----------------------------------------------------------------
            # CRITICAL FIX: Assign EOS token as PAD token.
            # GPT-2 defaults to no pad token, causing training batching to fail.
            # ----------------------------------------------------------------
            tokenizer.pad_token = tokenizer.eos_token
            print(f"|-- Tokenizer loaded and configured with pad_token: {tokenizer.pad_token}")

            hf_model = GPT2LMHeadModel.from_pretrained(model_id)
        except Exception as e:
            print(f"[CRITICAL ERROR] Could not connect to Hugging Face: {e}")
            return

        # 2. Synthesis
        synthesizer = ConceptualSynthesisEngine()
        _ = synthesizer.abstract_from_model(hf_model)

        # 3. Optimization
        initial_graph = build_call_graph_from_hf_model(hf_model)
        optimizer = ConceptualGraphOptimizer()
        optimized_graph = optimizer.optimize_model_graph(initial_graph)
        
        # 4. Assembly
        assembler = AGICoreAssembler()
        agi_core = assembler.assemble(optimized_graph, hf_model, tokenizer)
        
        # Initial Save
        save_agi_core(agi_core)

    # 5. Operational Loop: Inference -> Adapt -> Save
    while True:
        print("\n--- AGI Core Operational Menu ---")
        print("1. Run Inference Validation")
        print("2. Provide New Data & Adapt Core (Train)")
        print("3. Save Current Core State to Disk")
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
            # Auto-save on shutdown
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
    # To run this simulation, ensure libraries installed:
    # pip install transformers torch networkx
    try:
        run_hugging_face_pipeline()
    except ImportError:
        print("\n[Error] This simulation requires the 'transformers', 'torch', and 'networkx' libraries.")
        print("Please install them using: pip install transformers torch networkx")
    except KeyboardInterrupt:
        print("\n[System] Pipeline interrupted by user.")
