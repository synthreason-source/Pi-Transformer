#!/usr/bin/env python3
import sys
import time
import serial
import threading
import random
import torch

# Import your V18-RP engine components (make sure your original script is named v18_rp.py)
from v18_rp import V18RPEngine, tokenize, detokenize, PUNCT_TOKENS, COGNITIVE_TOKENS

class AutonomicSignalReader:
    """
    Reads continuous serial stream from the Arduino Uno in a background thread.
    Maintains a dynamic rolling baseline to normalize biological signals to [0.0 - 1.0].
    """
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.current_arousal = 0.5  # Neutral starting point
        self.raw_val = 0.0
        self.running = False
        
        self._min_val = 1023.0
        self._max_val = 0.0

    def start(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"[*] Connected to electrodes on {self.port}")
        except Exception as e:
            print(f"[!] Failed to connect to serial: {e}")
            print("[!] Generating without live bio-feedback (arousal locked at 0.5)")

    def _read_loop(self):
        while self.running:
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    val = float(line)
                    self.raw_val = val
                    
                    if val < self._min_val: self._min_val = val
                    if val > self._max_val: self._max_val = val
                    
                    self._min_val += 0.01 
                    self._max_val -= 0.01

                    range_span = self._max_val - self._min_val
                    if range_span > 10.0:
                        self.current_arousal = max(0.0, min(1.0, (val - self._min_val) / range_span))
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

def format_token_for_stream(token):
    """Formats cognitive tokens and punctuation for cleaner console streaming."""
    if token in PUNCT_TOKENS:
        return token
    if token.startswith("[") and token.endswith("]"):
        return " " + token.strip("[]").lower()
    return " " + token

def stream_autonomic_passage(engine: V18RPEngine, bio_reader: AutonomicSignalReader, 
                             num_sentences=4, tokens_per_sent=40, seed_text="", instruction_text=""):
    """
    Mirrors generate_passage_rp exactly, utilizing all internal V18 modules
    (iso_stacker, traces, norm caches), but injected with live hardware data.
    """
    walker = engine.walker
    lm = engine.lm
    
    # Fully engage instruction distributions if requested
    if instruction_text.strip():
        walker.instr_dist.set_instruction(instruction_text)
    elif seed_text.strip():
        walker.instr_dist.set_instruction(seed_text)

    # Clear previous run tracking
    walker._step_traces.clear()
    walker._csns_syn_norms.clear()
    walker._csns_trans_norms.clear()
    
    head_list = list(lm.heads.keys())
    if not head_list:
        print("[!] Engine vocabulary is empty. Train first.")
        return

    # Fully recreate V18 Seed mapping logic
    seed_w1 = seed_w2 = None
    seed_toks = tokenize(seed_text) if seed_text else []
    if len(seed_toks) >= 2:
        seed_w1, seed_w2 = seed_toks[-2], seed_toks[-1]
    elif len(seed_toks) == 1:
        matches = [p for p in head_list if p[1] == seed_toks[0]]
        if matches: 
            seed_w1, seed_w2 = random.choice(matches)
            
    if seed_w1 is None or (seed_w1, seed_w2) not in lm.heads:
        seed_w1, seed_w2 = random.choice(head_list)

    global_step = 0
    print("\n--- [ BIO-SYNC NEUROSYMBOLIC STREAM STARTED ] ---\n")
    
    # Print the initialization seed to console
    for t in seed_toks:
        print(format_token_for_stream(t).strip() if t == seed_toks[0] else format_token_for_stream(t), end='', flush=True)

    for sent_idx in range(num_sentences):
        # FIX: Correct explicit block assignment to prevent tuple unpacking crashes
        if sent_idx == 0:
            w1, w2 = seed_w1, seed_w2
            init_toks = [w1, w2]
        else:
            w1, w2 = random.choice(head_list)
            init_toks = []

        plan_seeds = seed_toks if seed_toks and sent_idx == 0 else [w1, w2]
        
        # fully engage the underlying Random Walk tracker
        walker.begin_sentence(seed_tokens=plan_seeds, total_tokens=tokens_per_sent)
        toks = list(init_toks)

        for step in range(tokens_per_sent):
            arousal = bio_reader.current_arousal
            
            # Autonomic shift mapping: 
            # High arousal -> high temp (chaotic/divergent), low constraint (and_weight)
            # Low arousal -> low temp (focused/logical), high constraint
            dynamic_temp = 0.8 + (arousal * 2.0)
            dynamic_and_weight = 0.5 + (arousal * 0.4)
            
            # Step the V18 engine probabilities
            cands, probs = walker.walk_probs(w1, w2, temp=dynamic_temp, and_weight=dynamic_and_weight)
            if not cands: break
            
            nxt = cands[torch.multinomial(probs, 1).item()]
            
            # Full Engine Trace Recording
            walker.record_step_trace(global_step, nxt, cands, probs, dynamic_and_weight)
            walker.push_token(nxt, tokens_per_sent)
            
            global_step += 1
            toks.append(nxt)
            
            sys.stdout.write(format_token_for_stream(nxt))
            sys.stdout.flush()
            
            time.sleep(0.05) 
            
            if nxt in PUNCT_TOKENS and len(toks) > 8: break
            w1, w2 = w2, nxt

        # Full Engine Semantic Stacker update
        sent_text = detokenize(toks)
        walker.iso_stacker.add(toks, walker.geo, sent_text)
        
        print(f"\n   [System | Arousal: {arousal:.2f} | Temp: {dynamic_temp:.2f} | Context Nodes: {len(toks)}]")
        
    print("\n\n--- [ BIO-SYNC STREAMING ENDED ] ---")

if __name__ == "__main__":
    PORT = 'COM4' # Change this if you are not using COM3
    
    reader = AutonomicSignalReader(port=PORT)
    reader.start()
    
    print("[*] Calibrating physiological baseline for 4 seconds... Please remain still.")
    time.sleep(4)
    
    try:
        engine = V18RPEngine.load("v18rp_engine.pkl")
        # You can now tweak generation limits here!
        stream_autonomic_passage(
            engine, 
            reader, 
            num_sentences=8, 
            tokens_per_sent=50, 
            seed_text="Quantum physics suggests that"
        )
    except FileNotFoundError:
        print("[!] Could not find 'v18rp_engine.pkl'. Please train the engine first.")
        
    reader.stop()
