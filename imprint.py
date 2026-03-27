#!/usr/bin/env python3
import os
import time

# Import the engine from your original script
from v18_rp import V18RPEngine
# Import the bio-signal reader to capture the autonomic imprint
from neural import AutonomicSignalReader

def train_and_save():
    print("=== V18-RP Engine Training ===")
    
    # --- AUTONOMIC IMPRINT CAPTURE ---
    print("[*] Establishing autonomic connection for user mind imprint...")
    # Ensure this port matches your Arduino's active COM port
    bio_reader = AutonomicSignalReader(port='COM4', baudrate=115200)
    bio_reader.start()
    
    print("[*] Calibrating physiological kernel for 5 seconds... Concentrate on the corpus material.")
    time.sleep(5)
    
    # Extract the user's mind state baseline
    user_mind_kernel = {
        "base_arousal": bio_reader.current_arousal,
        "calibrated_min": bio_reader._min_val,
        "calibrated_max": bio_reader._max_val
    }
    
    print(f"[*] Autonomic imprint captured (Base Arousal: {user_mind_kernel['base_arousal']:.3f})")
    bio_reader.stop()
    # ---------------------------------

    # 1. Provide some training text.
    with open("xab", "r", encoding="utf-8") as f:
        corpus = f.read()

    # 2. Initialize the empty engine
    print("[*] Initializing empty V18-RP Engine...")
    engine = V18RPEngine()
    
    # Inject the biometric kernel directly into the engine before training
    # This ensures the neurosymbolic graph is forever tied to this physical baseline
    engine.autonomic_kernel = user_mind_kernel

    # 3. Train the engine (this builds the RP graphs, LSH indexes, and Nyström approximations)
    print("[*] Training on corpus...")
    engine.train(corpus)

    # 4. Save the compiled engine to the pickle file
    engine.save("v18rp_engine.pkl")
    print("\n[+] Success! 'v18rp_engine.pkl' has been generated with your autonomic imprint embedded.")
    print("[+] You can now run the autonomic streaming script.")

if __name__ == "__main__":
    train_and_save()
