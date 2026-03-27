#!/usr/bin/env python3
import os

# Import the engine from your original script (rename paste.txt to v18_rp.py)
from v18_rp import V18RPEngine

def train_and_save():
    print("=== V18-RP Engine Training ===")
    
    # 1. Provide some training text.
    # You can load a real text file here (e.g., a quantum physics paper or a book)
    # For this example, we use a small hardcoded corpus.

    with open("singlekb.txt", "r", encoding="utf-8") as f:
        corpus = f.read()
    
    # 2. Initialize the empty engine
    print("[*] Initializing empty V18-RP Engine...")
    engine = V18RPEngine()
    
    # 3. Train the engine (this builds the RP graphs, LSH indexes, and Nyström approximations)
    print("[*] Training on corpus...")
    engine.train(corpus)
    
    # 4. Save the compiled engine to the pickle file
    engine.save("v18rp_engine.pkl")
    print("\n[+] Success! 'v18rp_engine.pkl' has been generated.")
    print("[+] You can now run the autonomic streaming script.")

if __name__ == "__main__":
    train_and_save()
