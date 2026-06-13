import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def cot_generation_loop(prompt, model_name="gpt2", max_reasoning_tokens=150):
    # 1. Initialize standard GPT-2 components
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Define an ordered sequence of cognitive triggers we want the model to hit in a loop
    target_triggers = ["because", "therefore"]
    trigger_index = 0  # Tracks our progress through the target triggers
    loop_count = 1     # Tracks how many full cycles we've completed
    
    # Encode the initial prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"--- Starting Sequential Reasoning Phase ---")
    print(f"Target path: {' -> '.join(target_triggers)} (Looping)\n")
    
    # 2. Continuous generation loop (Simulating the CoT stream)
    for _ in range(max_reasoning_tokens):
        # Forward pass through GPT-2
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Simple greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the new token to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Decode the newest token to check it
        decoded_token = tokenizer.decode(next_token[0]).strip().lower()
        print(decoded_token, end=" ", flush=True)
        
        # 3. Check for the CURRENT active cognitive trigger
        current_target = target_triggers[trigger_index]
        
        if current_target in decoded_token:
            print(f"\n\n[Match Found: '{current_target}' in Cycle {loop_count}]")
            
            # Advance index using modulo to create an infinite loop structure
            trigger_index = (trigger_index + 1) % len(target_triggers)
            
            # If the index resets to 0, we completed one full loop of all triggers
            if trigger_index == 0:
                print(f"[Finished Trigger Loop Cycle {loop_count}! Starting next cycle...]\n")
                loop_count += 1
                    
    # 4. Inject the transition token (forcing the <BOS> shift)
    print("\n--- Injecting Transition Token & Generating Final Answer ---")
    
    # Fallback to standard token ID if tokenizer.bos_token_id is None
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 50256
    bos_token_id = torch.tensor([[bos_id]])
    input_ids = torch.cat([input_ids, bos_token_id], dim=-1)
    
    # Generate the final conclusion based on the reasoning path
    final_output = model.generate(
        input_ids, 
        max_new_tokens=40, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(final_output[0], skip_special_tokens=False)

# Example Usage
prompt = "The sky is blue during the day but turns red at sunset."
final_text = cot_generation_loop(prompt)
print("\n--- Final Output ---")
print(final_text)
