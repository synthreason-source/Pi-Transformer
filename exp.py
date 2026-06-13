import torch
from itertools import cycle
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def cot_generation_loop(prompt, model_name="gpt2", max_reasoning_tokens=150, trigger_patience=20):
    # 1. Initialize components
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    target_triggers = ["because", "therefore"]
    trigger_cycle = cycle(target_triggers)
    current_target = next(trigger_cycle)  
    
    trigger_count = 0  
    tokens_since_last_trigger = 0 # Track to prevent infinite loops
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"--- Starting Sequential Reasoning Phase ---")
    print(f"Trigger sequence loop initialized: {' -> '.join(target_triggers)} -> (repeats)\n")
    
    # 2. Continuous generation loop
    for _ in range(max_reasoning_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Using basic sampling instead of rigid greedy decoding to boost natural variety
            # Filter logits slightly to avoid complete gibberish
            filtered_logits = torch.topk(next_token_logits, k=50).values
            # For this demo, we'll stick to a slightly relaxed greedy/top-k approach:
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the new token
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        tokens_since_last_trigger += 1
        
        # Decode and clean the newest token
        decoded_token = tokenizer.decode(next_token[0]).strip().lower()
        print(tokenizer.decode(next_token[0]), end="", flush=True)
        
        # 3. Check for target trigger
        if current_target in decoded_token:
            trigger_count += 1
            print(f"\n\n[Cognitive Stage {trigger_count} Cleared: '{current_target}']")
            current_target = next(trigger_cycle)
            print(f"[Next required trigger: '{current_target}']\n")
            tokens_since_last_trigger = 0
            
        # 4. DYNAMIC INJECTION: If the model is looping/stuck, force the trigger!
        elif tokens_since_last_trigger >= trigger_patience:
            trigger_count += 1
            print(f"\n\n[Patience Exceeded. Forcing Trigger Stage {trigger_count}: '{current_target}']")
            
            # Encode and append the forced trigger token
            forced_ids = tokenizer.encode(" " + current_target, return_tensors="pt")
            input_ids = torch.cat([input_ids, forced_ids], dim=-1)
            
            current_target = next(trigger_cycle)
            print(f"[Next required trigger: '{current_target}']\n")
            tokens_since_last_trigger = 0
            
        if trigger_count == 4: # Graceful breakout after a few loops
            print("\n[Target loop count reached. Breaking reasoning phase.]")
            break
                    
    # 5. Inject the transition token & generate final answer
    print("\n--- Injecting Transition Token & Generating Final Answer ---")
    
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 50256
    bos_token_id = torch.tensor([[bos_id]])
    input_ids = torch.cat([input_ids, bos_token_id], dim=-1)
    
    # Use sampling for the final generation to ensure a varied, clean answer
    final_output = model.generate(
        input_ids, 
        max_new_tokens=50,  
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(final_output[0], skip_special_tokens=False)

# Example Usage
prompt = "The sky is blue during the day but turns red at sunset."
final_text = cot_generation_loop(prompt)
print("\n--- Final Output ---")
print(final_text)
