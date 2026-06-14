import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class StableGPT2:

    def __init__(self, model_name="gpt2"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INIT] Loading {model_name} on {self.device}")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[INIT] Model ready\n")

    # -----------------------------
    # INPUT CLASSIFICATION
    # -----------------------------
    def classify(self, text):

        text = text.strip()

        if text.endswith("?"):
            return "question"

        if len(text.split()) <= 3:
            return "fragment"

        return "statement"

    # -----------------------------
    # CORE GENERATION (SAFE)
    # -----------------------------
    def generate(self, prompt, max_new_tokens=20):

        print("\n[DEBUG PROMPT]")
        print(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.55,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.25,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        result = decoded[len(prompt):].strip()

        print("\n[RAW OUTPUT]")
        print(result)

        # HARD SAFETY: remove weird leading punctuation ONLY
        result = result.lstrip(".,;:- ")

        # HARD STOP: keep only first sentence
        if "." in result:
            result = result.split(".")[0] + "."

        return result

    # -----------------------------
    # FRAGMENT HANDLER (FIXED)
    # -----------------------------
    def handle_fragment(self, text):

        prompt = f"""
Explain in one simple sentence.

Term: {text}

Answer:
"""

        return self.generate(prompt, max_new_tokens=200)

    # -----------------------------
    # QUESTION HANDLER
    # -----------------------------
    def handle_question(self, text):

        prompt = f"""
Question: {text}

Answer in one clear sentence:
"""

        return self.generate(prompt, max_new_tokens=250)

    # -----------------------------
    # STATEMENT HANDLER
    # -----------------------------
    def handle_statement(self, text):

        prompt = f"""
Statement: {text}

Explain briefly in one sentence:
"""

        return self.generate(prompt, max_new_tokens=250)

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def run(self, text):

        print("\n================================")
        print(f"INPUT: {text}")
        print("================================")

        input_type = self.classify(text)

        print(f"[TYPE] {input_type}")

        if input_type == "fragment":

            print("[MODE] fragment → explanation")
            return self.handle_fragment(text)

        elif input_type == "question":

            print("[MODE] question → answer")
            return self.handle_question(text)

        else:

            print("[MODE] statement → explanation")
            return self.handle_statement(text)


# -----------------------------
# CLI LOOP
# -----------------------------
def main():

    agent = StableGPT2("gpt2")

    print("Stable GPT-2 System")
    print("Type 'exit' to quit\n")

    while True:

        user_input = input("USER: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            result = agent.run(user_input)

            print("\n================ FINAL =================")
            print(result)
            print("========================================\n")

        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
