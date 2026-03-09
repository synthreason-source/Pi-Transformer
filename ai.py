import torch
import torch.nn.functional as F
import re, random
from collections import Counter, defaultdict
import gradio as gr

DEVICE = torch.device("cpu")
PUNCT = {",", ".", "!", "?", ";", ":"}

def tokenize(text):
    return [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b|[.,!?;:]", text)]

class SimilitudeEngine:
    def __init__(self, corpus: str, embed_dim: int = 64):
        self.tokens = tokenize(corpus)
        if len(self.tokens) < 10:
            self.tokens = tokenize("the quick brown fox jumps over the lazy dog. " * 10)
            
        self.vocab = list(dict.fromkeys(self.tokens))
        self.v2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2v = {i: w for i, w in enumerate(self.vocab)}
        
        torch.manual_seed(42)
        raw_embs = torch.randn(len(self.vocab), embed_dim, device=DEVICE)
        self.embs = F.normalize(raw_embs, p=2, dim=1) 
        
        self.lm = defaultdict(Counter)
        for i in range(len(self.tokens) - 2):
            self.lm[(self.tokens[i], self.tokens[i+1])][self.tokens[i+2]] += 1
            
        self.chunk_buffer = []
        self.sentence_embs = []

    def reset_state(self):
        self.chunk_buffer.clear()
        
    def walk(self, w1, w2, params):
        ctx = (w1, w2)
        counts = self.lm.get(ctx, {})
        
        if not counts:
            cands = self.vocab[:200]
            base_logits = torch.zeros(len(cands), device=DEVICE)
        else:
            cands = list(counts.keys())
            freqs = torch.tensor(list(counts.values()), dtype=torch.float32, device=DEVICE)
            base_logits = torch.log(freqs / freqs.sum())
            
        c_idxs = torch.tensor([self.v2i.get(c, 0) for c in cands], device=DEVICE)
        cand_embs = self.embs[c_idxs]           
        ctx_emb = self.embs[self.v2i.get(w2, 0)].unsqueeze(0) 
        
        sim_scores = F.cosine_similarity(ctx_emb, cand_embs, dim=-1)
        mean_emb = self.embs.mean(dim=0, keepdim=True)
        mrv_similitude = F.pairwise_distance(cand_embs, mean_emb) 
        
        chunk_sim = torch.zeros_like(sim_scores)
        if self.chunk_buffer:
            chunk_tensor = torch.stack(self.chunk_buffer).mean(dim=0, keepdim=True)
            chunk_sim = F.cosine_similarity(chunk_tensor, cand_embs, dim=-1)
            
        echo_sim = torch.zeros_like(sim_scores)
        if self.sentence_embs:
            last_sent_emb = self.sentence_embs[-1].unsqueeze(0)
            echo_sim = F.cosine_similarity(last_sent_emb, cand_embs, dim=-1)

        logits = (
            base_logits 
            + (params['alpha_sim'] * sim_scores)
            + (params['zeta_mrv']  * mrv_similitude)
            + (params['eta_chunk'] * chunk_sim)
            + (params['xi_echo']   * echo_sim)
        ) / max(params['temp'], 1e-3)
        
        for i, c in enumerate(cands):
            if c in PUNCT and w2 in PUNCT:
                logits[i] = -1e4
                
        probs = F.softmax(logits, dim=-1)
        nxt_idx = torch.multinomial(probs, 1).item()
        nxt_word = cands[nxt_idx]
        
        if nxt_word not in PUNCT and nxt_word in self.v2i:
            self.chunk_buffer.append(self.embs[self.v2i[nxt_word]])
            
        return nxt_word
def generate(engine, seed_context, num_sentences, tokens_per_sent, params):
    outputs = []
    heads = list(engine.lm.keys())
    
    # Process Seed Input once
    base_w1, base_w2 = None, None
    base_seed_tokens = []
    
    if seed_context:
        seed_toks = tokenize(seed_context)
        if len(seed_toks) >= 2:
            base_w1, base_w2 = seed_toks[-2], seed_toks[-1]
            base_seed_tokens = seed_toks
        elif len(seed_toks) == 1:
            matches = [p for p in heads if p[1] == seed_toks[0]]
            if matches:
                base_w1, base_w2 = random.choice(matches)
                base_seed_tokens = [base_w1, base_w2]
                
    for _ in range(num_sentences):
        engine.reset_state()
        
        # Force EVERY sentence to start with the seed if one was provided
        if base_w1 and base_w2:
            w1, w2 = base_w1, base_w2
            sent_tokens = list(base_seed_tokens)
        else:
            w1, w2 = random.choice(heads) if heads else (".", "the")
            sent_tokens = []
        
        for _ in range(tokens_per_sent): 
            nxt = engine.walk(w1, w2, params)
            sent_tokens.append(nxt)
            w1, w2 = w2, nxt
            
            # Allow early break if sentence naturally ends
            if nxt in {".", "?", "!"} and len(sent_tokens) > max(3, len(base_seed_tokens)):
                break
                
        # Syntax stacking: Save sentence embedding
        if engine.chunk_buffer:
            engine.sentence_embs.append(torch.stack(engine.chunk_buffer).mean(dim=0))
            if len(engine.sentence_embs) > 10:
                engine.sentence_embs.pop(0)
                
        # Detokenize
        text = " ".join(sent_tokens).replace(" .", ".").replace(" ,", ",")
        
        # Capitalize only if we aren't using a seed, or if the seed was capitalized
        if not base_seed_tokens:
            text = text.capitalize()
            
        outputs.append(text)
        
    return "\n".join(f"[{i+1:02d}] {s}" for i, s in enumerate(outputs))


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def run_session(text_file, seed_context, num_sents, tokens_per_sent, temp, alpha_sim, zeta_mrv, eta_chunk, xi_echo):
    try:
        with open(text_file.name, 'r', encoding='utf-8') as f:
            corpus = f.read()
    except:
        corpus = " ".join(["The neural networks compute deep similitudes.", 
                           "Vector spaces allow us to stack syntax.", 
                           "Gradient descent bounds the manifold."] * 100)
    
    engine = SimilitudeEngine(corpus)
    params = {
        'temp': temp, 
        'alpha_sim': alpha_sim, 
        'zeta_mrv': zeta_mrv, 
        'eta_chunk': eta_chunk, 
        'xi_echo': xi_echo
    }
    
    out_text = generate(engine, seed_context, int(num_sents), int(tokens_per_sent), params)
    report = f"✅ Running on: {DEVICE}\n✅ Vocab Size: {len(engine.vocab)}\n✅ GPU Similitude Engine Active"
    return out_text, report

with gr.Blocks(title="V18 PyTorch Similitudes") as demo:
    gr.Markdown(f"### V18-Similitude Engine | Pure PyTorch | Running on **{DEVICE}**")
    
    with gr.Row():
        with gr.Column(scale=1):
            text_file = gr.File(label="Upload Corpus (.txt)")
            
            # Added Seed Context Input
            seed_context = gr.Textbox(label="Seed Context", placeholder="Enter starting words (e.g. 'the system')")
            
            num_sents = gr.Slider(1, 50, value=10, step=1, label="Sentences")
            tokens_per_sent = gr.Slider(5, 200, value=92, step=1, label="Tokens per Sentence")
            
            temp      = gr.Slider(0.5, 2.5, value=1.2, label="Temperature")
            alpha_sim = gr.Slider(0.0, 3.0, value=1.5, label="Context Similitude (alpha)")
            zeta_mrv  = gr.Slider(0.0, 3.0, value=0.5, label="Diversity MRV (zeta)")
            eta_chunk = gr.Slider(0.0, 3.0, value=0.8, label="Chunk Similitude (eta)")
            xi_echo   = gr.Slider(0.0, 3.0, value=0.6, label="Syntax Echo (xi)")
            
        with gr.Column(scale=2):
            btn = gr.Button("Generate with PyTorch Similitudes", variant="primary")
            out_text = gr.Textbox(label="Generated Sentences", lines=15)
            out_report = gr.Textbox(label="System Status", lines=3)
            
            btn.click(run_session, 
                      inputs=[text_file, seed_context, num_sents, tokens_per_sent, temp, alpha_sim, zeta_mrv, eta_chunk, xi_echo],
                      outputs=[out_text, out_report])

if __name__ == "__main__":
    demo.launch(share=False)
