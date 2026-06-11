
import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_len = 3 
    batch_size = 64
    epochs = 8  
    lr = 1e-3

    d_model = 96
    d_coef = 32
    d_obj = 96
    d_mem = 96

    temp_step = 0.08
    lambda_transport = 0.1
    lambda_contrastive = 0.2

def tokenize(text):
    text = text.lower()
    out = []
    cur = []
    for ch in text:
        if ch.isalnum() or ch == "'":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out

def build_vocab(tokens):
    vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(set(tokens))
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos

class TextWindowDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.samples = []
        for i in range(len(ids) - seq_len + 1):
            window = ids[i:i+seq_len]
            self.samples.append((window[:-1], window[-1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class ContentAddressedMemory:
    def __init__(self, slots, dim, device):
        self.slots = slots
        self.dim = dim
        self.device = device
        self.memory_bank = torch.zeros(slots, dim, device=device)
        
    def update(self, keys, feats):
        with torch.no_grad():
            flat_keys = keys.view(-1)
            flat_feats = feats.view(-1, feats.size(-1))
            slot_indices = flat_keys % self.slots
            for idx, feat in zip(slot_indices, flat_feats):
                self.memory_bank[idx] = 0.90 * self.memory_bank[idx] + 0.10 * feat

    def read(self, keys):
        flat_keys = keys.view(-1)
        slot_indices = flat_keys % self.slots
        retrieved = self.memory_bank[slot_indices]
        return retrieved.view(keys.size(0), -1, self.dim).mean(dim=1)

class TransportPotential(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1)
        )

    def forward(self, x):
        with torch.enable_grad():
            x_col = x.clone().requires_grad_(True)
            phi = self.net(x_col).sum()
            
            grad = torch.autograd.grad(
                outputs=phi, 
                inputs=x_col, 
                create_graph=True, 
                retain_graph=True
            )[0]
        return grad

class ChoiceGeometricTrigramNOCN(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        self.word_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        context_tokens = cfg.seq_len - 1
        
        self.coeff_encoder = nn.Sequential(
            nn.Linear(cfg.d_model * context_tokens, cfg.d_coef),
            nn.Tanh(),
            nn.Linear(cfg.d_coef, cfg.d_coef)
        )

        self.temporal_ode = nn.Sequential(
            nn.Linear((cfg.d_model * context_tokens) + cfg.d_coef, cfg.d_obj),
            nn.SiLU(),
            nn.Linear(cfg.d_obj, cfg.d_obj)
        )

        self.transport = TransportPotential(cfg.d_obj)
        self.mem_project = nn.Linear(cfg.d_mem, cfg.d_obj)

        # Barycentric Network targeting Candidate Triangles
        self.barycentric_net = nn.Sequential(
            nn.Linear(cfg.d_obj * 2, cfg.d_obj),
            nn.LayerNorm(cfg.d_obj),
            nn.SiLU(),
            nn.Linear(cfg.d_obj, 3) 
        )
        
        nn.init.orthogonal_(self.barycentric_net[3].weight)
        nn.init.zeros_(self.barycentric_net[3].bias)

        self.contrast_proj = nn.Linear(cfg.d_obj, cfg.d_obj)

    def encode_context(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.word_emb(x) + self.pos_emb(pos)[None, :]
        h = h.view(x.size(0), -1) 
        c = self.coeff_encoder(h)
        return h, c

    def temporal_dynamics(self, h, c):
        z = torch.cat([h, c], dim=-1)
        dz = self.temporal_ode(z)
        return h.mean(dim=-1, keepdim=True).expand(-1, self.cfg.d_obj) + self.cfg.temp_step * dz

    def forward(self, x, mem, choice_vertex_ids):
        h, c = self.encode_context(x)
        obj = self.temporal_dynamics(h, c)

        transported = self.transport(obj)
        mem_features = self.mem_project(mem)
        
        combined_state = torch.cat([transported, mem_features], dim=-1)

        raw_weights = self.barycentric_net(combined_state)
        mix_weights = F.softmax(raw_weights / 0.15, dim=-1) 

        # Pull representations of candidate vertices
        v1 = self.word_emb(choice_vertex_ids[:, 0]) 
        v2 = self.word_emb(choice_vertex_ids[:, 1])
        v3 = self.word_emb(choice_vertex_ids[:, 2])

        # Synthesize target vector position within candidate geometric bounds
        synthesized_point = (mix_weights[:, 0].unsqueeze(-1) * v1) + \
                            (mix_weights[:, 1].unsqueeze(-1) * v2) + \
                            (mix_weights[:, 2].unsqueeze(-1) * v3)

        # Compute cosine matching across total vocabulary structure
        synthesized_point_norm = F.normalize(synthesized_point, p=2, dim=-1)
        vocab_table_norm = F.normalize(self.word_emb.weight, p=2, dim=-1)
        
        logits = torch.matmul(synthesized_point_norm, vocab_table_norm.t()) * 16.0

        return {
            "logits": logits,
            "transported": transported,
            "mix_weights": mix_weights
        }

    def loss_fn(self, out, y):
        recon = F.cross_entropy(out["logits"], y)
        transported = out["transported"]
        ma_loss = torch.mean((torch.var(transported, dim=-1) - 1.0) ** 2)

        z = F.normalize(self.contrast_proj(transported), p=2, dim=-1)
        sim = z @ z.t()
        eye = torch.eye(sim.size(0), device=sim.device)
        contrast = ((sim - eye) ** 2).mean()

        return recon + self.cfg.lambda_transport * ma_loss + self.cfg.lambda_contrastive * contrast

def train():
    random.seed(Config.seed)
    torch.manual_seed(Config.seed)

    filename = input("Filename: ").strip()
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenize(text)
    vocab, stoi, itos = build_vocab(tokens)
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size} unique symbols.")

    token_ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    dataset = TextWindowDataset(token_ids, Config.seq_len)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True)

    model = ChoiceGeometricTrigramNOCN(vocab_size, Config).to(Config.device)
    memory = ContentAddressedMemory(slots=128, dim=Config.d_mem, device=Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(Config.epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(Config.device), y.to(Config.device)
            
            # --- THE FIX: LANDMARKS ARE CANDIDATE CHOICES, NOT HISTORY ---
            # Slot 0 is always the TRUE word. Slots 1 & 2 are random negatives.
            neg1 = torch.randint(0, vocab_size, (x.size(0), 1), device=Config.device)
            neg2 = torch.randint(0, vocab_size, (x.size(0), 1), device=Config.device)
            choice_vertices = torch.cat([y.unsqueeze(1), neg1, neg2], dim=1)
            
            # Randomly shuffle choices so slot 0 isn't always the true target
            shuffled_indices = torch.stack([torch.randperm(3, device=Config.device) for _ in range(x.size(0))], dim=0)
            choice_vertices = torch.gather(choice_vertices, 1, shuffled_indices)
            
            mem_vectors = memory.read(x)
            optimizer.zero_grad()
            
            out = model(x, mem_vectors, choice_vertex_ids=choice_vertices)
            loss = model.loss_fn(out, y)
            
            loss.backward()
            optimizer.step()

            memory.update(x, out["transported"])
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{Config.epochs:02d} | Path Loss: {total_loss/len(dataloader):.5f}")
        
    return model, memory, stoi, itos

def generate_geometric_trigrams(model, memory, stoi, itos, seeds, max_new_tokens=10):
    model.eval()
    needed = Config.seq_len - 1 
    batch_size = len(seeds)
    
    batched_ids = []
    for seed in seeds:
        tokens = tokenize(seed)
        input_ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
        if len(input_ids) < needed:
            input_ids = [stoi["<bos>"]] * (needed - len(input_ids)) + input_ids
        else:
            input_ids = input_ids[-needed:]
        batched_ids.append(input_ids)
        
    input_tensor = torch.tensor(batched_ids, dtype=torch.long, device=Config.device)
    generated_trajectories = [[] for _ in range(batch_size)]
    blending_metrics = [[] for _ in range(batch_size)]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            mem_vectors = memory.read(input_tensor)
            
            # Dynamically select 3 random candidate terms from the vocabulary
            # The model must figure out how to look at the context to pick the best coordinate lane
            choice_vertices = torch.randint(0, len(stoi), (batch_size, 3), device=Config.device)
            
            out = model(input_tensor, mem_vectors, choice_vertex_ids=choice_vertices)
            logits = out["logits"]
            mix_weights = out["mix_weights"]
        
        logits[:, stoi["<unk>"]] = -float("Inf")
        probs_batch = F.softmax(logits, dim=-1)
        
        next_tokens = []
        for b in range(batch_size):
            probs_individual = probs_batch[b]
            next_id = torch.multinomial(probs_individual, num_samples=1).item()
            
            token_string = itos[next_id]
            current_weights = mix_weights[b].tolist()
            formatted_weights = [f"{w:.3f}" for w in current_weights]
            
            v1_str = itos[choice_vertices[b, 0].item()]
            v2_str = itos[choice_vertices[b, 1].item()]
            v3_str = itos[choice_vertices[b, 2].item()]
            
            generated_trajectories[b].append(token_string)
            blending_metrics[b].append((token_string, f"Candidates({v1_str}, {v2_str}, {v3_str}): {formatted_weights}"))
            next_tokens.append(next_id)
            
        next_tokens_tensor = torch.tensor(next_tokens, dtype=torch.long, device=Config.device).unsqueeze(1)
        input_tensor = torch.cat([input_tensor[:, 1:], next_tokens_tensor], dim=1)
        
    return generated_trajectories, blending_metrics

if __name__ == "__main__":
    model, memory, stoi, itos = train()
    print("\n--- Geometric Choice Candidate Pipeline Online ---\n")
    while True:
        sample_seeds = [
            "the quick",
            "optimal transport",
            "neural networks",
            input("USER: ")
        ]
        
        outputs, metrics = generate_geometric_trigrams(model, memory, stoi, itos, sample_seeds, max_new_tokens=80)
        
        print("\n================== PARALLEL GEOMETRIC REALIZATION ==================")
        for i, seed in enumerate(sample_seeds):
            print(f"\nSTREAM {i+1} SEED Context: '{seed}'")
            print(f"DYNAMIC CHOICE LANES    : {metrics[i]}")
        print("====================================================================")
