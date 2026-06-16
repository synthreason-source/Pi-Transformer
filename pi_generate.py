import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. VOCAB
# =========================
class Vocab:
    def __init__(self, words):
        self.word2idx = {}
        self.idx2word = []

        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)

    def __len__(self):
        return len(self.idx2word)

    def encode(self, words):
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def decode(self, ids):
        return " ".join(self.idx2word[i] for i in ids)


# =========================
# 2. N-GRAM DATASET
# =========================
class NGramDataset(Dataset):
    def __init__(self, file_path, n=5):
        text = open(file_path, "r", encoding="utf-8").read().lower()
        words = text.split()

        self.vocab = Vocab(words)
        self.n = n

        ids = self.vocab.encode(words)

        if len(ids) < n + 1:
            raise ValueError("Corpus too small for given n")

        self.data = [
            (ids[i:i+n], ids[i+n])
            for i in range(len(ids) - n)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return x, y


# =========================
# 3. SUPER PROB MINI-GENERATORS
# =========================
class SuperProbMiniGenerators:
    """
    Creates multiple mini-generators (small models) that run in parallel
    and vote on the final output probability distribution.
    
    How it works:
    1. Create N mini-models with different random initializations
    2. Each mini-generator produces its own probability distribution
    3. Combine probabilities via weighted averaging (ensemble)
    4. The "super probability" is the combined distribution
    
    This creates a more robust probability surface by averaging
    multiple perspectives on the same context.
    """
    
    def __init__(self, vocab_size, n=5, base_d=128, num_mini_generators=5):
        self.vocab_size = vocab_size
        self.n = n
        self.base_d = base_d
        self.num_mini_generators = num_mini_generators
        
        # Create mini-generators with different initializations
        self.mini_generators = []
        for i in range(num_mini_generators):
            mini = NGramModel(vocab_size, n, d=base_d)
            # Different random init for each
            self._random_init(mini, seed=i)
            self.mini_generators.append(mini)
        
        # Weights for each mini-generator (can be learned or fixed)
        self.generator_weights = torch.ones(num_mini_generators) / num_mini_generators
        
        # Temperature for super probability smoothing
        self.super_temperature = 1.0
    
    def _random_init(self, model, seed):
        """Random initialization with different seed for diversity."""
        torch.manual_seed(seed)
        for param in model.parameters():
            param.data = torch.randn_like(param) * 0.1
    
    def set_generator_weight(self, generator_idx, weight):
        """Set weight for a specific mini-generator."""
        self.generator_weights[generator_idx] = weight
    
    def set_super_temperature(self, temp):
        """Control sharpness of super probability (lower = sharper)."""
        self.super_temperature = temp
    
    @torch.no_grad()
    def get_super_probs(self, context_tensor, device):
        """
        Get super probability from all mini-generators.
        
        Args:
            context_tensor: (1, n) context tensor
            device: torch device
        
        Returns:
            super_probs: Combined probability distribution (vocab_size,)
            all_probs: List of individual probs from each mini-generator
        """
        all_probs = []
        all_logits = []
        
        # Run each mini-generator
        for i, mini in enumerate(self.mini_generators):
            mini.eval()
            logits = mini(context_tensor)
            all_logits.append(logits)
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[0])  # (vocab_size,)
        
        # Normalize weights
        weights = F.softmax(self.generator_weights, dim=-1).to(device)
        
        # Weighted average of probabilities (super probability)
        super_probs = torch.zeros(self.vocab_size, device=device)
        for i, probs in enumerate(all_probs):
            super_probs += weights[i] * probs
        
        # Apply temperature smoothing
        super_probs = super_probs / self.super_temperature
        super_probs = F.softmax(super_probs, dim=-1)
        
        return super_probs, all_probs
    
    def get_super_logits(self, context_tensor, device):
        """
        Get super logits (before softmax) from ensemble.
        Useful for combining with other logits processors.
        """
        all_logits = []
        
        for mini in self.mini_generators:
            mini.eval()
            logits = mini(context_tensor)
            all_logits.append(logits[0])
        
        # Weighted average of logits
        weights = F.softmax(self.generator_weights, dim=-1).to(device)
        super_logits = torch.zeros(self.vocab_size, device=device)
        
        for i, logits in enumerate(all_logits):
            super_logits += weights[i] * logits
        
        return super_logits
    
    @torch.no_grad()
    def get_top_k_consensus(self, context_tensor, device, k=5):
        """
        Get top-k tokens that multiple mini-generators agree on.
        
        Returns:
            consensus_tokens: List of (token_id, agreement_score)
        """
        all_probs = []
        
        for mini in self.mini_generators:
            mini.eval()
            logits = mini(context_tensor)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[0])
        
        # Count how many generators agree on top-k
        token_agreement = torch.zeros(self.vocab_size, device=device)
        
        for probs in all_probs:
            top_k_ids = probs.topk(k).indices
            for tid in top_k_ids:
                token_agreement[tid] += 1
        
        # Normalize to agreement score (0-1)
        token_agreement = token_agreement / len(self.mini_generators)
        
        # Get tokens with high agreement
        consensus = []
        for tid, score in enumerate(token_agreement):
            if score >= 0.5:  # At least 50% agreement
                consensus.append((tid, score.item()))
        
        consensus.sort(key=lambda x: x[1], reverse=True)
        return consensus[:k]
    
    def status(self, super_probs):
        """Return status string."""
        top_id = super_probs.max(dim=-1).indices.item()
        top_prob = super_probs.max(dim=-1).values.item()
        
        # Check agreement across mini-generators
        agreement_scores = self.get_top_k_consensus(
            torch.arange(1, self.n + 1, dtype=torch.long).unsqueeze(0), 
            super_probs.device, 
            k=1
        )
        
        if agreement_scores:
            agreement = agreement_scores[0][1]
            return f"super_prob: top={top_id} prob={top_prob:.3f} agreement={agreement:.2f}"
        else:
            return f"super_prob: top={top_id} prob={top_prob:.3f}"
    
    def train_mini_generators(self, loader, opt_base, device, epochs=1):
        """Train all mini-generators on the dataset."""
        for epoch in range(epochs):
            total_loss = 0
            
            for step, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                
                # Average loss across all mini-generators
                total_logits_loss = 0
                for mini in self.mini_generators:
                    mini.train()
                    logits = mini(x)
                    loss = F.cross_entropy(logits, y)
                    total_logits_loss += loss
                
                avg_loss = total_logits_loss / len(self.mini_generators)
                
                # Update all mini-generators
                for mini in self.mini_generators:
                    opt_base.zero_grad()
                    avg_loss.backward()
                    opt_base.step()
                
                total_loss += avg_loss.item()
                
                if step % 50 == 0:
                    print(f"  super_gen step {step} | loss {avg_loss.item():.4f}")
            
            print(f"  Epoch {epoch} | super_gen loss {total_loss / len(loader):.4f}")


# =========================
# 4. N-GRAM MODEL
# =========================
class NGramModel(nn.Module):
    def __init__(self, vocab_size, n=5, d=128):
        super().__init__()
        self.n = n

        self.emb = nn.Embedding(vocab_size, d)
        self.fc1 = nn.Linear(n * d, 256)
        self.fc2 = nn.Linear(256, vocab_size)

    def forward(self, x):
        e = self.emb(x)                 # (B, n, d)
        e = e.reshape(x.size(0), -1)    # (B, n*d)
        h = F.relu(self.fc1(e))
        return self.fc2(h)


# =========================
# 5. TRAIN LOOP
# =========================
def train(model, loader, opt, device):
    model.train()
    total_loss = 0

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        ce_loss = F.cross_entropy(logits, y, reduction='none').mean()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        loss = ce_loss - 0.001 * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"  step {step} | loss {loss.item():.4f}")

    return total_loss / len(loader)


# =========================
# 6. LOGITS CONSTRAINT ENGINE
# =========================
class IsomorphismLogitsProcessor:
    def __init__(self, target_mass=0.5):
        self.target_mass = target_mass

    def __call__(self, logits):
        logits = logits.clone()
        probs = F.softmax(logits, dim=-1)
        half = logits.size(-1) // 2
        mass = probs[:, :half].sum(dim=-1, keepdim=True)
        diff = mass - self.target_mass

        if diff.abs().max() > 1e-3:
            logits[:, :half] -= diff * 1.5

        return logits


# =========================
# 7. BENCHMARK LOGITS PROCESSOR
# =========================
class BenchmarkLogitsProcessor:
    def __init__(self, token_loss_map, alpha=0.1, max_boost=5.0):
        self.token_loss_map = token_loss_map
        self.alpha = alpha
        self.max_boost = max_boost

    def update(self, token_loss_map):
        self.token_loss_map = token_loss_map

    def __call__(self, logits):
        if not self.token_loss_map:
            return logits
        logits = logits.clone()
        for vocab_id, loss in self.token_loss_map.items():
            boost = min(loss * self.alpha, self.max_boost)
            logits[:, vocab_id] += boost
        return logits


# =========================
# 8. STREAK TRACKER
# =========================
class StreakTracker:
    def __init__(self, threshold=0.15, alpha=0.018, max_streak=8):
        self.threshold = threshold
        self.alpha = alpha
        self.max_streak = max_streak
        self.streak = 0

    def reset(self):
        self.streak = 0

    def __call__(self, logits):
        logits = logits.clone()
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1).values.item()

        if top_prob >= self.threshold:
            self.streak = min(self.streak + 1, self.max_streak)
        else:
            self.streak = 0

        if self.streak > 0:
            multiplier = 1.0 + self.streak * self.alpha
            logits = logits * multiplier

        return logits

    @property
    def status(self):
        bar = "█" * self.streak + "░" * (self.max_streak - self.streak)
        return f"streak [{bar}] {self.streak}/{self.max_streak}"


# =========================
# 9. TEST BENCHMARK
# =========================
class TestBenchmark:
    def __init__(self, qa_pairs, vocab, n):
        self.qa_pairs = qa_pairs
        self.vocab = vocab
        self.n = n

    @torch.no_grad()
    def evaluate(self, model, device, epoch):
        model.eval()
        print(f"\n── Benchmark after epoch {epoch} ──")

        all_token_losses = []
        token_loss_map = {}

        for q_str, a_str in self.qa_pairs:
            q_ids = self.vocab.encode(re.sub(r"[^\w\s]", "", q_str.lower()).split())
            if len(q_ids) < self.n:
                q_ids = [0] * (self.n - len(q_ids)) + q_ids

            a_ids = self.vocab.encode(re.sub(r"[^\w\s]", "", a_str.lower()).split())
            if not a_ids:
                continue

            context = q_ids[-self.n:]
            token_losses = []

            for target_id in a_ids:
                x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(x)
                target = torch.tensor([target_id], device=device)
                loss = F.cross_entropy(logits, target).item()
                token_losses.append(loss)

                token_loss_map[target_id] = token_loss_map.get(target_id, 0.0) + loss

                context = context[1:] + [target_id]

            cumsum = 0.0
            tokens = re.sub(r"[^\w\s]", "", a_str.lower()).split()
            print(f"  Q: {q_str!r}")
            for i, (tok, l) in enumerate(zip(tokens, token_losses)):
                cumsum += l
                known = tok in self.vocab.word2idx
                flag = "" if known else " [OOV]"
                print(f"    [{i+1:3d}] {tok:<20s} loss={l:6.3f}  cumsum={cumsum:8.3f}{flag}")

            all_token_losses.extend(token_losses)

        if all_token_losses:
            global_cumsum = 0.0
            print(f"\n  Global cumsum stream ({len(all_token_losses)} tokens):")
            for i, l in enumerate(all_token_losses):
                global_cumsum += l
                if (i + 1) % 10 == 0 or i == len(all_token_losses) - 1:
                    pct = (i + 1) / len(all_token_losses)
                    filled = int(pct * 30)
                    bar = "█" * filled + "░" * (30 - filled)
                    print(f"\r  [{bar}] {i+1}/{len(all_token_losses)} cumsum={global_cumsum:.3f}", end="", flush=True)
            print()
            avg = global_cumsum / len(all_token_losses)
            print(f"  avg token loss = {avg:.4f} | total cumsum = {global_cumsum:.3f}")

        print("────────────────────────────────")
        model.train()
        return all_token_losses, token_loss_map


# =========================
# 10. GENERATION WITH SUPER PROB MINI-GENERATORS
# =========================
@torch.no_grad()
def generate(model, vocab, prompt, device, iso_processor, bench_processor, streak_tracker, 
             super_gen=None, use_super_probs=False, max_len=300):
    model.eval()
    streak_tracker.reset()

    words = re.sub(r"[^\w\s]", "", prompt.lower()).split()
    ids = vocab.encode(words)

    if len(ids) < model.n:
        ids = [0] * (model.n - len(ids)) + ids

    for step in range(max_len):
        context = ids[-model.n:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        if use_super_probs and super_gen is not None:
            # Use super probability from mini-generators instead of base model
            super_probs, all_probs = super_gen.get_super_probs(x, device)
            
            # Show super prob status
            if step % 50 == 0:
                print(f"  {super_gen.status(super_probs)}")
            
            # Apply other processors to super_probs (convert back to logits)
            super_logits = super_probs.log()
            super_logits = iso_processor(super_logits.unsqueeze(0))
            super_logits = bench_processor(super_logits)
            super_logits = streak_tracker(super_logits)
            
            # Convert back to probs and sample
            probs = F.softmax(super_logits, dim=-1)[0]
            next_id = torch.multinomial(probs, 1).item()
        else:
            # Normal generation from base model
            logits = model(x)
            logits = iso_processor(logits)
            logits = bench_processor(logits)
            logits = streak_tracker(logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        
        ids.append(next_id)

    return vocab.decode(ids)


# =========================
# 11. MAIN
# =========================
def main(txt_path="corpus.txt", n=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NGramDataset(txt_path, n=n)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate)

    model = NGramModel(len(dataset.vocab), n=n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    qa_pairs = [
        ("what is the meaning of", "life is a journey full of purpose"),
        ("the quick brown fox",    "jumps over the lazy dog"),
        ("to be or not to",        "be that is the question"),
    ]

    benchmark     = TestBenchmark(qa_pairs, vocab=dataset.vocab, n=n)
    bench_processor  = BenchmarkLogitsProcessor(token_loss_map={}, alpha=0.000000008, max_boost=1.0)
    iso_processor    = IsomorphismLogitsProcessor(target_mass=0.000000008)
    streak_tracker   = StreakTracker(threshold=0.000000008, alpha=0.000000008, max_streak=118)
    
    # Create super probability mini-generators
    super_gen = SuperProbMiniGenerators(
        vocab_size=len(dataset.vocab),
        n=n,
        base_d=128,
        num_mini_generators=55  # 5 mini-generators in ensemble
    )
    super_gen.set_super_temperature(0.1)  # Slightly sharper super probability

    print("Training base n-gram model...")
    for epoch in range(2):
        loss = train(model, loader, opt, device)
        print(f"\nEpoch {epoch} | train loss {loss:.4f}")

        _, token_loss_map = benchmark.evaluate(model, device, epoch)
        bench_processor.update(token_loss_map)

    print("\nOptional: Train mini-generators for super probability...")
    print("  (Uncomment to train: super_gen.train_mini_generators(loader, opt, device, epochs=2))")
    
    print("\nGenerating with super probability mini-generators...\n")
    print("Use use_super_probs=True to enable ensemble generation")
    print()
    
    while True:
        user_input = input("USER: ")
        
        # Check for flag
        use_super = "--super" in user_input
        if use_super:
            user_input = user_input.replace("--super", "").strip()
        
        output = generate(
            model,
            dataset.vocab,
            user_input,
            device,
            iso_processor,
            bench_processor,
            streak_tracker,
            super_gen=super_gen,
            use_super_probs=use_super,
        )
        print(output)
        print()


if __name__ == "__main__":
    filepath = input("Filename: ")
    main(filepath, n=2)
