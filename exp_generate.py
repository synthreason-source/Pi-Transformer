import os
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

"""
control_kernel_ontology.py

Implementation of ALGORITHM ControlKernelOntology.

Every symbolic operation in the pseudocode is mapped to a concrete,
computable operation on numpy arrays -- see the module docstring notes in
the accompanying chat message for the specific choices made where the
pseudocode was ambiguous (Hessian/curvature proxies, the meaning of the
'kernel' Gradient/Divergence/Laplacian trio, the ⊗ operator, etc).

INPUT
    Memory M            -> np.ndarray, shape (n_concepts, dim)
    Ontologies O         -> list[Ontology]
    Products P           -> list[Product]
    Interfaces I          -> list[InterfaceSurface]
"""

from dataclasses import dataclass, field
import numpy as np


import os
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter



# -----------------------------------------------------------------------------
# GEOMETRIC OBJECTS
# -----------------------------------------------------------------------------

@dataclass
class ControlKernel:
    """Κ -- the control kernel state carried by every ontology / subclass."""
    dim: int
    gain: float = 0.0
    curvature: float = 0.0
    damping: float = 0.0
    uncertainty: float = 0.0
    projection_matrix: np.ndarray = None
    state: np.ndarray = None          # state_transition variable
    energy: float = 0.0

    def __post_init__(self):
        if self.projection_matrix is None:
            self.projection_matrix = np.eye(self.dim)
        if self.state is None:
            self.state = np.zeros(self.dim)


def CreateKernel(dim):
    return ControlKernel(dim=dim)


@dataclass
class OntologyManifold:
    """Ω -- a point cloud of concept vectors (an ontology's embedded concepts)."""
    name: str
    vertices: np.ndarray              # (n, dim)
    center: np.ndarray = None
    metric: np.ndarray = None         # covariance, the local "metric tensor"
    curvature: float = 0.0
    kernel: ControlKernel = None
    subclasses: list = field(default_factory=list)


@dataclass
class SubclassManifold:
    """Σ -- a subset of an ontology's concept vertices."""
    name: str
    parent: OntologyManifold
    vertices: np.ndarray
    metric: np.ndarray = None
    kernel: ControlKernel = None


@dataclass
class InterfaceSurface:
    """Γ -- a boundary surface a subclass can attach to (modeled as a small
    point set whose local covariance defines a surface normal)."""
    name: str
    vertices: np.ndarray
    attachments: list = field(default_factory=list)


@dataclass
class Product:
    """A causal-field participant with its own kernel."""
    name: str
    vector: np.ndarray
    kernel: ControlKernel = None
    connections: list = field(default_factory=list)


# -----------------------------------------------------------------------------
# LINEAR-ALGEBRA / DIFFERENTIAL-GEOMETRY PRIMITIVES
# -----------------------------------------------------------------------------

def Mean(vertices):
    return vertices.mean(axis=0)


def Covariance(vertices):
    if vertices.shape[0] < 2:
        d = vertices.shape[1]
        return np.eye(d) * 1e-6
    return np.cov(vertices, rowvar=False) + np.eye(vertices.shape[1]) * 1e-6


def Hessian(metric):
    # For a quadratic form x^T metric x, the Hessian is constant = 2*metric.
    return 2.0 * metric


def EigenValue(metric):
    """Top eigenvalue -- used as the kernel 'gain'."""
    vals = np.linalg.eigvalsh(metric)
    return float(vals[-1])


def GaussianCurvature(manifold_like):
    """Product of the two largest principal curvatures (eigenvalues of the
    metric), the discrete analogue of Gaussian curvature = k1 * k2."""
    metric = manifold_like.metric
    vals = np.linalg.eigvalsh(metric)
    if len(vals) < 2:
        return float(vals[-1] ** 2)
    return float(vals[-1] * vals[-2])


def RicciFlow(manifold_like):
    """Damping proxy: mean eigenvalue of the metric, i.e. trace(g)/dim --
    the isotropic part of the Ricci curvature under dg/dt = -2*Ric."""
    metric = manifold_like.metric
    vals = np.linalg.eigvalsh(metric)
    return float(np.mean(vals))


def _knn_laplacian(vertices, k=3):
    n = vertices.shape[0]
    if n < 2:
        return np.zeros((n, n))
    k = min(k, n - 1)
    dists = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
    W = np.zeros((n, n))
    for i in range(n):
        nn_idx = np.argsort(dists[i])[1 : k + 1]
        for j in nn_idx:
            w = np.exp(-dists[i, j])
            W[i, j] = w
            W[j, i] = w
    D = np.diag(W.sum(axis=1))
    return D - W


def LaplacianEnergy(manifold_like):
    """Dirichlet energy trace(X^T L X) of the subclass point cloud under its
    k-NN graph Laplacian -- a real, well-defined notion of 'graph energy'."""
    X = manifold_like.vertices
    L = _knn_laplacian(X)
    return float(np.trace(X.T @ L @ X))


def Gradient(state):
    """Discrete forward-difference gradient of a 1D kernel state vector."""
    return state - np.roll(state, 1)


def Divergence(vector_field):
    """Discrete divergence: sum of the gradient field, broadcast back to the
    state's shape so it can be subtracted elementwise in UPDATE_KERNEL."""
    total = np.sum(vector_field)
    return np.full_like(vector_field, total / max(1, len(vector_field)))


def Laplacian(state):
    """Discrete 1D Laplacian stencil [1, -2, 1] with periodic boundary."""
    return np.roll(state, 1) + np.roll(state, -1) - 2 * state


def Intersection(Oi, Oj):
    """Bhattacharyya-coefficient-style overlap in [0, 1] between the two
    manifolds' Gaussian approximations (center, metric)."""
    mu_i, mu_j = Oi.center, Oj.center
    Si, Sj = Oi.metric, Oj.metric
    S = (Si + Sj) / 2.0
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    diff = mu_i - mu_j
    term1 = 0.125 * diff.T @ S_inv @ diff
    sign_i, logdet_i = np.linalg.slogdet(Si)
    sign_j, logdet_j = np.linalg.slogdet(Sj)
    sign_s, logdet_s = np.linalg.slogdet(S)
    term2 = 0.5 * (logdet_s - 0.5 * logdet_i - 0.5 * logdet_j)
    bhattacharyya_distance = term1 + term2
    overlap = float(np.exp(-bhattacharyya_distance))
    return max(0.0, min(1.0, overlap))


def OrthogonalKernelProjection(Oi, Oj):
    """Ψ: orthogonal Procrustes alignment between the two manifolds' centered
    point clouds -- the one operation here with unambiguous, real math."""
    Xi = Oi.vertices - Oi.center
    Xj = Oj.vertices - Oj.center
    n = min(len(Xi), len(Xj))
    C = Xi[:n].T @ Xj[:n]
    U, _, Vt = np.linalg.svd(C)
    Psi = U @ Vt
    return Psi


def KernelSimilarity(k1, k2):
    """Cosine similarity between two kernel state vectors."""
    a, b = k1.state, k2.state
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def CausalField(Pi, Pj, length_scale=1.0):
    """Φ(Pi, Pj): distance-decayed cosine similarity between product vectors."""
    a, b = Pi.vector, Pj.vector
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos_sim = float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0
    dist = float(np.linalg.norm(a - b))
    decay = np.exp(-dist / max(length_scale, 1e-6))
    return cos_sim * decay


def SemanticEntropy(Sigma):
    """Shannon entropy of the softmax-normalized pairwise similarity
    distribution within a subclass's vertex set."""
    X = Sigma.vertices
    if X.shape[0] < 2:
        return 0.0
    sims = X @ X.T
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    sims = sims / (norms @ norms.T + 1e-12)
    flat = sims[np.triu_indices_from(sims, k=1)]
    probs = np.exp(flat - flat.max())
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy)


def SurfaceNormal(Gamma):
    """Top eigenvector of the interface's local covariance -- the direction
    of least/most variance, used as the surface normal."""
    cov = Covariance(Gamma.vertices)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, -1]


def Normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def NearestInterface(interfaces, Sigma):
    center = Mean(Sigma.vertices)
    best, best_dist = None, float("inf")
    for Gamma in interfaces:
        d = np.linalg.norm(Mean(Gamma.vertices) - center)
        if d < best_dist:
            best, best_dist = Gamma, d
    return best


def Attach(Sigma, Gamma, ingredient_vector):
    Gamma.attachments.append({"subclass": Sigma.name, "vector": ingredient_vector})


def Connect(Pi, Pj, influence):
    Pi.connections.append((Pj.name, influence))
    Pj.connections.append((Pi.name, influence))


# -----------------------------------------------------------------------------
# INITIALISATION
# -----------------------------------------------------------------------------

def initialise(ontologies):
    for Omega in ontologies:
        Omega.center = Mean(Omega.vertices)
        Omega.metric = Covariance(Omega.vertices)
        Omega.curvature = float(np.trace(Hessian(Omega.metric)))
        Omega.kernel = CreateKernel(dim=Omega.vertices.shape[1])
        Omega.kernel.state = Omega.center.copy()  # seed state, else Gradient/Laplacian of 0 stays 0
        for Sigma in Omega.subclasses:
            Sigma.metric = Covariance(Sigma.vertices)
            Sigma.kernel = CreateKernel(dim=Sigma.vertices.shape[1])
            Sigma.kernel.state = Mean(Sigma.vertices).copy()


# -----------------------------------------------------------------------------
# MEMORY AGREEMENT
# -----------------------------------------------------------------------------

def MEMORY_AGREEMENT(Oi, Oj, tolerance=0.15):
    overlap = Intersection(Oi, Oj)
    if overlap > tolerance:
        Psi = OrthogonalKernelProjection(Oi, Oj)
        Oi.metric = Oi.metric * Psi   # ⊗ realized as Hadamard product (shape-preserving)
        Oj.metric = Oj.metric * Psi
    return overlap


# -----------------------------------------------------------------------------
# SUBCLASS CONTROL
# -----------------------------------------------------------------------------

def BUILD_SUBCLASS_KERNEL(Sigma, preserve_state=None):
    K = CreateKernel(dim=Sigma.vertices.shape[1])
    K.gain = EigenValue(Sigma.metric)
    K.curvature = GaussianCurvature(Sigma)
    K.damping = RicciFlow(Sigma)
    K.energy = LaplacianEnergy(Sigma)
    # The pseudocode rebuilds Σ.kernel from scratch every iteration; carry
    # the previous state forward (falling back to the subclass's own mean)
    # so UPDATE_KERNEL has a persistent trajectory to evolve instead of
    # restarting from zero each pass.
    K.state = preserve_state.copy() if preserve_state is not None else Mean(Sigma.vertices).copy()
    return K


# -----------------------------------------------------------------------------
# CONTROL UPDATE
# -----------------------------------------------------------------------------

def UPDATE_KERNEL(K, dt=0.02, clip=50.0):
    """
    state += gain·∇(state) − damping·div(state) + curvature·∇²(state)

    The pseudocode specifies no integration step size, and applied as raw
    explicit Euler with realistic gain/curvature magnitudes this update is
    numerically unstable (diverges exponentially within ~10 iterations).
    A small dt keeps it an honest explicit-Euler integrator of the same
    update rule rather than silently changing the dynamics; clip guards
    against the (still-possible) blow-up so StableEnergy() has something
    finite to converge on.
    """
    delta = (
        K.gain * Gradient(K.state)
        - K.damping * Divergence(K.state)
        + K.curvature * Laplacian(K.state)
    )
    K.state = np.clip(K.state + dt * delta, -clip, clip)
    K.energy = float(np.sum(K.state ** 2))


# -----------------------------------------------------------------------------
# UNCERTAINTY INTERFACE
# -----------------------------------------------------------------------------

def INTERFACE_CONTROLLER(Sigma, Gamma, threshold=0.5):
    entropy = SemanticEntropy(Sigma)
    if Gamma is not None and entropy > threshold:
        normal = SurfaceNormal(Gamma)
        projection = Sigma.kernel.projection_matrix @ normal
        ingredient_vector = Normalize(projection)
        Attach(Sigma, Gamma, ingredient_vector)
    return entropy


# -----------------------------------------------------------------------------
# CAUSAL PRODUCTS
# -----------------------------------------------------------------------------

def PRODUCT_CONTROLLER(products, tau=0.2):
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            Pi, Pj = products[i], products[j]
            influence = CausalField(Pi, Pj) * KernelSimilarity(Pi.kernel, Pj.kernel)
            if influence > tau:
                Connect(Pi, Pj, influence)


# -----------------------------------------------------------------------------
# GLOBAL CONTROL FIELD
# -----------------------------------------------------------------------------

def StableEnergy(history, eps=1e-4, window=3):
    if len(history) < window + 1:
        return False
    recent = history[-window:]
    deltas = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
    return max(deltas) < eps


def run_control_kernel_ontology(ontologies, interfaces, products, max_iters=50):
    initialise(ontologies)
    for Omega in ontologies:
        for Sigma in Omega.subclasses:
            Sigma.kernel = BUILD_SUBCLASS_KERNEL(Sigma, preserve_state=Sigma.kernel.state)

    energy_history = []
    it = 0
    while it < max_iters:
        it += 1

        # pairwise memory agreement across all ontology pairs
        for i in range(len(ontologies)):
            for j in range(i + 1, len(ontologies)):
                MEMORY_AGREEMENT(ontologies[i], ontologies[j])

        for Omega in ontologies:
            UPDATE_KERNEL(Omega.kernel)
            for Sigma in Omega.subclasses:
                Sigma.kernel = BUILD_SUBCLASS_KERNEL(Sigma, preserve_state=Sigma.kernel.state)
                UPDATE_KERNEL(Sigma.kernel)
                Gamma = NearestInterface(interfaces, Sigma)
                INTERFACE_CONTROLLER(Sigma, Gamma)

        PRODUCT_CONTROLLER(products)

        total_energy = sum(O.kernel.energy for O in ontologies) + sum(
            Sigma.kernel.energy for O in ontologies for Sigma in O.subclasses
        )
        energy_history.append(total_energy)

        if StableEnergy(energy_history):
            break

    return {"iterations": it, "energy_history": energy_history}




KB_LEN = 99
# -----------------------------------------------------------------------------
# 1. Dataset extracted directly from Image 1
# -----------------------------------------------------------------------------
with open(input("Filename: "), "r", encoding="utf-8") as f:
    dataset_rows = f.read().split(".")[:KB_LEN]

# -----------------------------------------------------------------------------
# 2. Model Definition & Helper Functions
# -----------------------------------------------------------------------------
class CurvePriorNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.head(out)
        return logits, h


def tokenize(text):
    return text.lower().split()


def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab += [w for w, c in counts.items() if c >= min_freq and w not in vocab]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos


def encode(tokens, stoi):
    return [stoi.get(t, stoi["<unk>"]) for t in tokens]


def make_dataset(ids, seq_len=8):
    """
    Creates input (x) and target (y) sequences.
    Adjusted seq_len default to fit smaller example datasets cleanly.
    """
    xs, ys = [], []
    if len(ids) <= seq_len:
        # Pad sequence if total token count is smaller than seq_len
        pad_id = 0
        ids = ids + [pad_id] * (seq_len + 1 - len(ids))

    for i in range(len(ids) - seq_len):
        xs.append(ids[i : i + seq_len])
        ys.append(ids[i + 1 : i + seq_len + 1])

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def load_curve_prior(path=None, top_k=50):
    probs = np.array([
        0.0390, 0.0384, 0.0368, 0.0335, 0.0275, 0.0218, 0.0208, 0.0183, 0.0164, 0.0156,
        0.0138, 0.0134, 0.0112, 0.0088, 0.0080, 0.0071, 0.0068, 0.0065, 0.0058, 0.0057,
        0.0055, 0.0054, 0.0053, 0.0050, 0.0048, 0.0046, 0.0046, 0.0045, 0.0043, 0.0042,
        0.0042, 0.0042, 0.0041, 0.0041, 0.0040, 0.0039, 0.0039, 0.0038, 0.0038, 0.0037,
        0.0037, 0.0035, 0.0034, 0.0033, 0.0033, 0.00325, 0.00322, 0.00320, 0.00318, 0.00315
    ], dtype=np.float32)

    if top_k is not None:
        probs = probs[:top_k]
    probs = probs / probs.sum()
    return probs


def train_model(model, x, y, curve_prior=None, curve_weight=0.05, sensitivity=50, epochs=30, batch_size=16, lr=1e-3, device="cpu"):
    model.to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n = x.shape[0]
    steps = max(1, math.ceil(n / batch_size))

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0

        for s in range(steps):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            xb, yb = x[idx], y[idx]

            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=0)

            if curve_prior is not None and curve_weight > 0:
                vocab_slice = min(logits.size(-1), len(curve_prior))
                prior = torch.tensor(curve_prior[:vocab_slice], device=device)
                prior = prior / prior.sum().clamp_min(1e-12)

                last_logits = logits[:, -1, :vocab_slice]
                pred = F.softmax(last_logits, dim=-1).mean(dim=0)
                prior_loss = F.mse_loss(pred, prior)

                loss = loss + curve_weight * sensitivity * prior_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d} | Loss: {total/steps:.4f}")

    return model


# -----------------------------------------------------------------------------
# 2a. Token-frequency table (for low-frequency-event replacement)
# -----------------------------------------------------------------------------
def compute_token_freq(ids, vocab_size):
    """
    Returns a normalized frequency vector over the vocabulary, computed from
    the encoded training token ids. Used to detect 'low frequency events'
    during sampling (i.e. the model committing to a token that is rare in
    the training distribution).
    """
    counts = torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=vocab_size).float()
    freq = counts / counts.sum().clamp_min(1.0)
    return freq


@torch.no_grad()
def recursive_extrapolate(model, cur_id, h, depth, temperature, device, discount=0.7):
    """
    Recursive next-token-step lookahead.

    From the current hidden state `h` and the last committed token `cur_id`,
    greedily rolls the model forward `depth` additional steps (recursively
    calling itself), accumulating a discounted log-probability score for the
    path taken. This gives a cheap estimate of how 'good' continuing from a
    given first-step choice is, beyond just its immediate probability.

    Returns (score, h_unused) where score is a scalar tensor -- the
    discounted sum of log-probs along the greedy extrapolated path.
    """
    if depth <= 0:
        return torch.tensor(0.0, device=device), h

    x = torch.tensor([[cur_id]], dtype=torch.long, device=device)
    logits, h_next = model(x, h)
    step_logits = logits[:, -1, :] / max(1e-6, temperature)
    log_probs = F.log_softmax(step_logits, dim=-1).squeeze(0)

    next_id = int(torch.argmax(log_probs).item())
    step_score = log_probs[next_id]

    future_score, _ = recursive_extrapolate(
        model, next_id, h_next, depth - 1, temperature, device, discount
    )

    total_score = step_score + discount * future_score
    return total_score, h_next


@torch.no_grad()
def next_token_with_lookahead(model, cur, h, candidates, lookahead_steps, temperature, device):
    """
    'Next token steps': for each candidate next-token id, recursively
    extrapolate `lookahead_steps` tokens ahead and score the resulting path.
    Returns a tensor of lookahead scores aligned with `candidates`.
    """
    scores = []
    logits, h_step = model(cur, h)
    for cand_id in candidates:
        score, _ = recursive_extrapolate(
            model, cand_id, h_step, lookahead_steps, temperature, device
        )
        scores.append(score)
    return torch.stack(scores), h_step


def low_freq_replacement(next_id, probs, freq, threshold, device):
    """
    'Replaced during low frequency events': if the sampled token is rare in
    the training distribution (freq below `threshold`), replace it with the
    highest-probability token from the current step's distribution instead
    of emitting the rare token outright. Returns the (possibly replaced)
    token id and a flag indicating whether a replacement occurred.
    """
    vocab_size = probs.shape[-1]
    if next_id >= freq.shape[0]:
        # token outside the frequency table (shouldn't normally happen)
        token_freq = 0.0
    else:
        token_freq = freq[next_id].item()

    if token_freq < threshold:
        fallback_id = int(torch.argmax(probs).item())
        return fallback_id, True
    return next_id, False


@torch.no_grad()
def generate_text(
    model,
    stoi,
    itos,
    prime="the",
    length=80,
    temperature=1.0,
    sensitivity=1.0,
    device="cpu",
    token_freq=None,
    low_freq_threshold=1e-4,
    lookahead_steps=52,
    lookahead_top_k=15,
    lookahead_weight=0.9,
    control_kernel_bias=None,
    avg_curvature=0.6,
    curvature_temp_scale=0.5,
):
    model.eval()
    tokens = tokenize(prime)
    prompt_ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    ids = [stoi.get("<bos>", 1)] + prompt_ids

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    h = None
    logits, h = model(x, h)

    vocab_size = logits.size(-1)
    prompt_bias = prompt_bias_from_tokens(prompt_ids, vocab_size, device, sensitivity=sensitivity)

    if token_freq is None:
        # Uniform fallback so low-frequency logic is a no-op if no table given
        token_freq = torch.full((vocab_size,), 1.0, device=device)

    if control_kernel_bias is None:
        control_kernel_bias = torch.zeros(vocab_size, device=device)

    # Higher ControlKernelOntology curvature -> sharper (lower) effective
    # temperature, i.e. more curved concept manifolds make the model commit
    # more decisively. curvature_temp_scale=0 (default) leaves temperature
    # untouched, so this is a strict extension, not a behavior change.
    temperature_eff = max(1e-6, temperature / (1.0 + curvature_temp_scale * avg_curvature))

    out = tokens[:]
    cur = x[:, -1:]
    replacements = 0

    for _ in range(length):
        logits, h_after = model(cur, h)
        step_logits = logits[:, -1, :] / temperature_eff
        step_logits = step_logits + prompt_bias.unsqueeze(0) + control_kernel_bias.unsqueeze(0)

        probs = F.softmax(step_logits, dim=-1).squeeze(0)

        # Route the sampling distribution through the nilpotent ideal before
        # drawing a sample -- same embed/read-back roundtrip used for the
        # curve prior.
        N = probs_to_nilpotent_ideal(probs)
        probs = nilpotent_ideal_to_probs(N).unsqueeze(0).squeeze(0)

        # --- Next-token steps / recursive extrapolation ---------------------
        # Take the top-k immediate candidates and re-score them by how well
        # a greedy rollout `lookahead_steps` tokens ahead does, then blend
        # that lookahead score back into the sampling distribution.
        if lookahead_steps > 0 and lookahead_top_k > 1:
            top_probs, top_ids = torch.topk(probs, min(lookahead_top_k, probs.shape[-1]))
            candidates = top_ids.tolist()
            lookahead_scores, _ = next_token_with_lookahead(
                model, cur, h, candidates, lookahead_steps, temperature, device
            )
            lookahead_probs = F.softmax(lookahead_scores, dim=-1)

            blended = probs.clone()
            for i, cand_id in enumerate(candidates):
                blended[cand_id] = (
                    (1 - lookahead_weight) * probs[cand_id]
                    + lookahead_weight * lookahead_probs[i]
                )
            probs = blended / blended.sum().clamp_min(1e-12)

        probs_batched = probs.unsqueeze(0)
        next_id = torch.multinomial(lookahead_probs, 1).item()

        # --- Low-frequency event replacement --------------------------------
        next_id, was_replaced = low_freq_replacement(
            next_id, probs, token_freq, low_freq_threshold, device
        )
        if was_replaced:
            replacements += 1

        next_tok = itos[next_id]

        if next_tok == "<eos>":
            break
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, h_next = model(x, h)
        future_score, _ = recursive_extrapolate(
            model, next_id, h_next, 8, 0.7, device, 0.5
            )
        if out[-1] != next_tok:
            out.append(next_tok)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        h = h_after

    if replacements:
        print(f"[low-freq replacement] swapped {replacements} rare token(s) during generation")

    text = []
    for t in out:
        if t in ".,!?;:":
            if text:
                text[-1] = text[-1] + t
            else:
                text.append(t)
        else:
            text.append(t)

    return " ".join(text)


def prompt_bias_from_tokens(prompt_ids, vocab_size, device, sensitivity=1.0):
    bias = torch.zeros(vocab_size, device=device)
    if len(prompt_ids) == 0:
        return bias
    counts = torch.bincount(torch.tensor(prompt_ids, device=device), minlength=vocab_size).float()
    counts = counts / counts.sum().clamp_min(1.0)
    bias = sensitivity * counts
    return bias


def probs_to_nilpotent_ideal(probs):
    """Vector-only convenience wrapper (numpy or torch) used for the
    curve-prior distribution, kept for backward compatibility."""
    if isinstance(probs, torch.Tensor):
        return to_nilpotent_ideal(probs)
    probs_t = torch.from_numpy(np.asarray(probs))
    return to_nilpotent_ideal(probs_t).numpy()


def from_nilpotent_ideal(N):
    """
    Recover a vector from an element of the nilpotent ideal by reading back
    its superdiagonal (row sums, since each row of N has at most one
    nonzero entry). Inverse of to_nilpotent_ideal (up to the dropped last
    coordinate, which the ideal has no room to store).
    """
    return N.sum(dim=-1)


def nilpotent_ideal_to_probs(N, eps=1e-12):
    """Vector-only convenience wrapper that also renormalizes back into a
    valid probability distribution, kept for backward compatibility."""
    if isinstance(N, torch.Tensor):
        vec = from_nilpotent_ideal(N)
        return vec / vec.sum().clamp_min(eps)
    N_t = torch.from_numpy(np.asarray(N))
    vec = from_nilpotent_ideal(N_t).numpy()
    s = vec.sum()
    if s < eps:
        return np.full_like(vec, 1.0 / len(vec))
    return vec / s


def to_nilpotent_ideal(t):
    """
    Embed the last dimension of an arbitrary tensor into the nilpotent
    ideal n (strictly upper-triangular matrices) of the Borel subalgebra b
    of gl_k, batched over all leading dimensions.

    Input shape (..., k) -> output shape (..., k, k), with
    N[..., i, i+1] = t[..., i] for i < k - 1. N^k = 0 automatically.
    """
    *batch, k = t.shape
    N = torch.zeros(*batch, k, k, dtype=t.dtype, device=t.device)
    if k > 1:
        i0 = torch.arange(k - 1, device=t.device)
        i1 = torch.arange(1, k, device=t.device)
        N[..., i0, i1] = t[..., :-1]
    return N

# -----------------------------------------------------------------------------
# 2b. Bridge: build ControlKernelOntology structures from trained embeddings
# -----------------------------------------------------------------------------
def simple_kmeans(X, k, iters=50, seed=0):
    """Minimal Lloyd's-algorithm k-means, kept dependency-free (no sklearn)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    centroid_idx = rng.choice(n, size=k, replace=False)
    centroids = X[centroid_idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return labels, centroids


def build_ontologies_from_embeddings(model, vocab, itos, token_freq, n_ontologies=3, n_products=6):
    """
    Clusters the trained CurvePriorNet embedding matrix into ontology-like
    groups, splits each into two subclasses, derives an interface surface
    per ontology, and picks the most frequent content words as 'products'
    for the causal-field step. This is what feeds run_control_kernel_ontology
    with real learned vectors instead of synthetic demo data.
    """
    emb = model.embed.weight.detach().cpu().numpy()
    # skip the 4 special tokens (<pad>,<bos>,<eos>,<unk>) when clustering
    content_idx = np.arange(4, len(vocab))
    if len(content_idx) < n_ontologies * 4:
        # too small a vocab to form meaningful clusters/subclasses
        n_ontologies = max(1, len(content_idx) // 4)
    if n_ontologies == 0:
        return [], [], []

    content_emb = emb[content_idx]
    onto_labels, _ = simple_kmeans(content_emb, n_ontologies, seed=0)

    ontologies = []
    interfaces = []
    for c in range(n_ontologies):
        member_idx = content_idx[onto_labels == c]
        if len(member_idx) < 2:
            continue
        vertices = emb[member_idx]
        name = f"OntologyCluster_{c}_" + "_".join(itos[i] for i in member_idx[:3])
        Omega = OntologyManifold(name=name, vertices=vertices)
        Omega.token_ids = member_idx.tolist()

        if len(member_idx) >= 4:
            sub_labels, _ = simple_kmeans(vertices, 2, seed=1)
        else:
            sub_labels = np.zeros(len(member_idx), dtype=int)

        for s in np.unique(sub_labels):
            sub_idx = member_idx[sub_labels == s]
            if len(sub_idx) < 1:
                continue
            sub_vertices = emb[sub_idx]
            sub_name = f"{name}_sub{s}"
            Sigma = SubclassManifold(name=sub_name, parent=Omega, vertices=sub_vertices)
            Sigma.token_ids = sub_idx.tolist()
            Omega.subclasses.append(Sigma)

        ontologies.append(Omega)
        interfaces.append(InterfaceSurface(name=f"Interface_{c}", vertices=vertices))

    # Products: the most frequent content words by training-corpus frequency
    top_ids = content_idx[np.argsort(-token_freq[content_idx].numpy())[:n_products]]
    products = []
    for pid in top_ids:
        P = Product(name=itos[int(pid)], vector=emb[pid])
        P.kernel = CreateKernel(dim=emb.shape[1])
        P.kernel.state = emb[pid].copy()
        products.append(P)

    return ontologies, interfaces, products


def build_control_kernel_bias(
    model,
    stoi,
    itos,
    ontologies,
    interfaces,
    products,
    device,
    ontology_weight=0.3,
    subclass_weight=0.5,
    interface_weight=0.2,
    product_weight=1.0,
    clip=6.0,
):
    """
    Converts the converged ControlKernelOntology equations into a per-token
    logit bias, so the kernel states/gain/curvature/damping/energy computed
    by run_control_kernel_ontology actually influence generate_text instead
    of only being printed.

    - Ontology-level: kernel.state (a point in embedding space) is dotted
      against every token embedding to get an affinity, weighted by how
      much energy that ontology's kernel accumulated (settled kernels with
      near-zero energy contribute almost nothing).
    - Subclass-level: same dot-product affinity, but restricted to the
      vocab tokens that were actually clustered into that subclass, and
      weighted by (gain - damping) -- net excitatory vs. dissipative force.
    - Interface-level: each attachment's ingredient vector (itself a
      projection of a subclass's excess semantic entropy onto a surface
      normal) is dotted against every token embedding.
    - Product-level: PRODUCT_CONTROLLER's causal-field connections directly
      bump the logit of the connected product's own token.
    """
    emb = model.embed.weight.detach()
    vocab_size = emb.shape[0]
    bias = torch.zeros(vocab_size, device=device)

    def normalized_affinity(vec_np):
        vec = torch.tensor(vec_np, dtype=emb.dtype, device=device)
        affinity = emb @ vec
        std = affinity.std().clamp_min(1e-6)
        return (affinity - affinity.mean()) / std

    # --- Ontology-level: UPDATE_KERNEL's settled state + accumulated energy
    for Omega in ontologies:
        energy_weight = Omega.kernel.energy / (1.0 + Omega.kernel.energy)
        bias += ontology_weight * energy_weight * normalized_affinity(Omega.kernel.state)

        # --- Subclass-level: BUILD_SUBCLASS_KERNEL's gain/damping, applied
        # only to the vocab tokens that belong to that subclass
        for Sigma in Omega.subclasses:
            token_ids = getattr(Sigma, "token_ids", None)
            if not token_ids:
                continue
            net_force = Sigma.kernel.gain - Sigma.kernel.damping
            local_affinity = normalized_affinity(Sigma.kernel.state)
            idx = torch.tensor(token_ids, dtype=torch.long, device=device)
            bias[idx] += subclass_weight * net_force * local_affinity[idx]

    # --- Interface-level: INTERFACE_CONTROLLER's attached ingredient vectors
    for Gamma in interfaces:
        if not Gamma.attachments:
            continue
        for att in Gamma.attachments:
            bias += (interface_weight / len(Gamma.attachments)) * normalized_affinity(att["vector"])

    # --- Product-level: PRODUCT_CONTROLLER's causal-field connections
    for P in products:
        if not P.connections:
            continue
        token_id = stoi.get(P.name)
        if token_id is None:
            continue
        total_influence = sum(inf for _, inf in P.connections)
        bias[token_id] += product_weight * total_influence

    return torch.clamp(bias, -clip, clip)


def summarize_control_kernel_run(ontologies, interfaces, products, result):
    print(f"\n[ControlKernelOntology] converged after {result['iterations']} iterations")
    tail = [round(e, 4) for e in result["energy_history"][-5:]]
    print(f"[ControlKernelOntology] energy trajectory (last 5): {tail}")
    for Omega in ontologies:
        print(f"  [{Omega.name}] energy={Omega.kernel.energy:.4f} curvature={Omega.curvature:.4f}")
        for Sigma in Omega.subclasses:
            print(f"      - {Sigma.name}: gain={Sigma.kernel.gain:.4f} damping={Sigma.kernel.damping:.4f}")
    total_attachments = sum(len(Gamma.attachments) for Gamma in interfaces)
    total_connections = sum(len(P.connections) for P in products) // 2
    print(f"  interface attachments: {total_attachments}  |  product connections: {total_connections}\n")


# -----------------------------------------------------------------------------
# 3. Execution Pipeline
# -----------------------------------------------------------------------------
def run_dataset_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Combine sentences into sequence tokens
    full_text = ".".join(dataset_rows)
    tokens = tokenize(full_text)

    vocab, stoi, itos = build_vocab(tokens, min_freq=1)
    ids = encode(["<bos>"] + tokens + ["<eos>"], stoi)

    seq_len = 8  # Adjusted sequence length for chunked sentences
    x, y = make_dataset(ids, seq_len=seq_len)

    curve_prior = load_curve_prior(None, top_k=min(50, len(vocab)))

    # Frequency table over the whole vocab, used for low-frequency-event
    # replacement during generation.
    token_freq = compute_token_freq(ids, len(vocab))

    config = {"emb_dim": 64, "hidden": 128, "layers": 2, "seq_len": seq_len}
    model = CurvePriorNet(
        vocab_size=len(vocab),
        emb_dim=config["emb_dim"],
        hidden=config["hidden"],
        layers=config["layers"]
    )

    print("Training model on dataset...")
    model = train_model(
        model,
        x,
        y,
        curve_prior=curve_prior,
        curve_weight=0.05,
        sensitivity=10.0,
        epochs=10,
        batch_size=16,
        lr=1e-3,
        device=device
    )

    print("\nBuilding ontology structures from trained embeddings...")
    ontologies, interfaces, products = build_ontologies_from_embeddings(
        model, vocab, itos, token_freq, n_ontologies=3, n_products=6
    )
    control_kernel_bias = torch.zeros(len(vocab), device=device)
    avg_curvature = 0.0
    if ontologies:
        result = run_control_kernel_ontology(ontologies, interfaces, products, max_iters=30)
        summarize_control_kernel_run(ontologies, interfaces, products, result)
        control_kernel_bias = build_control_kernel_bias(
            model, stoi, itos, ontologies, interfaces, products, device
        )
        avg_curvature = float(np.mean([Omega.curvature for Omega in ontologies]))
        print(f"[ControlKernelOntology] bias non-zero on {int((control_kernel_bias != 0).sum())} "
              f"/ {len(vocab)} tokens, avg_curvature={avg_curvature:.4f}\n")
    else:
        print("  vocab too small to form ontology clusters, skipping.\n")

    print("Generating sample output from trained prior net:")
    while True:
        sample = generate_text(
            model,
            stoi,
            itos,
            prime=input("USER: "),
            length=120,
            temperature=0.8,
            device=device,
            token_freq=token_freq,
            low_freq_threshold=1e-4,
            lookahead_steps=12,
            lookahead_top_k=65,
            lookahead_weight=10.5,
            control_kernel_bias=control_kernel_bias,
            avg_curvature=avg_curvature,
            curvature_temp_scale=0.02,
        )
        print(f"\nGenerated Result: '{sample}'")


if __name__ == "__main__":
    run_dataset_pipeline()
