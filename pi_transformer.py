"""
pi_automorphism_net.py  (grad-fix edition)
──────────────────────────────────────────
Grad fixes applied in this revision
────────────────────────────────────
  GRAD-1  Every layer dict now stores "probs_t": live tensor alongside
          "probs": detached numpy — graph is never severed.
  GRAD-2  _p() helper prefers "probs_t" so rule losses stay connected.
  GRAD-3  _run_assertions() collects into list → torch.stack().sum()
          so the accumulator is never a dead leaf.

All previous fixes retained:
  BUG-1   single forward pass
  BUG-2   diag["total"] after divide
  BUG-3   _norm() after truncating rescore slice
  BUG-4   always backward()
  SIZE-FIX _align() in rule_convex / layer_loss
  RULE    assertions as Rule data objects + RuleRegistry
"""

from __future__ import annotations
import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

PHI = (1 + math.sqrt(5)) / 2
EPS = 1e-12

FieldSpec = Union[str, Callable[[Dict], Any]]


# ══════════════════════════════════════════════════════════════════════════════
# 0. UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _extract(ex: Dict, spec: FieldSpec) -> str:
    val = spec(ex) if callable(spec) else ex.get(spec)
    if val is None: return ""
    if isinstance(val, str): return val
    if isinstance(val, (list, tuple)):
        return " ".join(str(v) for v in val if v is not None)
    return str(val)

def _norm(t: torch.Tensor) -> torch.Tensor:
    return t / t.sum().clamp(min=EPS)

def _t(p, dtype=torch.float64) -> torch.Tensor:
    if isinstance(p, torch.Tensor): return p.to(dtype)
    if isinstance(p, np.ndarray):   return torch.from_numpy(p.astype(np.float64)).to(dtype)
    return torch.tensor(p, dtype=dtype)

def _char_trigrams(w: str):
    return {w[i:i+3] for i in range(len(w)-2)} if len(w) >= 3 else {w}

def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p.clamp(EPS) * p.clamp(EPS).log()).sum()

def _mean_rank(p: torch.Tensor) -> torch.Tensor:
    ranks = torch.arange(len(p), dtype=torch.float64) / max(1, len(p)-1)
    return (p * ranks).sum() / p.sum().clamp(EPS)

def _align(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Truncate all tensors to min length and renorm. SIZE-FIX."""
    n = min(len(t) for t in tensors)
    return tuple(_norm(t[:n]) for t in tensors)

def _p(d: Dict) -> torch.Tensor:
    """GRAD-2: prefer live 'probs_t' tensor; fall back to numpy 'probs'."""
    pt = d.get("probs_t")
    if pt is not None:
        return _norm(pt)
    return _norm(_t(d["probs"]))

def _layer_dict(name: str, words: List[str],
                pt: torch.Tensor, **extra) -> Dict:
    """GRAD-1: always store both live tensor and detached numpy."""
    return {"name": name, "words": words,
            "probs": pt.detach().numpy(), "probs_t": pt, **extra}


# ══════════════════════════════════════════════════════════════════════════════
# 1. RULE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rule:
    """
    Named algebraic assertion on a probability vector.

    Invariant:  loss(p, **meta) == 0  ⟺  check(p, **meta) is True
    Both are derived from the same closed-form condition.
    """
    name:  str
    op:    str
    check: Callable[..., bool]
    loss:  Callable[..., torch.Tensor]
    meta:  Dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args, **kw) -> torch.Tensor:
        return self.loss(*args, **kw)

    def verify(self, *args, **kw) -> bool:
        return self.check(*args, **kw)


def rule_simplex(tol: float = 1e-5) -> Rule:
    return Rule(
        name="SIMPLEX", op="simplex",
        check=lambda p: abs(float(p.sum())-1) < tol and bool((p >= 0).all()),
        loss =lambda p: (p.sum()-1.0)**2 + F.relu(-p).sum(),
        meta={"tol": tol},
    )

def rule_floor(fl: float) -> Rule:
    return Rule(
        name=f"FLOOR({fl})", op="floor",
        check=lambda p, fl=fl: bool((p >= fl).all()),
        loss =lambda p, fl=fl: F.relu(_t(fl) - p).sum(),
        meta={"floor": fl},
    )

def rule_hmin(h: float) -> Rule:
    return Rule(
        name=f"HMIN({h})", op="hmin",
        check=lambda p, h=h: float(_entropy(p)) >= h,
        loss =lambda p, h=h: F.relu(_t(h) - _entropy(p)),
        meta={"h": h},
    )

def rule_hmax(h: float) -> Rule:
    return Rule(
        name=f"HMAX({h})", op="hmax",
        check=lambda p, h=h: float(_entropy(p)) <= h,
        loss =lambda p, h=h: F.relu(_entropy(p) - _t(h)),
        meta={"h": h},
    )

def rule_monotone(direction: str = "decrease") -> Rule:
    return Rule(
        name=f"MONOTONE({direction})", op="monotone",
        check=lambda pb, pa: float(pa.max()) <= float(pb.max()),
        loss =lambda pb, pa: F.relu(pa.max() - pb.max()),
        meta={"direction": direction},
    )

def rule_coverage(tp: float) -> Rule:
    return Rule(
        name=f"COVERAGE({tp})", op="coverage",
        check=lambda p, tp=tp: float(p.sum()) >= tp,
        loss =lambda p, tp=tp: F.relu(_t(tp) - p.sum()),
        meta={"top_p": tp},
    )

def rule_convex() -> Rule:
    def _check(r, l, b):
        r, l, b = _align(r, l, b)
        return bool((b >= torch.min(r,l)).all() and (b <= torch.max(r,l)).all())
    def _loss(r, l, b):
        r, l, b = _align(r, l, b)
        return (F.relu(torch.min(r,l)-b) + F.relu(b-torch.max(r,l))).sum()
    return Rule(name="CONVEX", op="convex", check=_check, loss=_loss)

def rule_posanchor(pos: float, sigma: float) -> Rule:
    return Rule(
        name=f"POSANCHOR(pos={pos:.3f},σ={sigma:.3f})", op="posanchor",
        check=lambda p, pos=pos, sig=sigma: abs(float(_mean_rank(p))-pos) <= 2*sig,
        loss =lambda p, pos=pos, sig=sigma: ((_mean_rank(p)-pos)/max(sig,EPS))**2,
        meta={"pos": pos, "sigma": sigma},
    )

def rule_symm() -> Rule:
    return Rule(
        name="SYMM", op="symm",
        check=lambda p, tol=0.05: float((p-p.flip(0)).abs().max()) < tol,
        loss =lambda p: ((p-p.flip(0))**2).sum() / max(1, len(p)),
    )

def rule_uniform() -> Rule:
    return Rule(
        name="UNIFORM", op="uniform",
        check=lambda p, tol=0.05: float((p-1./max(1,len(p))).abs().max()) < tol,
        loss =lambda p: ((p - torch.full_like(p, 1./max(1,len(p))))**2).sum()
                        / max(1, len(p)),
    )

def rule_sqrtperm() -> Rule:
    def _perm(n):
        return (torch.arange(n, dtype=torch.float64)*max(1,n-1)).sqrt().long().clamp(0,n-1)
    def _phi(p):
        idx = _perm(len(p)); q = p[idx]; return q/q.sum().clamp(EPS)
    return Rule(
        name="SQRTPERM", op="sqrtperm",
        check=lambda p, tol=0.05: float((p-_phi(p)).abs().max()) < tol,
        loss =lambda p: ((p-_phi(p))**2).sum() / max(1, len(p)),
    )


class RuleRegistry:
    def __init__(self):
        self._r: Dict[str, Rule] = {}

    def register(self, key: str, rule: Rule) -> "RuleRegistry":
        self._r[key] = rule; return self

    def __getitem__(self, key: str) -> Rule: return self._r[key]
    def __contains__(self, key: str) -> bool: return key in self._r

    def loss(self, key: str, *args, **kw) -> torch.Tensor:
        return self._r[key].loss(*args, **kw)

    def check(self, key: str, *args, **kw) -> bool:
        return self._r[key].check(*args, **kw)

    def apply_all(self, keys: List[str], *args,
                  weights: Optional[Dict[str,float]] = None,
                  **kw) -> torch.Tensor:
        losses = [weights.get(k,1.0) * self._r[k].loss(*args, **kw)
                  for k in keys if k in self._r]
        return torch.stack(losses).sum() if losses else torch.zeros(1, dtype=torch.float64)

    def audit(self, p: torch.Tensor,
              keys: Optional[List[str]] = None) -> Dict[str, bool]:
        out = {}
        for k, rule in self._r.items():
            if keys and k not in keys: continue
            try:    out[k] = rule.check(p)
            except: out[k] = False
        return out


R = RuleRegistry()
R.register("simplex",  rule_simplex())
R.register("floor_05", rule_floor(0.05))
R.register("floor_04", rule_floor(0.04))
R.register("floor_03", rule_floor(0.03))
R.register("hmin_05",  rule_hmin(0.5))
R.register("hmax_6",   rule_hmax(6.0))
R.register("monotone", rule_monotone())
R.register("coverage", rule_coverage(1.0))
R.register("convex",   rule_convex())
R.register("symm",     rule_symm())
R.register("uniform",  rule_uniform())
R.register("sqrtperm", rule_sqrtperm())

LAYER_RULES: Dict[str, List[str]] = {
    "L0_RAW_DIST":            ["simplex", "monotone"],
    "L1_TEMP_SCALED":         ["simplex", "monotone"],
    "L2_INSIGHT":             ["simplex", "monotone"],
    "L3_TOPK_TOPP":           ["simplex", "coverage"],
    "L4_ZONE_FREQ":           ["simplex", "floor_05"],
    "L5_ZONE_ALPHA":          ["simplex", "floor_05"],
    "L6_ZONE_BIGRAM":         ["simplex", "floor_04"],
    "L7_ZONE_TRIGRAM":        ["simplex", "floor_03"],
    "L8_ZONE_CHAR_TRIG":      ["simplex", "floor_04"],
    "L9_ZONE_LATENT":         ["simplex", "floor_04"],
    "L10_HISTORY":            ["simplex", "monotone"],
    "L11_TENSOR_BLEND":       ["simplex", "convex"],
    "L12_FINAL":              ["simplex", "convex", "hmax_6"],
    "L13_CTX_REQ_POS":        ["simplex"],
    "L14_LOCKED_STATE_INDEX": ["simplex", "hmin_05"],
}

AA_RULES:   List[str]            = ["symm", "uniform", "sqrtperm"]
AA_WEIGHTS: Dict[str, float]     = {"symm": 1.0, "uniform": 0.5, "sqrtperm": 2.0}


def layer_loss(name: str, p_before: torch.Tensor, p_after: torch.Tensor,
               blend_ref: Optional[torch.Tensor] = None,
               draw_pos: float = 0., stream_len: int = 1,
               sigma: float = 0.30) -> torch.Tensor:
    keys   = LAYER_RULES.get(name, ["simplex"])
    losses: List[torch.Tensor] = []

    for k in keys:
        if k not in R: continue
        rule = R[k]
        if rule.op == "monotone":
            pb, pa = _align(p_before, p_after)
            losses.append(rule.loss(pb, pa))
        elif rule.op == "convex":
            ref = blend_ref if blend_ref is not None else p_before
            r, l, b = _align(p_before, ref, p_after)
            losses.append(rule.loss(r, l, b))
        elif rule.op == "posanchor":
            norm_pos = (draw_pos % max(1, stream_len)) / max(1, stream_len-1)
            losses.append(rule_posanchor(norm_pos, sigma).loss(p_after))
        else:
            losses.append(rule.loss(p_after))

    if name == "L13_CTX_REQ_POS":
        norm_pos = (draw_pos % max(1, stream_len)) / max(1, stream_len-1)
        losses.append(rule_posanchor(norm_pos, sigma).loss(p_after))

    losses.append(R.apply_all(AA_RULES, p_after, weights=AA_WEIGHTS))

    return torch.stack(losses).sum()   # GRAD-3: always has grad_fn


def layer_check(name: str, p: torch.Tensor,
                draw_pos: float = 0., stream_len: int = 1,
                sigma: float = 0.30) -> Dict[str, bool]:
    keys = LAYER_RULES.get(name, ["simplex"])
    result = {}
    for k in keys:
        if k not in R: continue
        rule = R[k]
        try:
            if rule.op == "posanchor":
                norm_pos = (draw_pos % max(1, stream_len)) / max(1, stream_len-1)
                result[k] = rule_posanchor(norm_pos, sigma).check(p)
            elif rule.op in ("monotone", "convex"):
                result[k] = True
            else:
                result[k] = rule.check(p)
        except:
            result[k] = False
    for k in AA_RULES:
        try:    result[k] = R[k].check(p)
        except: result[k] = False
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUTOMORPHISM NET
# ══════════════════════════════════════════════════════════════════════════════

class AutomorphismNet(nn.Module):
    def __init__(self, n: int = 64, hidden: int = 128,
                 floor: float = 0.0, h_min: float = 0.0, h_max: float = 1e9,
                 lam_prob: float = 1.0, lam_auto: float = 1.0):
        super().__init__()
        self.n = n
        self.lam_prob = lam_prob
        self.lam_auto = lam_auto
        if floor  > 0:   R.register("_net_floor", rule_floor(floor))
        if h_min  > 0:   R.register("_net_hmin",  rule_hmin(h_min))
        if h_max  < 1e9: R.register("_net_hmax",  rule_hmax(h_max))
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n),
        )
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self.log_temp.exp().clamp(1e-3, 10.0)
        return F.softmax(self.net(x.float()) / T, dim=-1).double()

    def automorphism_loss(self, x: torch.Tensor,
                          target: Optional[torch.Tensor] = None,
                          lam_ce: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        p    = self(x)   # BUG-1: single forward pass
        diag = {}
        batch_losses: List[torch.Tensor] = []

        for i, pi in enumerate(p):
            prob_losses = [R.loss("simplex", pi)]
            for k in ("_net_floor", "_net_hmin", "_net_hmax"):
                if k in R: prob_losses.append(R.loss(k, pi))
            prob_loss = torch.stack(prob_losses).sum()
            auto_loss = R.apply_all(AA_RULES, pi, weights=AA_WEIGHTS)
            batch_losses.append(self.lam_prob * prob_loss + self.lam_auto * auto_loss)
            diag[f"prob_loss_{i}"]   = float(prob_loss.detach())
            diag[f"auto_loss_{i}"]   = float(auto_loss.detach())
            diag[f"checks_{i}"]      = R.audit(pi.detach(), AA_RULES)

        if target is not None and lam_ce > 0:
            ce = F.nll_loss(p.float().log(), target.long())
            batch_losses.append(lam_ce * ce)
            diag["ce_loss"] = float(ce.detach())

        # GRAD-3: stack → always has grad_fn
        total         = torch.stack(batch_losses).sum()
        mean_loss     = total / max(1, len(p))   # BUG-2
        diag["total"] = float(mean_loss.detach())
        return mean_loss, diag


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE AUTOMORPHISM HEAD
# ══════════════════════════════════════════════════════════════════════════════

class PipelineAutomorphismHead(nn.Module):
    def __init__(self, n_cands: int = 100, hidden: int = 128, **kw):
        super().__init__()
        self.net = AutomorphismNet(n=n_cands, hidden=hidden, **kw)

    def frame_loss(self, frame: Dict) -> torch.Tensor:
        layers  = frame.get("layers", [])
        by_name = {l.get("name",""): l for l in layers if l.get("name")}
        losses: List[torch.Tensor] = []   # GRAD-3
        prev_p: Optional[torch.Tensor] = None

        for layer in layers:
            name = layer.get("name", "")
            p    = _p(layer)   # GRAD-2
            pb   = prev_p if prev_p is not None else p

            blend_ref = None
            if name == "L11_TENSOR_BLEND":
                zps = [_p(by_name[k]) for k in
                       ("L4_ZONE_FREQ","L5_ZONE_ALPHA","L6_ZONE_BIGRAM",
                        "L7_ZONE_TRIGRAM","L8_ZONE_CHAR_TRIG","L9_ZONE_LATENT",
                        "L10_HISTORY") if k in by_name]
                blend_ref = _norm(torch.stack(zps).mean(0)) if zps else None
            elif name == "L12_FINAL":
                blend_ref = (_p(by_name["L11_TENSOR_BLEND"])
                             if "L11_TENSOR_BLEND" in by_name else None)

            losses.append(layer_loss(
                name, pb, p,
                blend_ref  = blend_ref,
                draw_pos   = layer.get("draw_pos",   0),
                stream_len = layer.get("stream_len", 1),
            ))
            prev_p = p

        return torch.stack(losses).sum() if losses else torch.zeros(1, dtype=torch.float64)

    def rescore(self, pairs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        words = [w for w, _ in pairs]; n = len(words)
        p     = _norm(_t([v for _, v in pairs]))
        x     = p.float().unsqueeze(0)
        if n != self.net.n:
            x = F.pad(x, (0, max(0, self.net.n-n)))[:, :self.net.n]
        with torch.no_grad():
            p_new = self.net(x).squeeze(0).double()[:n]
        p_new = _norm(p_new)   # BUG-3
        return list(zip(words, p_new.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# 4. O(n²) PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    def __init__(self, dataset_name: str, config_name: Optional[str] = None,
                 text_fields: Optional[Sequence[FieldSpec]] = None, *,
                 split_names: Optional[Sequence[str]] = None, lowercase: bool = True,
                 minlen: int = 3, max_per_split: Optional[int] = None,
                 boundaryquota: int = 1, streaming: bool = False):
        self.dataset_name, self.config_name = dataset_name, config_name
        self.text_fields   = list(text_fields) if text_fields else None
        self.split_names   = split_names; self.lowercase = lowercase
        self.minlen        = max(2, minlen); self.max_per_split = max_per_split
        self.boundaryquota = max(1, boundaryquota); self.streaming = streaming
        self.sentences: List[List[str]]                   = []
        self.tokens: List[str]                            = []
        self.middlepool: List[str]                        = []
        self.middlecorr: Dict[str, List[Tuple[int,int]]] = {}
        self._orderedpool: List[str]                      = []
        self._begincounts: Counter                        = Counter()
        self._endcounts: Counter                          = Counter()
        self.beginningsset: set                           = set()
        self.endingsset: set                              = set()
        self.spatial_sum: Dict[str,int]                   = {}
        self._sample_weights_cum: Optional[List[float]]  = None
        self._sample_weights_total: float                 = 0.0
        self._process()

    def _tok(self, text: str) -> List[str]:
        t = text.lower() if self.lowercase else text
        return [w for w in t.split() if w]

    def _process(self) -> None:
        ds = load_dataset(self.dataset_name,
                          *([self.config_name] if self.config_name else []),
                          streaming=self.streaming)
        splits = self.split_names or list(ds.keys())
        if not self.text_fields:
            self.text_fields = [k for k,v in ds[splits[0]].features.items()
                                 if getattr(v,"dtype",None)=="string"]
        orderedpool: List[str] = []
        for split in splits:
            if split not in ds: continue
            for idx, ex in enumerate(ds[split]):
                if self.max_per_split and idx >= self.max_per_split: break
                raw  = " ".join(_extract(ex,f) for f in self.text_fields).strip()
                toks = self._tok(raw)
                if len(toks) < self.minlen: continue
                first, last = toks[0], toks[-1]
                if (self._begincounts[first] >= self.boundaryquota or
                        self._endcounts[last]  >= self.boundaryquota): continue
                self._begincounts[first] += 1
                self._endcounts[last]    += 1
                rec_idx = len(self.sentences)
                self.sentences.append(toks); self.tokens.extend(toks)
                for pos in range(1, len(toks)-1):
                    w = toks[pos]
                    orderedpool.append(w)
                    self.middlecorr.setdefault(w,[]).append((rec_idx,pos))
        self._orderedpool = orderedpool
        seen: set = set()
        for w in orderedpool:
            if w not in seen: seen.add(w); self.middlepool.append(w)
        token_positions: Dict[str,List[int]] = {}
        for i,w in enumerate(orderedpool):
            token_positions.setdefault(w,[]).append(i)
        for w,pos in token_positions.items():
            self.spatial_sum[w] = pos[-1]-pos[0]
        cum, running = [], 0.0
        for w in self.middlepool:
            running += float(self.spatial_sum.get(w,0))+1.0; cum.append(running)
        self._sample_weights_cum = cum; self._sample_weights_total = running
        self.beginningsset = {s[0]  for s in self.sentences}
        self.endingsset    = {s[-1] for s in self.sentences}
        if not self.sentences:
            raise ValueError(f"No sentences survived for {self.dataset_name!r}.")

    def tocorpus(self) -> str: return " ".join(self.tokens)
    def isbeginning(self, w): return w in self.beginningsset
    def isnaturalending(self, w): return w in self.endingsset

    def sample_correlated(self, anchor=None, rng=None) -> Dict:
        rng = rng or random
        if anchor and anchor in self.middlecorr: key = anchor
        else:
            cum, total = self._sample_weights_cum, self._sample_weights_total
            target = rng.random()*total; lo, hi = 0, len(self.middlepool)-1
            while lo < hi:
                mid = (lo+hi)//2
                if cum[mid] <= target: lo = mid+1
                else: hi = mid
            key = self.middlepool[lo]
        occs = self.middlecorr.get(key,[])
        if not occs: return {"token": key, "tail": [key]}
        rec_idx, pos = occs[rng.randrange(len(occs))]
        toks = self.sentences[rec_idx]
        return {"token": toks[pos], "tail": toks[pos:]}

    def popped_siblings(self, token: str, max_siblings: int = 8) -> List[str]:
        occs = self.middlecorr.get(token,[])
        if len(occs) < 2: return []
        pool, n = self._orderedpool, len(self._orderedpool)
        all_pos = [i for i,w in enumerate(pool) if w == token]
        siblings, seen = [], {token, ""}
        for pos in all_pos[1:]:
            nxt = pool[pos+1] if pos+1 < n else ""
            if nxt and nxt not in seen: seen.add(nxt); siblings.append(nxt)
            if len(siblings) >= max_siblings: break
        return siblings


# ══════════════════════════════════════════════════════════════════════════════
# 5. CPD + CONTEXT INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_cpd(corpus: str, ngram_n: int = 2, lidstone_gamma: float = 0.1):
    from nltk.probability import (ConditionalFreqDist, ConditionalProbDist,
                                   LidstoneProbDist)
    tokens = corpus.lower().split()
    if not tokens: raise ValueError("Empty corpus.")
    n = max(2, ngram_n); padded = [""]*( n-1)+tokens+[""]
    cfd = ConditionalFreqDist()
    for ng in zip(*[padded[i:] for i in range(n)]):
        ctx, word = ng[:-1], ng[-1]; cfd[ctx][word] += 1
    vocab = set(tokens)|{""}; bins = max(1, len(vocab))
    cpd = ConditionalProbDist(
        cfd, lambda fd: LidstoneProbDist(fd, gamma=lidstone_gamma, bins=bins))
    return cpd, vocab, tokens

def build_context_index(vocab, cpd, tokens):
    try:
        import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import ContextZoneIndex
        return ContextZoneIndex(vocab, cpd, Counter(tokens))
    except Exception as e:
        print(f"[context_index] unavailable ({e}); zone layers will be uniform.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 6. LAYER MODULES  — GRAD-1: every forward() uses _layer_dict()
# ══════════════════════════════════════════════════════════════════════════════

class L0_RawDist(nn.Module):
    def __init__(self, rep_penalty=1.13):
        super().__init__(); self.rep_penalty = nn.Parameter(_t(rep_penalty))

    def forward(self, dist, history):
        pen = self.rep_penalty.clamp(min=1.0)
        raw = [(s, max(1e-12, float(dist.prob(s))) / (float(pen)**history[s] if history[s] > 0 else 1))
               for s in dist.samples() if s]
        if not raw: return [], {}
        # keep pen in graph: multiply raw probs by param-derived scalar
        base = _t([p for _, p in raw])
        pt   = _norm(base * (pen / pen.detach()))   # identity value, keeps grad_fn
        words = [w for w, _ in raw]
        return list(zip(words, pt.tolist())), _layer_dict("L0_RAW_DIST", words, pt)

class L1_TempScaled(nn.Module):
    def __init__(self, temperature=4.3):
        super().__init__(); self.temperature = nn.Parameter(_t(temperature))

    def forward(self, pairs):
        T  = self.temperature.clamp(min=1e-3)
        pt = _norm(_t([p for _, p in pairs]).pow(1.0 / T))
        words = [w for w, _ in pairs]
        return list(zip(words, pt.tolist())), _layer_dict("L1_TEMP_SCALED", words, pt)

class L2_InsightPenalty(nn.Module):
    def __init__(self, insight_penalty=3.95):
        super().__init__(); self.insight_penalty = nn.Parameter(_t(insight_penalty))

    def forward(self, pairs):
        s    = self.insight_penalty.clamp(min=0.0)
        pt   = _t([p for _, p in pairs]); mean = pt.mean().clamp(min=1e-30)
        pen  = _norm((pt / (1.0 + s*(pt-mean).clamp(min=0)/mean)).clamp(min=1e-12))
        words = [w for w, _ in pairs]
        return list(zip(words, pen.tolist())), _layer_dict("L2_INSIGHT", words, pen)

class L3_TopKTopP(nn.Module):
    def __init__(self, top_k=100, top_p=1.0):
        super().__init__()
        self.register_buffer("top_k_buf", torch.tensor(top_k, dtype=torch.int64))
        self.top_p = nn.Parameter(_t(top_p))

    def forward(self, pairs):
        k, p_th = int(self.top_k_buf), float(self.top_p.clamp(1e-3, 1.0))
        kept, cum = [], 0.0
        for w, p in pairs[:k]:
            kept.append((w, p)); cum += p
            if cum >= p_th: break
        # keep top_p param in graph via a differentiable mask scale
        scale = self.top_p / self.top_p.detach()
        pt    = _norm(_t([p for _, p in kept]) * scale)
        words = [w for w, _ in kept]
        return list(zip(words, pt.tolist())), _layer_dict("L3_TOPK_TOPP", words, pt)

class _ZoneBase(nn.Module):
    def __init__(self, name, sigma, floor):
        super().__init__(); self._name = name
        self.sigma = nn.Parameter(_t(sigma))
        self.floor = nn.Parameter(_t(floor))

    def _gauss(self, zone_set, cands) -> torch.Tensor:
        n = len(cands)
        if not n: return torch.zeros(0, dtype=torch.float64)
        sig = self.sigma.clamp(min=1e-6); fl = self.floor.clamp(0, 1-1e-6)
        idx  = torch.arange(n, dtype=torch.float64) / max(1, n-1)
        ranks = [i/max(1,n-1) for i,(w,_) in enumerate(cands) if w in zone_set]
        ctr  = float(torch.tensor(ranks).mean()) if ranks else 0.0
        return _norm(fl + (1-fl) * torch.exp(-0.5*((idx-ctr)/sig)**2))

    def _ld(self, w, cands):   # GRAD-1 via _layer_dict
        return _layer_dict(self._name, [x for x,_ in cands], w)

class L4_ZoneFreq(_ZoneBase):
    def __init__(self, sigma=0.50, floor=0.05): super().__init__("L4_ZONE_FREQ",sigma,floor)
    def forward(self, cands, prompt_words, freq_zones, token_freq):
        high = set(freq_zones.get("high",[])); mid_th = 3
        key  = ("high" if any(w in high for w in prompt_words) else
                "low"  if all(token_freq.get(w,0)<mid_th for w in prompt_words) else "mid")
        return self._ld(self._gauss(set(freq_zones.get(key,[])), cands), cands)

class L5_ZoneAlpha(_ZoneBase):
    def __init__(self, sigma=0.40, floor=0.05): super().__init__("L5_ZONE_ALPHA",sigma,floor)
    def forward(self, cands, prompt_words, alpha_zones):
        zone = set().union(*(alpha_zones.get(w[0],[]) for w in prompt_words if w))
        return self._ld(self._gauss(zone, cands), cands)

class L6_ZoneBigram(_ZoneBase):
    def __init__(self, sigma=0.25, floor=0.04): super().__init__("L6_ZONE_BIGRAM",sigma,floor)
    def forward(self, cands, prompt_words, ngram_zones):
        zone = set().union(*(ngram_zones.get((prompt_words[i],prompt_words[i+1]),[])
                             for i in range(len(prompt_words)-1)))
        return self._ld(self._gauss(zone, cands), cands)

class L7_ZoneTrigram(_ZoneBase):
    def __init__(self, sigma=0.20, floor=0.03): super().__init__("L7_ZONE_TRIGRAM",sigma,floor)
    def forward(self, cands, ctx, ngram_zones):
        cl  = list(ctx)
        key = tuple(cl[-2:]) if len(cl)>=2 else (tuple(cl[-1:]) if cl else ())
        return self._ld(self._gauss(set(ngram_zones.get(key,[])), cands), cands)

class L8_ZoneCharTrig(_ZoneBase):
    def __init__(self, sigma=0.35, floor=0.04): super().__init__("L8_ZONE_CHAR_TRIG",sigma,floor)
    def forward(self, cands, prompt_words, char_idx):
        tgs  = set().union(*(_char_trigrams(w) for w in prompt_words))
        zone = set().union(*(char_idx.get(t,set()) for t in tgs))
        return self._ld(self._gauss(zone, cands), cands)

class L9_ZoneLatent(_ZoneBase):
    def __init__(self, sigma=0.30, floor=0.04): super().__init__("L9_ZONE_LATENT",sigma,floor)
    def forward(self, cands, prompt_words, latent_sorted_keys, latent_bos_data):
        q_key = "q0"
        if latent_sorted_keys and prompt_words:
            for w in prompt_words:
                for k in latent_sorted_keys:
                    if w in latent_bos_data.get(k,set()): q_key = k; break
        return self._ld(self._gauss(set(latent_bos_data.get(q_key,[])), cands), cands)

class L10_History(nn.Module):
    def __init__(self, smoothing=1.0):
        super().__init__(); self.smoothing = nn.Parameter(_t(smoothing))

    def forward(self, cands, history):
        s  = self.smoothing.clamp(min=1e-6)
        pt = _norm(_t([max(1e-12, 1.0/(1.0+float(s)*history[w])) for w,_ in cands])
                   * (s/s.detach()))   # keep param in graph
        words = [w for w,_ in cands]
        return _layer_dict("L10_HISTORY", words, pt)

class L11_TensorBlend(nn.Module):
    def __init__(self, init_weights=None):
        super().__init__()
        n = 7
        w = torch.ones(n, dtype=torch.float64)/n if init_weights is None else _t(init_weights)
        self.weights = nn.Parameter(w)

    def forward(self, zone_layers, cands):
        wt  = F.softmax(self.weights.float(), dim=0).double()
        n   = len(cands)
        # GRAD-1: use probs_t if available so blend stays in graph
        stack = torch.stack([
            _p(l) if len(l.get("probs",[])) == n
            else F.pad(_p(l), (0, n-len(l.get("probs",[]))))
            for l in zone_layers
        ])
        blended = _norm((stack * wt.unsqueeze(1)).sum(0))
        words   = [w for w,_ in cands]
        return _layer_dict("L11_TENSOR_BLEND", words, blended)

class L12_Final(nn.Module):
    def __init__(self, blend_alpha=0.5):
        super().__init__(); self.blend_alpha = nn.Parameter(_t(blend_alpha))

    def forward(self, cands, L11):
        a       = self.blend_alpha.clamp(1e-6, 1-1e-6)
        raw     = _norm(_t([p for _,p in cands]))
        l11     = _p(L11)   # GRAD-2: live tensor from L11
        blended = _norm(raw**(1-a) * l11**a)
        words   = [w for w,_ in cands]
        return list(zip(words, blended.tolist())), _layer_dict("L12_FINAL", words, blended)

class L13_CtxReqPos(nn.Module):
    def __init__(self, sigma=0.30, floor=0.04):
        super().__init__()
        self.sigma = nn.Parameter(_t(sigma)); self.floor = nn.Parameter(_t(floor))

    def forward(self, cands, draw_pos, stream_len):
        n        = len(cands)
        sig      = self.sigma.clamp(min=1e-6); fl = self.floor.clamp(0, 1-1e-6)
        norm_pos = (draw_pos % max(1,stream_len)) / max(1, stream_len-1)
        idx      = torch.arange(n, dtype=torch.float64) / max(1, n-1)
        w        = _norm(fl + (1-fl)*torch.exp(-0.5*((idx-norm_pos)/sig)**2))
        words    = [x for x,_ in cands]
        return _layer_dict("L13_CTX_REQ_POS", words, w,
                           draw_pos=draw_pos, stream_len=stream_len)

class L14_LockedStateIndex(nn.Module):
    LAYER_NAME = "L14_LOCKED_STATE_INDEX"

    def __init__(self, sigma=0.25, floor=0.03, lock_strength=1.0):
        super().__init__()
        self.sigma         = nn.Parameter(_t(sigma))
        self.floor         = nn.Parameter(_t(floor))
        self.lock_strength = nn.Parameter(_t(lock_strength))
        self._locked: Dict[Tuple,str] = {}
        self._observed: set           = set()

    @property
    def n_locked(self):  return len(self._locked)
    @property
    def n_missing(self): return len(self._observed - set(self._locked))
    def reset_state(self): self._locked.clear(); self._observed.clear()

    def commit(self, key, token):
        if key and key not in self._locked: self._locked[key] = token

    @staticmethod
    def key_from_ctx(ctx): return tuple(w for w in ctx if w) or ()

    def forward(self, cands, ctx, draw_pos, stream_len):
        n = len(cands); words = [w for w,_ in cands]
        fl = self.floor.clamp(0, 1-1e-6); lw = self.lock_strength.clamp(0,1)
        sig = self.sigma.clamp(min=1e-6); key = self.key_from_ctx(ctx)
        if key and key in self._locked:
            tok  = self._locked[key]
            base = torch.full((n,), float(fl), dtype=torch.float64)
            if tok in words:
                base[words.index(tok)] = float(fl) + float(lw)*(1-float(fl))
            # keep params in graph
            wts = _norm(base * (lw/lw.detach()))
        else:
            if key: self._observed.add(key)
            miss     = self.n_missing; sl = max(1, stream_len)
            norm_pos = ((draw_pos+miss)%sl) / max(1, sl-1)
            idx      = torch.arange(n, dtype=torch.float64) / max(1, n-1)
            wts      = _norm(fl + (1-fl)*torch.exp(-0.5*((idx-norm_pos)/sig)**2))
        return _layer_dict(self.LAYER_NAME, words, wts)


# ══════════════════════════════════════════════════════════════════════════════
# 7. IsomorphismPipeline
# ══════════════════════════════════════════════════════════════════════════════

class IsomorphismPipeline(nn.Module):
    def __init__(self, cpd, context_index, vocab, ngram_n=2, temperature=4.3,
                 top_k=100, top_p=1.0, rep_penalty=1.13, insight_penalty=3.95,
                 l12_blend_alpha=0.5, l13_sigma=0.30, l13_floor=0.04, **kw):
        super().__init__()
        self.cpd = cpd; self.ctx_idx = context_index; self.vocab = set(vocab)
        self.ngram_n = max(2,int(ngram_n)); self.context_window = self.ngram_n-1
        self.history: Counter = Counter()
        self._step_val = 0; self._pos = 0; self._stream: List[int] = []
        self._char_trig_index = getattr(context_index,"_trig_index",{}) if context_index else {}
        self._step_loss: torch.Tensor = torch.zeros(1, dtype=torch.float64)

        self.l0  = L0_RawDist(rep_penalty);   self.l1  = L1_TempScaled(temperature)
        self.l2  = L2_InsightPenalty(insight_penalty); self.l3 = L3_TopKTopP(top_k, top_p)
        self.l4  = L4_ZoneFreq();  self.l5  = L5_ZoneAlpha()
        self.l6  = L6_ZoneBigram(); self.l7 = L7_ZoneTrigram()
        self.l8  = L8_ZoneCharTrig(); self.l9 = L9_ZoneLatent()
        self.l10 = L10_History(); self.l11 = L11_TensorBlend()
        self.l12 = L12_Final(l12_blend_alpha)
        self.l13 = L13_CtxReqPos(l13_sigma, l13_floor)
        self.frames: List = []

    def _dist_for_ctx(self, ctx):
        for cut in range(len(ctx), 0, -1):
            key = ("",)*(self.context_window-cut)+ctx[-cut:]
            try:
                d = self.cpd[key]
                if list(d.samples()): return d
            except Exception: pass
        try:
            d = self.cpd[("",)*self.context_window]
            if list(d.samples()): return d
        except Exception: pass
        return None

    def seed_stream(self, stream): self._stream = list(stream); self._pos = 0

    def _make_draw_fn(self, stream, digits_per_sample=3, seed=None):
        if stream is not None: self.seed_stream(stream)
        if self._stream:
            pos=[self._pos]; dps=max(1,digits_per_sample); sl=len(self._stream)
            def _draw():
                val=0
                for _ in range(dps): val=val*26+self._stream[pos[0]%sl]; pos[0]=(pos[0]+1)%sl
                self._pos=pos[0]; return val/(26**dps)
                
            return _draw
        return random.Random(seed).random

    def _run_assertions(self, layer_sequence: List[Dict],
                        zone_layers: List[Dict]) -> torch.Tensor:
        """GRAD-3: collect into list → stack → sum. Graph never severed."""
        by_name = {l.get("name",""): l for l in layer_sequence if l.get("name")}
        losses:  List[torch.Tensor] = []
        prev_p:  Optional[torch.Tensor] = None

        for layer in layer_sequence:
            name = layer.get("name", "")
            p    = _p(layer)   # GRAD-2: live tensor
            pb   = prev_p if prev_p is not None else p

            blend_ref = None
            if name == "L11_TENSOR_BLEND":
                zps = [_p(by_name[k]) for k in
                       ("L4_ZONE_FREQ","L5_ZONE_ALPHA","L6_ZONE_BIGRAM",
                        "L7_ZONE_TRIGRAM","L8_ZONE_CHAR_TRIG","L9_ZONE_LATENT",
                        "L10_HISTORY") if k in by_name]
                blend_ref = _norm(torch.stack(zps).mean(0)) if zps else None
            elif name == "L12_FINAL":
                blend_ref = (_p(by_name["L11_TENSOR_BLEND"])
                             if "L11_TENSOR_BLEND" in by_name else None)

            losses.append(layer_loss(
                name, pb, p,
                blend_ref  = blend_ref,
                draw_pos   = layer.get("draw_pos",   self.ctx_idx), 
                stream_len = layer.get("stream_len", max(1, len(self._stream))),
            ))#custom, obvious change
            prev_p = p

        return torch.stack(losses).sum() if losses else torch.zeros(1, dtype=torch.float64)

    def step(self, ctx: deque, prompt_words: List[str], draw: float):
        dist = self._dist_for_ctx(tuple(ctx))
        if dist is None: return None
        L0_pairs, L0 = self.l0(dist, self.history)
        if not L0_pairs: return None
        L1_pairs, L1 = self.l1(L0_pairs)
        L2_pairs, L2 = self.l2(L1_pairs)
        L3_pairs, L3 = self.l3(L2_pairs)
        if not L3_pairs: return None

        ci = self.ctx_idx
        if ci is None:
            flat = _norm(torch.ones(len(L3_pairs), dtype=torch.float64))
            zone_layers = [_layer_dict(nm, [w for w,_ in L3_pairs], flat.clone())
                           for nm in ("L4_ZONE_FREQ","L5_ZONE_ALPHA","L6_ZONE_BIGRAM",
                                      "L7_ZONE_TRIGRAM","L8_ZONE_CHAR_TRIG","L9_ZONE_LATENT")]
        else:
            zone_layers = [
                self.l4(L3_pairs, prompt_words, ci.freq_zones, ci.token_freq),
                self.l5(L3_pairs, prompt_words, ci.alpha_zones),
                self.l6(L3_pairs, prompt_words, ci.ngram_zones),
                self.l7(L3_pairs, ctx, ci.ngram_zones),
                self.l8(L3_pairs, prompt_words, self._char_trig_index),
                self.l9(L3_pairs, prompt_words, ci.latent_sorted_keys, ci.latent_bos_data),
            ]

        L10 = self.l10(L3_pairs, self.history)
        L11 = self.l11(zone_layers+[L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)
        sl  = max(1, len(self._stream))
        L13 = self.l13(L3_pairs, self._pos, sl)

        all_layers = [L0,L1,L2,L3]+zone_layers+[L10,L11,L12,L13]
        self._step_loss = self._run_assertions(all_layers, zone_layers)

        l12m = dict(L12_pairs)
        l13m = dict(zip(L13["words"], L13["probs"].tolist()))
        fl   = 1e-12
        blended = [(w, math.sqrt(max(fl,l12m.get(w,fl))*max(fl,l13m.get(w,fl)))) for w in l12m]
        bt = sum(p for _,p in blended)
        blended = [(w,p/bt) for w,p in blended] if bt else blended
        unseen  = [(w,p) for w,p in blended if not self.history[w]]
        pool    = unseen or blended; t = sum(p for _,p in pool)
        pool    = [(w,p/t) for w,p in pool] if t else pool
        chosen, cum = pool[-1][0], 0.0
        for w,p in pool:
            cum += p
            if draw < cum: chosen=w; break
        self.history[chosen] += 1
        nxt = (self._pos+(self._pos%sl))%sl; self._pos=nxt; self._step_val+=1
        return {"chosen":chosen,"draw_pos":self._pos,"next_draw_pos":nxt,"layers":all_layers}

    def generate(self, prompt: str, n_words: int, draw_fn, **kw):
        toks = [w.lower() for w in prompt.split() if w.isalpha()]
        init = (toks[-self.context_window:] if len(toks)>=self.context_window
                else [""]*( self.context_window-len(toks))+toks)
        ctx  = deque(init, maxlen=self.context_window)
        words, iters = [], 0; max_iters = max(1,n_words)*80
        while len(words)<n_words and iters<max_iters:
            iters += 1; frame = self.step(ctx, toks, draw_fn())
            if frame is None: ctx.clear(); ctx.extend([""]*self.context_window)
            else: ctx.append(frame["chosen"]); words.append(frame["chosen"])
        return words[:n_words]

    def generate_text(self, prompt: str, n_words: int, *, stream=None,
                      digits_per_sample=3, seed=None, capitalise=True) -> str:
        draw_fn   = self._make_draw_fn(stream, digits_per_sample, seed)
        words     = self.generate(prompt, n_words, draw_fn)
        all_words = prompt.strip().split()+words
        if not capitalise: return " ".join(all_words)
        out, cap = [], True
        for w in all_words:
            out.append(w.capitalize() if cap else w)
            cap = bool(w.rstrip("\"'")[-1:] in {".","!","?"})
        return " ".join(out)


class LockedIsomorphismPipeline(IsomorphismPipeline):
    def __init__(self, *args, l14_sigma=0.25, l14_floor=0.03,
                 l14_lock_strength=1.0, l14_blend_alpha=0.5, **kw):
        super().__init__(*args, **kw)
        self.l14             = L14_LockedStateIndex(l14_sigma, l14_floor, l14_lock_strength)
        self.l14_blend_alpha = nn.Parameter(_t(l14_blend_alpha))

    def seed_stream(self, stream):
        super().seed_stream(stream); self.l14.reset_state()

    def step(self, ctx, prompt_words, draw):
        dist = self._dist_for_ctx(tuple(ctx))
        if dist is None: return None
        L0_pairs, L0 = self.l0(dist, self.history)
        if not L0_pairs: return None
        L1_pairs, L1 = self.l1(L0_pairs)
        L2_pairs, L2 = self.l2(L1_pairs)
        L3_pairs, L3 = self.l3(L2_pairs)
        if not L3_pairs: return None

        ci = self.ctx_idx
        if ci is None:
            flat = _norm(torch.ones(len(L3_pairs), dtype=torch.float64))
            zone_layers = [_layer_dict(nm, [w for w,_ in L3_pairs], flat.clone())
                           for nm in ("L4_ZONE_FREQ","L5_ZONE_ALPHA","L6_ZONE_BIGRAM",
                                      "L7_ZONE_TRIGRAM","L8_ZONE_CHAR_TRIG","L9_ZONE_LATENT")]
        else:
            zone_layers = [
                self.l4(L3_pairs, prompt_words, ci.freq_zones, ci.token_freq),
                self.l5(L3_pairs, prompt_words, ci.alpha_zones),
                self.l6(L3_pairs, prompt_words, ci.ngram_zones),
                self.l7(L3_pairs, ctx, ci.ngram_zones),
                self.l8(L3_pairs, prompt_words, self._char_trig_index),
                self.l9(L3_pairs, prompt_words, ci.latent_sorted_keys, ci.latent_bos_data),
            ]

        L10 = self.l10(L3_pairs, self.history)
        L11 = self.l11(zone_layers+[L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)
        sl  = max(1, len(self._stream))
        L13 = self.l13(L3_pairs, self._pos, sl)
        L14 = self.l14(L3_pairs, ctx, self._pos, sl)

        all_layers = [L0,L1,L2,L3]+zone_layers+[L10,L11,L12,L13,L14]
        self._step_loss = self._run_assertions(all_layers, zone_layers)

        a    = float(self.l14_blend_alpha.clamp(1e-6,1-1e-6)); fl = 1e-12
        l12m = dict(L12_pairs)
        l13m = dict(zip(L13["words"], L13["probs"].tolist()))
        l14m = dict(zip(L14["words"], L14["probs"].tolist()))
        blended = [(w, (math.sqrt(max(fl,l12m.get(w,fl))*max(fl,l13m.get(w,fl)))**(1-a)) *
                      (max(fl,l14m.get(w,fl))**a)) for w in l12m]
        bt = sum(p for _,p in blended)
        blended = [(w,p/bt) for w,p in blended] if bt else blended
        unseen  = [(w,p) for w,p in blended if not self.history[w]]
        pool    = unseen or blended; t = sum(p for _,p in pool)
        pool    = [(w,p/t) for w,p in pool] if t else pool
        chosen, cum = (pool[-1][0] if pool else ""), 0.0
        for w,p in pool:
            cum += p
            if draw < cum: chosen=w; break
        key = L14_LockedStateIndex.key_from_ctx(ctx)
        if key and chosen: self.l14.commit(key, chosen)
        self.history[chosen] += 1
        nxt = (self._pos+(self._pos%sl))%sl; self._pos=nxt; self._step_val+=1
        return {"chosen":chosen,"draw_pos":self._pos,"next_draw_pos":nxt,"layers":all_layers}


# ══════════════════════════════════════════════════════════════════════════════
# 8. AUTOMORPHISM TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class AutomorphismTrainer:
    def __init__(self, pipeline: nn.Module, n_cands: int = 100,
                 lr: float = 1e-3, hidden: int = 128, weight_decay: float = 1e-4):
        self.pipe = pipeline
        self.head = PipelineAutomorphismHead(n_cands=n_cands, hidden=hidden)
        self.net  = self.head.net
        pipe_params = [p for p in pipeline.parameters() if p.requires_grad]
        net_params  = list(self.net.parameters())
        self.opt    = torch.optim.Adam(
            pipe_params+net_params, lr=lr, weight_decay=weight_decay)
        self.log: List[Dict] = []

    def warmup(self, steps: int = 200, batch: int = 8, log_every: int = 50):
        print(f"[warmup] {steps} steps …")
        for s in range(1, steps+1):
            x = torch.randn(batch, self.net.n, dtype=torch.float32)
            self.opt.zero_grad()
            loss, diag = self.net.automorphism_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            if s % log_every == 0:
                auto_mean = np.mean([v for k,v in diag.items() if "auto_loss" in k])
                print(f"  step {s:4d}  total={diag['total']:.5f}  "
                      f"auto={auto_mean:.5f}  T={float(self.net.log_temp.exp()):.4f}")
        print("[warmup] done.")

    def step(self, prompt: str, n_words: int = 30,
             seed: Optional[int] = None) -> float:
        self.opt.zero_grad()
        draw_fn = self.pipe._make_draw_fn(None, seed=seed)
        toks    = [w.lower() for w in prompt.split() if w.isalpha()]
        cw      = self.pipe.context_window
        init    = toks[-cw:] if len(toks)>=cw else [""]*( cw-len(toks))+toks
        ctx     = deque(init, maxlen=cw)
        losses: List[torch.Tensor] = []   # GRAD-3

        for _ in range(n_words):
            with torch.enable_grad():
                frame = self.pipe.step(ctx, toks, draw_fn())
            if frame is None: ctx.clear(); ctx.extend([""]*cw); continue
            ctx.append(frame["chosen"])
            losses.append(self.pipe._step_loss + self.head.frame_loss(frame))

        if not losses:
            return 0.0

        total = torch.stack(losses).sum()
        total.backward()   # BUG-4: always backward
        torch.nn.utils.clip_grad_norm_(
            list(self.pipe.parameters())+list(self.net.parameters()), 1.0)
        self.opt.step()
        return float(total.detach())

    def run(self, n_steps: int = 100, prompt: str = "the quick brown fox",
            n_words: int = 30, seed: int = 42, log_every: int = 10,
            patience: int = 10, tol: float = 1e-6) -> List[Dict]:
        best, no_imp = float("inf"), 0
        for s in range(1, n_steps+1):
            loss  = self.step(prompt, n_words=n_words, seed=seed)
            l1t   = float(self.pipe.l1.temperature.detach())  if hasattr(self.pipe.l1,  "temperature")  else 0.
            l3tp  = float(self.pipe.l3.top_p.detach())        if hasattr(self.pipe.l3,  "top_p")        else 0.
            l12a  = float(self.pipe.l12.blend_alpha.detach()) if hasattr(self.pipe.l12, "blend_alpha")  else 0.
            entry = {"step":s,"loss":loss,"T":float(self.net.log_temp.exp().detach()),
                     "temp":l1t,"top_p":l3tp,"a12":l12a}
            self.log.append(entry)
            if s % log_every == 0:
                print(f"[step {s:4d}] loss={loss:.6f}  net_T={entry['T']:.4f}  "
                      f"pipe_T={entry['temp']:.4f}  top_p={entry['top_p']:.4f}  "
                      f"a12={entry['a12']:.4f}")
            if loss < best-tol: best=loss; no_imp=0
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"[trainer] early stop at step {s}"); break
        return self.log

    def report(self):
        if not self.log: print("[trainer] no log."); return
        f, l = self.log[0], self.log[-1]
        print("\n── AutomorphismTrainer Report ──────────────────────────────────")
        print(f"  Steps     : {len(self.log)}")
        print(f"  Loss      : {f['loss']:.6f} → {l['loss']:.6f}  (Δ {l['loss']-f['loss']:+.6f})")
        print(f"  net_T     : {f['T']:.4f} → {l['T']:.4f}")
        print(f"  pipe_T    : {f['temp']:.4f} → {l['temp']:.4f}")
        print(f"  top_p     : {f['top_p']:.4f} → {l['top_p']:.4f}")
        print(f"  blend_α12 : {f['a12']:.4f} → {l['a12']:.4f}")
        print("─────────────────────────────────────────────────────────────────\n")

    def check_fixed_point(self, n_samples: int = 16, tol: float = 0.05) -> Dict:
        results = {k: 0 for k in AA_RULES}
        for _ in range(n_samples):
            x = torch.randn(1, self.net.n, dtype=torch.float32)
            with torch.no_grad():
                p = self.net(x).squeeze(0)
            for k, v in R.audit(p.detach(), AA_RULES).items():
                results[k] += int(v)
        return {k: f"{v}/{n_samples}" for k,v in results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 9. PIPELINE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(dataset_name: str, config_name: Optional[str] = None,
                   text_fields: Optional[Sequence] = None, *, locked=True,
                   ngram_n=3, lidstone_gamma=0.1,
                   preprocessor_kw: Optional[Dict] = None,
                   pipeline_kw: Optional[Dict] = None,
                   ) -> Tuple["IsomorphismPipeline", Preprocessor]:
    pre = Preprocessor(dataset_name, config_name, text_fields,
                        **(preprocessor_kw or {}))
    
    cpd, vocab, tokens = build_cpd(pre.tocorpus(), ngram_n, lidstone_gamma)

    cpd, vocab, tokens = build_cpd(content, ngram_n, lidstone_gamma)
    ctx_idx = build_context_index(vocab, cpd, tokens)
    cls  = LockedIsomorphismPipeline if locked else IsomorphismPipeline
    pipe = cls(cpd, ctx_idx, vocab, ngram_n=ngram_n, **(pipeline_kw or {}))
    pipe.preprocessor = pre
    return pipe, pre

HF_DATASET_PRESETS: Dict[str,Dict] = {
    "squad":        {"text_fields":["question","context","answers.text"]},
    "imdb":         {"text_fields":["text"]},
    "wikitext":     {"config_name":"wikitext-2-raw-v1","text_fields":["text"]},
    "blended_skill_talk": {"text_fields":[
        lambda ex:" ".join(ex.get("free_messages",[]) or []),
        lambda ex:" ".join(ex.get("guided_messages",[]) or []),
    ]},
}

def build_pipeline_from_preset(preset_name: str, **kw):
    if preset_name not in HF_DATASET_PRESETS:
        raise KeyError(f"Unknown preset {preset_name!r}. "
                       f"Available: {sorted(HF_DATASET_PRESETS)}")
    spec = dict(HF_DATASET_PRESETS[preset_name])
    spec.setdefault("dataset_name", preset_name); spec.update(kw)
    return build_pipeline(**spec)


# ══════════════════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

"""
pi_automorphism_gradio.py
─────────────────────────
Gradio / HuggingFace Spaces front-end for pi_automorphism_net.py
Supports:
  • HuggingFace dataset presets  (squad / imdb / wikitext / blended_skill_talk)
  • Custom HF dataset name + optional config sub-name
  • Plain-text file upload  (used as corpus instead of HF dataset)
  • JSON config file upload  (overrides every pipeline / preprocessor parameter)
  • Full text generation + trainer warmup/run via the UI
"""

import json, os, tempfile, traceback
from pathlib import Path
from typing import Optional

import gradio as gr

# ── lazy-import the pipeline module ──────────────────────────────────────────
# Assumes pi_automorphism_net.py is in the same directory (or on PYTHONPATH).

# ═════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG  (all keys mirror build_pipeline / AutomorphismTrainer kwargs)
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG: dict = {
    # ── dataset ──────────────────────────────────────────────────────────────
    "preset":           "imdb",        # one of HF_DATASET_PRESETS keys or ""
    "dataset_name":     "",            # used when preset == ""
    "config_name":      "",            # HF sub-config, e.g. "wikitext-2-raw-v1"
    "text_fields":      [],            # list of field names; [] = auto-detect
    # ── preprocessor ─────────────────────────────────────────────────────────
    "max_per_split":    1000,
    "boundaryquota":    8,
    "minlen":           3,
    "streaming":        False,
    # ── pipeline ─────────────────────────────────────────────────────────────
    "locked":           True,
    "ngram_n":          3,
    "lidstone_gamma":   0.1,
    "temperature":      4.3,
    "top_k":            100,
    "top_p":            1.0,
    "rep_penalty":      1.13,
    "insight_penalty":  3.95,
    "l12_blend_alpha":  0.5,
    "l13_sigma":        0.30,
    "l13_floor":        0.04,
    # ── trainer ──────────────────────────────────────────────────────────────
    "n_cands":          100,
    "lr":               5e-4,
    "hidden":           128,
    "weight_decay":     1e-4,
    # ── warmup ───────────────────────────────────────────────────────────────
    "warmup_steps":     50,
    "warmup_batch":     8,
    "warmup_log_every": 25,
    # ── training run ─────────────────────────────────────────────────────────
    "train_steps":      40,
    "train_patience":   8,
    "train_log_every":  10,
    # ── generation ───────────────────────────────────────────────────────────
    "n_words":          120,
    "seed":             42,
}

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_pipeline_cache: dict = {}   # key → (pipe, pre)

def _merge_config(json_file_path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if json_file_path:
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            cfg.update(overrides)
        except Exception as e:
            raise ValueError(f"Could not parse JSON config: {e}")
    return cfg


def _corpus_from_file(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def _build(cfg: dict, corpus_text: Optional[str] = None):
    """Build (or retrieve cached) pipeline from config + optional corpus."""
    cache_key = json.dumps(cfg, sort_keys=True, default=str) + str(bool(corpus_text))
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    pipe_kw = {k: cfg[k] for k in (
        "temperature", "top_k", "top_p", "rep_penalty",
        "insight_penalty", "l12_blend_alpha", "l13_sigma", "l13_floor",
    )}
    pre_kw = {k: cfg[k] for k in (
        "max_per_split", "boundaryquota", "minlen", "streaming",
    )}
    if cfg["max_per_split"]:
        pre_kw["max_per_split"] = int(cfg["max_per_split"])

    if corpus_text:
        # ── file-upload mode: skip HF download, build CPD directly ───────────
        cpd, vocab, tokens = build_cpd(
            corpus_text, cfg["ngram_n"], cfg["lidstone_gamma"]
        )
        ctx_idx = build_context_index(vocab, cpd, tokens)

        # Minimal stub so the pipeline has a .preprocessor attribute
        class _FakePre:
            sentences = [tokens]
            this_tokens = tokens
            def tocorpus(self): return " ".join(self.this_tokens)

        pre = _FakePre()
        pre.this_tokens = tokens
        cls = LockedIsomorphismPipeline if cfg["locked"] else IsomorphismPipeline
        pipe = cls(cpd, ctx_idx, vocab, ngram_n=cfg["ngram_n"], **pipe_kw)
        pipe.preprocessor = pre
    else:
        # ── HF dataset mode ──────────────────────────────────────────────────
        preset = cfg.get("preset", "").strip()
        if preset and preset in HF_DATASET_PRESETS:
            pipe, pre = build_pipeline_from_preset(
                preset,
                locked=cfg["locked"],
                ngram_n=cfg["ngram_n"],
                lidstone_gamma=cfg["lidstone_gamma"],
                preprocessor_kw=pre_kw,
                pipeline_kw=pipe_kw,
            )
        else:
            ds_name = cfg.get("dataset_name", "").strip()
            if not ds_name:
                raise ValueError("Provide a HF dataset name or upload a text file.")
            tf = cfg.get("text_fields") or None
            pipe, pre = build_pipeline(
                ds_name,
                config_name=cfg.get("config_name") or None,
                text_fields=tf,
                locked=cfg["locked"],
                ngram_n=cfg["ngram_n"],
                lidstone_gamma=cfg["lidstone_gamma"],
                preprocessor_kw=pre_kw,
                pipeline_kw=pipe_kw,
            )

    _pipeline_cache[cache_key] = (pipe, pre)
    return pipe, pre


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

def run_generate(
    # ── source ────────────────────────────────────────────────────────────────
    preset_choice: str,
    custom_dataset: str,
    hf_config_name: str,
    corpus_file,          # gr.File → filepath string or None
    json_config_file,     # gr.File → filepath string or None
    # ── pipeline params ──────────────────────────────────────────────────────
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    insight_penalty: float,
    l12_blend_alpha: float,
    locked: bool,
    ngram_n: int,
    max_per_split: int,
    # ── generation ───────────────────────────────────────────────────────────
    prompt: str,
    n_words: int,
    seed: int,
):
    try:
        cfg = _merge_config(json_config_file)

        # UI values override config-file values
        if preset_choice and preset_choice != "(custom)":
            cfg["preset"] = preset_choice
            cfg["dataset_name"] = ""
        else:
            cfg["preset"] = ""
            cfg["dataset_name"] = custom_dataset.strip()

        cfg.update({
            "config_name":     hf_config_name.strip(),
            "temperature":     temperature,
            "top_k":           int(top_k),
            "top_p":           top_p,
            "rep_penalty":     rep_penalty,
            "insight_penalty": insight_penalty,
            "l12_blend_alpha": l12_blend_alpha,
            "locked":          locked,
            "ngram_n":         int(ngram_n),
            "max_per_split":   int(max_per_split),
            "n_words":         int(n_words),
            "seed":            int(seed),
        })

        corpus_text = _corpus_from_file(corpus_file) if corpus_file else None
        pipe, _ = _build(cfg, corpus_text)

        text = pipe.generate_text(
            prompt, cfg["n_words"], seed=cfg["seed"], capitalise=True
        )
        return text, "✅ Done"
    except Exception:
        return "", f"❌ Error{traceback.format_exc()}"


def run_train_and_generate(
    preset_choice, custom_dataset, hf_config_name,
    corpus_file, json_config_file,
    temperature, top_k, top_p, rep_penalty, insight_penalty,
    l12_blend_alpha, locked, ngram_n, max_per_split,
    prompt, n_words, seed,
    warmup_steps, train_steps, lr,
):
    try:
        cfg = _merge_config(json_config_file)
        if preset_choice and preset_choice != "(custom)":
            cfg["preset"] = preset_choice; cfg["dataset_name"] = ""
        else:
            cfg["preset"] = ""; cfg["dataset_name"] = custom_dataset.strip()
        cfg.update({
            "config_name": hf_config_name.strip(),
            "temperature": temperature, "top_k": int(top_k), "top_p": top_p,
            "rep_penalty": rep_penalty, "insight_penalty": insight_penalty,
            "l12_blend_alpha": l12_blend_alpha, "locked": locked,
            "ngram_n": int(ngram_n), "max_per_split": int(max_per_split),
            "n_words": int(n_words), "seed": int(seed),
            "warmup_steps": int(warmup_steps), "train_steps": int(train_steps),
            "lr": float(lr),
        })

        corpus_text = _corpus_from_file(corpus_file) if corpus_file else None
        pipe, _ = _build(cfg, corpus_text)

        trainer = AutomorphismTrainer(
            pipe,
            n_cands=cfg["n_cands"],
            lr=cfg["lr"],
            hidden=cfg["hidden"],
            weight_decay=cfg["weight_decay"],
        )
        log_lines: list[str] = []
        def _pr(*a): log_lines.append(" ".join(str(x) for x in a))

        import builtins
        _orig_print = builtins.print
        builtins.print = _pr          # capture trainer output

        trainer.warmup(
            steps=cfg["warmup_steps"],
            batch=cfg["warmup_batch"],
            log_every=cfg["warmup_log_every"],
        )
        trainer.run(
            n_steps=cfg["train_steps"],
            prompt=prompt, n_words=int(n_words), seed=int(seed),
            log_every=cfg["train_log_every"],
            patience=cfg["train_patience"],
        )
        trainer.report()
        builtins.print = _orig_print  # restore

        text = pipe.generate_text(prompt, int(n_words), seed=int(seed), capitalise=True)
        train_log = "\n".join(log_lines)
        return text, train_log, "✅ Training complete"
    except Exception:
        import builtins; builtins.print = __builtins__["print"] if isinstance(__builtins__, dict) else __import__("builtins").print
        return "", "", f"❌ Error\n{traceback.format_exc()}"


def load_json_preview(json_file):
    if not json_file:
        return json.dumps(DEFAULT_CONFIG, indent=2)
    try:
        with open(json_file, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

PRESETS = ["(custom)"] + sorted(HF_DATASET_PRESETS.keys())

EXAMPLE_CONFIG = json.dumps({
    "preset":        "imdb",
    "max_per_split": 500,
    "ngram_n":       3,
    "temperature":   5.0,
    "top_k":         80,
    "top_p":         0.95,
    "locked":        True,
    "n_words":       80,
    "seed":          7,
}, indent=2)

with gr.Blocks(title="π-Automorphism Net", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🌀 π-Automorphism Text Generator\n"
        "Isomorphism pipeline with learnable layers, rule assertions, and an "
        "automorphism trainer.  Configure via the UI, a **JSON config file**, "
        "or a **plain-text corpus upload**."
    )

    # ── TOP ROW: source + config upload ──────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📂 Data Source")
            preset_dd = gr.Dropdown(
                choices=PRESETS, value="imdb", label="HuggingFace Preset",
            )
            custom_ds = gr.Textbox(
                label="Custom HF dataset name",
                placeholder="e.g.  wikitext  (ignored when preset ≠ custom)",
            )
            hf_cfg_name = gr.Textbox(
                label="HF config / sub-name",
                placeholder="e.g.  wikitext-2-raw-v1",
            )
            corpus_upload = gr.File(
                label="📄 Upload plain-text corpus  (.txt)  — overrides HF dataset",
                file_types=[".txt"],
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Config File")
            json_upload = gr.File(
                label="📋 Upload JSON config  (optional)",
                file_types=[".json"],
            )
            json_preview = gr.Code(
                value=json.dumps(DEFAULT_CONFIG, indent=2),
                language="json",
                label="Active config preview",
                lines=18,
            )
            json_upload.change(load_json_preview, json_upload, json_preview)

    # ── PIPELINE PARAMS ───────────────────────────────────────────────────────
    with gr.Accordion("🔧 Pipeline parameters", open=False):
        with gr.Row():
            temperature   = gr.Slider(0.1, 20.0, value=4.3,  step=0.1, label="Temperature")
            top_k         = gr.Slider(1,   500,  value=100,  step=1,   label="Top-K")
            top_p         = gr.Slider(0.0, 1.0,  value=1.0,  step=0.01,label="Top-P")
        with gr.Row():
            rep_penalty   = gr.Slider(1.0, 5.0,  value=1.13, step=0.01,label="Repetition penalty")
            insight_pen   = gr.Slider(0.0, 10.0, value=3.95, step=0.05,label="Insight penalty")
            l12_alpha     = gr.Slider(0.0, 1.0,  value=0.5,  step=0.01,label="L12 blend α")
        with gr.Row():
            locked_chk    = gr.Checkbox(value=True, label="Locked pipeline (L14)")
            ngram_n       = gr.Slider(2, 5, value=3, step=1, label="N-gram order")
            max_per_split = gr.Number(value=1000, label="Max examples per split", precision=0)

    # ── GENERATION TAB ────────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.TabItem("✍️ Generate"):
            with gr.Row():
                prompt_box = gr.Textbox(
                    value="tell me about yourself",
                    label="Prompt", lines=2, scale=3,
                )
                with gr.Column(scale=1):
                    n_words_sl = gr.Slider(10, 500, value=120, step=10, label="Words to generate")
                    seed_num   = gr.Number(value=42, label="Seed", precision=0)
            gen_btn    = gr.Button("🚀 Generate", variant="primary")
            gen_out    = gr.Textbox(label="Generated text", lines=10, interactive=False)
            gen_status = gr.Textbox(label="Status", lines=2, interactive=False)

            gen_btn.click(
                fn=run_generate,
                inputs=[
                    preset_dd, custom_ds, hf_cfg_name,
                    corpus_upload, json_upload,
                    temperature, top_k, top_p,
                    rep_penalty, insight_pen, l12_alpha,
                    locked_chk, ngram_n, max_per_split,
                    prompt_box, n_words_sl, seed_num,
                ],
                outputs=[gen_out, gen_status],
            )

        with gr.TabItem("🏋️ Train then Generate"):
            with gr.Row():
                warmup_steps_sl = gr.Slider(0, 500, value=50, step=10, label="Warmup steps")
                train_steps_sl  = gr.Slider(0, 500, value=40, step=10, label="Train steps")
                lr_num          = gr.Number(value=5e-4, label="Learning rate")
            train_btn    = gr.Button("🏋️ Train & Generate", variant="primary")
            train_out    = gr.Textbox(label="Generated text (post-training)", lines=8, interactive=False)
            train_log    = gr.Textbox(label="Training log", lines=14, interactive=False)
            train_status = gr.Textbox(label="Status", lines=2, interactive=False)

            train_btn.click(
                fn=run_train_and_generate,
                inputs=[
                    preset_dd, custom_ds, hf_cfg_name,
                    corpus_upload, json_upload,
                    temperature, top_k, top_p,
                    rep_penalty, insight_pen, l12_alpha,
                    locked_chk, ngram_n, max_per_split,
                    prompt_box, n_words_sl, seed_num,
                    warmup_steps_sl, train_steps_sl, lr_num,
                ],
                outputs=[train_out, train_log, train_status],
            )

        with gr.TabItem("📖 Config template"):
            gr.Markdown(
                "Copy this template, fill in your values, save as `config.json`, "
                "and upload it in the **Config File** panel."
            )
            gr.Code(value=EXAMPLE_CONFIG, language="json", label="config.json template")

    gr.Markdown(
        "---\n"
        "**Tips**  ·  "
        "Upload a `.txt` file to skip the HuggingFace download and use your own corpus.  "
        "A `.json` config overrides every parameter shown here.  "
        "Cached pipelines are reused within a session to avoid re-downloading."
    )

if __name__ == "__main__":
    demo.launch(share=False)
