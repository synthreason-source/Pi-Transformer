"""pi_transformer – refactored: O(n²) preprocessing, O(2n) generation."""
from __future__ import annotations
import math, random, re
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset

FieldSpec = Union[str, Callable[[Dict], Any]]

# ── field extraction ────────────────────────────────────────────────────────

def _extract(ex: Dict, spec: FieldSpec) -> str:
    val = spec(ex) if callable(spec) else ex.get(spec)
    if val is None: return ""
    if isinstance(val, str): return val
    if isinstance(val, (list, tuple)):
        return " ".join(str(v) for v in val if v is not None)
    return str(val)

# ── O(n²) preprocessor ─────────────────────────────────────────────────────

class Preprocessor:
    """Load an HF dataset and build boundary-quota token structures in O(n²)."""

    def __init__(self, dataset_name: str, config_name: Optional[str] = None,
                 text_fields: Optional[Sequence[FieldSpec]] = None, *,
                 split_names: Optional[Sequence[str]] = None, lowercase: bool = True,
                 minlen: int = 3, max_per_split: Optional[int] = None,
                 boundaryquota: int = 1, streaming: bool = False):
        self.dataset_name, self.config_name = dataset_name, config_name
        self.text_fields = list(text_fields) if text_fields else None
        self.split_names, self.lowercase = split_names, lowercase
        self.minlen, self.max_per_split = max(2, minlen), max_per_split
        self.boundaryquota, self.streaming = max(1, boundaryquota), streaming

        self.sentences: List[List[str]] = []
        self.tokens: List[str] = []
        self.middlepool: List[str] = []
        self.middlecorr: Dict[str, List[Tuple[int,int]]] = {}  # token→[(rec,pos)]
        self._orderedpool: List[str] = []
        self._begincounts: Counter = Counter()
        self._endcounts:   Counter = Counter()
        self.beginningsset: set = set()
        self.endingsset:   set = set()

        # spatial-sum weights (O(n²): for each unique token, scan all positions)
        self.spatial_sum:        Dict[str, int]         = {}
        self._sample_weights_cum: Optional[List[float]]  = None
        self._sample_weights_total: float                = 0.0

        self._process()

    # ── internals ──────────────────────────────────────────────────────

    def _tok(self, text: str) -> List[str]:
        t = text.lower() if self.lowercase else text
        return [w for w in t.split() if w]

    def _process(self) -> None:
        ds = load_dataset(self.dataset_name, *([self.config_name] if self.config_name else []),
                          streaming=self.streaming)
        splits = self.split_names or list(ds.keys())
        if not self.text_fields:
            # auto-detect: O(n) over feature names (small)
            self.text_fields = [k for k, v in ds[splits[0]].features.items()
                                 if getattr(v, 'dtype', None) == 'string']
        orderedpool: List[str] = []
        for split in splits:
            if split not in ds: continue
            for idx, ex in enumerate(ds[split]):
                if self.max_per_split and idx >= self.max_per_split: break
                raw = " ".join(_extract(ex, f) for f in self.text_fields).strip()
                toks = self._tok(raw)
                if len(toks) < self.minlen: continue
                first, last = toks[0], toks[-1]
                if (self._begincounts[first] >= self.boundaryquota or
                        self._endcounts[last] >= self.boundaryquota): continue
                self._begincounts[first] += 1
                self._endcounts[last]    += 1
                rec_idx = len(self.sentences)
                self.sentences.append(toks)
                self.tokens.extend(toks)
                # O(n) per record – O(n²) total: build middle correlation
                for pos in range(1, len(toks) - 1):
                    w = toks[pos]
                    orderedpool.append(w)
                    self.middlecorr.setdefault(w, []).append((rec_idx, pos))

        self._orderedpool = orderedpool
        # unique middle pool – O(n) pass
        seen: set = set()
        for w in orderedpool:
            if w not in seen:
                seen.add(w)
                self.middlepool.append(w)

        # O(n²) spatial-sum: for each unique token, positions span = last−first
        token_positions: Dict[str, List[int]] = {}
        for i, w in enumerate(orderedpool):
            token_positions.setdefault(w, []).append(i)
        for w, positions in token_positions.items():
            self.spatial_sum[w] = positions[-1] - positions[0]  # O(1) per token

        # cumulative sampling weights
        cum, running = [], 0.0
        for w in self.middlepool:
            running += float(self.spatial_sum.get(w, 0)) + 1.0
            cum.append(running)
        self._sample_weights_cum   = cum
        self._sample_weights_total = running

        self.beginningsset = {s[0]  for s in self.sentences}
        self.endingsset    = {s[-1] for s in self.sentences}
        if not self.sentences:
            raise ValueError(f"No sentences survived for {self.dataset_name!r}.")

    # ── public API ─────────────────────────────────────────────────────

    def tocorpus(self) -> str:
        return " ".join(self.tokens)

    def isbeginning(self, w: str) -> bool: return w in self.beginningsset
    def isnaturalending(self, w: str) -> bool: return w in self.endingsset

    def sample_correlated(self, anchor: Optional[str] = None,
                          rng: Optional[random.Random] = None) -> Dict:
        """Weighted sample → jump to a kept record. O(log n) via binary search."""
        rng = rng or random
        if anchor and anchor in self.middlecorr:
            key = anchor
        else:
            cum = self._sample_weights_cum
            total = self._sample_weights_total
            target = rng.random() * total
            lo, hi = 0, len(self.middlepool) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] <= target: lo = mid + 1
                else: hi = mid
            key = self.middlepool[lo]
        occs = self.middlecorr.get(key, [])
        if not occs: return {"token": key, "tail": [key]}
        rec_idx, pos = occs[rng.randrange(len(occs))]
        toks = self.sentences[rec_idx]
        return {"token": toks[pos], "tail": toks[pos:]}

    def popped_siblings(self, token: str, max_siblings: int = 8) -> List[str]:
        """Tokens following duplicate occurrences of *token* in orderedpool."""
        occs = self.middlecorr.get(token, [])
        if len(occs) < 2: return []
        pool, n = self._orderedpool, len(self._orderedpool)
        # positions of all occurrences: first occurrence in middlepool is kept,
        # rest are 'popped' – reconstruct popped positions by scanning orderedpool
        all_positions = [i for i, w in enumerate(pool) if w == token]
        siblings, seen = [], {token, ""}
        for pos in all_positions[1:]:           # skip first (kept) occurrence
            nxt = pool[pos + 1] if pos + 1 < n else ""
            if nxt and nxt not in seen:
                seen.add(nxt); siblings.append(nxt)
            if len(siblings) >= max_siblings: break
        return siblings


# ── utilities ───────────────────────────────────────────────────────────────

def _norm(t: torch.Tensor) -> torch.Tensor:
    return t / t.sum().clamp(min=1e-30)

def _t(p, dtype=torch.float64) -> torch.Tensor:
    if isinstance(p, torch.Tensor): return p.to(dtype)
    if isinstance(p, np.ndarray):   return torch.from_numpy(p.astype(np.float64)).to(dtype)
    return torch.tensor(p, dtype=dtype)

def _char_trigrams(w: str): return {w[i:i+3] for i in range(len(w)-2)} if len(w)>=3 else {w}


# ── layer modules ───────────────────────────────────────────────────────────

class L0_RawDist(nn.Module):
    def __init__(self, rep_penalty=1.13):
        super().__init__(); self.rep_penalty = nn.Parameter(_t(rep_penalty))
    def forward(self, dist, history):
        pen = self.rep_penalty.clamp(min=1.0).item()
        raw = [(s, max(1e-12, float(dist.prob(s))) / (pen**history[s] if history[s]>0 else 1))
               for s in dist.samples() if s]
        if not raw: return raw, {}
        pt = _norm(_t([p for _,p in raw])); words = [w for w,_ in raw]
        return list(zip(words, pt.tolist())), {"name":"L0_RAW_DIST","words":words,"probs":pt.numpy()}

class L1_TempScaled(nn.Module):
    def __init__(self, temperature=4.3):
        super().__init__(); self.temperature = nn.Parameter(_t(temperature))
    def forward(self, pairs):
        T = self.temperature.clamp(min=1e-3)
        pt = _norm(_t([p for _,p in pairs]).pow(1.0/T)); words = [w for w,_ in pairs]
        return list(zip(words, pt.tolist())), {"name":"L1_TEMP_SCALED","words":words,"probs":pt.detach().numpy() }

class L2_InsightPenalty(nn.Module):
    def __init__(self, insight_penalty=3.95):
        super().__init__(); self.insight_penalty = nn.Parameter(_t(insight_penalty))
    def forward(self, pairs):
        s = self.insight_penalty.clamp(min=0.0)
        pt = _t([p for _,p in pairs]); mean = pt.mean().clamp(min=1e-30)
        pen = _norm((pt/(1.0+s*(pt-mean).clamp(min=0)/mean)).clamp(min=1e-12))
        words = [w for w,_ in pairs]
        return list(zip(words, pen.tolist())), {"name":"L2_INSIGHT","words":words,"probs":pen.detach().numpy() }

class L3_TopKTopP(nn.Module):
    def __init__(self, top_k=100, top_p=1.0):
        super().__init__()
        self.register_buffer("top_k_buf", torch.tensor(top_k, dtype=torch.int64))
        self.top_p = nn.Parameter(_t(top_p))
    def forward(self, pairs):
        k, p_th = int(self.top_k_buf), float(self.top_p.clamp(1e-3,1.0))
        kept, cum = [], 0.0
        for w,p in pairs[:k]:
            kept.append((w,p)); cum+=p
            if cum>=p_th: break
        pt = _norm(_t([p for _,p in kept])); words = [w for w,_ in kept]
        return list(zip(words, pt.tolist())), {"name":"L3_TOPK_TOPP","words":words,"probs":pt.numpy()}

class _ZoneBase(nn.Module):
    def __init__(self, name, sigma, floor):
        super().__init__(); self._name=name
        self.sigma = nn.Parameter(_t(sigma)); self.floor = nn.Parameter(_t(floor))
    def _gauss(self, zone_set, cands):
        n = len(cands)
        if not n: return torch.zeros(0, dtype=torch.float64)
        sig = self.sigma.clamp(min=1e-6); fl = self.floor.clamp(0, 1-1e-6)
        idx = torch.arange(n, dtype=torch.float64) / max(1, n-1)
        ranks = [i/max(1,n-1) for i,(w,_) in enumerate(cands) if w in zone_set]
        ctr = float(torch.tensor(ranks).mean()) if ranks else 0.0
        return _norm(fl + (1-fl)*torch.exp(-0.5*((idx-ctr)/sig)**2))
    def _layer(self, w, cands): return {"name":self._name,"words":[x for x,_ in cands],"probs":w.numpy()}

class L4_ZoneFreq(_ZoneBase):
    def __init__(self, sigma=0.50, floor=0.05): super().__init__("L4_ZONE_FREQ",sigma,floor)
    def forward(self, cands, prompt_words, freq_zones, token_freq):
        high = set(freq_zones.get("high",[])); mid_th = 3
        key = "high" if any(w in high for w in prompt_words) else \
              "low"  if all(token_freq.get(w,0)<mid_th for w in prompt_words) else "mid"
        return self._layer(self._gauss(set(freq_zones.get(key,[])), cands), cands)

class L5_ZoneAlpha(_ZoneBase):
    def __init__(self, sigma=0.40, floor=0.05): super().__init__("L5_ZONE_ALPHA",sigma,floor)
    def forward(self, cands, prompt_words, alpha_zones):
        zone = set().union(*(alpha_zones.get(w[0],[]) for w in prompt_words if w))
        return self._layer(self._gauss(zone, cands), cands)

class L6_ZoneBigram(_ZoneBase):
    def __init__(self, sigma=0.25, floor=0.04): super().__init__("L6_ZONE_BIGRAM",sigma,floor)
    def forward(self, cands, prompt_words, ngram_zones):
        zone = set().union(*(ngram_zones.get((prompt_words[i],prompt_words[i+1]),[])
                             for i in range(len(prompt_words)-1)))
        return self._layer(self._gauss(zone, cands), cands)

class L7_ZoneTrigram(_ZoneBase):
    def __init__(self, sigma=0.20, floor=0.03): super().__init__("L7_ZONE_TRIGRAM",sigma,floor)
    def forward(self, cands, ctx, ngram_zones):
        cl = list(ctx); key = tuple(cl[-2:]) if len(cl)>=2 else (tuple(cl[-1:]) if cl else ())
        return self._layer(self._gauss(set(ngram_zones.get(key,[])), cands), cands)

class L8_ZoneCharTrig(_ZoneBase):
    def __init__(self, sigma=0.35, floor=0.04): super().__init__("L8_ZONE_CHAR_TRIG",sigma,floor)
    def forward(self, cands, prompt_words, char_idx):
        tgs = set().union(*(_char_trigrams(w) for w in prompt_words))
        zone = set().union(*(char_idx.get(t,set()) for t in tgs))
        return self._layer(self._gauss(zone, cands), cands)

class L9_ZoneLatent(_ZoneBase):
    def __init__(self, sigma=0.30, floor=0.04): super().__init__("L9_ZONE_LATENT",sigma,floor)
    def forward(self, cands, prompt_words, latent_sorted_keys, latent_bos_data):
        n = len(latent_sorted_keys)
        q_key = "q0"
        if n and prompt_words:
            for w in prompt_words:
                for i, k in enumerate(latent_sorted_keys):
                    if w in latent_bos_data.get(k, set()):
                        q_key = k; break
        return self._layer(self._gauss(set(latent_bos_data.get(q_key,[])), cands), cands)

class L10_History(nn.Module):
    def __init__(self, smoothing=1.0):
        super().__init__(); self.smoothing = nn.Parameter(_t(smoothing))
    def forward(self, cands, history):
        s = self.smoothing.clamp(min=1e-6)
        pt = _norm(_t([max(1e-12, 1.0/(1.0+s.item()*history[w])) for w,_ in cands]))
        words = [w for w,_ in cands]
        return {"name":"L10_HISTORY","words":words,"probs":pt.detach().numpy() }

class L11_TensorBlend(nn.Module):
    def __init__(self, init_weights=None):
        super().__init__()
        n = 7  # L4..L10
        w = torch.ones(n, dtype=torch.float64)/n if init_weights is None else _t(init_weights)
        self.weights = nn.Parameter(w)
    def forward(self, zone_layers, cands):
        wt = F.softmax(self.weights.float(), dim=0).double()
        n = len(cands)
        stack = torch.stack([_t(l["probs"]) if len(l["probs"])==n
                             else F.pad(_t(l["probs"]),(0,n-len(l["probs"])))
                             for l in zone_layers])
        blended = _norm((stack * wt.unsqueeze(1)).sum(0))
        words = [w for w,_ in cands]
        return {"name":"L11_TENSOR_BLEND","words":words,"probs":blended.detach().numpy() }

class L12_Final(nn.Module):
    def __init__(self, blend_alpha=0.5):
        super().__init__(); self.blend_alpha = nn.Parameter(_t(blend_alpha))
    def forward(self, cands, L11):
        a = self.blend_alpha.clamp(1e-6, 1-1e-6)
        raw = _norm(_t([p for _,p in cands]))
        l11 = _norm(_t(L11["probs"]))
        blended = _norm(raw**(1-a) * l11**a)
        words = [w for w,_ in cands]
        return list(zip(words, blended.tolist())), {"name":"L12_FINAL","words":words,"probs":blended.detach().numpy() }

class L13_CtxReqPos(nn.Module):
    def __init__(self, sigma=0.30, floor=0.04):
        super().__init__(); self.sigma=nn.Parameter(_t(sigma)); self.floor=nn.Parameter(_t(floor))
    def forward(self, cands, draw_pos, stream_len):
        n = len(cands); sig=self.sigma.clamp(min=1e-6); fl=self.floor.clamp(0,1-1e-6)
        norm_pos = (draw_pos % max(1, stream_len)) / max(1, stream_len-1)
        idx = torch.arange(n, dtype=torch.float64)/max(1,n-1)
        w = _norm(fl+(1-fl)*torch.exp(-0.5*((idx-norm_pos)/sig)**2))
        words = [x for x,_ in cands]
        return {"name":"L13_CTX_REQ_POS","words":words,"probs":w.detach().numpy()}

class L14_LockedStateIndex(nn.Module):
    LAYER_NAME = "L14_LOCKED_STATE_INDEX"
    def __init__(self, sigma=0.25, floor=0.03, lock_strength=1.0):
        super().__init__()
        self.sigma=nn.Parameter(_t(sigma)); self.floor=nn.Parameter(_t(floor))
        self.lock_strength=nn.Parameter(_t(lock_strength))
        self._locked: Dict[Tuple,str]={}; self._observed: set=set()
    @property
    def n_locked(self): return len(self._locked)
    @property
    def n_missing(self): return len(self._observed - set(self._locked))
    def reset_state(self): self._locked.clear(); self._observed.clear()
    def commit(self, key, token):
        if key and key not in self._locked: self._locked[key]=token
    @staticmethod
    def key_from_ctx(ctx): return tuple(w for w in ctx if w) or ()
    def forward(self, cands, ctx, draw_pos, stream_len):
        n=len(cands); words=[w for w,_ in cands]
        fl=self.floor.clamp(0,1-1e-6); lw=self.lock_strength.clamp(0,1)
        sig=self.sigma.clamp(min=1e-6); key=self.key_from_ctx(ctx)
        if key and key in self._locked:
            tok=self._locked[key]; wts=torch.full((n,),float(fl),dtype=torch.float64)
            if tok in words: wts[words.index(tok)]=float(fl)+float(lw)*(1-float(fl))
            wts=_norm(wts)
        else:
            if key: self._observed.add(key)
            miss=self.n_missing; sl=max(1,stream_len)
            norm_pos=((draw_pos+miss)%sl)/max(1,sl-1)
            idx=torch.arange(n,dtype=torch.float64)/max(1,n-1)
            wts=_norm(fl+(1-fl)*torch.exp(-0.5*((idx-norm_pos)/sig)**2))
        return {"name":self.LAYER_NAME,"words":words,"probs":wts.detach().numpy()}


# ── CPD builder ─────────────────────────────────────────────────────────────

def build_cpd(corpus: str, ngram_n: int = 2, lidstone_gamma: float = 0.1):
    from nltk.probability import ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist, FreqDist
    tokens = corpus.lower().split()
    if not tokens: raise ValueError("Empty corpus.")
    n = max(2, ngram_n); padded = [""]*( n-1) + tokens + [""]
    cfd = ConditionalFreqDist()
    for ng in zip(*[padded[i:] for i in range(n)]):
        ctx, word = ng[:-1], ng[-1]
        cfd[ctx][word] += 1
    vocab = set(tokens)|{""}
    bins = max(1, len(vocab))
    cpd = ConditionalProbDist(cfd, lambda fd: LidstoneProbDist(fd, gamma=lidstone_gamma, bins=bins))
    return cpd, vocab, tokens


def build_context_index(vocab, cpd, tokens):
    try:
        import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import ContextZoneIndex
        return ContextZoneIndex(vocab, cpd, Counter(tokens))
    except Exception as e:
        print(f"[context_index] unavailable ({e}); zone layers will be uniform.")
        return None


# ── IsomorphismPipeline ─────────────────────────────────────────────────────

class IsomorphismPipeline(nn.Module):
    """14-layer isomorphic probability pipeline (L0–L13)."""

    def __init__(self, cpd, context_index, vocab, ngram_n=2, temperature=4.3,
                 top_k=100, top_p=1.0, rep_penalty=1.13, insight_penalty=3.95,
                 l12_blend_alpha=0.5, l13_sigma=0.30, l13_floor=0.04, **kw):
        super().__init__()
        self.cpd=cpd; self.ctx_idx=context_index; self.vocab=set(vocab)
        self.ngram_n=max(2,int(ngram_n)); self.context_window=self.ngram_n-1
        self.history: Counter = Counter()
        self._step=0; self._pos=0; self._stream: List[int]=[]
        self._char_trig_index = getattr(context_index,"_trig_index",{}) if context_index else {}

        self.l0  = L0_RawDist(rep_penalty)
        self.l1  = L1_TempScaled(temperature)
        self.l2  = L2_InsightPenalty(insight_penalty)
        self.l3  = L3_TopKTopP(top_k, top_p)
        self.l4  = L4_ZoneFreq();   self.l5 = L5_ZoneAlpha()
        self.l6  = L6_ZoneBigram(); self.l7 = L7_ZoneTrigram()
        self.l8  = L8_ZoneCharTrig(); self.l9 = L9_ZoneLatent()
        self.l10 = L10_History(); self.l11 = L11_TensorBlend()
        self.l12 = L12_Final(l12_blend_alpha)
        self.l13 = L13_CtxReqPos(l13_sigma, l13_floor)
        self.frames: List = []

    def _dist_for_ctx(self, ctx):
        for cut in range(len(ctx), 0, -1):
            key = ("",)*(self.context_window-cut) + ctx[-cut:]
            try:
                d = self.cpd[key]
                if list(d.samples()): return d
            except Exception: pass
        try:
            d = self.cpd[("",)*self.context_window]
            if list(d.samples()): return d
        except Exception: pass
        return None

    def seed_stream(self, stream):
        self._stream = list(stream); self._pos = 0

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
            flat = _norm(torch.ones(len(L3_pairs), dtype=torch.float64)).numpy()
            zone_layers = [{"name":n,"words":[w for w,_ in L3_pairs],"probs":flat.copy()}
                           for n in ("L4_ZONE_FREQ","L5_ZONE_ALPHA","L6_ZONE_BIGRAM",
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
        L11 = self.l11(zone_layers + [L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)
        L13 = self.l13(L3_pairs, self._pos, max(1,len(self._stream)))

        l12m = dict(L12_pairs); l13m = dict(zip(L13["words"], L13["probs"].tolist()))
        fl = 1e-12
        blended = [(w, math.sqrt(max(fl,l12m.get(w,fl))*max(fl,l13m.get(w,fl)))) for w in l12m]
        bt = sum(p for _,p in blended)
        blended = [(w,p/bt) for w,p in blended] if bt else blended

        unseen = [(w,p) for w,p in blended if not self.history[w]]
        pool = unseen or blended
        t = sum(p for _,p in pool); pool = [(w,p/t) for w,p in pool] if t else pool

        chosen, cum = pool[-1][0], 0.0
        for w,p in pool:
            cum += p
            if draw < cum: chosen=w; break

        self.history[chosen] += 1
        sl = max(1, len(self._stream))
        nxt = (self._pos + (self._pos % sl)) % sl
        self._pos = nxt; self._step += 1
        return {"chosen": chosen, "draw_pos": self._pos, "next_draw_pos": nxt,
                "layers": [L0,L1,L2,L3]+zone_layers+[L10,L11,L12,L13]}

    def generate(self, prompt: str, n_words: int, draw_fn, **kw):
        """O(2n): one forward pass per word, one reseed pass per failure."""
        toks = [w.lower() for w in prompt.split() if w.isalpha()]
        init = toks[-self.context_window:] if len(toks)>=self.context_window \
               else [""]*(self.context_window-len(toks))+toks
        ctx = deque(init, maxlen=self.context_window)
        words, iters = [], 0
        max_iters = max(1, n_words) * 80
        while len(words) < n_words and iters < max_iters:
            iters += 1
            frame = self.step(ctx, toks, draw_fn())
            if frame is None:           # reseed — O(n) only when dead context
                ctx.clear(); ctx.extend([""]*self.context_window)
            else:
                ctx.append(frame["chosen"]); words.append(frame["chosen"])
        return words[:n_words]

    def generate_text(self, prompt: str, n_words: int, *, stream=None,
                      digits_per_sample=3, seed=None, capitalise=True) -> str:
        draw_fn = self._make_draw_fn(stream, digits_per_sample, seed)
        words = self.generate(prompt, n_words, draw_fn)
        all_words = prompt.strip().split() + words
        if not capitalise: return " ".join(all_words)
        out, cap = [], True
        for w in all_words:
            out.append(w.capitalize() if cap else w)
            cap = bool(w.rstrip("\"'")[-1:] in {".", "!", "?"})
        return " ".join(out)


class LockedIsomorphismPipeline(IsomorphismPipeline):
    """IsomorphismPipeline + L14_LockedStateIndex (15 layers)."""
    def __init__(self, *args, l14_sigma=0.25, l14_floor=0.03,
                 l14_lock_strength=1.0, l14_blend_alpha=0.5, **kw):
        super().__init__(*args, **kw)
        self.l14 = L14_LockedStateIndex(l14_sigma, l14_floor, l14_lock_strength)
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
            flat = _norm(torch.ones(len(L3_pairs), dtype=torch.float64)).numpy()
            zone_layers = [{"name":n,"words":[w for w,_ in L3_pairs],"probs":flat.copy()}
                           for n in ("L4_ZONE_FREQ","L5_ZONE_ALPHA","L6_ZONE_BIGRAM",
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
        L11 = self.l11(zone_layers + [L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)
        sl = max(1, len(self._stream))
        L13 = self.l13(L3_pairs, self._pos, sl)
        L14 = self.l14(L3_pairs, ctx, self._pos, sl)

        a = float(self.l14_blend_alpha.clamp(1e-6, 1-1e-6))
        fl = 1e-12
        l12m = dict(L12_pairs); l13m = dict(zip(L13["words"],L13["probs"].tolist()))
        l14m = dict(zip(L14["words"],L14["probs"].tolist()))
        blended = [(w, (math.sqrt(max(fl,l12m.get(w,fl))*max(fl,l13m.get(w,fl)))**(1-a)) *
                      (max(fl,l14m.get(w,fl))**a)) for w in l12m]
        bt = sum(p for _,p in blended)
        blended = [(w,p/bt) for w,p in blended] if bt else blended

        unseen = [(w,p) for w,p in blended if not self.history[w]]
        pool = unseen or blended
        t = sum(p for _,p in pool); pool = [(w,p/t) for w,p in pool] if t else pool

        chosen, cum = (pool[-1][0] if pool else ""), 0.0
        for w,p in pool:
            cum += p
            if draw < cum: chosen=w; break

        key = L14_LockedStateIndex.key_from_ctx(ctx)
        if key and chosen: self.l14.commit(key, chosen)
        self.history[chosen] += 1
        nxt = (self._pos + (self._pos % sl)) % sl
        self._pos = nxt; self._step += 1
        return {"chosen": chosen, "draw_pos": self._pos, "next_draw_pos": nxt,
                "layers": [L0,L1,L2,L3]+zone_layers+[L10,L11,L12,L13,L14]}


# ── pipeline factory ────────────────────────────────────────────────────────

def build_pipeline(dataset_name: str, config_name: Optional[str] = None,
                   text_fields: Optional[Sequence] = None, *, locked=True,
                   ngram_n=3, lidstone_gamma=0.1, minlen=3,
                   preprocessor_kw: Optional[Dict]=None,
                   pipeline_kw: Optional[Dict]=None,
                   ) -> Tuple[IsomorphismPipeline, Preprocessor]:
    pre = Preprocessor(dataset_name, config_name, text_fields,
                       minlen=minlen, **(preprocessor_kw or {}))
    cpd, vocab, tokens = build_cpd(pre.tocorpus(), ngram_n, lidstone_gamma)
    ctx_idx = build_context_index(vocab, cpd, tokens)
    cls = LockedIsomorphismPipeline if locked else IsomorphismPipeline
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
        raise KeyError(f"Unknown preset {preset_name!r}. Available: {sorted(HF_DATASET_PRESETS)}")
    spec = dict(HF_DATASET_PRESETS[preset_name])
    spec.setdefault("dataset_name", preset_name)
    spec.update(kw)
    return build_pipeline(**spec)


# ── demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe, pre = build_pipeline_from_preset(
        "blended_skill_talk", locked=True, ngram_n=3,
        preprocessor_kw=dict(boundaryquota=8, max_per_split=1000),
    )
    while True:
        prompt = input("USER: ")
        if not prompt.strip(): break
        print(pipe.generate_text(prompt, n_words=800, seed=42))
        print()