"""CUDA-first stochastic n-gram generator.

Run:
  pip install torch gradio
  python app_cuda.py
  python app_cuda.py --share

All dense scoring, adaptive binding similarity, subset selection gains,
logit construction, top-k filtering, and sampling use PyTorch tensors on CUDA
when available. Text parsing and persistence necessarily remain Python-side.
"""
from __future__ import annotations
import argparse,json,math,random,re
from collections import Counter,defaultdict
from pathlib import Path
from typing import Dict,List,Optional,Tuple
import torch

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA=DEVICE.type=='cuda'
BOS,EOS='<bos>','<eos>'
SPECIAL={BOS,EOS}
MODEL_PATH='model_cuda.json';BINDINGS_PATH='bindings_cuda.json';DEFAULT_CORPUS='corpus.txt'
ALPHA=.05;TEMP=.8;TOP_K=20;MAX_NEW=800;MAX_SUBSET=5;BEAM=24
Vec=Dict[str,float]

def tokenize(text):return re.findall(r"[a-zA-ZÀ-ÿ0-9_]+|[.!?,;:{}()[\]<>+=*/%^_-]",text.lower())
def sentences(text):
    out=[]
    for x in re.split(r'(?<=[.!?])\s+|\n+',text.replace('\r\n','\n')):
        x=x.strip()
        if x:out.append(x if x[-1] in '.!?' else x+'.')
    return out
def bow(tokens):return Counter(x for x in tokens if x not in SPECIAL)

def matrix(vectors,keys):
    return torch.tensor([[v.get(k,0.) for k in keys] for v in vectors],dtype=torch.float32,device=DEVICE)
def gpu_cosine(query:Vec,vectors:List[Vec]):
    keys=sorted(set(query).union(*(v.keys() for v in vectors))) if vectors else []
    if not keys:return torch.zeros(len(vectors),device=DEVICE)
    q=torch.tensor([query.get(k,0.) for k in keys],dtype=torch.float32,device=DEVICE)
    x=matrix(vectors,keys)
    return (x@q)/(torch.linalg.vector_norm(x,dim=1)*torch.linalg.vector_norm(q)).clamp_min(1e-12)

def gpu_gain(target:Vec,vectors:List[Vec]):
    keys=sorted(target)
    if not keys:return torch.zeros(len(vectors),device=DEVICE)
    t=torch.tensor([target[k] for k in keys],dtype=torch.float32,device=DEVICE)
    return torch.minimum(matrix(vectors,keys),t).sum(1)

class Bindings:
    def __init__(self,threshold=.35,momentum=.85):self.threshold=threshold;self.momentum=momentum;self.contexts={};self.links={}
    def update(self,text,incremental=False):
        incoming=defaultdict(Counter)
        for s in sentences(text):
            t=tokenize(s)
            for i,w in enumerate(t):incoming[w].update(t[max(0,i-4):min(len(t),i+5)])
        if incremental and self.contexts:
            merged=defaultdict(Counter)
            for w,v in self.contexts.items():merged[w].update(v)
            for w,v in incoming.items():
                for k,x in v.items():merged[w][k]=self.momentum*merged[w][k]+(1-self.momentum)*x
            self.contexts=dict(merged)
        else:self.contexts=dict(incoming)
        self.rebuild()
    def rebuild(self):
        words=list(self.contexts);vectors=[{k:float(x) for k,x in self.contexts[w].items()} for w in words]
        if not words:self.links={};return
        scores=gpu_cosine_matrix(vectors)
        links={}
        for i,w in enumerate(words):
            row=scores[i].clone();row[i]=-1
            values,idx=torch.topk(row,k=min(8,len(words)-1))
            links[w]={words[int(j)]:float(v) for v,j in zip(values.detach().cpu(),idx.detach().cpu()) if float(v)>=self.threshold}
        self.links=links
    def expand(self,v):
        out=defaultdict(float)
        for w,x in v.items():
            out[w]+=x
            for z,s in self.links.get(w,{}).items():out[z]+=x*s
        return dict(out)
    def save(self):Path(BINDINGS_PATH).write_text(json.dumps({'threshold':self.threshold,'momentum':self.momentum,'links':self.links,'contexts':{w:dict(v) for w,v in self.contexts.items()}},indent=2),encoding='utf8')
    @classmethod
    def load(cls):
        x=cls()
        if not Path(BINDINGS_PATH).exists():return x
        d=json.loads(Path(BINDINGS_PATH).read_text(encoding='utf8'));x.threshold=d.get('threshold',.35);x.momentum=d.get('momentum',.85);x.links=d.get('links',{});x.contexts={w:Counter(v) for w,v in d.get('contexts',{}).items()};return x
    def summary(self):return f'Device: {DEVICE}\nBinding tokens: {len(self.links)}\nBinding links: {sum(len(x) for x in self.links.values())}'

def gpu_cosine_matrix(vectors):
    if not vectors:return torch.empty((0,0),device=DEVICE)
    keys=sorted(set().union(*(v.keys() for v in vectors)));x=matrix(vectors,keys);n=torch.linalg.vector_norm(x,dim=1,keepdim=True).clamp_min(1e-12);return (x@x.T)/(n@n.T)

class Ref:
    def __init__(self,i,s,t,freq,features):self.i=i;self.s=s;self.t=t;self.freq=freq;self.f=features
class Match:
    def __init__(self,rank,ref,score,vector,selected,endpoint):self.rank=rank;self.ref=ref;self.score=score;self.vector=vector;self.selected=selected;self.endpoint=endpoint

class Corpus:
    def __init__(self,text='',bindings=None):
        self.bindings=bindings or Bindings();self.bindings.update(text) if text else None;ss=sentences(text);freq=Counter(s.lower() for s in ss);self.refs=[]
        for i,s in enumerate(ss):
            t=tokenize(s);f=self.bindings.expand({k:float(v) for k,v in bow(t).items()});self.refs.append(Ref(i,s,t,freq[s.lower()],f))
    def search(self,prompt,limit=5):
        target=self.bindings.expand({k:float(v) for k,v in bow(tokenize(prompt)).items()});vectors=[r.f for r in self.refs];scores=gpu_cosine(target,vectors);gains=gpu_gain(target,vectors);chosen=[];covered=defaultdict(float)
        for _ in range(min(MAX_SUBSET,len(self.refs))):
            residual={k:max(0.,v-covered[k]) for k,v in target.items()};rg=gpu_gain(residual,vectors);rg[[i for i in range(len(self.refs)) if i in chosen]]=0
            j=int(torch.argmax(rg).item())
            if float(rg[j])<=0:break
            chosen.append(j)
            for k,v in self.refs[j].f.items():covered[k]+=v
        rows=[]
        for i,r in enumerate(self.refs):
            if i in chosen or float(gains[i])>0:
                overlap=len(set(tokenize(prompt))&set(r.t))/max(1,len(set(tokenize(prompt))|set(r.t)));score=.7*float(scores[i])+.2*overlap+.1*math.log1p(r.freq);rows.append(Match(0,r,score,float(scores[i]),i in chosen,i not in chosen))
        rows.sort(key=lambda x:x.score,reverse=True)
        for i,x in enumerate(rows[:limit],1):x.rank=i
        return rows[:limit]

class NGram:
    def __init__(self):self.vocab={};self.inv=[];self.uni=Counter();self.bi=defaultdict(Counter);self.tri=defaultdict(Counter)
    def ingest(self,text):
        for s in sentences(text):
            q=[BOS,BOS,*tokenize(s),EOS];self.uni.update(q)
            for a,b in zip(q,q[1:]):self.bi[a][b]+=1
            for a,b,c in zip(q,q[1:],q[2:]):self.tri[(a,b)][c]+=1
        self.inv=sorted(self.uni);self.vocab={w:i for i,w in enumerate(self.inv)};return self
    def gpu_logits(self,history):
        a,b=history[-2:];counts=self.tri.get((a,b)) or self.bi.get(b) or self.uni;v=torch.full((len(self.inv),),-float('inf'),device=DEVICE)
        total=sum(counts.values())+ALPHA*len(counts)
        for w,c in counts.items():v[self.vocab[w]]=math.log((c+ALPHA)/total)
        return v
    def bias_from(self,seeds):
        out={}
        for w in seeds:
            c=self.bi.get(w); total=sum(c.values()) if c else 0
            if total:
                for z,n in c.items():out[z]=out.get(z,0.)+n/total
        return out
    def sample(self,history,bias=None):
        logits=self.gpu_logits(history)
        if bias:
            for w,x in bias.items():
                if w in self.vocab:logits[self.vocab[w]]+=x
        vals,ids=torch.topk(logits,k=min(TOP_K,len(logits)));p=torch.softmax(vals/TEMP,0);return self.inv[int(ids[torch.multinomial(p,1)].item())]
    def generate(self,prompt,biasfn=None):
        out=[BOS,BOS,*tokenize(prompt)]
        for i in range(MAX_NEW):
            b=biasfn(out) if biasfn and i%8==0 else None;x=self.sample(out,b)
            out.append(x)
        return ' '.join(x for x in out if x not in SPECIAL)
    def save(self):Path(MODEL_PATH).write_text(json.dumps({'vocab':self.inv,'uni':dict(self.uni),'bi':{str(k):dict(v) for k,v in self.bi.items()},'tri':{str(k):dict(v) for k,v in self.tri.items()}},indent=2),encoding='utf8')

NGram.model_bias=lambda self,seeds: self.bias_from(seeds) if hasattr(self,'bias_from') else {}
class State:
    def __init__(self):self.model=None;self.corpus=None;self.bindings=Bindings()
    def train(self,text):self.model=NGram().ingest(text);self.model.save();self.bindings=Bindings();self.bindings.update(text);self.bindings.save();self.corpus=Corpus(text,self.bindings)
S=State()
def train(file):
    if file is None:return 'No file','',''
    text=Path(getattr(file,'name',file)).read_text(encoding='utf8',errors='replace');S.train(text);return 'Trained',text[:1000],S.bindings.summary()
def generate(prompt):
    if not S.model:return 'Train first','',''
    target=S.corpus.bindings.expand({k:float(v) for k,v in bow(tokenize(prompt)).items()});state=dict(target)
    def bias(history):
        seeds=sorted(state,key=lambda x:state[x]*S.corpus.bindings.threshold,reverse=True)[:12];return S.model_bias(seeds) if hasattr(S,'model_bias') else {}
    text=S.model.generate(prompt,bias);return 'CUDA device: '+str(DEVICE),format_matches(S.corpus.search(prompt)),text
def format_matches(rows):return '\n'.join(f'{x.rank}. {x.ref.s} score={x.score:.3f} vector={x.vector:.3f}'+(' [subset]' if x.selected else ' [endpoint]') for x in rows) or 'No matches.'
def main():
    p=argparse.ArgumentParser();p.add_argument('--share',action='store_true');p.add_argument('--server-name',default='127.0.0.1');p.add_argument('--server-port',type=int,default=7860);a=p.parse_args();import gradio as gr
    with gr.Blocks(title="CUDA N-Gram") as ui:
        gr.Markdown(
            f"# CUDA-first stochastic n-gram generator\n"
            f"Device: `{DEVICE}`"
        )

        gr.Markdown("## 1. Corpus and training")

        corpus_file = gr.File(
            label="Corpus text file",
            file_types=["file"],
        )

        train_button = gr.Button(
            "Train model",
            variant="primary",
        )

        training_status = gr.Textbox(
            label="Training status",
            lines=2,
        )

        corpus_preview = gr.Textbox(
            label="Corpus preview",
            lines=6,
        )

        binding_summary = gr.Textbox(
            label="GPU model and adaptive-binding summary",
            lines=5,
        )

        train_button.click(
            fn=train,
            inputs=[corpus_file],
            outputs=[
                training_status,
                corpus_preview,
                binding_summary,
            ],
        )

        gr.Markdown("## 2. CUDA stochastic generation")

        prompt_input = gr.Textbox(
            label="Text prompt",
            placeholder="Enter a prompt in the corpus language...",
            lines=3,
        )

        generate_button = gr.Button(
            "Generate text",
            variant="primary",
        )

        generation_status = gr.Textbox(
            label="Generation and CUDA status",
            lines=3,
        )

        corpus_matches = gr.Textbox(
            label="Subset-sum corpus matches and optimized endpoints",
            lines=10,
        )

        generated_text = gr.Textbox(
            label="Generated text",
            lines=12,
        )

        generate_button.click(
            fn=generate,
            inputs=[prompt_input],
            outputs=[
                generation_status,
                corpus_matches,
                generated_text,
            ],
        )
        ui.launch(server_name=a.server_name,server_port=a.server_port,share=a.share)
if __name__=='__main__':main()
