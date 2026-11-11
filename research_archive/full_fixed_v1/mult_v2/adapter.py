"""Explicit physiological-signal adaptation of ACL2019 MulT, using official encoders."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
D=Path(__file__).resolve().parent.parent
OFFICIAL=D.parent/'published_sources_verified/MulT'
sys.path.insert(0,str(D));sys.path.insert(0,str(OFFICIAL))
from core_models import make_core
from modules.transformer import TransformerEncoder

def schema(rows):
 return sorted({m for r in rows for m in r['tokens']})

class DirectionalFusion(nn.Module):
 def __init__(self,modalities,K,input_dim=64,layers=5,heads=5,dropout=True):
  super().__init__();self.modalities=tuple(modalities);assert len(set(modalities))==len(modalities)>0
  self.input_dim=input_dim;self.dim=30;self.mem_dim=30*max(1,len(modalities)-1);self.out_dropout=0.
  self.projections=nn.ModuleDict({m:nn.Conv1d(input_dim,30,1,bias=False) for m in modalities})
  def enc(d,n):
   x=TransformerEncoder(d,heads,n,attn_dropout=.1 if dropout else 0.,relu_dropout=.1 if dropout else 0.,res_dropout=.1 if dropout else 0.,embed_dropout=.25 if dropout else 0.,attn_mask=False)
   # Official empty FloatTensor(1) is a dtype/device sentinel, not a learned value.
   # Deterministically initialize it for stable state hashes; forward never uses its value.
   x.embed_positions._float_tensor.zero_();return x
  self.cross=nn.ModuleDict({a:nn.ModuleDict({b:enc(30,layers) for b in modalities if b!=a}) for a in modalities})
  self.memory=nn.ModuleDict({m:enc(self.mem_dim,max(layers,3)) for m in modalities})
  combined=self.mem_dim*len(modalities)
  self.proj1=nn.Linear(combined,combined);self.proj2=nn.Linear(combined,combined);self.out_layer=nn.Linear(combined,K)
 def forward(self,sequences):
  # One record, actual nonpadding sequence lengths. No other record acts as context.
  assert sequences and set(sequences)<=set(self.modalities)
  x={m:(v.T[None] if self.input_dim==30 else self.projections[m](v.T[None])).permute(2,0,1) for m,v in sequences.items()}
  first=next(iter(x.values()));features=[]
  for a in self.modalities:
   if a not in x:
    features.append(first.new_zeros(1,self.mem_dim));continue
   peers=[b for b in self.modalities if b!=a]
   if not peers:h=x[a]
   else:
    has_peer=any(b in x for b in peers)
    branches=[]
    for b in peers:
     if b in x:branches.append(self.cross[a][b](x[a],x[b],x[b]))
     elif not has_peer:branches.append(x[a]) # declared no-peer self-memory fallback
     else:branches.append(torch.zeros_like(x[a])) # absent source slot, no fictitious keys
    h=torch.cat(branches,-1)
   features.append(self.memory[a](h)[-1])
  features=torch.cat(features,-1)
  return self.out_layer(features+self.proj2(F.dropout(F.relu(self.proj1(features)),p=self.out_dropout,training=self.training)))

class Classifier(nn.Module):
 def __init__(self,arm,K,modalities):
  super().__init__();assert arm=='MulT';self.modalities=tuple(modalities)
  self.core=make_core('no_temporal');self.fusion=DirectionalFusion(modalities,K)
 def forward(self,b,augment=False):
  out=self.core(b,modality_drop_p=0.,channel_drop_p=.1 if augment else 0.)
  result=[]
  for i in range(len(b['dataset'])):
   seq={}
   for m,t in out['tokens'].items():
    hits=(out['owners'][m]==i).nonzero().flatten()
    if len(hits):
     assert len(hits)==1;j=int(hits[0]);mask=b[m]['wmask'][j]
     assert mask.any();seq[m]=t[j,mask]
   assert seq and set(seq)<=set(self.modalities)
   # Per-record modality dropout guarantees at least one observed stream survives.
   if self.training and augment and len(seq)>1:
    for m in list(seq):
     if len(seq)>1 and torch.rand(())<.25:del seq[m]
   # MulT's original attention implementation runs FP32; signal CNN uses BF16 on GPU.
   with torch.autocast(device_type=next(iter(seq.values())).device.type,enabled=False):
    result.append(self.fusion({m:x.float() for m,x in seq.items()}))
  return torch.cat(result)
