"""Explicit signal-encoder adaptation of the official ACL2018 LMF factor computation."""
import sys
from pathlib import Path
import torch
from torch import nn
D=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(D))
from core_models import make_core
class FactorFusion(nn.Module):
 def __init__(self,modalities,K,dim=64,rank=4):
  super().__init__();self.K=K;self.rank=rank
  self.factors=nn.ParameterDict({m:nn.Parameter(torch.empty(rank,dim+1,K)) for m in modalities})
  self.weights=nn.Parameter(torch.empty(1,rank));self.bias=nn.Parameter(torch.zeros(1,K))
  for p in self.factors.values():nn.init.xavier_normal_(p)
  nn.init.xavier_normal_(self.weights)
 def forward(self,z,owners,B):
  assert z and set(z)<=set(self.factors)
  first=next(iter(z.values()));seen=torch.zeros(B,device=first.device,dtype=torch.long)
  # Products use FP32, matching the published reference's FP32 factor computation.
  with torch.autocast(device_type=first.device.type,enabled=False):
   value=torch.ones(B,self.rank,self.K,device=first.device,dtype=torch.float32)
   for m,x in z.items():
    ix=owners[m];assert ix.unique().numel()==len(ix)
    aug=torch.cat([torch.ones(len(x),1,device=x.device),x.float()],dim=1)
    term=torch.einsum('bd,rdk->brk',aug,self.factors[m])
    neutral=torch.ones_like(value).index_copy(0,ix,term)
    value=value*neutral;seen.index_add_(0,ix,torch.ones_like(ix))
   value=value.masked_fill((seen==0)[:,None,None],0)
   return torch.einsum('brk,qr->bqk',value,self.weights).squeeze(1)+self.bias
class Classifier(nn.Module):
 def __init__(self,arm,K):
  super().__init__();assert arm=='LMF'
  self.core=make_core('no_temporal')
  self.fusion=FactorFusion(list(self.core.mod_index),K)
 def forward(self,b,augment=False):
  out=self.core(b,modality_drop_p=.25 if augment else 0,channel_drop_p=.1 if augment else 0)
  return self.fusion(out['z'],out['owners'],len(b['dataset']))
