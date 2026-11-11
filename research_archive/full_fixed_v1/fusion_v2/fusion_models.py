"""Explicit mean-initialized modality fusion ablations; no change to old sources."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from torch import nn
from core_models import make_core as old_make_core
from rehabfm.model import RehabFM
from rehabfm.config import MODALITIES
MODS=tuple(sorted(MODALITIES))
class Fusion(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  if kind=='concat':self.projection=nn.Parameter(torch.eye(64).repeat(len(MODS),1,1))
  elif kind=='attention':
   self.score=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1));nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
  else:raise ValueError(kind)
 def forward(self,out,B):
  first=next(iter(out['z'].values()));z=first.new_zeros(B,len(MODS),64);T=first.new_zeros(B,len(MODS),12,64);present=torch.zeros(B,len(MODS),device=first.device,dtype=torch.bool)
  for m,v in out['z'].items():
   k=MODS.index(m);idx=out['owners'][m];z[idx,k]=v;T[idx,k]=out['tokens'][m];present[idx,k]=True
  valid=present.any(1); count=present.sum(1).clamp_min(1)
  if self.kind=='concat':
   # One block per named modality: concatenate zero-filled slots, apply linear
   # blocks, normalize by number present. No bias preserves all-zero windows.
   result=torch.einsum('bmd,mdh->bmh',z,self.projection).sum(1)/count[:,None]
   tokens=torch.einsum('bmtd,mdh->bmth',T,self.projection).sum(1)/count[:,None,None]
   weights=present.to(z.dtype)/count[:,None]
  else:
   logits=self.score(z).squeeze(-1).float().masked_fill(~present,-torch.inf)
   logits=torch.where(valid[:,None],logits,torch.zeros_like(logits))
   weights=logits.softmax(1).to(z.dtype)*present.to(z.dtype)
   result=(weights[:,:,None]*z).sum(1);tokens=(weights[:,:,None,None]*T).sum(1)
  return result,tokens,weights
class FusionRehabFM(RehabFM):
 def __init__(self,kind):
  super().__init__(n_datasets=48,geometry=True,native=False)
  # Preserve the old decoder/predictor initialization random stream exactly.
  with torch.random.fork_rng(devices=[]):self.fusion=Fusion(kind)
 def forward(self,batch,*args,**kwargs):
  out=super().forward(batch,*args,**kwargs);out['fusion_z'],out['fusion_T'],out['fusion_weights']=self.fusion(out,len(batch['y']));return out
def make_core(arm):return FusionRehabFM(arm)
def fuse(out,B):return out['fusion_z']
def aux_loss(out,batch,heads,weights):
 ids=batch['dataset'].unique();assert len(ids)==1;key=str(int(ids[0]));y=batch['y']
 if y.unique().numel()<2:return None
 return nn.functional.cross_entropy(heads[key](fuse(out,len(y))).float(),y,weight=weights[key])
