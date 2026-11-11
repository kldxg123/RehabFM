"""Historical shared backbone, explicit architecture/loss ablation interfaces."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import torch
from torch import nn
from rehabfm.model import RehabFM
class NoTemporal(nn.Module):
 def forward(self,x,src_key_padding_mask=None,**kwargs):return x
def make_core(arm):
 model=RehabFM(n_datasets=48,geometry=True,native=False)
 if arm=='no_temporal':model.encoder=NoTemporal()
 return model
def fuse(out,B):
 first=next(iter(out['z'].values()));s=first.new_zeros(B,64);n=first.new_zeros(B)
 for m,z in out['z'].items():s.index_add_(0,out['owners'][m],z);n.index_add_(0,out['owners'][m],n.new_ones(len(z)))
 return s/n.clamp_min(1)[:,None]
def aux_loss(out,batch,heads,weights):
 ids=batch['dataset'].unique();assert len(ids)==1
 key=str(int(ids[0]));y=batch['y']
 if y.unique().numel()<2:return None
 return nn.functional.cross_entropy(heads[key](fuse(out,len(y))).float(),y,weight=weights[key])
