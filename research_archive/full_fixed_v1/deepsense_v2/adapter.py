"""Signal adaptation of official DeepSense Android noConv3D architecture (TF1 equations)."""
import sys,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
D=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(D))
def schema(rows):
 return {m:max(int(r['tokens'][m].shape[1]) for r in rows if m in r['tokens']) for m in sorted({m for r in rows for m in r['tokens']})}
def spectral(x):
 # Explicit adaptation:128 samples/window, first10 nonnegative bins, includingDC.
 if x.shape[-1]!=128:x=F.interpolate(x.reshape(-1,1,x.shape[-1]),size=128,mode='linear',align_corners=False).reshape(*x.shape[:-1],128)
 z=torch.fft.rfft(x.float(),dim=-1,norm='forward')[...,:10]
 return torch.stack([z.abs(),torch.angle(z)],-1).permute(0,1,3,2,4).flatten(2) # B,W,frequency x channel x pair
class TFGRU(nn.Module):
 def __init__(self,input_dim,hidden=120):
  super().__init__();self.hidden=hidden;self.gates=nn.Linear(input_dim+hidden,2*hidden);self.candidate=nn.Linear(input_dim+hidden,hidden)
  nn.init.xavier_uniform_(self.gates.weight);nn.init.ones_(self.gates.bias);nn.init.xavier_uniform_(self.candidate.weight);nn.init.zeros_(self.candidate.bias)
 def forward(self,x,h):
  r,u=self.gates(torch.cat([x,h],-1)).sigmoid().chunk(2,-1)
  c=self.candidate(torch.cat([x,r*h],-1)).tanh()
  return u*h+(1-u)*c
class TFBN(nn.Module):
 def __init__(self,d):
  super().__init__();self.weight=nn.Parameter(torch.ones(d));self.bias=nn.Parameter(torch.zeros(d));self.register_buffer('mean',torch.zeros(d));self.register_buffer('variance',torch.ones(d))
 def forward(self,x):
  if self.training:
   mu=x.mean((0,2));var=x.var((0,2),unbiased=False)
   with torch.no_grad():self.mean.mul_(.999).add_(mu.detach(),alpha=.001);self.variance.mul_(.999).add_(var.detach(),alpha=.001)
  else:mu,var=self.mean,self.variance
  return (x-mu[None,:,None])*torch.rsqrt(var[None,:,None]+.001)*self.weight[None,:,None]+self.bias[None,:,None]
def same_padding(length,kernel,stride):
 total=max(0,(math.ceil(length/stride)-1)*stride+kernel-length)
 return total//2,total-total//2
def structured_drop(x,B,W,p,training):
 if not training:return x
 mask=F.dropout(torch.ones(B,x.shape[1],1,device=x.device,dtype=x.dtype),p=p,training=True)
 return (x.reshape(B,W,x.shape[1],x.shape[2])*mask[:,None]).reshape_as(x)
class ConvBlock(nn.Module):
 def __init__(self,din,k,stride=1,same=False):
  super().__init__();self.k=k;self.stride=stride;self.same=same;self.conv=nn.Conv1d(din,64,k,stride=stride);self.bn=TFBN(64)
  nn.init.xavier_uniform_(self.conv.weight);nn.init.zeros_(self.conv.bias)
 def forward(self,x):
  if self.same:x=F.pad(x,same_padding(x.shape[-1],self.k,self.stride))
  return self.bn(self.conv(x)).relu()
class DeepSenseNet(nn.Module):
 def __init__(self,channels,K):
  super().__init__();self.channels=dict(channels);M=len(channels);assert M>0
  self.individual=nn.ModuleDict({m:nn.ModuleList([ConvBlock(1,6*c,2*c),ConvBlock(64,3),ConvBlock(64,3)]) for m,c in channels.items()})
  self.merge=nn.ModuleList([ConvBlock(64,M*8,M,True),ConvBlock(64,M*6,M,True),ConvBlock(64,M*4,M,True)])
  length=4*M
  for _ in range(3):length=math.ceil(length/M)
  self.gru1=TFGRU(64*length);self.gru2=TFGRU(120);self.head=nn.Linear(120,K);nn.init.xavier_uniform_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,features,masks):
  B,W=next(iter(features.values())).shape[:2];chunks=[];valid=torch.zeros(B,W,dtype=torch.bool,device=next(iter(features.values())).device)
  for m,blocks in self.individual.items():
   x=features[m].reshape(B*W,1,-1)
   for j,block in enumerate(blocks):
    x=block(x)
    if j<2:x=structured_drop(x,B,W,.2,self.training)
   assert x.shape[-1]==4
   x=x.reshape(B,W,64,4)*masks[m][:,:,None,None];chunks.append(x.permute(0,1,3,2));valid|=masks[m]
  assert valid.any(1).all()
  # Official Android order: B,W,frequency,sensor,channels then flatten frequency/sensor.
  x=torch.stack(chunks,3).reshape(B*W,4*len(chunks),64).transpose(1,2);x=structured_drop(x,B,W,.2,self.training)
  for j,block in enumerate(self.merge):
   x=block(x)
   if j<2:x=structured_drop(x,B,W,.2,self.training)
  x=x.transpose(1,2).reshape(B,W,-1);h1=x.new_zeros(B,120);h2=x.new_zeros(B,120);out=[]
  for t in range(W):
   h1=torch.where(valid[:,t,None],self.gru1(x[:,t],h1),h1)
   d1=F.dropout(h1,.5,self.training)
   h2=torch.where(valid[:,t,None],self.gru2(d1,h2),h2)
   out.append(F.dropout(h2,.5,self.training)*valid[:,t,None])
  return self.head(torch.stack(out,1).sum(1)/valid.sum(1)[:,None])
class Classifier(nn.Module):
 def __init__(self,arm,K,modalities):
  super().__init__();assert arm=='DeepSense';self.modalities=dict(modalities);self.core=DeepSenseNet(modalities,K)
 def forward(self,b,augment=False):
  B=len(b['dataset']);dev=b['dataset'].device;features={};masks={};W=12
  for m,C in self.modalities.items():
   x=torch.zeros(B,W,C,128,device=dev);mask=torch.zeros(B,W,dtype=torch.bool,device=dev)
   if m in b:
    p=b[m];v=p['x'].float();n,w,c,tw=v.shape;assert w<=W and c<=C
    cm=p['cmask']
    if augment and self.training:
     keep=torch.rand_like(cm.float())>=.1;keep=keep|(~keep.any(-1,keepdim=True));cm=cm&keep
    v=v*cm[:,:,:,None]
    if tw!=128:v=F.interpolate(v.reshape(-1,1,tw),128,mode='linear',align_corners=False).reshape(n,w,c,128)
    x[p['index'],:w,:c]=v;mask[p['index'],:w]=p['wmask']
   features[m]=spectral(x);masks[m]=mask
  assert not {k for k,v in b.items() if isinstance(v,dict) and 'x' in v}-set(self.modalities)
  if augment and self.training and len(masks)>1:
   for i in range(B):
    present=[m for m in masks if masks[m][i].any()]
    for m in list(present):
     if len(present)>1 and torch.rand(())<.25:masks[m][i]=False;features[m][i]=0;present.remove(m)
  return self.core(features,masks)
