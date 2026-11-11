"""Declared operator adaptations, NOT reproduction of the pretrained SleepFM system.
Official classes are compiled unchanged from the pinned source AST.
"""
from common import *
import ast,math,torch
from torch import nn
from torch.nn import functional as F
from rehabfm.model import RehabFM
from core_models import make_core as original_core
source=O/'official_source/sleepfm/models/models.py'
tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['Tokenizer','AttentionPooling','PositionalEncoding','SetTransformer']]
from einops import rearrange
ns=dict(torch=torch,nn=nn,F=F,math=math,rearrange=rearrange)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
OfficialSet=ns['SetTransformer'];OfficialPool=ns['AttentionPooling']
class NoCondition(RehabFM):
 def _cond(self,mod,like,dataset_idx=None):return like.new_zeros(like.shape[0],64)
class PoolReplacement(nn.Module):
 def __init__(self):
  super().__init__();self.pool=OfficialPool(64,num_heads=4,dropout=.3)
 def forward(self,x,mask):
  B,W,C,D=x.shape;v=x.reshape(B*W,C,D);m=mask.reshape(B*W,C);empty=~m.any(-1)
  safe=m.clone();safe[empty,0]=True
  z=self.pool(v,~safe);z=z.masked_fill(empty[:,None],0)
  return z.reshape(B,W,D)
from shape_safe_layernorm import ShapeSafeLayerNorm
def temporal_pool_forward(pool,x,mask):
 if not x.is_cuda:return pool(x,mask)
 from torch.nn.attention import sdpa_kernel,SDPBackend
 with sdpa_kernel(SDPBackend.MATH):return pool(x,mask)
class SleepAdapter(nn.Module):
 forward=RehabFM.forward
 def __init__(self):
  super().__init__()
  self.geometry=True
  # Official released base configuration: 128 wide, 8 heads, SIX layers.
  self.base=OfficialSet(1,640,128,8,6,pooling_head=8,dropout=.3,max_seq_length=128)
  self.bridge=nn.Linear(128,64)
  for i,layer in enumerate(self.base.patch_embedding.tokenizer):
   if isinstance(layer,nn.LayerNorm) and len(layer.normalized_shape)>1:
    safe=ShapeSafeLayerNorm(layer.normalized_shape,eps=layer.eps,elementwise_affine=layer.elementwise_affine)
    safe.load_state_dict(layer.state_dict());self.base.patch_embedding.tokenizer[i]=safe
 def encode_modality(self,mod,x,cmask,wmask=None,dataset_idx=None):
  B,W,C,L=x.shape
  # Declared input adaptation: each original 5-s channel resampled to 640.
  x=F.interpolate(x.reshape(-1,1,L).float(),size=640,mode='linear',align_corners=False).reshape(B,W,C,640)
  v=x.permute(0,2,1,3).reshape(B,C,W*640)
  ch=self.base.patch_embedding(v).permute(0,2,1,3).reshape(B*W,C,128)
  valid=cmask.reshape(B*W,C);empty=~valid.any(1);safe=valid.clone();safe[empty,0]=True
  tok=self.base.spatial_pooling(ch,~safe).masked_fill(empty[:,None],0).reshape(B,W,128)
  tok=self.base.layer_norm(self.base.positional_encoding(tok))
  wm=torch.ones(B,W,dtype=torch.bool,device=x.device) if wmask is None else wmask
  safe_w=wm.clone();safe_w[~wm.any(1),0]=True
  tok=self.base.transformer_encoder(tok,src_key_padding_mask=~safe_w)
  z=temporal_pool_forward(self.base.temporal_pooling,tok,~safe_w)
  z=self.bridge(z).masked_fill((~wm.any(1))[:,None],0)
  tok=self.bridge(tok)*wm[:,:,None]
  return tok.float(),z.float()
def make_core(arm):
 if arm=='sleepfm_arch':return SleepAdapter()
 if arm=='no_condition':return NoCondition(n_datasets=48,geometry=True,native=False)
 m=original_core('full')
 if arm=='official_pool':m.chan_pool=PoolReplacement()
 assert arm in ['full','official_pool','ce_only']
 return m
