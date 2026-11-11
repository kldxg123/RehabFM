"""Published UniTS supervised-x64 model, explicit waveform input adapter.

Official UniTS and selected timm layer files are loaded without edits.
Only the timm package namespace is assembled here, avoiding torchvision imports
and any installation/change to the environment of running experiments.
"""
import sys,types,importlib.util,importlib,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def official_model():
 layers=R/'vendor/timm_layers_v1_0_22/timm/layers'
 receipt=json.loads((layers.parents[1]/'SOURCE_RECEIPT.json').read_text())
 for x in receipt['files']:assert sha(layers.parents[1]/x['path'])==x['sha256'],x['path']
 if 'timm' not in sys.modules:
  pkg=types.ModuleType('timm');pkg.__path__=[str(layers.parent)];sys.modules['timm']=pkg
  lp=types.ModuleType('timm.layers');lp.__path__=[str(layers)];sys.modules['timm.layers']=lp
  lp.Mlp=importlib.import_module('timm.layers.mlp').Mlp
  lp.DropPath=importlib.import_module('timm.layers.drop').DropPath
 p=R/'vendor/UniTS/models/UniTS.py';rec=json.loads((R/'vendor/UniTS/SOURCE_RECEIPT.json').read_text())
 assert sha(p)==next(x['sha256'] for x in rec['files'] if x['path']=='models/UniTS.py')
 spec=importlib.util.spec_from_file_location('official_units_pinned',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.Model
def make_model(schemas):
 args=SimpleNamespace(prompt_num=10,d_model=64,stride=16,patch_len=16,dropout=.1,e_layers=3,n_heads=8)
 configs=[(s['task'],dict(dataset=s['task'],enc_in=sum(s['channels'].values()),task_name='classification',num_class=s['K'])) for s in schemas]
 return official_model()(args,configs,pretrain=False)
def fit_schema(task,rows,K):
 # Only legal fold-training rows determine modality/channel slots.
 channels={}
 for r in rows:
  for m,t in r['tokens'].items():channels[m]=max(channels.get(m,0),t.shape[1])
 return dict(task=task,K=K,channels=dict(sorted(channels.items())),window_samples=128,windows=12)
def transform(row,schema):
 # No labels, IDs or fitted statistics are accessed by this transform.
 tokens=row['tokens']  # Unseen modalities have no fitted channel slots and are ignored.
 arrays=[]
 for m,C in schema['channels'].items():
  a=torch.zeros(12,C,128)
  if m in tokens:
   x=torch.as_tensor(tokens[m],dtype=torch.float32);assert x.ndim==3 and x.shape[0]==12 and torch.isfinite(x).all()
   n=min(C,x.shape[1]);a[:,:n]=F.adaptive_avg_pool1d(x[:,:n],128)
  arrays.append(a)
 out=torch.cat(arrays,dim=1).permute(0,2,1).reshape(1536,-1).contiguous()
 assert torch.isfinite(out).all();return out
