"""Isolated incremental study. Private material is only opened on the server."""
from pathlib import Path
import os,sys,json,hashlib,time
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
R=Path('/path/to/rehabfm-workspace/rehabfm_paper_completion_20261026/full_fixed_v1')
sys.path[:0]=[str(O),str(R),str(R/'vendor'),str(R.parent)]
ARMS=['full','no_condition','official_pool','ce_only','sleepfm_arch']
SEEDS=[0,1101,2202]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def progress(p,**v):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(updated_unix=time.time(),pid=os.getpid(),**v),indent=2));q.replace(p)
def verify():
 fr=read(O/'freeze.json')
 for p,h in fr['files'].items():assert sha(p)==h,p
 return fr
def inherit(arm,seed):return seed==0 and arm in ['full','ce_only']
def runpath(arm,seed):return R/'runs'/f'{arm}_f0' if inherit(arm,seed) else O/'runs'/f'{arm}_s{seed}'
def exact(a,b):
 import torch
 if torch.is_tensor(a):assert torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b
def device(b,dev):
 import torch
 return {k:v.to(dev) if torch.is_tensor(v) else device(v,dev) if isinstance(v,dict) else v for k,v in b.items()}
