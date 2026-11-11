"""Training-only input schema and strict fixed-prediction contracts.

No model fitting, fixed truth reads or scientific result selection in this file.
"""
from pathlib import Path
import ast,hashlib,json
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def canonical_describe():
 from scipy.signal import resample
 from rehabfm.config import MODALITIES
 p=Path('/path/to/rehabfm-workspace/rehabfm_methods_v2_20261015/extract_spatial.py')
 assert sha(p)=='e373161ac678d9aeba6a2ba97c56ec5911c87d883bab835e973c9a7d7976e38d'
 node=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='describe')
 ns=dict(np=np,resample=resample,MODALITIES=MODALITIES);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 return ns['describe']

def fit_statistics_schema(training_rows,describe):
 """Only legal training-token shapes determine named modality/channel slots."""
 caps={};lengths={}
 for row in training_rows:
  for mod,value in row['tokens'].items():
   a=np.asarray(value);assert a.ndim==3 and a.shape[0]==12 and a.shape[1]>0 and np.isfinite(a).all()
   caps[mod]=max(caps.get(mod,0),a.shape[1]);lengths.setdefault(mod,a.shape[2])
 assert caps
 result=[]
 for mod,C in sorted(caps.items()):
  template=np.ones((12,C,lengths[mod]),np.float32);width=len(describe(template,mod));assert width>0
  result.append(dict(modality=mod,channels=C,width=width))
 return result

def statistics(rows,schema,describe):
 """No query IDs, labels or collection statistics; per-record fixed forward."""
 out=[]
 for row in rows:
  parts=[]
  for s in schema:
   mod=s['modality'];C=s['channels'];width=s['width']
   if mod not in row['tokens']:v=np.zeros(width,np.float32)
   else:
    a=np.asarray(row['tokens'][mod],dtype=np.float32);assert a.ndim==3 and a.shape[0]==12 and np.isfinite(a).all()
    # Align channels before descriptor blocks/covariances, not by cutting a
    # concatenated descriptor halfway through a different feature section.
    x=np.zeros((12,C,a.shape[-1]),np.float32);keep=min(C,a.shape[1]);x[:,:keep]=a[:,:keep]
    if not np.any(x):v=np.zeros(width,np.float32)
    else:v=describe(x,mod);assert v.shape==(width,)
   parts.append(v)
  out.append(np.concatenate(parts))
 matrix=np.stack(out);assert np.isfinite(matrix).all();return matrix

def prediction_contract(prediction,source_rows,expected_rows,K,probability=None):
 p=np.asarray(prediction);ix=np.asarray(source_rows);expected=np.asarray(expected_rows)
 assert p.ndim==1 and len(p)==len(expected)>0 and np.array_equal(ix,expected)
 assert np.isfinite(p).all() and np.equal(p,p.astype(np.int64)).all() and ((p>=0)&(p<K)).all()
 if probability is not None:
  a=np.asarray(probability);assert a.shape==(len(p),K) and np.isfinite(a).all() and (a>=0).all() and np.allclose(a.sum(1),1,atol=1e-5)
  assert np.array_equal(p,a.argmax(1))
 return True

def patient_votes(truth,prediction,groups,K):
 """Scoring-only: retain original lowest-class tie rule, one vote/patient."""
 y=np.asarray(truth);p=np.asarray(prediction);g=np.asarray(groups);yy=[];pp=[]
 for subject in np.unique(g):
  mask=g==subject;labels=np.unique(y[mask]);assert len(labels)==1
  yy.append(int(labels[0]));pp.append(int(np.bincount(p[mask],minlength=K).argmax()))
 return np.asarray(yy),np.asarray(pp)

def metrics(truth,prediction,K):
 from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
 y=np.asarray(truth);p=np.asarray(prediction);cm=confusion_matrix(y,p,labels=np.arange(K));present=cm.sum(1)>0
 return dict(correct=int((y==p).sum()),total=len(y),accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(np.mean(np.diag(cm)[present]/cm.sum(1)[present])),macro_f1=float(f1_score(y,p,labels=np.arange(K),average='macro',zero_division=0)),confusion=cm.tolist())
