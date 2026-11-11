"""Prediction-only neural fixed panel, including explicit exact-result reuse.

The source stays in the full-fit tree so official-model relative paths retain
their identities. The separately frozen evaluation panel is an argument.
No fitting, gradients, original truth, patient IDs or query-set statistics.
"""
from pathlib import Path
import argparse,sys,json,os,re,shutil
import numpy as np
import torch,joblib
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R/'vendor'),str(R.parent)]
from fixed_interfaces import read,sha,prediction_contract,canonical_describe,statistics
from full_model_io002 import make_architecture,state_digest
from train_core import Items,device
from rehabfm.data import collate_fn
from torch.utils.data import DataLoader


def wr(path,value):
 with Path(path).open('x') as h:json.dump(value,h,indent=2)


def extract_core(model,rows,factory,batch_size):
 features=[];tokens=[]
 with torch.no_grad():
  for raw in DataLoader(Items(rows),batch_size=batch_size,shuffle=False,collate_fn=collate_fn,num_workers=0):
   batch=device(raw);out=model(batch);B=len(batch['y'])
   if factory=='fusion':z=out['fusion_z'];t=out['fusion_T']
   else:
    from core_models import fuse
    z=fuse(out,B);first=next(iter(out['tokens'].values()));t=first.new_zeros(B,12,64);count=first.new_zeros(B)
    for mod,v in out['tokens'].items():t.index_add_(0,out['owners'][mod],v);count.index_add_(0,out['owners'][mod],count.new_ones(len(v)))
    assert (count>0).all();t=t/count[:,None,None]
   f=torch.cat([z,t.mean(1),t.std(1,unbiased=False),t.max(1).values],1)
   assert f.shape==(B,256) and t.shape==(B,12,64) and torch.isfinite(f).all() and torch.isfinite(t).all()
   features.append(f.cpu().numpy());tokens.append(t.cpu().numpy())
 return np.concatenate(features),np.concatenate(tokens)


def main(panel,job):
 P=Path(panel).resolve();assert P.parent==R/'fixed_eval_panels' and re.fullmatch('[A-Za-z0-9_-]+',job)
 freeze=read(P/'fixed_evaluation_freeze.json');assert freeze['status']=='frozen_before_fixed_inference'
 def check(path):
  p=Path(path);assert str(p) in freeze['files'] and sha(p)==freeze['files'][str(p)],str(p);return p
 check(Path(__file__));check(R/'full_model_io002.py');check(R/'fixed_interfaces.py');check(P/'models_for_fixed.json')
 registry=read(P/'models_for_fixed.json');spec=registry['jobs'][job];out=P/'predictions_private'/job;assert not out.exists();out.mkdir(parents=True)
 query=check(spec['query_path']);q=torch.load(query,map_location='cpu',weights_only=False);rows=q['items']
 assert q['task']==spec['task'] and len(rows)==spec['fixed_query_records'] and all(x['y']==0 and x['subject']==-1 for x in rows)
 expected=np.asarray(q['original_row_indices']);prob=None;cache_sha=None;source_reuse=None
 torch.set_num_threads(3)
 if spec['kind'].startswith('inherit_'):
  if spec['kind']=='inherit_statistical':
   source=check(spec['reference_prediction']);seal=read(check(spec['reference_seal']))
   assert seal['task']==spec['task'] and seal['method']=='B4' and seal['records']==len(rows)
   assert seal['status']=='prediction_sealed_no_scoring' and seal['query_input_sha256']==sha(query)
  else:
   assert spec['kind']=='inherit_job';source=P/'predictions_private'/spec['reference_job']/'prediction.npz';seal=read(source.parent/'seal.json')
   assert seal['task']==spec['task'] and seal['records']==len(rows) and seal['status']=='prediction_sealed_no_scoring'
   assert seal['evaluation_freeze_sha256']==sha(P/'fixed_evaluation_freeze.json') and seal['registry_sha256']==sha(P/'models_for_fixed.json')
  assert sha(source)==seal['prediction_sha256']
  with np.load(source,allow_pickle=False) as pack:
   pred=pack['prediction'];prob=pack['probability'] if 'probability' in pack.files else None
   prediction_contract(pred,pack['source_rows'],expected,spec['K'],prob)
   assert set(pack.files)<=set(['prediction','probability','source_rows'])
  shutil.copyfile(source,out/'prediction.npz');assert sha(out/'prediction.npz')==sha(source);source_reuse=dict(path=str(source),sha256=sha(source));new_forwards=0
 else:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2';torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0);torch.cuda.manual_seed_all(0)
  cp=check(spec['checkpoint_path']);state=torch.load(cp,map_location='cpu',weights_only=False);assert state['epoch']==60
  if spec['kind']=='direct':
   model=make_architecture('direct',spec['family'],spec['K'],state.get('modalities'));model.load_state_dict(state['model'],strict=True);model=model.cuda().eval();before=state_digest(model.state_dict());parts=[]
   with torch.no_grad():
    for raw in DataLoader(Items(rows),batch_size=spec['batch_size'],shuffle=False,collate_fn=collate_fn,num_workers=0):parts.append(model(device(raw)).float().softmax(-1).cpu().numpy())
   prob=np.concatenate(parts);pred=prob.argmax(1);assert state_digest(model.state_dict())==before==spec['model_sha256']
  elif spec['kind']=='units':
   model=make_architecture('units',schemas=state['schemas']);model.load_state_dict(state['model'],strict=True);model=model.cuda().eval();before=state_digest(model.state_dict())
   from full_model_io002 import load_source
   transform=load_source('fixed_units_transform',R/'units_v3/adapter.py').transform
   tid=next(i for i,s in enumerate(state['schemas']) if s['task']==spec['task']);schema=state['schemas'][tid];assert schema['K']==spec['K'];parts=[]
   with torch.no_grad():
    for start in range(0,len(rows),spec['batch_size']):
     x=torch.stack([transform(r,schema) for r in rows[start:start+spec['batch_size']]]).cuda();parts.append(model.classification(x,None,tid).float().softmax(-1).cpu().numpy())
   prob=np.concatenate(parts);pred=prob.argmax(1);assert state_digest(model.state_dict())==before==spec['model_sha256']
  else:
   assert spec['kind']=='core_readout';key=spec['feature_cache_key'];assert re.fullmatch('[A-Za-z0-9_-]+',key)
   cache=P/'feature_cache_private'/key
   if cache.exists():
    cs=read(cache/'seal.json');assert cs['status']=='frozen_query_features_no_truth' and cs['query_sha256']==sha(query) and cs['checkpoint_sha256']==sha(cp) and cs['model_sha256']==spec['model_sha256'] and cs['batch_size']==spec['batch_size'] and cs['evaluation_freeze_sha256']==sha(P/'fixed_evaluation_freeze.json')
    assert cs['feature_sha256']==sha(cache/'features.npz')
    with np.load(cache/'features.npz',allow_pickle=False) as z:
     assert set(z.files)=={'F','T','source_rows'} and np.array_equal(z['source_rows'],expected);F=z['F'];T=z['T']
   else:
    cache.mkdir(parents=True);model=make_architecture(spec['core_factory'],spec['arm']);model.load_state_dict(state['model'],strict=True);model=model.cuda().eval();model.unseen_conditioning=dict(mode='zero',known_ids=sorted(map(int,state['schema'])),modality_rows={});before=state_digest(model.state_dict())
    F,T=extract_core(model,rows,spec['core_factory'],spec['batch_size']);assert state_digest(model.state_dict())==before==spec['model_sha256']
    np.savez_compressed(cache/'features.npz',F=F,T=T,source_rows=expected)
    wr(cache/'seal.json',dict(status='frozen_query_features_no_truth',query_sha256=sha(query),checkpoint_sha256=sha(cp),model_sha256=before,batch_size=spec['batch_size'],feature_sha256=sha(cache/'features.npz'),evaluation_freeze_sha256=sha(P/'fixed_evaluation_freeze.json'),labels_or_patient_ids_used=False,new_gradient_updates=0))
   assert F.shape==(len(rows),256) and T.shape==(len(rows),12,64) and np.isfinite(F).all() and np.isfinite(T).all();cache_sha=sha(cache/'features.npz')
   view=spec['effective_readout'];readout=check(spec['readout_path'])
   if view=='T':
    from evaluate_core import SequenceHead
    st=torch.load(readout,map_location='cpu',weights_only=False);assert st['epochs']==150;head=SequenceHead().cuda();head.load_state_dict(st['model'],strict=True);head.eval();x=torch.as_tensor(T,device='cuda');x=((x-st['mean'].cuda())/st['std'].cuda()).clamp(-10,10)
    with torch.no_grad():prob=head(x).softmax(-1).cpu().numpy()
    pred=prob.argmax(1)
   else:
    assert view in ['F','H'];x=F
    if view=='H':x=np.concatenate([statistics(rows,read(check(spec['statistics_schema'])),canonical_describe()),F],1)
    estimator=joblib.load(readout);pred=estimator.predict(x).astype(np.int64)
  prediction_contract(pred,expected,expected,spec['K'],prob)
  data=dict(prediction=pred.astype(np.int64),source_rows=expected)
  if prob is not None:data['probability']=prob
  np.savez_compressed(out/'prediction.npz',**data);new_forwards=1
 prediction_contract(pred,expected,expected,spec['K'],prob)
 seal=dict(status='prediction_sealed_no_scoring',job=job,method=spec['method'],task=spec['task'],K=spec['K'],records=len(rows),prediction_sha256=sha(out/'prediction.npz'),query_input_sha256=sha(query),registry_sha256=sha(P/'models_for_fixed.json'),evaluation_freeze_sha256=sha(P/'fixed_evaluation_freeze.json'),true_query_labels_read_by_predictor=False,patient_ids_used_by_predictor=False,new_model_updates=0,new_prediction_jobs=new_forwards,exact_reuse=source_reuse,feature_cache_sha256=cache_sha,accuracy_computed=False)
 wr(out/'seal.json',seal);print(json.dumps({k:seal[k] for k in ['status','job','records','new_prediction_jobs','accuracy_computed']}))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--job',required=True);a=p.parse_args();main(a.panel,a.job)
