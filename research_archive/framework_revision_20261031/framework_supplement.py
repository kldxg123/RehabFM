"""Frozen, post-hoc framework applicability study. All private data stay server-side.
No GPU, backbone updates, score-based selection or edits to completed studies.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,math,time,traceback,tarfile,subprocess
sys.dont_write_bytecode=True
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
R=Path('/path/to/rehabfm-workspace/rehabfm_paper_completion_20261026/full_fixed_v1')
O=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R/'vendor'),str(R.parent)]
from fixed_interfaces import read,sha,statistics,canonical_describe,metrics
from statistical_baselines import model as classical
TASKS=['moveport','hsstroke','llmi','psul','mgposture','strokedepth']
SEEDS=[1101,2202,3303]
FRACTIONS=[.25,.5,1.]
P=R/'fixed_eval_panels/neural_common_v1'
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2)
def progress(**v):
 q=O/'progress.tmp';q.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),**v),indent=2));q.replace(O/'progress.json')
def spec(t):return read(P/'models_for_fixed.json')['jobs']['Core_full_H__'+t]
def train_paths(t):return [R/'readouts_private/core_full'/f'{t}_training_features.npz',R/'statistical_fullfits_v1/training_private'/f'{t}_training.npz']
def arrays(t):
 a,b=train_paths(t)
 with np.load(a,allow_pickle=False) as z:F,y,g=z['F'],z['y'],z['group']
 with np.load(b,allow_pickle=False) as z:S=z['x'];assert np.array_equal(z['y'],y) and np.array_equal(z['group'],g)
 return S,F,y,g
def selection(g,f,seed):
 groups=np.random.default_rng(seed).permutation(np.unique(g));n=max(1,math.ceil(f*len(groups)))
 return np.flatnonzero(np.isin(g,groups[:n]))
def prepare():
 assert not (O/'freeze.json').exists()
 roles=read(R/'full_training_roles002.json')['tasks'];assert sorted(TASKS)==sorted(t for t,r in roles.items() if not r['core_training'])
 assert all(roles[t]['training_groups']>=4 for t in TASKS)
 paths=[Path(__file__),R/'full_training_roles002.json',P/'models_for_fixed.json',P/'fixed_evaluation_freeze.json',R/'full_training_source_freeze.json',R/'readouts_private/core_full/training_readout_receipt.json',R/'core_models.py',R/'vendor/rehabfm/model.py',R/'vendor/rehabfm/config.py',R/'fixed_interfaces.py',R.parent/'statistical_baselines.py']
 registry=read(P/'models_for_fixed.json')
 original=read(P/'fixed_evaluation_freeze.json')['files']
 for t in roles:
  s=spec(t);assert s['kind']=='core_readout' and s['effective_readout']=='H' and s['arm']=='full'
  paths.extend([Path(s[k]) for k in ['checkpoint_path','query_path','statistics_schema','readout_path']])
  cache=P/'feature_cache_private'/s['feature_cache_key'];paths.extend([cache/'seal.json',cache/'features.npz',P/'predictions_private'/('Core_full_H__'+t)/'seal.json'])
  if t in TASKS:paths.extend(train_paths(t))
  paths.append(R/'cohort_private'/f'{t}.pt')
 for name in ['full_training_roles002.json','core_models.py','vendor/rehabfm/model.py','vendor/rehabfm/config.py','fixed_interfaces.py']:
  p=R/name;assert str(p) in original and sha(p)==original[str(p)],str(p)
 # Generated group-budget checks use no project data or labels.
 g=np.repeat(np.arange(12),np.arange(1,13))
 for seed in SEEDS:
  a,b,c=[selection(g,f,seed) for f in FRACTIONS];assert set(a)<=set(b)<=set(c)==set(range(len(g)))
 policy=dict(status='frozen_before_new_fits',study='post_hoc_frozen_backbone_task_adaptation',tasks=TASKS,seeds=SEEDS,fractions=FRACTIONS,views=['S','F','H'],expected_fits=162,forest_trees=300,forest_max_features='sqrt',patient_group_sampling='same nested random group prefixes across S/F/H; ceil(fraction*groups); no class or score resampling',preprocessing='VarianceThreshold and StandardScaler fitted only to selected training groups',backbone='existing full 60-epoch model and features; zero updates; seed repetitions are readout/sampling seeds, not backbone seeds',test='previously exposed original fixed subset of 6 tasks / 467 records; exploratory, not independent confirmation',query_labels='read only after all 162 predictions are sealed',selection='no model, threshold, task, fraction or seed chosen from new results',resource='CPU only, one numerical/forest thread, nice19',limits='metadata group checks do not establish global identity; retained original K; no zero-shot claim',source_hashes={str(p):sha(p) for p in sorted(set(paths))},truth_contract_path=registry['original_test_contract'],truth_contract_sha256=original[registry['original_test_contract']],created_unix=time.time())
 save(O/'freeze.json',policy);print(json.dumps({k:policy[k] for k in ['status','expected_fits','tasks']}))
def check():
 fr=read(O/'freeze.json')
 for path,h in fr['source_hashes'].items():assert sha(path)==h,path
 return fr
def route_audit():
 from core_models import make_core
 from train_core import Items,digest
 from rehabfm.data import collate_fn
 torch.set_num_threads(1);torch.manual_seed(0)
 reg=read(P/'models_for_fixed.json');roles=read(R/'full_training_roles002.json')['tasks']
 state=torch.load(spec('moveport')['checkpoint_path'],map_location='cpu',weights_only=False)
 model=make_core('full');model.load_state_dict(state['model'],strict=True);model.eval();assert isinstance(model.encoder,torch.nn.TransformerEncoder) and len(model.encoder.layers)==3
 model.unseen_conditioning=dict(mode='zero',known_ids=sorted(map(int,state['schema'])),modality_rows={})
 before=digest(model);rows=[];calls=[]
 hook=model.encoder.register_forward_pre_hook(lambda m,a:calls.append(dict(batch=int(a[0].shape[0]),tokens=int(a[0].shape[1]))))
 for t in roles:
  s=spec(t);cache=P/'feature_cache_private'/s['feature_cache_key'];seal=read(cache/'seal.json')
  assert seal['feature_sha256']==sha(cache/'features.npz') and seal['checkpoint_sha256']==sha(s['checkpoint_path']) and seal['query_sha256']==sha(s['query_path'])
  with np.load(cache/'features.npz',allow_pickle=False) as z:assert z['F'].shape==(s['fixed_query_records'],256) and np.isfinite(z['F']).all()
  data=torch.load(R/'cohort_private'/f'{t}.pt',map_location='cpu',weights_only=False)
  # Runtime hooks on legal training examples covering every available modality.
  selected=[];seen=set()
  for i,item in enumerate(data['items']):
   mods=set(item['tokens'])
   if not mods<=seen:selected.append(i);seen|=mods
  total=0
  for i in selected:
   b=collate_fn([Items(data['items'])[i]]);calls.clear()
   with torch.no_grad():out=model(b)
   expected={m for m,v in b.items() if isinstance(v,dict) and 'x' in v}
   assert len(calls)==len(out['tokens'])==len(expected) and set(out['tokens'])==expected
   assert len(calls)>0 and all(c['tokens']==13 for c in calls)
   total+=len(calls)
  assert seen==set(roles[t]['modalities'])
  rows.append(dict(task=t,query_records=s['fixed_query_records'],readout='H',kind=s['kind'],fixed_feature_hash_verified=True,modalities=sorted(seen),legal_training_runtime_examples=len(selected),actual_encoder_calls=total,all_actual_modalities_reached_transformer=True))
 hook.remove();assert digest(model)==before==spec('moveport')['model_sha256']
 save(O/'path_audit.json',dict(status='passed',tasks=rows,external_tasks=22,internal_tasks=1,external_query_records=sum(x['query_records'] for x in rows if x['task']!='inhouse'),internal_raw_query_records=next(x['query_records'] for x in rows if x['task']=='inhouse'),neural_bypass_tasks=0,new_backbone_updates=0,new_fixed_predictions=0,evidence='all fixed cache seals verified plus trained-checkpoint runtime hooks on legal training examples covering every task/modality; not rerunning old fixed predictions',model_hash=before))
def run():
 check();assert not (O/'private').exists();(O/'private').mkdir();(O/'cases').mkdir();route_audit()
 roles=read(R/'full_training_roles002.json')['tasks'];cases=[]
 with threadpool_limits(limits=1):
  for t in TASKS:
   S,F,y,g=arrays(t);s=spec(t);q=torch.load(s['query_path'],map_location='cpu',weights_only=False)
   with np.load(P/'feature_cache_private'/s['feature_cache_key']/'features.npz',allow_pickle=False) as z:Fq=z['F'];indices=z['source_rows']
   assert np.array_equal(indices,np.asarray(q['original_row_indices'])) and all(r['y']==0 and r['subject']==-1 for r in q['items'])
   Sq=statistics(q['items'],read(s['statistics_schema']),canonical_describe());assert Sq.shape[1]==S.shape[1]
   np.savez_compressed(O/'private'/f'{t}_query_features.npz',S=Sq,F=Fq,source_rows=indices)
   for seed in SEEDS:
    for fraction in FRACTIONS:
     ix=selection(g,fraction,seed)
     for view in ['S','F','H']:
      key=f'{t}_{seed}_{int(fraction*100)}_{view}';progress(status='running',active=key,completed=len(cases),expected=162)
      X={'S':S,'F':F,'H':np.concatenate([S,F],1)}[view];Xq={'S':Sq,'F':Fq,'H':np.concatenate([Sq,Fq],1)}[view]
      estimator=classical('B4');estimator.set_params(extratreesclassifier__random_state=seed,extratreesclassifier__n_jobs=1);estimator.fit(X[ix],y[ix])
      pred=estimator.predict(Xq).astype(np.int64);assert set(pred)<=set(range(s['K'])) and len(pred)==s['fixed_query_records']
      modelp=O/'private'/f'{key}.joblib';joblib.dump(estimator,modelp,compress=3)
      assert np.array_equal(joblib.load(modelp).predict(Xq),pred)
      predp=O/'private'/f'{key}.npz';np.savez_compressed(predp,prediction=pred,source_rows=indices,training_indices=ix)
      item=dict(key=key,task=t,seed=seed,fraction=fraction,view=view,K=s['K'],fit_rows=len(ix),fit_groups=len(np.unique(g[ix])),available_groups=len(np.unique(g)),visible_training_classes=len(np.unique(y[ix])),original_classes=s['K'],feature_width=X.shape[1],model_sha256=sha(modelp),prediction_sha256=sha(predp),query_feature_sha256=sha(O/'private'/f'{t}_query_features.npz'),query_records=len(pred),truth_read_for_fit_or_selection=False)
      save(O/'cases'/f'{key}.json',item);cases.append(item)
 save(O/'all_predictions_sealed.json',dict(status='all_162_predictions_sealed_before_scoring',cases=cases,freeze_sha256=sha(O/'freeze.json')))
 progress(status='sealed_pending_independent_audit',completed=162,expected=162,active=None)
def audit():
 fr=check();sealed=read(O/'all_predictions_sealed.json');assert len(sealed['cases'])==162 and sealed['freeze_sha256']==sha(O/'freeze.json')
 assert sha(fr['truth_contract_path'])==fr['truth_contract_sha256'];contract=read(fr['truth_contract_path']);results=[]
 with threadpool_limits(limits=1):
  for c in sealed['cases']:
   key=c['key'];t=c['task'];S,F,y,g=arrays(t);ix=selection(g,c['fraction'],c['seed'])
   assert not set(g)&set(contract['external'][t]['group'])
   modelp=O/'private'/f'{key}.joblib';predp=O/'private'/f'{key}.npz';assert sha(modelp)==c['model_sha256'] and sha(predp)==c['prediction_sha256']
   qp=O/'private'/f'{t}_query_features.npz';assert sha(qp)==c['query_feature_sha256']
   with np.load(qp,allow_pickle=False) as z:Xq={'S':z['S'],'F':z['F'],'H':np.concatenate([z['S'],z['F']],1)}[c['view']]
   estimator=joblib.load(modelp);assert estimator[-1].n_estimators==300 and estimator[-1].random_state==c['seed']
   X={'S':S,'F':F,'H':np.concatenate([S,F],1)}[c['view']]
   assert np.array_equal(estimator[0].variances_,np.var(X[ix],axis=0)) or np.allclose(estimator[0].variances_,np.var(X[ix],axis=0),rtol=1e-6,atol=1e-10)
   assert np.allclose(estimator[1].mean_,estimator[0].transform(X[ix]).mean(0),rtol=1e-6,atol=1e-7)
   with np.load(predp,allow_pickle=False) as z:
    pred=z['prediction'];assert np.array_equal(z['training_indices'],ix) and np.array_equal(z['source_rows'],contract['external'][t]['source_rows']) and np.array_equal(estimator.predict(Xq),pred)
   truth=np.asarray(contract['external'][t]['truth']);cm=np.zeros((c['K'],c['K']),np.int64);np.add.at(cm,(truth,pred),1)
   m=metrics(truth,pred,c['K']);assert cm.tolist()==m['confusion'] and int(cm.trace())==m['correct']
   results.append(dict(**c,**m));progress(status='independent_audit',audited=len(results),expected=162)
 groups=[]
 for f in FRACTIONS:
  for v in ['S','F','H']:
   seedrows=[]
   for seed in SEEDS:
    rr=[r for r in results if r['fraction']==f and r['view']==v and r['seed']==seed];assert len(rr)==6 and sum(x['total'] for x in rr)==467
    seedrows.append(dict(seed=seed,accuracy=sum(x['correct'] for x in rr)/467,task_mean_BA=float(np.mean([x['balanced_accuracy'] for x in rr])),task_mean_macro_f1=float(np.mean([x['macro_f1'] for x in rr]))))
   groups.append(dict(fraction=f,view=v,records=467,mean_accuracy=float(np.mean([x['accuracy'] for x in seedrows])),sd_accuracy=float(np.std([x['accuracy'] for x in seedrows],ddof=1)),mean_task_BA=float(np.mean([x['task_mean_BA'] for x in seedrows])),seeds=seedrows))
 save(O/'audited_results.json',dict(status='passed',fits=162,results=results,summary=groups,original_query_records=467,backbone_seeds=1,readout_sampling_seeds=3,new_backbone_updates=0,all_predictions_restored_exactly=True,all_confusions_recomputed=True,all_training_group_budgets_verified=True,all_metadata_train_test_group_intersections_zero=True,scoring_after_all_predictions_sealed=True,exposed_benchmark=True,freeze_sha256=sha(O/'freeze.json')))
 progress(status='audited_complete_archive_pending',completed=162,expected=162)
def archive():
 assert read(O/'audited_results.json')['status']=='passed';arc=O.parent/(O.name+'_completed001.tar.gz');assert not arc.exists()
 paths=[p for p in O.rglob('*') if p.is_file() and p.name not in ['progress.json','manager.log','launch.json'] and p.suffix not in ['.pyc'] and '__pycache__' not in str(p)]
 hashes={str(p.relative_to(O)):sha(p) for p in paths};save(O/'archive_manifest.json',hashes)
 with tarfile.open(arc,'w:gz') as t:
  for p in paths:t.add(p,arcname=str(p.relative_to(O)))
 assert subprocess.run(['gzip','--test',str(arc)]).returncode==0
 with tarfile.open(arc,'r:gz') as t:
  assert len(t.getmembers())==len(hashes)
  for m in t:
   h=hashlib.sha256();f=t.extractfile(m)
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==hashes[m.name]
 save(O/'archive_receipt.json',dict(status='passed',archive=str(arc),sha256=sha(arc),bytes=arc.stat().st_size,files=len(hashes),gzip_test_returncode=0,all_payload_bytes_verified=True))
 progress(status='completed',completed=162,expected=162)
if __name__=='__main__':
 try:
  if sys.argv[1]=='all':run();audit();archive()
  else:globals()[sys.argv[1]]()
 except Exception as e:
  save(O/('failure_'+sys.argv[1]+'.json'),dict(status='failed_preserved',type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
