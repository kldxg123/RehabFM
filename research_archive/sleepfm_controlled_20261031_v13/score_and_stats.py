from common import *
import numpy as np
from fixed_interfaces import metrics,patient_votes
from fixed_statistics_math001 import holm,summarize_difference
P=R/'fixed_eval_panels/neural_common_v1'
def registry():
 reg=read(P/'models_for_fixed.json');cases=[]
 for seed in SEEDS:
  for arm in ARMS:
   if inherit(arm,seed):
    for task in read(R/'full_training_roles002.json')['tasks']:
     for view in ['F','H']:
      key=f'Core_{arm}_{view}__{task}';s=reg['jobs'][key];folder=P/'predictions_private'/key;seal=read(folder/'seal.json');assert sha(folder/'prediction.npz')==seal['prediction_sha256']
      cases.append(dict(arm=arm,seed=seed,task=task,view=view,K=s['K'],prediction_path=str(folder/'prediction.npz'),prediction_sha256=seal['prediction_sha256'],inherited=True))
   else:
    folder=O/'readouts_private'/f'{arm}_s{seed}';a=read(folder/'independent_prediction_audit.json');assert a['status']=='passed' and a['sealed_sha256']==sha(folder/'all_sealed.json');cases+=read(folder/'all_sealed.json')['cases']
 assert len(cases)==690
 for c in cases:assert sha(c['prediction_path'])==c['prediction_sha256']
 return reg,cases
def paired(curves,truth):
 pairs=[(f'full_{v}',f'{a}_{v}') for a in ARMS if a!='full' for v in ['F','H']]+[('full_H','S'),('full_H','full_F')]
 results=[]
 for scope in ['external','internal']:
  for j,(a,b) in enumerate(pairs):
   rng=np.random.default_rng(20261031+j);boot=np.zeros(10000);perm=np.zeros(10000);point=0.
   tasks=[t for t in truth if (t=='inhouse')==(scope=='internal')];N=10 if scope=='internal' else 1731
   for t in tasks:
    g=truth[t]['group'];diff=curves[a][t]-curves[b][t];ug=np.unique(g);sizes=np.array([(g==q).sum() for q in ug]);tot=np.array([diff[g==q].sum() for q in ug]);ix=rng.integers(len(ug),size=(10000,len(ug)));sign=rng.choice([-1,1],size=(10000,len(ug)))
    point+=diff.sum()/N;boot+=tot[ix].sum(1)/sizes[ix].sum(1)*len(g)/N;perm+=(tot[None,:]*sign).sum(1)/N
   results.append(dict(scope=scope,left=a,right=b,**summarize_difference(point,boot,perm)))
  adjusted=holm([x['p_unadjusted'] for x in results if x['scope']==scope])
  for x,p in zip([x for x in results if x['scope']==scope],adjusted):x['holm_p']=float(p)
 return results
def main():
 fr=verify();reg,cases=registry();save(O/'all_predictions_sealed.json',dict(status='all_690_sealed_before_truth',cases=cases,freeze_sha256=sha(O/'freeze.json')))
 contract=read(reg['original_test_contract']);rows=[];correct={};truth={}
 for c in cases:
  t=c['task'];f=contract['clinical'] if t=='inhouse' else contract['external'][t];y=np.array(f['truth']);g=np.array(f['group'])
  with np.load(c['prediction_path']) as z:p=z['prediction'];assert np.array_equal(z['source_rows'],f['source_rows'])
  if t=='inhouse':y,p=patient_votes(y,p,g,c['K']);g=np.unique(g);assert len(y)==10
  truth[t]=dict(y=y,group=g);correct.setdefault((c['arm'],c['view'],c['seed']),{})[t]=(y==p).astype(float)
  rows.append(dict(arm=c['arm'],seed=c['seed'],task=t,view=c['view'],K=c['K'],**metrics(y,p,c['K'])))
 curves={}
 for a in ARMS:
  for v in ['F','H']:curves[f'{a}_{v}']={t:np.mean([correct[(a,v,s)][t] for s in SEEDS],axis=0) for t in truth}
 curves['S']={}
 for t in truth:
  # Exact previously audited common-input S predictions, no refitting.
  s=reg['jobs']['Core_full_S__'+t];pp=P/'predictions_private'/f'Core_full_S__{t}'/'prediction.npz';seal=read(pp.parent/'seal.json');assert sha(pp)==seal['prediction_sha256']
  with np.load(pp) as z:p=z['prediction']
  f=contract['clinical'] if t=='inhouse' else contract['external'][t];y=np.array(f['truth']);g=np.array(f['group'])
  if t=='inhouse':y,p=patient_votes(y,p,g,s['K'])
  curves['S'][t]=(y==p).astype(float)
 agg=[]
 for a in ARMS:
  for v in ['F','H']:
   seeds=[]
   for s in SEEDS:
    ext=[r for r in rows if (r['arm'],r['view'],r['seed'])==(a,v,s) and r['task']!='inhouse'];inn=next(r for r in rows if (r['arm'],r['view'],r['seed'],r['task'])==(a,v,s,'inhouse'));assert len(ext)==22 and sum(r['total'] for r in ext)==1731
    seeds.append(dict(seed=s,correct=sum(r['correct'] for r in ext),total=1731,accuracy=sum(r['correct'] for r in ext)/1731,internal_correct=inn['correct'],internal_total=10,mean_task_BA=float(np.mean([r['balanced_accuracy'] for r in ext]))))
   agg.append(dict(arm=a,view=v,seeds=seeds,external_mean=float(np.mean([r['accuracy'] for r in seeds])),external_sd=float(np.std([r['accuracy'] for r in seeds],ddof=1))))
 stats=paired(curves,truth)
 save(O/'results_pending_audit.json',dict(status='scored_pending_independent_audit',task_results=rows,aggregate=agg,paired_statistics=stats,original_fixed_external_records=1731,internal_patients=10,exploratory_exposed_benchmark=True,backbone_seeds=SEEDS,all_original_classes_retained=True,no_best_seed_selection=True))
 # Private score vectors stay on server and support independent statistical recomputation.
 np.savez_compressed(O/'score_vectors_private.npz',**{f'{m}__{t}':x for m,d in curves.items() for t,x in d.items()},**{f'groups__{t}':d['group'] for t,d in truth.items()})
if __name__=='__main__':main()
