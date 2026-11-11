"""Independent complete neural fixed-panel prediction and metric audit."""
from pathlib import Path
import argparse,json,sys,datetime
import numpy as np
import torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from fixed_interfaces import read,sha
P=R/'fixed_eval_panels/neural_common_v1'


def inputs():
 freeze=read(P/'fixed_evaluation_freeze.json');assert freeze['status']=='frozen_before_fixed_inference'
 for p,h in freeze['files'].items():assert sha(p)==h,p
 reg=read(P/'models_for_fixed.json');assert len(reg['jobs'])==805 and len(set(reg['expected_methods']))==35
 for method in reg['expected_methods']:
  v=[x for x in reg['jobs'].values() if x['method']==method];assert len(v)==23 and len({x['task'] for x in v})==23
  assert sum(x['fixed_query_records'] for x in v if x['task']!='inhouse')==1731 and next(x['fixed_query_records'] for x in v if x['task']=='inhouse')==163
 return reg


def predictions():
 dest=P/'independent_prediction_audit.json';assert not dest.exists();reg=inputs();queries={};checks=[];fresh=0;stat=0;mix=0;caches=set()
 for key,spec in reg['jobs'].items():
  task=spec['task']
  if task not in queries:
   q=torch.load(spec['query_path'],map_location='cpu',weights_only=False);assert q['task']==task and len(q['items'])==spec['fixed_query_records'] and all(x['y']==0 and x['subject']==-1 for x in q['items']);queries[task]=np.asarray(q['original_row_indices'])
  folder=P/'predictions_private'/key;seal=read(folder/'seal.json');target=folder/'prediction.npz'
  assert seal['status']=='prediction_sealed_no_scoring' and seal['job']==key and seal['method']==spec['method'] and seal['task']==task and seal['K']==spec['K'] and seal['records']==spec['fixed_query_records']
  assert seal['new_model_updates']==0 and seal['true_query_labels_read_by_predictor'] is False and seal['patient_ids_used_by_predictor'] is False
  assert seal['registry_sha256']==sha(P/'models_for_fixed.json') and seal['evaluation_freeze_sha256']==sha(P/'fixed_evaluation_freeze.json') and seal['query_input_sha256']==sha(spec['query_path']) and seal['prediction_sha256']==sha(target)
  with np.load(target,allow_pickle=False) as z:
   assert set(z.files)<=set(['prediction','probability','source_rows']);p=z['prediction'];assert p.dtype==np.int64 and p.shape==(spec['fixed_query_records'],) and np.array_equal(z['source_rows'],queries[task]) and ((0<=p)&(p<spec['K'])).all()
   if 'probability' in z.files:
    a=z['probability'];assert a.shape==(len(p),spec['K']) and np.isfinite(a).all() and (a>=0).all() and np.allclose(a.sum(1),1,atol=1e-5) and np.array_equal(a.argmax(1),p)
  if spec['kind'].startswith('inherit_'):
   assert seal['new_prediction_jobs']==0 and seal['exact_reuse']
   source=Path(spec['reference_prediction']) if spec['kind']=='inherit_statistical' else P/'predictions_private'/spec['reference_job']/'prediction.npz'
   assert sha(source)==sha(target)==seal['exact_reuse']['sha256'];stat+=spec['kind']=='inherit_statistical';mix+=spec['kind']=='inherit_job'
  else:assert seal['new_prediction_jobs']==1 and seal['exact_reuse'] is None;fresh+=1
  if spec['kind']=='core_readout':
   cache=P/'feature_cache_private'/spec['feature_cache_key'];cs=read(cache/'seal.json');assert cs['model_sha256']==spec['model_sha256'] and cs['checkpoint_sha256']==sha(spec['checkpoint_path']) and cs['query_sha256']==sha(spec['query_path'])
   assert cs['labels_or_patient_ids_used'] is False and cs['feature_sha256']==seal['feature_cache_sha256']==sha(cache/'features.npz')
   with np.load(cache/'features.npz',allow_pickle=False) as z:assert set(z.files)=={'F','T','source_rows'} and z['F'].shape==(len(queries[task]),256) and z['T'].shape==(len(queries[task]),12,64) and np.array_equal(z['source_rows'],queries[task]) and np.isfinite(z['F']).all() and np.isfinite(z['T']).all()
   caches.add(spec['feature_cache_key'])
  checks.append(dict(job=key,records=spec['fixed_query_records'],prediction_sha256=sha(target),kind=spec['kind'],original_rows_and_classes_preserved=True))
 assert (fresh,stat,mix,len(caches))==(490,161,154,161)
 result=dict(status='passed',jobs=805,methods=35,new_prediction_jobs=490,exact_statistical_reuses=161,exact_within_panel_reuses=154,core_feature_caches=161,checks=checks,truth_file_opened=False,new_fits=0,new_gradients=0,source_sha256=sha(Path(__file__)))
 with dest.open('x') as h:json.dump(result,h,indent=2)
 return dict(status='passed',jobs=805,truth_file_opened=False)


def metric(y,p,K):
 cm=np.zeros((K,K),np.int64)
 for a,b in zip(y,p):assert 0<=int(a)<K and 0<=int(b)<K;cm[int(a),int(b)]+=1
 s=cm.sum(1);c=cm.sum(0);d=cm.diagonal();f=np.divide(2*d,s+c,out=np.zeros(K,float),where=(s+c)>0)
 return dict(correct=int(d.sum()),total=int(cm.sum()),accuracy=float(d.sum()/cm.sum()),balanced_accuracy=float(np.mean(d[s>0]/s[s>0])),macro_f1=float(f.mean()),confusion=cm.tolist())


def scores():
 dest=P/'independent_result_audit.json';final=P/'audited_fixed_results.json';assert not dest.exists() and not final.exists();reg=inputs();assert read(P/'independent_prediction_audit.json')['status']=='passed'
 result=read(P/'fixed_results_pending_independent_audit.json');assert result['status']=='scored_pending_independent_audit' and len(result['task_results'])==805 and {x['method'] for x in result['methods']}==set(reg['expected_methods'])
 found={x['job']:x for x in result['task_results']};assert set(found)==set(reg['jobs']);contract=read(reg['original_test_contract']);checks=[]
 for key,spec in reg['jobs'].items():
  c=contract['clinical'] if spec['task']=='inhouse' else contract['external'][spec['task']];y=np.asarray(c['truth'],np.int64);g=np.asarray(c['group'],np.int64)
  with np.load(P/'predictions_private'/key/'prediction.npz',allow_pickle=False) as z:
   assert np.array_equal(z['source_rows'],np.asarray(c['source_rows'])) and len(y)==spec['fixed_query_records'];p=z['prediction']
   if spec['task']=='inhouse':
    yy=[];pp=[]
    for u in sorted(set(g.tolist())):
     mask=g==u;truth=set(y[mask].tolist());assert len(truth)==1;counts=[int((p[mask]==i).sum()) for i in range(spec['K'])];yy.append(next(iter(truth)));pp.append(counts.index(max(counts)))
    assert len(yy)==10;y=np.asarray(yy);p=np.asarray(pp)
   m=metric(y,p,spec['K']);actual=found[key]
   for name,value in m.items():
    if isinstance(value,float):assert abs(actual[name]-value)<=1e-12,(key,name)
    else:assert actual[name]==value,(key,name)
   checks.append(dict(method=spec['method'],task=spec['task'],**m))
 for row in result['methods']:
  rows=[x for x in checks if x['method']==row['method']];ext=[x for x in rows if x['task']!='inhouse'];c=next(x for x in rows if x['task']=='inhouse');assert len(ext)==22 and sum(x['total'] for x in ext)==1731
  assert row['external_correct']==sum(x['correct'] for x in ext) and row['external_total']==1731 and row['internal_correct']==c['correct'] and row['internal_total']==10
  assert abs(row['external_accuracy']-row['external_correct']/1731)<1e-15 and abs(row['internal_accuracy']-c['correct']/10)<1e-15
 audit=dict(status='passed',methods=35,jobs=805,external_tasks=22,external_fixed_records_per_method=1731,internal_fixed_patients=10,metrics_independently_recomputed=True,new_fits=0,new_predictions=0,result_sha256=sha(P/'fixed_results_pending_independent_audit.json'),source_sha256=sha(Path(__file__)),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with dest.open('x') as h:json.dump(audit,h,indent=2)
 result.update(status='passed_independent_fixed_test_audit',independent_result_audit_sha256=sha(dest),qualifications=reg['qualifications'],input_panel='common-waveform, full legal training')
 with final.open('x') as h:json.dump(result,h,indent=2)
 return dict(status='passed',methods=35,jobs=805)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['predictions','scores']);print(json.dumps(predictions() if p.parse_args().action=='predictions' else scores()))
