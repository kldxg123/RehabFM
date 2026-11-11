"""Separate scoring: only after all declared fixed predictions are sealed."""
from pathlib import Path
import json,sys
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from fixed_interfaces import sha,read,prediction_contract,patient_votes,metrics

def main():
 out=R/'fixed_results_pending_independent_audit.json';assert not out.exists();freeze=read(R/'fixed_evaluation_freeze.json');assert freeze['status']=='frozen_before_fixed_inference'
 assert freeze['files'][str(Path(__file__))]==sha(Path(__file__)) and freeze['files'][str(R/'fixed_interfaces.py')]==sha(R/'fixed_interfaces.py')
 registry=read(R/'models_for_fixed.json');assert freeze['files'][str(R/'models_for_fixed.json')]==sha(R/'models_for_fixed.json')
 assert registry['jobs'] and registry['expected_methods'] and {x['method'] for x in registry['jobs'].values()}==set(registry['expected_methods'])
 # First check every completion seal and file hash. Truth is opened only after
 # every declared method/task job has a complete, immutable prediction array.
 seals={}
 for key,spec in registry['jobs'].items():
  folder=R/'predictions_private'/key;seal=read(folder/'seal.json');assert seal['status']=='prediction_sealed_no_scoring' and seal['job']==key
  assert seal['evaluation_freeze_sha256']==sha(R/'fixed_evaluation_freeze.json') and seal['registry_sha256']==sha(R/'models_for_fixed.json')
  assert sha(folder/'prediction.npz')==seal['prediction_sha256'] and seal['records']==spec['fixed_query_records'];seals[key]=seal
 contract_path=Path(registry['original_test_contract']);assert freeze['files'][str(contract_path)]==sha(contract_path);contract=read(contract_path);results=[]
 for key,s in registry['jobs'].items():
  fixed=contract['clinical'] if s['task']=='inhouse' else contract['external'][s['task']];truth=np.asarray(fixed['truth'],np.int64);groups=np.asarray(fixed['group'],np.int64);expected=np.asarray(fixed['source_rows'],np.int64)
  with np.load(R/'predictions_private'/key/'prediction.npz',allow_pickle=False) as p:
   prediction=p['prediction'];prediction_contract(prediction,p['source_rows'],expected,s['K'],p['probability'] if 'probability' in p.files else None)
   assert len(truth)==len(prediction)
   if s['task']=='inhouse':
    yy,pp=patient_votes(truth,prediction,groups,s['K']);assert len(yy)==10
   else:yy,pp=truth,prediction
   results.append(dict(job=key,method=s['method'],task=s['task'],prediction_sha256=seals[key]['prediction_sha256'],**metrics(yy,pp,s['K'])))
 aggregate=[]
 for method in sorted({x['method'] for x in results}):
  rows=[x for x in results if x['method']==method];external=[x for x in rows if x['task']!='inhouse'];internal=[x for x in rows if x['task']=='inhouse']
  assert len(rows)==23 and len(external)==22 and len(internal)==1 and sum(x['total'] for x in external)==1731 and internal[0]['total']==10
  correct=sum(x['correct'] for x in external);aggregate.append(dict(method=method,external_correct=correct,external_total=1731,external_accuracy=correct/1731,internal_correct=internal[0]['correct'],internal_total=10,internal_accuracy=internal[0]['accuracy']))
 payload=dict(status='scored_pending_independent_audit',scope='original fixed external22/1731 and internal10 patients',methods=aggregate,task_results=results,true_query_labels_used_only_after_all_predictions_sealed=True,evaluation_freeze_sha256=sha(R/'fixed_evaluation_freeze.json'),original_test_contract_sha256=sha(contract_path),global_source_identity_certified=False,known_exposed_test=True,new_model_updates=0,no_test_score_model_selection=True)
 with out.open('x') as f:json.dump(payload,f,indent=2)
 print(json.dumps(dict(status=payload['status'],methods=len(aggregate),task_results=len(results))))
if __name__=='__main__':main()
