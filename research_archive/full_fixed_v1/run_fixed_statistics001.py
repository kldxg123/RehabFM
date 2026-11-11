"""Locked analysis of audited actual fixed predictions, never OOF/inference."""
from pathlib import Path
import sys,json,datetime
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from fixed_interfaces import read,sha,patient_votes
from fixed_statistics_math001 import cluster_draws,holm,mcnemar,summarize_difference
O=R/'fixed_statistics_v1';N=R/'fixed_eval_panels/neural_common_v1';C=R/'fixed_eval_panels/statistics_common_v1'


def main():
 assert not O.exists()
 # Complete all model/result audits before opening any patient prediction/truth.
 for panel,jobs in [(N,805),(C,115)]:
  assert read(panel/'independent_prediction_audit.json')['status']=='passed' and read(panel/'independent_prediction_audit.json')['jobs']==jobs
  assert read(panel/'independent_result_audit.json')['status']=='passed' and read(panel/'audited_fixed_results.json')['status']=='passed_independent_fixed_test_audit'
 lock=read(R/'fixed_statistics_plan001_freeze.json');assert lock['plan_sha256']==sha(R/'fixed_statistics_plan001.json')
 plan=read(R/'fixed_statistics_plan001.json');assert len(plan['pairs'])==36 and plan['draws']==10000
 nr=read(N/'models_for_fixed.json');cr=read(C/'models_for_fixed.json');assert nr['original_test_contract']==cr['original_test_contract'];contract_path=Path(nr['original_test_contract'])
 models=sorted({x[k] for x in plan['pairs'] for k in ['a','b']});paths={};files=[Path(__file__),R/'audit_fixed_statistics_results001.py',R/'fixed_statistics_math001.py',R/'fixed_interfaces.py',R/'fixed_statistics_plan001.json',R/'fixed_statistics_plan001_freeze.json',R/'neural_evaluation_plan003.json',N/'models_for_fixed.json',C/'models_for_fixed.json',contract_path]
 for panel in [N,C]:files += [panel/n for n in ['fixed_evaluation_freeze.json','independent_prediction_audit.json','independent_result_audit.json','audited_fixed_results.json']]
 tasks=plan['partitions']['all_external']+['inhouse']
 for model in models:
  panel=C if model in ['B0','B1','B2','B3','B4'] else N;registry=cr if panel==C else nr
  for task in tasks:
   candidates=[(k,v) for k,v in registry['jobs'].items() if v['method']==model and v['task']==task];assert len(candidates)==1;key,spec=candidates[0];folder=panel/'predictions_private'/key
   seal=read(folder/'seal.json');assert seal['status']=='prediction_sealed_no_scoring' and seal['prediction_sha256']==sha(folder/'prediction.npz')
   paths[(model,task)]=folder/'prediction.npz';files += [folder/'prediction.npz',folder/'seal.json']
 O.mkdir();private=O/'private';private.mkdir()
 freeze=dict(status='frozen_before_fixed_statistics_computation',files={str(p):sha(p) for p in sorted(set(files))},models=models,comparisons=36,external_tests=108,internal_tests=36,new_inference=False,new_training=False)
 with (O/'source_freeze.json').open('x') as h:json.dump(freeze,h,indent=2)
 contract=read(contract_path);rng=np.random.default_rng(plan['seed']);draws=plan['draws'];by_task={};names={m:i for i,m in enumerate(models)};clinical=None
 for task in tasks:
  data=contract['clinical'] if task=='inhouse' else contract['external'][task];truth=np.asarray(data['truth'],np.int64);groups=np.asarray(data['group'],np.int64);expected=np.asarray(data['source_rows'],np.int64);preds=[]
  for model in models:
   with np.load(paths[(model,task)],allow_pickle=False) as z:
    assert np.array_equal(z['source_rows'],expected) and len(z['prediction'])==len(truth);preds.append(z['prediction'])
  preds=np.stack(preds)
  if task=='inhouse':
   correct=[]
   for pred in preds:
    yy,pp=patient_votes(truth,pred,groups,3);assert len(yy)==10;correct.append(yy==pp)
   clinical=np.asarray(correct);np.savez_compressed(private/'clinical.npz',correct=clinical)
  else:
   correct=preds==truth[None,:];part=cluster_draws(correct,groups,draws,rng);np.savez_compressed(private/(task+'.npz'),correct=correct,groups=groups,**part);by_task[task]=part
 assert sum(len(contract['external'][t]['truth']) for t in plan['partitions']['all_external'])==1731
 external=[];boot_external=[];perm_external=[]
 for partition,selected in plan['partitions'].items():
  total=sum(len(contract['external'][t]['truth']) for t in selected);assert total==plan['original_fixed_records_by_partition'][partition]
  for pair in plan['pairs']:
   a=names[pair['a']];b=names[pair['b']];point=0.;bootstrap=np.zeros(draws);permutation=np.zeros(draws)
   for task in selected:
    d=by_task[task];n=len(contract['external'][task]['truth']);point+=(d['point_accuracy'][a]-d['point_accuracy'][b])*n/total
    bootstrap+=(d['bootstrap_accuracy'][a]-d['bootstrap_accuracy'][b])*n/total
    permutation+=d['signs'].astype(np.int64)@(d['correct_by_group'][a]-d['correct_by_group'][b])/total
   external.append(dict(partition=partition,external_fixed_records=total,**pair,**summarize_difference(point,bootstrap,permutation)));boot_external.append(bootstrap);perm_external.append(permutation)
 adj=holm([x['p_unadjusted'] for x in external])
 for x,p in zip(external,adj):x['p_Holm']=float(p)
 indices=rng.integers(0,10,size=(draws,10));internal=[];boot_internal=[]
 for pair in plan['pairs']:
  a=clinical[names[pair['a']]];b=clinical[names[pair['b']]];delta=a.astype(float)-b.astype(float);boot=delta[indices].mean(1);lo,hi=np.quantile(boot,[.025,.975]);internal.append(dict(**pair,internal_fixed_patients=10,difference=float(delta.mean()),CI95=[float(lo),float(hi)],p_unadjusted=mcnemar(a,b),a_correct=int(a.sum()),b_correct=int(b.sum())));boot_internal.append(boot)
 adj=holm([x['p_unadjusted'] for x in internal])
 for x,p in zip(internal,adj):x['p_Holm']=float(p)
 np.savez_compressed(private/'draws.npz',external_bootstrap=np.stack(boot_external),external_permutation=np.stack(perm_external),internal_bootstrap=np.stack(boot_internal),internal_indices=indices)
 result=dict(status='computed_pending_independent_audit',scope='Actual original fixed predictions only; no OOF',external=external,internal=internal,models=models,draws=draws,seed=plan['seed'],plan_sha256=sha(R/'fixed_statistics_plan001.json'),source_freeze_sha256=sha(O/'source_freeze.json'),private_hashes={p.name:sha(p) for p in private.glob('*.npz')},new_training_updates=0,new_fixed_predictions=0,interpretation=plan['registration'],limitations=plan['limitations'],completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with (O/'results.json').open('x') as h:json.dump(result,h,indent=2)
 print(json.dumps(dict(status=result['status'],external_tests=len(external),internal_tests=len(internal),new_fixed_predictions=0)))


if __name__=='__main__':main()
