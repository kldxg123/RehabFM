from framework_supplement import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
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
   vf=VarianceThreshold(threshold=1e-10).fit(X[ix]);sc=StandardScaler().fit(vf.transform(X[ix]))
   assert np.array_equal(estimator[0].variances_,np.var(X[ix].astype(np.float64),axis=0))
   assert np.array_equal(estimator[0].variances_,vf.variances_) and np.array_equal(estimator[0].get_support(),vf.get_support())
   assert all(np.array_equal(getattr(estimator[1],k),getattr(sc,k)) for k in ['mean_','var_','scale_','n_samples_seen_'])
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

if __name__=='__main__':
 try:
  assert not (O/'audited_results.json').exists() and read(O/'failure_all.json')['status']=='failed_preserved'
  d=read(O/'preprocessing_audit_diagnosis001.json');assert d['float64_numpy_max_abs']==0 and d['reconstructed_variance_exact'] and d['reconstructed_mean_exact'] and d['reconstructed_scale_exact']
  save(O/'audit002_freeze.json',dict(auditor_sha256=sha(Path(__file__)),original_script_sha256=sha(O/'framework_supplement.py'),original_freeze_sha256=sha(O/'freeze.json'),sealed_predictions_sha256=sha(O/'all_predictions_sealed.json'),original_failure_sha256=sha(O/'failure_all.json'),diagnosis_sha256=sha(O/'preprocessing_audit_diagnosis001.json'),change='audit uses training-equivalent float64 variance plus exact preprocessing reconstruction; no tolerance, model, seed, threshold or prediction change',new_tree_fits=0,new_backbone_updates=0))
  audit();archive()
 except Exception as e:
  save(O/'failure_audit002.json',dict(status='failed_preserved',type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
