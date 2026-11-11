"""Locked, training-pool grouped OOF baselines. Never load fixed data or neural F."""
from pathlib import Path
import argparse, hashlib, json, os, time, traceback, fcntl
import numpy as np
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent
V=Path('/path/to/rehabfm-workspace/rehabfm_unified_context_20261019_v1/clean_subset_all23_20261021_v2')
def rd(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def wr(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def model(arm):
 if arm=='B0':return DummyClassifier(strategy='most_frequent')
 alg={'B1':lambda:RidgeClassifier(alpha=1.0,class_weight=None),
      'B2':lambda:SVC(C=1.,kernel='rbf',gamma='scale',class_weight=None,probability=False,cache_size=1024),
      'B3':lambda:RandomForestClassifier(n_estimators=300,max_features='sqrt',min_samples_leaf=1,random_state=0,n_jobs=4),
      'B4':lambda:ExtraTreesClassifier(n_estimators=300,max_features='sqrt',min_samples_leaf=1,random_state=0,n_jobs=4)}[arm]()
 return make_pipeline(VarianceThreshold(1e-10),StandardScaler(),alg)
def metrics(y,p,K):
 cm=confusion_matrix(y,p,labels=np.arange(K));n=len(y)
 return dict(correct=int((y==p).sum()),total=n,accuracy=float((y==p).mean()),balanced_accuracy=float(balanced_accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=np.arange(K),average='macro',zero_division=0)),confusion=cm.tolist(),classes=K)
def vote(y,p,g):
 yy=[];pp=[]
 for u in np.unique(g):
  a=y[g==u];assert len(np.unique(a))==1
  yy.append(a[0]);pp.append(np.bincount(p[g==u]).argmax())
 return np.array(yy),np.array(pp)
def preflight():
 x=np.random.default_rng(0).normal(size=(36,8));y=np.repeat(np.arange(3),12)
 for a in ['B0','B1','B2','B3','B4']:
  m=model(a);m.fit(x,y);p=m.predict(x[:3]);assert p.shape==(3,)
 yy,pp=vote(np.array([0,0,1,1]),np.array([0,1,1,1]),np.array([10,10,20,20]))
 assert yy.tolist()==[0,1] and pp.tolist()==[0,1]
 assert metrics(np.array([0,2]),np.array([0,2]),3)['confusion']==[[1,0,0],[0,0,0],[0,0,1]]
 wr(R/'statistical_generated_admission.json',dict(status='passed',models=5,vote_tie_rule='smallest class',original_class_axis=True,project_data_loaded=False,source_sha256=sha(__file__)))
def freeze():
 assert not (R/'statistical_source_freeze.json').exists()
 review=rd(V/'clean_subset045_review.json');assert review['review_passed']
 paths=[Path(__file__),R/'statistical_policy.json',R/'statistical_generated_admission.json',V/'clean_subset045_review.json',V/'training_pool_manifest.json',V/'audit_private/fold_rows.npz']
 assert sha(V/'training_pool_manifest.json')==review['training_manifest_sha256']
 assert sha(V/'audit_private/fold_rows.npz')==review['fold_rows_sha256']
 m=rd(V/'training_pool_manifest.json')
 for t in m['tasks']:
  assert not t['learned_body_features_consumed'];p=V/'training_only'/t['filename'];assert sha(p)==t['sha256'];paths.append(p)
 wr(R/'statistical_source_freeze.json',dict(status='frozen_before_project_model_fitting',files={str(p):sha(p) for p in paths},planned_cases=345,created_unix=time.time()))
def aggregate(manifest):
 results=[]
 for arm in ['B0','B1','B2','B3','B4']:
  rows=[]
  for t in manifest['tasks']:
   paths=[R/'predictions_private'/f'{arm}_{t["task"]}_f{f}.npz' for f in range(3)]
   if not all(p.exists() for p in paths):continue
   packs=[np.load(p,allow_pickle=False) for p in paths]
   i=np.concatenate([z['rows'] for z in packs]);y=np.concatenate([z['truth'] for z in packs]);p=np.concatenate([z['prediction'] for z in packs]);g=np.concatenate([z['group'] for z in packs])
   assert sorted(i.tolist())==list(range(t['rows']))
   if t['task']=='inhouse':y,p=vote(y,p,g);assert len(y)==19
   rows.append(dict(task=t['task'],weight=t['fixed_test_rows'],**metrics(y,p,t['classes'])))
   for z in packs:z.close()
  ext=[x for x in rows if x['task']!='inhouse']
  item=dict(arm=arm,completed_tasks=len(rows),tasks=rows)
  if len(rows)==23:
   weight=sum(t['weight'] for t in ext);assert weight==1708
   item.update(standardized_OOF_accuracy=sum(t['accuracy']*t['weight'] for t in ext)/weight,mean_task_BA=float(np.mean([t['balanced_accuracy'] for t in ext])),mean_task_macro_f1=float(np.mean([t['macro_f1'] for t in ext])),internal_patient=next(t for t in rows if t['task']=='inhouse'))
  results.append(item)
 wr(R/'statistical_summary.json',dict(status='completed' if all(x['completed_tasks']==23 for x in results) else 'incomplete',evaluation='training_pool_OOF_not_fixed_test',input='common_nonlearned_S_spatial_not_each_historical_native_S_recipe',weight_denominator=1708,seed=0,arms=results,source_freeze_sha256=sha(R/'statistical_source_freeze.json'),missing_neural_and_published_model_experiments=True))
def run():
 lock=(R/'statistical.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'statistical_completed_receipt.json').exists(),'Do not rerun complete work'
 fr=rd(R/'statistical_source_freeze.json')
 for p,h in fr['files'].items():assert sha(p)==h,p
 assert rd(R/'statistical_generated_admission.json')['source_sha256']==sha(__file__)
 for d in ['models_private','predictions_private','case_receipts']:(R/d).mkdir(exist_ok=True)
 m=rd(V/'training_pool_manifest.json');folds=np.load(V/'audit_private/fold_rows.npz',allow_pickle=False)
 progress=dict(status='running',pid=os.getpid(),started_unix=time.time(),completed_cases=0,expected_cases=345,failures=[],fixed_test_loaded=False,new_neural_gradient_updates=0)
 wr(R/'statistical_progress.json',progress)
 with threadpool_limits(limits=4):
  for t in m['tasks']:
   name=t['task']
   with np.load(V/'training_only'/t['filename'],allow_pickle=False) as z:x,y,g=z['x'],z['y'],z['group']
   assert np.isfinite(x).all() and set(np.unique(y))<=set(range(t['classes']))
   if name=='inhouse':assert len(np.unique(g))==19 and not (set(g.tolist()) & {9000+i for i in [5,7,13,16,17,21,22,25,27,29]})
   for fold in range(3):
    tr=folds[f'{name}_fold{fold}_fit'];va=folds[f'{name}_fold{fold}_query'];assert not set(g[tr])&set(g[va]);assert len(tr)+len(va)==len(x)
    for arm in ['B0','B1','B2','B3','B4']:
     key=f'{arm}_{name}_f{fold}';p=R/'predictions_private'/f'{key}.npz';mp=R/'models_private'/f'{key}.joblib';rp=R/'case_receipts'/f'{key}.json'
     progress['active_case']=key;wr(R/'statistical_progress.json',progress)
     if rp.exists():
      rr=rd(rp);assert rr['prediction_sha256']==sha(p) and rr['model_sha256']==sha(mp)
     else:
      assert not p.exists() and not mp.exists(),'Partial output requires audit, not overwrite'
      start=time.time();clf=model(arm);clf.fit(x[tr],y[tr]);pred=clf.predict(x[va]).astype(np.int64)
      assert pred.shape==y[va].shape
      joblib.dump(clf,mp,compress=3)
      assert np.array_equal(joblib.load(mp).predict(x[va]),pred)
      np.savez_compressed(p,rows=va,truth=y[va],prediction=pred,group=g[va])
      rr=dict(arm=arm,task=name,fold=fold,fit_rows=len(tr),oof_rows=len(va),features=x.shape[1],classes=t['classes'],elapsed_seconds=time.time()-start,model_sha256=sha(mp),prediction_sha256=sha(p),input_sha256=t['sha256'],source_freeze_sha256=sha(R/'statistical_source_freeze.json'),exact_model_reload_predictions=True,query_data_used_for_fitting=False)
      wr(rp,rr)
     progress['completed_cases']+=1;wr(R/'statistical_progress.json',progress)
   aggregate(m)
 progress.update(status='completed',active_case=None,finished_unix=time.time());wr(R/'statistical_progress.json',progress)
 aggregate(m);assert progress['completed_cases']==345
 wr(R/'statistical_completed_receipt.json',dict(status='passed',cases=345,task_arms=115,models_restored=345,patient_folds=69,fixed_test_loaded=False,new_neural_gradient_updates=0,summary_sha256=sha(R/'statistical_summary.json'),source_freeze_sha256=sha(R/'statistical_source_freeze.json')))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['admit','freeze','run']);a=ap.parse_args()
 try:{'admit':preflight,'freeze':freeze,'run':run}[a.action]()
 except Exception as e:
  wr(R/('statistical_'+a.action+'_failure.json'),dict(status='failed',type=type(e).__name__,message=str(e),time=time.time()))
  raise
