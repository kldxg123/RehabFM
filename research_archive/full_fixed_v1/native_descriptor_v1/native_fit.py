"""Full legal native-descriptor classical fits. Query arrays never opened."""
from pathlib import Path
import sys,json,os,hashlib,time
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P.parent.parent)]
from fixed_interfaces import read,sha
from statistical_baselines import model


def write(path,value):
 with Path(path).open('x') as h:json.dump(value,h,indent=2)


def main():
 assert not os.environ.get('CUDA_VISIBLE_DEVICES') and not (P/'training_source_freeze.json').exists() and not (P/'training_fits_completed.json').exists()
 manifest=read(P/'data_manifest.json');assert manifest['tasks']==23 and manifest['external_original_fixed_records']==1731
 paths=[Path(__file__),P/'native_audit.py',P/'fixed_interfaces.py',P/'data_manifest.json',P/'policy.json',P.parent.parent/'statistical_baselines.py']+[Path(x['training_path']) for x in manifest['entries']]
 assert all('query_private' not in p.parts for p in paths)
 write(P/'training_source_freeze.json',dict(status='frozen_before_native_training_only_fits',files={str(p):sha(p) for p in paths},planned_fits=115,query_arrays_in_training_freeze=False,versions=dict(numpy=np.__version__,sklearn=__import__('sklearn').__version__,joblib=joblib.__version__)))
 (P/'models_private').mkdir();cases=[]
 with threadpool_limits(limits=4):
  for e in manifest['entries']:
   assert sha(e['training_path'])==e['training_sha256']
   with np.load(e['training_path'],allow_pickle=False) as z:assert set(z.files)=={'x','y','group'};x=z['x'];y=z['y'];groups=z['group']
   assert len(x)==e['training_rows'] and x.shape[1]==e['features'] and np.isfinite(x).all() and ((0<=y)&(y<e['K'])).all()
   prefix=np.unique(np.linspace(0,len(y)-1,min(12,len(y)),dtype=np.int64))
   for arm in ['B0','B1','B2','B3','B4']:
    key=arm+'_'+e['task'];progress=dict(status='fitting_legal_training_only',pid=os.getpid(),active=key,completed_fits=len(cases),expected_fits=115,new_fixed_predictions=0)
    (P/'fit_progress.json').write_text(json.dumps(progress,indent=2));start=time.time();est=model(arm);est.fit(x,y);pred=np.asarray(est.predict(x[prefix]),np.int64);path=P/'models_private'/(key+'.joblib');assert not path.exists();joblib.dump(est,path,compress=3)
    assert np.array_equal(joblib.load(path).predict(x[prefix]),pred)
    cases.append(dict(key=key,method=arm,task=e['task'],K=e['K'],training_rows=len(y),training_groups=len(np.unique(groups)),features=x.shape[1],model_path=str(path),model_sha256=sha(path),training_sha256=e['training_sha256'],probe_rows=prefix.tolist(),training_probe_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),elapsed_seconds=time.time()-start,query_inputs_read=False))
    print(json.dumps(dict(completed_fits=len(cases),active=key,new_fixed_predictions=0)),flush=True)
 assert len(cases)==115
 write(P/'training_fits_completed.json',dict(status='fit_complete_pending_independent_audit',fits=115,cases=cases,new_neural_updates=0,new_fixed_predictions=0,training_source_freeze_sha256=sha(P/'training_source_freeze.json')))
 (P/'fit_progress.json').write_text(json.dumps(dict(status='fit_complete_pending_independent_audit',pid=os.getpid(),active=None,completed_fits=115,new_fixed_predictions=0),indent=2))


if __name__=='__main__':main()
