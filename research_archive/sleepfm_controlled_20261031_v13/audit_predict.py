"""Independent classifier/preprocessing/prediction restoration, no truth."""
from common import *
import argparse,numpy as np,joblib
from threadpoolctl import threadpool_limits
def main(arm,seed):
 verify();out=O/'readouts_private'/f'{arm}_s{seed}';sealed=read(out/'all_sealed.json');assert len(sealed['cases'])==46
 with threadpool_limits(limits=2):
  for c in sealed['cases']:
   t,v=c['task'],c['view'];assert sha(c['prediction_path'])==c['prediction_sha256'] and sha(c['readout_path'])==c['readout_sha256']
   assert sha(out/f'{t}_training.npz')==c['training_features_sha256'] and sha(out/f'{t}_query.npz')==c['query_features_sha256']
   with np.load(out/f'{t}_training.npz') as a:X=a['F'] if v=='F' else np.concatenate([a['S'],a['F']],1);y=a['y'];g=a['group']
   with np.load(R/'statistical_fullfits_v1/training_private'/f'{t}_training.npz') as a:assert np.array_equal(y,a['y']) and np.array_equal(g,a['group'])
   est=joblib.load(c['readout_path']);vt,sc,forest=list(est.named_steps.values())
   expected=np.var(X.astype(np.float64),axis=0)>1e-10;assert np.array_equal(vt.get_support(),expected)
   assert np.allclose(sc.mean_,X[:,expected].astype(np.float64).mean(0),atol=1e-7,rtol=1e-6)
   assert forest.n_estimators==300 and forest.random_state==0
   with np.load(out/f'{t}_query.npz') as a:Xq=a['F'] if v=='F' else np.concatenate([a['S'],a['F']],1);ix=a['source_rows']
   with np.load(c['prediction_path']) as a:assert np.array_equal(est.predict(Xq),a['prediction']) and np.array_equal(ix,a['source_rows'])
 save(out/'independent_prediction_audit.json',dict(status='passed',cases=46,all_readouts_restored=True,training_only_preprocessing_verified=True,all_predictions_recomputed=True,truth_loaded=False,sealed_sha256=sha(out/'all_sealed.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.arm,a.seed)
