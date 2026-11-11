"""Pure saved-estimator prediction on original native descriptors, no truth."""
from pathlib import Path
import sys,argparse,json,os
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from fixed_interfaces import read,sha,prediction_contract


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);key=parser.parse_args().job
 assert not os.environ.get('CUDA_VISIBLE_DEVICES');fr=read(P/'fixed_evaluation_freeze.json');reg=read(P/'models_for_fixed.json');s=reg['jobs'][key]
 for p in [Path(__file__),P/'fixed_interfaces.py',P/'models_for_fixed.json',Path(s['query_path']),Path(s['estimator_path'])]:assert fr['files'][str(p)]==sha(p)
 folder=P/'predictions_private'/key;assert not folder.exists();folder.mkdir(parents=True)
 with np.load(s['query_path'],allow_pickle=False) as z:assert set(z.files)=={'x','source_rows'};x=z['x'];indices=z['source_rows']
 assert x.shape==(s['fixed_query_records'],s['features']) and np.isfinite(x).all()
 est=joblib.load(s['estimator_path'])
 with threadpool_limits(limits=4):pred=np.asarray(est.predict(x),np.int64)
 prediction_contract(pred,indices,indices,s['K']);np.savez_compressed(folder/'prediction.npz',prediction=pred,source_rows=indices)
 seal=dict(status='prediction_sealed_no_scoring',job=key,method=s['method'],task=s['task'],K=s['K'],records=len(pred),prediction_sha256=sha(folder/'prediction.npz'),registry_sha256=sha(P/'models_for_fixed.json'),evaluation_freeze_sha256=sha(P/'fixed_evaluation_freeze.json'),query_input_sha256=sha(s['query_path']),estimator_sha256=sha(s['estimator_path']),true_query_labels_read_by_predictor=False,patient_ids_used_by_predictor=False,new_model_updates=0)
 with (folder/'seal.json').open('x') as h:json.dump(seal,h,indent=2)
 print(json.dumps(dict(status=seal['status'],job=key,records=len(pred))))


if __name__=='__main__':main()
