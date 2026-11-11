"""Fit training-only readouts, seal fixed predictions without opening truth."""
from common import *
import argparse,torch,numpy as np,joblib
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from study_models import make_core
from train_core import Items,digest
from rehabfm.data import collate_fn
from core_models import fuse
from fixed_interfaces import statistics,canonical_describe,prediction_contract
from statistical_baselines import model as classical
def extract(model,rows):
 parts=[]
 with torch.no_grad():
  for raw in DataLoader(Items(rows),batch_size=16,shuffle=False,collate_fn=collate_fn):
   b=device(raw,'cuda');p=model(b);B=len(b['y']);z=fuse(p,B)
   first=next(iter(p['tokens'].values()));t=first.new_zeros(B,12,64);count=first.new_zeros(B)
   for m,v in p['tokens'].items():t.index_add_(0,p['owners'][m],v);count.index_add_(0,p['owners'][m],count.new_ones(len(v)))
   assert (count>0).all();t=t/count[:,None,None];f=torch.cat([z,t.mean(1),t.std(1,unbiased=False),t.max(1).values],1)
   assert f.shape==(B,256) and torch.isfinite(f).all();parts.append(f.cpu().numpy())
 return np.concatenate(parts)
def main(arm,seed):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3';verify();assert not inherit(arm,seed)
 torch.set_num_threads(3);torch.manual_seed(0);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 folder=runpath(arm,seed);assert read(folder/'independent_audit.json')['status']=='passed'
 out=O/'readouts_private'/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=False)
 cp=folder/'checkpoint_e060.pt';st=torch.load(cp,map_location='cpu',weights_only=False);m=make_core(arm);m.load_state_dict(st['model']);m=m.cuda().eval()
 m.unseen_conditioning=dict(mode='zero',known_ids=sorted(map(int,st['schema'])),modality_rows={});before=digest(m)
 reg=read(R/'fixed_eval_panels/neural_common_v1/models_for_fixed.json');roles=read(R/'full_training_roles002.json')['tasks'];cases=[]
 save(out/'model_binding.json',dict(checkpoint_sha256=sha(cp),model_sha256=before,source_freeze_sha256=sha(O/'freeze.json'),fixed_predictions_started=False))
 with threadpool_limits(limits=3):
  for task in roles:
   spec=reg['jobs']['Core_full_H__'+task];tr=torch.load(R/'cohort_private'/f'{task}.pt',map_location='cpu',weights_only=False)['items'];Ftr=extract(m,tr)
   with np.load(R/'statistical_fullfits_v1/training_private'/f'{task}_training.npz') as a:S=a['x'];y=a['y'];g=a['group']
   assert np.array_equal(y,[int(r['y'])//2 if task=='inhouse' else int(r['y']) for r in tr]);assert np.array_equal(g,[int(r['subject']) for r in tr])
   np.savez_compressed(out/f'{task}_training.npz',S=S,F=Ftr,y=y,group=g)
   estimators={}
   for view,X in [('F',Ftr),('H',np.concatenate([S,Ftr],1))]:
    est=classical('B4');est.set_params(extratreesclassifier__n_jobs=3);est.fit(X,y);mp=out/f'{task}_{view}.joblib';joblib.dump(est,mp,compress=3);estimators[view]=est
   q=torch.load(spec['query_path'],map_location='cpu',weights_only=False);rows=q['items'];assert all(r['y']==0 and r['subject']==-1 for r in rows)
   Fq=extract(m,rows);Sq=statistics(rows,read(spec['statistics_schema']),canonical_describe());ix=np.asarray(q['original_row_indices'])
   np.savez_compressed(out/f'{task}_query.npz',S=Sq,F=Fq,source_rows=ix)
   for view,X in [('F',Fq),('H',np.concatenate([Sq,Fq],1))]:
    p=estimators[view].predict(X).astype(np.int64);prediction_contract(p,ix,ix,spec['K']);pp=out/f'{task}_{view}_prediction.npz';np.savez_compressed(pp,prediction=p,source_rows=ix)
    cases.append(dict(arm=arm,seed=seed,task=task,view=view,K=spec['K'],prediction_path=str(pp),prediction_sha256=sha(pp),readout_path=str(out/f'{task}_{view}.joblib'),readout_sha256=sha(out/f'{task}_{view}.joblib'),training_features_sha256=sha(out/f'{task}_training.npz'),query_features_sha256=sha(out/f'{task}_query.npz'),query_input_sha256=sha(spec['query_path']),query_records=len(p)))
   assert digest(m)==before
   progress(out/'progress.json',status='fitting_and_sealing_without_truth',tasks=len(cases)//2,expected=23)
 save(out/'all_sealed.json',dict(status='sealed_no_truth',cases=cases,model_sha256=before,checkpoint_sha256=sha(cp),new_backbone_updates=0,freeze_sha256=sha(O/'freeze.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.arm,a.seed)
