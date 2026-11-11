"""Training-only S/F/H and clinical temporal readouts for new full-fit cores.

No fixed query arrays, original test labels, OOF scores or model selection.
The S estimator is the byte-exact already-fitted B4 baseline on the same legal
training descriptors; F/H and the clinical sequence head are fitted here.
"""
from pathlib import Path
import sys,os,json,argparse,hashlib
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R/'vendor'),str(R.parent)]
from fixed_interfaces import sha,read
from train_core import digest,Items,device
from rehabfm.data import collate_fn
from torch.utils.data import DataLoader
from statistical_baselines import model as classical


def wr(path,value):
 with Path(path).open('x') as h:json.dump(value,h,indent=2)


def main(a):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
 torch.set_num_threads(3);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 freeze=read(R/'full_training_source_freeze.json')
 for path,h in freeze['files'].items():assert sha(path)==h,path
 assert read(R/'statistical_fullfits_v1/independent_training_audit.json')['status']=='passed'
 if a.factory=='fusion':
  assert a.arm in ['concat','attention'] and a.task is None
  sys.path.insert(0,str(R/'fusion_v2'));from fusion_models import make_core
  run=R/'fusion_v2/runs'/f'{a.arm}_f0'
 elif a.factory=='task_full':
  assert a.arm=='full' and a.task
  from core_models import make_core
  run=R/'task_full_v1/runs'/f'{a.task}_f0'
 else:
  assert a.factory=='core' and a.arm in ['full','no_temporal','no_ce','ce_only'] and a.task is None
  from core_models import make_core
  run=R/'runs'/f'{a.arm}_f0'
 receipt=read(run/'training_receipt.json');cp=run/'checkpoint_e060.pt'
 assert receipt['epochs_saved']==60 and receipt['exact_model_restore'] and sha(cp)==receipt['checkpoint_sha256']
 state=torch.load(cp,map_location='cpu',weights_only=False);assert state['epoch']==60
 model=make_core(a.arm);model.load_state_dict(state['model'],strict=True);model=model.cuda().eval()
 initial=digest(model);assert initial==receipt['model_sha256']
 model.unseen_conditioning=dict(mode='zero',known_ids=sorted(map(int,state['schema'])),modality_rows={})
 key=f'{a.factory}_{a.arm}'+('_'+a.task if a.task else '')
 out=R/'readouts_private'/key;assert not out.exists();out.mkdir(parents=True)
 cohort=read(R/'cohort_receipt.json');entries=[e for e in cohort['tasks'] if not a.task or e['task']==a.task]
 assert len(entries)==(1 if a.task else 23)
 audits=[];additional_updates=0
 with threadpool_limits(limits=3):
  for e in entries:
   name=e['task'];source=R/'cohort_private'/f'{name}.pt';assert sha(source)==e['sha256']
   data=torch.load(source,map_location='cpu',weights_only=False);rows=data['items'];K=data['target_classes']
   y=np.array([int(r['y'])//2 if name=='inhouse' else int(r['y']) for r in rows]);g=np.array([int(r['subject']) for r in rows])
   features=[];tokens=[]
   with torch.no_grad():
    for raw in DataLoader(Items(rows),batch_size=16,shuffle=False,collate_fn=collate_fn,num_workers=0):
     b=device(raw);p=model(b);B=len(b['y'])
     if a.factory=='fusion':z=p['fusion_z'];t=p['fusion_T']
     else:
      from core_models import fuse
      z=fuse(p,B);first=next(iter(p['tokens'].values()));t=first.new_zeros(B,12,64);count=first.new_zeros(B)
      for mod,v in p['tokens'].items():t.index_add_(0,p['owners'][mod],v);count.index_add_(0,p['owners'][mod],count.new_ones(len(v)))
      assert (count>0).all();t=t/count[:,None,None]
     f=torch.cat([z,t.mean(1),t.std(1,unbiased=False),t.max(1).values],1)
     assert f.shape==(B,256) and torch.isfinite(f).all();features.append(f.cpu().numpy());tokens.append(t.cpu().numpy())
   assert digest(model)==initial
   F=np.concatenate(features);T=np.concatenate(tokens)
   feature_path=out/f'{name}_training_features.npz';np.savez_compressed(feature_path,F=F,T=T,y=y,group=g)
   stroot=R/'statistical_fullfits_v1';schema=stroot/'training_private'/f'{name}_schema.json'
   with np.load(stroot/'training_private'/f'{name}_training.npz',allow_pickle=False) as q:
    S=q['x'];assert np.array_equal(q['y'],y) and np.array_equal(q['group'],g) and len(S)==len(F)
   views={}
   s_model=stroot/'models_private'/f'B4_{name}.joblib'
   views['S']=dict(readout_path=str(s_model),sha256=sha(s_model),inherited_exact_statistical_B4=True,new_fits=0)
   for view,X in [('F',F),('H',np.concatenate([S,F],1))]:
    est=classical('B4');est.fit(X,y);target=out/f'{name}_{view}.joblib';joblib.dump(est,target,compress=3)
    prefix=est.predict(X[:3]).astype(np.int64);assert np.array_equal(joblib.load(target).predict(X[:3]),prefix)
    views[view]=dict(readout_path=str(target),sha256=sha(target),inherited_exact_statistical_B4=False,new_fits=1,training_prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest())
   if name=='inhouse':
    from evaluate_core import SequenceHead
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    X=torch.tensor(T,dtype=torch.float32,device='cuda');mu=X.mean((0,1),keepdim=True);sd=X.std((0,1),unbiased=False,keepdim=True).clamp_min(.05);X=((X-mu)/sd).clamp(-10,10);Y=torch.tensor(y,device='cuda')
    head=SequenceHead().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    assert sum(p.numel() for p in head.parameters())==166275
    initial_head=digest(head);checkpoints=out/'inhouse_sequence_checkpoints';checkpoints.mkdir()
    for epoch in range(1,151):
     head.train();opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(X),Y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step()
     # Saving extra evidence does not change RNG or any optimizer/forward step.
     checkpoint=checkpoints/f'checkpoint_e{epoch:03d}.pt'
     torch.save(dict(model=head.state_dict(),optimizer=opt.state_dict(),mean=mu.cpu(),std=sd.cpu(),epochs=epoch,training_source_sha256=sha(source)),checkpoint)
    additional_updates+=150;head.eval();target=checkpoints/'checkpoint_e150.pt';saved=torch.load(target,map_location='cpu',weights_only=False)
    restored=SequenceHead().cuda();restored.load_state_dict(saved['model'],strict=True);restored.eval();restopt=torch.optim.AdamW(restored.parameters());restopt.load_state_dict(saved['optimizer'])
    assert digest(restored)==digest(head)!=initial_head and restopt.state_dict()['param_groups']==saved['optimizer']['param_groups']
    for key,fields in saved['optimizer']['state'].items():
     for field,v in fields.items():assert torch.equal(v.detach().cpu(),restopt.state_dict()['state'][key][field].detach().cpu()) if torch.is_tensor(v) else v==restopt.state_dict()['state'][key][field]
    with torch.no_grad():assert torch.equal(head(X),restored(X))
    views['T']=dict(readout_path=str(target),sha256=sha(target),epochs=150,updates=150,checkpoints=150,parameters=166275,exact_model_optimizer_restore=True)
    del head,restored,opt,restopt,X,Y;torch.cuda.empty_cache()
   audits.append(dict(task=name,K=K,training_rows=len(rows),training_groups=len(set(g.tolist())),statistics_schema=str(schema),statistics_schema_sha256=sha(schema),training_source_sha256=sha(source),training_features_sha256=sha(feature_path),core_sha256=initial,checkpoint_sha256=sha(cp),views=views,fixed_query_arrays_read=False))
   (out/'progress.json').write_text(json.dumps(dict(status='training_readouts_only',completed_tasks=len(audits),expected_tasks=len(entries),new_fixed_predictions=0)))
   print(json.dumps(dict(task=name,completed_tasks=len(audits),new_fixed_predictions=0)),flush=True)
 assert digest(model)==initial
 result=dict(status='training_readouts_complete_pending_independent_audit',factory=a.factory,arm=a.arm,gradient_task=a.task,tasks=audits,core_checkpoint_path=str(cp),core_checkpoint_sha256=sha(cp),core_sha256=initial,
      new_classical_readout_fits=2*len(audits),inherited_S_baseline_estimators=len(audits),new_core_updates=0,additional_clinical_readout_updates=additional_updates,new_fixed_predictions=0,query_arrays_read=False,source_freeze_sha256=sha(R/'full_training_source_freeze.json'),fitter_sha256=sha(Path(__file__)))
 wr(out/'training_readout_receipt.json',result)
 (out/'progress.json').write_text(json.dumps(dict(status=result['status'],completed_tasks=len(audits),new_fixed_predictions=0)))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--factory',choices=['core','fusion','task_full'],required=True);p.add_argument('--arm',required=True);p.add_argument('--task');main(p.parse_args())
