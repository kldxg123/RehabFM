from common import *
import argparse,torch
from torch import nn
from study_models import make_core
from rehabfm.model import MaskedStatsDecoder
from train_core import digest
def components(arm,state):
 model=make_core(arm);decoder=MaskedStatsDecoder()
 mods=sorted({k.split('.')[0] for k in state['predictors']})
 predictors=nn.ModuleDict({m:nn.Linear(64,32) for m in mods})
 heads=nn.ModuleDict({k:nn.Linear(64,v['n_classes']) for k,v in state['schema'].items()})
 proj=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,128))
 c=dict(model=model,decoder=decoder,predictors=predictors,aux_heads=heads,projection=proj)
 opt=torch.optim.AdamW([dict(params=list(model.parameters())+list(decoder.parameters())+list(predictors.parameters()),lr=.001),dict(params=list(heads.parameters())+list(proj.parameters()),lr=.001)],weight_decay=.05,betas=(.9,.98))
 for k,m in c.items():m.load_state_dict(state[k],strict=True);exact(m.state_dict(),state[k])
 opt.load_state_dict(state['optimizer']);exact(opt.state_dict(),state['optimizer'])
 for group,sg in zip(opt.param_groups,state['optimizer']['param_groups']):
  for p,i in zip(group['params'],sg['params']):
   for k,v in state['optimizer']['state'].get(i,{}).items():
    if torch.is_tensor(v):
     assert torch.isfinite(v).all()
     if k in ['exp_avg','exp_avg_sq']:assert v.shape==p.shape
 return c,opt
def audit_gradient_accounting(arm,seed,folder,h):
 from structural_zero_policy import explainable,validate_counts
 import subprocess
 evidence=[];zero=0
 ledger=read(folder/'zero_loss_evidence.json') if (folder/'zero_loss_evidence.json').exists() else []
 for r in h:
  missing=r['updates']-r['nonzero_gradient_steps'];assert 0<=missing<=r['updates']
  if not missing:continue
  if arm=='sleepfm_arch' and seed==0 and r['epoch']==2:
   proof=O/'preserved_v8_failure/sleepfm_v7_epoch2_counter_diagnosis003.json';d=read(proof)
   assert sha(O/'partial_origin/history.json')==d['history_sha256']
   assert sha(folder/'checkpoint_e001.pt')==d['checkpoint_sha256'] and sha(folder/'checkpoint_e002.pt')==d['target_checkpoint_sha256']
   assert read(folder/'history.json')[:2]==read(O/'partial_origin/history.json')
   assert d['model_parameters_exact_match'] and all(d['saved_epoch2_exact_restoration_after_diagnostic_replay'].values())
   zs=[x for x in d['records'] if x.get('native_norm')=='0.0'];assert len(zs)==missing
   for z in zs:
    assert z['all_gradients_finite_before_clip'] and z['max_abs_finite_gradient']==0 and z['stable_FP64_norm']==0
    assert explainable(arm,z['batch_records'],z['unique_labels'],z['remaining_modalities'],z['valid_masked_windows'],z['loss_terms'],z['loss'],z['variance_std_min'])
   evidence.append(dict(epoch=2,proof=str(proof),sha256=sha(proof),structural_zero_calls=missing,diagnostic_replay_updates=175,formal_extra_updates=0))
  else:
   rows=[x for x in ledger if x['epoch']==r['epoch']];assert len(rows)==missing,'zero-loss private evidence missing'
   from manager import gpu_ready,paused,PY
   for c in rows:
    assert sha(c['path'])==c['sha256'] and explainable(**c['metadata'])
    output=folder/f"zero_case_audit_e{c['epoch']:03d}_b{c['step']:03d}.json"
    if not output.exists():
     while paused() or not gpu_ready():time.sleep(30)
     env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',PYTHONDONTWRITEBYTECODE='1')
     assert subprocess.run([PY,'-B','-u',str(O/'zero_loss_evidence.py'),'--case',c['path'],'--receipt',str(output)],cwd=O,env=env).returncode==0
    q=read(output);assert q['status']=='passed_exact_inactive_loss_proof' and q['case_sha256']==c['sha256'] and q['optimizer_updates']==0
    evidence.append(dict(epoch=r['epoch'],step=c['step'],proof=str(output),sha256=sha(output),structural_zero_calls=1))
  validate_counts(r['updates'],r['nonzero_gradient_steps'],missing);zero+=missing
 return dict(optimizer_calls=sum(r['updates'] for r in h),nonzero_gradient_calls=sum(r['nonzero_gradient_steps'] for r in h),proven_structural_zero_calls=zero,unexplained_zero_calls=0,evidence=evidence,rule='Exact inactive objectives incl satisfied variance hinge; historical replay/full-state equality or frozen one-batch private proof; no numeric tolerance')

def audit(arm,seed,admission=False):
 torch.set_num_threads(2);verify()
 folder=O/'gpu_admission_private'/f'{arm}_s{seed}' if admission else runpath(arm,seed)
 # Completed originals are referenced, never re-audited or overwritten.
 assert admission or not inherit(arm,seed)
 n=1 if admission else 60;h=read(folder/'history.json');assert len(h)==n
 cps=sorted(folder.glob('checkpoint_e*.pt'));assert len(cps)==n
 accounting=audit_gradient_accounting(arm,seed,folder,h)
 assert sum(x['updates'] for x in h)==(2 if admission else 10500)
 checkpoints=[]
 for i,p in enumerate(cps,1):
  s=torch.load(p,map_location='cpu',weights_only=False);assert s['epoch']==i
  assert s['updates']==sum(r['updates'] for r in h[:i]);checkpoints.append(dict(epoch=i,sha256=sha(p)))
  if i==n:c,opt=components(arm,s)
  # Every checkpoint: finite tensors plus counters; final exact model/optimizer restoration.
  for k in ['model','decoder','predictors','aux_heads','projection']:
   assert all(torch.isfinite(v).all() for v in s[k].values() if torch.is_tensor(v))
 receipt=read(folder/'training_receipt.json');assert receipt['checkpoint_sha256']==sha(cps[-1])
 assert digest(c['model'])==receipt['model_sha256']!=receipt['initial_model_sha256']
 save(folder/'independent_audit.json',dict(status='passed',arm=arm,seed=seed,admission=admission,epochs=n,updates=s['updates'],gradient_accounting=accounting,checkpoints=checkpoints,exact_model_optimizer_restoration=True,training_receipt_sha256=sha(folder/'training_receipt.json'),fixed_test_loaded=False))
 print(json.dumps(dict(status='passed',arm=arm,seed=seed,epochs=n)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--admission',action='store_true');a=p.parse_args();audit(a.arm,a.seed,a.admission)
