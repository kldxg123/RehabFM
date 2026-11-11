"""No optimizer.step; reconstruct objective availability under frozen epoch RNG and sampler.
For CE-only, label-homogeneous batches are an exact weight/RNG-independent skip.
For other arms, model parameters stay fixed while RNG control flow follows the trainer.
"""
from common import *
import torch,numpy as np,argparse,types
from torch import nn
import train_study as T
from structural_zero_policy import explainable,validate_counts
from stable_clip import clip_grad_norm_equivalent
def reconstruct(arm,seed,epoch,destination):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3';verify()
 folder=runpath(arm,seed);h=read(folder/'history.json');row=h[epoch-1]
 assert row['epoch']==epoch and row['nonzero_gradient_steps']<row['updates']
 torch.set_num_threads(3);torch.manual_seed(seed);np.random.seed(seed)
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 source=(O/'reference/train_study_v7.py').read_text();prefix=source[source.index(' receipt=json.loads'):source.index(' history=[];updates=0')]
 lines=[x[1:] for x in prefix.splitlines() if not x.strip().startswith('out=') and not x.strip().startswith("wr(out/'protocol.json'")]
 ns=dict(vars(T));ns['a']=types.SimpleNamespace(arm=arm,seed=seed,fold=0,epochs=60,admission=False)
 exec(compile('\n'.join(lines),'<frozen no-write diagnostic construction>','exec'),ns)
 if epoch>1:
  state=torch.load(folder/f'checkpoint_e{epoch-1:03d}.pt',map_location='cpu',weights_only=False)
  for key,mod in [('model','model'),('decoder','decoder'),('predictors','predictors'),('aux_heads','heads'),('projection','proj')]:ns[mod].load_state_dict(state[key],strict=True)
 model=ns['model'];mods=ns['modules'];before=[T.digest(m) for m in mods]
 torch.manual_seed(seed+(epoch-1)*100003);torch.cuda.manual_seed_all(seed+(epoch-1)*100003);ns['sampler'].seed=seed+epoch-1
 for m in mods:m.train()
 found=[];count=0
 for step,batch in enumerate(ns['dl']):
  batch=T.device(batch);masks=T.apply_window_mask(batch,.3)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   pred=model(batch,modality_drop_p=.25,channel_drop_p=.1);N=len(batch['y']);model._stat_cache={m:batch[m]['wave'] for m in pred['z']}
   terms=dict(aux=T.aux_loss(pred,batch,ns['heads'],ns['weights']))
   if arm!='ce_only':terms.update(masked=T.masked_stats_loss(model,ns['decoder'],batch,masks),loo=T.loo_cl_loss(pred['z'],pred['owners'],.07,N),xpred=T.xpred_loss(model,pred['z'],pred['owners'],ns['predictors'],N),var=T.variance_loss(pred['z']),supcon=T.supcon_loss(pred['z'],pred['owners'],batch['y'],ns['proj'],N))
   loss=sum(z.sum()*0 for z in pred['z'].values())
   for k,w in [('masked',1),('loo',1),('xpred',.5),('var',1),('supcon',.5),('aux',1 if arm=='ce_only' else .3)]:
    if terms.get(k) is not None:loss=loss+w*terms[k]
  assert torch.isfinite(loss)
  for p in ns['params']:p.grad=None
  loss.backward();gn=clip_grad_norm_equivalent(ns['params'],1.);assert torch.isfinite(gn)
  values={k:None if v is None else float(v.detach()) for k,v in terms.items()}
  meta=dict(arm=arm,batch_records=N,unique_labels=int(batch['y'].unique().numel()),remaining_modalities=len(pred['z']),valid_masked_windows=sum(int((masks[m]&batch[m]['wmask']).sum()) for m in masks),terms=values,total=float(loss.detach()))
  if float(gn)==0:
   assert any(p.grad is not None for p in ns['params']),'disconnected objective'
   assert explainable(**meta),'unexplained zero in diagnostic; original audit not accepted'
   found.append(dict(step=step+1,gradient_norm=0.,**meta))
  else:assert not explainable(**meta),'structural zero unexpectedly has gradient'
  count+=1
 assert count==row['updates'];validate_counts(row['updates'],row['nonzero_gradient_steps'],len(found))
 assert before==[T.digest(m) for m in mods]
 save(destination,dict(status='passed_structural_availability_proof',arm=arm,seed=seed,epoch=epoch,history_sha256=sha(folder/'history.json'),checkpoint_sha256=sha(folder/f'checkpoint_e{epoch:03d}.pt'),diagnostic_batches=count,diagnostic_optimizer_updates=0,all_components_byte_unchanged=True,parameter_held_fixed_not_identical_training_trajectory=True,structural_zero_steps=found,source_freeze_sha256=sha(O/'freeze.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--destination',required=True);a=p.parse_args();reconstruct(a.arm,a.seed,a.epoch,Path(a.destination))
