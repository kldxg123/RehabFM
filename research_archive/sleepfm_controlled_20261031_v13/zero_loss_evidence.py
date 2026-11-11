"""Private one-batch evidence for mathematically inactive losses; no optimizer updates in audit."""
from common import *
import torch
from structural_zero_policy import explainable
def cpu_copy(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_copy(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
 return x
def capture(folder,epoch,step,model,decoder,predictors,heads,proj,opt,schema,weights,batch,masks,pred,terms,loss,rng_cpu,rng_cuda,buffers):
 N=len(batch['y']);minimum=min(float(z.detach().float().std(0).min()) for z in pred['z'].values()) if N>=2 else None
 meta=dict(arm=folder.name.rsplit('_s',1)[0],batch_records=N,unique_labels=int(batch['y'].unique().numel()),remaining_modalities=len(pred['z']),valid_masked_windows=sum(int((masks[m]&batch[m]['wmask']).sum()) for m in masks),terms={k:None if v is None else float(v.detach()) for k,v in terms.items()},total=float(loss.detach()),variance_std_min=minimum)
 assert explainable(**meta),'unexplained true zero-gradient remains fatal'
 components=dict(model=model.state_dict(),decoder=decoder.state_dict(),predictors=predictors.state_dict(),aux_heads=heads.state_dict(),projection=proj.state_dict(),optimizer=opt.state_dict(),schema=schema)
 state=cpu_copy(components)
 for k,v in buffers.items():state['model'][k]=cpu_copy(v)
 state.update(batch=cpu_copy(batch),window_masks=cpu_copy(masks),weights=cpu_copy(weights),rng_cpu=cpu_copy(rng_cpu),rng_cuda=cpu_copy(rng_cuda),metadata=meta)
 dst=folder/'zero_loss_private';dst.mkdir(exist_ok=True);p=dst/f'e{epoch:03d}_b{step:03d}.pt';assert not p.exists();torch.save(state,p)
 ledger=folder/'zero_loss_evidence.json';rows=read(ledger) if ledger.exists() else []
 rows.append(dict(epoch=epoch,step=step,path=str(p),sha256=sha(p),metadata=meta));q=ledger.with_suffix('.tmp');q.write_text(json.dumps(rows,indent=2));q.replace(ledger)
def audit_case(case,receipt):
 from audit_training import components
 from stable_clip import clip_grad_norm_equivalent
 from train_core import digest
 import train_study as T
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3';verify();torch.set_num_threads(3)
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 s=torch.load(case,map_location='cpu',weights_only=False);mods,opt=components(s['metadata']['arm'],s)
 for m in mods.values():m.cuda().train()
 batch=device(s['batch'],'cuda');masks=device(s['window_masks'],'cuda');weights=device(s['weights'],'cuda')
 torch.set_rng_state(s['rng_cpu']);torch.cuda.set_rng_state(s['rng_cuda'])
 model=mods['model'];params=[p for m in mods.values() for p in m.parameters()];before={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
 with torch.autocast('cuda',dtype=torch.bfloat16):
  pred=model(batch,modality_drop_p=.25,channel_drop_p=.1);N=len(batch['y']);model._stat_cache={m:batch[m]['wave'] for m in pred['z']}
  terms=dict(aux=T.aux_loss(pred,batch,mods['aux_heads'],weights))
  if s['metadata']['arm']!='ce_only':terms.update(masked=T.masked_stats_loss(model,mods['decoder'],batch,masks),loo=T.loo_cl_loss(pred['z'],pred['owners'],.07,N),xpred=T.xpred_loss(model,pred['z'],pred['owners'],mods['predictors'],N),var=T.variance_loss(pred['z']),supcon=T.supcon_loss(pred['z'],pred['owners'],batch['y'],mods['projection'],N))
  loss=sum(z.sum()*0 for z in pred['z'].values())
  for k,w in [('masked',1),('loo',1),('xpred',.5),('var',1),('supcon',.5),('aux',1 if s['metadata']['arm']=='ce_only' else .3)]:
   if terms.get(k) is not None:loss=loss+w*terms[k]
 meta=dict(arm=s['metadata']['arm'],batch_records=N,unique_labels=int(batch['y'].unique().numel()),remaining_modalities=len(pred['z']),valid_masked_windows=sum(int((masks[m]&batch[m]['wmask']).sum()) for m in masks),terms={k:None if v is None else float(v.detach()) for k,v in terms.items()},total=float(loss.detach()),variance_std_min=min(float(z.detach().float().std(0).min()) for z in pred['z'].values()) if N>=2 else None)
 assert explainable(**meta);loss.backward();gn=clip_grad_norm_equivalent(params,1.);assert float(gn)==0
 assert any(p.grad is not None for p in params) and all(torch.isfinite(p.grad).all() and not torch.count_nonzero(p.grad) for p in params if p.grad is not None)
 assert all(torch.equal(p.detach().cpu(),before[k]) for k,p in model.named_parameters())
 save(receipt,dict(status='passed_exact_inactive_loss_proof',case_sha256=sha(case),metadata=meta,optimizer_updates=0,model_parameters_unchanged=True,BN_buffers_follow_training_mode_only=True))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();audit_case(Path(a.case),Path(a.receipt))
