"""Fold-local from-scratch historical training; no OOF scoring/model selection."""
import sys,os,json,time,math,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,ConcatDataset,DataLoader
from common import *
from study_models import make_core
from stable_clip import clip_grad_norm_equivalent,events as numeric_events
from zero_loss_evidence import capture
from core_models import aux_loss
from rehabfm.data import collate_fn
from rehabfm.model import MaskedStatsDecoder
from rehabfm.train_pretrain import RoundRobinBatchSampler,apply_window_mask,masked_stats_loss,loo_cl_loss,xpred_loss,variance_loss,supcon_loss

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def wr(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def digest(module):
 h=hashlib.sha256()
 for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class Items(Dataset):
 def __init__(self,items):self.items=items
 def __len__(self):return len(self.items)
 def __getitem__(self,i):
  r=dict(self.items[i])
  for k in ['tokens','stats','waves']:r[k]={m:x.numpy() if torch.is_tensor(x) else x for m,x in r[k].items()}
  return r
def device(batch):return {k:v.cuda() if torch.is_tensor(v) else {a:b.cuda() if torch.is_tensor(b) else b for a,b in v.items()} for k,v in batch.items()}
def run(a):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3','GPU3 only'
 torch.set_num_threads(3);torch.manual_seed(a.seed);np.random.seed(a.seed)
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 if not a.admission:
  fr=verify()
  for p,h in fr['files'].items():assert sha(p)==h,p
  assert a.epochs==60 and not a.max_batches
 receipt=json.loads((R/'cohort_receipt.json').read_text());folds=np.load(R/'cohort_private/folds.npz',allow_pickle=False)
 out=O/('gpu_admission_private' if a.admission else 'runs')/f'{a.arm}_s{a.seed}'
 resuming=not a.admission and a.arm=='sleepfm_arch' and a.seed==0 and (out/'checkpoint_e002.pt').exists()
 if not resuming:out.mkdir(parents=True,exist_ok=False)
 parts=[];tasks=[];audit=[]
 for e in receipt['tasks']:
  if not e['core_training']:continue
  t=torch.load(R/'cohort_private'/f'{e["task"]}.pt',map_location='cpu',weights_only=False)
  tr=folds[f'{e["task"]}_f{a.fold}_train'];va=folds[f'{e["task"]}_f{a.fold}_query']
  allrows=t.pop('items');rows=[allrows[i] for i in tr]
  assert not {int(r['subject']) for r in rows}&{int(allrows[i]['subject']) for i in va}
  t['fit']=rows;tasks.append(t);parts.extend([Items(rows)]*t['repeat'])
  audit.append(dict(task=t['name'],train_rows=len(tr),oof_rows=len(va),train_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),query_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest()))
 concat=ConcatDataset(parts);sampler=RoundRobinBatchSampler(concat.cumulative_sizes,32,0)
 dl=DataLoader(concat,batch_sampler=sampler,collate_fn=collate_fn,num_workers=0)
 model=make_core(a.arm).cuda();decoder=MaskedStatsDecoder().cuda()
 mods=sorted({m for t in tasks for r in t['fit'] for m in r['tokens']})
 predictors=nn.ModuleDict({m:nn.Linear(64,32) for m in mods}).cuda()
 initial=digest(model);torch.manual_seed(100+a.seed)
 heads=nn.ModuleDict();weights={};schema={}
 for t in tasks:
  key=str(t['dataset_idx']);counts=torch.bincount(torch.tensor([int(r['y']) for r in t['fit']]),minlength=t['n_classes']).float()
  heads[key]=nn.Linear(64,t['n_classes']);w=counts.clamp_min(1).rsqrt();w[counts==0]=0;weights[key]=w.cuda();schema[key]=dict(n_classes=t['n_classes'],counts=counts.tolist())
 heads=heads.cuda();proj=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,128)).cuda()
 modules=[model,decoder,predictors,heads,proj];params=[p for m in modules for p in m.parameters()]
 opt=torch.optim.AdamW([dict(params=list(model.parameters())+list(decoder.parameters())+list(predictors.parameters()),lr=.001),dict(params=list(heads.parameters())+list(proj.parameters()),lr=.001)],weight_decay=.05,betas=(.9,.98))
 if not resuming:
  wr(out/'protocol.json',dict(arm=a.arm,fold=a.fold,admission=a.admission,initial_model_sha256=initial,core_parameters=sum(p.numel() for p in model.parameters()),batches_per_epoch=len(dl),presentations_per_epoch=len(concat),epochs=a.epochs,patient_audit=audit,schema=schema,cohort_sha256=sha(R/'cohort_receipt.json'),initialization='random_from_scratch_no_checkpoint',fixed_test_loaded=False,source_freeze_sha256=sha(O/'freeze.json')))
 history=[];updates=0;start=time.time();start_epoch=0;prior_seconds=0.
 if resuming:
  assert read(O/'partial_resume_audit.json')['status']=='passed'
  saved=torch.load(out/'checkpoint_e002.pt',map_location='cpu',weights_only=False)
  for key,module in [('model',model),('decoder',decoder),('predictors',predictors),('aux_heads',heads),('projection',proj)]:module.load_state_dict(saved[key],strict=True)
  opt.load_state_dict(saved['optimizer']);assert saved['epoch']==2 and saved['updates']==350 and saved['initial_sha256']==initial
  torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state(saved['rng_cuda'])
  history=read(out/'history.json');assert len(history)==2 and history==read(O/'partial_origin/history.json')
  updates=350;start_epoch=2;prior_seconds=sum(x['seconds'] for x in history)
  wr(out/'resume_protocol.json',dict(status='restored_for_continuation',saved_epochs=2,saved_updates=350,checkpoint_sha256=sha(out/'checkpoint_e002.pt'),all_components_optimizer_and_RNG_restored=True,source_freeze_sha256=sha(O/'freeze.json'),admission_weights_used=False))
 for ep in range(start_epoch,a.epochs):
  torch.manual_seed(a.seed+ep*100003);torch.cuda.manual_seed_all(a.seed+ep*100003);sampler.seed=a.seed+ep
  for m in modules:m.train()
  n=0;sums={};ts=time.time();nonzero=0
  for step,batch in enumerate(dl):
   if a.max_batches and step>=a.max_batches:break
   phase_total=30*len(dl);s=(ep%30)*len(dl)+step;factor=min(1.,(s+1)/max(1,.02*phase_total))*.5*(1+math.cos(math.pi*s/phase_total))
   opt.param_groups[0]['lr']=(.001 if ep<30 else .0003)*factor;opt.param_groups[1]['lr']=.001*factor
   batch=device(batch);masks=apply_window_mask(batch,.3)
   zero_cpu=torch.get_rng_state();zero_cuda=torch.cuda.get_rng_state();zero_buffers={k:v.detach().clone() for k,v in model.named_buffers()}
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pred=model(batch,modality_drop_p=.25,channel_drop_p=.1);B=len(batch['y']);model._stat_cache={m:batch[m]['wave'] for m in pred['z']}
    terms={'aux':aux_loss(pred,batch,heads,weights)}
    if a.arm!='ce_only':terms.update(masked=masked_stats_loss(model,decoder,batch,masks),loo=loo_cl_loss(pred['z'],pred['owners'],.07,B),xpred=xpred_loss(model,pred['z'],pred['owners'],predictors,B),var=variance_loss(pred['z']),supcon=supcon_loss(pred['z'],pred['owners'],batch['y'],proj,B))
    loss=sum(z.sum()*0 for z in pred['z'].values())
    for k,w in [('masked',1),('loo',1),('xpred',.5),('var',1),('supcon',.5),('aux',0 if a.arm=='no_ce' else (.3 if a.arm!='ce_only' else 1))]:
     if terms.get(k) is not None and w:loss=loss+w*terms[k]
   assert torch.isfinite(loss),'Nonfinite loss'
   opt.zero_grad(set_to_none=True);loss.backward();gn=clip_grad_norm_equivalent(params,1.);assert torch.isfinite(gn)
   if float(gn)==0:
    capture(out,ep+1,step+1,model,decoder,predictors,heads,proj,opt,schema,weights,batch,masks,pred,terms,loss,zero_cpu,zero_cuda,zero_buffers)
   opt.step();n+=1;updates+=1;nonzero+=int(float(gn)>0)
   for k,v in terms.items():
    if v is not None:sums[k]=sums.get(k,0)+float(v.detach())
   sums['total']=sums.get('total',0)+float(loss.detach())
   if step%20==0:wr(out/'progress.json',dict(status='training',arm=a.arm,fold=a.fold,epoch_in_progress=ep+1,epochs_saved=ep,updates=updates,batches_per_epoch=len(dl),pid=os.getpid(),updated_unix=time.time()))
  row=dict(epoch=ep+1,updates=n,cumulative_updates=updates,nonzero_gradient_steps=nonzero,seconds=time.time()-ts,losses={k:v/n for k,v in sums.items()});history.append(row)
  state=dict(model=model.state_dict(),decoder=decoder.state_dict(),predictors=predictors.state_dict(),aux_heads=heads.state_dict(),projection=proj.state_dict(),optimizer=opt.state_dict(),epoch=ep+1,updates=updates,arm=a.arm,fold=a.fold,schema=schema,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),initial_sha256=initial)
  cp=out/f'checkpoint_e{ep+1:03d}.pt';tmp=Path(str(cp)+'.tmp');torch.save(state,tmp);tmp.replace(cp)
  wr(out/'history.json',history);wr(out/'progress.json',dict(status='training',arm=a.arm,fold=a.fold,epochs_saved=ep+1,updates=updates,checkpoint_sha256=sha(cp),batches_per_epoch=len(dl),pid=os.getpid(),updated_unix=time.time()))
  wr(out/'numeric_recovery_events.json',dict(events=numeric_events,formal_optimizer_calls=updates,threshold=1.,native_finite_path_exactly_preserved=True))
  print(json.dumps(row),flush=True)
 final=out/f'checkpoint_e{a.epochs:03d}.pt';saved=torch.load(final,map_location='cpu',weights_only=False)
 check=make_core(a.arm);check.load_state_dict(saved['model'],strict=True);assert digest(check)==digest(model) and digest(model)!=initial
 assert int(max(float(v['step']) for v in saved['optimizer']['state'].values()))<=updates
 done=dict(status='training_completed_awaiting_oof_readouts',arm=a.arm,fold=a.fold,epochs_saved=a.epochs,updates=updates,expected_updates=len(dl)*a.epochs if not a.admission else a.max_batches*a.epochs,checkpoint_sha256=sha(final),model_sha256=digest(check),initial_model_sha256=initial,exact_model_restore=True,wall_seconds=prior_seconds+time.time()-start,pid=os.getpid(),fixed_test_loaded=False)
 assert updates==done['expected_updates'];wr(out/'progress.json',done);wr(out/'training_receipt.json',done)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--fold',type=int,choices=[0,1,2],required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--admission',action='store_true');p.add_argument('--max-batches',type=int,default=0);a=p.parse_args();assert a.fold==0;assert a.admission or not inherit(a.arm,a.seed);run(a)
