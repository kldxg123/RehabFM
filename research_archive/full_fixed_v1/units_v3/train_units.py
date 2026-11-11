"""Fold-local joint UniTS training; all private arrays and states stay remote."""
import os,sys,json,time,math,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from adapter import make_model,fit_schema,transform,sha
from admit_cpu import digest
R=Path(__file__).resolve().parent;D=R.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def load_fold(fold,admission=False):
 cohort=json.loads((D/'cohort_receipt.json').read_text())['tasks'];folds=np.load(D/'cohort_private/folds.npz',allow_pickle=False);schemas=[];data=[];audits=[]
 for e in cohort:
  name=e['task'];t=torch.load(D/'cohort_private'/f'{name}.pt',map_location='cpu',weights_only=False);tr=folds[f'{name}_f{fold}_train'];va=folds[f'{name}_f{fold}_query'];rows=t['items'];fit=[rows[int(i)] for i in tr]
  assert not {int(r['subject']) for r in fit}&{int(rows[int(i)]['subject']) for i in va}
  schema=fit_schema(name,fit,t['target_classes']);schemas.append(schema)
  chosen=fit[:4] if admission else fit
  x=torch.stack([transform(r,schema) for r in chosen]);y=torch.tensor([int(r['y'])//2 if name=='inhouse' else int(r['y']) for r in chosen]);counts=torch.bincount(torch.tensor([int(r['y'])//2 if name=='inhouse' else int(r['y']) for r in fit]),minlength=t['target_classes']).float();w=counts.clamp_min(1).rsqrt();w[counts==0]=0
  data.append(dict(task=name,x=x,y=y,weight=w,repeat=t['repeat']))
  audits.append(dict(task=name,fit_rows=len(tr),query_rows=len(va),train_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),query_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest(),core_gradient=True,K=t['target_classes']))
  del t,rows,fit,chosen
 return schemas,data,audits
def schedule(data,ep):
 g=torch.Generator().manual_seed(ep);lists=[]
 for i,d in enumerate(data):
  indices=torch.cat([torch.randperm(len(d['y']),generator=g) for _ in range(d['repeat'])]);lists.append([(i,x) for x in indices.split(4)])
 return [q[j] for j in range(max(map(len,lists))) for q in lists if j<len(q)]
def evaluate(model,schemas,fold,out):
 folds=np.load(D/'cohort_private/folds.npz',allow_pickle=False);model.eval();ident=digest(model);records=[]
 for task_id,s in enumerate(schemas):
  name=s['task'];t=torch.load(D/'cohort_private'/f'{name}.pt',map_location='cpu',weights_only=False);va=folds[f'{name}_f{fold}_query'];rows=[t['items'][int(i)] for i in va];probs=[]
  with torch.no_grad():
   for start in range(0,len(rows),4):
    x=torch.stack([transform(r,s) for r in rows[start:start+4]]).cuda();p=model.classification(x,None,task_id).float().softmax(-1);assert p.shape==(len(x),s['K']) and torch.isfinite(p).all();probs.append(p.cpu().numpy())
  probability=np.concatenate(probs);pred=probability.argmax(1);truth=np.array([int(r['y'])//2 if name=='inhouse' else int(r['y']) for r in rows]);groups=np.array([int(r['subject']) for r in rows]);p=out/f'{name}_prediction.npz'
  assert not p.exists();np.savez_compressed(p,probability=probability,prediction=pred,truth=truth,group=groups,rows=va)
  assert digest(model)==ident;records.append(dict(task=name,K=s['K'],query_rows=len(va),model_sha256=ident,prediction_sha256=sha(p)));write(out/'evaluation_progress.json',dict(status='running',completed=len(records),expected=23))
 write(out/'evaluation_receipt.json',dict(status='passed',tasks=23,records=records,model_sha256=ident,fixed_test_loaded=False))
def main(a):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2';torch.set_num_threads(3);torch.manual_seed(0);np.random.seed(0)
 if not a.admission:
  fr=json.loads((R/'source_freeze.json').read_text())
  for p,h in fr['files'].items():assert sha(p)==h,p
 out=R/('admission_private' if a.admission else 'runs')/f'f{a.fold}';out.mkdir(parents=True,exist_ok=False)
 schemas,data,audits=load_fold(a.fold,a.admission);model=make_model(schemas).cuda();initial=digest(model);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-6)
 steps_per_epoch=math.ceil(len(schedule(data,0))/8);epochs=1 if a.admission else 60;updates=0;hist=[]
 write(out/'protocol.json',dict(schemas=schemas,patient_audit=audits,fold=a.fold,epochs=epochs,admission=a.admission,initial_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),expected_updates=6 if a.admission else steps_per_epoch*60,steps_per_epoch=steps_per_epoch,source_freeze_sha256=sha(R/'source_freeze.json') if not a.admission else None,trainer_sha256=sha(__file__),fixed_test_loaded=False))
 for ep in range(epochs):
  torch.manual_seed(ep*100003);torch.cuda.manual_seed_all(ep*100003);model.train();plan=schedule(data,ep)
  if a.admission:
   # Six actual optimizer updates, cycling every legal task (including K20/22).
   plan=[(i,torch.arange(min(4,len(d['y'])))) for i,d in enumerate(data)]*2
  losses=[];start=time.time();opt.zero_grad(set_to_none=True)
  for j,(task_id,ix) in enumerate(plan):
   d=data[task_id];x=d['x'][ix].cuda();y=d['y'][ix].cuda();remaining=min(8,len(plan)-(j//8)*8);s=ep*steps_per_epoch+j//8;total=60*steps_per_epoch;factor=min(1.,(s+1)/max(1,.02*total))*.5*(1+math.cos(math.pi*s/total));opt.param_groups[0]['lr']=1e-4*factor
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.classification(x,None,task_id);loss=torch.nn.functional.cross_entropy(logits.float(),y,weight=d['weight'].cuda())
   assert torch.isfinite(loss);(loss/remaining).backward();losses.append(float(loss.detach()))
   if (j+1)%8==0 or j+1==len(plan):
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),100.);assert torch.isfinite(gn) and gn>0;opt.step();opt.zero_grad(set_to_none=True);updates+=1
   if j%40==0:write(out/'progress.json',dict(status='training',epoch_in_progress=ep+1,epochs_saved=ep,updates=updates,microbatch=j+1,microbatches_total=len(plan),pid=os.getpid(),updated_unix=time.time()))
  cp=out/f'checkpoint_e{ep+1:03d}.pt';tmp=Path(str(cp)+'.tmp');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),schemas=schemas,epoch=ep+1,updates=updates,fold=a.fold,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),tmp);tmp.replace(cp)
  hist.append(dict(epoch=ep+1,updates=updates,microbatches=len(plan),mean_microbatch_loss=float(np.mean(losses)),seconds=time.time()-start));write(out/'history.json',hist)
  write(out/'progress.json',dict(status='training',epochs_saved=ep+1,updates=updates,pid=os.getpid()))
 model.eval();saved=torch.load(cp,map_location='cpu',weights_only=False);rest=make_model(schemas).cuda();rest.load_state_dict(saved['model']);rest.eval();assert digest(rest)==digest(model)!=initial
 newopt=torch.optim.AdamW(rest.parameters());newopt.load_state_dict(saved['optimizer']);rr=newopt.state_dict();assert rr['param_groups']==saved['optimizer']['param_groups']
 for k,st in saved['optimizer']['state'].items():
  for name,v in st.items():assert torch.equal(rr['state'][k][name].detach().cpu(),v.detach().cpu()) if torch.is_tensor(v) else rr['state'][k][name]==v
 assert updates==(6 if a.admission else steps_per_epoch*60)
 batch_checks=[]
 if a.admission:
  with torch.no_grad():
   for name in ['inhouse','ulred','phytmo','gaitrec','hsstroke']:
    tid=next(i for i,s in enumerate(schemas) if s['task']==name);x=data[tid]['x'][:3].cuda();A=model.classification(x,None,tid);B=torch.cat([model.classification(x[i:i+1],None,tid) for i in range(len(x))]);assert torch.allclose(A,B,atol=1e-4,rtol=1e-4);assert torch.equal(A,rest.classification(x,None,tid));batch_checks.append(dict(task=name,max_batch_error=float((A-B).abs().max())))
 del rest,newopt;torch.cuda.empty_cache()
 result=dict(status='admission_passed' if a.admission else 'training_completed',fold=a.fold,epochs=epochs,updates=updates,model_sha256=digest(model),checkpoint_sha256=sha(cp),model_optimizer_restored=True,trained_parameter_tensors=len(saved['optimizer']['state']),total_parameter_tensors=len(list(model.parameters())),batch_checks=batch_checks,query_truth_used_for_prediction=False,fixed_test_loaded=False)
 write(out/'training_receipt.json',result)
 # Evaluation is performed separately for all three folds with the frozen slot rule.
 write(out/'progress.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=[0,1,2],required=True);p.add_argument('--admission',action='store_true');main(p.parse_args())
