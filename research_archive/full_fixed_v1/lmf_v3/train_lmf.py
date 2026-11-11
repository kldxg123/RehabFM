"""Task-local matched CNN frontend with explicitly adapted official LMF fusion."""
import os,json,time,math,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from core_models import make_core,fuse
from train_core import Items,sha,wr,digest
from rehabfm.data import collate_fn
from rehabfm.train_pretrain import apply_window_mask
R=Path(__file__).resolve().parent
from adapter import Classifier
D=R.parent
def move(b,dev):return {k:v.to(dev) if torch.is_tensor(v) else {a:z.to(dev) if torch.is_tensor(z) else z for a,z in v.items()} for k,v in b.items()}
def run(a):
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.set_num_threads(3);torch.manual_seed(0);np.random.seed(0);dev='cpu' if a.cpu else 'cuda'
 if not a.cpu:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
 if not a.admission:
  fr=json.loads((R/'source_freeze.json').read_text())
  for p,h in fr['files'].items():
   if not p.endswith('.pt') or p.endswith('/'+a.task+'.pt'):assert sha(p)==h,p
 t=torch.load(D/'cohort_private'/f'{a.task}.pt',map_location='cpu',weights_only=False);rows=t['items'];K=t['target_classes'];f=np.load(D/'cohort_private/folds.npz',allow_pickle=False);tr=f[f'{a.task}_f{a.fold}_train'];va=f[f'{a.task}_f{a.fold}_query'];assert not {int(rows[i]['subject']) for i in tr}&{int(rows[i]['subject']) for i in va}
 fit=[dict(rows[i]) for i in tr]
 if a.task=='inhouse':
  for r in fit:r['y']=int(r['y'])//2
 if a.admission:fit=fit[:4]
 fit=fit*(1 if a.admission else t['repeat']);root=R/(('admission_cpu_private' if a.cpu else 'admission_gpu_private') if a.admission else 'runs');out=root/f'{a.arm}_{a.task}_f{a.fold}';out.mkdir(parents=True,exist_ok=False)
 model=Classifier(a.arm,K).to(dev);initial=digest(model);count=torch.bincount(torch.tensor([int(x['y']) for x in fit]),minlength=K).float();w=count.clamp_min(1).rsqrt();w[count==0]=0;w=w.to(dev)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05,betas=(.9,.98));updates=0;hist=[];epochs=2 if a.admission else 60;bs=2 if a.admission else 32
 for ep in range(epochs):
  torch.manual_seed(ep*100003);dl=DataLoader(Items(fit),batch_size=bs,shuffle=True,collate_fn=collate_fn,num_workers=0,generator=torch.Generator().manual_seed(ep));model.train();loss_sum=0.
  for step,raw in enumerate(dl):
   phase=30*len(dl);s=(ep%30)*len(dl)+step;factor=min(1.,(s+1)/max(1,.02*phase))*.5*(1+math.cos(math.pi*s/phase));opt.param_groups[0]['lr']=(.001 if ep<30 else .0003)*factor
   b=move(raw,dev);apply_window_mask(b,.3)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev=='cuda'):
    logits=model(b,True);loss=nn.functional.cross_entropy(logits.float(),b['y'],weight=w)
   assert torch.isfinite(loss);opt.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();updates+=1;loss_sum+=float(loss.detach())
  state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep+1,updates=updates,arm=a.arm,task=a.task,fold=a.fold,K=K,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if dev=='cuda' else None)
  cp=out/f'checkpoint_e{ep+1:03d}.pt';torch.save(state,cp);hist.append(dict(epoch=ep+1,updates=updates,loss=loss_sum/len(dl)));wr(out/'history.json',hist);wr(out/'progress.json',dict(status='training',epochs_saved=ep+1,updates=updates,pid=os.getpid(),arm=a.arm,task=a.task,fold=a.fold))
 model.eval();saved=torch.load(cp,map_location='cpu',weights_only=False);rest=Classifier(a.arm,K).to(dev);rest.load_state_dict(saved['model']);rest.eval();assert digest(rest)==digest(model)!=initial
 op2=torch.optim.AdamW(rest.parameters());op2.load_state_dict(saved['optimizer']);assert op2.state_dict()['param_groups']==saved['optimizer']['param_groups']
 for k,st in saved['optimizer']['state'].items():
  for name,v in st.items():assert torch.equal(v.detach().cpu(),op2.state_dict()['state'][k][name].detach().cpu()) if torch.is_tensor(v) else v==op2.state_dict()['state'][k][name]
 result=dict(status='admission_passed' if a.admission else 'training_completed_awaiting_fixed_evaluation',arm=a.arm,task=a.task,fold=a.fold,epochs=epochs,updates=updates,checkpoint_sha256=sha(cp),model_sha256=digest(model),parameters=sum(p.numel() for p in model.parameters()),fold_training_rows=len(tr),fold_query_rows=len(va),fixed_test_loaded=False,trainer_sha256=sha(__file__),adapter_sha256=sha(R/'adapter.py'),model_optimizer_restored=True,device=dev)
 if a.admission:
  with torch.no_grad():assert torch.equal(model(b),rest(b))
 if not a.admission:result['source_freeze_sha256']=sha(R/'source_freeze.json')
 wr(out/'receipt.json',result);wr(out/'progress.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['LMF'],required=True);p.add_argument('--task',required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--admission',action='store_true');p.add_argument('--cpu',action='store_true');run(p.parse_args())
