"""Original clipping exactly when native norm finite; FP64 accumulation only for finite-gradient overflow."""
import torch
from torch import nn
events=[]
@torch.no_grad()
def clip_grad_norm_equivalent(parameters,max_norm):
 params=list(parameters);grads=[p.grad for p in params if p.grad is not None]
 assert grads,'disconnected objective'
 native=nn.utils.get_total_norm(grads,2.,False,None)
 clip_grad_norm_equivalent.last_fallback=False
 if torch.isfinite(native) and (float(native)!=0 or not any(torch.count_nonzero(g) for g in grads)):return nn.utils.clip_grad_norm_(params,max_norm)
 assert all(torch.isfinite(g).all() for g in grads),'nonfinite gradient elements remain fatal'
 pieces=[torch.linalg.vector_norm(g.detach().double(),2) for g in grads]
 stable=torch.linalg.vector_norm(torch.stack(pieces),2)
 assert torch.isfinite(stable) and stable>0,'nonfinite true norm remains fatal'
 coefficient=(float(max_norm)/(stable+1e-6)).clamp(max=1.)
 for g in grads:g.mul_(coefficient.to(device=g.device,dtype=g.dtype))
 assert all(torch.isfinite(g).all() for g in grads)
 clip_grad_norm_equivalent.last_fallback=True
 events.append(dict(native_norm=str(float(native)),stable_norm=float(stable),max_norm=float(max_norm)))
 return stable
clip_grad_norm_equivalent.last_fallback=False
