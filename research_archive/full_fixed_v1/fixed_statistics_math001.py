"""Paired group bootstrap/permutation math, independent of model inference."""
import math
import numpy as np


def holm(pvalues):
 p=np.asarray(pvalues,float);assert p.ndim==1 and np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
 order=np.argsort(p,kind='stable');out=np.empty_like(p);running=0.
 for j,i in enumerate(order):running=max(running,(len(p)-j)*p[i]);out[i]=min(1.,running)
 return out


def mcnemar(a,b):
 a=np.asarray(a,bool);b=np.asarray(b,bool);ab=int((a&~b).sum());ba=int((~a&b).sum());n=ab+ba
 return min(1.,2*sum(math.comb(n,j) for j in range(min(ab,ba)+1))/2**n) if n else 1.


def cluster_draws(correct,groups,draws,rng):
 """All methods share draws; records of one metadata group stay together."""
 correct=np.asarray(correct,np.int64);groups=np.asarray(groups);assert correct.ndim==2 and correct.shape[1]==len(groups)
 unique=np.unique(groups);counts=np.array([(groups==g).sum() for g in unique],np.int64)
 totals=np.stack([correct[:,groups==g].sum(1) for g in unique],1)
 indices=rng.integers(0,len(unique),size=(draws,len(unique)),dtype=np.int64)
 signs=rng.choice(np.array([-1,1],np.int8),size=(draws,len(unique)))
 boot=totals[:,indices].sum(2)/counts[indices].sum(1)[None,:]
 return dict(group_sizes=counts,correct_by_group=totals,indices=indices,signs=signs,bootstrap_accuracy=boot,point_accuracy=correct.mean(1))


def summarize_difference(point,bootstrap,permutation):
 b=np.asarray(bootstrap,float);p=np.asarray(permutation,float);assert b.ndim==p.ndim==1 and len(b)==len(p)
 lo,hi=np.quantile(b,[.025,.975]);tail=int((np.abs(p)>=abs(point)-1e-12).sum())
 return dict(difference=float(point),CI95=[float(lo),float(hi)],p_unadjusted=float((1+tail)/(len(p)+1)))
