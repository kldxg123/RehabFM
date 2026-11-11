"""Independent original-fixed statistic reconstruction; no inference or fitting."""
from pathlib import Path
import sys,json,math,datetime
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from fixed_interfaces import read,sha
O=R/'fixed_statistics_v1';N=R/'fixed_eval_panels/neural_common_v1';C=R/'fixed_eval_panels/statistics_common_v1'


def adjusted(pvalues):
 # Independently implement sorted cumulative maxima, with stable equal-p ranks.
 p=np.asarray(pvalues,float);order=np.lexsort((np.arange(len(p)),p));q=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)));out=np.zeros_like(q);out[order]=q;return out


def exact_discordance(first,second):
 n10=int(np.logical_and(first,np.logical_not(second)).sum());n01=int(np.logical_and(second,np.logical_not(first)).sum());n=n10+n01
 if not n:return 1.
 # Symmetric upper/lower binomial tails rather than the producer's combination sum.
 distribution=np.array([math.factorial(n)/(math.factorial(k)*math.factorial(n-k))/2**n for k in range(n+1)])
 return float(min(1,2*distribution[:min(n10,n01)+1].sum()))


def votes(y,p,g,K):
 targets=[];predictions=[]
 for value in sorted(set(g.tolist())):
  mask=g==value;labels=set(y[mask].tolist());assert len(labels)==1
  counts=[int(np.count_nonzero(p[mask]==label)) for label in range(K)]
  targets.append(next(iter(labels)));predictions.append(counts.index(max(counts)))
 return np.asarray(targets),np.asarray(predictions)


def close(a,b):
 assert np.asarray(a).shape==np.asarray(b).shape and np.allclose(a,b,atol=2e-12,rtol=0),(a,b)


def main():
 dest=O/'independent_audit.json';assert not dest.exists();plan=read(R/'fixed_statistics_plan001.json');result=read(O/'results.json');freeze=read(O/'source_freeze.json')
 assert result['status']=='computed_pending_independent_audit' and result['plan_sha256']==sha(R/'fixed_statistics_plan001.json')==read(R/'fixed_statistics_plan001_freeze.json')['plan_sha256']
 assert result['source_freeze_sha256']==sha(O/'source_freeze.json') and freeze['comparisons']==36 and freeze['external_tests']==108 and freeze['internal_tests']==36
 for p,h in freeze['files'].items():assert sha(p)==h,p
 for name,h in result['private_hashes'].items():assert sha(O/'private'/name)==h,name
 registries={str(panel):read(panel/'models_for_fixed.json') for panel in [N,C]};summaries={str(panel):read(panel/'audited_fixed_results.json') for panel in [N,C]}
 assert all(s['status']=='passed_independent_fixed_test_audit' for s in summaries.values())
 assert registries[str(N)]['original_test_contract']==registries[str(C)]['original_test_contract'];contract=read(registries[str(N)]['original_test_contract'])
 models=result['models'];assert models==sorted({p[k] for p in plan['pairs'] for k in ['a','b']});model_index={m:i for i,m in enumerate(models)};D=plan['draws'];assert result['draws']==D==10000 and result['seed']==plan['seed']
 rng=np.random.default_rng(plan['seed']);task_data={};clinical=None;correct_totals={m:0 for m in models}
 for task in plan['partitions']['all_external']+['inhouse']:
  c=contract['clinical'] if task=='inhouse' else contract['external'][task];truth=np.asarray(c['truth']);groups=np.asarray(c['group']);correct=[]
  for model in models:
   panel=C if model in ['B0','B1','B2','B3','B4'] else N;reg=registries[str(panel)];keys=[key for key,v in reg['jobs'].items() if v['method']==model and v['task']==task];assert len(keys)==1;key=keys[0]
   path=panel/'predictions_private'/key/'prediction.npz';assert sha(path)==read(path.parent/'seal.json')['prediction_sha256']
   with np.load(path,allow_pickle=False) as z:
    assert np.array_equal(z['source_rows'],np.asarray(c['source_rows']));pred=z['prediction'];assert pred.shape==truth.shape
   if task=='inhouse':yy,pp=votes(truth,pred,groups,3);assert len(yy)==10;row=yy==pp
   else:row=truth==pred;correct_totals[model]+=int(row.sum())
   # Cross-check against the independently audited original fixed result table.
   match=[x for x in summaries[str(panel)]['task_results'] if x['job']==key];assert len(match)==1 and match[0]['correct']==int(row.sum()) and match[0]['total']==len(row)
   correct.append(row)
  correct=np.asarray(correct)
  if task=='inhouse':
   clinical=correct
   with np.load(O/'private/clinical.npz',allow_pickle=False) as z:assert set(z.files)=={'correct'} and np.array_equal(z['correct'],correct)
  else:
   unique=np.unique(groups);sizes=np.array([int(np.count_nonzero(groups==g)) for g in unique]);group_correct=np.column_stack([correct[:,groups==g].sum(1) for g in unique])
   # Reconstruct RNG from the locked seed and original task order.
   indices=rng.integers(0,len(unique),size=(D,len(unique)),dtype=np.int64);signs=rng.choice(np.array([-1,1],np.int8),size=(D,len(unique)))
   with np.load(O/'private'/(task+'.npz'),allow_pickle=False) as z:
    for name,value in [('correct',correct),('groups',groups),('group_sizes',sizes),('correct_by_group',group_correct),('indices',indices),('signs',signs)]:assert np.array_equal(z[name],value),(task,name)
    # Work method-by-method instead of the producer's three-dimensional gather.
    denominator=sizes[indices].sum(axis=1);boot=np.array([v[indices].sum(axis=1)/denominator for v in group_correct]);point=correct.sum(axis=1)/correct.shape[1]
    close(z['bootstrap_accuracy'],boot);close(z['point_accuracy'],point)
   task_data[task]=dict(n=correct.shape[1],correct=correct,group_correct=group_correct,boot=boot,point=point,signs=signs)
 assert sum(v['n'] for v in task_data.values())==1731
 with np.load(O/'private/draws.npz',allow_pickle=False) as z:
  ext_boot=z['external_bootstrap'];ext_perm=z['external_permutation'];int_boot=z['internal_bootstrap'];int_indices=z['internal_indices'];assert ext_boot.shape==ext_perm.shape==(108,D) and int_boot.shape==(36,D)
  assert np.array_equal(int_indices,rng.integers(0,10,size=(D,10)))
 external=[];index=0
 for partition,tasks in plan['partitions'].items():
  total=sum(task_data[t]['n'] for t in tasks);assert total==plan['original_fixed_records_by_partition'][partition]
  for pair in plan['pairs']:
   a=model_index[pair['a']];b=model_index[pair['b']];ac=sum(int(task_data[t]['correct'][a].sum()) for t in tasks);bc=sum(int(task_data[t]['correct'][b].sum()) for t in tasks);point=(ac-bc)/total
   boot=sum((task_data[t]['boot'][a]-task_data[t]['boot'][b])*(task_data[t]['n']/total) for t in tasks)
   perm=np.zeros(D)
   for t in tasks:
    d=task_data[t];delta=d['group_correct'][a]-d['group_correct'][b]
    # Independent columnwise signed reduction of whole-group swaps.
    for g,value in enumerate(delta):perm+=d['signs'][:,g]*float(value)/total
   close(ext_boot[index],boot);close(ext_perm[index],perm)
   lo,hi=np.percentile(boot,[2.5,97.5]);p=(np.count_nonzero(np.abs(perm)>=abs(point)-1e-12)+1)/(D+1);row=result['external'][index]
   assert row['partition']==partition and row['external_fixed_records']==total and all(row[k]==pair[k] for k in pair)
   close(row['difference'],point);close(row['CI95'],[lo,hi]);close(row['p_unadjusted'],p);external.append(p);index+=1
 assert index==len(result['external'])==108
 close([x['p_Holm'] for x in result['external']],adjusted(external))
 internal=[]
 for i,pair in enumerate(plan['pairs']):
  a=clinical[model_index[pair['a']]];b=clinical[model_index[pair['b']]];delta=a.astype(np.int64)-b.astype(np.int64);boot=delta[int_indices].sum(1)/10;close(int_boot[i],boot);row=result['internal'][i]
  assert all(row[k]==pair[k] for k in pair) and row['internal_fixed_patients']==10 and row['a_correct']==int(a.sum()) and row['b_correct']==int(b.sum())
  close(row['difference'],delta.sum()/10);close(row['CI95'],np.percentile(boot,[2.5,97.5]));p=exact_discordance(a,b);close(row['p_unadjusted'],p);internal.append(p)
 assert len(result['internal'])==36;close([x['p_Holm'] for x in result['internal']],adjusted(internal))
 audit=dict(status='passed',original_external_fixed_records=1731,original_internal_fixed_patients=10,comparisons=36,external_tests=108,internal_tests=36,draws_per_test=D,all_private_arrays_reconstructed_from_original_predictions=True,RNG_replayed_from_locked_seed=True,Holm_families=[108,36],source_sha256=sha(Path(__file__)),results_sha256=sha(O/'results.json'),new_fits=0,new_predictions=0,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with dest.open('x') as h:json.dump(audit,h,indent=2)
 result.update(status='passed_independent_fixed_statistics_audit',independent_audit_sha256=sha(dest))
 with (O/'audited_results.json').open('x') as h:json.dump(result,h,indent=2)
 print(json.dumps(audit))


if __name__=='__main__':main()
