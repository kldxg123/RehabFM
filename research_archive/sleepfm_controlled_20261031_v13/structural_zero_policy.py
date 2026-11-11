"""Strict classification of unavailable objectives; no tolerance, no data-dependent score selection."""
import math
def explainable(arm, batch_records, unique_labels, remaining_modalities, valid_masked_windows, terms, total, variance_std_min=None):
 if not isinstance(batch_records,int) or batch_records<1:return False
 if not isinstance(unique_labels,int) or not 1<=unique_labels<=batch_records:return False
 if not math.isfinite(total) or total!=0:return False
 if any(v is not None and not math.isfinite(v) for v in terms.values()):return False
 if arm=='ce_only':
  return unique_labels==1 and set(terms)=={'aux'} and terms['aux'] is None
 common=(arm in ['full','no_condition','official_pool','sleepfm_arch'] and unique_labels==1 and remaining_modalities==1 and valid_masked_windows==0 and set(terms)=={'aux','masked','loo','xpred','var','supcon'} and terms['masked']==0 and all(terms[k] is None for k in ['aux','loo','xpred','supcon']))
 if not common:return False
 if batch_records==1:return terms['var'] is None
 return terms['var']==0 and variance_std_min is not None and math.isfinite(variance_std_min) and variance_std_min>=1.
def validate_counts(updates,nonzero,explained):
 assert isinstance(updates,int) and isinstance(nonzero,int) and isinstance(explained,int)
 assert updates>0 and 0<=nonzero<=updates and explained>=0
 assert nonzero+explained==updates,'unexplained zero gradients or inaccurate accounting'
