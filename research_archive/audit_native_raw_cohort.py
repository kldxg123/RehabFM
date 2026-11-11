"""Read-only cohort feasibility audit; no training/fixed-file reads."""
from pathlib import Path
import collections,hashlib,json,time
import numpy as np
import torch
R=Path(__file__).resolve().parent
B=Path('/path/to/rehabfm-workspace/rehabfm_unified_context_20261019_v1')
V=B/'clean_subset_all23_20261021_v2'
P=Path('/path/to/rehabfm-workspace/rehabfm_refine_20261015_1300/all23_training_pool_r2_server_only.pt')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(16*1024*1024),b''):h.update(z)
 return h.hexdigest()
def arr(v):return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
def main():
 torch.set_num_threads(2)
 assert not (R/'native_raw_cohort_feasibility.json').exists()
 manifest=json.loads((V/'training_pool_manifest.json').read_text());review=json.loads((V/'clean_subset045_review.json').read_text())
 assert sha(V/'training_pool_manifest.json')==review['training_manifest_sha256']
 tasks=torch.load(P,map_location='cpu',weights_only=False)
 assert isinstance(tasks,list) and len(tasks)==23
 mapping={t['name']:t for t in tasks};rows=[];allok=True
 for e in manifest['tasks']:
  t=mapping[e['task']];pool=t['fit']+t['val']
  with np.load(V/'training_only'/e['filename'],allow_pickle=False) as z:g=z['group'];y=z['y']
  allowed=set(g.tolist());kept=[r for r in pool if int(r['subject']) in allowed]
  rg=np.array([int(r['subject']) for r in kept]);ry=np.array([int(r['y']) for r in kept]);ry=ry//2 if t['name']=='inhouse' else ry
  if t['name']=='inhouse':assert len(np.unique(rg))==19 and not (set(rg.tolist()) & {9000+i for i in [5,7,13,16,17,21,22,25,27,29]})
  same=collections.Counter(zip(rg.tolist(),ry.tolist()))==collections.Counter(zip(g.tolist(),y.tolist()))
  hashes=collections.defaultdict(set);valid=True;modshape={}
  for r in kept:
   h=hashlib.sha256()
   for m,v in sorted(r['tokens'].items()):
    a=arr(v);valid &= bool(np.isfinite(a).all());h.update(m.encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes())
    modshape.setdefault(m,[]).append(list(a.shape))
   hashes[h.hexdigest()].add(int(r['subject']))
  collisions=sum(len(v)>1 for v in hashes.values())
  row=dict(task=t['name'],raw_original_rows=len(pool),raw_legal_rows=len(kept),descriptor_rows=len(g),raw_patients=len(set(rg.tolist())),descriptor_patients=len(allowed),same_group_target_multiset=same,finite=valid,cross_group_exact_token_collisions=collisions,modalities={m:{'records':len(v),'window_range':[min(x[0] for x in v),max(x[0] for x in v)],'channels_range':[min(x[1] for x in v),max(x[1] for x in v)],'time_widths':sorted(set(x[2] for x in v))} for m,v in modshape.items()})
  allok &= same and valid and collisions==0;rows.append(row)
  print(json.dumps(row),flush=True)
 out=dict(status='aligned' if allok else 'alignment_gaps_require_explicit_protocol',tasks=rows,source_path=str(P),source_sha256=sha(P),manifest_sha256=sha(V/'training_pool_manifest.json'),fixed_data_loaded=False,neural_updates=0,completed_unix=time.time())
 (R/'native_raw_cohort_feasibility.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
