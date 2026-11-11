from common import *
import numpy as np
from score_and_stats import registry,paired
verify();reg,cases=registry();r=read(O/'results_pending_audit.json');contract=read(reg['original_test_contract']);assert len(r['task_results'])==690
for c,row in zip(cases,r['task_results']):
 t=c['task'];f=contract['clinical'] if t=='inhouse' else contract['external'][t];y=np.array(f['truth']);g=np.array(f['group'])
 with np.load(c['prediction_path']) as z:p=z['prediction'];assert np.array_equal(z['source_rows'],f['source_rows'])
 if t=='inhouse':
  yy=[];pp=[]
  for u in np.unique(g):
   mask=g==u;assert len(np.unique(y[mask]))==1;yy.append(y[mask][0]);pp.append(np.bincount(p[mask],minlength=c['K']).argmax())
  y=np.array(yy);p=np.array(pp)
 cm=np.zeros((c['K'],c['K']),int);np.add.at(cm,(y,p),1);assert row['confusion']==cm.tolist() and row['correct']==int(cm.trace()) and row['total']==int(cm.sum())
 den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(c['K'],float),where=den>0).mean();nz=cm.sum(1)>0;ba=(np.diag(cm)[nz]/cm.sum(1)[nz]).mean()
 assert abs(f1-row['macro_f1'])<1e-12 and abs(ba-row['balanced_accuracy'])<1e-12
with np.load(O/'score_vectors_private.npz') as pack:
 curves={};truth={}
 for key in pack.files:
  m,t=key.split('__');v=pack[key]
  if m=='groups':truth[t]=dict(group=v)
  else:curves.setdefault(m,{})[t]=v
stats=paired(curves,truth);assert stats==r['paired_statistics']
for a in r['aggregate']:
 for s in a['seeds']:
  ext=[x for x in r['task_results'] if (x['arm'],x['view'],x['seed'])==(a['arm'],a['view'],s['seed']) and x['task']!='inhouse'];assert s['correct']==sum(x['correct'] for x in ext) and s['total']==sum(x['total'] for x in ext)==1731
r['status']='passed_independent_confusion_and_statistics_audit';r['source_results_sha256']=sha(O/'results_pending_audit.json');save(O/'audited_results.json',r)
