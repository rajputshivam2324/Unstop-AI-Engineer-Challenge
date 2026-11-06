"""Independent worker per method/split: bounded retrieve-then-verify + exact evidence.

No heldout invocation is permitted until code/config/model hashes are frozen.
The pair benchmark evaluates semantic relationships, not misconduct.
"""
from pathlib import Path
import sys,os,json,time,resource,argparse,hashlib,dataclasses
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from integrity_engine.config import load_config
from integrity_engine.types import ProcessedDocument,Page
from integrity_engine.text_processing import build_chunks
from integrity_engine.alignment import TemplateRegistry,DocumentAligner,evidence_pair_scores
from integrity_engine.enhanced_alignment import EnhancedDocumentAligner
from integrity_engine.embeddings import build_embedder
from integrity_engine.indexing import build_index,SelfMatchExclusion,search_neighbors,pairwise_doc_similarity
from integrity_engine.alignment import EvidenceGate
from integrity_engine.metrics import prf
ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['development','heldout','regression'],required=True);ap.add_argument('--mode',choices=['baseline','boundary','semantic_sts','semantic_nli','semantic_hybrid'],required=True);ap.add_argument('--out',required=True)
a=ap.parse_args();output=Path(a.out)
if output.exists():raise SystemExit('Refusing overwrite: '+str(output))
if a.split=='heldout':
 freeze=json.loads((R/'improvements/final_method_freeze.json').read_text())
 for name,h in freeze['sha256'].items():
  if hashlib.sha256((R/name).read_bytes()).hexdigest()!=h:raise SystemExit('Frozen method changed: '+name)
cfg=load_config(R/'config/default.toml');cfg=dataclasses.replace(cfg,matching=dataclasses.replace(cfg.matching,mode=a.mode))
t0=time.monotonic();emb=build_embedder(cfg.embedding,str(R/'cache/hf'));model_load=time.monotonic()-t0
# Establish verifier load independently of measured per-case inference (same convention all modes).
verifier_load=0.;shared_verifier=None
if a.mode.startswith('semantic_'):
 from integrity_engine.semantic import build_verifier
 ts=time.monotonic();shared_verifier=build_verifier(a.mode.removeprefix('semantic_'),str(R/'cache/hf'));shared_verifier.load();verifier_load=time.monotonic()-ts

def score_docs(docs):
 ts=time.monotonic();chunks=[c for d in docs for c in build_chunks(d,cfg)]
 reg=TemplateRegistry();reg.register_all_chunks(chunks)
 mat=emb.encode_corpus([c.text for c in chunks]);idx=build_index(mat);exclude=SelfMatchExclusion([c.document_id for c in chunks])
 neighbors={i:search_neighbors(idx,mat[i:i+1],cfg.pairwise.chunk_query_topk,exclude,i) for i in range(len(chunks))}
 cp=pairwise_doc_similarity(chunks,neighbors,cfg.pairwise,{},EvidenceGate(chunks,emb,cfg))
 al=(DocumentAligner if a.mode=='baseline' else EnhancedDocumentAligner)(emb,reg,cfg,chunks)
 if shared_verifier:al.verifier=shared_verifier
 pairs=evidence_pair_scores(docs,al,cp,{},cfg.pairwise.chunk_query_topk)
 score=max([p.score for p in pairs] or [0.])
 passages,reverse=[],[]
 if score>=cfg.pairwise.min_pair_score_floor:
  passages,_=al.align(docs[0],docs[1]);reverse,_=al.align(docs[1],docs[0])
 return {'predicted':bool(passages or reverse),'score':score,'seconds':time.monotonic()-ts,'passages':[dataclasses.asdict(p) for p in passages]}

results=[]
if a.split in ('development','heldout'):
 filename='heldout_disjoint' if a.split=='heldout' else a.split
 rows=json.loads((R/f'improvements/data/{filename}.json').read_text())
 for row in rows:
  docs=[ProcessedDocument(f'{i+1:02d}_{side}.docx',Path(side),'docx','ok',pages=[Page(None,row[side],0,len(row[side]))],full_text=row[side]) for i,side in enumerate(('left','right'))]
  res=score_docs(docs);res.update({'id':row['id'],'kind':row['kind'],'label':row['label']});results.append(res)
else:
 e=json.loads((R/'validation/extraction.json').read_text())
 for names,label,kind in [(['01_human_speech.pdf','11_human_crossmodal.mp4'],1,'cross_modal'),(['06_policy_argument.docx','07_policy_rewording.docx'],1,'paraphrase'),(['30_assertion.pdf','31_negation.pdf'],0,'contradiction')]:
  docs=[]
  for name in names:
   v=e[name];docs.append(ProcessedDocument(name,Path(name),Path(name).suffix[1:],'ok',pages=[Page(**p) for p in v['pages']],full_text=v['full_text'],metadata=v['metadata']))
  res=score_docs(docs);res.update({'id':' + '.join(names),'kind':kind,'label':label});results.append(res)
tp=sum(r['predicted'] and r['label'] for r in results);fp=sum(r['predicted'] and not r['label'] for r in results);fn=sum(not r['predicted'] and r['label'] for r in results);tn=len(results)-tp-fp-fn
p,r,f1=prf(tp,fp,fn);times=sorted(v['seconds'] for v in results)
payload={'split':a.split,'mode':a.mode,'config':dataclasses.asdict(cfg.matching),'metrics':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':p,'recall':r,'f1':f1},'timing':{'embedding_load_s':model_load,'verifier_load_s':verifier_load,'cases_total_s':sum(times),'median_case_s':times[len(times)//2],'p95_case_s':times[min(len(times)-1,int(.95*len(times)))],'wall_s':time.monotonic()-t0},'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'verifier_pairs_scored':shared_verifier.pairs_scored if shared_verifier else 0,'truncated_pairs':shared_verifier.truncated_pairs if shared_verifier else 0,'cases':results}
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2));print({k:v for k,v in payload.items() if k!='cases'})
