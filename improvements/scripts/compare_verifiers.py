"""Development-only model comparison. Heldout is deliberately not opened."""
from pathlib import Path
import sys,json,time,resource,os
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
os.environ['HF_HUB_OFFLINE']='1'
from integrity_engine.semantic import SemanticVerifier
from integrity_engine.embeddings import Embedder
rows=json.loads((R/'improvements/data/development.json').read_text());pairs=[(r['left'],r['right']) for r in rows]
out={}
for name in ('sts','nli'):
 t=time.monotonic();v=SemanticVerifier(name,str(R/'cache/hf'));v.load();load=time.monotonic()-t
 t=time.monotonic();scores=v.score_pairs(pairs);elapsed=time.monotonic()-t
 out[name]={'load_seconds':load,'inference_seconds':elapsed,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'scores':[{'id':r['id'],'kind':r['kind'],'label':r['label'],'similarity':s.similarity,'contradiction':s.contradiction,'truncated':s.truncated} for r,s in zip(rows,scores)]};v.release()
 for rr in out[name]['scores']:print(name,rr,flush=True)
(R/'improvements/runs/development_verifiers.json').write_text(json.dumps(out,indent=2))
