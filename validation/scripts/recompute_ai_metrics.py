from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'benchmarks'));sys.path.insert(0,str(R))
import bench_ai_detection as a
from integrity_engine.config import load_config
cfg=load_config(R/'config/default.toml');cal,test=a.split(a.load_pairs(50,42),42);scores=json.loads((R/'benchmarks/results/ai_components.json').read_text());rows=a.to_rows(test,scores)
x=a.evaluate(rows,cfg,cfg.ai.perplexity_mu,cfg.ai.perplexity_k,cfg.ai.burst_mu,cfg.ai.burst_k)
old=json.loads((R/'benchmarks/results/ai_detection.json').read_text())['test_with_shipped_config']
(R/'validation/ai_metrics_recomputed.json').write_text(json.dumps({'method':'Recomputed from preserved cached component outputs and reconstructed test qids; no model inference rerun.','result':x,'matches_saved':x==old},indent=2));print(x['fused_at_ai_likely_band'])
