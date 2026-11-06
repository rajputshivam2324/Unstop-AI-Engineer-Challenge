"""Run original CLI unchanged, preserving prior reports and logging exact commands."""
from pathlib import Path
import os,json,subprocess,time,datetime,sys
R=Path(__file__).resolve().parents[2];V=R/'validation';env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for kind,folder in [('main','corpus'),('edge','edge_corpus'),('semantic','probe_corpus')]:
 out=V/'runs'/kind
 if out.exists():raise SystemExit(f'Refusing overwrite: {out}')
 out.mkdir()
 cmd=[str(R/'.venv/bin/python'),str(R/'run_integrity_check.py'),'--input_dir',str(V/folder),'--output_dir',str(out),'--skip-web']
 start=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (out/'stdout.txt').open('w') as so,(out/'stderr.txt').open('w') as se:p=subprocess.run(cmd,cwd=R,env=env,stdout=so,stderr=se)
 (out/'execution.json').write_text(json.dumps({'command':cmd,'environment_overrides':{k:env[k] for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM')},'start_utc':stamp,'exit_code':p.returncode,'subprocess_wall_s':time.monotonic()-start},indent=2))
 print(kind,p.returncode,flush=True)
