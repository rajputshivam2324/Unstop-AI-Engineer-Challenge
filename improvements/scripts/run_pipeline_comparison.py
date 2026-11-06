"""Full unchanged-schema CLI comparison on the prior 13-file development corpus."""
from pathlib import Path
import subprocess,time,json,datetime
R=Path(__file__).resolve().parents[2]
for mode in ['baseline','boundary','semantic_nli','semantic_hybrid']:
 out=R/f'improvements/runs/pipeline_{mode}'
 if out.exists():raise SystemExit('Refusing overwrite: '+str(out))
 out.mkdir()
 cmd=[str(R/'.venv/bin/python'),str(R/'run_integrity_check.py'),'--input_dir',str(R/'validation/corpus'),'--output_dir',str(out),'--skip-web','--matching-mode',mode]
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
 with (out/'stdout.txt').open('w') as so,(out/'stderr.txt').open('w') as se:
  p=subprocess.run(cmd,cwd=R,stdout=so,stderr=se,env={**__import__('os').environ,'HF_HUB_OFFLINE':'1'})
 (out/'execution.json').write_text(json.dumps({'command':cmd,'start_utc':start,'exit_code':p.returncode,'subprocess_s':time.monotonic()-t},indent=2))
 print(mode,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
