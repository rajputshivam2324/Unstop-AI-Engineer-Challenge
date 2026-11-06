"""Sequential fresh processes, refusing to overwrite previous result files."""
from pathlib import Path
import subprocess,sys,argparse,json,datetime,time
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','regression','heldout'],required=True);a=p.parse_args()
for mode in ['baseline','boundary','semantic_sts','semantic_nli','semantic_hybrid']:
 prefix=R/f'improvements/runs/{a.split}_{mode}_final'
 cmd=[str(R/'.venv/bin/python'),str(R/'improvements/scripts/benchmark_matching.py'),'--split',a.split,'--mode',mode,'--out',str(prefix.with_suffix('.json'))]
 if prefix.with_suffix('.json').exists():raise SystemExit('Refusing overwrite')
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
 with prefix.with_suffix('.stdout').open('w') as out,prefix.with_suffix('.stderr').open('w') as err:
  result=subprocess.run(cmd,cwd=R,stdout=out,stderr=err)
 prefix.with_suffix('.execution.json').write_text(json.dumps({'command':cmd,'start_utc':started,'wall_s':time.monotonic()-t,'exit_code':result.returncode},indent=2))
 print(a.split,mode,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
