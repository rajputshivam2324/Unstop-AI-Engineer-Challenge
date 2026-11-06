"""Original retrieval benchmark; outputs redirected to a NEW validation folder."""
from pathlib import Path
import os,sys,argparse,json,time,datetime
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'benchmarks'));sys.path.insert(0,str(R))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import bench_retrieval as b
p=argparse.ArgumentParser();p.add_argument('--output',default='validation/reproduction/retrieval');a=p.parse_args()
out=Path(a.output).resolve()
if out.exists():raise SystemExit(f'Refusing to overwrite {out}')
out.mkdir(parents=True)
b.RESULTS=out
s=time.monotonic();code=b.main()
(out/'execution.json').write_text(json.dumps({'script':'benchmarks/bench_retrieval.py','output_override_only':str(out),'exit_code':code,'wall_seconds':time.monotonic()-s,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
raise SystemExit(code)
