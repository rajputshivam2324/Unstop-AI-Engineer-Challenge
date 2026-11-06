"""Serial final comparisons: avoid concurrent benchmark CPU contention."""
from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[2];py=str(R/'.venv/bin/python')
commands=[['improvements/scripts/run_comparison.py','--split','regression'],['improvements/scripts/run_pipeline_comparison.py'],['improvements/scripts/run_comparison.py','--split','heldout']]
for cmd in commands:
 p=subprocess.run([py,*cmd],cwd=R)
 if p.returncode:raise SystemExit(p.returncode)
