from pathlib import Path
import subprocess
R=Path(__file__).resolve().parents[1]
for p in sorted((R/'corpus').glob('*.docx')):
 d=R/'qa'/p.stem;d.mkdir(exist_ok=True)
 subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/unstop-validation-lo','--headless','--convert-to','pdf','--outdir',str(d),str(p)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 subprocess.run(['pdftoppm','-scale-to','1000','-png',str(d/(p.stem+'.pdf')),str(d/'page')],check=True)
for p in sorted((R/'corpus').glob('*.pdf')):
 d=R/'qa'/p.stem;d.mkdir(exist_ok=True)
 subprocess.run(['pdftoppm','-scale-to','1000','-png',str(p),str(d/'page')],check=True)
