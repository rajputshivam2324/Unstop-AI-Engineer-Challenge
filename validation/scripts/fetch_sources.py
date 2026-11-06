from pathlib import Path
import subprocess,json,hashlib,datetime
root=Path(__file__).resolve().parents[1]/'sources'
urls={'jfk.ogg':'https://upload.wikimedia.org/wikipedia/commons/d/d5/JFK_inaugural_address.ogg?download=1','jfk.txt':'https://www.gutenberg.org/cache/epub/3/pg3.txt','alice.txt':'https://www.gutenberg.org/cache/epub/11/pg11.txt','time_machine.txt':'https://www.gutenberg.org/cache/epub/35/pg35.txt','jfk_license.html':'https://commons.wikimedia.org/wiki/File:JFK_inaugural_address.ogg'}
rows=[]
for name,url in urls.items():
 p=root/name
 if not p.exists():
  subprocess.run(['curl','-fL','--max-time','60',url,'-o',str(p)],check=True)
 rows.append({'file':name,'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(root/'downloads.json').write_text(json.dumps({'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows},indent=2))
print(rows)
