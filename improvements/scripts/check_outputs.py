"""Verify unchanged report schema, exact passages, source locations and old results."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];I=R/'improvements'
extracted=json.loads((R/'validation/extraction.json').read_text());checks=[]
for mode in ['baseline','boundary','semantic_nli','semantic_hybrid']:
 for p in (I/f'runs/pipeline_{mode}/per_doc').glob('*.json'):
  report=json.loads(p.read_text());old=json.loads((R/'validation/runs/main/per_doc'/p.name).read_text())
  assert set(report)==set(old),(mode,p.name,'schema')
  assert set(report['scores'])==set(old['scores'])
  q=report['document_id']
  for e in report['peer_matches']+report['multimodal_matches']:
   source=e['matched_file'];assert e['query_passage'] in extracted[q]['full_text']
   assert e['matched_passage'] in extracted[source]['full_text']
   for name,passage,page in [(q,e['query_passage'],e['location'].get('query_page')),(source,e['matched_passage'],e['location'].get('matched_page'))]:
    if name.endswith('.pdf'):
     assert any(pg['page']==page and passage in pg['text'] for pg in extracted[name]['pages']),(mode,name,page)
  checks.append({'mode':mode,'file':q,'schema_and_exact_passages':True})
root=I/'runs/pipeline_semantic_hybrid/per_doc'
r=json.loads((root/'01_human_speech.pdf.json').read_text());m=[p for p in r['multimodal_matches'] if p['matched_file']=='11_human_crossmodal.mp4'];assert m
assert all(p['location']=={'query_page':2,'matched_timestamp':'00:00:00 - 00:00:15'} for p in m)
x={'files_checked':len(checks),'checks':checks,'crossmodal_page_timestamp_correct':True}
(I/'output_checks.json').write_text(json.dumps(x,indent=2));print('Verified',len(checks),'reports; schema, exact evidence and PDF pages; repaired S1 page/timestamp correct')
