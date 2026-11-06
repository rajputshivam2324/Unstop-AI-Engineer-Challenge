"""Pre-inference data audit: reserve a source-disjoint PAWS split.

The first proposed test-row slice shares 76 sentence strings with the old PAWS
benchmark. It remains preserved, unscored; it is NOT used as the final test.
"""
from pathlib import Path
import json,subprocess,datetime,hashlib
R=Path(__file__).resolve().parents[2];D=R/'improvements/data'
p=D/'heldout_disjoint.json'
if p.exists():raise SystemExit('Refusing overwrite')
url='https://datasets-server.huggingface.co/rows?dataset=google-research-datasets%2Fpaws&config=labeled_final&split=validation&offset=0&length=100'
raw=subprocess.check_output(['curl','-fL','--max-time','60',url]);(D/'paws_validation_holdout_raw.json').write_bytes(raw)
rows=[{'id':f'paws-validation:{i}','kind':'positive' if r['row']['label'] else 'adversarial_negative','label':int(r['row']['label']),'left':r['row']['sentence1'],'right':r['row']['sentence2'],'provenance':'PAWS labeled_final validation 0..99; CC BY 4.0; final heldout for this project, not used for calibration'} for i,r in enumerate(json.loads(raw)['rows'])]
assert len(rows)==100
norm=lambda s:' '.join(s.lower().split())
old=[json.loads(l) for l in (R/'benchmarks/data/paws_test.jsonl').read_text().splitlines() if l]
prior={norm(r[k]) for r in old for k in ('sentence1','sentence2')}
prior|={norm(r[k]) for r in json.loads((D/'development.json').read_text()) for k in ('left','right')}
shared={norm(r[k]) for r in rows for k in ('left','right')}&prior
p.write_text(json.dumps(rows,indent=2))
(D/'final_holdout_freeze.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':url,'n_pairs':len(rows),'exact_normalized_sentence_overlap_with_development_and_old_paws':len(shared),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reason_for_replacement':'Initial unscored test slice shared 76 sentence strings with previously evaluated PAWS test rows. No model outputs were inspected for either candidate when making this change.','scope':'Public human-labelled adversarial semantic-equivalence test. Not an externally blinded private test; pretraining/source-article overlap cannot be ruled out.'},indent=2))
assert not shared,'Final holdout must be sentence-disjoint; do not score overlapping candidate.'
print('Final heldout frozen; 100 pairs, zero exact normalized sentence overlap; contents not printed')
