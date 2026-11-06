"""Read-only scoring of new outputs and preservation checks; prior results untouched."""
from pathlib import Path
import json,itertools,hashlib,datetime
R=Path(__file__).resolve().parents[2];I=R/'improvements'
truth={tuple(sorted(p['files'])):p['relation'] for p in json.loads((R/'validation/manifest.json').read_text())['positive_pairs']}
metrics={};details={}
for mode in ('baseline','boundary','semantic_nli','semantic_hybrid'):
 root=I/f'runs/pipeline_{mode}';reports={}
 if not (root/'summary/latest.json').exists():continue
 for p in (root/'per_doc').glob('*.json'):
  row=json.loads(p.read_text());reports[row['document_id']]=row
 predictions={};issues=[]
 for name,r in reports.items():
  for p in r['peer_matches']+r['multimodal_matches']:
   if p['match_type']=='common_source':continue
   key=tuple(sorted((name,p['matched_file'])));predictions.setdefault(key,[]).append({'query_file':name,**p})
   if name==p['matched_file']:issues.append('self match')
   if 'verified candidate' in (p['note'] or '') and p['evidence_sentence_scores']:issues.append('window reported as sentence scores')
 tp=len(set(predictions)&set(truth));fp=len(set(predictions)-set(truth));fn=len(set(truth)-set(predictions));tn=78-tp-fp-fn
 precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn);f1=2*tp/(2*tp+fp+fn)
 summary=json.loads((root/'summary/latest.json').read_text())
 metrics[mode]={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':f1,'runtime':summary['runtime'],'output_count_json':len(reports),'output_count_md':len(list((root/'per_doc').glob('*.md'))),'schema_issues':issues}
 details[mode]={'positive_pairs':[{'pair':list(k),'expected':v,'detected':k in predictions,'evidence':predictions.get(k,[])} for k,v in truth.items()],'false_positives':[{'pair':list(k),'evidence':predictions[k]} for k in predictions if k not in truth],'per_file':[{'file':n,'risk':r['total_integrity_risk'],'scores':r['scores'],'warnings':r['warnings']} for n,r in reports.items()]}
# Compare baseline report values to original frozen baseline, ignoring timestamps.
baseline_mismatches=[]
for p in (I/'runs/pipeline_baseline/per_doc').glob('*.json'):
 old=json.loads((R/'validation/runs/main/per_doc'/p.name).read_text());new=json.loads(p.read_text())
 old.pop('generated_at_utc',None);new.pop('generated_at_utc',None)
 if old!=new:baseline_mismatches.append(p.name)
prior=json.loads((I/'prior_results_freeze.json').read_text());changed=[p for p,h in prior['sha256'].items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
freeze=json.loads((I/'final_method_freeze.json').read_text());code_changed=[p for p,h in freeze['sha256'].items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pipeline_metrics':metrics,'pipeline_details':details,'baseline_substantive_report_mismatches':baseline_mismatches,'changed_prior_result_files':changed,'changed_final_method_files':code_changed}
(I/'comparison.json').write_text(json.dumps(x,indent=2));print(json.dumps({k:v for k,v in x.items() if k!='pipeline_details'},indent=2));assert not changed and not code_changed
