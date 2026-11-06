"""Independent report scorer, keeping every main pair including misses."""
from pathlib import Path
import json,re,itertools,math,hashlib,datetime,subprocess
V=Path(__file__).resolve().parents[1];R=V.parent
m=json.loads((V/'manifest.json').read_text());E=json.loads((V/'extraction.json').read_text())
reports={p.name[:-5]:json.loads(p.read_text()) for p in (V/'runs/main/per_doc').glob('*.json')}
truth={tuple(sorted(x['files'])):x['relation'] for x in m['positive_pairs']}
ids=sorted(reports);rows=[];s=json.loads((V/'runs/main/summary/latest.json').read_text());matrix=s['peer_similarity_matrix']
for a,b in itertools.combinations(ids,2):
 passages=[]
 for q,t in [(a,b),(b,a)]:
  passages.extend({'query_file':q,**x} for x in reports[q]['peer_matches']+reports[q]['multimodal_matches'] if x['matched_file']==t and x['match_type']=='overlap')
 score=max(matrix[a].get(b,0),matrix[b].get(a,0));expected=(a,b) in truth
 pred=bool(passages)
 rows.append({'files':[a,b],'expected_relationship':truth.get((a,b),'negative'),'detected_relationships':sorted(set(p['overlap_kind'] for p in passages)),'doc_pair_score':score,'passage_cosines':[p['similarity_score'] for p in passages],'predicted_any_reported_evidence':pred,'predicted_at_0_20':score>=.2,'status':('TP' if pred else 'FN') if expected else ('FP' if pred else 'TN'),'risks':[reports[a]['total_integrity_risk'],reports[b]['total_integrity_risk']],'evidence':passages})
def prf(rows,field):
 tp=sum(r[field] and r['expected_relationship']!='negative' for r in rows);fp=sum(r[field] and r['expected_relationship']=='negative' for r in rows);fn=sum(not r[field] and r['expected_relationship']!='negative' for r in rows);tn=len(rows)-tp-fp-fn
 return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
def wilson(k,n):
 z=1.96;d=1+z*z/n;c=(k/n+z*z/(2*n))/d;w=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/d
 return [max(0,c-w),min(1,c+w)]
met=prf(rows,'predicted_any_reported_evidence');met['recall_wilson95']=wilson(met['tp'],met['tp']+met['fn']);met['precision_wilson95']=wilson(met['tp'],met['tp']+met['fp'])
# Independent AUC via pairwise ranks; ties get half credit. No external dependency.
ps=[r['doc_pair_score'] for r in rows if r['expected_relationship']!='negative'];ns=[r['doc_pair_score'] for r in rows if r['expected_relationship']=='negative']
auc=sum((a>b)+.5*(a==b) for a in ps for b in ns)/(len(ps)*len(ns))
# AI truth: seven known AI-written files, one unique unmixed human work.
ai_ids=['02_alice_source.pdf','04_library_hours.pdf','05_library_storage.pdf','06_policy_argument.docx','07_policy_rewording.docx','08_library_transport.docx','09_library_privacy.docx','10_library_access.docx']
ai={'tp':sum(reports[i]['ai_analysis']['label']=='AI_LIKELY' for i in ai_ids[1:]),'fn':sum(reports[i]['ai_analysis']['label']!='AI_LIKELY' for i in ai_ids[1:]),'fp':int(reports[ai_ids[0]]['ai_analysis']['label']=='AI_LIKELY'),'tn':int(reports[ai_ids[0]]['ai_analysis']['label']!='AI_LIKELY'),'warning':'7 synthetic positives, 1 unique pure-human negative; human speech PDF mixes synthetic cover and is excluded, copy is not an additional independent human. No population accuracy claim.'}
from importlib.machinery import SourceFileLoader
asr=SourceFileLoader('validation_bench_asr',str(R/'benchmarks/bench_asr.py')).load_module()
wer=[]
for c in m['clip_sources']:
 d=E[c['file']];duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(V/'corpus'/c['file'])],text=True))
 wer.append({'file':c['file'],'duration_s':duration,'reference':c['reference_transcript'],'hypothesis':d['full_text'],'wer':asr.wer(c['reference_transcript'],d['full_text']),'segments':d['metadata']['segments'],'reference_interval_s':[0,duration]})
checks={'report_count_json':len(reports),'report_count_md':len(list((V/'runs/main/per_doc').glob('*.md'))),'matrix_rows':len(matrix),'matrix_offdiagonal_entries':sum(len(r) for r in matrix.values()),'self_matches':sum(p['matched_file']==i for i,r in reports.items() for p in r['peer_matches']+r['multimodal_matches']),'schema_ok':all(list(r)[:6]==['document_id','total_integrity_risk','scores','peer_matches','multimodal_matches','explanations'] for r in reports.values()),'textual_evidence_checks':[]}
for row in rows:
 for p in row['evidence']:
  q=p['query_file'];t=p['matched_file']
  checks['textual_evidence_checks'].append({'query':q,'source':t,'query_span_exact':p['query_passage'] in E[q]['full_text'],'source_span_exact':p['matched_passage'] in E[t]['full_text'],'query_page_correct':any(pg['page']==p['location'].get('query_page') and p['query_passage'] in pg['text'] for pg in E[q]['pages']) if q.endswith('pdf') else None})
perfile=[];log=(V/'runs/main/stderr.txt').read_text()
for i,r in sorted(reports.items()):
 mt=re.search(r'ingested '+re.escape(i)+r'\s+status=\w+\s+chars=\s*\d+ \(([0-9.]+)s\)',log)
 perfile.append({'file':i,'status':r['status'],'peer_pct':r['scores']['peer_similarity_percentage'],'external_pct':r['scores']['external_similarity_percentage'],'ai_pct':r['scores']['ai_generated_probability'],'ai_label':r['ai_analysis']['label'],'risk':r['total_integrity_risk'],'warnings':r['warnings'],'ingestion_s_from_pipeline_log':float(mt.group(1)) if mt else None})
output={'n_files':len(ids),'n_pairs':len(rows),'metrics_any_reported_evidence':met,'metrics_score_at_0_20':prf(rows,'predicted_at_0_20'),'document_pair_auc':auc,'ai_exploratory':ai,'asr':wer,'checks':checks,'per_file':perfile,'pairs':rows}
(V/'results.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k not in ('pairs','per_file')},indent=2))
# Reject changed frozen production files, old reports, old benchmark outputs, input fixtures.
f=json.loads((V/'freeze.json').read_text());changed=[p for p,h in f['hashes'].items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
changed_fixtures=[p for p,h in m['files'].items() if hashlib.sha256((V/p).read_bytes()).hexdigest()!=h['sha256']]
(V/'preservation_check.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_files_checked':len(f['hashes']),'changed_original_files':changed,'changed_fixtures':changed_fixtures},indent=2));assert not changed and not changed_fixtures
