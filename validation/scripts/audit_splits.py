import sys,json,random,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'benchmarks'))
import bench_ai_detection as a,bench_retrieval as b
cal,test=a.split(a.load_pairs(50,42),42)
canon=lambda t:' '.join(re.findall(r'\w+',t.lower()))
ct={canon(p[k]) for p in cal for k in ('human','ai')};tt={canon(p[k]) for p in test for k in ('human','ai')}
# Exact same construction sequence, without invoking models or modifying benchmark outputs.
rng=random.Random(42);answers=b.human_answers();subsets=sorted(answers)
for s in subsets:rng.shuffle(answers[s])
segments=[];cursor={s:0 for s in subsets};i=0
while len(segments)<b.N_DOCS:
 s=subsets[i%len(subsets)];i+=1
 if cursor[s]+3<=len(answers[s]):segments.append(answers[s][cursor[s]:cursor[s]+3]);cursor[s]+=3
order=list(range(b.N_DOCS));rng.shuffle(order);truth={};pos=0
for _ in range(b.N_COPY):
 x,y=order[pos:pos+2];pos+=2;segments[y][1]=segments[x][1];truth[frozenset((x,y))]='exact_copy'
for p1,p2 in b.mrpc_passages(b.N_PARA,rng):
 x,y=order[pos:pos+2];pos+=2;segments[x][1]=p1;segments[y][1]=p2;truth[frozenset((x,y))]='paraphrase'
used={canon(s) for g in segments for s in g}
# Preserve exact qids and fingerprints for traceability, no private student data.
x={'ai_calibration_questions':len(cal),'ai_test_questions':len(test),'question_id_overlap':sorted({p['qid'] for p in cal}&{p['qid'] for p in test}),'exact_normalized_text_overlap':len(ct&tt),'retrieval_segments_matching_calibration_text':len(used&ct),'retrieval_segments_matching_ai_test_text':len(used&tt),'retrieval_positive_pairs':len(truth),'retrieval_all_pairs':b.N_DOCS*(b.N_DOCS-1)//2,'retrieval_negative_pairs':b.N_DOCS*(b.N_DOCS-1)//2-len(truth),'mrpc_retrieval_source':'mrpc_test.jsonl (same split used for sentence-level paraphrase benchmark)','ai_split_qids':{'calibration':[p['qid'] for p in cal],'test':[p['qid'] for p in test]},'data_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'benchmarks/data').glob('*.jsonl')}}
(R/'validation/split_audit.json').write_text(json.dumps(x,indent=2));print({k:v for k,v in x.items() if k not in ('ai_split_qids','data_sha256')})
