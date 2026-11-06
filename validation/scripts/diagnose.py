"""Read-only stage diagnostics on frozen fixtures; no threshold changes."""
from pathlib import Path
import sys,os,json,time,dataclasses,re,subprocess
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from integrity_engine.config import load_config
from integrity_engine.ingestion import extract_any
from integrity_engine.ingestion.mp4 import release_asr
from integrity_engine.text_processing import build_chunks,token_count
from integrity_engine.alignment import TemplateRegistry,DocumentAligner,token_jaccard
from integrity_engine.embeddings import build_embedder
cfg=load_config(R/'config/default.toml');V=R/'validation';docs=[];extracted={}
for folder in ['corpus','probe_corpus']:
 for p in sorted((V/folder).iterdir()):
  t=time.monotonic();d=extract_any(p,cfg);docs.append(d)
  extracted[d.document_id]={'full_text':d.full_text,'pages':[dataclasses.asdict(x) for x in d.pages],'metadata':d.metadata,'status':d.status,'warnings':d.warnings,'diagnostic_extract_seconds':time.monotonic()-t}
release_asr();(V/'extraction.json').write_text(json.dumps(extracted,indent=2))
main=[d for d in docs if d.path.parent.name=='corpus'];chunks=[c for d in main for c in build_chunks(d,cfg)]
reg=TemplateRegistry(cfg.alignment.template_min_docs,cfg.alignment.template_hamming_max,cfg.alignment.template_min_tokens,cfg.alignment.verbatim_template_min_tokens);reg.register_all_chunks(chunks)
emb=build_embedder(cfg.embedding,str(R/'cache/hf'));al=DocumentAligner(emb,reg,cfg,chunks);by={d.document_id:d for d in docs}
x={}
for a,b in [('01_human_speech.pdf','11_human_crossmodal.mp4'),('06_policy_argument.docx','07_policy_rewording.docx')]:
 sa,va=al._doc_sentences(by[a]);sb,vb=al._doc_sentences(by[b]);cos=va@vb.T
 runs,rejected=al.accepted_runs(by[a],by[b]);whole=emb.encode_corpus([by[a].full_text,by[b].full_text])
 x[a+' + '+b]={'left_sentences':[s.text for s in sa],'right_sentences':[s.text for s in sb],'cosine_matrix':cos.tolist(),'accepted_runs':runs,'rejected_runs':rejected,'raw_document_cosine':float(whole[0]@whole[1]),'document_jaccard':token_jaccard(by[a].full_text,by[b].full_text),'forward_coverage':al.coverage(by[a],by[b]),'reverse_coverage':al.coverage(by[b],by[a])}
# Component probes are explicitly exploratory and excluded from main-corpus metrics.
from integrity_engine.types import ProcessedDocument,Page
borrowed='The council should reserve the abandoned railway cutting as a flood basin. Storm water would enter through the old maintenance tunnel and spread across the gravel bed. A raised footpath would let residents cross the site without stepping into the overflow channel.'
ends=['A surveyor must inspect the northern abutment.','The park manager will count nesting birds.','An electrician should isolate the platform lights.']
trio=[ProcessedDocument(f'4{i}_borrowed.docx',Path(f'4{i}_borrowed.docx'),'docx','ok',pages=[Page(None,borrowed+' '+e,0,len(borrowed+' '+e))],full_text=borrowed+' '+e,sha256=str(i)) for i,e in enumerate(ends)]
cc=[c for d in trio for c in build_chunks(d,cfg)];rr=TemplateRegistry(cfg.alignment.template_min_docs,cfg.alignment.template_hamming_max,cfg.alignment.template_min_tokens,cfg.alignment.verbatim_template_min_tokens);rr.register_all_chunks(cc);aa=DocumentAligner(emb,rr,cfg,cc)
x['exploratory_three_copy']={'expected':'all three unordered pairs share deliberately borrowed distinctive analysis; not an authorised prompt','documents':{d.document_id:d.full_text for d in trio},'template_sentence_count':len(rr.template_sentences),'template_chunks':sum(c.is_template for c in cc),'pair_coverages':{a.document_id+' + '+b.document_id:[aa.coverage(a,b),aa.coverage(b,a)] for i,a in enumerate(trio) for b in trio[i+1:]}}
from integrity_engine.external_check import LocalReferenceKB,ddg_query
from reference_kb.kb_texts import all_passages
kb=LocalReferenceKB(emb,cfg,skip_web=True);kb.build();kid,text=all_passages()[0];d=ProcessedDocument('local_control',Path('local_control'),'docx','ok',pages=[Page(None,text,0,len(text))],full_text=text)
x['local_kb_positive_control']={'expected_source':kid,'query':text,'result':dataclasses.asdict(kb.search_doc(d))}
# Deterministic failure transport probe, evaluation-only replacement of transport function.
import integrity_engine.external_check as ext
saved=ext.ddg_query
try:
 ext.ddg_query=lambda *args:None
 hits,mock=kb._web_tier(['A distinctive test sentence with enough words to exercise the disclosed fallback mechanism.'])
 x['web_failure_control']={'hit_count':len(hits),'mock_engaged':mock}
finally:ext.ddg_query=saved
(V/'diagnostics.json').write_text(json.dumps(x,indent=2));emb.release()
print(json.dumps(x,indent=2))
