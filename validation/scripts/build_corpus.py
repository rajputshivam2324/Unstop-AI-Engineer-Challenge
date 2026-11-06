"""Independent fixtures. No engine imports, tuning, or production-file writes."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil,re
from fpdf import FPDF
from docx import Document
from docx.shared import Inches,Pt,RGBColor
R=Path(__file__).resolve().parents[1]
C=R/'corpus'; C.mkdir(exist_ok=True)
def ascii_text(s):
 return s.replace('\u2019',"'").replace('\u2018',"'").replace('\u201c','"').replace('\u201d','"').replace('\u2014','-').replace('\u2013','-')
def pdf(name,pages):
 p=FPDF();p.set_auto_page_break(True,18);p.set_margins(20,18,20)
 for text in pages:
  p.add_page();p.set_font('Helvetica',size=11);p.multi_cell(0,6,ascii_text(text))
 p.set_creation_date(datetime.datetime(2026,11,6,tzinfo=datetime.timezone.utc));p.output(str(C/name))
def docx(name,text):
 d=Document();d.core_properties.author='Independent validation fixtures';d.core_properties.created=datetime.datetime(2026,11,6);d.core_properties.modified=datetime.datetime(2026,11,6)
 sec=d.sections[0];sec.top_margin=sec.bottom_margin=Inches(.8)
 style=d.styles['Normal'];style.font.name='Calibri';style.font.size=Pt(11);style.font.color.rgb=RGBColor(0,0,0)
 for para in text.split('\n\n'):d.add_paragraph(para)
 d.save(C/name)
# Exactly 150 whitespace-delimited words. Entire prompt is evaluator-authored synthetic text.
prompt='''Northbridge Public Library asks the university review panel to compare proposals for improving access across three neighbourhood branches. Each candidate should submit an independent recommendation grounded in the fictional facts here. The annual budget is fixed, the buildings remain publicly owned, and the council must publish its reasoning before the next funding meeting. Discuss the proposal assigned to you, identify who benefits, describe an implementation obstacle, and name an observation that could change your decision. Do not invent survey findings or treat an illustrative calculation as measured evidence. Keep any quotations separate from your own analysis and explain uncertainty in language. The panel welcomes disagreement when the argument follows from stated assumptions. Your response should be understandable to residents without specialist training. Include a practical first step, a responsible team, and a review date. This shared institutional prompt is provided for context and should not be assessed as original analysis.'''
assert len(prompt.split())==150,len(prompt.split())
answers=[
'''Extending evening opening should begin at the branch beside the bus interchange. Shift workers can reach that site after the last industrial shift, whereas an extra morning hour mainly serves existing visitors. A four week trial needs a named duty manager and a transport timetable posted at the entrance. Count entries in the additional hour rather than asking everyone whether they support libraries. Stop the extension if staffing costs displace the children's reading sessions. The argument concerns timing of access, not the total number of registered borrowers.''',
'''The first expenditure should replace damaged shelving in the archive room. A box inventory shows what needs safe storage before anyone promises a larger collection. Moving fragile material twice creates avoidable handling work, so the caretaker should plan repairs while the room is closed for its normal cleaning day. Suppliers should quote for removable units that fit the existing door. I would reconsider if the conservation officer found that humidity, rather than furniture, was responsible for the damage. Public evening hours have little bearing on this physical storage problem.''',
'''A mobile return service would help residents who already borrow books but cannot carry them back. Use the existing council van on its quiet route day and let carers reserve a collection by telephone. Staff must separate returned volumes from parcels carrying food. The transport supervisor can log missed pickups and journey distance for six weeks. Low demand would justify replacing the route with collection lockers at the clinic. This proposal addresses the burden of returning heavy books rather than buying more copies.''',
'''Borrower privacy deserves attention before a new recommendation screen is purchased. The catalogue should retain the active loan record and erase a completed transaction when its administrative purpose ends. A resident ought to search for sensitive health information without building a permanent profile at the council. Ask the information officer to document deletion behaviour using a test account. If a statutory retention obligation applies, publish its scope before changing the software. Aggregate circulation totals can support planning without exposing a person's reading history.''',
'''Wheelchair access depends on the route between the pavement and the reading desk. A ramp is insufficient when a heavy inner door cannot be opened without assistance. Invite an access group to inspect that route with the facilities team and rank the barriers by the effort they impose. Start with the door closer and a clear turning space near the entrance. Repeat the visit after installation rather than accepting a supplier photograph as proof. A lift upgrade should wait for an engineering survey of the upper floor.'''
]
speech='The world is very different now, for man holds in his mortal hands the power to abolish all forms of human poverty and all forms of human life.'
# Page 1 is deliberately unrelated to page 2; the known source page is fixed before inference.
pdf('01_human_speech.pdf',['Archive extract for a source location check\n\nThis cover identifies an archival speech excerpt. The recording was delivered by John F Kennedy on 20 January 1961. Page two contains the selected passage. This fixture uses an existing human recording, rather than a synthesized narration. No claim about academic misconduct follows from its intentional reuse in an evaluation.','John F Kennedy inaugural address excerpt\n\n'+speech])
alice=ascii_text((R/'sources/alice.txt').read_text(encoding='utf-8-sig'))
start=alice.index('Alice was beginning');stop=alice.index('There was nothing',start)
alice=' '.join(alice[start:stop].split())
pdf('02_alice_source.pdf',[alice]);pdf('03_alice_copy.pdf',[alice])
pdf('04_library_hours.pdf',[prompt+'\n\n'+answers[0]])
pdf('05_library_storage.pdf',[prompt+'\n\n'+answers[1]])
para_a='''After a city removes library fines, the outstanding debt disappears, so apparent improvement in account balances says nothing about whether books come back sooner. Measure days overdue among loans issued after the change instead. Compare the same calendar weeks with a neighbouring council that keeps its old rules, because school holidays can alter borrowing. Keep replacement charges unchanged during the experiment so the policy contrast is interpretable. Success means shorter delays in returning books without fewer loans to residents who previously owed money.'''
para_b='''Cancelling penalties wipes the ledger clean by definition; that accounting result cannot establish that borrowers have become more punctual. The useful outcome is how late newly borrowed volumes are returned. An adjacent authority retaining penalties provides a comparison over corresponding weeks, helping separate the reform from seasonal school closures. Liability for lost items should stay constant while overdue fees are tested. Support the reform if materials arrive back faster and people formerly carrying debts continue using the service.'''
docx('06_policy_argument.docx',para_a);docx('07_policy_rewording.docx',para_b)
for name,ans in zip(['08_library_transport.docx','09_library_privacy.docx','10_library_access.docx'],answers[2:]):docx(name,prompt+'\n\n'+ans)
# Source preparation alignment uses a DIFFERENT ASR model solely to locate cuts.
# Clip boundaries and source text are fixed here, before running the evaluated tiny model.
clips=[('11_human_crossmodal.mp4',51.0,15.0,speech),('12_human_allies.mp4',180.5,12.0,'To those old allies whose cultural and spiritual origins we share, we pledge the loyalty of faithful friends.'),('13_human_revolution.mp4',66.8,10.5,'And yet the same revolutionary beliefs for which our forebears fought are still at issue around the globe.')]
for name,start,length,text in clips:
 subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=c=navy:s=640x360:r=25','-ss',str(start),'-i',str(R/'sources/jfk.ogg'),'-t',str(length),'-map','0:v','-map','1:a','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac',str(C/name)],check=True)
# Edge cases: valid document plus empty PDF, corrupt PDF and no-audio MP4.
E=R/'edge_corpus';E.mkdir(exist_ok=True);shutil.copyfile(C/'05_library_storage.pdf',E/'05_valid_control.pdf')
(E/'20_empty.pdf').write_bytes(b'');(E/'21_corrupt.pdf').write_bytes(b'%PDF-1.7\nnot a valid xref or object table\n%%EOF')
subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=c=black:s=320x240:r=25','-t','12','-an','-c:v','libx264',str(E/'22_silent.mp4')],check=True)
# Minimal factual reversal is a semantic negative but a textual-reuse positive.
P=R/'probe_corpus';P.mkdir(exist_ok=True)
positive='The blue filter removed arsenic from the river water during the controlled trial. The treatment reduced contamination below the laboratory reporting limit. Investigators recommended deploying the filter in every village before the rainy season.'
reverse='The blue filter did not remove arsenic from the river water during the controlled trial. The treatment did not reduce contamination below the laboratory reporting limit. Investigators did not recommend deploying the filter in every village before the rainy season.'
for name,t in [('30_assertion.pdf',positive),('31_negation.pdf',reverse)]:
 pdf(name,[t]);shutil.move(C/name,P/name)
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':'Labels frozen before engine execution. No thresholds or models modified. Main pair detection: any non-template report passage (engine floor 0.05). Secondary comparison: document-pair score >=0.20 fixed from shipped risk threshold.','positive_pairs':[{'files':['01_human_speech.pdf','11_human_crossmodal.mp4'],'relation':'cross_modal','pdf_page':2,'video_interval_s':[0,15]},{'files':['02_alice_source.pdf','03_alice_copy.pdf'],'relation':'exact_copy'},{'files':['06_policy_argument.docx','07_policy_rewording.docx'],'relation':'paraphrase'}],'negative_pairs':'All remaining unordered pairs in main corpus; authorised shared prompt is excluded from positive labels.','template_files':['04_library_hours.pdf','05_library_storage.pdf','08_library_transport.docx','09_library_privacy.docx','10_library_access.docx'],'template_prompt':prompt,'template_words':150,'clip_sources':[{'file':n,'source':'jfk.ogg','offset_s':s,'duration_s':t,'reference_transcript':v} for n,s,t,v in clips],'provenance':{'01_human_speech.pdf':'Human 1961 speech body; evaluator-authored cover and excerpt heading. Mixed authorship, excluded from document-level AI accuracy.','02_alice_source.pdf':'Human Lewis Carroll public-domain original excerpt.','03_alice_copy.pdf':'Identical human-authored excerpt; counted once for unique-human AI metrics.','04_05_06_07_08_09_10':'AI-authored by this evaluation assistant. Answers 04,05,08,09,10 independently composed with no reused answer passages; 07 intentionally rewords 06. These are synthetic coverage, not real student submissions.','videos':'Genuine human speech, single archival speaker; static colour video is synthetic packaging. No TTS.'},'probe_truth':{'30_assertion.pdf + 31_negation.pdf':{'semantic_equivalent':False,'textual_reuse':True,'reason':'Controlled negation transformation. An overlap flag alone is not an FP for reuse, but a same-meaning paraphrase assertion is wrong.'}},'files':{}}
for folder in [C,E,P]:
 for f in sorted(folder.iterdir()):manifest['files'][str(f.relative_to(R))]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Built',len(list(C.iterdir())),'main submissions;',len(list(E.iterdir())),'edge files;',len(list(P.iterdir())),'semantic probes')
