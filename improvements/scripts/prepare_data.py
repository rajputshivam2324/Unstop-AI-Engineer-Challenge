"""Freeze development fixtures and unseen public PAWS rows before model comparison."""
from pathlib import Path
import json,datetime,hashlib,subprocess
R=Path(__file__).resolve().parents[2];D=R/'improvements/data'
if (D/'freeze.json').exists():raise SystemExit('Data already frozen; refusing replacement')
dev=[
('d01','positive','The coastal clinic replaced paper appointment slips with numbered telephone reservations. Missed visits fell because patients could change their booking without travelling to the reception desk.','A seaside health centre introduced call based scheduling in place of handwritten tickets. Allowing people to rearrange a visit remotely reduced the number who failed to attend.'),
('d02','positive','The experimental bus lane cut journey times during the morning peak. The evaluation compared identical weeks on an untreated route to separate the change from school holidays.','Commuters reached their destinations faster after a dedicated lane opened. Analysts used a route without the intervention over corresponding dates to account for seasonal school closures.'),
('d03','positive','The farmer stored seeds in sealed metal drums to keep out moisture. Germination improved because the contents stayed dry throughout the monsoon.','A grower protected planting grain from damp weather using airtight steel containers. Keeping the grain dry during the rainy season increased the proportion that sprouted.'),
('d04','positive','The museum lets visitors reserve a quiet hour before the main doors open. This arrangement gives people sensitive to noise a chance to see exhibits without a crowd.','An early session at the gallery can be booked by guests who find loud environments difficult. They can inspect the collection peacefully before general admission begins.'),
('d05','positive','The accountant excluded donated equipment from cash expenditure. Counting the gift as a payment would exaggerate the amount the charity spent.','The charity received machinery without paying for it. Its financial reviewer therefore left that donation out of the total money disbursed, preventing an inflated spending figure.'),
('d06','positive','The council repairs leaks before installing larger water tanks. Extra storage cannot solve a shortage when much of the stored supply escapes through broken pipes.','Officials plan to mend damaged mains ahead of expanding reservoir capacity. Building more holding space is ineffective if the water continues to drain away through faults in the network.'),
('d07','same_topic','The clinic uses telephone reservations to let patients rearrange appointments. This reduces missed visits.','The clinic recorded the cost of telephone equipment during its annual budget review. The audit recommended replacing damaged handsets.'),
('d08','same_topic','The museum offers early quiet sessions for people sensitive to noise. Visitors can explore without crowds.','The museum rents its main hall for evening concerts. Ticket sales pay for the restoration of its roof.'),
('d09','same_topic','The farmer keeps seeds dry inside sealed metal drums. This improves germination during the monsoon.','The farmer sells surplus steel drums to a neighbouring workshop. Buyers pay more for containers without dents.'),
('d10','same_topic','The accountant excludes donated machinery from cash spending. This prevents expenditure being overstated.','The accountant values donated machinery for the insurance register. The charity needs enough cover if the workshop catches fire.'),
('d11','contradiction','The new bus lane reduced travel time during the morning rush. The city should keep the lane open.','The new bus lane increased travel time during the morning rush. The city should close the lane.'),
('d12','contradiction','The filter removed arsenic from the water. Its output was safe to drink.','The filter did not remove arsenic from the water. Its output was unsafe to drink.'),
('d13','contradiction','Every patient in the trial recovered after taking the medicine. The researchers recommended its use.','No patient in the trial recovered after taking the medicine. The researchers advised against its use.'),
('d14','contradiction','The retailer acquired the supplier after the merger. The supplier is now owned by the retailer.','The supplier acquired the retailer after the merger. The retailer is now owned by the supplier.'),
('d15','same_topic','The council extended library opening hours to help shift workers. Staff counted visits after sunset.','The council bought books about shift work for the library. Staff placed them on a display beside the entrance.'),
('d16','positive','Charging households by the weight of unrecycled rubbish encourages sorting. A fixed monthly fee offers no financial reward for reducing waste.','Residents save money by separating recyclables when disposal bills depend on the mass of residual trash. A flat subscription lacks that incentive.'),
('d17','positive','The school delayed the examination until the flooded bridge reopened. Students on the far bank otherwise could not reach the classroom.','Pupils across the river were unable to attend while floodwater closed the crossing. Their test was postponed until access was restored.'),
('d18','same_topic','The school postponed an examination because the bridge was flooded. Pupils could not get to the classroom.','The school used an examination question about bridge construction. Pupils calculated how a flood would load the piers.'),
]
# Append the confirmed failure and existing meaning-reversal mechanism as DEVELOPMENT only.
e=json.loads((R/'validation/extraction.json').read_text())
dev.append(('confirmed_docx','positive',e['06_policy_argument.docx']['full_text'],e['07_policy_rewording.docx']['full_text']))
rows=[{'id':i,'kind':k,'label':int(k=='positive'),'left':a,'right':b,'provenance':'AI-authored controlled development fixture; not human submissions'} for i,k,a,b in dev]
(D/'development.json').write_text(json.dumps(rows,indent=2))
# Download a contiguous test slice not used by the earlier PAWS benchmark (which used rows 0..1999).
url='https://datasets-server.huggingface.co/rows?dataset=google-research-datasets%2Fpaws&config=labeled_final&split=test&offset=2000&length=100'
raw=subprocess.check_output(['curl','-fL','--max-time','60',url]);(D/'paws_holdout_raw.json').write_bytes(raw)
payload=json.loads(raw);blind=[{'id':f'paws:{2000+i}','kind':'positive' if r['row']['label'] else 'adversarial_negative','label':int(r['row']['label']),'left':r['row']['sentence1'],'right':r['row']['sentence2'],'provenance':'PAWS labeled_final test rows 2000..2099; CC BY 4.0; previously unused by this repo benchmark'} for i,r in enumerate(payload['rows'])]
assert len(blind)==100
(D/'heldout.json').write_text(json.dumps(blind,indent=2))
(D/'freeze.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rule':'Development may select method and thresholds. Heldout is not scored or inspected until method/config/code are frozen; no changes based on heldout outcomes. Not blinded to an external evaluator; publicly labelled heldout test. Original validation corpus is development/regression.','public_source':url,'public_license':'https://github.com/google-research-datasets/paws/blob/master/LICENSE','hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file()}},indent=2))
print('Frozen',len(rows),'development pairs and',len(blind),'public heldout pairs; heldout content not printed')
