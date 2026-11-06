import os,json,subprocess
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
from faster_whisper import WhisperModel
root=Path(__file__).resolve().parents[1]
subprocess.run(['ffmpeg','-y','-v','error','-i',str(root/'sources/jfk.ogg'),'-t','210','-ar','16000','-ac','1',str(root/'sources/jfk_first210.wav')],check=True)
m=WhisperModel('base',device='cpu',compute_type='int8',download_root=str(root.parent/'cache/hf'))
segments,_=m.transcribe(str(root/'sources/jfk_first210.wav'),language='en',word_timestamps=True)
x=[{'start':s.start,'end':s.end,'text':s.text,'words':[{'start':w.start,'end':w.end,'word':w.word} for w in s.words]} for s in segments]
(root/'sources/preparation_alignment.json').write_text(json.dumps(x,indent=2))
for s in x:print(s['start'],s['end'],s['text'])
