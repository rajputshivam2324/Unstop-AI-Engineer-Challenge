"""Render one static benchmark image directly from preserved measurements."""
from pathlib import Path
import json, math, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODES = ['baseline', 'boundary', 'semantic_sts', 'semantic_nli', 'semantic_hybrid']
DATA = {m: json.loads((ROOT / f'improvements/runs/heldout_{m}_final.json').read_text())
        for m in MODES}
FULL = json.loads((ROOT / 'improvements/comparison.json').read_text())
for p in ['/usr/share/fonts/open-sans/OpenSans-Regular.ttf',
          '/usr/share/fonts/open-sans/OpenSans-Semibold.ttf',
          '/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-Bold.otf']:
    if Path(p).exists(): font_manager.fontManager.addfont(p)
plt.rcParams.update({'font.family': ['Open Sans', 'DejaVu Sans'], 'font.size': 11,
                     'axes.titlesize': 15, 'axes.titleweight': 'semibold',
                     'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 11,
                     'text.color': '#1B2D45', 'axes.labelcolor': '#637287',
                     'xtick.color': '#637287', 'ytick.color': '#1B2D45',
                     'axes.edgecolor': '#DCE3EB', 'savefig.facecolor': '#F7F9FC'})
NAVY, TEAL, MUTED, GRID = '#354C70', '#007F80', '#637287', '#E1E7EF'
ORANGE, RED, LIGHTBLUE = '#DE7640', '#C45460', '#87B9D4'
fig = plt.figure(figsize=(16, 12), facecolor='#F7F9FC')
fig.text(.055, .958, 'MULTIMODAL INTEGRITY ENGINE', fontfamily='Montserrat',
         fontsize=27, fontweight='bold')
fig.text(.055, .924, 'BENCHMARK IMAGE  /  Baseline vs semantic hybrid', fontsize=13, color=MUTED)
fig.text(.945, .956, 'FROZEN MEASUREMENTS', ha='right', fontsize=10, color=TEAL, weight='semibold')
fig.text(.945, .931, 'CPU  ·  06 NOV 2026  ·  7c07ab7', ha='right', fontsize=9, color=MUTED)
fig.add_artist(plt.Line2D([.055,.945],[.900,.900],transform=fig.transFigure,color=GRID,lw=1))
# Header measurements all refer to the held-out semantic task, never the regression set.
b, h = DATA['baseline']['metrics'], DATA['semantic_hybrid']['metrics']
fig.text(.055, .875, 'HELD-OUT SEMANTIC TEST', fontsize=10, color=MUTED, weight='semibold')
fig.text(.055, .840, f"F1  {b['f1']*100:.1f}%  to  {h['f1']*100:.1f}%", fontsize=26, weight='semibold')
fig.text(.055, .812, f"+{(h['f1']-b['f1'])*100:.1f} percentage points  ·  100 human-labelled pairs", fontsize=10, color=MUTED)
for xpos, label, value in [(.535,'HYBRID PRECISION',h['precision']),(.750,'HYBRID RECALL',h['recall'])]:
    fig.text(xpos,.873,label,fontsize=10,color=MUTED,weight='semibold')
    fig.text(xpos,.835,f'{100*value:.1f}%',fontsize=29,color=TEAL,weight='semibold')
    old = b['precision'] if 'PRECISION' in label else b['recall']
    fig.text(xpos,.811,f'Baseline {100*old:.1f}%',fontsize=10,color=MUTED)


def axes(rect, title, subtitle, heading_offset=0):
    ax=fig.add_axes(rect,facecolor='none')
    ax.set_title(title,loc='left',x=heading_offset,pad=34)
    ax.text(heading_offset,1.09,subtitle,transform=ax.transAxes,fontsize=9,color=MUTED)
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='both',length=0,pad=7)
    return ax

# 1. Five methods; preserve the weak STS result alongside the selected method.
ax=axes([.135,.532,.200,.205],'F1 by matching method','Same 100 held-out PAWS pairs',heading_offset=-.40)
labels=['Baseline','Boundary only','STS verifier','NLI verifier','Semantic hybrid']
vals=[100*DATA[m]['metrics']['f1'] for m in MODES]
colors=[NAVY,'#A8B6CA','#A8B6CA','#6F9CA7',TEAL]
y=np.arange(5)
ax.barh(y,vals,height=.58,color=colors)
for yi,v in zip(y,vals):ax.text(v+1.7,yi,f'{v:.1f}%',va='center',fontsize=10,weight='semibold')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,100)
ax.set_xticks([0,25,50,75,100]);ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
ax.xaxis.grid(True,color=GRID,lw=.7);ax.set_axisbelow(True)

# 2. Pie/donut accounting for every case, not just successful detections.
ax=fig.add_axes([.390,.542,.240,.185],facecolor='none')
ax.set_title('Every held-out outcome',loc='center',pad=38,fontsize=15,weight='semibold')
ax.text(.5,1.13,'Hybrid  ·  47 positive / 53 negative labels',ha='center',transform=ax.transAxes,fontsize=9,color=MUTED)
counts=[h['tp'],h['tn'],h['fp'],h['fn']]
wedges,_=ax.pie(counts,startangle=90,counterclock=False,colors=[TEAL,LIGHTBLUE,RED,ORANGE],
               wedgeprops={'width':.31,'edgecolor':'#F7F9FC','linewidth':2})
ax.text(0,.04,'100',ha='center',va='center',fontsize=31,weight='semibold')
ax.text(0,-.25,'pairs',ha='center',fontsize=10,color=MUTED)
ax.legend(wedges,[f'True positives  {h["tp"]}',f'True negatives  {h["tn"]}',
                  f'False positives  {h["fp"]}',f'False negatives  {h["fn"]}'],
          loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False,
          handlelength=.8,handletextpad=.5,columnspacing=1.4,fontsize=9)

# 3. Waterfall: follow actual case IDs to distinguish removed and new mistakes.
bfp={r['id'] for r in DATA['baseline']['cases'] if r['predicted'] and not r['label']}
hfp={r['id'] for r in DATA['semantic_hybrid']['cases'] if r['predicted'] and not r['label']}
removed,added=len(bfp-hfp),len(hfp-bfp)
assert len(bfp)-removed+added==len(hfp)
ax=axes([.710,.532,.235,.205],'False-positive waterfall','Semantic negatives  ·  case IDs tracked')
bottoms=[0,len(bfp)-removed,len(bfp)-removed,0]
heights=[len(bfp),removed,added,len(hfp)]
ax.bar(range(4),heights,bottom=bottoms,width=.58,color=[NAVY,TEAL,ORANGE,TEAL])
for i,level in enumerate([len(bfp),len(bfp)-removed,len(hfp)]):
    ax.plot([i+.29,i+.71],[level,level],color='#9AA9BA',lw=1,ls=(0,(3,3)))
for i,v in enumerate([str(len(bfp)),f'−{removed}',f'+{added}',str(len(hfp))]):
    top=bottoms[i]+heights[i]
    ax.text(i,top+2.3,v,ha='center',va='bottom',fontsize=12,weight='semibold')
ax.set_xticks(range(4),['Baseline','Removed','New','Hybrid']);ax.set_ylim(0,60)
ax.set_yticks([0,20,40,60]);ax.yaxis.grid(True,color=GRID,lw=.7);ax.set_axisbelow(True)
ax.text(.5,-.26,f'{len(bfp)-len(hfp)} fewer false positives; {added} new errors remain',ha='center',transform=ax.transAxes,fontsize=9,color=MUTED)

# 4. True per-case histograms, common bins, all samples retained.
ax=axes([.055,.230,.280,.185],'Latency distribution','Held-out worker  ·  100 pair timings per method')
bt=np.array([r['seconds'] for r in DATA['baseline']['cases']])
ht=np.array([r['seconds'] for r in DATA['semantic_hybrid']['cases']])
end=math.ceil(max(bt.max(),ht.max())*10)/10
bins=np.linspace(0,end,18)
ax.hist(bt,bins=bins,color=NAVY,alpha=.78,label='Baseline',edgecolor='none')
ax.hist(ht,bins=bins,color=TEAL,alpha=.64,label='Hybrid',edgecolor='none')
for v,c in [(np.median(bt),NAVY),(np.median(ht),TEAL)]:ax.axvline(v,color=c,lw=1.5,ls='--')
ax.set_xlabel('Seconds per pair');ax.set_ylabel('Pairs');ax.set_xlim(0,end)
ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=4))
ax.yaxis.grid(True,color=GRID,lw=.7);ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=9,loc='upper right')
ax.text(0,-.36,f'Median: {np.median(bt):.3f}s to {np.median(ht):.3f}s',transform=ax.transAxes,fontsize=10,color=MUTED)

# 5. Memory measured in the same held-out workers, separate from the full AI pipeline.
ax=axes([.405,.230,.225,.185],'Verification memory','Held-out worker peak RSS  ·  MiB')
mem=[DATA['baseline']['peak_rss_mb'],DATA['semantic_hybrid']['peak_rss_mb']]
ax.barh([1,0],mem,color=[NAVY,TEAL],height=.42)
for y,v in zip([1,0],mem):ax.text(v+25,y,f'{v:,.0f}',va='center',fontsize=11,weight='semibold')
ax.set_yticks([1,0],['Baseline','Hybrid']);ax.set_xlim(0,1350);ax.set_ylim(-.6,1.6)
ax.set_xticks([0,500,1000]);ax.xaxis.grid(True,color=GRID,lw=.7);ax.set_axisbelow(True)
ax.text(0,-.36,'Worker totals: 11.37s to 40.36s\nInitial model loading excluded',transform=ax.transAxes,fontsize=9,color=MUTED,linespacing=1.5)

# 6. Clearly distinguish repair/regression results from held-out generalization.
ax=axes([.710,.230,.235,.185],'Known regression repair','F1  ·  13 mixed files  ·  78 document pairs')
p=FULL['pipeline_metrics']
reg=[100*p['baseline']['f1'],100*p['semantic_hybrid']['f1']]
ax.barh([1,0],reg,color=[NAVY,TEAL],height=.42)
for y,v in zip([1,0],reg):ax.text(v-3,y,f'{v:.0f}%',va='center',ha='right',color='white',fontsize=11,weight='semibold')
ax.set_yticks([1,0],['Baseline','Hybrid']);ax.set_xlim(0,100);ax.set_ylim(-.6,1.6)
ax.set_xticks([0,50,100]);ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
ax.xaxis.grid(True,color=GRID,lw=.7);ax.set_axisbelow(True)
ax.text(0,-.36,f'Full CLI: {p["baseline"]["runtime"]["elapsed_s"]:.2f}s to {p["semantic_hybrid"]["runtime"]["elapsed_s"]:.2f}s\nBoth confirmed misses repaired; no FP pairs',transform=ax.transAxes,fontsize=9,color=MUTED,linespacing=1.5)

fig.add_artist(plt.Line2D([.055,.945],[.115,.115],transform=fig.transFigure,color=GRID,lw=1))
fig.text(.055,.087,'TEST SCOPE',fontsize=9,weight='semibold',color=TEAL)
fig.text(.150,.087,'Held-out semantic equivalence and known regression repair are separate evaluations.',fontsize=10,color=MUTED)
fig.text(.055,.060,'Small, purpose-selected samples. Semantic false positives are not misconduct findings. Baseline remains the default.',fontsize=10,color=MUTED)
fig.text(.055,.032,'SOURCE  improvements/runs/heldout_*_final.json  +  improvements/comparison.json  |  All charts use recorded measurements.',fontsize=8.5,color=MUTED)

png=OUT/'engine_benchmark.png'
fig.savefig(png,dpi=240,facecolor=fig.get_facecolor())
plt.close(fig)
paths=[ROOT/f'improvements/runs/heldout_{m}_final.json' for m in MODES]+[ROOT/'improvements/comparison.json']
(OUT/'image_provenance.json').write_text(json.dumps({
    'image':png.name,'dimensions_px':[3840,2880],
    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'matplotlib':matplotlib.__version__,
    'false_positive_waterfall':{'baseline':len(bfp),'removed':removed,'new':added,'hybrid':len(hfp)},
    'heldout_hybrid_metrics':h,
},indent=2))
print(png)
