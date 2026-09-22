import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]/'results/lerobot_pipeline_ab'
s=json.loads((OUT/'summary.json').read_text());names=['Serial','Decode / upload pipeline'];colors=['#718da8','#288e82'];modes=['serial','pipeline']
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,field,title,ylabel in [(axes[0],'wall_s','Same 2048 videos: completion time','Wall time (minutes)'),(axes[1],'video_hours_per_machine_day','8-GPU throughput projection','Source video hours / day')]:
 vals=[s['phases'][m][field]/(60 if field=='wall_s' else 1) for m in modes];bars=ax.bar(names,vals,color=colors,width=.55)
 for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val,f'{val:.2f}',ha='center',va='bottom')
 ax.set(title=title,ylabel=ylabel,ylim=(0,max(vals)*1.18));ax.grid(axis='y',alpha=.2)
fig.suptitle(f"Omega 4fps / window 240 / overlap 36: {s['speedup']:.2f}x measured speedup\nSame models and BOS readback checks; startup excluded, fill/drain included",fontsize=11)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180)
