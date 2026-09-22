import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/omega_window_study'
def main():
 d=json.loads((OUT/'summary.json').read_text());assert d['complete'];rows=d['summary'];chosen=d['selection'];fig,axes=plt.subplots(2,2,figsize=(11,8))
 for ax,(key,title,ylabel) in zip(axes.flat,[('ate_cm','Dense trajectory accuracy','ATE Sim(3) RMSE (cm)'),('rpe_rot_deg','Rotation at 30fps output','1s RPE rotation (degrees)'),('anchor10_tail15_cm','Tail error after initial 10s alignment','Tail 15s RMSE (cm)'),('inference_s_per_video_minute','Inference cost','Seconds per video minute')]):
  for fps,color in [(2,'#5985b7'),(4,'#d08328'),(8,'#148572')]:
   rs=sorted([r for r in rows if r['fps']==fps],key=lambda r:r['overlap']);ax.plot([r['overlap'] for r in rs],[r[key] for r in rs],'o-',label=f'{fps} fps',color=color)
  ax.scatter(chosen['overlap'],chosen[key],marker='*',s=230,color='black',zorder=5,label='Selected');ax.set(title=title,xlabel='Overlap (frames)',ylabel=ylabel,xticks=[24,36,48]);ax.grid(alpha=.2)
 axes[0,0].legend();fig.suptitle('VGGT-Omega: 20 full HOT3D videos, window = 240 frames');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'configuration_comparison.png',dpi=180);plt.close(fig)
 # Same-sequence comparison makes overlap effects assessable without mixing datasets.
 allrows=[json.loads(p.read_text()) for p in (OUT/'predictions').glob('*/*.json')];fps=chosen['fps'];ids=sorted(set(r['id'] for r in allrows));fig,axes=plt.subplots(1,2,figsize=(13,5));colors=['#5985b7','#d08328','#148572']
 for ov,color in zip([24,36,48],colors):
  rs={r['id']:r for r in allrows if r['fps']==fps and r['overlap']==ov};x=np.arange(len(ids));axes[0].plot(x,[rs[i]['ate_sim3_rmse_m']*100 for i in ids],'o-',color=color,label=f'overlap {ov}',markersize=4);axes[1].plot(x,[rs[i]['anchor10_tail15_rmse_m']*100 for i in ids],'o-',color=color,label=f'overlap {ov}',markersize=4)
 for ax,title in zip(axes,['ATE Sim(3) RMSE (cm)','Tail 15s RMSE after first 10s alignment (cm)']):
  ax.set(title=title,xticks=np.arange(len(ids)),xticklabels=[i.split('_')[-1] for i in ids]);ax.tick_params(axis='x',rotation=75,labelsize=8);ax.grid(alpha=.2)
 axes[0].legend();fig.suptitle(f'Paired overlap comparison at {fps} fps');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'paired_overlap.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
