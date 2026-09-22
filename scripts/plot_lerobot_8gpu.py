import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]/'results/lerobot_8gpu_4fps_ov36'
rows=[json.loads(l) for l in (OUT/'monitor.jsonl').read_text().splitlines()]
s=json.loads((OUT/'summary.json').read_text());t=np.array([r['elapsed_s'] for r in rows]);v=np.array([r['video_s'] for r in rows]);fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(t/60,v/3600);ax[0].set(xlabel='Wall time (minutes)',ylabel='Completed source video (hours)',title='8 resident workers, including BOS roundtrip');ax[0].grid(alpha=.2)
ax[1].bar(np.arange(8),[g['video_hours_per_day'] for g in s['per_gpu']],color='#348985');ax[1].axhline(s['video_hours_per_gpu_day_average'],color='#cf7145',linestyle='--',label='Average in simultaneous 8-GPU run');ax[1].set(xlabel='GPU index',ylabel='Projected video hours / day',title='Contribution during the same 10-minute interval');ax[1].set_xticks(range(8));ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.2)
fig.suptitle('Omega: 4 fps / window 240 / overlap 36');fig.tight_layout();fig.savefig(OUT/'throughput.png',dpi=180)
