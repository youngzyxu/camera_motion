import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'results/lerobot_4fps_ov36'
tasks=[json.loads(l) for l in (out/'worklist.jsonl').read_text().splitlines()]
def get(t):
 p=Path(t['video']);meta=p.parents[3]/'meta'/t['chunk']/(t['episode']+'.json');r=json.loads(meta.read_text())['info'];return {'task_id':t['task_id'],'duration_s':r['length']/r['capture_fps'],'fps':r['capture_fps'],'bytes':p.stat().st_size,'chunk':t['chunk']}
with ThreadPoolExecutor(8) as pool:rows=list(pool.map(get,tasks))
durations=np.array([r['duration_s'] for r in rows]);summary={'count':len(rows),'chunks':len(set(r['chunk'] for r in rows)),'video_hours':sum(durations)/3600,'duration_s_mean':float(durations.mean()),'duration_s_p10_p50_p90_p99_max':np.percentile(durations,[10,50,90,99,100]).tolist(),'fraction_over60s':float(np.mean(durations>60)),'source_fps_values':sorted(set(r['fps'] for r in rows))}
(out/'sample_profile.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2))
