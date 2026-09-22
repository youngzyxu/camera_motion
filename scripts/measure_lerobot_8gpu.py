"""Measure a fixed 600-second completion interval while eight workers remain alive."""
import json,sqlite3,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/lerobot_8gpu_4fps_ov36'
while not all((OUT/f'gpu{i}/metadata.json').exists() for i in range(8)):time.sleep(.5)
start=time.time();deadline=time.monotonic()+600
(OUT/'measurement_start.json').write_text(json.dumps({'unix_s':start,'duration_requested_s':600,'rule':'all eight models ready; count server-accepted completions within interval; leave services running'},indent=2))
print('MEASUREMENT_START',start,flush=True)
while True:
 now=time.time();db=sqlite3.connect('file:'+str(OUT/'queue.sqlite')+'?mode=ro',uri=True)
 records=[json.loads(r[0]) for r in db.execute("SELECT result FROM tasks WHERE status='done'")];counts=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'));db.close()
 selected=[r for r in records if r['accepted_at']>=start]
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
 with (OUT/'monitor.jsonl').open('a') as f:f.write(json.dumps({'unix_s':now,'elapsed_s':now-start,'completed':len(selected),'video_s':sum(r['metrics']['video_seconds'] for r in selected),'counts':counts,'gpu':gpu})+'\n')
 remaining=deadline-time.monotonic()
 if remaining<=0:break
 time.sleep(min(5,remaining))
end=time.time();selected=[r for r in selected if r['accepted_at']<=end]
(OUT/'measurement.json').write_text(json.dumps({'start_unix_s':start,'end_unix_s':end,'wall_s':end-start,'gpus':8,'completed':len(selected),'video_s':sum(r['metrics']['video_seconds'] for r in selected),'records':selected},indent=2))
print('MEASUREMENT_COMPLETE',len(selected),end-start,flush=True)
