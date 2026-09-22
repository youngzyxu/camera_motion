import json,time,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results/lerobot_4fps_ov36';out=root/'monitor.jsonl'
while True:
 rows=[];p=root/'measured/completed.jsonl'
 if p.exists():
  for line in p.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
 r={'time':time.time(),'completed':len(rows),'video_s':sum(r['video_seconds'] for r in rows),'gpu_util_memoryutil_mib_watts':gpu}
 with out.open('a') as f:f.write(json.dumps(r)+'\n')
 if (root/'measured/summary.json').exists():break
 time.sleep(5)
