"""Bounded production measurement from accepted result log deltas (no DB scan)."""
import argparse,json,time,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from process_camera_lerobot import rpc
p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=180);a=p.parse_args()
run=ROOT/'runs/chenqi_egomixed_0920_5090';top=json.loads((run/'topology.json').read_text())
def records():
 rows={}
 for node in top['client_nodes']:
  for f in (run/node).glob('gpu*/*/completed.jsonl'):
   for line in f.read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    rows[r['task']['task_id']]={**r,'node':node}
 return rows
old=records();begin=time.time();before=rpc(top['server_url'],'status',{})
time.sleep(a.seconds);after=rpc(top['server_url'],'status',{});elapsed=time.time()-begin
new=records();rows=[r for k,r in new.items() if k not in old];hours=sum(r['video_seconds'] for r in rows)/3600
report={'start_utc':datetime.datetime.fromtimestamp(begin,datetime.timezone.utc).isoformat(),'seconds':elapsed,'completed_log_delta':len(rows),'accepted_count_delta':after['counts'].get('done',0)-before['counts'].get('done',0),'video_hours':hours,'video_hours_per_day':hours/elapsed*86400,'source':'PFS mirrored completion logs, up to 30s per-node mirror lag; boundaries approximate','before':before,'after':after,'nodes':{}}
for node in top['client_nodes']:
 rr=[r for r in rows if r['node']==node]
 report['nodes'][node]={'completed':len(rr),'video_hours':sum(r['video_seconds'] for r in rr)/3600,'mean_inference_s':sum(r['inference_s'] for r in rr)/max(1,len(rr)),'mean_prefetch_s':sum(r.get('prefetch_s',0) for r in rr)/max(1,len(rr)),'mean_upload_s':sum(r['upload_s'] for r in rr)/max(1,len(rr))}
(run/'production_benchmark_5090.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
