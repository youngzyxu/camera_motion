"""Publish streaming preparation/processing state; finalize only after the input scan ends."""
import datetime,hashlib,json,pathlib,sqlite3,sys,time,os
PROJECT=pathlib.Path(__file__).resolve().parents[1];ROOT=PROJECT/'runs/chenqi_egomixed_0920_5090';sys.path.insert(0,str(PROJECT))
from process_camera_lerobot import rpc,transfer
REMOTE='bos:/liberai-web-humandata/processing/camera_motion_feibiao/chenqi_egomixed_0920'
def publish(p,name):transfer(p,REMOTE+'/'+name)
def status():
 inv=json.loads((ROOT/'inventory.json').read_text());top=json.loads((ROOT/'topology.json').read_text());q=rpc(top['server_url'],'status',{})
 return {'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inventory':inv,'topology':top,'counts':q['counts'],'rpc_ms':q.get('rpc_ms',{}),'input_scan_complete':inv['complete'],'progress_of_discovered_tasks':q['counts'].get('done',0)/max(sum(q['counts'].values()),1),'progress_is_full_batch':inv['complete']}
def finalize(s):
 out=ROOT/'camera_pose_results_all.json';tmp=out.with_suffix('.tmp');c=sqlite3.connect('file:'+json.loads((ROOT/'topology.json').read_text())['queue_db']+'?mode=ro',uri=True,timeout=60)
 n=0;seconds=0.;h=hashlib.sha256();seen=set()
 with tmp.open('wb',buffering=8*1024*1024) as f:
  def emit(b):f.write(b);h.update(b)
  emit(b'[\n')
  for raw,res in c.execute("SELECT task,result FROM tasks WHERE status='done' ORDER BY rowid"):
   t=json.loads(raw);r=json.loads(res);m=r['metrics'];assert r['ok'] and m['readback_verified'] and t['task_id']==r['task_id'];assert t['batch']=='chenqi_egomixed_0920'
   key=(t['dataset'],t['episode_index'],t['camera']);assert key not in seen;seen.add(key)
   record={'task':t,'result':r,'source_queue':'global','camera_npz_uri':m['remote']+'/camera.npz','camera_json_uri':m['remote']+'/camera.json','success_uri':m['remote']+'/SUCCESS.json'}
   if n:emit(b',\n')
   emit(json.dumps(record,ensure_ascii=False,separators=(',',':')).encode());n+=1;seconds+=m['video_seconds']
  emit(b'\n]\n')
 assert n==s['inventory']['selected_tasks']
 tmp.replace(out);publish(out,out.name)
 back=ROOT/'bos_readback.tmp';transfer(REMOTE+'/'+out.name,back);hh=hashlib.sha256()
 with back.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):hh.update(b)
 assert hh.hexdigest()==h.hexdigest();back.unlink()
 result={'records':n,'video_hours':seconds/3600,'manifest_camera_hours':s['inventory']['selected_hours'],'manifest_episode_hours':s['inventory']['episode_hours'],'sha256':h.hexdigest(),'bytes':out.stat().st_size,'bos_readback_sha256_verified':True,'bos_uri':REMOTE+'/'+out.name,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inventory':s['inventory']}
 summary=ROOT/'final_summary.json';summary.write_text(json.dumps(result,indent=2));publish(summary,'camera_pose_results_all.summary.json');print('ALL_DONE_AND_PUBLISHED',json.dumps(result),flush=True)
def main():
 watch='--watch' in sys.argv
 while True:
  try:
   s=status();tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(s,indent=2));tmp.replace(ROOT/'status.json');print(json.dumps({'scan_complete':s['input_scan_complete'],'discovered':s['inventory']['selected_tasks'],'counts':s['counts']}),flush=True)
   if not watch:return
   publish(ROOT/'status.json','_run_chenqi_egomixed_0920_5090/status.json');publish(ROOT/'inventory.json','_run_chenqi_egomixed_0920_5090/inventory.json')
   if s['input_scan_complete'] and not s['counts'].get('pending',0) and not s['counts'].get('running',0):
    if s['counts'].get('failed',0):print('ATTENTION_TERMINAL_FAILURES',s['counts'],flush=True);return
    if not (ROOT/'final_summary.json').exists():finalize(s)
    return
  except Exception as e:
   if not watch:raise
   print('MONITOR_ERROR',repr(e),flush=True)
  time.sleep(300)
if __name__=='__main__':main()
