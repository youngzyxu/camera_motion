"""Verified partial export plus final append-only merge of remaining accepted results."""
import argparse,datetime,hashlib,json,os,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from process_camera_lerobot import BCECMD
RUN=PROJECT/'runs/chenqi_egomixed_0920_5090'
REMOTE='bos:/liberai-web-humandata/processing/camera_motion_feibiao/chenqi_egomixed_0920'

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def transfer(src,dst):
 for attempt in range(3):
  try:
   p=subprocess.run([BCECMD,'bos','cp',str(src),str(dst),'--yes','--disable-bar','--disable-task-progress'],capture_output=True,text=True,timeout=1800)
   if p.returncode==0:return
   error=p.stdout[-500:]+p.stderr[-500:]
  except subprocess.TimeoutExpired:error='Transfer timed out after 1800s'
  time.sleep(attempt+1)
 raise RuntimeError(error)

def publish(out,summary,summary_name):
 transfer(out,REMOTE+'/'+out.name)
 back=out.with_suffix('.readback');transfer(REMOTE+'/'+out.name,back)
 assert digest(back)==summary['sha256'],'BOS readback SHA256 mismatch';back.unlink()
 summary.update(bos_readback_sha256_verified=True,bos_uri=REMOTE+'/'+out.name,published_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 p=out.parent/summary_name;p.write_text(json.dumps(summary,indent=2));transfer(p,REMOTE+'/'+summary_name)
 return summary

def record(task_id,raw,res):
 t=json.loads(raw);r=json.loads(res);m=r['metrics']
 assert r['ok'] and m['readback_verified'] and task_id==t['task_id']==r['task_id']
 assert t['batch']=='chenqi_egomixed_0920'
 assert m['remote'].endswith('/'+r['lease_id'])
 return {'task':t,'result':r,'source_queue':'global','camera_npz_uri':m['remote']+'/camera.npz','camera_json_uri':m['remote']+'/camera.json','success_uri':m['remote']+'/SUCCESS.json'},float(m['video_seconds']),r['lease_id']

def export_partial(db,out):
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 ids=out.with_suffix('.ids.sqlite');ids_tmp=ids.with_suffix('.tmp');idx=sqlite3.connect(ids_tmp)
 idx.execute('CREATE TABLE included(id TEXT PRIMARY KEY,lease TEXT NOT NULL)')
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True,timeout=60);c.execute('BEGIN')
 expected=c.execute("SELECT count(*) FROM tasks WHERE status='done'").fetchone()[0]
 snapshot=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=out.with_suffix('.tmp');n=0;seconds=0.;h=hashlib.sha256()
 try:
  with tmp.open('wb',buffering=8*1024*1024) as f:
   def emit(b):f.write(b);h.update(b)
   emit(b'[\n')
   for tid,raw,res in c.execute("SELECT id,task,result FROM tasks WHERE status='done' ORDER BY rowid"):
    item,sec,lease=record(tid,raw,res)
    idx.execute('INSERT INTO included VALUES (?,?)',(tid,lease))
    if n:emit(b',\n')
    emit(json.dumps(item,ensure_ascii=False,separators=(',',':')).encode());n+=1;seconds+=sec
    if n%100000==0:print('PARTIAL_EXPORTED',n,flush=True)
   emit(b'\n]\n')
  assert n==expected
  idx.commit();idx.close();c.close();os.replace(tmp,out);os.replace(ids_tmp,ids)
 except BaseException:
  idx.close();c.close();raise
 return {'kind':'partial','complete':False,'snapshot_utc':snapshot,'records':n,'video_hours':seconds/3600,'video_seconds':seconds,'sha256':h.hexdigest(),'bytes':out.stat().st_size,'local_json':str(out),'ids_path':str(ids),'ids_sha256':digest(ids),'contains_pose_arrays':False,'unique_task_ids':True}

def merge_final(db,base,out,expected):
 out=Path(out);src=Path(base['local_json']);ids=Path(base['ids_path'])
 assert base['bos_readback_sha256_verified'] and digest(ids)==base['ids_sha256']
 # Keep the million-key membership index on the server's local disk/cache.
 local_ids=Path(db).parent/(out.stem+'.base_ids.sqlite');shutil.copyfile(ids,local_ids)
 assert digest(local_ids)==base['ids_sha256']
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True,timeout=60)
 c.execute('ATTACH DATABASE ? AS exported',('file:'+str(local_ids)+'?mode=ro',))
 c.execute('PRAGMA exported.cache_size=-262144')
 c.execute('BEGIN');counts=dict(c.execute('SELECT status,count(*) FROM tasks GROUP BY status'))
 assert counts.get('done',0)==expected and not sum(v for k,v in counts.items() if k!='done'),counts
 assert c.execute('SELECT count(*) FROM exported.included').fetchone()[0]==base['records']
 assert c.execute("SELECT count(*) FROM exported.included b LEFT JOIN tasks t ON t.id=b.id WHERE t.id IS NULL OR t.status!='done' OR t.token!=b.lease").fetchone()[0]==0,'Base records changed since export'
 tmp=out.with_suffix('.tmp');h=hashlib.sha256();original=hashlib.sha256();n=base['records'];seconds=base['video_seconds'];delta=0
 with tmp.open('wb',buffering=8*1024*1024) as f:
  def emit(b):f.write(b);h.update(b)
  with src.open('rb') as old:
   left=src.stat().st_size-3
   while left:
    b=old.read(min(left,8*1024*1024));assert b
    original.update(b);emit(b);left-=len(b)
   tail=old.read();original.update(tail);assert tail==b'\n]\n'
  assert original.hexdigest()==base['sha256'],'Base JSON hash changed'
  for tid,raw,res in c.execute("SELECT t.id,t.task,t.result FROM tasks t NOT INDEXED WHERE t.status='done' AND NOT EXISTS(SELECT 1 FROM exported.included b WHERE b.id=t.id) ORDER BY t.rowid"):
   item,sec,_=record(tid,raw,res)
   if n:emit(b',\n')
   emit(json.dumps(item,ensure_ascii=False,separators=(',',':')).encode());n+=1;delta+=1;seconds+=sec
  emit(b'\n]\n')
 c.close();local_ids.unlink();assert n==expected;os.replace(tmp,out)
 return {'kind':'complete','complete':True,'records':n,'base_records':base['records'],'delta_records':delta,'video_hours':seconds/3600,'sha256':h.hexdigest(),'bytes':out.stat().st_size,'merged_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}

def finalize(s):
 base=json.loads((RUN/'partial_export.json').read_text());top=json.loads((RUN/'topology.json').read_text());out=RUN/'camera_pose_results_all.json'
 result=merge_final(top['queue_db'],base,out,s['inventory']['selected_tasks'])
 result.update(inventory=s['inventory'],manifest_camera_hours=s['inventory']['selected_hours'],manifest_episode_hours=s['inventory']['episode_hours'])
 publish(out,result,'camera_pose_results_all.summary.json')
 (RUN/'final_summary.json').write_text(json.dumps(result,indent=2))
 print('ALL_DONE_AND_PUBLISHED',json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true',required=True);a=p.parse_args()
 top=json.loads((RUN/'topology.json').read_text());stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 out=RUN/f'camera_pose_results_partial_{stamp}.json';meta=export_partial(top['queue_db'],out)
 print('PARTIAL_LOCAL_COMPLETE',json.dumps(meta),flush=True)
 publish(out,meta,out.stem+'.summary.json')
 tmp=RUN/'partial_export.tmp';tmp.write_text(json.dumps(meta,indent=2));os.replace(tmp,RUN/'partial_export.json')
 print('PARTIAL_PUBLISHED',json.dumps(meta),flush=True)
if __name__=='__main__':main()
