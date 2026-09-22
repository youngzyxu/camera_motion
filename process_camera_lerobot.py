#!/usr/bin/env python3
"""LeRobot camera poses: manifest preparation, durable HTTP server, GPU client.
Design follows l3_cut/web_humandata_processing/server_client; standalone SQLite queue.
"""
import argparse, hashlib, json, os, random, shutil, socket, sqlite3, subprocess, sys, threading, time, uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.error import URLError, HTTPError
QUEUE_HTTP=build_opener(ProxyHandler({}))  # Private queue traffic must not use an outbound HTTP proxy.
ROOT=Path(__file__).resolve().parent
BCECMD='/mnt/pfs/pfs-yc2F4O/modelTeam/code/xuzhiyong/bin/bcecmd-0.5.17-1/bcecmd'
DEFAULT_DATA='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-human-dataset/processed_lerobot/web_first_36K/dataset'
DEFAULT_BOS='bos:/liberai-web-humandata/processing/for_next_20260901_150304_249934232/temp_camera_motion'
CONFIG={'model':'VGGT-Omega-1B-512','fps':4,'window':240,'overlap':36,'stride':204,'stitch':'orientation_sim3_irls12_blend_v1','include_last_frame':True,'sampling_version':'4fps_include_last_v1'}

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False))

def prepare(a):
 root=Path(a.dataset).resolve();rng=random.Random(a.seed)
 chunks=sorted((root/'videos').glob('chunk-*'))
 if a.sample_chunks: chunks=rng.sample(chunks,min(a.sample_chunks,len(chunks)))
 tasks=[]
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 # Full-corpus preparation streams records instead of retaining millions in RAM.
 if not a.sample_chunks and not a.per_chunk:
  count=0
  with out.open('w') as f:
   for chunk in chunks:
    for video in sorted(chunk.glob('*/*.mp4')):
     rel=video.relative_to(root/'videos');task={'task_id':hashlib.sha256(str(rel).encode()).hexdigest()[:24],'video':str(video),'relative_video':str(rel),'episode':video.stem,'camera':video.parent.name,'chunk':chunk.name}
     f.write(json.dumps(task)+'\n');count+=1
  dump(str(out)+'.meta.json',{'dataset':str(root),'tasks':count,'actual_chunks':len(chunks),'sampling':'all videos, streaming in chunk order'})
  print('PREPARED',count,flush=True);return
 for chunk in chunks:
  files=sorted(chunk.glob('*/*.mp4'))
  if a.per_chunk:files=rng.sample(files,min(a.per_chunk,len(files)))
  for p in files:
   rel=p.relative_to(root/'videos');tasks.append({'task_id':hashlib.sha256(str(rel).encode()).hexdigest()[:24],'video':str(p),'relative_video':str(rel),'episode':p.stem,'camera':p.parent.name,'chunk':chunk.name})
 rng.shuffle(tasks);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:
  for t in tasks:f.write(json.dumps(t)+'\n')
 dump(str(out)+'.meta.json',{'dataset':str(root),'tasks':len(tasks),'seed':a.seed,'sample_chunks':a.sample_chunks,'per_chunk':a.per_chunk,'actual_chunks':len(chunks),'sampling':'uniform chunks, uniform files within chunks, shuffled order'})
 print('PREPARED',len(tasks),flush=True)

class Queue:
 def __init__(self,path,manifest,lease=180,max_attempts=3):
  self.path=str(Path(path).resolve())
  self.lock=threading.Lock();self.lease=lease;self.max_attempts=max_attempts
  self.db=sqlite3.connect(path,check_same_thread=False,timeout=60);self.db.execute('PRAGMA journal_mode=WAL')
  self.db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, task TEXT, status TEXT, token TEXT, deadline REAL, attempts INTEGER, result TEXT)')
  self.db.execute('CREATE INDEX IF NOT EXISTS tasks_status_deadline ON tasks(status,deadline)')
  self.db.execute('CREATE INDEX IF NOT EXISTS tasks_pending ON tasks(status)')
  with self.db:
   with open(manifest) as f:
    for line in f:
     t=json.loads(line);self.db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,\'pending\',NULL,0,0,NULL)',(t['task_id'],json.dumps(t)))
 def call(self,op,p):
  if op=='status':
   # Large-batch statistics must never block claims, heartbeats or result commits.
   reader=sqlite3.connect('file:'+self.path+'?mode=ro',uri=True,timeout=60)
   try:return {'counts':dict(reader.execute('SELECT status,count(*) FROM tasks GROUP BY status').fetchall())}
   finally:reader.close()
  with self.lock,self.db:
   now=time.time()
   self.db.execute("UPDATE tasks SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'pending' END WHERE status='running' AND deadline<?",(self.max_attempts,now))
   if op=='status':return {'counts':dict(self.db.execute('SELECT status,count(*) FROM tasks GROUP BY status').fetchall())}
   if op=='claim':
    row=self.db.execute("SELECT id,task FROM tasks WHERE status='pending' ORDER BY rowid LIMIT 1").fetchone()
    if not row:return {'task':None,'exhausted':not self.db.execute("SELECT 1 FROM tasks WHERE status='running' LIMIT 1").fetchone()}
    token=uuid.uuid4().hex;self.db.execute("UPDATE tasks SET status='running',token=?,deadline=?,attempts=attempts+1 WHERE id=?",(token,now+self.lease,row[0]));return {'task':{**json.loads(row[1]),'lease_id':token,'worker_id':p.get('client_id','unknown')}}
   row=self.db.execute('SELECT status,token,attempts FROM tasks WHERE id=?',(p['task_id'],)).fetchone()
   if not row or row[1]!=p['lease_id']:return {'accepted':False,'reason':'stale lease'}
   if op=='complete' and row[0]=='done':return {'accepted':True}
   if row[0]!='running':return {'accepted':False,'reason':'not running'}
   if op=='heartbeat':self.db.execute('UPDATE tasks SET deadline=? WHERE id=?',(now+self.lease,p['task_id']))
   elif op=='complete':
    p={**p,'accepted_at':now}
    status='done' if p['ok'] else ('failed' if row[2]>=self.max_attempts else 'pending')
    self.db.execute('UPDATE tasks SET status=?,result=? WHERE id=?',(status,json.dumps(p),p['task_id']))
   else:raise ValueError(op)
   return {'accepted':True}

def serve(a):
 q=Queue(a.db,a.manifest,a.lease_seconds,a.max_attempts)
 class Handler(BaseHTTPRequestHandler):
  def do_POST(self):
   try:
    p=json.loads(self.rfile.read(int(self.headers['Content-Length'])));r=q.call(self.path.strip('/'),p);body=json.dumps(r).encode();self.send_response(200)
   except Exception as e:body=json.dumps({'error':str(e)}).encode();self.send_response(400)
   self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def log_message(self,*args):pass
 server=ThreadingHTTPServer((a.host,a.port),Handler)
 print('SERVER_READY',a.host,a.port,flush=True);server.serve_forever()

def rpc(server,op,p):
 data=json.dumps(p).encode();last=None
 attempt=0;persistent=os.environ.get('CAMERA_RPC_RETRY')=='1' and op!='status'
 while persistent or attempt<(1 if op=='status' else 3):
  try:
   with QUEUE_HTTP.open(Request(server.rstrip('/')+'/'+op,data=data,headers={'Content-Type':'application/json'}),timeout=120 if op=='status' else 20) as r:result=json.load(r)
   if 'error' in result:raise RuntimeError(result['error'])
   return result
  except Exception as e:
   last=e;attempt+=1
   if persistent and (isinstance(e,HTTPError) and e.code<500 or not isinstance(e,(URLError,TimeoutError,ConnectionError,OSError))):raise
   time.sleep(min(attempt,10))
 raise last

def transfer(src,dst):
 for attempt in range(3):
  p=subprocess.run([BCECMD,'bos','cp',str(src),str(dst),'--yes','--disable-bar','--disable-task-progress'],capture_output=True,text=True,timeout=120)
  if p.returncode==0:return
  time.sleep(attempt+1)
 raise RuntimeError('BOS transfer failed: '+p.stdout[-500:]+p.stderr[-500:])

def client(a,model=None):
 os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True');sys.path.insert(0,str(ROOT/'scripts'))
 import torch,numpy as np
 from run_omega_video import run_video,load_model,predict
 stage=Path(a.stage).resolve();stage.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(a.cpu_threads);torch.manual_seed(0)
 startup=time.monotonic()
 if model is None:
  model=load_model('omega');predict(model,torch.rand(2,3,512,512,device='cuda'),'omega');torch.cuda.synchronize()
 begin=time.monotonic();records=[];failures=[];stop_at=begin+a.seconds if a.seconds else float('inf')
 meta={'config':CONFIG,'gpu':str(torch.cuda.get_device_properties(0)),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'model_load_and_warmup_s':begin-startup,'seconds_requested':a.seconds,'bos_prefix':a.bos_prefix,'pid':os.getpid(),'client_id':socket.gethostname()+':'+str(os.getpid()),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cpu_threads':a.cpu_threads}
 dump(stage/'metadata.json',meta);print('CLIENT_READY',flush=True)
 while time.monotonic()<stop_at and not (a.stop_file and Path(a.stop_file).exists()):
  try:response=rpc(a.server,'claim',{'client_id':socket.gethostname()+':'+str(os.getpid())})
  except Exception as e:
   if not a.stay_alive:raise
   print('SERVER_RETRY',str(e),flush=True);time.sleep(5);continue
  task=response['task']
  if task is None:
   if response['exhausted'] and not a.stay_alive:break
   time.sleep(1);continue
  token={'task_id':task['task_id'],'lease_id':task['lease_id']};stop=threading.Event();heartbeat_errors=[]
  def beat(token=token,stop=stop,heartbeat_errors=heartbeat_errors):
   while not stop.wait(30):
    try:
     if not rpc(a.server,'heartbeat',token)['accepted']:heartbeat_errors.append('lease rejected');return
    except Exception as e:heartbeat_errors.append(str(e))
  thread=threading.Thread(target=beat,daemon=True);thread.start();t=time.monotonic()
  try:
   dest=stage/'tasks'/task['task_id']/task['lease_id']/'camera';dest.parent.mkdir(parents=True,exist_ok=True)
   row=run_video(model,Path(task['video']),4,240,36,dest)
   row.update(task=task,config=CONFIG);dump(str(dest)+'.json',row)
   remote=a.bos_prefix.rstrip('/')+'/'+task['chunk']+'/'+task['camera']+'/'+task['episode']+'/'+task['lease_id']
   u=time.monotonic();hashes={}
   for suffix in ['.npz','.json']:
    src=Path(str(dest)+suffix);hashes[suffix]=hashlib.sha256(src.read_bytes()).hexdigest();transfer(src,remote+'/camera'+suffix)
    if a.verify_upload:
     check=dest.parent/('readback'+suffix);transfer(remote+'/camera'+suffix,check)
     assert hashlib.sha256(check.read_bytes()).hexdigest()==hashes[suffix],'BOS readback checksum mismatch';check.unlink()
   marker={'task_id':task['task_id'],'lease_id':task['lease_id'],'config':CONFIG,'sha256':hashes,'video':task['video'],'remote':remote,'readback_verified':a.verify_upload}
   dump(dest.parent/'SUCCESS.json',marker);transfer(dest.parent/'SUCCESS.json',remote+'/SUCCESS.json')
   row.update(upload_s=time.monotonic()-u,task_wall_s=time.monotonic()-t,remote=remote,readback_verified=a.verify_upload)
   if heartbeat_errors:raise RuntimeError(str(heartbeat_errors))
   assert rpc(a.server,'complete',{**token,'ok':True,'metrics':row})['accepted'],'Completion rejected'
   records.append(row);print('COMPLETE',len(records),task['episode'],round(row['video_seconds'],2),round(row['task_wall_s'],2),flush=True)
   with (stage/'completed.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if not a.keep_local:shutil.rmtree(dest.parent)
  except Exception as e:
   failure={**token,'error':str(e)};failures.append(failure);print('FAILED',json.dumps(failure),flush=True)
   rpc(a.server,'complete',{**token,'ok':False,'error':str(e)})
   with (stage/'failed.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
   torch.cuda.empty_cache()
  finally:stop.set();thread.join(timeout=2)
 elapsed=time.monotonic()-begin;video=sum(r['video_seconds'] for r in records)
 summary={**meta,'completed':len(records),'failed_attempts':len(failures),'wall_s':elapsed,'wall_including_startup_s':time.monotonic()-startup,'video_s':video,'video_hours_per_gpu_day':video/elapsed*24,'video_hours_per_gpu_day_including_startup':video/(time.monotonic()-startup)*24,'inference_s':sum(r['inference_s'] for r in records),'upload_s':sum(r['upload_s'] for r in records),'peak_allocated_gib':max((r['peak_allocated_gib'] for r in records),default=0),'stop_rule':'stop claiming at deadline, finish current video including upload; divide by actual elapsed'}
 dump(stage/'summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)

def export_results(a):
 db=sqlite3.connect('file:'+str(Path(a.db).resolve())+'?mode=ro',uri=True)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);count=0
 with out.open('w') as f:
  for task_id,result in db.execute("SELECT id,result FROM tasks WHERE status='done' ORDER BY rowid"):
   r=json.loads(result);m=r['metrics'];f.write(json.dumps({'task_id':task_id,'remote':m['remote'],'video':m['video'],'task':m['task'],'config':m['config'],'video_seconds':m['video_seconds']})+'\n');count+=1
 db.close();print('EXPORTED',count,flush=True)

def enqueue(a):
 if not Path(a.db).is_file():raise ValueError('Start the server database first')
 db=sqlite3.connect(a.db,timeout=30);added=0;batch=[]
 def put(batch):
  before=db.total_changes
  with db:db.executemany("INSERT OR IGNORE INTO tasks VALUES (?,?,'pending',NULL,0,0,NULL)",batch)
  return db.total_changes-before
 with open(a.manifest) as f:
  for line in f:
   t=json.loads(line);batch.append((t['task_id'],json.dumps(t)))
   if len(batch)>=1000:added+=put(batch);batch=[]
 if batch:added+=put(batch)
 db.close();print('ENQUEUED',added,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 s=sub.add_parser('prepare');s.add_argument('--dataset',default=DEFAULT_DATA);s.add_argument('--output',required=True);s.add_argument('--seed',type=int,default=20260909);s.add_argument('--sample-chunks',type=int,default=0);s.add_argument('--per-chunk',type=int,default=0);s.set_defaults(fn=prepare)
 s=sub.add_parser('server');s.add_argument('--manifest',required=True);s.add_argument('--db',required=True);s.add_argument('--host',default='127.0.0.1');s.add_argument('--port',type=int,default=18764);s.add_argument('--lease-seconds',type=float,default=180);s.add_argument('--max-attempts',type=int,default=3);s.set_defaults(fn=serve)
 s=sub.add_parser('client');s.add_argument('--server',default='http://127.0.0.1:18764');s.add_argument('--stage',required=True);s.add_argument('--bos-prefix',default=DEFAULT_BOS+'/omega_4fps_w240_ov36');s.add_argument('--seconds',type=float,default=0);s.add_argument('--cpu-threads',type=int,default=6);s.add_argument('--verify-upload',action='store_true');s.add_argument('--keep-local',action='store_true');s.add_argument('--pipeline',action='store_true');s.add_argument('--reuse-cuda-cache',action='store_true',help='Keep CUDA allocator cache between videos; retain clearing on inference errors');s.add_argument('--stay-alive',action='store_true',help='Keep loaded model alive when queue is empty');s.add_argument('--stop-file',help='Finish current task and exit when this file exists');s.set_defaults(fn=client)
 s=sub.add_parser('export');s.add_argument('--db',required=True);s.add_argument('--output',required=True);s.set_defaults(fn=export_results)
 s=sub.add_parser('enqueue');s.add_argument('--db',required=True);s.add_argument('--manifest',required=True);s.set_defaults(fn=enqueue)
 a=p.parse_args()
 if a.mode=='client' and a.pipeline:
  os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True');sys.path.insert(0,str(ROOT/'scripts'))
  import torch
  from benchmark_camera import load_model,predict
  from lerobot_pipeline import run
  torch.set_num_threads(a.cpu_threads);model=load_model('omega');predict(model,torch.rand(2,3,512,512,device='cuda'),'omega');run(a,model)
 else:a.fn(a)
if __name__=='__main__':main()
