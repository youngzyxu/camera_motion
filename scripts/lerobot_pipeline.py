"""Bounded decode -> GPU -> upload pipeline; preserves single-video inference."""
import gc, hashlib, json, os, queue, shutil, socket, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from benchmark_camera import predict
from window_trajectory import stitch
from frame_sampling import sample_indices
from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio, _max_size_target_shape
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from process_camera_lerobot import CONFIG,dump,rpc,transfer

class Job:
 def __init__(self,task,stage,server):
  self.task=task;self.token={k:task[k] for k in ['task_id','lease_id']};self.started=time.monotonic();self.stop=threading.Event();self.errors=[]
  self.dest=stage/'tasks'/task['task_id']/task['lease_id']/'camera';self.dest.parent.mkdir(parents=True,exist_ok=True)
  self.thread=threading.Thread(target=self.beat,args=(server,),daemon=True);self.thread.start()
 def beat(self,server):
  while not self.stop.wait(30):
   try:
    if not rpc(server,'heartbeat',self.token)['accepted']:self.errors.append('stale lease');return
   except Exception as e:print('HEARTBEAT_RETRY',str(e),flush=True)
 def close(self):
  self.stop.set();self.thread.join(timeout=25)
  if hasattr(self,'local_video'):self.local_video.unlink(missing_ok=True)

def decode(job,packets):
 """Only CPU work; retain <=240 images plus a bounded packet queue."""
 video=getattr(job,'local_video',Path(job.task['video']))
 threads=int(os.environ.get('CAMERA_DECODE_THREADS','0'))
 cap=cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,threads]) if threads else cv2.VideoCapture(str(video))
 srcfps=cap.get(5);n=int(cap.get(7))
 if not np.isfinite(srcfps) or srcfps<=0 or n<1:cap.release();raise ValueError('Invalid source FPS or frame count')
 idx=sample_indices(n,srcfps,4);wanted=set(idx.tolist())
 job.idx=idx;job.srcfps=srcfps;job.n=n;job.decode_s=0.;job.queue_put_wait_s=0.
 buffers=[];seen=0;start=0;last_full=0;spans=[];segment=time.monotonic()
 def emit(end):
  nonlocal segment
  job.decode_s+=time.monotonic()-segment
  t=time.monotonic();packets.put(('window',job,start,end,buffers));job.queue_put_wait_s+=time.monotonic()-t;spans.append((start,end));segment=time.monotonic()
 try:
  for f in range(n):
   if not cap.grab():raise ValueError(f'Incomplete decode {f}/{n}')
   if f not in wanted:continue
   ok,frame=cap.retrieve()
   if not ok:raise ValueError(f'Cannot retrieve {f}')
   im=_crop_to_supported_aspect_ratio(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)));w,h=im.size;th,tw=_max_size_target_shape(h/w,512,16)
   buffers.append(to_tensor(im.resize((tw,th),Image.Resampling.BICUBIC)));seen+=1
   if len(buffers)==240:emit(seen);last_full=seen;buffers=buffers[-36:];start=seen-36
  if seen!=len(idx):raise ValueError('Missing sampled frames')
  if seen>last_full:emit(seen)
  job.decode_s+=time.monotonic()-segment;job.spans=spans;packets.put(('end',job))
 finally:cap.release()

def run(a,model):
 stage=Path(a.stage).resolve();stage.mkdir(parents=True,exist_ok=True);begin=time.monotonic();deadline=begin+a.seconds if a.seconds else float('inf')
 packets=queue.Queue(maxsize=2);slots=threading.Semaphore(4);lock=threading.Lock();records=[];failures=[];upload_futures=[]
 def failed(job,error):
  record={**job.token,'error':str(error),'time':time.time()}
  try:rpc(a.server,'complete',{**job.token,'ok':False,'error':str(error)})
  finally:
   with lock:
    failures.append(record)
    with (stage/'failed.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   job.close();slots.release()
  print('FAILED',str(error),flush=True)
 prefetch=os.environ.get('CAMERA_PREFETCH_LOCAL')=='1';prepared=queue.Queue(maxsize=2)
 def decode_prepared():
  try:
   while True:
    job=prepared.get()
    if job is None:break
    try:decode(job,packets)
    except Exception as e:packets.put(('error',job,e))
  finally:packets.put(('stop',))
 def producer():
  try:
   while time.monotonic()<deadline and not (a.stop_file and Path(a.stop_file).exists()):
    slots.acquire()
    if a.stop_file and Path(a.stop_file).exists():slots.release();break
    claim_started=time.monotonic()
    try:response=rpc(a.server,'claim',{'client_id':socket.gethostname()+':'+str(os.getpid())})
    except Exception as error:
     slots.release()
     if not a.stay_alive:raise
     print('CLAIM_RETRY',str(error),flush=True);time.sleep(1);continue
    task=response['task']
    if task is None:
     slots.release()
     if response['exhausted'] and not a.stay_alive:break
     time.sleep(2);continue
    job=Job(task,stage,a.server);job.claim_rpc_s=time.monotonic()-claim_started
    try:
     if prefetch:
      source=Path(task['video']);job.local_video=job.dest.parent/('source'+source.suffix)
      t=time.monotonic()
      if shutil.disk_usage(stage).free<source.stat().st_size+2*1024**3:raise RuntimeError('Local staging disk below reserve')
      shutil.copyfile(source,job.local_video);job.prefetch_s=time.monotonic()-t
      prepared.put(job)
     else:decode(job,packets)
    except Exception as e:
     if prefetch:failed(job,e)
     else:packets.put(('error',job,e))
  except Exception as e:packets.put(('fatal',e))
  finally:
   if prefetch:prepared.put(None)
   else:packets.put(('stop',))
 def upload(job,row):
  try:
   t=time.monotonic();task=job.task;dest=job.dest
   remote=a.bos_prefix.rstrip('/')+'/'+task['chunk']+'/'+task['camera']+'/'+task['episode']+'/'+task['lease_id'];hashes={}
   # Transfer retries use already-computed files; no GPU re-run for transient errors.
   for suffix in ['.npz','.json']:
    src=Path(str(dest)+suffix);hashes[suffix]=hashlib.sha256(src.read_bytes()).hexdigest();transfer(src,remote+'/camera'+suffix)
    if a.verify_upload:
     check=dest.parent/('readback'+suffix);transfer(remote+'/camera'+suffix,check)
     assert hashlib.sha256(check.read_bytes()).hexdigest()==hashes[suffix],'BOS checksum mismatch';check.unlink()
   marker={'task_id':task['task_id'],'lease_id':task['lease_id'],'config':CONFIG,'sha256':hashes,'video':task['video'],'remote':remote,'readback_verified':a.verify_upload}
   dump(dest.parent/'SUCCESS.json',marker);transfer(dest.parent/'SUCCESS.json',remote+'/SUCCESS.json')
   row.update(prefetch_s=getattr(job,'prefetch_s',0.),upload_s=time.monotonic()-t,task_wall_s=time.monotonic()-job.started,remote=remote,readback_verified=a.verify_upload)
   if job.errors:raise RuntimeError(str(job.errors))
   ack_started=time.monotonic()
   assert rpc(a.server,'complete',{**job.token,'ok':True,'metrics':row})['accepted'],'Completion rejected'
   row['completed_unix']=time.time();row['complete_ack_s']=time.monotonic()-ack_started;row['claim_rpc_s']=getattr(job,'claim_rpc_s',0.)
   with lock:
    records.append(row)
    with (stage/'completed.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if not a.keep_local:shutil.rmtree(dest.parent)
   print('COMPLETE',len(records),task['episode'],round(row['video_seconds'],2),flush=True)
   job.close();slots.release()
  except Exception as e:failed(job,e)
 if prefetch:threading.Thread(target=decode_prepared,daemon=True).start()
 producer_thread=threading.Thread(target=producer,daemon=True);producer_thread.start();pool=ThreadPoolExecutor(max_workers=int(os.environ.get('CAMERA_UPLOAD_WORKERS','2')))
 active=None;gpu_wait=0.;fatal=None
 while True:
  t=time.monotonic();packet=packets.get();gpu_wait+=time.monotonic()-t;kind=packet[0]
  if kind=='stop':break
  if kind=='fatal':fatal=packet[1];continue
  job=packet[1]
  if kind=='error':failed(job,packet[2]);active=None;continue
  if active is not job:
   active=job;job.preds=[];job.inference_s=0.;job.h2d_s=0.;job.gpu_error=None
   if not getattr(a,'reuse_cuda_cache',False):torch.cuda.empty_cache()
   torch.cuda.reset_peak_memory_stats()
  if kind=='window':
   if job.gpu_error:continue
   try:
    _,_,start,end,buffer=packet;t=time.monotonic();x=torch.stack(buffer).cuda();torch.cuda.synchronize();job.h2d_s+=time.monotonic()-t;job.hw=x.shape[-2:]
    t=time.monotonic();p,k=predict(model,x,'omega');torch.cuda.synchronize();job.inference_s+=time.monotonic()-t;job.preds.append((p,k));del x,buffer
   except Exception as e:job.gpu_error=e;torch.cuda.empty_cache()
  elif kind=='end':
   if job.gpu_error:failed(job,job.gpu_error);active=None;continue
   try:
    t=time.monotonic();pose,k,joins=stitch(job.preds,job.spans);stitch_s=time.monotonic()-t
    assert len(pose)==len(job.idx) and np.isfinite(pose).all()
    row={'video':job.task['video'],'fps':4,'window':240,'overlap':36,'windows':len(job.spans),'sampled_frames':len(job.idx),'source_frames':job.n,'source_fps':job.srcfps,'video_seconds':job.n/job.srcfps,'max_window_input_frames':max(e-s for s,e in job.spans),'processed_window_frames':sum(e-s for s,e in job.spans),'inference_s':job.inference_s,'upload_and_stack_s':job.h2d_s,'decode_preprocess_s':job.decode_s,'decoder_queue_wait_s':job.queue_put_wait_s,'stitch_s':stitch_s,'end_to_end_s':time.monotonic()-job.started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'joins':joins,'scale_fallbacks':sum(j['scale_fallback'] for j in joins),'stitch_version':'orientation_sim3_irls12_blend_v1','status':'ok','task':job.task,'config':CONFIG,'pipeline':('prefetch1_decode1_queue2_gpu1_upload'+os.environ.get('CAMERA_UPLOAD_WORKERS','2')+'_inflight4' if prefetch else 'decode1_queue2_gpu1_upload2_inflight4'),'include_last_frame':True,'sampling_version':'4fps_include_last_v1','preprocessing':'max_size512','pose_convention':'OpenCV c2w; first camera origin; arbitrary translation scale'}
    t=time.monotonic();np.savez_compressed(str(job.dest)+'.npz',c2w=pose.astype(np.float32),intrinsics=k,frame_indices=job.idx,timestamps_s=job.idx/job.srcfps,input_hw=np.array(job.hw),source_fps=job.srcfps);row['save_s']=time.monotonic()-t;dump(str(job.dest)+'.json',row)
    job.preds=[];upload_futures.append(pool.submit(upload,job,row))
   except Exception as e:failed(job,e)
   active=None
 producer_thread.join();pool.shutdown(wait=True)
 for f in upload_futures:f.result()
 elapsed=time.monotonic()-begin;video=sum(r['video_seconds'] for r in records)
 result={'completed':len(records),'failed_attempts':len(failures),'wall_s':elapsed,'video_s':video,'video_hours_per_gpu_day':video/elapsed*24,'gpu_queue_wait_s':gpu_wait,'pipeline':('prefetch1_decode1_queue2_gpu1_upload'+os.environ.get('CAMERA_UPLOAD_WORKERS','2')+'_inflight4' if prefetch else 'decode1_queue2_gpu1_upload2_inflight4'),'config':CONFIG,'reuse_cuda_cache':getattr(a,'reuse_cuda_cache',False)}
 dump(stage/'summary.json',result)
 if fatal:raise fatal
 return result
