"""One supervisor per node, local weights/staging, one model per GPU, graceful drain."""
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/mnt/pfs/pfs-yc2F4O/miniconda3/envs/camera_motion/bin/python'
WEIGHTS=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/xuzhiyong/backup/vggt_omega_1b_512.pt')
BOS='bos:/liberai-web-humandata/processing/camera_motion_feibiao/chenqi_egomixed_0920/results'

def main():
 p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--run',required=True)
 p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--seconds',type=int,default=0)
 a=p.parse_args();run=Path(a.run);config=json.loads((run/'topology.json').read_text())
 assert a.node in config['client_nodes']
 shared=run/a.node;shared.mkdir(exist_ok=True);local=Path('/tmp/camera_motion_5090')/run.name;local.mkdir(parents=True,exist_ok=True)
 guard=(local/'node.lock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stop=local/'STOP';stop.unlink(missing_ok=True)
 def stopped():return stop.exists() or (shared/'STOP').exists() or (run/'STOP').exists()
 def terminate(*_):stop.touch()
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 if stopped():raise SystemExit('Run/node STOP marker is present')
 weight=local/'omega.pt'
 if not weight.exists() or weight.stat().st_size!=WEIGHTS.stat().st_size:
  tmp=weight.with_suffix('.tmp');shutil.copyfile(WEIGHTS,tmp);os.replace(tmp,weight)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTORCH_ALLOC_CONF='expandable_segments:True',CAMERA_OMEGA_WEIGHTS=str(weight),CAMERA_PREFETCH_LOCAL='1',CAMERA_DECODE_THREADS='1',CAMERA_UPLOAD_WORKERS='1',CAMERA_RPC_RETRY='1')
 procs={};stages={};restarts={};next_start={};handles=[];finished=set()
 def launch(gpu):
  stage=local/f'gpu{gpu}'/time.strftime('%Y%m%d_%H%M%S',time.gmtime());stage.mkdir(parents=True,exist_ok=True)
  log=(stage/'worker.log').open('a');handles.append(log)
  cmd=[PY,'-u',str(ROOT/'process_camera_lerobot.py'),'client','--pipeline','--reuse-cuda-cache','--server',config['server_url'],'--stage',str(stage),'--seconds',str(a.seconds),'--stop-file',str(stop),'--verify-upload','--cpu-threads','4','--bos-prefix',BOS]
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=gpu
  procs[gpu]=subprocess.Popen(cmd,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT)
  stages[gpu]=stage;restarts[gpu]=restarts.get(gpu,0)+1
  print('LAUNCH',gpu,procs[gpu].pid,flush=True)
 def sync():
  for gpu,stage in stages.items():
   dest=shared/f'gpu{gpu}'/stage.name;dest.mkdir(parents=True,exist_ok=True)
   for name in ['metadata.json','completed.jsonl','failed.jsonl','summary.json','worker.log']:
    src=stage/name
    if src.exists():shutil.copyfile(src,dest/name)
  (shared/'state.json').write_text(json.dumps({'unix':time.time(),'node':a.node,'stopping':stopped(),'workers':{g:{'pid':p.pid,'returncode':p.poll(),'launches':restarts[g],'stage':str(stages[g])} for g,p in procs.items()}},indent=2))
 try:
  for gpu in a.gpus.split(','):
   if stopped():break
   launch(gpu);time.sleep(3)
  last_sync=0
  while procs:
   if (shared/'STOP').exists() or (run/'STOP').exists():stop.touch()
   for gpu,proc in list(procs.items()):
    code=proc.poll()
    if code is None:continue
    if stopped() or code==0:finished.add(gpu)
    elif restarts[gpu]>=3:
     print('GPU_DISABLED_AFTER_3_FAILURES',gpu,flush=True);finished.add(gpu)
    elif time.time()>=next_start.get(gpu,0):
     sync();launch(gpu);next_start[gpu]=time.time()+30
   if time.time()-last_sync>=30:sync();last_sync=time.time()
   if len(finished)==len(procs):break
   time.sleep(2)
 finally:
  stop.touch()
  for proc in procs.values():proc.wait()
  sync()
  for f in handles:f.close()
 print('NODE_STOPPED',flush=True)
if __name__=='__main__':main()
