"""Same 2048 videos, same eight resident models: serial then pipeline."""
import argparse,json,os,sys,time,sqlite3
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/lerobot_pipeline_ab';sys.path.insert(0,str(ROOT))

def worker(gpu):
 os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
 import torch
 from benchmark_camera import load_model,predict
 from process_camera_lerobot import client,dump
 from lerobot_pipeline import run
 torch.set_num_threads(6);torch.manual_seed(0);t=time.monotonic();model=load_model('omega');predict(model,torch.rand(2,3,512,512,device='cuda'),'omega');torch.cuda.synchronize()
 dump(OUT/f'ready_gpu{gpu}.json',{'pid':os.getpid(),'gpu':gpu,'startup_s':time.monotonic()-t})
 for mode,port in [('serial',18768),('pipeline',18769)]:
  while not (OUT/f'{mode}_start.json').exists():time.sleep(.02)
  a=SimpleNamespace(stage=str(OUT/mode/f'gpu{gpu}'),server=f'http://127.0.0.1:{port}',seconds=0,stop_file=None,stay_alive=False,verify_upload=True,keep_local=True,cpu_threads=6,bos_prefix=f'bos:/liberai-web-humandata/processing/for_next_20260901_150304_249934232/temp_camera_motion/pipeline_ab_20260909/{mode}')
  if mode=='serial':client(a,model)
  else:run(a,model)
  dump(OUT/f'{mode}_done_gpu{gpu}.json',{'unix_s':time.time()})
 # Retain loaded models after the comparison; can continue production through command files.
 dump(OUT/f'resident_gpu{gpu}.json',{'pid':os.getpid(),'ready':True})
 while not (OUT/'STOP').exists():
  command=OUT/f'continue_gpu{gpu}.json'
  if command.exists():
   spec=json.loads(command.read_text());command.unlink();a=SimpleNamespace(**spec);run(a,model)
  time.sleep(1)

def control():
 from process_camera_lerobot import dump
 while not all((OUT/f'ready_gpu{i}.json').exists() for i in range(8)):time.sleep(.5)
 for mode in ['serial','pipeline']:
  start=time.time();dump(OUT/f'{mode}_start.json',{'unix_s':start})
  while not all((OUT/f'{mode}_done_gpu{i}.json').exists() for i in range(8)):
   db=sqlite3.connect(OUT/f'{mode}.sqlite');counts=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'));db.close()
   with (OUT/f'{mode}_monitor.jsonl').open('a') as f:f.write(json.dumps({'unix_s':time.time(),'elapsed_s':time.time()-start,'counts':counts})+'\n')
   time.sleep(2)
  end=max(json.loads((OUT/f'{mode}_done_gpu{i}.json').read_text())['unix_s'] for i in range(8));dump(OUT/f'{mode}_timing.json',{'start_unix_s':start,'end_unix_s':end,'wall_s':end-start})
  print('PHASE_DONE',mode,end-start,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int);ap.add_argument('--control',action='store_true');a=ap.parse_args()
 if a.control:control()
 else:worker(a.gpu)
