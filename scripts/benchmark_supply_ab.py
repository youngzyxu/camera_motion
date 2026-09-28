"""Same-input A/B on a drained node; BOS reads only, isolated local output.
Upload/readback is emulated at the measured 0.86 s/video so this is not a
production end-to-end throughput claim. Every output array is compared exactly.
"""
import argparse,json,os,sys,time,hashlib,shutil
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args()
 if a.wait_pid:
  until=time.monotonic()+900
  while (Path('/proc')/str(a.wait_pid)/'cmdline').exists():
   cmd=(Path('/proc')/str(a.wait_pid)/'cmdline').read_bytes()
   if b'process_camera_lerobot.py' not in cmd and b'benchmark_supply_ab.py' not in cmd:break
   if time.monotonic()>until:raise TimeoutError('Production worker did not drain; GPU not touched')
   time.sleep(2)
 import numpy as np
 import torch
 import lerobot_pipeline as lp
 from benchmark_camera import load_model,predict
 from process_camera_lerobot import Queue
 torch.set_num_threads(4);torch.manual_seed(0)
 out=a.run.resolve();local=Path('/tmp/camera_supply_ab_20260924')/out.name/f'gpu{a.gpu}';local.mkdir(parents=True,exist_ok=True)
 tasks=json.loads((out/'sample.json').read_text())[a.gpu::8]
 manifest=local/'manifest.jsonl';manifest.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
 original_transfer=lp.transfer;results=[];reference={};input_reference={};mismatches=[]
 model=load_model('omega');predict(model,torch.rand(2,3,512,512,device='cuda'),'omega');torch.cuda.synchronize()
 order=['baseline','optimized','optimized','baseline'] if a.gpu%2==0 else ['optimized','baseline','baseline','optimized']
 for trial,mode in enumerate(order):
  os.environ['CAMERA_PRESTACK']=os.environ['CAMERA_ASYNC_REUSE']='1' if mode=='optimized' else '0'
  stage=local/f'{trial}_{mode}';stage.mkdir(exist_ok=True)
  q=Queue(stage/'queue.sqlite',manifest)
  lp.rpc=lambda server,op,p:q.call(op,p)
  # Cold remote objects may affect reads; alternating ABBA/BAAB balances order.
  remote={}
  def transfer(src,dst):
   src,dst=str(src),str(dst)
   if dst.startswith('bos:/supply-ab/'):
    remote[dst]=Path(src).read_bytes();time.sleep(.172)
   elif src.startswith('bos:/supply-ab/'):
    Path(dst).write_bytes(remote[src]);time.sleep(.172)
   else:original_transfer(src,dst)
  lp.transfer=transfer
  args=SimpleNamespace(stage=str(stage),seconds=0,stop_file=None,server='local',stay_alive=False,bos_prefix='bos:/supply-ab',verify_upload=True,keep_local=True,reuse_cuda_cache=True)
  summary=lp.run(args,model);rows=[json.loads(l) for l in (stage/'completed.jsonl').read_text().splitlines()]
  assert summary['completed']==len(tasks) and summary['failed_attempts']==0,summary
  for row in rows:
   tid=row['task']['task_id'];files=list((stage/'tasks'/tid).glob('*/camera/camera.npz'))
   # job.dest is tasks/id/lease/camera, with camera.npz alongside that stem.
   files=list((stage/'tasks'/tid).glob('*/camera.npz'))
   assert len(files)==1,(tid,files)
   with np.load(files[0]) as data:
    arrays={k:data[k].copy() for k in data.files}
   if tid not in reference:reference[tid]=arrays
   else:
    for k,value in arrays.items():
     if not np.array_equal(reference[tid][k],value,equal_nan=True):mismatches.append({'trial':trial,'task':tid,'field':k,'max_abs':float(np.max(np.abs(reference[tid][k]-value)))})
  summary.update(trial=trial,mode=mode,reused=sum(bool(x.get('reused_camera')) for x in rows),inference_s=sum(x.get('inference_s',0) for x in rows if not x.get('reused_camera')),input_gpu_s=sum(x.get('upload_and_stack_s',0) for x in rows if not x.get('reused_camera')))
  results.append(summary);q.db.close()
  (out/f'gpu{a.gpu}.json').write_text(json.dumps({'gpu':a.gpu,'trials':results,'mismatches':mismatches,'complete':trial==len(order)-1},indent=2))
  print('TRIAL_DONE',trial,mode,summary['wall_s'],'mismatches',len(mismatches),flush=True)
  shutil.rmtree(stage)
 assert not mismatches,mismatches
if __name__=='__main__':main()
