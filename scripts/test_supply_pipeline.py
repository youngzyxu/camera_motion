"""Slow reuse must not block fresh decoding; rejected reuse drains via inference."""
import json,os,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import lerobot_pipeline as lp
import reuse_camera
from process_camera_lerobot import Queue

def main():
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);tasks=[]
  for name in ['reuse','fresh','fallback']:
   video=root/(name+'.mp4');video.write_bytes(b'fixture')
   t={'task_id':name,'video':str(video),'chunk':'chunk-000','camera':'ego','episode':name}
   if name!='fresh':t['reuse_camera']={'remote':'fixture'}
   tasks.append(t)
  manifest=root/'manifest';manifest.write_text(''.join(json.dumps(t)+'\n' for t in tasks));q=Queue(root/'queue',manifest)
  decoded=threading.Event();reuse_done=threading.Event();overlapped=[];predicted=[];remote={}
  def prepare(task,dest,transfer,config):
   if task['task_id']=='fallback':raise ValueError('Injected invalid reuse')
   overlapped.append(decoded.wait(5));assert overlapped[-1],'Fresh decode blocked by reuse'
   np.savez_compressed(str(dest)+'.npz',dummy=np.array([1]))
   return {'reused_camera':True,'video_seconds':.5,'task':task,'config':config}
  def decode(job,packets):
   decoded.set();job.idx=np.arange(2);job.srcfps=4;job.n=2;job.decode_s=0.;job.queue_put_wait_s=0.;job.spans=[(0,2)]
   packets.put(('window',job,0,2,[torch.zeros(3,8,8),torch.ones(3,8,8)]));packets.put(('end',job))
  def predict(model,x,method):
   assert reuse_done.wait(5),'Reuse completion still blocked behind GPU inference'
   predicted.append(x.clone());return np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(3),(2,1,1))
  def transfer(src,dst):
   src,dst=str(src),str(dst)
   if dst.startswith('bos:'):remote[dst]=Path(src).read_bytes()
   else:Path(dst).write_bytes(remote[src])
  def rpc(server,op,p):
   if op=='complete' and p.get('ok'):
    assert any('/'+p['task_id']+'/' in k and k.endswith('/SUCCESS.json') for k in remote)
    if p['task_id']=='reuse':reuse_done.set()
   return q.call(op,p)
  a=SimpleNamespace(stage=str(root/'stage'),seconds=0,stop_file=None,server='mock',stay_alive=False,bos_prefix='bos:/test',verify_upload=True,keep_local=False,reuse_cuda_cache=True)
  with patch.dict(os.environ,{'CAMERA_PREFETCH_LOCAL':'1','CAMERA_ASYNC_REUSE':'1'}),patch.object(reuse_camera,'prepare_reuse',prepare),patch.object(lp,'rpc',rpc),patch.object(lp,'transfer',transfer),patch.object(lp,'decode',decode),patch.object(lp,'predict',predict),patch.object(torch.Tensor,'cuda',lambda x:x),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(torch.cuda,'max_memory_reserved',return_value=0):
   summary=lp.run(a,None)
  assert summary['completed']==3 and summary['failed_attempts']==0,summary
  assert overlapped==[True] and len(predicted)==2
  rows=[json.loads(x) for x in (root/'stage/completed.jsonl').read_text().splitlines()]
  assert sum(bool(r.get('reused_camera')) for r in rows)==1
  assert any(r.get('reuse_fallback_reason')=='Injected invalid reuse' for r in rows)
  assert q.call('status',{})['counts']=={'done':3}
 print('PASS reuse/decoding overlap, rejected reuse falls back once, exact completion count and upload commit order')
if __name__=='__main__':main()
