"""Offline failure/backpressure/commit-order checks; no GPU or external BOS writes."""
import tempfile,json,sys,hashlib,shutil,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import lerobot_pipeline as lp
from process_camera_lerobot import Queue

def main(reuse_cuda_cache=False,prefetch=False):
 with tempfile.TemporaryDirectory() as tmp:
  tmp=Path(tmp);manifest=tmp/'worklist';tasks=[{'task_id':s,'video':s+'.mp4','chunk':'chunk-000','camera':'ego','episode':s} for s in ['good','bad']];manifest.write_text(''.join(json.dumps(t)+'\n' for t in tasks));q=Queue(tmp/'queue.db',manifest)
  if prefetch:
   for task in tasks:
    task['video']=str(tmp/(task['task_id']+'.mp4'));Path(task['video']).write_bytes(b'video-fixture')
    q.db.execute('UPDATE tasks SET task=? WHERE id=?',(json.dumps(task),task['task_id']))
   q.db.commit()
  remote={};calls=[]
  def rpc(server,op,p):
   if op=='complete' and p['ok']:
    assert any(k.endswith('/SUCCESS.json') for k in remote),'Acknowledged before BOS marker'
    calls.append('accepted')
   return q.call(op,p)
  def transfer(src,dst):
   src=str(src);dst=str(dst)
   if dst.startswith('bos:'):remote[dst]=Path(src).read_bytes()
   else:Path(dst).write_bytes(remote[src])
  def decode(job,packets):
   if prefetch:assert job.local_video.read_bytes()==b'video-fixture'
   if job.task['task_id']=='bad':raise ValueError('Injected corrupt video')
   job.idx=np.arange(2);job.srcfps=4;job.n=2;job.decode_s=.01;job.queue_put_wait_s=0.;job.spans=[(0,2)]
   packets.put(('window',job,0,2,[torch.zeros(3,8,8),torch.ones(3,8,8)]));packets.put(('end',job))
  def predict(model,x,method):
   calls.append('predict');return np.tile(np.eye(4),(len(x),1,1)),np.tile(np.eye(3),(len(x),1,1))
  a=SimpleNamespace(stage=str(tmp/'stage'),seconds=0,stop_file=None,server='mock',stay_alive=False,bos_prefix='bos:/test',verify_upload=True,keep_local=True,reuse_cuda_cache=reuse_cuda_cache)
  with patch.dict(os.environ,{'CAMERA_PREFETCH_LOCAL':'1' if prefetch else '0'}),patch.object(lp,'rpc',rpc),patch.object(lp,'transfer',transfer),patch.object(lp,'decode',decode),patch.object(lp,'predict',predict),patch.object(torch.Tensor,'cuda',lambda t:t),patch.object(torch.cuda,'empty_cache') as clear_cache,patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(torch.cuda,'max_memory_reserved',return_value=0):
   result=lp.run(a,None)
   assert clear_cache.call_count == (0 if reuse_cuda_cache else 1), clear_cache.call_count
  assert result['completed']==1 and result['failed_attempts']==3,result
  assert calls==['predict','accepted'],calls
  assert q.call('status',{})['counts']=={'done':1,'failed':1}
 print('PASS decode failure retries, bounded pipeline drain, BOS readback and commit order; good video inferred exactly once')
if __name__=='__main__':main();main(True);main(True,True)
