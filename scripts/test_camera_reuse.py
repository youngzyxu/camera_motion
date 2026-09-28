"""Migration must reject corrupt/stale results and avoid inference on valid reuse."""
import json,hashlib,tempfile,sys,os,shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from process_camera_lerobot import CONFIG,Queue
from reuse_camera import prepare_reuse
import lerobot_pipeline as lp

def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);video=root/'source.mp4';video.write_bytes(b'landed video')
  metadata=root/'metadata.json';metadata.write_text(json.dumps({'info':{'episode_index':1,'length':2,'capture_fps':4.}}))
  old=root/'old';old.mkdir();remote='bos:/old/result';new='bos:/new/results'
  task={'task_id':'new-task','video':str(video),'metadata':str(metadata),'episode_index':1,'source_frames_metadata':2,'source_fps_metadata':4.,'chunk':'chunk-000','camera':'ego','episode':'episode_000001','reuse_camera':{'remote':remote,'task_id':'old-task','lease_id':'old-lease'}}
  row={'video':str(video),'config':CONFIG,'include_last_frame':True,'sampling_version':CONFIG['sampling_version'],'source_frames':2,'source_fps':4.,'sampled_frames':2,'video_seconds':.5,'task':{'task_id':'old-task'}}
  np.savez_compressed(old/'camera.npz',c2w=np.tile(np.eye(4),(2,1,1)),intrinsics=np.tile(np.eye(3),(2,1,1)),frame_indices=np.array([0,1]),timestamps_s=np.array([0,.25]),source_fps=4.)
  (old/'camera.json').write_text(json.dumps(row));marker={'task_id':'old-task','lease_id':'old-lease','video':str(video),'readback_verified':True,'config':CONFIG,'sha256':{s:hashlib.sha256((old/('camera'+s)).read_bytes()).hexdigest() for s in ['.npz','.json']}}
  (old/'SUCCESS.json').write_text(json.dumps(marker));uploaded={}
  def transfer(src,dst):
   src,dst=str(src),str(dst)
   if src.startswith(remote):shutil.copyfile(old/src.rsplit('/',1)[-1],dst)
   elif src.startswith(new):Path(dst).write_bytes(uploaded[src])
   else:uploaded[dst]=Path(src).read_bytes()
  dest=root/'scratch/camera';dest.parent.mkdir()
  assert prepare_reuse(task,dest,transfer,CONFIG)['task']['task_id']=='new-task'
  # Checksum mismatch never becomes a successful migrated artifact.
  (old/'camera.json').write_text('corrupt')
  try:prepare_reuse(task,dest,transfer,CONFIG)
  except AssertionError:pass
  else:raise AssertionError('Corrupt JSON accepted')
  (old/'camera.json').write_text(json.dumps(row))
  # Valid JSON/hashes but wrong end-frame sample must also be rejected.
  original=(old/'camera.npz').read_bytes()
  np.savez_compressed(old/'camera.npz',c2w=np.tile(np.eye(4),(2,1,1)),intrinsics=np.tile(np.eye(3),(2,1,1)),frame_indices=np.array([0,0]),timestamps_s=np.array([0,0]),source_fps=4.)
  marker['sha256']['.npz']=hashlib.sha256((old/'camera.npz').read_bytes()).hexdigest();(old/'SUCCESS.json').write_text(json.dumps(marker))
  try:prepare_reuse(task,dest,transfer,CONFIG)
  except AssertionError:pass
  else:raise AssertionError('Missing last frame accepted')
  (old/'camera.npz').write_bytes(original);marker['sha256']['.npz']=hashlib.sha256(original).hexdigest();(old/'SUCCESS.json').write_text(json.dumps(marker))
  manifest=root/'manifest.jsonl';manifest.write_text(json.dumps(task)+'\n');q=Queue(root/'q.sqlite',manifest)
  def rpc(server,op,p):
   if op=='complete' and p.get('ok'):
    assert any(k.startswith(new) and k.endswith('/SUCCESS.json') for k in uploaded)
    assert p['metrics']['reused_camera'] and p['metrics']['readback_verified']
   return q.call(op,p)
  a=SimpleNamespace(stage=str(root/'stage'),seconds=0,stop_file=None,server='mock',stay_alive=False,bos_prefix=new,verify_upload=True,keep_local=False,reuse_cuda_cache=True)
  with patch.dict(os.environ,{'CAMERA_PREFETCH_LOCAL':'1'}),patch.object(lp,'rpc',rpc),patch.object(lp,'transfer',transfer),patch.object(lp,'predict',side_effect=AssertionError('Reuse incorrectly invoked inference')):
   result=lp.run(a,None)
  assert result['completed']==1 and result['failed_attempts']==0,result
  assert q.call('status',{})['counts']=={'done':1}
  data=[json.loads(v) for k,v in uploaded.items() if k.endswith('/camera.json')][0]
  assert data['task']['task_id']=='new-task' and data['reused_from']==remote
 print('PASS: checksum rejection, last-frame rejection, new namespace/identity, no inference on valid reuse, success after upload/readback')
if __name__=='__main__':main()
