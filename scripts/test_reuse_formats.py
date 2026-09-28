"""Legacy metadata never bypasses core config or exact NPZ sampling validation."""
import hashlib,json,tempfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from process_camera_lerobot import CONFIG
from reuse_camera import prepare_reuse

def main():
 with tempfile.TemporaryDirectory() as d:
  r=Path(d);video=r/'source.mp4';video.write_bytes(b'video');meta=r/'metadata.json';meta.write_text(json.dumps({'info':{'episode_index':7,'length':9,'capture_fps':10.}}))
  task={'task_id':'new','video':str(video),'metadata':str(meta),'episode_index':7,'source_frames_metadata':9,'source_fps_metadata':10.}
  old={k:v for k,v in CONFIG.items() if k not in ('include_last_frame','sampling_version')}
  row={'video':str(video),'episode_index':7,'config':old,'source_frames':9,'source_fps':10.,'sampled_frames':4,'video_seconds':.9,'fps':4,'window':240,'overlap':36,'stitch_version':CONFIG['stitch'],'preprocessing':'max_size512'}
  def payload(indices):
   np.savez_compressed(r/'camera.npz',frame_indices=np.array(indices),timestamps_s=np.array(indices)/10,c2w=np.tile(np.eye(4),(4,1,1)),intrinsics=np.tile(np.eye(3),(4,1,1)),source_fps=10.)
  payload([0,3,5,8]);(r/'camera.json').write_text(json.dumps(row))
  def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
  def marker():
   m={'task_id':'old','lease_id':'lease','video':str(video),'readback_verified':True,'config':old,'sha256':{s:digest(r/('camera'+s)) for s in ('.npz','.json')}}
   (r/'SUCCESS.json').write_text(json.dumps(m));return m
  marker();task['reuse_camera']={'remote':'bos:/fixture','task_id':'old','lease_id':'lease','allow_legacy_sampling_metadata':True}
  def transfer(src,dst):Path(dst).write_bytes((r/str(src).rsplit('/',1)[-1]).read_bytes())
  dest=r/'out/camera';dest.parent.mkdir()
  result=prepare_reuse(task,dest,transfer,CONFIG)
  assert result['config']==CONFIG and result['include_last_frame'] and result['original_config']==old
  assert (r/'camera.npz').read_bytes()==Path(str(dest)+'.npz').read_bytes()
  payload([0,3,5,7]);marker()
  try:prepare_reuse(task,dest,transfer,CONFIG)
  except AssertionError:pass
  else:raise AssertionError('Legacy missing last frame accepted')
  payload([0,3,5,8]);marker();old['overlap']=24;marker()
  try:prepare_reuse(task,dest,transfer,CONFIG)
  except AssertionError:pass
  else:raise AssertionError('Legacy wrong overlap accepted')
  old['overlap']=36;marker()
  # Independently published sample artifacts have a hashed batch receipt.
  row.pop('config');row['include_last_frame']=True;(r/'camera.json').write_text(json.dumps(row))
  complete={'status':'complete','all_last_frames_preserved':True,'count':1,'artifacts_published_and_sha256_verified':3,**{k:CONFIG[k] for k in ('model','fps','window','overlap')}}
  (r/'COMPLETE.json').write_text(json.dumps(complete))
  task['reuse_camera']={'kind':'published_sample','npz_uri':'bos:/fixture/camera.npz','json_uri':'bos:/fixture/camera.json','catalog_complete_uri':'bos:/fixture/COMPLETE.json','catalog_complete_sha256':digest(r/'COMPLETE.json'),'sha256':{s:digest(r/('camera'+s)) for s in ('.npz','.json')}}
  result=prepare_reuse(task,dest,transfer,CONFIG);assert result['reuse_kind']=='published_sample' and result['task']['task_id']=='new'
  complete['model']='wrong-model';(r/'COMPLETE.json').write_text(json.dumps(complete));task['reuse_camera']['catalog_complete_sha256']=digest(r/'COMPLETE.json')
  try:prepare_reuse(task,dest,transfer,CONFIG)
  except AssertionError:pass
  else:raise AssertionError('Wrong model sample accepted')
 print('PASS legacy exact-array reuse, missing-last rejection, core-config rejection, published sample proof and model checks')
if __name__=='__main__':main()
