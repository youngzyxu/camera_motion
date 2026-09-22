"""Audit fixed-cohort A/B results, including numerical equivalence and BOS hashes."""
import json,sqlite3,hashlib,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/lerobot_pipeline_ab'

def main():
 allrows={};phases={};locations={};audit={'passed':True,'checked_npz_pairs':0,'bitwise_equal_npz_arrays':0,'max_pose_abs_difference':0.,'max_intrinsics_abs_difference':0.,'BOS_readback_verified_files':0}
 for mode in ['serial','pipeline']:
  db=sqlite3.connect(OUT/f'{mode}.sqlite');states=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'));records=[json.loads(r[0])['metrics'] for r in db.execute("SELECT result FROM tasks WHERE status='done'")];db.close()
  assert states=={'done':2048},states
  maps={}
  for gpu in range(8):
   for line in (OUT/mode/f'gpu{gpu}/completed.jsonl').read_text().splitlines():
    r=json.loads(line);maps[r['task']['task_id']]=(gpu,r)
  assert len(maps)==2048
  allrows[mode]=maps
  timing=json.loads((OUT/f'{mode}_timing.json').read_text());video=sum(r['video_seconds'] for r in records)
  phases[mode]={**timing,'videos':len(records),'video_s':video,'video_hours_per_machine_day':video/timing['wall_s']*24,'video_hours_per_gpu_day_average':video/timing['wall_s']*3,'inference_worker_s':sum(r['inference_s'] for r in records),'upload_worker_s':sum(r['upload_s'] for r in records),'peak_worker_allocated_gib':max(r['peak_allocated_gib'] for r in records),'attempts_over_one':0}
  db=sqlite3.connect(OUT/f'{mode}.sqlite');phases[mode]['attempts_over_one']=db.execute('SELECT count(*) FROM tasks WHERE attempts>1').fetchone()[0];db.close()
  locations[mode]={}
  for taskid,(gpu,r) in maps.items():
   d=OUT/mode/f'gpu{gpu}/tasks'/taskid/r['task']['lease_id'];locations[mode][taskid]=d
   marker=json.loads((d/'SUCCESS.json').read_text());assert marker['readback_verified'] and r['readback_verified']
   for suffix in ['.npz','.json']:
    assert hashlib.sha256((d/('camera'+suffix)).read_bytes()).hexdigest()==marker['sha256'][suffix];audit['BOS_readback_verified_files']+=1
 flat=[]
 assert set(allrows['serial'])==set(allrows['pipeline'])
 for taskid in allrows['serial']:
  a=np.load(locations['serial'][taskid]/'camera.npz');b=np.load(locations['pipeline'][taskid]/'camera.npz')
  assert set(a.files)==set(b.files)
  for k in a.files:assert np.isfinite(b[k]).all() and a[k].shape==b[k].shape
  assert np.array_equal(a['frame_indices'],b['frame_indices']) and np.array_equal(a['timestamps_s'],b['timestamps_s'])
  p=b['c2w'];assert np.allclose(p[0],np.eye(4),atol=1e-5);rot=p[:,:3,:3];assert np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-5)
  error=float(np.max(abs(a['c2w']-b['c2w'])));kerror=float(np.max(abs(a['intrinsics']-b['intrinsics'])));audit['checked_npz_pairs']+=1;audit['bitwise_equal_npz_arrays']+=int(all(np.array_equal(a[k],b[k]) for k in a.files));audit['max_pose_abs_difference']=max(audit['max_pose_abs_difference'],error);audit['max_intrinsics_abs_difference']=max(audit['max_intrinsics_abs_difference'],kerror)
  assert np.allclose(a['c2w'],b['c2w'],atol=1e-5,rtol=1e-5) and np.allclose(a['intrinsics'],b['intrinsics'],atol=1e-5,rtol=1e-5)
  sr=allrows['serial'][taskid][1];pr=allrows['pipeline'][taskid][1];assert sr['sampled_frames']==pr['sampled_frames'] and sr['windows']==pr['windows'] and sr['video_seconds']==pr['video_seconds']
  flat.append({'task_id':taskid,'video_seconds':sr['video_seconds'],'serial_inference_s':sr['inference_s'],'pipeline_inference_s':pr['inference_s'],'pipeline_decode_s':pr['decode_preprocess_s'],'serial_upload_s':sr['upload_s'],'pipeline_upload_s':pr['upload_s'],'pose_max_abs_difference':error,'intrinsics_max_abs_difference':kerror,'serial_remote':sr['remote'],'pipeline_remote':pr['remote']})
 summary={'config':{'fps':4,'window':240,'overlap':36},'phases':phases,'speedup':phases['serial']['wall_s']/phases['pipeline']['wall_s'],'wall_time_reduction_fraction':1-phases['pipeline']['wall_s']/phases['serial']['wall_s'],'audit':audit,'scope':'same fixed 2048 videos, 8 resident models, serial then pipeline; startup excluded, fill/drain and all BOS readback checks included'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
 with (OUT/'paired.csv').open('w') as f:w=csv.DictWriter(f,list(flat[0]));w.writeheader();w.writerows(flat)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
