"""Re-score all window-cache configurations with the final common stitch implementation."""
import json,time
from pathlib import Path
import numpy as np
from run_window_study import OUT,evaluate
from window_trajectory import windows,stitch

def restitch_all():
 count=0
 for path in sorted((OUT/'predictions').glob('*/*fps_ov*.json')):
  row=json.loads(path.read_text());sid=row['id'];fps=row['fps'];ov=row['overlap'];data=np.load(OUT/'data'/sid/'gt.npz');n=int(data['source_frames']);srcfps=float(data['source_fps']);idx=np.unique(np.floor(np.arange(0,n/srcfps,1/fps)*srcfps+.5).astype(np.int64));idx=idx[idx<n];spans=windows(len(idx),240,ov);pred=[]
  for start,end in spans:
   z=np.load(OUT/'window_cache'/sid/f'{fps}fps_{start}_{end}.npz');pred.append((z['c2w'],z['intrinsics']))
  t=time.perf_counter();p,k,joins=stitch(pred,spans);row['stitch_s']=time.perf_counter()-t;row['joins']=joins;row['scale_fallbacks']=sum(j['scale_fallback'] for j in joins);row['stitch_version']='orientation_sim3_irls12_blend_v1';row.update(evaluate(p,idx,data['camera_to_world'],srcfps,joins))
  np.savez_compressed(path.with_suffix('.npz'),c2w=p,intrinsics=k,frame_indices=idx,timestamps_s=idx/srcfps,input_hw=np.array([512,512]),source_fps=srcfps);path.write_text(json.dumps(row,indent=2));count+=1
 print('RESTITCHED',count)
 return count
if __name__=='__main__':restitch_all()
