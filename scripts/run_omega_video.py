#!/usr/bin/env python3
"""Bounded-frame-memory batched video pose reconstruction with overlap stitching."""
import os
if not os.environ.get('PYTORCH_ALLOC_CONF') and not os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
 os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
import argparse,json,time,gc
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from benchmark_camera import ROOT,load_model,predict
from window_trajectory import stitch
from frame_sampling import sample_indices
from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio,_max_size_target_shape

def run_video(model,video,fps,window,overlap,dest,include_last_frame=True):
 if fps<=0 or not 0<overlap<window:raise ValueError('Invalid fps/window/overlap')
 cap=cv2.VideoCapture(str(video));srcfps=cap.get(5);n=int(cap.get(7));assert srcfps>0 and n>0
 if fps>srcfps:raise ValueError('Requested fps exceeds source fps')
 idx=np.unique(np.floor(np.arange(0,n/srcfps,1/fps)*srcfps+.5).astype(np.int64));idx=idx[idx<n]
 if include_last_frame:idx=sample_indices(n,srcfps,fps)
 wanted=set(idx.tolist());last_frame=None
 buffers=[];preds=[];spans=[];seen=0;last_full_end=0;start=0;inference=0.;upload=0.;hw=None
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 def infer_buffer(buffer,start,end):
  nonlocal inference,upload,hw
  t=time.perf_counter();x=torch.stack(buffer).cuda();torch.cuda.synchronize();upload+=time.perf_counter()-t;hw=x.shape[-2:]
  t=time.perf_counter();p,k=predict(model,x,'omega');torch.cuda.synchronize();inference+=time.perf_counter()-t;preds.append((p,k));spans.append((start,end));del x
  print(f'WINDOW {video.name}: samples [{start},{end})',flush=True)
 try:
  for f in range(n):
   if not cap.grab():raise ValueError(f'Incomplete decode at {f}/{n}')
   if f not in wanted:continue
   ok,frame=cap.retrieve()
   if not ok:raise ValueError(f'Cannot retrieve frame {f}')
   if include_last_frame and f==n-1:last_frame=frame
   im=_crop_to_supported_aspect_ratio(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)));w,h=im.size;th,tw=_max_size_target_shape(h/w,512,16);buffers.append(to_tensor(im.resize((tw,th),Image.Resampling.BICUBIC)));seen+=1
   if len(buffers)==window:
    infer_buffer(buffers,start,seen);last_full_end=seen;buffers=buffers[-overlap:];start=seen-overlap
  if seen!=len(idx):raise ValueError('Missing sampled frames')
  if seen>last_full_end:infer_buffer(buffers,start,seen)
 finally:cap.release()
 t=time.perf_counter();pose,k,joins=stitch(preds,spans);stitch_s=time.perf_counter()-t;elapsed=time.perf_counter()-begin
 assert len(pose)==len(idx) and np.isfinite(pose).all()
 row={'video':str(video),'fps':fps,'window':window,'overlap':overlap,'windows':len(spans),'max_window_input_frames':max(e-s for s,e in spans),'processed_window_frames':sum(e-s for s,e in spans),'stitch_version':'orientation_sim3_irls12_blend_v1','source_fps':srcfps,'source_frames':n,'video_seconds':n/srcfps,'sampled_frames':len(idx),'inference_s':inference,'upload_and_stack_s':upload,'stitch_s':stitch_s,'end_to_end_s':elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'scale_fallbacks':sum(j['scale_fallback'] for j in joins),'joins':joins,'status':'ok','preprocessing':'max_size512','pose_convention':'OpenCV c2w; first camera origin; arbitrary translation scale'}
 dest.parent.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
 row['include_last_frame']=include_last_frame
 if include_last_frame:
  assert idx[-1]==n-1 and last_frame is not None
  tail=Path(str(dest)+'.last_frame.png')
  if not cv2.imwrite(str(tail),last_frame):raise IOError(f'Cannot write {tail}')
  row['last_frame_path']=str(tail)
 np.savez_compressed(Path(str(dest)+'.npz'),c2w=pose.astype(np.float32),intrinsics=k,frame_indices=idx,timestamps_s=idx/srcfps,input_hw=np.array(hw),source_fps=srcfps);row['save_s']=time.perf_counter()-t;Path(str(dest)+'.json').write_text(json.dumps(row,indent=2));return row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input');ap.add_argument('--manifest');ap.add_argument('--output',required=True);ap.add_argument('--config',default=str(ROOT/'configs/omega_video.json'));ap.add_argument('--fps',type=float);ap.add_argument('--overlap',type=int);ap.add_argument('--window',type=int);a=ap.parse_args()
 config=json.loads(Path(a.config).read_text()) if Path(a.config).exists() else {}
 fps=a.fps if a.fps is not None else config.get('fps');ov=a.overlap if a.overlap is not None else config.get('overlap');win=a.window or config.get('window',240)
 if fps is None or ov is None:raise ValueError('Provide selected config or explicit --fps and --overlap')
 if bool(a.input)==bool(a.manifest):raise ValueError('Provide exactly one --input or --manifest')
 if a.manifest:jobs=json.loads(Path(a.manifest).read_text())
 else:
  inp=Path(a.input);files=[inp] if inp.is_file() else sorted(inp.rglob('*.mp4'));jobs=[{'id':str(p.relative_to(inp).with_suffix('')).replace('/','__') if inp.is_dir() else p.stem,'video':str(p)} for p in files]
 torch.set_num_threads(6);torch.manual_seed(0);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 t=time.perf_counter();model=load_model('omega');torch.cuda.synchronize();load=time.perf_counter()-t;warm=torch.rand(2,3,512,512,device='cuda');predict(model,warm,'omega');del warm;torch.cuda.synchronize()
 (out/'metadata.json').write_text(json.dumps({'fps':fps,'overlap':ov,'window':win,'model_load_s':load,'gpu':str(torch.cuda.get_device_properties(0)),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'allocator':os.environ.get('PYTORCH_ALLOC_CONF'),'config':config},indent=2))
 for job in jobs:
  dest=out/job['id']
  if Path(str(dest)+'.json').exists():continue
  row=run_video(model,Path(job['video']),fps,win,ov,dest);print('DONE',job['id'],round(row['end_to_end_s'],2),flush=True);gc.collect()
if __name__=='__main__':main()
