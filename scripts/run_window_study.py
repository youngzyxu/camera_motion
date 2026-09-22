"""Full HOT3D 2/4/8fps x overlap24/36/48, disjoint per-sequence GPU shards."""
import argparse,json,time,gc,os
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation,Slerp
from benchmark_camera import ROOT,load_model,predict
from window_trajectory import windows,stitch
from evaluate_camera import metrics,fit_sim3,angle
OUT=ROOT/'results/omega_window_study'

def densify(p,idx,end):
 ts=np.arange(end+1);out=np.tile(np.eye(4),(len(ts),1,1));out[:,:3,:3]=Slerp(idx,Rotation.from_matrix(p[:,:3,:3]))(ts).as_matrix()
 for d in range(3):out[:,d,3]=np.interp(ts,idx,p[:,d,3])
 return out

def evaluate(p,idx,gt,srcfps,joins):
 end=int((len(gt)-1)//15*15);dense=densify(p,idx,end);truth=gt[:end+1];m=metrics(dense,truth,np.arange(end+1),srcfps)
 # Forward-only evaluation alignment uses the same first 10 seconds at every fps.
 anchor=min(len(dense),int(srcfps*10)+1);s,r,t=fit_sim3(dense[:anchor,:3,3],truth[:anchor,:3,3]);pos=s*(dense[:,:3,3]@r.T)+t;err=np.linalg.norm(pos-truth[:,:3,3],axis=1)
 m.update(anchor10_ate_rmse_m=float(np.sqrt(np.mean(err**2))),anchor10_tail15_rmse_m=float(np.sqrt(np.mean(err[-int(srcfps*15):]**2))),anchor10_endpoint_m=float(err[-1]))
 dt=int(srcfps);dp=np.linalg.inv(dense[:-dt])@dense[dt:];dg=np.linalg.inv(truth[:-dt])@truth[dt:];rr=angle(dp[:,:3,:3]@dg[:,:3,:3].transpose(0,2,1));rt=np.linalg.norm(m['sim3_scale']*dp[:,:3,3]-dg[:,:3,3],axis=1)
 mask=np.zeros(len(rr),bool)
 for j in joins:
  for b in [idx[j['start']],idx[j['overlap_end']-1]]:mask|=abs(np.arange(len(rr))+dt/2-b)<=dt
 m.update(seam_pairs=int(mask.sum()),seam_rotation_p95_deg=float(np.percentile(rr[mask],95)) if mask.any() else None,seam_translation_rmse_m=float(np.sqrt(np.mean(rt[mask]**2))) if mask.any() else None,dense_end_frame=end)
 onep=np.linalg.inv(dense[:-1])@dense[1:];oneg=np.linalg.inv(truth[:-1])@truth[1:]
 one_rot=angle(onep[:,:3,:3]@oneg[:,:3,:3].transpose(0,2,1));one_trans=np.linalg.norm(m['sim3_scale']*onep[:,:3,3]-oneg[:,:3,3],axis=1);one_mask=np.zeros(len(one_rot),bool)
 for j in joins:
  for b in [idx[j['start']],idx[j['overlap_end']-1]]:one_mask|=abs(np.arange(len(one_rot))+.5-b)<=dt
 m.update(seam_step_pairs=int(one_mask.sum()),seam_step_rot_p95_deg=float(np.percentile(one_rot[one_mask],95)) if one_mask.any() else None,seam_step_trans_rmse_m=float(np.sqrt(np.mean(one_trans[one_mask]**2))) if one_mask.any() else None)
 speed=angle(dg[:,:3,:3]);fast=speed>=np.percentile(speed,90);m['fast_motion_rpe_rot_deg']=float(rr[fast].mean())
 return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=7);a=ap.parse_args()
 torch.set_num_threads(6);torch.manual_seed(0);np.random.seed(0);model=load_model('omega');warm=torch.rand(2,3,512,512,device='cuda');predict(model,warm,'omega');del warm;torch.cuda.synchronize()
 meta={'shard':a.shard,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'gpu':str(torch.cuda.get_device_properties(0)),'window':240,'overlaps':[24,36,48],'fps':[2,4,8],'allocator':os.environ.get('PYTORCH_ALLOC_CONF'),'alignment':'orientation-constrained robust Sim3; translation blend and SO3 interpolation; no GT used in stitching','timing':'sum of measured cached unique window inference times; preprocessing excluded; window cache reused across overlaps'}
 (OUT/f'worker_{a.shard}.json').write_text(json.dumps(meta,indent=2))
 jobs=json.loads((ROOT/'results/hot3d_manifest.json').read_text())[a.shard::a.shards]
 for j in jobs:
  sid=j['id'];data=OUT/'data'/sid
  while not (data/'ready.json').exists():time.sleep(2)
  z=np.load(data/'gt.npz');gt=z['camera_to_world'];n=int(z['source_frames']);srcfps=float(z['source_fps']);master=z['frame_indices_8fps'];rgb=np.load(data/'rgb8.npy',mmap_mode='r');dest=OUT/'predictions'/sid;dest.mkdir(parents=True,exist_ok=True);cache=OUT/'window_cache'/sid;cache.mkdir(parents=True,exist_ok=True)
  for fps in [2,4,8]:
   idx=np.unique(np.floor(np.arange(0,n/srcfps,1/fps)*srcfps+.5).astype(np.int64));idx=idx[idx<n];mapping=np.searchsorted(master,idx);assert np.array_equal(master[mapping],idx)
   for ov in [24,36,48]:
    key=f'{fps}fps_ov{ov}';rec=dest/(key+'.json')
    if rec.exists():continue
    spans=windows(len(idx),240,ov);predictions=[];inference=0;peak=0;transfer=0
    for start,end in spans:
     name=f'{fps}fps_{start}_{end}';f=cache/(name+'.npz');jf=cache/(name+'.json')
     if not jf.exists():
      gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
      cpu=torch.from_numpy(np.asarray(rgb[mapping[start:end]]).copy()).permute(0,3,1,2).contiguous();x=cpu.cuda().float().div_(255);torch.cuda.synchronize();h2d=time.perf_counter()-t;t=time.perf_counter();pose,k=predict(model,x,'omega');torch.cuda.synchronize();sec=time.perf_counter()-t;mem=torch.cuda.max_memory_allocated()/2**30
      np.savez_compressed(f,c2w=pose,intrinsics=k);jf.write_text(json.dumps({'inference_s':sec,'input_transfer_s':h2d,'peak_allocated_gib':mem,'frames':end-start}));del cpu,x
     w=np.load(f);predictions.append((w['c2w'],w['intrinsics']));info=json.loads(jf.read_text());inference+=info['inference_s'];transfer+=info['input_transfer_s'];peak=max(peak,info['peak_allocated_gib'])
    t=time.perf_counter();pose,k,joins=stitch(predictions,spans);stitch_s=time.perf_counter()-t
    assert np.isfinite(pose).all();m=evaluate(pose,idx,gt,srcfps,joins)
    row={'id':sid,'fps':fps,'overlap':ov,'window':240,'sampled_frames':len(idx),'video_seconds':n/srcfps,'windows':len(spans),'inference_s':inference,'input_transfer_s':transfer,'stitch_s':stitch_s,'peak_allocated_gib':peak,'scale_fallbacks':sum(j['scale_fallback'] for j in joins),'joins':joins,**m}
    np.savez_compressed(dest/(key+'.npz'),c2w=pose,intrinsics=k,frame_indices=idx,timestamps_s=idx/srcfps,input_hw=np.array([512,512]),source_fps=srcfps)
    rec.write_text(json.dumps(row,indent=2));print('DONE',sid,key,'windows',len(spans),'ATE_cm',round(m['ate_sim3_rmse_m']*100,3),'inference_s',round(inference,2),flush=True)
  del rgb;gc.collect()
 print('WORKER_COMPLETE',a.shard,flush=True)
if __name__=='__main__':main()
