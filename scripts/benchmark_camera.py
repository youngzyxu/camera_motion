#!/usr/bin/env python3
"""Single-GPU, full-sequence camera-only benchmark; no frame cap or window stitching."""
import argparse, gc, json, os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'vggt_omega_repo'), str(ROOT/'vggttt_repo')]
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

def load_model(method):
    if method == 'omega':
        from vggt_omega.models import VGGTOmega
        model = VGGTOmega()
        model.load_state_dict(torch.load(os.environ.get('CAMERA_OMEGA_WEIGHTS','/mnt/pfs/pfs-yc2F4O/modelTeam/code/xuzhiyong/backup/vggt_omega_1b_512.pt'),map_location='cpu',weights_only=True),strict=True)
        model.dense_head = None
    else:
        from vggttt.nets.vggt.models.vggt import VGGT
        model = VGGT.from_pretrained('/mnt/pfs/pfs-yc2F4O/hf_cache/hub/models--nvidia--vgg-ttt/snapshots/2bd0869b45f919791e80cda723925be2f030fe02',local_files_only=True)
        model.depth_head = model.point_head = model.track_head = None
    return model.eval().cuda()

@torch.no_grad()
def predict(model, images, method):
    h,w=images.shape[-2:]
    if method=='omega':
        from vggt_omega.utils.pose_enc import encoding_to_camera
        pred=model(images)
        extr,k=encoding_to_camera(pred['pose_enc'],(h,w))
    else:
        # Exactly the camera branch in upstream infer(), with its default one TTT
        # update pass and memory-efficient chunking. Depth/unprojection omitted.
        from vggttt.nets.ttt import TTTOperator
        from vggttt.nets.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        ops=[TTTOperator(0,None,True,False,True)]*getattr(model,"benchmark_ttt_steps",1)+[TTTOperator(0,None,False,True,False)]
        kwargs={'info':{'ttt_op_order':ops},'chunk_size':20*37*37,'track_details':False,'offload_to_cpu':False}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tokens,_,_=model.aggregator(images[None],attn_kwargs=kwargs,add_first_view_token=True,memory_efficient_inference=True)
        with torch.autocast('cuda',enabled=False):
            enc=model.camera_head(tokens[-1][:,:,0].clone().cuda(non_blocking=True))[-1]
            extr,k=pose_encoding_to_extri_intri(enc,image_size_hw=(h,w))
    w2c=torch.eye(4,device=extr.device).expand(extr.shape[1],4,4).clone()
    w2c[:,:3]=extr[0]
    c2w=torch.linalg.inv(w2c)
    return c2w.float().cpu().numpy(),k[0].float().cpu().numpy()

def preprocess(path, fps, method):
    cap=cv2.VideoCapture(str(path))
    srcfps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if srcfps<=0 or total<=0: raise ValueError(f'Invalid video: {path}')
    # Exact requested temporal sampling; do not round 30/fps to an integer stride.
    idx=np.unique(np.floor(np.arange(0,total/srcfps,1/fps)*srcfps+0.5).astype(np.int64))
    idx=idx[idx<total]; wanted=set(idx.tolist()); out=[]; decoded=0
    for i in range(total):
        ok=cap.grab()
        if not ok: break
        decoded+=1
        if i not in wanted: continue
        ok,frame=cap.retrieve()
        if not ok: raise ValueError(f'Cannot decode frame {i}')
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if method=='omega':
            from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio,_max_size_target_shape
            im=_crop_to_supported_aspect_ratio(Image.fromarray(rgb)); w,h=im.size
            th,tw=_max_size_target_shape(h/w,512,16)
            out.append(to_tensor(im.resize((tw,th),Image.Resampling.BICUBIC)))
        else:
            from vggttt.nets.vggt.img import load_and_preprocess_images
            out.append(load_and_preprocess_images([rgb])[0])
    cap.release()
    if decoded!=total or len(out)!=len(idx): raise ValueError(f'Incomplete decode {decoded}/{total}, samples {len(out)}/{len(idx)}')
    return torch.stack(out),idx,srcfps,total

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['omega','ttt'],required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--fps',type=float,nargs='+',default=[2,4,8,16]);p.add_argument('--limit',type=int);p.add_argument('--ttt-steps',type=int,default=1);args=p.parse_args()
    torch.set_num_threads(8);torch.manual_seed(0);np.random.seed(0)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(Path(args.manifest).read_text())
    if args.limit: jobs=jobs[:args.limit]
    t=time.perf_counter();model=load_model(args.method);model.benchmark_ttt_steps=args.ttt_steps;torch.cuda.synchronize();load_s=time.perf_counter()-t
    meta={'method':args.method,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_properties':str(torch.cuda.get_device_properties(0)),'model_load_s':load_s,'mode':'camera_only_full_sequence','preprocessing':'max_size512' if args.method=='omega' else 'upstream_crop518','ttt_steps':args.ttt_steps if args.method=='ttt' else None,'offload_to_cpu':False,'repeats':1}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
    # Warm up independently of each measured video. Startup reported separately.
    warm=torch.rand(2,3,512 if args.method=='omega' else 518,512 if args.method=='omega' else 518,device='cuda')
    t=time.perf_counter();predict(model,warm,args.method);torch.cuda.synchronize();meta['warmup_s']=time.perf_counter()-t;del warm
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    for fps in args.fps:
      for job in jobs:
        key=f"{job['id']}__{fps:g}fps";record_path=out/(key+'.json')
        if record_path.exists(): continue
        row={'id':job['id'],'video':job['video'],'method':args.method,'fps':fps,'status':'running'}
        images=None;gpu_images=None;poses=None;k=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        try:
            images,idx,srcfps,total=preprocess(job['video'],fps,args.method)
            row.update(n_frames=len(idx),source_fps=srcfps,source_frames=total,video_seconds=total/srcfps,input_hw=list(images.shape[-2:]),preprocess_s=time.perf_counter()-start)
            print('START '+key+' '+str(images.shape),flush=True)
            t=time.perf_counter();gpu_images=images.cuda();torch.cuda.synchronize();row['h2d_s']=time.perf_counter()-t
            t=time.perf_counter();poses,k=predict(model,gpu_images,args.method);torch.cuda.synchronize();row['inference_s']=time.perf_counter()-t
            if not np.isfinite(poses).all() or not np.isfinite(k).all(): raise ValueError('Nonfinite prediction')
            row['end_to_end_s']=time.perf_counter()-start
            t=time.perf_counter();np.savez_compressed(out/(key+'.npz'),c2w=poses,intrinsics=k,frame_indices=idx,timestamps_s=idx/srcfps,input_hw=np.array(images.shape[-2:]),source_fps=srcfps)
            row.update(save_s=time.perf_counter()-t,status='ok')
        except torch.cuda.OutOfMemoryError as e:
            row.update(status='oom',error=str(e),elapsed_to_failure_s=time.perf_counter()-start)
        except Exception as e:
            row.update(status='error',error=traceback.format_exc(),elapsed_to_failure_s=time.perf_counter()-start)
        row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        row['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
        record_path.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
        del images,gpu_images,poses,k;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
