import json,sys,time
from pathlib import Path
from benchmark_camera import ROOT,preprocess,predict
import torch,numpy as np
method=sys.argv[1];steps=int(sys.argv[2]) if len(sys.argv)>2 else 1;torch.set_num_threads(8)
job=json.loads((ROOT/'results/ego_manifest.json').read_text())[0]
images,_,_,_=preprocess(job['video'],2,method);images=images[:4].cuda()
if method=='omega':
 from vggt_omega.models import VGGTOmega
 from vggt_omega.utils.pose_enc import encoding_to_camera
 model=VGGTOmega();model.load_state_dict(torch.load('/mnt/pfs/pfs-yc2F4O/modelTeam/code/xuzhiyong/backup/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True));model=model.eval().cuda()
 with torch.no_grad():
  out=model(images);e,k=encoding_to_camera(out['pose_enc'],images.shape[-2:]);w=torch.eye(4,device='cuda').repeat(4,1,1);w[:,:3]=e[0];reference=torch.linalg.inv(w).cpu().numpy();ki=k[0].cpu().numpy()
else:
 from vggttt.nets.vggt.models.vggt import VGGT
 model=VGGT.from_pretrained('/mnt/pfs/pfs-yc2F4O/hf_cache/hub/models--nvidia--vgg-ttt/snapshots/2bd0869b45f919791e80cda723925be2f030fe02',local_files_only=True).eval().cuda()
 with torch.no_grad():out=model.infer(images,memory_efficient_inference=True,offload_to_cpu=False,num_ttt_steps=steps)
 reference=out['pose'].numpy();ki=out['intrinsics'].numpy()
model.benchmark_ttt_steps=steps
a,b=predict(model,images,method)
result={'method':method,'frames':4,'ttt_steps':steps if method=='ttt' else None,'max_abs_pose_difference':float(np.max(np.abs(a-reference))),'max_abs_intrinsics_difference':float(np.max(np.abs(b-ki)))}
assert np.allclose(a,reference,atol=1e-5,rtol=1e-5),result
assert np.allclose(b,ki,atol=1e-4,rtol=1e-5),result
(ROOT/(f'results/{method}'+('_steps2' if steps==2 else '')+'_branch_verification.json')).write_text(json.dumps(result,indent=2));print(result)
