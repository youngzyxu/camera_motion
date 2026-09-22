"""Worst supported square window, using production inference and precision."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from benchmark_camera import load_model,predict
import numpy as np
torch.set_num_threads(4);torch.manual_seed(0)
m=load_model('omega');predict(m,torch.rand(2,3,512,512,device='cuda'),'omega')
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
p,k=predict(m,torch.rand(240,3,512,512,device='cuda'),'omega');torch.cuda.synchronize()
r={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'frames':len(p),'finite':bool(np.isfinite(p).all() and np.isfinite(k).all()),'seconds':time.monotonic()-t,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
print(json.dumps(r),flush=True)
assert r['finite'] and len(p)==240
