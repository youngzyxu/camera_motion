"""GT-free overlap registration and blending for camera-to-world windows."""
import numpy as np
from scipy.spatial.transform import Rotation

def windows(n,size=240,overlap=36):
 if not 0<overlap<size:raise ValueError('Require 0 < overlap < window')
 out=[];start=0
 while True:
  end=min(start+size,n);out.append((start,end))
  if end==n:return out
  start=end-overlap

def align_overlap(source,target,prior_scale=1.):
 """Use camera orientations to constrain rotation; robust LS estimates positive scale."""
 x=source[:,:3,3];y=target[:,:3,3];rels=target[:,:3,:3]@source[:,:3,:3].transpose(0,2,1)
 weights=np.ones(len(x));fallback=False
 for _ in range(12):
  weights=weights/weights.sum();r=Rotation.from_matrix(rels).mean(weights=weights).as_matrix()
  xm=np.sum(weights[:,None]*x,0);ym=np.sum(weights[:,None]*y,0);xc=x-xm;yc=y-ym
  var=np.sum(weights*np.sum(xc*xc,1));candidate=np.sum(weights*np.sum((xc@r.T)*yc,1))/max(var,1e-30)
  # Model-gauge threshold only handles near-zero motion; flag all fallback cases.
  fallback=var<1e-8 or candidate<=0 or not np.isfinite(candidate)
  s=prior_scale if fallback else candidate;t=ym-s*r@xm
  poserr=np.linalg.norm(s*(x@r.T)+t-y,axis=1);roterr=Rotation.from_matrix(rels@r.T).magnitude()
  score=poserr/max(np.median(poserr),1e-7)+roterr/max(np.median(roterr),1e-5)
  weights=np.minimum(1,3/np.maximum(score,1e-10))
 aligned=source.copy();aligned[:,:3,:3]=r@source[:,:3,:3];aligned[:,:3,3]=s*(source[:,:3,3]@r.T)+t
 return s,r,t,{'scale':float(s),'scale_fallback':bool(fallback),'overlap_position_rmse_gauge':float(np.sqrt(np.mean(poserr**2))),'overlap_rotation_mean_deg':float(np.degrees(roterr).mean()),'effective_pairs':float(weights.sum()**2/np.sum(weights**2)),'overlap_center_variance':float(var)}

def stitch(predictions,ranges):
 n=ranges[-1][1];out=np.empty((n,4,4));K=np.empty((n,3,3));logs=[];prior=1.
 for wi,((pose,ki),(start,end)) in enumerate(zip(predictions,ranges)):
  pose=pose.astype(np.float64)
  if wi==0:out[start:end]=pose;K[start:end]=ki;continue
  previous_end=ranges[wi-1][1];overlap=previous_end-start
  s,r,t,log=align_overlap(pose[:overlap],out[start:previous_end],prior);prior=s
  pose[:,:3,:3]=r@pose[:,:3,:3];pose[:,:3,3]=s*(pose[:,:3,3]@r.T)+t
  a=np.linspace(0,1,overlap);old=out[start:previous_end].copy()
  rel=old[:,:3,:3].transpose(0,2,1)@pose[:overlap,:3,:3]
  pose[:overlap,:3,:3]=old[:,:3,:3]@Rotation.from_rotvec(Rotation.from_matrix(rel).as_rotvec()*a[:,None]).as_matrix()
  pose[:overlap,:3,3]=(1-a[:,None])*old[:,:3,3]+a[:,None]*pose[:overlap,:3,3]
  ki=ki.copy();ki[:overlap]=(1-a[:,None,None])*K[start:previous_end]+a[:,None,None]*ki[:overlap]
  out[start:end]=pose;K[start:end]=ki;logs.append({'start':start,'overlap_end':previous_end,'end':end,**log})
 out=np.linalg.inv(out[0])@out
 return out,K,logs

def self_test():
 rng=np.random.default_rng(6);n=900;gt=np.tile(np.eye(4),(n,1,1));t=np.arange(n)/100;gt[:,:3,3]=np.stack([t,np.sin(t),np.cos(t*.3)],1);gt[:,:3,:3]=Rotation.from_rotvec(np.stack([t*.01,t*.03,t*.1],1)).as_matrix()
 for overlap in [24,36,48]:
  spans=windows(n,240,overlap);ps=[]
  for start,end in spans:
   r=Rotation.random(random_state=rng).as_matrix();s=rng.uniform(.2,3);tr=rng.normal(size=3);p=gt[start:end].copy();p[:,:3,:3]=r@p[:,:3,:3];p[:,:3,3]=s*(p[:,:3,3]@r.T)+tr;ps.append((p,np.tile(np.eye(3),(len(p),1,1))))
  joined,_,logs=stitch(ps,spans)
  # First window defines arbitrary global scale, estimate that one scale only.
  expected=np.linalg.inv(gt[0])@gt;scale=np.linalg.norm(joined[-1,:3,3])/np.linalg.norm(expected[-1,:3,3]);expected[:,:3,3]*=scale
  assert np.allclose(joined,expected,atol=1e-8),np.max(abs(joined-expected));assert len(joined)==n
 line=np.tile(np.eye(4),(48,1,1));line[:,0,3]=np.arange(48)/10
 r=Rotation.from_euler('xyz',[.1,.3,-.2]).as_matrix();target=line.copy();target[:,:3,:3]=r;target[:,:3,3]=2*(line[:,:3,3]@r.T)+[1,3,4];target[:4,:3,3]+=100
 s,rfit,t,log=align_overlap(line,target)
 assert abs(s-2)<.01 and np.linalg.norm(t-[1,3,4])<.03 and np.allclose(rfit,r,atol=1e-8)
 static=np.tile(np.eye(4),(24,1,1));assert align_overlap(static,static)[3]['scale_fallback']
 print('Window coverage, Sim3 gauges, rotation blend, and static fallback tests passed')
if __name__=='__main__':self_test()
