#!/usr/bin/env python3
"""Evaluate c2w trajectories with one sequence-level Sim(3), and 1s RPE."""
import argparse,json
from pathlib import Path
import numpy as np

def fit_sim3(x,y):
    xc=x-x.mean(0);yc=y-y.mean(0)
    u,d,vt=np.linalg.svd(yc.T@xc/len(x));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
    r=(u*sign)@vt
    var=np.mean(np.sum(xc*xc,axis=1))
    if var<1e-12: raise ValueError('Degenerate predicted trajectory: zero translation variance')
    s=float(np.dot(d,sign)/var);t=y.mean(0)-s*r@x.mean(0)
    return s,r,t

def angle(r):
    return np.degrees(np.arccos(np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1)))

def metrics(pred,gt,idx,srcfps):
    pred=np.asarray(pred,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64)
    if pred.shape!=gt.shape or not np.isfinite(pred).all() or not np.isfinite(gt).all():raise ValueError('Invalid poses')
    s,r,t=fit_sim3(pred[:,:3,3],gt[:,:3,3]);aligned=s*(pred[:,:3,3]@r.T)+t
    errors=np.linalg.norm(aligned-gt[:,:3,3],axis=1)
    rot=angle((r@pred[:,:3,:3])@gt[:,:3,:3].transpose(0,2,1))
    pairs=[]
    for i,f in enumerate(idx):
        j=int(np.searchsorted(idx,f+srcfps)); candidates=[k for k in [j-1,j] if i<k<len(idx)]
        if not candidates:continue
        j=min(candidates,key=lambda j:abs(idx[j]-f-srcfps))
        if abs((idx[j]-f)/srcfps-1)<=0.5/srcfps+1e-8:pairs.append((i,j))
    if not pairs: raise ValueError('No 1-second RPE pairs')
    ii,jj=np.array(pairs).T
    dp=np.linalg.inv(pred[ii])@pred[jj];dg=np.linalg.inv(gt[ii])@gt[jj]
    # Relative rotation error is alignment independent; translations use one global scale.
    rr=angle(dp[:,:3,:3]@dg[:,:3,:3].transpose(0,2,1))
    rt=np.linalg.norm(s*dp[:,:3,3]-dg[:,:3,3],axis=1)
    path=float(np.linalg.norm(np.diff(gt[:,:3,3],axis=0),axis=1).sum())
    return {'n_frames':len(pred),'ate_sim3_rmse_m':float(np.sqrt(np.mean(errors**2))),'ate_sim3_median_m':float(np.median(errors)), 'rotation_sim3_mean_deg':float(rot.mean()),'rpe_1s_rot_mean_deg':float(rr.mean()),'rpe_1s_trans_rmse_m':float(np.sqrt(np.mean(rt**2))),'rpe_1s_pairs':len(ii),'sim3_scale':s,'gt_path_length_m':path,'endpoint_vector_error_m':float(np.linalg.norm((aligned[-1]-aligned[0])-(gt[-1,:3,3]-gt[0,:3,3]))),'ate_squared_sum':float(np.sum(errors**2)),'rpe_translation_squared_sum':float(np.sum(rt**2)),'rpe_rotation_sum':float(rr.sum())}

def self_test():
    rng=np.random.default_rng(7);gt=np.tile(np.eye(4),(100,1,1));gt[:,:3,3]=rng.normal(size=(100,3))
    a=.7;r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    pred=gt.copy();pred[:,:3,:3]=r;pred[:,:3,3]=2*(gt[:,:3,3]@r.T)+[3,5,-2]
    m=metrics(pred,gt,np.arange(100),10)
    assert m['ate_sim3_rmse_m']<1e-10 and m['rpe_1s_trans_rmse_m']<1e-10 and m['rpe_1s_rot_mean_deg']<1e-5,m
    pred[50,0,3]+=.5;m=metrics(pred,gt,np.arange(100),10);assert m['ate_sim3_rmse_m']>.01
    print('Synthetic Sim3/RPE checks passed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--predictions');p.add_argument('--output');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return
    rows=[];missing=[]
    for job in json.loads(Path(a.manifest).read_text()):
      files=sorted(Path(a.predictions).glob(job['id']+'__*fps.npz'))
      if not files:missing.append(job['id'])
      for f in files:
        z=np.load(f);gt=np.load(job['gt'])['camera_to_world'];idx=z['frame_indices']
        if idx[-1]>=len(gt):raise ValueError('GT/frame index mismatch')
        m=metrics(z['c2w'],gt[idx],idx,float(z['source_fps']));m.update(id=job['id'],prediction=str(f),fps=float(f.stem.split('__')[-1][:-3]));rows.append(m)
    summary={}
    for fps in sorted(set(r['fps'] for r in rows)):
        rs=[r for r in rows if r['fps']==fps];n=sum(r['n_frames'] for r in rs);pairs=sum(r['rpe_1s_pairs'] for r in rs)
        summary[str(fps)]={'sequences':len(rs),'frames':n,'ate_sim3_micro_rmse_m':float(np.sqrt(sum(r['ate_squared_sum'] for r in rs)/n)),'ate_sim3_macro_rmse_m':float(np.mean([r['ate_sim3_rmse_m'] for r in rs])),'rpe_1s_rot_mean_deg':sum(r['rpe_rotation_sum'] for r in rs)/pairs,'rpe_1s_trans_rmse_m':float(np.sqrt(sum(r['rpe_translation_squared_sum'] for r in rs)/pairs))}
    report={'summary':summary,'missing':missing,'per_sequence':rows,'protocol':'One full-sequence Sim3 from predicted/GT camera centers; RPE at 1 sec; scale arbitrary before evaluation; GT never supplied to model.'}
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({'summary':summary,'missing':missing},indent=2))
if __name__=='__main__':main()
