#!/usr/bin/env python3
"""Normalize sampled OpenCV poses to frame 0; optional time interpolation."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

def main():
 p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');p.add_argument('--fps',type=float);p.add_argument('--translation-scale',type=float,default=1.0);a=p.parse_args()
 z=np.load(a.input);ts=z['timestamps_s'];poses=z['c2w'].astype(np.float64);k=z['intrinsics']
 poses=np.linalg.inv(poses[0])@poses;poses[:,:3,3]*=a.translation_scale
 if a.fps:
  if a.fps<=0:raise ValueError('fps must be positive')
  # Never extrapolate past the final measured pose.
  new_ts=np.arange(ts[0],ts[-1]+1e-9,1/a.fps);new=np.tile(np.eye(4),(len(new_ts),1,1))
  new[:,:3,:3]=Slerp(ts,Rotation.from_matrix(poses[:,:3,:3]))(new_ts).as_matrix()
  for d in range(3):new[:,d,3]=np.interp(new_ts,ts,poses[:,d,3])
  new_k=np.empty((len(new_ts),3,3))
  for i in range(3):
   for j in range(3):new_k[:,i,j]=np.interp(new_ts,ts,k[:,i,j])
  poses,k,ts=new,new_k,new_ts
 np.savez_compressed(a.output,c2w=poses.astype(np.float32),w2c=np.linalg.inv(poses).astype(np.float32),intrinsics=k.astype(np.float32),timestamps_s=ts,input_hw=z['input_hw'],translation_scale=a.translation_scale,coordinate_convention='OpenCV: x right, y down, z forward; first camera is world origin',scale_unit='arbitrary unless externally calibrated')
 print(f'Wrote {len(ts)} poses to {a.output}; interval [{ts[0]:.3f}, {ts[-1]:.3f}] s')
if __name__=='__main__':main()
