"""Prepare full continuous HOT3D sequences with official camera GT and sampled rectified RGB."""
import sys,json,csv,subprocess,hashlib,concurrent.futures
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];BENCH=ROOT/'hand_pose_bench/EgoHand4D-Bench'
sys.path[:0]=[str(BENCH/'scripts'),str(BENCH/'src')]
from build_benchmark_viewer_assets import project_fisheye624,rotate_camera_xy,rotate_points_clockwise
OUT=ROOT/'results/omega_window_study/data'
def prepare(j):
 sid=j['id'];dest=OUT/sid;dest.mkdir(exist_ok=True)
 if (dest/'ready.json').exists():return sid
 root=Path('/mnt/pfs/pfs-yc2F4O/guangxian/Hot3DAria_downloads')/sid
 video=next(root.glob('*preview_rgb.mp4'));cap=cv2.VideoCapture(str(video));n=int(cap.get(7));fps=cap.get(5);assert fps==30
 desc=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags=description','-of','json',str(video)]))
 device=np.array(json.loads(desc['format']['tags']['description']),dtype=np.int64);assert len(device)==n
 rows=list(csv.DictReader(open(root/'timecode_devicetime_mapping.csv')));dev=np.array([int(r['devicetime_ns']) for r in rows]);tc=np.array([int(r['timecode_ns']) for r in rows]);order=np.argsort(dev);dev=dev[order];tc=tc[order]
 k=np.clip(np.searchsorted(dev,device),1,len(dev)-1);k-=abs(dev[k-1]-device)<abs(dev[k]-device);delta=int(max(abs(dev[k]-device)));assert delta<=2, (sid,delta)
 timestamps=tc[k];assert np.all(np.diff(timestamps)>0)
 head={int(r['timestamp[ns]']):r for r in csv.DictReader(open(root/'headset_trajectory.csv'))};T=np.tile(np.eye(4),(n,1,1))
 for i,t in enumerate(timestamps):
  r=head[int(t)];T[i,:3,:3]=Rotation.from_quat([float(r['q_wo_'+k]) for k in ['x','y','z','w']]).as_matrix();T[i,:3,3]=[float(r[f't_wo_{k}[m]']) for k in 'xyz']
 camera=next(c for c in json.loads((root/'camera_models.json').read_text()) if c['label']=='camera-rgb');cal=camera['T_Device_Camera'];q0=cal['quaternion_wxyz'];dc=np.eye(4);dc[:3,:3]=Rotation.from_quat(q0[1:]+q0[:1]).as_matrix();dc[:3,3]=cal['translation_xyz']
 oldc=np.load(Path(j['gt']).parent.parent/'hot3d_raw_v5_camera_native'/f'{sid}.npz');q=int(oldc['clockwise_quarters']);rot=np.eye(4);rot[:3,:3]=Rotation.from_euler('z',-q*90,degrees=True).as_matrix();gt=T@dc@rot
 old=np.load(j['gt'])['camera_to_world'];err=float(np.max(abs(gt[:1800]-old)));assert err<1e-8,err;assert np.array_equal(timestamps[:1800],oldc['timestamps'])
 params=np.array(camera['projectionParams']);h=w=1408;y,x=np.mgrid[:h,:w];rays=np.stack([(x-(w-1)/2)/params[0],(y-(h-1)/2)/params[0],np.ones_like(x)],-1)
 mapping=rotate_points_clockwise(project_fisheye624(rotate_camera_xy(rays,-q),params),q,w,h).astype(np.float32)
 idx=np.unique(np.floor(np.arange(0,n/fps,1/8)*fps+.5).astype(np.int64));idx=idx[idx<n];lookup={int(f):i for i,f in enumerate(idx)}
 images=np.lib.format.open_memmap(dest/'rgb8.npy',mode='w+',dtype=np.uint8,shape=(len(idx),512,512,3));cv2.setNumThreads(1);count=0
 for f in range(n):
  ok=cap.grab();assert ok,(sid,f)
  if f not in lookup:continue
  ok,im=cap.retrieve();assert ok
  im=cv2.remap(im,mapping,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  images[lookup[f]]=np.asarray(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((512,512),Image.Resampling.BICUBIC));count+=1
 cap.release();images.flush();del images;assert count==len(idx)
 np.savez_compressed(dest/'gt.npz',camera_to_world=gt,timestamps_ns=timestamps,frame_indices_8fps=idx,source_fps=fps,source_frames=n)
 meta={'id':sid,'source_video':str(video),'source_frames':n,'source_fps':fps,'seconds':n/fps,'sampled_frames_8fps':len(idx),'old_gt_max_abs_difference':err,'device_time_rounding_max_ns':delta,'clockwise_quarters':q,'preprocessing':'official static rotaware remap 1408; PIL bicubic 512; uint8 lossless cache; no intermediate video re-encoding','first_1800_gt_exact':True}
 (dest/'ready.json').write_text(json.dumps(meta,indent=2));print('READY',sid,n,flush=True);return sid
if __name__=='__main__':
 jobs=json.loads((ROOT/'results/hot3d_manifest.json').read_text())
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(prepare,jobs))
 (OUT.parent/'manifest.json').write_text(json.dumps([{'id':j['id'],'data':str(OUT/j['id'])} for j in jobs],indent=2))
 print('ALL_READY',len(jobs),flush=True)
