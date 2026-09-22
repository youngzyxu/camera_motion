"""Audit saved trajectories and append measured deployment performance to the report."""
import csv,json
from pathlib import Path
import numpy as np
from window_trajectory import windows
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/omega_window_study'
VERSION='orientation_sim3_irls12_blend_v1'

def audit(p,source_frames,source_fps):
 r=json.loads(p.read_text());z=np.load(p.with_suffix('.npz'))
 idx=np.unique(np.floor(np.arange(0,source_frames/source_fps,1/r['fps'])*source_fps+.5).astype(np.int64));idx=idx[idx<source_frames]
 assert np.array_equal(idx,z['frame_indices']),str(p)
 pose=z['c2w'];assert len(pose)==r['sampled_frames']==len(idx)
 assert all(np.isfinite(z[k]).all() for k in z.files)
 assert np.allclose(pose[0],np.eye(4),atol=1e-5)
 rot=pose[:,:3,:3];err=float(np.max(abs(rot.transpose(0,2,1)@rot-np.eye(3))))
 assert err<1e-5 and np.allclose(np.linalg.det(rot),1,atol=1e-5)
 assert np.allclose(pose[:,3,:],np.array([0,0,0,1]),atol=1e-6)
 assert np.allclose(z['timestamps_s'],idx/source_fps)
 spans=windows(len(idx),240,r['overlap']);assert len(spans)==r['windows']
 assert r['window']==240 and r['stitch_version']==VERSION
 assert len(r['joins'])==len(spans)-1
 for j,(s,e),(_,pe) in zip(r['joins'],spans[1:],spans):
  assert (j['start'],j['end'],j['overlap_end'])==(s,e,pe)
 if 'max_window_input_frames' in r:
  assert r['max_window_input_frames']==max(e-s for s,e in spans)<=240
  assert r['processed_window_frames']==sum(e-s for s,e in spans)
 return r,err

def main():
 manifest=json.loads((OUT/'manifest.json').read_text());errors=[];count=0
 for job in manifest:
  gt=np.load(Path(job['data'])/'gt.npz')
  for fps in [2,4,8]:
   for ov in [24,36,48]:
    r,e=audit(OUT/'predictions'/job['id']/f'{fps}fps_ov{ov}.json',int(gt['source_frames']),float(gt['source_fps']));errors.append(e);count+=1
 summary=json.loads((OUT/'summary.json').read_text());assert summary['complete']
 assert len({r['dense_frames'] for r in summary['summary']})==1
 jobs=json.loads((ROOT/'results/ego_manifest.json').read_text());perf=[]
 for ov in [24,36,48]:
  rows=[]
  for job in jobs:
   p=OUT/f'ego_8fps_ov{ov}'/(job['id']+'.json');r=json.loads(p.read_text());r,e=audit(p,r['source_frames'],r['source_fps']);assert r['status']=='ok' and r['fps']==8 and r['overlap']==ov;rows.append(r);errors.append(e)
  longest=max(rows,key=lambda r:r['video_seconds'])
  perf.append({'fps':8,'overlap':ov,'videos':len(rows),'video_total_s':sum(r['video_seconds'] for r in rows),'inference_total_s':sum(r['inference_s'] for r in rows),'end_to_end_total_s':sum(r['end_to_end_s'] for r in rows),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in rows),'scale_fallbacks':sum(r['scale_fallbacks'] for r in rows),'longest_video_s':longest['video_seconds'],'longest_samples':longest['sampled_frames'],'longest_windows':longest['windows'],'longest_inference_s':longest['inference_s'],'longest_end_to_end_s':longest['end_to_end_s']})
 (OUT/'ego_performance.json').write_text(json.dumps(perf,indent=2))
 with (OUT/'ego_performance.csv').open('w') as f:
  w=csv.DictWriter(f,list(perf[0]));w.writeheader();w.writerows(perf)
 checks={'passed':True,'hot3d_cases':count,'ego_cases':len(jobs)*3,'maximum_rotation_orthogonality_error':max(errors),'dense_frames_per_configuration':summary['summary'][0]['dense_frames'],'checks':['exact sampled frame coverage','finite pose/intrinsics/timestamps','first pose identity','proper rotation matrices','window ranges and overlaps','consistent final stitching version','same dense evaluation frame count']}
 (OUT/'artifact_audit.json').write_text(json.dumps(checks,indent=2))
 marker='\n## 实际批量视频运行验证\n'
 report=(OUT/'REPORT.md').read_text().split(marker)[0]
 lines=[marker,'在本机独立GPU上，每个overlap各顺序跑完ego_hand0723全部11条视频，8fps、窗口240。模型在各自批次常驻。下表为真实整段运行，不是缓存窗口的耗时累加。','','| overlap | 成功视频 | 推理总秒 | 端到端总秒 | 峰值显存/GiB | 最长视频推理秒 | 最长视频端到端秒 |','|---:|---:|---:|---:|---:|---:|---:|']
 for p in perf:lines.append(f"| {p['overlap']} | {p['videos']}/11 | {p['inference_total_s']:.2f} | {p['end_to_end_total_s']:.2f} | {p['peak_allocated_gib']:.2f} | {p['longest_inference_s']:.2f} | {p['longest_end_to_end_s']:.2f} |")
 lines+=['',f"最长视频{perf[0]['longest_video_s']:.2f}秒，各配置均输出{perf[0]['longest_samples']}个观测姿态；窗口输入最多240帧。各配置尺度回退次数：{[p['scale_fallbacks'] for p in perf]}。",'端到端计时包含读取、预处理、上传、推理和拼接，不含模型加载、预热和输出文件写入。每配置单次运行，独立GPU之间共享CPU与文件系统；不能视为稳定延迟置信区间。ego数据无GT，本项只验证耗时、显存和输出完整性。', '', 'HOT3D精度数据使用完整视频、无损缓存与相同30fps评估时间轴；与此前60秒、仅4fps观测位置的评估协议不同，数值不能直接横向比较。8fps主要减少下游稠密轨迹的旋转插值误差；48帧overlap改善的是本实验的平移轨迹精度，接缝旋转指标与其他overlap接近。', '', '产物审计全部通过，见 [artifact_audit.json](artifact_audit.json)。拼接的已知Sim3、旋转融合、共线运动、离群点及静止退化自测见 `scripts/window_trajectory.py`。性能明细见 [ego_performance.csv](ego_performance.csv)。', '', '![配置比较](configuration_comparison.png)', '', '![逐视频overlap比较](paired_overlap.png)', '', '汇总复现：依次运行 `scripts/restitch_window_study.py`、`scripts/summarize_window_study.py`、`scripts/plot_window_study.py`、`scripts/finalize_window_report.py`。原始推理与数据准备协议见 PROTOCOL.md。']
 (OUT/'REPORT.md').write_text(report+'\n'.join(lines)+'\n')
 print(json.dumps({'audit':checks,'performance':perf},indent=2))
if __name__=='__main__':main()
