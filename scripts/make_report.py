#!/usr/bin/env python3
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np
from evaluate_camera import fit_sim3
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
METHODS=[('omega','VGGT-Omega'),('ttt','VGG-T3 (1 step)'),('ttt_steps2','VGG-T3 (2 steps)')]

def read_rows(folder):return [json.loads(p.read_text()) for p in sorted(folder.glob('*fps.json'))]
def write_csv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def main():
 lines=['# Camera pose benchmark — 2026-09-09','', '本次将用户所说的“20条”解释为本地 HOT3D 20 条修正旋转后的去畸变视频。数据来源、环境、指标及复现说明见 [README](../README.md)。','', '每次推理使用单张约96 GB GPU；驱动型号为 `NVIDIA RPBZZZ6`，compute capability 12.0。相机输出专用路径，不运行深度头。Ω 最长边512；VGG-T³ crop518。运行一次、先独立预热；不同任务在不同空闲卡并行，共享CPU和存储。','', '## HOT3D 精度（4 fps，每条完整60秒）','','每条轨迹采用一个全局 Sim(3) 对齐；ATE / RPE 平移使用GT米制尺度，仅作为评估。模型原始输出不具备已验证的米制尺度。去畸变使用官方静态标定；GT K/pose不传给模型。','', '| 方法 | 完成 | ATE RMSE / cm（micro） | ATE RMSE / cm（macro） | 1秒RPE旋转 / ° | 1秒RPE平移 / cm |','|---|---:|---:|---:|---:|---:|']
 accuracy=[];complete=True
 for name,label in METHODS:
  dest=OUT/f'hot3d_{name}_accuracy.json'
  subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_camera.py'),'--manifest',str(OUT/'hot3d_manifest.json'),'--predictions',str(OUT/f'hot3d_{name}'),'--output',str(dest)],check=True,stdout=subprocess.DEVNULL)
  d=json.loads(dest.read_text());s=d['summary'].get('4.0')
  if not s or s['sequences']!=20:complete=False
  if s:
   lines.append(f"| {label} | {s['sequences']}/20 | {s['ate_sim3_micro_rmse_m']*100:.2f} | {s['ate_sim3_macro_rmse_m']*100:.2f} | {s['rpe_1s_rot_mean_deg']:.2f} | {s['rpe_1s_trans_rmse_m']*100:.2f} |")
   for row in d['per_sequence']:accuracy.append({'method':name,**row})
 lines+=['','原始逐条精度见 [accuracy.csv](accuracy.csv)。不足20条时仅为中间结果，不作为完整比较。','', '## ego_hand0723 耗时（全部11条）','','`成功推理/端到端合计`只累加成功片段；若不足11条，不能理解为完整测试集完成时间。端到端包括视频读取、预处理、上传及推理，不含模型加载/预热/输出保存。OOM是实际尝试后的结果，未截断视频或自动切窗口。','', '| 方法 | 采样fps | 成功/11 | OOM | 成功推理合计 / s | 成功端到端合计 / s | 成功峰值显存 / GiB |','|---|---:|---:|---:|---:|---:|---:|']
 timing=[];summary=[]
 for name,label in METHODS:
  rows=read_rows(OUT/f'ego_{name}');timing.extend([{'variant':name,**r} for r in rows])
  if len(rows)!=44:complete=False
  for fps in [2,4,8,16]:
   allr=[r for r in rows if r['fps']==fps];rs=[r for r in allr if r['status']=='ok'];oom=sum(r['status']=='oom' for r in allr)
   s={'method':name,'fps':fps,'attempted':len(allr),'ok':len(rs),'oom':oom,'error':sum(r['status']=='error' for r in allr),'success_inference_s':sum(r['inference_s'] for r in rs),'success_end_to_end_s':sum(r['end_to_end_s'] for r in rs),'success_peak_allocated_gib':max([r['peak_allocated_gib'] for r in rs],default=0)};summary.append(s)
   lines.append(f"| {label} | {fps} | {len(rs)}/11 | {oom} | {s['success_inference_s']:.2f} | {s['success_end_to_end_s']:.2f} | {s['success_peak_allocated_gib']:.2f} |")
 lines+=['','### 相同成功片段的公平耗时比较','','以下每个fps都只选三种配置共同成功的片段；完整失败记录仍保留在上表。','', '| fps | 共同成功片段数 | Ω推理合计/s | T³-1推理合计/s | T³-2推理合计/s |','|---:|---:|---:|---:|---:|']
 for fps in [2,4,8,16]:
  maps=[{r['id']:r for r in timing if r['variant']==name and r['fps']==fps and r['status']=='ok'} for name,_ in METHODS];common=set(maps[0]).intersection(*[set(m) for m in maps[1:]])
  totals=[sum(m[k]['inference_s'] for k in common) for m in maps];lines.append(f"| {fps} | {len(common)} | "+' | '.join(f'{v:.2f}' for v in totals)+' |')
 fixed_maps=[{r['id']:r for r in timing if r['variant']==name and r['fps']==fps and r['status']=='ok'} for name,_ in METHODS for fps in [2,4,8,16]]
 fixed_ids=set(fixed_maps[0]).intersection(*[set(m) for m in fixed_maps[1:]])
 fixed_rows=[]
 lines+=['','### 全部采样率固定相同片段的耗时比较','',f'固定使用在所有方法、所有fps下均成功的 {len(fixed_ids)} 个片段，避免各fps的片段集合不同。数值为纯推理合计秒数。','', '| fps | Ω/s | T³-1/s | T³-2/s |','|---:|---:|---:|---:|']
 for fps in [2,4,8,16]:
  rr={'fps':fps,'clips':len(fixed_ids)}
  for name,_ in METHODS:rr[name]=sum(r['inference_s'] for r in timing if r['variant']==name and r['fps']==fps and r['id'] in fixed_ids)
  fixed_rows.append(rr);lines.append(f"| {fps} | {rr['omega']:.2f} | {rr['ttt']:.2f} | {rr['ttt_steps2']:.2f} |")
 write_csv(OUT/'timing_fixed_common.csv',fixed_rows)
 lines+=['','### 最长片段：979b… / seg03，388.67秒','','| 方法 | fps | 采样帧数 | 状态 | 推理/s | 端到端/s | 峰值/GiB |','|---|---:|---:|---|---:|---:|---:|']
 for r in timing:
  if r['id']=='979b3835-190b-57d0-8cec-86638e47fb74_seg03':
   lines.append(f"| {r['variant']} | {r['fps']:g} | {r.get('n_frames','?')} | {r['status']} | {r.get('inference_s',float('nan')):.2f} | {r.get('end_to_end_s',float('nan')):.2f} | {r['peak_allocated_gib']:.2f} |")
 lines+=['','`nan`表示该项失败，不能得到成功推理耗时。逐条细节、失败原因和elapsed_to_failure_s见 [timing.csv](timing.csv)。','', '## 产物与验证','','- `ego_{omega,ttt,ttt_steps2}/*.npz`：采样帧c2w、内参、帧号与时间戳。','- `hot3d_{omega,ttt,ttt_steps2}/*.npz`：20条精度集原始预测，未用GT校准输出。','- [输入审计](hot3d_input_audit.json)：20条视频哈希、GT哈希、帧数、坐标一致性。','- `*_branch_verification.json`：相机分支与上游完整推理数值对照。','- [环境版本](environment.json)；每个结果目录的metadata.json包含GPU UUID、输入模式、加载/预热耗时。','- [轨迹图](trajectories_xy.png)：GT与Sim(3)对齐后的预测，仅为评估可视化。','- [速度与精度图](comparison.png)。','- `scripts/export_trajectory.py`：首帧归一化、c2w/w2c及可选30fps插值，保留任意尺度标记。','','## 状态','', '所有规定组合均已尝试，精度3组各20条完成。' if complete else '运行中：尚有组合未完成。']
 diagnostic=read_rows(OUT/'diagnostic_omega_expandable')
 lines+=['','## Ω显存分配诊断（独立于主表）','', '`PYTORCH_ALLOC_CONF=expandable_segments:True`，GPU7，最长388.67秒片段，4fps/1555帧，其他推理设置相同。']
 if diagnostic:
  d=diagnostic[0]
  lines.append(f"状态：{d['status']}；峰值allocated {d['peak_allocated_gib']:.2f} GiB。")
  if d['status']=='ok':lines.append(f"推理 {d['inference_s']:.2f}s，端到端 {d['end_to_end_s']:.2f}s。原始配置OOM存在可通过分配策略避免的因素，不能将其当作该模型的绝对容量上限。")
  else:lines.append(f"复测仍失败，elapsed_to_failure={d['elapsed_to_failure_s']:.2f}s；本次分配策略未解决此输入的显存问题。")
 else:
  complete=False;lines.append('复测进行中。')
 if len(accuracy)==60:
  aa={name:{r['id']:r for r in accuracy if r['method']==name} for name,_ in METHODS}
  wins=sum(aa['omega'][k]['ate_sim3_rmse_m']<aa['ttt_steps2'][k]['ate_sim3_rmse_m'] for k in aa['omega'])
  lines+=['','## 结果解释','',f'在这20条固定输入和4fps设置下，Ω在 {wins}/20 条的ATE低于VGG-T³（2次更新）。T³的2次更新显著优于默认1次，但仍有更明显的轨迹误差。该精度结论不等同于ego_hand0723的精度：后者本次仅作为耗时集。', '长视频高采样率的整段推理需同时考虑显存和耗时；本次没有测试滑窗拼接、CPU offload、进一步裁剪主干缓存或其他分辨率，不能把这里的OOM当作所有实现的容量上限。', '用于WAM/VideoGEN时，优先保留原始采样时间戳和首帧相对轨迹；米制尺度需外部来源。插值只用于适配目标时间网格。']
 write_csv(OUT/'accuracy.csv',accuracy);write_csv(OUT/'timing.csv',timing);write_csv(OUT/'timing_summary.csv',summary)
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (OUT/'summary.json').write_text(json.dumps({'complete':complete,'timing':summary},indent=2))
 print('Report written; complete:',complete)
 if len(accuracy)==60:plot(accuracy,timing)

def plot(accuracy,timing):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=['#137c66','#e48a24','#6753bc']
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for (name,label),color in zip(METHODS,colors):
  rs=[r for r in timing if r['variant']==name and r['status']=='ok'];axes[0].scatter([r['n_frames'] for r in rs],[r['inference_s'] for r in rs],label=label,c=color,alpha=.75,s=25)
 axes[0].set(xscale='log',yscale='log',xlabel='Sampled frames',ylabel='Camera inference (s)',title='EgoHand: successful full-sequence runs');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
 scores=[np.sqrt(np.mean([r['ate_sim3_rmse_m']**2 for r in accuracy if r['method']==name]))*100 for name,_ in METHODS]
 axes[1].bar([label for _,label in METHODS],scores,color=colors);axes[1].set(ylabel='ATE Sim(3) RMSE (cm)',title='HOT3D: 20 x 60s, 4 fps (lower is better)')
 for i,v in enumerate(scores):axes[1].text(i,v,f'{v:.2f}',ha='center',va='bottom')
 fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
 jobs=json.loads((OUT/'hot3d_manifest.json').read_text());fig,axes=plt.subplots(5,4,figsize=(16,19))
 for ax,job in zip(axes.flat,jobs):
  gt=np.load(job['gt'])['camera_to_world'];origin=gt[0,:3,3]
  ax.plot(gt[:,0,3]-origin[0],gt[:,1,3]-origin[1],color='black',lw=1.5,label='GT')
  for (name,label),color in zip(METHODS,colors):
   z=np.load(OUT/f'hot3d_{name}'/(job['id']+'__4fps.npz'));p=z['c2w'][:,:3,3];s,r,t=fit_sim3(p,gt[z['frame_indices'],:3,3]);q=s*(p@r.T)+t-origin;ax.plot(q[:,0],q[:,1],color=color,lw=.9,label=label)
  ax.set_title(job['id'],fontsize=9);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.set_xlabel('world X (m)',fontsize=8);ax.set_ylabel('world Y (m)',fontsize=8)
 axes.flat[0].legend(fontsize=7);fig.suptitle('HOT3D camera trajectories: per-sequence Sim(3) alignment, XY projection',fontsize=14);fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(OUT/'trajectories_xy.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
