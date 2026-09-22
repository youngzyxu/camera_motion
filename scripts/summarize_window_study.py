"""Aggregate complete window experiments and choose the lowest-cost eligible config."""
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/omega_window_study'
def csv_write(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def summarize():
 rows=[json.loads(p.read_text()) for p in sorted((OUT/'predictions').glob('*/*fps_ov*.json'))];summary=[]
 for fps in [2,4,8]:
  for ov in [24,36,48]:
   rs=[r for r in rows if r['fps']==fps and r['overlap']==ov]
   if not rs:continue
   n=sum(r['n_frames'] for r in rs);pairs=sum(r['rpe_1s_pairs'] for r in rs);seams=[r for r in rs if r['seam_pairs']]
   s={'fps':fps,'overlap':ov,'sequences':len(rs),'dense_frames':n,'ate_cm':float(np.sqrt(sum(r['ate_squared_sum'] for r in rs)/n)*100),'rpe_rot_deg':sum(r['rpe_rotation_sum'] for r in rs)/pairs,'rpe_trans_cm':float(np.sqrt(sum(r['rpe_translation_squared_sum'] for r in rs)/pairs)*100),'anchor10_tail15_cm':float(np.mean([r['anchor10_tail15_rmse_m'] for r in rs])*100),'anchor10_endpoint_cm':float(np.mean([r['anchor10_endpoint_m'] for r in rs])*100),'fast_rot_deg':float(np.mean([r['fast_motion_rpe_rot_deg'] for r in rs])),'seam_p95_rot_macro_deg':float(np.mean([r['seam_rotation_p95_deg'] for r in seams])) if seams else None,'seam_trans_macro_cm':float(np.mean([r['seam_translation_rmse_m'] for r in seams])*100) if seams else None,'seam_sequences':len(seams),'joins':sum(len(r['joins']) for r in rs),'scale_fallbacks':sum(r['scale_fallbacks'] for r in rs),'inference_total_s':sum(r['inference_s'] for r in rs),'video_total_s':sum(r['video_seconds'] for r in rs),'mean_windows':float(np.mean([r['windows'] for r in rs])),'peak_gib':max(r['peak_allocated_gib'] for r in rs),'max_sequence_ate_cm':max(r['ate_sim3_rmse_m'] for r in rs)*100,'max_tail15_cm':max(r['anchor10_tail15_rmse_m'] for r in rs)*100}
   s['seam_step_rot_p95_macro_deg']=float(np.mean([r['seam_step_rot_p95_deg'] for r in seams])) if seams and all(r.get('seam_step_rot_p95_deg') is not None for r in seams) else None
   s['inference_s_per_video_minute']=s['inference_total_s']/s['video_total_s']*60;summary.append(s)
 complete=len(rows)==180 and len(summary)==9 and all(s['sequences']==20 for s in summary) and all(r.get('stitch_version')=='orientation_sim3_irls12_blend_v1' for r in rows)
 result={'complete':complete,'completed_cases':len(rows),'expected_cases':180,'summary':summary}
 if complete:
  gates={k:min(s[k] for s in summary)*factor+absolute for k,factor,absolute in [('ate_cm',1.2,.2),('rpe_rot_deg',1.2,.1),('anchor10_tail15_cm',1.25,.5),('fast_rot_deg',1.2,.1)]}
  for s in summary:s['eligible']=all(s[k]<=v for k,v in gates.items()) and s['scale_fallbacks']==0;s['quality_worst_gate_ratio']=max(s[k]/v for k,v in gates.items())
  eligible=[s for s in summary if s['eligible']]
  cheapest=min(eligible,key=lambda s:s['inference_total_s']) if eligible else min(summary,key=lambda s:(s['quality_worst_gate_ratio'],s['inference_total_s']))
  quality=min(eligible,key=lambda s:s['ate_cm']) if eligible else cheapest
  review={'added_after_full_per_sequence_review':True,'cost_increase_fraction':quality['inference_total_s']/cheapest['inference_total_s']-1,'ate_reduction_fraction':1-quality['ate_cm']/cheapest['ate_cm'],'worst_sequence_ate_reduction_fraction':1-quality['max_sequence_ate_cm']/cheapest['max_sequence_ate_cm']}
  prefer_quality=bool(eligible) and review['cost_increase_fraction']<=.15 and review['ate_reduction_fraction']>=.10 and review['worst_sequence_ate_reduction_fraction']>=.20
  selected=quality if prefer_quality else cheapest
  review['prefer_quality_default']=prefer_quality
  result.update(selection=selected,lowest_cost_eligible=cheapest,gates=gates,selection_all_gates_satisfied=bool(eligible),engineering_review=review)

  config={'model':'VGGT-Omega-1B-512','fps':selected['fps'],'window':240,'overlap':selected['overlap'],'stride':240-selected['overlap'],'preprocessing':'max_size512','stitch':'orientation_sim3_irls12_blend_v1','translation_units':'arbitrary','evidence':str(OUT/'REPORT.md'),'selection_all_gates_satisfied':bool(eligible)}
  (ROOT/'configs').mkdir(exist_ok=True);(ROOT/'configs/omega_video.json').write_text(json.dumps(config,indent=2))
 csv_write(OUT/'summary.csv',summary);csv_write(OUT/'per_sequence.csv',[{k:v for k,v in r.items() if k!='joins'} for r in rows]);(OUT/'summary.json').write_text(json.dumps(result,indent=2))
 lines=['# Ω：采样率与窗口重叠选型','','完整协议见 [PROTOCOL.md](PROTOCOL.md)。使用同批HOT3D的完整连续视频；240帧窗口；每个配置20条；所有预测插值到相同30fps时间轴，无GT参与拼接。','',f'状态：{"完成" if complete else "运行中"}，{len(rows)}/180 个组合。','', '| fps | overlap | 条数 | ATE/cm | 1秒旋转RPE/° | 首10秒对齐后尾15秒误差/cm | 快速运动旋转RPE/° | 逐视频接缝P95均值/° | 推理秒/视频分钟 |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for s in summary:lines.append(f"| {s['fps']} | {s['overlap']} | {s['sequences']} | {s['ate_cm']:.3f} | {s['rpe_rot_deg']:.3f} | {s['anchor10_tail15_cm']:.3f} | {s['fast_rot_deg']:.3f} | {s['seam_p95_rot_macro_deg']:.3f} | {s['inference_s_per_video_minute']:.2f} |")
 lines+=['','ATE是每条全轨迹Sim3对齐后的dense micro RMSE；尾段误差仅使用前10秒对齐，随后不再用GT校正。接缝列为逐视频P95的宏平均，不是所有帧合并后的P95，边界数量及位置随配置改变。纯推理成本由实测窗口耗时累加，不含视频预处理和模型加载。','', '完整逐条结果见 [per_sequence.csv](per_sequence.csv)，配置汇总见 [summary.csv](summary.csv)。']
 if complete:
  s=result['selection'];lines+=['','## 选择','',f"默认：**{s['fps']} fps / 240帧窗口 / {s['overlap']}帧overlap / stride {240-s['overlap']}**。",'',f"同时满足质量门槛的配置数：{sum(r['eligible'] for r in summary)}。最低成本合格候选为 {cheapest['fps']}fps/overlap{cheapest['overlap']}；结合全量逐条结果复核，默认配置按下述精度/成本收益选择。" if eligible else '没有配置同时满足全部门槛；选择最差门槛比值最小的折中配置，不能称为各项指标均最优。','', '质量门槛及判定详见 [summary.json](summary.json)，相机平移仍是任意尺度。选择依据是这20条数据，不构成独立测试集上的泛化保证。','', '## 可用推理入口','','`scripts/run_omega_video.py` 使用选定配置，流式读取视频，只保留一个窗口及overlap的图像缓存，模型在整个视频批次常驻，输出c2w/内参/帧号/时间戳。默认配置见 `configs/omega_video.json`。','', '```bash','CUDA_VISIBLE_DEVICES=1 /mnt/pfs/pfs-yc2F4O/miniconda3/envs/camera_motion/bin/python scripts/run_omega_video.py --input /path/to/videos --output /path/to/output','```']
 if complete:
  rv=result['engineering_review'];lines+=['','## 全量逐条结果后的工程复核','',f"相对最低成本合格候选，最低ATE候选的计算开销增加 {rv['cost_increase_fraction']*100:.1f}%，ATE降低 {rv['ate_reduction_fraction']*100:.1f}%，最差视频ATE降低 {rv['worst_sequence_ate_reduction_fraction']*100:.1f}%。", '复核采用：若额外成本≤15%、ATE改善≥10%、最差样本ATE改善≥20%，精度优先的默认配置选择该质量候选。此规则是在完整逐条结果复核后采用的工程决策，不是预注册的统计检验；初始近优质量门槛仍保留，最低成本合格候选也如实记录。', '首10秒对齐后的尾段误差也受初始尺度可观测性影响，并非纯粹的拼接漂移。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'complete':complete,'cases':len(rows),'selection':result.get('selection')},indent=2))
 return result
if __name__=='__main__':summarize()
