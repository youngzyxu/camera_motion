"""Audit exact 10-minute completion cohort and report actual eight-GPU throughput."""
import json,csv,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/lerobot_8gpu_4fps_ov36'
sys.path.insert(0,str(ROOT/'scripts'))
from window_trajectory import windows

def main():
 measure=json.loads((OUT/'measurement.json').read_text());records=measure['records'];elapsed=measure['wall_s'];assign={};metas=[]
 for gpu in range(8):
  metas.append(json.loads((OUT/f'gpu{gpu}/metadata.json').read_text()))
  for line in (OUT/f'gpu{gpu}/completed.jsonl').read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   assign[r['task']['lease_id']]=gpu
 per_gpu=[{'gpu':i,'videos':0,'video_s':0.,'inference_s':0.,'upload_s':0.,'peak_allocated_gib':0.} for i in range(8)]
 flat=[];maxrot=0;bytes_total=0;fallback=0;maxframes=0
 for record in records:
  r=record['metrics'];t=r['task'];gpu=assign[t['lease_id']];d=OUT/f'gpu{gpu}'/'tasks'/t['task_id']/t['lease_id'];z=np.load(d/'camera.npz');p=z['c2w'];idx=np.unique(np.floor(np.arange(0,r['source_frames']/r['source_fps'],.25)*r['source_fps']+.5).astype(np.int64));idx=idx[idx<r['source_frames']]
  assert np.array_equal(z['frame_indices'],idx) and len(p)==r['sampled_frames']
  assert all(np.isfinite(z[k]).all() for k in z.files)
  assert np.allclose(p[0],np.eye(4),atol=1e-5)
  rot=p[:,:3,:3];err=float(np.max(abs(rot.transpose(0,2,1)@rot-np.eye(3))));maxrot=max(maxrot,err)
  assert err<1e-5 and np.allclose(np.linalg.det(rot),1,atol=1e-5)
  spans=windows(len(p),240,36);assert len(spans)==r['windows'] and r['max_window_input_frames']<=240
  for j,(s,e),(_,pe) in zip(r['joins'],spans[1:],spans):assert (j['start'],j['end'],j['overlap_end'])==(s,e,pe)
  assert len(r['joins'])==len(spans)-1 and r['readback_verified']
  marker=json.loads((d/'SUCCESS.json').read_text());assert marker['remote']==r['remote']
  for suffix in ['.npz','.json']:
   f=d/('camera'+suffix);assert hashlib.sha256(f.read_bytes()).hexdigest()==marker['sha256'][suffix];bytes_total+=f.stat().st_size
  assert measure['start_unix_s']<=record['accepted_at']<=measure['end_unix_s']
  g=per_gpu[gpu];g['videos']+=1
  for k in ['video_s','inference_s','upload_s']:g[k]+=r['video_seconds'] if k=='video_s' else r[k]
  g['peak_allocated_gib']=max(g['peak_allocated_gib'],r['peak_allocated_gib']);fallback+=r['scale_fallbacks'];maxframes=max(maxframes,r['max_window_input_frames'])
  flat.append({'video':r['video'],'camera':t['camera'],'gpu':gpu,'task_id':t['task_id'],'lease_id':t['lease_id'],'chunk':t['chunk'],'episode':t['episode'],'video_seconds':r['video_seconds'],'windows':r['windows'],'inference_s':r['inference_s'],'upload_s':r['upload_s'],'task_wall_s':r['task_wall_s'],'accepted_at':record['accepted_at'],'remote':r['remote']})
 for g in per_gpu:g['video_hours_per_day']=g['video_s']/elapsed*24
 for filename,rows in [('per_gpu.csv',per_gpu),('per_video.csv',flat)]:
  with (OUT/filename).open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 video=sum(r['video_seconds'] for r in flat);assert abs(video-measure['video_s'])<1e-6
 assert len({r['task_id'] for r in flat})==len(flat)
 failures=[]
 for gpu in range(8):
  p=OUT/f'gpu{gpu}/failed.jsonl'
  if p.exists():failures.extend(p.read_text().splitlines())
 durations=[r['video_seconds'] for r in flat]
 summary={'gpus':8,'wall_s':elapsed,'completed':len(flat),'video_s':video,'video_hours_per_machine_day':video/elapsed*24,'video_hours_per_gpu_day_average':video/elapsed*3,'failed_attempts_seen_by_report_time':len(failures),'inference_gpu_seconds':sum(g['inference_s'] for g in per_gpu),'upload_worker_seconds':sum(g['upload_s'] for g in per_gpu),'peak_worker_allocated_gib':max(g['peak_allocated_gib'] for g in per_gpu),'duration_s_p10_p50_p90_max':np.percentile(durations,[10,50,90,100]).tolist(),'chunks':len({r['chunk'] for r in flat}),'multiwindow_videos':sum(r['windows']>1 for r in flat),'model_load_warmup_seconds_by_gpu':[m['model_load_and_warmup_s'] for m in metas],'per_gpu':per_gpu,'benchmark_rule':'server accepted completions during shared 600-second interval, after all 8 models ready; services continue'}
 audit={'passed':True,'videos':len(flat),'unique_tasks':len(flat),'all_BOS_npz_json_SHA256_roundtrips_verified':True,'max_rotation_orthogonality_error':maxrot,'max_window_frames':maxframes,'scale_fallbacks':fallback,'output_bytes_npz_json':bytes_total,'checks':['exact source frame sampling','finite arrays','first pose identity','proper rotations','window spans and overlaps','server accepted within time interval','BOS readback checksum']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
 with (OUT/'accepted_results.jsonl').open('w') as f:
  for r in flat:f.write(json.dumps({k:r[k] for k in ['task_id','chunk','camera','episode','video','remote']})+'\n')
 rate=summary['video_hours_per_machine_day'];avg=rate/8
 lines=['# LeRobot 八卡 PRO6000 吞吐实测','','**4fps / 窗口240帧 / overlap36帧 / stride204帧**。GPU0–7各一个常驻Ω模型，共用HTTP任务server。', '', f"本机八卡整机实测外推 **{rate:.1f} 视频小时/天**，平均每卡 **{avg:.1f} 视频小时/天**。平均每卡值来自八卡同时运行，不是独立单卡10分钟测量。", '', '| 项目 | 实测 |','|---|---:|',f'| 统一测量区间 | {elapsed:.2f} 秒 |',f'| 完整成功并回传的视频 | {len(flat)} 条 |',f'| 累计原视频时长 | {video/60:.2f} 分钟 / {video/3600:.3f} 小时 |',f'| 失败尝试（截至报告） | {len(failures)} |',f"| 最大单worker allocated显存 | {summary['peak_worker_allocated_gib']:.2f} GiB |",f"| 多窗口视频 | {summary['multiwindow_videos']} 条 |",f'| 最长成功视频 | {max(durations):.2f} 秒 |', '', f'公式：{video:.2f} / {elapsed:.2f} × 24 = {rate:.1f} 视频小时/整机天；除以8得到每卡平均。', '', '## 每卡实际贡献','','| GPU | 视频数 | 视频分钟 | 折算视频小时/天 | 峰值allocated/GiB |','|---:|---:|---:|---:|---:|']
 for g in per_gpu:lines.append(f"| {g['gpu']} | {g['videos']} | {g['video_s']/60:.2f} | {g['video_hours_per_day']:.1f} | {g['peak_allocated_gib']:.2f} |")
 lines+=['', '任务动态分配，片段长度和分辨率不同；各卡短时贡献差异不能解释为硬件性能差异。GPU0还保留着用户原有约13GiB进程，本次未终止该进程。表中显存是worker自身分配峰值，不包含原有进程。', '', '## 计时范围与样本','', '输入直接从PFS读原MP4；CPU预处理、GPU推理、拼接、NPZ/JSON写出、BOS上传、逐条BOS下载SHA256校验和server确认均在处理链路内。所有模型完成加载/预热后开始统一600秒测量，以区间内server接受的完整任务计数；起止边界可能有跨区间任务，这是持续运行的完成吞吐口径。', '', '模型启动不计入稳态600秒；各卡加载/预热耗时另存summary.json。10分钟结束后服务继续常驻，不为统计排空队列。每日产能为短时外推，不是24小时持续运行实测，也不含BOS源视频下载。', '', f"任务池从128个chunk随机抽4096条，seed20260909；每chunk32条后随机打乱。测量完成视频覆盖{summary['chunks']}个chunk，时长中位数{np.median(durations):.2f}秒，90分位{np.percentile(durations,90):.2f}秒。chunk内文件数不同时，此抽样不是严格全库均匀抽样。", '', '## 服务与产物','', '独立入口：[process_camera_lerobot.py](../../process_camera_lerobot.py)，部署、续跑、追加任务与停止说明：[PROCESS_CAMERA_LEROBOT.md](../../PROCESS_CAMERA_LEROBOT.md)。服务PID见 [services.json](services.json)。当前任务池共4096条，完成后八个worker保持模型就绪，支持向现有队列追加任务。', '', 'BOS：`bos:/liberai-web-humandata/processing/for_next_20260901_150304_249934232/temp_camera_motion/benchmark_20260909_8gpu_4fps_ov36/`。结果映射见 [accepted_results.jsonl](accepted_results.jsonl)；每个episode/lease子目录包含camera.npz、camera.json及最后写入的SUCCESS.json。', '', '全部被计数产物均通过帧覆盖、数值、旋转矩阵、窗口拼接边界和BOS回读校验，见 [artifact_audit.json](artifact_audit.json)、[逐视频统计](per_video.csv)、[汇总JSON](summary.json)。', '', f'无GT拼接，尺度回退记录共{fallback}次；回退仅表示重叠段尺度估计退化，不等于解码或处理失败。平移仍为任意尺度，首相机为原点、OpenCV轴约定，内参对应预处理分辨率。原始LeRobot parquet未修改。', '', '## 可复现性','', '原始任务池：`../lerobot_4fps_ov36/worklist.jsonl`；逐卡原始日志及完成记录保留在本目录。测量程序为`scripts/measure_lerobot_8gpu.py`，报告及产物校验程序为`scripts/report_lerobot_8gpu.py`。旧单卡测试按用户要求中途切换八卡，未作为完整10分钟单卡基准。']
 lines+=['','另外对32条视频（含长片段）使用ffprobe交叉检查时长，最大差异低于1微秒，见 [duration_crosscheck.json](duration_crosscheck.json)。','','![八卡吞吐](throughput.png)']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
