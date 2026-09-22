"""Audit completed benchmark and report one-GPU throughput including BOS roundtrips."""
import csv,json,sqlite3,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/lerobot_4fps_ov36'
sys.path.insert(0,str(ROOT/'scripts'))
from window_trajectory import windows

def main():
 summary=json.loads((OUT/'measured/summary.json').read_text());rows=[json.loads(l) for l in (OUT/'measured/completed.jsonl').read_text().splitlines()]
 assert len(rows)==summary['completed'] and len({r['task']['task_id'] for r in rows})==len(rows)
 db=sqlite3.connect(OUT/'queue_measured.sqlite');states=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'));assert states.get('done',0)==len(rows);assert states.get('running',0)==0
 maxrot=0;bytes_total=0
 for r in rows:
  t=r['task'];d=OUT/'measured/tasks'/t['task_id']/t['lease_id'];z=np.load(d/'camera.npz');p=z['c2w'];idx=np.unique(np.floor(np.arange(0,r['source_frames']/r['source_fps'],1/4)*r['source_fps']+.5).astype(np.int64));idx=idx[idx<r['source_frames']]
  assert np.array_equal(z['frame_indices'],idx) and len(p)==r['sampled_frames']
  assert all(np.isfinite(z[k]).all() for k in z.files)
  assert np.allclose(p[0],np.eye(4),atol=1e-5)
  rot=p[:,:3,:3];error=float(np.max(abs(rot.transpose(0,2,1)@rot-np.eye(3))));maxrot=max(maxrot,error)
  assert error<1e-5 and np.allclose(np.linalg.det(rot),1,atol=1e-5)
  spans=windows(len(p),240,36);assert len(spans)==r['windows'] and r['max_window_input_frames']<=240
  assert r['readback_verified']
  marker=json.loads((d/'SUCCESS.json').read_text())
  for suffix in ['.npz','.json']:
   f=d/('camera'+suffix);assert hashlib.sha256(f.read_bytes()).hexdigest()==marker['sha256'][suffix];bytes_total+=f.stat().st_size
  accepted=json.loads(db.execute('SELECT result FROM tasks WHERE id=?',(t['task_id'],)).fetchone()[0]);assert accepted['metrics']['remote']==r['remote']
 db.close()
 durations=np.array([r['video_seconds'] for r in rows]);sample=json.loads((OUT/'sample_profile.json').read_text())['summary'];total=summary['wall_s'];rate=summary['video_hours_per_gpu_day']
 audit={'passed':True,'videos':len(rows),'all_BOS_NPZ_JSON_roundtrips_SHA256_verified':True,'max_rotation_orthogonality_error':maxrot,'output_bytes_npz_json':bytes_total,'queue_states':states,'multiwindow_videos':sum(r['windows']>1 for r in rows),'scale_fallbacks':sum(r['scale_fallbacks'] for r in rows),'duration_s_p10_p50_p90_max':np.percentile(durations,[10,50,90,100]).tolist(),'actual_chunks':len({r['task']['chunk'] for r in rows})}
 (OUT/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
 flat=[{k:r[k] for k in ['video','video_seconds','sampled_frames','windows','inference_s','end_to_end_s','save_s','upload_s','task_wall_s','peak_allocated_gib','scale_fallbacks','remote']} for r in rows]
 with (OUT/'per_video.csv').open('w') as f:w=csv.DictWriter(f,list(flat[0]));w.writeheader();w.writerows(flat)
 lines=['# LeRobot 相机轨迹单卡吞吐','','配置：**Ω 4fps / 240帧窗口 / 36帧overlap / stride204**。只使用本机GPU1的一张卡，一个client，结果写BOS并逐条下载回读校验。','',f"实测外推：**{rate:.1f} 视频小时 / 单GPU / 天**；计入模型加载与预热则为 **{summary['video_hours_per_gpu_day_including_startup']:.1f} 小时/天**。",'', '| 项目 | 实测 |','|---|---:|',f"| 正式处理墙钟（含BOS） | {total:.2f} 秒 |",f"| 模型加载与预热 | {summary['model_load_and_warmup_s']:.2f} 秒 |",f"| 成功视频 | {len(rows)} 条 |",f"| 成功原视频总时长 | {summary['video_s']/60:.2f} 分钟（{summary['video_s']/3600:.4f} 小时） |",f"| 失败尝试 | {summary['failed_attempts']} |",f"| GPU推理累计 | {summary['inference_s']:.2f} 秒 |",f"| BOS上传+回读校验累计 | {summary['upload_s']:.2f} 秒 |",f"| 峰值allocated显存 | {summary['peak_allocated_gib']:.2f} GiB |",f"| 多窗口视频 | {audit['multiwindow_videos']} 条 |",f"| 最长成功视频 | {max(durations):.2f} 秒 |",'',f"计算公式：{summary['video_s']:.2f} ÷ {total:.2f} × 24 = {rate:.1f} 视频小时/天。计时从模型就绪开始，到最后一个视频完成上传和server确认结束；不把未完成任务计入分子。",'', '## 样本与适用范围','',f"预先从128个chunk抽取4096条视频，seed20260909，随机顺序处理到10分钟停止领取。本次成功视频来自{audit['actual_chunks']}个chunk，时长中位数{np.median(durations):.2f}秒；任务池中位数{sample['duration_s_p10_p50_p90_p99_max'][1]:.2f}秒。本次结果是episode吞吐，不是把短视频先合并成长视频后的吞吐。",'', '每日值是这次短时运行的外推，不是连续24小时实测。输入从PFS读取，包含每条结果BOS上传和回读，不包含BOS源视频下载。其他机器的CPU、存储和网络负载会改变端到端吞吐；此处不将单卡结果直接当作整机多卡实测。', '', f"GPU驱动报告：`{summary['gpu']}`，`CUDA_VISIBLE_DEVICES={summary['visible_devices']}`。",'', '## 产物与验证','', '独立入口：[process_camera_lerobot.py](../../process_camera_lerobot.py)，部署与恢复说明：[PROCESS_CAMERA_LEROBOT.md](../../PROCESS_CAMERA_LEROBOT.md)。', '', f"BOS路径：`{summary['bos_prefix']}`。每个episode/lease目录有camera.npz、camera.json和最后写入的SUCCESS.json；下载映射见 [accepted_results.jsonl](accepted_results.jsonl)。", '', '所有成功输出均检查完整采样帧覆盖、有限数值、首姿态单位阵、旋转正交性和窗口边界，且BOS NPZ/JSON均逐条下载并校验SHA256。见 [artifact_audit.json](artifact_audit.json)、[per_video.csv](per_video.csv)、[原始汇总](measured/summary.json)、[实验协议](PROTOCOL.md)。', '', '拼接不使用GT。平移为任意尺度；首帧相机原点，OpenCV轴约定；内参对应预处理图像。未改写LeRobot源parquet。', '', '本次短视频多、每条上传固定开销明显。当前版本一个client顺序处理视频，未做跨任务解码/上传重叠；此结果为已验证的当前实现吞吐，不是硬件理论上限。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'summary':summary,'audit':audit},indent=2))
if __name__=='__main__':main()
