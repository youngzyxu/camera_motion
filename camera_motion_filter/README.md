# Camera motion filter：10K L3，1s infill

用户确认后的执行口径：只筛选 `results/l3_selection_20260916/l3_10K_hours.json` 的2,435,014条；保留完整原字段并新增camera_motion_filter判定信息；最后交付一份留存JSON（另有等价gzip压缩副本）。不重新补足10K，不裁剪视频。

## 本次采样约定

- handpose 两批：沿用网页全量统计，约4FPS并保留末帧。读取对应批次parquet里的 observation.mano.camera_ego_0_pose，而不是手部有效性mask。
- camera_motion_hand2：按已保存的frame_indices和timestamps_s计算。存在末帧就包含；缺少末帧不补点、不插值、不重新推理。统计保留/缺少末帧条数。
- 因此本次camera旧结果与网页的速度公式和过滤逻辑一致，但未包含末帧的条目采样覆盖区间不同，这是用户明确接受的口径。
- 进行中的及后续生产另行切换至4fps_include_last_v1；原10K引用旧lease版本保持不变。

## 指标与规则

令相邻camera-to-world旋转为R_i、R_(i+1)，位置p_i、p_(i+1)，dt=t_(i+1)-t_i。
Rrel=R_i.T@R_(i+1)。skew=[Rrel32-Rrel23,Rrel13-Rrel31,Rrel21-Rrel12]。
旋转速度=degrees(atan2(norm(skew)/2,(trace(Rrel)-1)/2))/dt；平移速度=norm(p_(i+1)-p_i)/dt。

1. 旋转 >19°/s 或平移 >0.40/s，先取布尔invalid并集；等于阈值不invalid。
2. 找原始并集内的连续合格(False)区间，按边界t[end]-t[start]求持续秒数。≤1秒（沿用1e-9容差）全部改为invalid，包括中间、首尾以及整个短合格覆盖区间。
3. 最终invalid区间数量/全部相邻采样区间数量 >50%，或完整视频duration_sec<1秒，过滤整条。等于50%保留。
4. 不使用时长加权、不累计曲线面积，不增加平滑或尺度归一化。有效采样点不足2个单列原因并不保留；读取/非有限数/时间轴错误独立报错，最终汇总要求0错误。
5. 保留小时按整条视频时长计；invalid比例只按区间数计。handpose的估计米制平移与Omega任意尺度沿用网页相同数值阈值，不声称物理尺度等价。

## 可追溯代码

- reference/sources.json：原网页源码路径与SHA256；reference/包含完整指标、infill、采样、handpose解码及基线源码快照。
- code/metrics.py：本次现有pose→速度→并集→1秒infill判定，直接调用冻结源码的infill_mask。
- code/test_metrics.py：边界和100组与网页公式/规则一致性测试。
- code/prepare.py：冻结两批handpose逐条CSV，提取本次10K中camera任务，不采入后续新完成数据。
- code/verify_handpose.py：随机128条从parquet重新计算，核对逐条基线。
- code/run_camera.py：三个分片执行器，64个有界I/O线程/机，优化bcecmd下载NPZ，按2000条checkpoint；错误最多重试3次。BOS挂载小文件读取较慢，执行中已改为CLI直接读取。
- code/finalize.py：按批次内episode和原dataset episode双重对齐，核对handpose基线、输入SHA256、全量唯一性、保留JSON的数量与时长，生成retention.csv/summary.json。

## 输出

output/l3_10K_retained_infill1s.json（及.json.gz）：唯一留存清单，保留所有原始字段，新增camera_motion_filter对象。
output/all_metrics_1s.csv：全部输入逐条指标、决定、原因，含未保留数据。
output/retention.csv、summary.json：分三批及总计的条数/小时留存统计。
output/camera_chunks/：分片指标及errors，支持断点继续；output/tasks/记录固定输入集合。
output/handpose_recompute_audit.json：128条独立重算核对。上游COMPLETE归档包含677条网页交集一致性证据；本次没有将这677条说成重新运行的测试。

## 生产末帧修复

scripts/lerobot_pipeline.py统一使用frame_sampling.sample_indices；scripts/run_omega_video.py默认include_last_frame=True；process_camera_lerobot.CONFIG加入include_last_frame及sampling_version。保留4fps/240/36和所有上传回读校验。
通过STOP_GPUN让各GPU排空在途任务后重启；排空中的旧任务仍按旧口径产出，新worker开始的任务保证末帧。新旧版本由结果JSON/config和NPZ索引区分。测试覆盖单帧、采样末帧、240帧边界、36帧重叠及队列失败/提交顺序；恢复后逐卡抽查实际BOS NPZ端点。

原待确认文档 PROTOCOL_PENDING_CONFIRMATION.md 仅作讨论存档，本README为最终确认执行口径。
