# 10K L3 camera motion 过滤口径（待用户确认，尚未执行全量过滤）

## 输入与来源

输入 `results/l3_selection_20260916/l3_10K_hours.json`，2,435,014条，10,000.014839945494小时。保留其batch、episode_id、source_episode_id、l3_id、video_path及全部原字段。

- part_1_28K：713,803条，2,857.12188254小时。
- part_2_remaining：882,971条，3,541.71201961小时。
- camera_motion_hand2：838,240条，3,601.18093780小时。

handpose的camera pose来自对应落盘parquet中的 `observation.mano.camera_ego_0_pose`，不使用手部有效性mask替代相机运动指标。xyz + rot6d解码沿用slam_comparison/build.py：前两行经Gram–Schmidt归一化，第三行为叉积。Omega取c2w的R和位置p。沿用原有平移尺度，不额外归一化；handpose估计米制与Omega任意尺度使用同一数值阈值，并不意味着物理速度等价。

## 与网页一致的指标

用frame_sampling.sample_indices(n, fps, 4)，最近帧floor(x+0.5)、去重、强制包含n-1末帧；t_i=frame_index_i/fps，dt_i=t_(i+1)-t_i。不虚构末帧后的区间。

Rrel_i = R_i.T @ R_(i+1)。令skew=[Rrel32-Rrel23,Rrel13-Rrel31,Rrel21-Rrel12]：

- angle_i = degrees(atan2(norm(skew)/2, (trace(Rrel)-1)/2))。
- rotation_speed_i = angle_i / dt_i，单位deg/s。
- translation_speed_i = norm(p_(i+1)-p_i) / dt_i，沿用各pose原单位/s。

不增加平滑、截断、尺度归一化或其他阈值。每条至少需要2个相机采样点；缺失/非有限/时间轴异常单列错误，不能默认为通过。

## 过滤规则

1. 旋转速度 >19 deg/s 或平移速度 >0.40/s，取布尔invalid并集。同一区间只计一次，等于阈值不invalid。
2. 在原始并集上找所有连续False合格段。用采样边界t[end]-t[start]计算段长。
3. 对1秒、1.5秒分别计算：段长 <= gap+1e-9 时，将整个段标成invalid。包含中间、首尾；整个覆盖区间合格但足够短时也会被全部填充。两版本都从原并集计算，不能递进填充。
4. 最终比例 = 填充后invalid区间数 / 全部相邻采样区间数。按点数，不是时长加权，不是曲线积分。
5. 比例 >0.5 或完整clip时长 n/fps <1秒，过滤整条。等于50%保留；不裁剪视频、不只扣invalid时长。
6. 报告两版本，网页当前默认1.5秒，1秒为对照。小时留存统计被保留完整L3的duration_sec。

## handpose基线比较

复用并核对 upstream handpose_population/output/part_1_28K_clips.csv 和 part_2_remaining_clips.csv，按(batch,episode_index)以及source_episode_index双重对齐，验证这10K内handpose子集恰好覆盖全部1,596,774条。抽样重算parquet指标，复核上游677网页交集验证记录。基线统计必须精确一致；任何差异单独列出。

- 1s：1,109,881条，4,994.6477小时。
- 1.5s：1,011,859条，4,711.0518小时。

分别报告每个batch和合计：输入数/小时、保留数/小时、留存率、过滤原因、错误数；新增camera与已有handpose分开对比。过滤后不重新补齐10K。

## 必须确认的采样差异

当前生产lerobot_pipeline.py仅4fps采样，未调用保留末帧的sample_indices。网页run_camera_sample.py明确使用run_video(...include_last_frame=True)。已存档的3个生产NPZ抽查全部缺少末帧，缺失0.067–0.134秒，见sampling_audit.json。这不是全量比例估计。

严格与网页相同的推荐路径：先审计camera子集的frame_indices；对不满足网页采样索引的条目，以include_last_frame=True重新推理完整视频窗口并拼接，写独立目录，保留原结果。已有严格满足索引的条目可复用。不能重复最后pose/补时间戳冒充真实末帧；仅追加末帧或只重算尾窗，也不能保证等同网页的联合多帧推理结果。

若选择直接使用现有Omega采样点，速度和infill代码可相同，但采样口径不完全一致，必须标记为单独近似版本，不可宣称严格对齐网页。该选项尚未得到用户确认。

## 目录与交付

reference/已归档网页指标/过滤/采样源码、handpose基线报告与汇总CSV，sources.json记录原路径和SHA256。待确认后在此目录新增执行入口、逐条指标、过滤决定、分批统计、1s/1.5s保留与剔除JSON清单、错误清单和最终报告。原始10K清单和逐视频结果只读。
