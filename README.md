# Camera pose reconstruction benchmark

本目录对比 VGGT-Ω 与 VGG-T³，目标为下游 WAM / video generation 提供视频相机轨迹。结论和逐条统计见 `results/REPORT.md`（完整测试后生成）。

## 仓库与环境

两个仓库从 `../../backup/` 移动到当前目录，未修改上游模型代码。已有旧 venv 的解释器路径失效；新环境为：

```bash
PY=/mnt/pfs/pfs-yc2F4O/miniconda3/envs/camera_motion/bin/python
```

该环境是基于同目录 `havor` 的 `--system-site-packages` venv，复用 PyTorch 2.10.0+cu130，额外复用了旧 venv 的 roma 1.5.7。本项目脚本显式设置两个仓库的 Python import 路径。

权重直接读取现有本地文件，没有重新下载：

- Ω：`/mnt/pfs/pfs-yc2F4O/modelTeam/code/xuzhiyong/backup/vggt_omega_1b_512.pt`
- VGG-T³：`/mnt/pfs/pfs-yc2F4O/hf_cache/hub/models--nvidia--vgg-ttt/snapshots/2bd0869b45f919791e80cda723925be2f030fe02/`

硬件按照驱动原样记录为 NVIDIA RPBZZZ6，compute capability 12.0、约 96 GB 显存；不将不透明设备名称自动改写成零售产品型号。每个推理进程只使用一张 GPU；各任务运行于独立空闲卡。

## 数据与口径

- 耗时：`results/ego_manifest.json`，用户指定 ego_hand0723 的全部 11 个片段，完整时长，2/4/8/16 fps。包含一个约 389 秒的长片段。
- 精度：`results/hot3d_manifest.json`，暂按本地找到的 HOT3D 20 条、每条 60 秒、修正 preview 旋转后的静态去畸变视频；4 fps、每条 240 帧。用户未给出额外 20 条清单时采用此假设。
- GT：`hot3d_raw_v5_world_native/*` 的 camera_to_world，与同帧相机 GT 和去畸变视频旋转元数据核验一致。详见 `results/hot3d_input_audit.json`。去畸变本身使用官方标定，模型未接收 GT 内参或 GT 位姿。
- 采样选择最接近目标时刻的原视频帧，保存真实 frame_indices 和 frame_index/source_fps；不会把 30/16 四舍五入为固定 stride。
- Ω 使用 max_size512、patch16；VGG-T³ 使用上游 crop518、patch14。记录实际输入尺寸；不是严格相同 token 数量的架构消融。
- 相机专用路径：移除不使用的深度/点云/跟踪头输出，保持主干和相机头不变。VGG-T³ 使用上游 memory_efficient_inference=True，offload_to_cpu=False；默认 infer 的 1 次 TTT 更新，以及 `--ttt-steps 2`（权重配置中 num_steps=2）的对照。与完整推理的数值对照见 `*_branch_verification.json`。
- 每个视频完整一次前向，不进一步裁剪主干缓存，无截断、无窗口拼接、无 CPU offload；OOM 明确计作失败，不用短视频平均值代替完整数据集总耗时。
- 模型常驻显存、单独预热，每个组合测量一次。CUDA 同步计时；inference_s 包含相机解码和结果 GPU→CPU，end_to_end_s 加视频读取/预处理及输入上传，不含模型加载、预热和磁盘输出。模型加载/预热和 save_s 分开记录。多任务共享 CPU/文件系统，端到端耗时存在 I/O 负载影响，单次运行不是稳定延迟的置信区间。
- ATE：每条轨迹单一全局 Sim(3) 对齐相机中心，报告米单位 RMSE。该指标去除了整体尺度与坐标系自由度，不证明模型能恢复真实米制尺度。
- RPE：相隔 1 秒的相机相对旋转误差（度）和相对平移 RMSE（米）；平移仅使用上述全局尺度。汇总同时保留逐条结果，不能将 GT 对齐后的轨迹当成无 GT 的部署输出。

## 复现

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/benchmark_camera.py --method omega --manifest results/ego_manifest.json --output results/ego_omega --fps 2 4 8 16
CUDA_VISIBLE_DEVICES=2 "$PY" scripts/benchmark_camera.py --method ttt --manifest results/ego_manifest.json --output results/ego_ttt --fps 2 4 8 16
CUDA_VISIBLE_DEVICES=3 "$PY" scripts/benchmark_camera.py --method omega --manifest results/hot3d_manifest.json --output results/hot3d_omega --fps 4
CUDA_VISIBLE_DEVICES=4 "$PY" scripts/benchmark_camera.py --method ttt --manifest results/hot3d_manifest.json --output results/hot3d_ttt --fps 4
"$PY" scripts/evaluate_camera.py --manifest results/hot3d_manifest.json --predictions results/hot3d_omega --output results/hot3d_omega_accuracy.json
"$PY" scripts/evaluate_camera.py --manifest results/hot3d_manifest.json --predictions results/hot3d_ttt --output results/hot3d_ttt_accuracy.json
```

附加2次更新比较使用同样命令，增加 `--ttt-steps 2`，并分别输出到 `ego_ttt_steps2` 和 `hot3d_ttt_steps2`。最后执行 `"$PY" scripts/make_report.py` 汇总并生成图表。

脚本跳过已有逐条 JSON。若需重测，换新 output 路径；不能在同一输出目录混合参数。`evaluate_camera.py --self-test` 校验已知 Sim(3) 下的指标不变性及扰动响应。

## 下游轨迹格式

每个成功输入的 NPZ 包含 `c2w[N,4,4]`、`intrinsics[N,3,3]`、`timestamps_s[N]`、`frame_indices[N]`、`input_hw[2]`、`source_fps`。OpenCV 约定：x 向右、y 向下、z 向前。内参对应模型预处理后的图像尺寸，不能直接当原视频内参。

```bash
"$PY" scripts/export_trajectory.py results/ego_omega/VIDEO__2fps.npz trajectory_30fps.npz --fps 30
```

导出工具把第一帧相机设为世界原点，额外输出 w2c，使用旋转 SLERP 与平移线性插值，只输出第一与最后一个观测姿态之间的时间范围。插值不增加真实相机观测，也不外推尾帧。位移默认仍为任意尺度；若有外部尺度标定，可显式传入 `--translation-scale`。不同 WAM/VideoGEN 的轴方向、c2w/w2c、绝对/相对/归一化尺度协议仍需按目标模型适配。

额外显存诊断：`results/diagnostic_omega_expandable/` 使用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 重试最长片段4fps，单独报告，不混入主表。

诊断实测：最长片段4fps/1555帧在默认分配器OOM，但使用上述可扩展分配器后成功（推理336.63秒，峰值allocated 93.85 GiB）。这个成功样本只在独立诊断目录，未覆盖主表默认配置的失败记录。复现：

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" scripts/benchmark_camera.py --method omega --manifest results/ego_longest_manifest.json --output results/diagnostic_omega_expandable --fps 4
```

## Ω 批量视频推荐配置（完整窗口实验）

默认采用 **8fps / 240帧窗口 / 48帧overlap / stride192**，即30秒窗口、6秒重叠、24秒步长。结论来自同批20条HOT3D完整视频的180组实验（2/4/8fps × 24/36/48 overlap），统一插值到30fps评价；另在ego_hand0723全部11条视频上实跑三个overlap验证耗时与显存。

[实验报告](results/omega_window_study/REPORT.md)、[默认配置](configs/omega_video.json)、[完整性审计](results/omega_window_study/artifact_audit.json)。该报告使用新的完整视频与稠密评估协议，应与上面的旧整段基准分开解读。

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/pfs/pfs-yc2F4O/miniconda3/envs/camera_motion/bin/python scripts/run_omega_video.py --input /path/to/videos --output /path/to/output
```

输入可以是单个MP4或视频目录，也可用 `--manifest results/ego_manifest.json` 代替 `--input`。模型常驻、流式读取、用重叠姿态做无GT的Sim(3)对齐与融合；输出8fps观测姿态，若需要稠密下游轨迹可继续用上面的 `export_trajectory.py`。平移仍为任意尺度，内参对应预处理尺寸。重跑不同参数请使用新的输出目录。

## LeRobot server-client批处理（4fps / overlap36）

按批量生产需求单独提供 [process_camera_lerobot.py](process_camera_lerobot.py)：4fps、240帧窗口、36帧overlap，HTTP任务租约分发、SQLite断点恢复、GPU模型常驻和BOS结果回传。使用方法见 [PROCESS_CAMERA_LEROBOT.md](PROCESS_CAMERA_LEROBOT.md)，八卡常驻运行、10分钟整机实测见 [LeRobot吞吐报告](results/lerobot_8gpu_4fps_ov36/REPORT.md)。该入口使用独立固定配置，不读取上面精度实验选出的8fps默认配置。

## 综合试验报告与流水线加速

[飞书：《camera pose消融试验记录报告》](https://ucn81zkkano6.feishu.cn/docx/Ealed6Zy0o9Bbhxo3LgcDIoAnld)。固定2048条、同一八卡模型的实测：串行706.82秒，解码/上传流水线406.47秒，加速1.74倍，2048对NPZ逐位一致。八卡外推1994.8视频小时/天，平均每卡249.4小时/天；包括BOS回读校验，不包括模型启动。详见 `results/lerobot_pipeline_ab/summary.json`。

## RTX 5090 多机续跑

`deploy_5090/` 提供本批 chenqi egomixed 数据的 18 台 × 8 卡部署：统一队列、
本地预取/常驻模型、持久化断点、网络异常重连和吞吐统计。
部署说明见 [deploy_5090/README.md](deploy_5090/README.md)，
停机和换机操作见 [deploy_5090/OPERATIONS.md](deploy_5090/OPERATIONS.md)。
