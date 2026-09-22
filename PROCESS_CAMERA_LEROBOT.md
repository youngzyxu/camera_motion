# LeRobot 相机轨迹批处理

入口：[process_camera_lerobot.py](process_camera_lerobot.py)。固定使用4fps、240帧窗口、36帧overlap、204帧步长；一个视频一个任务，不跨episode对齐。沿用`l3_cut/web_humandata_processing/server_client`的租约分发设计，独立实现SQLite持久化、心跳续租、失败重试与过期回收。

## 准备与启动

在camera_motion目录执行。server不需要GPU依赖；client使用已有camera_motion环境及本地Ω权重。

```bash
PY=/mnt/pfs/pfs-yc2F4O/miniconda3/envs/camera_motion/bin/python
mkdir -p results/lerobot_production
python process_camera_lerobot.py prepare \
  --output results/lerobot_production/worklist.jsonl
python process_camera_lerobot.py server \
  --manifest results/lerobot_production/worklist.jsonl \
  --db results/lerobot_production/queue.sqlite \
  --host 0.0.0.0 --port 18764
```

另一个终端启动client；每张GPU一个进程、独立stage目录。跨机器部署时，client须能访问清单中的PFS路径、Ω权重和BOS凭据，将`SERVER_IP`替换为server的内网地址。HTTP服务供可信内网使用，无内置身份认证。

```bash
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 "$PY" process_camera_lerobot.py client \
  --server http://SERVER_IP:18764 \
  --stage results/lerobot_production/gpu1 \
  --bos-prefix bos:/liberai-web-humandata/processing/for_next_20260901_150304_249934232/temp_camera_motion/omega_4fps_w240_ov36
```

不指定`--seconds`就一直运行到任务耗尽。`--seconds 600`用于10分钟测试，达到时间后完成当前视频再停；`--verify-upload`逐条下载并校验NPZ/JSON；`--keep-local`保留本地产物。默认上传并确认完成后删除当前任务的本地产物以控制磁盘占用。模型在整个client生命周期常驻，图像缓存最多240帧。

## 结果与断点恢复

BOS层级：`PREFIX/chunk-XXXX/observation.images.image_ego_0/episode_XXXXXX/LEASE_ID/`。每次尝试使用独立lease目录，避免超时旧worker覆盖新worker结果。先写`camera.npz`、`camera.json`，最后写`SUCCESS.json`，其中记录配置与文件SHA256。server数据库的done记录给出被接受的结果地址；恢复后可能存在未被server接受的旧lease产物，不能仅凭目录存在就计数。

NPZ包含`c2w`、`intrinsics`、`frame_indices`、`timestamps_s`、`input_hw`、`source_fps`。时间戳相对episode起点；输出4fps观测姿态。OpenCV坐标轴，首帧相机为世界原点，平移任意尺度，内参对应预处理图像。不会写回或覆盖LeRobot原始data/parquet中的字段。

server重启使用相同清单与SQLite文件，保留done状态；原先running的任务在租约过期后重新分配。client每30秒续租，默认租约180秒；最多尝试3次，最终失败保留在数据库。若改变模型/采样/拼接配置，应使用新的数据库与BOS前缀。避免两台server同时操作同一队列数据库。

查看队列：

```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://SERVER_IP:18764/status
```

每个client的`completed.jsonl`记录逐条耗时、BOS地址和视频信息；退出时`summary.json`给出实际墙钟、成功视频时长与每日视频小时数。失败尝试另存`failed.jsonl`。

## 本次实测

按最新要求改为八卡共同运行，结果汇总在[吞吐报告](results/lerobot_8gpu_4fps_ov36/REPORT.md)。这是PFS读视频、BOS写结果的整机八卡测试，包含逐条BOS回读校验；不包含从BOS下载源视频的成本。每日值是短时测量外推。此前单卡预实验因切换八卡而提前停止，不作为完整单卡10分钟测量。

导出server已接受的结果清单（读取SQLite，不加载GPU模型）：

```bash
python process_camera_lerobot.py export \
  --db results/lerobot_production/queue.sqlite \
  --output results/lerobot_production/accepted_results.jsonl
```

该清单把episode、camera、原视频和BOS地址关联起来，供后续合并或下载使用。

## 八卡常驻模式

本次已经启动八个client，GPU0–7各一个模型，共用18766端口的任务server。运行状态及PID见 `results/lerobot_8gpu_4fps_ov36/services.json`。加入`--stay-alive`后，队列暂时为空时模型也不会退出。

当前4096条抽样任务完成后，如需加入完整数据清单，可先prepare生成全量清单，再追加到运行中的数据库；已有task_id不会重复插入：

```bash
python process_camera_lerobot.py enqueue \
  --db results/lerobot_8gpu_4fps_ov36/queue.sqlite \
  --manifest /path/to/full_worklist.jsonl
```

准备全量清单采用逐chunk流式写入，避免在内存同时保留数百万任务。追加任务分批事务提交，不要求重启server或重新加载模型。

当前八个worker共用优雅停止标记：

```bash
touch results/lerobot_pipeline_ab/STOP
```

worker会完成当前视频上传后退出，server仍保留队列。重启worker前移除该STOP文件。八卡10分钟统计使用统一时间区间内server接受的完成任务；计时完成不会停止常驻worker。最终报告位于 `results/lerobot_8gpu_4fps_ov36/REPORT.md`，不能将平均每卡吞吐称作独立单卡实测。

## 解码与上传流水线

在client命令后增加 `--pipeline` 启用：一个CPU解码线程、最多两个待推理窗口、一个GPU消费者、两个异步BOS上传线程，最多四个已领取任务在途。仍然逐视频独立推理，仍支持 `--verify-upload`；等待上传期间持续续租，回读校验成功后才向server确认。

```bash
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 "$PY" process_camera_lerobot.py client \
  --server http://SERVER_IP:18764 \
  --stage results/lerobot_production/pipeline_gpu1 \
  --pipeline --stay-alive --verify-upload \
  --stop-file results/lerobot_production/STOP \
  --bos-prefix bos:/liberai-web-humandata/processing/for_next_20260901_150304_249934232/temp_camera_motion/pipeline_4fps_w240_ov36
```

`--seconds`或停止文件仅停止继续领取，已进入流水线的任务会处理并上传完毕。图像按窗口缓存，长视频不提前完整解码。不要只根据显存空闲无限增加CPU预取；解码缓冲与两个队列窗口也占用主机内存。

串行与流水线的固定2048条对照、配对数值和BOS校验位于 `results/lerobot_pipeline_ab/`。最终统一报告是 `camera pose消融试验记录报告.md`，并同步到飞书。

已通过固定2048条数值一致性和耗时对照，当前八个常驻模型已切换流水线消费原18766队列。服务PID和停止标记见 `results/lerobot_pipeline_ab/services.json`；旧串行worker已退出，原始GPU0用户进程保留。
