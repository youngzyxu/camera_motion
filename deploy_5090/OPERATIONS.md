# Operator handoff

All 18 clients use the SAME input/output batch and central queue. Original Pro6000
workers and server remain stopped. Do not start their launch_node.py scripts.

## Where to check

- Deployment status: `python deploy_5090/manage.py status` (live server counts).
- Shared status: `runs/chenqi_egomixed_0920_5090/status.json` (five-minute monitor).
- Per-node state: `<run>/<5090-N>/state.json` (30-second copies).
- Node/server/monitor logs: `<run>/<5090-N>/*.log`.
- Per-GPU accepted completion logs: `<run>/<5090-N>/gpuN/<timestamp>/completed.jsonl`.
- Local staging path: `/tmp/camera_motion_5090/chenqi_egomixed_0920_5090/`.
- BOS result layout is unchanged: `<dataset>/chunk-xxx/<camera_key>/episode_xxxxxx/<lease_id>/`.

A client process existing does not prove successful inference: inspect completed.jsonl
and `gpu_output_audit.json`. A node that fails three worker launches disables that
GPU, so inspect launch counts/returncodes when calculating throughput.

## Resume after server failure

On the SAME server with local disk retained, `manage.py server` reuses the active
local SQLite database and WAL. Do not copy over it.

When local disk is lost, server.py restores the new shared checkpoint if it exists;
only if absent does it restore the immutable Pro6000 stop checkpoint. On a new host,
update server_node/bind_host/server_url and queue_db together, ensure the old server
is stopped, then launch server and clients. Recovered leases expire in <=600 seconds
and return to pending without using computational failure attempts.

The checkpoint timestamp is `<run>/global/checkpoint.status.json`. The full snapshot
is created by SQLite online backup and published with an atomic rename. It is not
safe to copy a live SQLite main file without its WAL or the backup API.

## Graceful stop

`python deploy_5090/manage.py stop` creates node STOP files. Supervisors propagate
them to the local stop file, stop new claims and drain decoding/inference/uploads.
Wait for all node supervisor processes to exit. Then stop monitor and server with
SIGTERM using their recorded PID files after checking command lines. Wait for server
exit and final shared checkpoint. Keep local files until snapshot completion.

## Artifacts

`monitor.py` will create `camera_pose_results_all.json` only after all 1,169,951 tasks
are done, with no failed/running/pending tasks, and verify its BOS readback hash.
Its target is the ORIGINAL batch output parent directory, so pre-migration and
post-migration successes are merged together. NPZ payloads remain separate files.

## 已完成结果提前汇总（2026-09-22）

`export_results.py --partial` 在 server 本机读取一致的 SQLite 事务快照，输出
`camera_pose_results_partial_<UTC>.json` 及 `.summary.json`，上传 BOS 并回读 SHA256。
这份 JSON 与最终汇总采用同一条目结构，包含完整 task/result 和 NPZ/JSON/SUCCESS URI。
不内嵌位姿数组。它只代表快照时刻已被队列接受的成功结果，不能当作全量完成。

本地 `.ids.sqlite` 保存 task_id 与 accepted lease，用于差集。只有部分汇总上传、
回读校验成功后，才原子发布 `partial_export.json` 供 monitor 使用。

全批完成时 monitor 复用已验证的 JSON 字节，查询未导出的任务并追加；核对已导出
任务仍保持原 accepted lease、总数与原清单一致、部分文件及 ID 库哈希不变。
最终仍输出 `camera_pose_results_all.json` 并进行完整 BOS 回读校验，随后发布 summary。
原部分 JSON 保留，可提前供下游使用；最终文件替代它成为全量索引，不要把两者再次拼接。

`partial_export_requested.json` 表示用户已要求增量汇总。如果部分汇总未成功，monitor
会等待并报错，避免静默改回重新全量解析。失败时查看 `5090-1/partial_export.log`。
