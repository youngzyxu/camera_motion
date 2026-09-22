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
