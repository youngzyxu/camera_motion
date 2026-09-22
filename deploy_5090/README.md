# Camera motion 5090 multi-node resume

Resume the existing `chenqi_egomixed_0920` manifest and BOS result namespace. Model,
4 FPS + real last frame, 240-frame window, 36-frame overlap, preprocessing and stitch
remain unchanged. No inference batching. Source cameras below 4 FPS retain real frames.

## Origin and machines

- Original input: `/mnt/pfs/pfs-yc2F4O/chenqi/pretrain/human/ego_mixed_scaling/exports/10kh_camera_pose_manifest.parquet`
- Output: `bos:/liberai-web-humandata/processing/camera_motion_feibiao/chenqi_egomixed_0920/results`
- Restore: `runs/chenqi_egomixed_0920/global/checkpoint.sqlite`
- Initial counts: 443,805 done, 726,146 pending; 1,169,951 total.
- Deployment: `runs/chenqi_egomixed_0920_5090`; isolated from old stopped Pro6000 deployment.
- SSH `/root/.ssh/config`: `5090-1` through `5090-18`, 8 GPUs each (144 total).
- Unique server: 5090-1, `10.64.115.219:18818`; all clients share its queue.
- Active SQLite: `/tmp/camera_motion_5090/chenqi_egomixed_0920_5090/queue.sqlite` on 5090-1.
- Latest shared snapshot: `runs/chenqi_egomixed_0920_5090/global/checkpoint.sqlite`.
- Existing Python environment: `.../miniconda3/envs/camera_motion/bin/python`
  (torch 2.10.0+cu130; symlink into havor). No shared environment upgrades.

## Optimizations and reference

Reference: `/mnt/pfs/pfs-yc2F4O/modelTeam/code/xuzhiyong/l3_quality_check_deploy/quality_pipeline`.
Adopt bounded stages, explicit concurrency/decoder threads, local staging, durable
lease recovery, upload retries and graceful draining. Do not copy its 64 decoder
threads: its 16-frame scoring workload differs from 240-frame camera reconstruction.

1. Single SQLite writer coalesces at most 128 requests / 2 ms into a transaction.
   Replies are sent only after successful commit. Malformed requests roll back their
   savepoint without affecting other requests. Status and health avoid the DB lock.
2. Local WAL with NORMAL synchronization. Passive WAL flush runs separately every
   5 seconds; PFS online snapshot runs separately, waiting 300 seconds after its
   previous completion. Final snapshot on graceful server shutdown.
3. Per GPU: one model, one prefetched-video copier, one decoder (FFmpeg 1 thread),
   a two-packet window queue, one uploader, at most four leased videos. Intermediate
   data and weights use local disk. Eight workers/node start 3 seconds apart.
4. Keep CUDA allocator cache; preserve existing BCE 0.5.17-1 no-progress flags,
   retry transfers and verify NPZ/JSON uploads before accepting SUCCESS.
5. Transport errors retry while retaining the loaded model (`CAMERA_RPC_RETRY=1`);
   invalid 4xx requests fail instead of looping forever. Each upload is accepted
   idempotently by task/lease ID.
6. Local logs/state copied to PFS every 30 seconds, monitor publishes BOS state
   every 5 minutes and generates the original final merged JSON on completion.

NORMAL WAL survives process restarts; machine/local disk loss restores the latest
shared snapshot and may redo tasks completed since that snapshot. This is **not**
a zero-data-loss guarantee for acknowledged work after the last PFS snapshot.
Accepted output URI remains authoritative when more than one lease directory exists.
Server startup prefers the new deployment checkpoint; only the first start uses
original Pro6000 checkpoint. Never restore `seed.sqlite` for this continuation.

## Commands

Run from project root, using an interpreter with access to the shared mount:

```
python deploy_5090/manage.py status
python deploy_5090/manage.py clients --hosts 5090-4,5090-5
python deploy_5090/manage.py stop                     # drain all 18 clients
```

`manage.py init` creates a new deployment once from SSH host aliases and recorded
inventory. `server` starts the singleton. Node supervisors refuse duplicate local
instances and refuse STOP markers; unsuccessful GPU workers retry at most three
launches, then stay disabled for diagnosis. They exit normally when the queue drains.
Do not run server simultaneously on two hosts. To move it: drain clients, stop
monitor/server, wait for final checkpoint, change topology, then launch replacement.
To resume deliberately stopped clients, clear their run/node STOP markers first.
Old Pro6000 STOP markers remain in place.

For shutdown: request client drain, verify all node processes exited and queue
running=0 (or explicitly recover orphan leases after server shutdown), SIGTERM monitor
and server using recorded PID files, and wait for final checkpoint completion.
Do not SIGKILL the server while it is snapshotting.

## Validation

- `python deploy_5090/test_server.py`: 144 concurrent claim/completion workers,
  unique leases, idempotent completion, stale rejection, bad-request rollback,
  restart persistence, lease expiry without consuming retries, SQLite integrity.
- `scripts/test_lerobot_pipeline.py`: upload acceptance order, failures/retries,
  local prefetch content and pipeline drain, both CUDA cache modes.
- `deploy_5090/smoke.py`: real RTX5090 240x512x512 inference, finite outputs;
  23.84 GiB peak allocation / 26.75 GiB reservation, 25.51 sec (synthetic input;
  not an end-to-end video throughput measurement).
- `deploy_5090/audit.py`: BOS readback, last frame, finite poses/intrinsics per GPU.
- `runtime/5090/inventory.json`, `runtime/5090/smoke.log` and deployment logs
  are runtime evidence, intentionally excluded from Git.

Upstream VGGT source repositories and weights remain external to this Git repo;
paths are preserved on the shared filesystem. `CAMERA_OMEGA_WEIGHTS` optionally
selects a local copy. Full source plus model/environment provenance is required
when moving to a machine without the shared mount.

Network outage validation exposed model reloads in the initial non-stay-alive client.
The final deployment rolls all nodes onto persistent transport retry; this was an
operational correction, not a model or sampling change. `test_rpc.py` validates
recovery beyond three failed attempts, fatal HTTP 400 and bounded status requests.
