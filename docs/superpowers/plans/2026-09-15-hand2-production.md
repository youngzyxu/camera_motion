# Hand-quality-2 camera production operational plan

Goal: Run existing verified Omega pipeline on landed web_first_36K hand_quality==2 only, across pro6000-1/2/7. User explicitly authorized clearing GPU workloads and starting production.
Architecture: existing HTTP/SQLite task server per machine, deterministic episode modulo 3 partitions, eight existing pipeline clients per machine. Stream verified metadata/video tasks into indexed persistent queues. No changes to model, frame sampling, preprocessing, or stitching.
Spec: conversation 2026-09-15. Output bos:/liberai-web-humandata/processing/for_next_20260901_150304_249934232/camera_pose_motion/results.

- [ ] Verify input metadata score and landed video; capture exclusions and counts.
- [x] Identify machines; clear authorized GPU processes; verify environments and BOS.
- [x] Start persistent indexed queues and manifest producer.
- [x] Start 24 supervised pipeline workers with 4fps/240/36 and BOS roundtrip checks.
- [x] Check successful results on each machine; audit score==2, output shape, checksum and worker count.
- [ ] Publish run status, resume instructions and final inventory when producer completes.

Operational constraints: never overwrite source LeRobot files; keep per-lease output and accepted result map. Do not claim complete reconstruction when only deployment is complete. Local host equals pro6000-1 by matching SSH endpoint ED25519 host key. Keep launcher/logs frozen in runs/hand2_20260915. Workers periodically restart after draining to bound resident log/future accumulation in existing client.

Deployment verified: 24 GPU workers produced successes. Independently downloaded one accepted result per machine and verified score, config, SHA256, pose shape, finite values and timestamp count. Inventory continues; final count is pending. README and five-minute status publisher installed.
