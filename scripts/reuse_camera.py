"""Validate historical camera artifacts before publishing them in a new batch."""
import hashlib
import json
from pathlib import Path

import numpy as np

from frame_sampling import sample_indices


def check_config(historical, expected, allow_legacy=False):
    for key, value in expected.items():
        if allow_legacy and key in ('include_last_frame', 'sampling_version') and key not in historical:
            continue
        assert historical.get(key) == value, f'Historical configuration mismatch: {key}'


def prepare_reuse(task, dest, transfer, config):
    source = task['reuse_camera']
    kind = source.get('kind', 'success_marker')
    legacy = source.get('allow_legacy_sampling_metadata', False)
    if kind == 'published_sample':
        # This separately published catalog has per-artifact hashes and a
        # batch completion receipt, rather than one SUCCESS marker per video.
        proof = dest.parent / 'original_COMPLETE.json'
        transfer(source['catalog_complete_uri'], proof)
        assert hashlib.sha256(proof.read_bytes()).hexdigest() == source['catalog_complete_sha256']
        complete = json.loads(proof.read_text())
        assert complete['status'] == 'complete' and complete['all_last_frames_preserved']
        for key in ('model', 'fps', 'window', 'overlap'):
            assert complete[key] == config[key], f'Sample catalog mismatch: {key}'
        assert complete['artifacts_published_and_sha256_verified'] == 3 * complete['count']
        hashes = source['sha256']
        uris = {'.npz': source['npz_uri'], '.json': source['json_uri']}
        remote = source['npz_uri']
    else:
        assert kind == 'success_marker', 'Unsupported historical artifact kind'
        remote = source['remote'].rstrip('/')
        marker_path = dest.parent / 'original_SUCCESS.json'
        transfer(remote + '/SUCCESS.json', marker_path)
        marker = json.loads(marker_path.read_text())
        assert marker['task_id'] == source['task_id']
        assert marker['lease_id'] == source['lease_id']
        assert marker['video'] == task['video']
        assert marker['readback_verified']
        check_config(marker['config'], config, legacy)
        hashes = marker['sha256']
        uris = {suffix: remote + '/camera' + suffix for suffix in ('.npz', '.json')}
    for suffix in ['.npz', '.json']:
        target = Path(str(dest) + suffix)
        transfer(uris[suffix], target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[suffix]
    row = json.loads(Path(str(dest) + '.json').read_text())
    assert row['video'] == task['video']
    if kind == 'published_sample':
        assert row['episode_index'] == task['episode_index']
        for key in ('fps', 'window', 'overlap'):
            assert row[key] == config[key]
        assert row['stitch_version'] == config['stitch']
        assert row['preprocessing'] == 'max_size512' and row['include_last_frame']
    else:
        check_config(row['config'], config, legacy)
        if not legacy:
            assert row['include_last_frame'] and row['sampling_version'] == config['sampling_version']
    # Re-numbered handpose files are never used to identify a source video.
    metadata = json.loads(Path(task['metadata']).read_text())['info']
    n, fps = row['source_frames'], row['source_fps']
    assert metadata['episode_index'] == task['episode_index']
    assert n == metadata['length'] == task['source_frames_metadata']
    assert np.isclose(fps, metadata['capture_fps'], rtol=1e-5, atol=1e-5)
    assert np.isclose(task['source_fps_metadata'], metadata['capture_fps'], rtol=1e-5, atol=1e-5)
    assert Path(task['video']).stat().st_size > 0
    expected = sample_indices(n, fps, config['fps'])
    with np.load(str(dest) + '.npz', allow_pickle=False) as z:
        assert np.array_equal(z['frame_indices'], expected)
        assert expected[0] == 0 and expected[-1] == n - 1
        assert len(z['c2w']) == len(z['intrinsics']) == len(expected) == row['sampled_frames']
        assert z['c2w'].shape[-2:] == (4, 4)
        assert z['intrinsics'].shape[-2:] == (3, 3)
        assert np.isfinite(z['c2w']).all() and np.isfinite(z['intrinsics']).all()
        assert np.allclose(z['timestamps_s'], expected / fps)
        assert np.isclose(float(z['source_fps']), fps)
    row.update(original_config=row.get('config'), config=dict(config),
               include_last_frame=True, sampling_version=config['sampling_version'],
               task=task, reused_camera=True, reused_from=remote, reuse_kind=kind,
               reuse_validation='sha256_core_config_source_metadata_exact_sampling_finite_v2')
    for key in ['remote', 'readback_verified', 'completed_unix', 'upload_s', 'task_wall_s',
                'camera_pose_path', 'last_frame_path']:
        row.pop(key, None)
    return row
