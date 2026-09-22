"""Time-based sampling that always retains both episode endpoints."""
import numpy as np


def sample_indices(num_frames, source_fps, target_fps):
    if num_frames < 1 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError('Invalid video length or source FPS')
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError('Invalid target FPS')
    fps = min(source_fps, target_fps)
    indices = np.floor(np.arange(0, num_frames / source_fps, 1 / fps) * source_fps + .5).astype(np.int64)
    return np.unique(np.append(indices[indices < num_frames], num_frames - 1))
