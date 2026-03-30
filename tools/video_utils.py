"""
Video processing utility module.
Provides unified video frame sampling and timestamp computation.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import warnings

try:
    import decord
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    warnings.warn("decord not available, falling back to cv2")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def sample_frames(
    video_path: str,
    max_frames: int = 32,
    strategy: str = "uniform"
) -> Tuple[List[Image.Image], List[int], List[float], float]:
    """
    Unified video frame sampling function, ensuring consistency with model inference sampling.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to sample (should match inference max_frames)
        strategy: Sampling strategy, currently supports "uniform"
    
    Returns:
        frames_pil: List of PIL.Image
        frame_ids: Frame indices in the original video (0-based)
        timestamps: Timestamp for each frame (seconds)
        fps: Video FPS
    
    Note:
        - Sampling strategy must match model inference to avoid frame mismatch
        - Prefers decord (faster), falls back to cv2
    """
    if strategy != "uniform":
        raise NotImplementedError(f"Strategy '{strategy}' not implemented yet")
    
    # Prefer decord
    if HAS_DECORD:
        return _sample_frames_decord(video_path, max_frames)
    elif HAS_CV2:
        return _sample_frames_cv2(video_path, max_frames)
    else:
        raise RuntimeError("Neither decord nor cv2 is available")


def _sample_frames_decord(
    video_path: str,
    max_frames: int
) -> Tuple[List[Image.Image], List[int], List[float], float]:
    """Sample video frames using decord"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    # Uniform sampling
    if total_frames <= max_frames:
        frame_ids = list(range(total_frames))
    else:
        # Uniformly spaced sampling
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frame_ids = indices.tolist()
    
    # Read frames
    frames_np = vr.get_batch(frame_ids).asnumpy()  # (N, H, W, C)
    frames_pil = [Image.fromarray(frame) for frame in frames_np]
    
    # Compute timestamps
    timestamps = [frame_id / fps for frame_id in frame_ids]
    
    return frames_pil, frame_ids, timestamps, fps


def _sample_frames_cv2(
    video_path: str,
    max_frames: int
) -> Tuple[List[Image.Image], List[int], List[float], float]:
    """Sample video frames using cv2 (fallback)"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Uniform sampling
    if total_frames <= max_frames:
        frame_ids = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frame_ids = indices.tolist()
    
    # Read frames
    frames_pil = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            # cv2 reads BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(frame_rgb))
        else:
            warnings.warn(f"Failed to read frame {frame_id}")
    
    cap.release()
    
    # Compute timestamps
    timestamps = [frame_id / fps for frame_id in frame_ids]
    
    return frames_pil, frame_ids, timestamps, fps


def get_video_info(video_path: str) -> dict:
    """
    Get basic video information.
    
    Returns:
        dict: {
            'total_frames': int,
            'fps': float,
            'duration': float (seconds),
            'width': int,
            'height': int
        }
    """
    if HAS_DECORD:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        width, height = vr[0].shape[1], vr[0].shape[0]
    elif HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        raise RuntimeError("Neither decord nor cv2 is available")
    
    duration = total_frames / fps if fps > 0 else 0
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height
    }
