"""
视频处理工具模块
提供统一的视频采帧和时间戳计算功能
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
    统一的视频采帧函数，确保与模型推理时的采样策略一致
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大采样帧数（应与推理时的max_frames一致）
        strategy: 采样策略，目前支持 "uniform"（均匀采样）
    
    Returns:
        frames_pil: PIL.Image 列表
        frame_ids: 帧在原视频中的索引（从0开始）
        timestamps: 每帧对应的时间戳（秒）
        fps: 视频的 FPS
    
    注意：
        - 采样策略必须与模型推理时一致，避免"模型看的是A帧，对齐用的是B帧"
        - 优先使用 decord（更快），否则使用 cv2
    """
    if strategy != "uniform":
        raise NotImplementedError(f"Strategy '{strategy}' not implemented yet")
    
    # 优先使用 decord
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
    """使用 decord 采样视频帧"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    # 均匀采样
    if total_frames <= max_frames:
        frame_ids = list(range(total_frames))
    else:
        # 均匀间隔采样
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frame_ids = indices.tolist()
    
    # 读取帧
    frames_np = vr.get_batch(frame_ids).asnumpy()  # (N, H, W, C)
    frames_pil = [Image.fromarray(frame) for frame in frames_np]
    
    # 计算时间戳
    timestamps = [frame_id / fps for frame_id in frame_ids]
    
    return frames_pil, frame_ids, timestamps, fps


def _sample_frames_cv2(
    video_path: str,
    max_frames: int
) -> Tuple[List[Image.Image], List[int], List[float], float]:
    """使用 cv2 采样视频帧（fallback）"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 均匀采样
    if total_frames <= max_frames:
        frame_ids = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frame_ids = indices.tolist()
    
    # 读取帧
    frames_pil = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            # cv2 读取的是 BGR，转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(frame_rgb))
        else:
            warnings.warn(f"Failed to read frame {frame_id}")
    
    cap.release()
    
    # 计算时间戳
    timestamps = [frame_id / fps for frame_id in frame_ids]
    
    return frames_pil, frame_ids, timestamps, fps


def get_video_info(video_path: str) -> dict:
    """
    获取视频基本信息
    
    Returns:
        dict: {
            'total_frames': int,
            'fps': float,
            'duration': float (秒),
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
