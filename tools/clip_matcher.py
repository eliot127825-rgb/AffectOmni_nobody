"""
CLIP 图文匹配模块
使用 CLIP/OpenCLIP 进行事件与视频帧的相似度匹配
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import warnings

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    warnings.warn("open_clip not available, trying clip")

try:
    import clip
    import torch
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class CLIPMatcher:
    """CLIP 图文匹配器"""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        use_original_clip: bool = False
    ):
        """
        初始化 CLIP 匹配器
        
        Args:
            model_name: 模型名称，如 "ViT-B-32", "ViT-L-14"
            pretrained: 预训练权重，如 "openai", "laion2b_s34b_b79k"
            device: 设备 "cuda" 或 "cpu"
            use_original_clip: 是否强制使用原版 CLIP（离线友好）
        """
        self.device = device
        
        # 优先使用原版 CLIP（离线友好）或根据参数选择
        if use_original_clip or not HAS_OPEN_CLIP:
            if HAS_CLIP:
                self._init_clip(model_name)
            else:
                raise RuntimeError("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        else:
            # 尝试使用 open_clip，失败则 fallback
            try:
                self._init_open_clip(model_name, pretrained)
            except Exception as e:
                warnings.warn(f"OpenCLIP initialization failed: {e}. Falling back to original CLIP.")
                if HAS_CLIP:
                    self._init_clip(model_name)
                else:
                    raise RuntimeError("Neither open_clip nor clip is available.")
        
    def _init_open_clip(self, model_name: str, pretrained: str):
        """初始化 OpenCLIP"""
        self.backend = "open_clip"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        print(f"✅ Loaded OpenCLIP: {model_name} ({pretrained})")
    
    def _init_clip(self, model_name: str):
        """初始化原版 CLIP（fallback）"""
        self.backend = "clip"
        # 原版 CLIP 模型名称格式：ViT-B/32, ViT-L/14, RN50 等
        # 只替换第二个 - 为 /（例如 ViT-B-32 -> ViT-B/32）
        if model_name.count('-') >= 2:
            parts = model_name.split('-')
            clip_model_name = f"{parts[0]}-{parts[1]}/{'-'.join(parts[2:])}"
        else:
            clip_model_name = model_name
        
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.model.eval()
        print(f"✅ Loaded CLIP: {clip_model_name}")
    
    def match_events_to_frames(
        self,
        events: List[str],
        frames_pil: List[Image.Image],
        return_scores: bool = False
    ) -> Dict[str, int]:
        """
        将事件匹配到最相似的帧
        
        Args:
            events: 事件描述列表，如 ["woman picks up rose", "man smiles"]
            frames_pil: 视频帧列表（PIL.Image）
            return_scores: 是否返回相似度分数
        
        Returns:
            event_to_frame: {event: best_frame_id}
            如果 return_scores=True，还返回 {event: (best_frame_id, score)}
        """
        if not events or not frames_pil:
            return {}
        
        # 编码所有帧
        frame_features = self._encode_images(frames_pil)  # (N_frames, D)
        
        # 编码所有事件
        text_features = self._encode_texts(events)  # (N_events, D)
        
        # 计算相似度矩阵
        # (N_events, N_frames)
        similarity_matrix = text_features @ frame_features.T
        
        # 为每个事件找到最匹配的帧
        best_frames = np.argmax(similarity_matrix, axis=1)
        
        if return_scores:
            best_scores = np.max(similarity_matrix, axis=1)
            return {
                event: (int(best_frames[i]), float(best_scores[i]))
                for i, event in enumerate(events)
            }
        else:
            return {
                event: int(best_frames[i])
                for i, event in enumerate(events)
            }
    
    def get_similarity_matrix(
        self,
        events: List[str],
        frames_pil: List[Image.Image]
    ) -> np.ndarray:
        """
        获取完整的相似度矩阵（用于 DP 约束）
        
        Returns:
            similarity_matrix: (N_events, N_frames) 的相似度矩阵
        """
        frame_features = self._encode_images(frames_pil)
        text_features = self._encode_texts(events)
        return text_features @ frame_features.T
    
    def _encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """批量编码图像"""
        import torch
        
        # 预处理图像
        image_inputs = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        # 编码
        with torch.no_grad():
            if self.backend == "open_clip":
                image_features = self.model.encode_image(image_inputs)
            else:  # clip
                image_features = self.model.encode_image(image_inputs)
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        import torch
        
        # Tokenize
        if self.backend == "open_clip":
            text_inputs = self.tokenizer(texts).to(self.device)
        else:  # clip
            text_inputs = clip.tokenize(texts).to(self.device)
        
        # 编码
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()


def match_with_monotonic_constraint(
    similarity_matrix: np.ndarray,
    lambda_smooth: float = 0.3
) -> List[int]:
    """
    使用 DP（Viterbi）强制事件帧号单调非递减
    
    Args:
        similarity_matrix: (N_events, N_frames) 相似度矩阵
        lambda_smooth: 平滑惩罚系数，越大越倾向于帧号平滑增长
    
    Returns:
        best_frames: 每个事件的最佳帧号 (长度为 N_events)
    
    目标函数：
        maximize: sum_i S[i, f_i] - lambda * |f_i - f_{i-1}|
        constraint: f_i >= f_{i-1}
    """
    N_events, N_frames = similarity_matrix.shape
    
    if N_events == 0:
        return []
    
    # DP 表: dp[i][f] = 前 i 个事件，第 i 个事件选择帧 f 的最大得分
    dp = np.full((N_events, N_frames), -np.inf)
    backtrack = np.zeros((N_events, N_frames), dtype=int)
    
    # 初始化：第一个事件可以选任意帧
    dp[0, :] = similarity_matrix[0, :]
    
    # DP 转移
    for i in range(1, N_events):
        for f in range(N_frames):
            # 第 i 个事件选择帧 f
            # 第 i-1 个事件只能选择 <= f 的帧（单调约束）
            for f_prev in range(f + 1):
                # 转移代价：相似度 - 帧跳跃惩罚
                transition_score = dp[i-1, f_prev] - lambda_smooth * abs(f - f_prev)
                score = similarity_matrix[i, f] + transition_score
                
                if score > dp[i, f]:
                    dp[i, f] = score
                    backtrack[i, f] = f_prev
    
    # 回溯找到最优路径
    best_frames = []
    best_last_frame = np.argmax(dp[N_events - 1, :])
    
    # 从后往前回溯
    f = best_last_frame
    for i in range(N_events - 1, -1, -1):
        best_frames.append(f)
        if i > 0:
            f = backtrack[i, f]
    
    best_frames.reverse()
    return best_frames


# 便捷函数
def create_matcher(
    model_name: str = "ViT-B-32",
    device: str = "cuda"
) -> CLIPMatcher:
    """创建 CLIP 匹配器的便捷函数"""
    return CLIPMatcher(model_name=model_name, device=device)
