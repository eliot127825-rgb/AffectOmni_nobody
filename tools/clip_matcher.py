"""
CLIP image-text matching module.
Uses CLIP/OpenCLIP for event-to-video-frame similarity matching.
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
    """CLIP image-text matcher"""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        use_original_clip: bool = False
    ):
        """
        Initialize the CLIP matcher.
        
        Args:
            model_name: Model name, e.g. "ViT-B-32", "ViT-L-14"
            pretrained: Pretrained weights, e.g. "openai", "laion2b_s34b_b79k"
            device: Device "cuda" or "cpu"
            use_original_clip: Whether to force using the original CLIP (offline-friendly)
        """
        self.device = device
        
        # Prefer original CLIP (offline-friendly) or choose based on parameter
        if use_original_clip or not HAS_OPEN_CLIP:
            if HAS_CLIP:
                self._init_clip(model_name)
            else:
                raise RuntimeError("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        else:
            # Try open_clip, fallback on failure
            try:
                self._init_open_clip(model_name, pretrained)
            except Exception as e:
                warnings.warn(f"OpenCLIP initialization failed: {e}. Falling back to original CLIP.")
                if HAS_CLIP:
                    self._init_clip(model_name)
                else:
                    raise RuntimeError("Neither open_clip nor clip is available.")
        
    def _init_open_clip(self, model_name: str, pretrained: str):
        """Initialize OpenCLIP"""
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
        """Initialize original CLIP (fallback)"""
        self.backend = "clip"
        # Original CLIP model name format: ViT-B/32, ViT-L/14, RN50, etc.
        # Only replace the second '-' with '/' (e.g. ViT-B-32 -> ViT-B/32)
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
        Match events to the most similar frames.
        
        Args:
            events: List of event descriptions, e.g. ["woman picks up rose", "man smiles"]
            frames_pil: List of video frames (PIL.Image)
            return_scores: Whether to return similarity scores
        
        Returns:
            event_to_frame: {event: best_frame_id}
            If return_scores=True, returns {event: (best_frame_id, score)}
        """
        if not events or not frames_pil:
            return {}
        
        # Encode all frames
        frame_features = self._encode_images(frames_pil)  # (N_frames, D)
        
        # Encode all events
        text_features = self._encode_texts(events)  # (N_events, D)
        
        # Compute similarity matrix
        # (N_events, N_frames)
        similarity_matrix = text_features @ frame_features.T
        
        # Find the best matching frame for each event
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
        Get the full similarity matrix (for DP constraints).
        
        Returns:
            similarity_matrix: (N_events, N_frames) similarity matrix
        """
        frame_features = self._encode_images(frames_pil)
        text_features = self._encode_texts(events)
        return text_features @ frame_features.T
    
    def _encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Batch encode images"""
        import torch
        
        # Preprocess images
        image_inputs = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        # Encode
        with torch.no_grad():
            if self.backend == "open_clip":
                image_features = self.model.encode_image(image_inputs)
            else:  # clip
                image_features = self.model.encode_image(image_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Batch encode texts"""
        import torch
        
        # Tokenize
        if self.backend == "open_clip":
            text_inputs = self.tokenizer(texts).to(self.device)
        else:  # clip
            text_inputs = clip.tokenize(texts).to(self.device)
        
        # Encode
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()


def match_with_monotonic_constraint(
    similarity_matrix: np.ndarray,
    lambda_smooth: float = 0.3
) -> List[int]:
    """
    Use DP (Viterbi) to enforce monotonically non-decreasing event frame indices.
    
    Args:
        similarity_matrix: (N_events, N_frames) similarity matrix
        lambda_smooth: Smoothing penalty coefficient; larger values favor smoother frame progression
    
    Returns:
        best_frames: Best frame index for each event (length N_events)
    
    Objective:
        maximize: sum_i S[i, f_i] - lambda * |f_i - f_{i-1}|
        constraint: f_i >= f_{i-1}
    """
    N_events, N_frames = similarity_matrix.shape
    
    if N_events == 0:
        return []
    
    # DP table: dp[i][f] = max score for first i events, with event i assigned to frame f
    dp = np.full((N_events, N_frames), -np.inf)
    backtrack = np.zeros((N_events, N_frames), dtype=int)
    
    # Initialization: the first event can be assigned to any frame
    dp[0, :] = similarity_matrix[0, :]
    
    # DP transition
    for i in range(1, N_events):
        for f in range(N_frames):
            # Event i is assigned to frame f
            # Event i-1 can only be assigned to frames <= f (monotonic constraint)
            for f_prev in range(f + 1):
                # Transition cost: similarity - frame jump penalty
                transition_score = dp[i-1, f_prev] - lambda_smooth * abs(f - f_prev)
                score = similarity_matrix[i, f] + transition_score
                
                if score > dp[i, f]:
                    dp[i, f] = score
                    backtrack[i, f] = f_prev
    
    # Backtrack to find the optimal path
    best_frames = []
    best_last_frame = np.argmax(dp[N_events - 1, :])
    
    # Backtrack from end to start
    f = best_last_frame
    for i in range(N_events - 1, -1, -1):
        best_frames.append(f)
        if i > 0:
            f = backtrack[i, f]
    
    best_frames.reverse()
    return best_frames


# Convenience function
def create_matcher(
    model_name: str = "ViT-B-32",
    device: str = "cuda"
) -> CLIPMatcher:
    """Convenience function to create a CLIP matcher"""
    return CLIPMatcher(model_name=model_name, device=device)
