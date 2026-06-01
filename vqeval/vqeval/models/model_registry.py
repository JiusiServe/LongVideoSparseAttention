"""Shared model registry with lazy loading to avoid duplicate VRAM usage.

Uses open_clip (instead of transformers) for CLIP, torch.hub for DINOv2,
and OpenCV Farneback for optical flow. This avoids segfaults on Python 3.14
caused by transformers/torchvision C extension incompatibilities.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import cv2
import numpy as np
import torch


class ModelRegistry:
    """Singleton registry for shared ML models.

    Ensures each model (CLIP, DINO, etc.) is loaded only once
    and shared across evaluators. Models are lazy-loaded on first request.
    """

    _instance: Optional[ModelRegistry] = None
    _lock = threading.Lock()

    def __new__(cls, device: str = "cuda"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, device: str = "cuda"):
        if self._initialized:
            return
        self.device = device
        self._models: dict[str, Any] = {}
        self._processors: dict[str, Any] = {}
        self._initialized = True

    def get_clip_model(self) -> tuple:
        """Get CLIP ViT-B/16 model and tokenizer via open_clip."""
        if "clip" not in self._models:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai", device=self.device
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-16")
            self._models["clip"] = model
            self._processors["clip_preprocess"] = preprocess
            self._processors["clip_tokenizer"] = tokenizer
        return self._models["clip"], self._processors["clip_preprocess"]

    def get_clip_tokenizer(self):
        """Get the CLIP tokenizer (loads model if needed)."""
        if "clip_tokenizer" not in self._processors:
            self.get_clip_model()
        return self._processors["clip_tokenizer"]

    def get_dino_model(self) -> Any:
        """Get DINOv2 ViT-B/14 model via torch.hub."""
        if "dino" not in self._models:
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True
            )
            model = model.to(self.device)
            model.eval()
            self._models["dino"] = model
        return self._models["dino"]

    def get_pyiqa_metric(self, metric_name: str):
        """Get a pyiqa metric model."""
        key = f"pyiqa_{metric_name}"
        if key not in self._models:
            import pyiqa

            model = pyiqa.create_metric(metric_name, device=self.device)
            self._models[key] = model
        return self._models[key]

    def get_lpips_model(self):
        """Get LPIPS perceptual distance model."""
        if "lpips" not in self._models:
            import lpips

            model = lpips.LPIPS(net="alex").to(self.device)
            model.eval()
            self._models["lpips"] = model
        return self._models["lpips"]

    @torch.no_grad()
    def compute_clip_image_embeddings(
        self, images_tensor: torch.Tensor, batch_size: int = 16
    ) -> torch.Tensor:
        """Compute CLIP visual embeddings for a batch of images.

        Args:
            images_tensor: (N, 3, H, W) float tensor normalized [0,1]
            batch_size: batch size for inference

        Returns:
            (N, D) normalized embedding tensor
        """
        model, _ = self.get_clip_model()
        all_embeds = []

        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i : i + batch_size]
            # Resize and normalize for CLIP
            batch_resized = torch.nn.functional.interpolate(
                batch, size=(224, 224), mode="bilinear", align_corners=False
            )
            # CLIP normalization (openai stats)
            mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                device=batch.device,
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                device=batch.device,
            ).view(1, 3, 1, 1)
            batch_norm = (batch_resized - mean) / std

            embeds = model.encode_image(batch_norm)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.float())

        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def compute_dino_embeddings(
        self, images_tensor: torch.Tensor, batch_size: int = 16
    ) -> torch.Tensor:
        """Compute DINOv2 embeddings for a batch of images.

        Args:
            images_tensor: (N, 3, H, W) float tensor normalized [0,1]
            batch_size: batch size for inference

        Returns:
            (N, D) normalized embedding tensor
        """
        model = self.get_dino_model()
        all_embeds = []

        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i : i + batch_size]
            batch_resized = torch.nn.functional.interpolate(
                batch, size=(224, 224), mode="bilinear", align_corners=False
            )
            # DINOv2 normalization (ImageNet stats)
            mean = torch.tensor(
                [0.485, 0.456, 0.406], device=batch.device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [0.229, 0.224, 0.225], device=batch.device
            ).view(1, 3, 1, 1)
            batch_norm = (batch_resized - mean) / std

            # DINOv2 via torch.hub returns a dict with 'x_norm_clstoken'
            # or just the CLS token depending on the version
            outputs = model(batch_norm)
            if isinstance(outputs, dict):
                embeds = outputs.get(
                    "x_norm_clstoken", outputs.get("cls_token", None)
                )
                if embeds is None:
                    # Fallback: use the first key
                    embeds = next(iter(outputs.values()))
            else:
                # Direct tensor output (CLS token)
                embeds = outputs

            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.float())

        return torch.cat(all_embeds, dim=0)

    def compute_optical_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        max_size: int = 480,
    ) -> torch.Tensor:
        """Compute optical flow between two frames using OpenCV Farneback.

        Uses CPU-based Farneback algorithm — stable on all Python versions
        and doesn't consume VRAM.

        Args:
            frame1, frame2: (1, 3, H, W) or (3, H, W) float tensors [0,1]
            max_size: downscale the longest edge before computing flow

        Returns:
            (1, 2, H, W) flow field tensor on CPU
        """
        if frame1.dim() == 4:
            frame1 = frame1[0]
        if frame2.dim() == 4:
            frame2 = frame2[0]

        # Convert to numpy uint8 grayscale
        f1_np = (frame1.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        f2_np = (frame2.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Downscale if needed
        h, w = f1_np.shape[:2]
        longest = max(h, w)
        if longest > max_size:
            scale = max_size / longest
            new_w = int(w * scale)
            new_h = int(h * scale)
            f1_np = cv2.resize(f1_np, (new_w, new_h))
            f2_np = cv2.resize(f2_np, (new_w, new_h))

        gray1 = cv2.cvtColor(f1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2_np, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # flow shape: (H, W, 2) -> convert to (1, 2, H, W) tensor
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
        return flow_tensor

    @torch.no_grad()
    def compute_clip_text_embedding(self, text: str) -> torch.Tensor:
        """Compute CLIP text embedding for a given string."""
        model, _ = self.get_clip_model()
        tokenizer = self.get_clip_tokenizer()

        tokens = tokenizer([text]).to(self.device)
        embeds = model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds.float()

    def clear(self):
        """Release all models and free VRAM."""
        self._models.clear()
        self._processors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def reset(cls):
        """Reset the singleton (mainly for testing)."""
        cls._instance = None
