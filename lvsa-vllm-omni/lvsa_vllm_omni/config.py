"""LVSA configuration for vllm-omni backend."""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LVSAConfig:
    """Configuration for LVSA sparse windowed attention.

    All frame-based parameters are in **video frames** (not latent frames).
    They are divided by ``vae_temporal_factor`` internally when building
    the sparse pattern.

    Can be loaded from environment variables (``LVSA_*``) or a JSON string.
    """

    window_size: int = 12
    n_first_frames: int = 4
    key_frame_interval: int = 16
    auto_keyframes: bool = True
    rotate_keyframes: bool = False
    expand_window: bool = True  # extend local window outward when global frames overlap
    backend: str = "sdpa"  # "sdpa" | "flashinfer"
    vae_temporal_factor: int = 4  # Wan=4, HunyuanVideo=4
    total_latent_frames: Optional[int] = None  # override; auto-detected if None
    sparsity_scale: float = 1.0  # <1 = more sparse, >1 = less sparse (scales auto_kfi target)
    reference_latent_frames: int = 21  # model's training-horizon latent-frame count
    # (Wan 1.3B/14B = 21; HunyuanVideo 1.5 = 33). Determines the per-query
    # attention budget at ≤ reference and must match the model to avoid
    # accidental sparsity when T_lat <= this value.

    @classmethod
    def from_env(cls) -> "LVSAConfig":
        """Build config from ``LVSA_*`` environment variables.

        Reads: LVSA_WINDOW_SIZE, LVSA_N_FIRST_FRAMES, LVSA_KEY_FRAME_INTERVAL,
        LVSA_AUTO_KEYFRAMES, LVSA_ROTATE_KEYFRAMES, LVSA_EXPAND_WINDOW,
        LVSA_BACKEND, LVSA_VAE_TEMPORAL_FACTOR, LVSA_TOTAL_LATENT_FRAMES,
        LVSA_CONFIG (JSON string overriding all others).
        """
        # If LVSA_CONFIG is set, parse it as JSON first
        json_str = os.environ.get("LVSA_CONFIG")
        if json_str:
            return cls.from_json(json_str)

        def _int(key: str, default: int) -> int:
            return int(os.environ.get(key, default))

        def _bool(key: str, default: bool) -> bool:
            val = os.environ.get(key)
            if val is None:
                return default
            return val.lower() in ("1", "true", "yes")

        def _float(key: str, default: float) -> float:
            return float(os.environ.get(key, default))

        def _str(key: str, default: str) -> str:
            return os.environ.get(key, default)

        total_lat = os.environ.get("LVSA_TOTAL_LATENT_FRAMES")

        return cls(
            window_size=_int("LVSA_WINDOW_SIZE", 12),
            n_first_frames=_int("LVSA_N_FIRST_FRAMES", 4),
            key_frame_interval=_int("LVSA_KEY_FRAME_INTERVAL", 16),
            auto_keyframes=_bool("LVSA_AUTO_KEYFRAMES", True),
            rotate_keyframes=_bool("LVSA_ROTATE_KEYFRAMES", False),
            expand_window=_bool("LVSA_EXPAND_WINDOW", True),
            backend=_str("LVSA_BACKEND", "sdpa"),
            vae_temporal_factor=_int("LVSA_VAE_TEMPORAL_FACTOR", 4),
            total_latent_frames=int(total_lat) if total_lat else None,
            sparsity_scale=_float("LVSA_SPARSITY_SCALE", 1.0),
            reference_latent_frames=_int("LVSA_REFERENCE_LATENT_FRAMES", 21),
        )

    @classmethod
    def from_json(cls, s: str) -> "LVSAConfig":
        """Parse a JSON config string into LVSAConfig."""
        data = json.loads(s)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def latent_window_size(self) -> int:
        return self.window_size // self.vae_temporal_factor

    @property
    def latent_n_first_frames(self) -> int:
        return self.n_first_frames // self.vae_temporal_factor

    @property
    def latent_key_frame_interval(self) -> int:
        return self.key_frame_interval // self.vae_temporal_factor


# ── Patches-per-frame resolution ──────────────────────────────────────
# Geometry detection needs a candidate set of "patches per frame" (P) values
# to match against `seq_len = T_lat * P + enc_tokens`. Historically this was
# a hardcoded `{1560}` (Wan + HunyuanVideo at 480p) which silently fell back
# to dense for any other resolution or for tests using small synthetic P.
#
# Resolution order (first match wins):
#   1. ``LVSA_PATCHES_PER_FRAME`` — explicit override (comma-separated list or int).
#      Use this in tests / non-480p configs.
#   2. Derivation from ``LVSA_VIDEO_HEIGHT``, ``LVSA_VIDEO_WIDTH``,
#      ``LVSA_VAE_SPATIAL_FACTOR`` (default 8), ``LVSA_PATCH_SIZE`` (default 2).
#   3. Built-in default: {1560} (Wan/HunyuanVideo at 480×832).

_DEFAULT_PPF = 1560


def candidate_patches_per_frame() -> list[int]:
    """Return the list of candidate P values geometry detection should try.

    Ordered by specificity: explicit env override first, then resolution
    derivation, then the built-in default. Callers iterate until one matches.
    """
    # 1. explicit override
    override = os.environ.get("LVSA_PATCHES_PER_FRAME")
    if override:
        try:
            return [int(x.strip()) for x in override.split(",") if x.strip()]
        except ValueError:
            pass  # fall through to derivation

    # 2. derive from resolution + model geometry
    try:
        h = int(os.environ.get("LVSA_VIDEO_HEIGHT", "0"))
        w = int(os.environ.get("LVSA_VIDEO_WIDTH", "0"))
        sf = int(os.environ.get("LVSA_VAE_SPATIAL_FACTOR", "8"))
        ps = int(os.environ.get("LVSA_PATCH_SIZE", "2"))
        if h > 0 and w > 0 and sf > 0 and ps > 0:
            ph = (h // sf) // ps
            pw = (w // sf) // ps
            if ph > 0 and pw > 0:
                return [ph * pw]
    except ValueError:
        pass

    # 3. built-in default
    return [_DEFAULT_PPF]
