"""Register LVSA as a diffusion attention backend in vllm-omni.

Call ``register_lvsa_backend()`` before launching the server to add
LVSA to the ``DiffusionAttentionBackendEnum`` and backend registry.

Usage from CLI::

    python -c "from lvsa_vllm_omni.register import register_lvsa_backend; register_lvsa_backend()" && vllm-omni serve ...

Or via the wrapper entry point::

    python -m lvsa_vllm_omni.serve /models/Wan2.1-T2V-1.3B-Diffusers --port 8100
"""

import os


def register_lvsa_backend():
    """Register LVSA in vllm-omni's backend enum and registry.

    This is the entry point called by ``vllm_omni.plugins.load_omni_general_plugins``.
    vllm-omni invokes this in the main process, engine process, AND every
    diffusion worker — so we can trigger all the LVSA installation steps here.

    Steps:
    1. Extend ``DiffusionAttentionBackendEnum`` with ``LVSA``.
    2. If ``LVSA_HUNYUAN_HOOK=1``, patch ``HunyuanVideo15Attention`` for
       dual-stream LVSA.
    3. If ``LVSA_WAN_HOOK=1``, patch ``WanSelfAttention`` for LVSA on
       pre-sharded sequences.

    Multi-GPU (Ring SP / Ulysses SP) falls through to vllm-omni's default
    parallel attention path; LVSA's per-rank LVSA pattern is not applied at
    ring degree > 1 in this release.
    """
    from aenum import extend_enum
    from vllm_omni.diffusion.attention.backends.registry import (
        DiffusionAttentionBackendEnum,
        register_diffusion_backend,
    )

    # Add LVSA to the enum if not already present
    if not hasattr(DiffusionAttentionBackendEnum, "LVSA"):
        extend_enum(
            DiffusionAttentionBackendEnum,
            "LVSA",
            "lvsa_vllm_omni.backend.LVSABackend",
        )

    # Register in the backend loader
    register_diffusion_backend(
        DiffusionAttentionBackendEnum.LVSA,
        "lvsa_vllm_omni.backend.LVSABackend",
    )

    # Trigger the optional monkey-patches. Each function is env-var guarded
    # and a no-op if the trigger is not set.
    maybe_install_hunyuan_hook()
    maybe_install_wan_hook()


def maybe_install_hunyuan_hook():
    """Install HunyuanVideo LVSA hook if LVSA_HUNYUAN_HOOK=1.

    Patches HunyuanVideo15Attention at the class level so that
    all instances (including those created in worker processes)
    use the LVSA forward path.

    Must be called before the model is instantiated.
    Reads total_latent_frames from LVSA_TOTAL_LATENT_FRAMES env var at forward time.
    """
    if os.environ.get("LVSA_HUNYUAN_HOOK", "").lower() not in ("1", "true", "yes"):
        return

    try:
        from lvsa_vllm_omni.hunyuan_hook import install_hunyuan_lvsa_hook
        T_lat = os.environ.get("LVSA_TOTAL_LATENT_FRAMES")
        if T_lat:
            install_hunyuan_lvsa_hook(total_latent_frames=int(T_lat))
        else:
            print("[LVSA] Warning: LVSA_HUNYUAN_HOOK=1 but LVSA_TOTAL_LATENT_FRAMES not set")
    except Exception as e:
        print(f"[LVSA] Hook installation failed: {e}")


def maybe_install_wan_hook():
    """Install Wan LVSA hook if LVSA_WAN_HOOK=1.

    Patches WanSelfAttention at the class level. Required for Wan because
    vllm-omni's _sp_plan pre-shards sequences before the attention backend
    sees them, breaking LVSA's geometry detection. Hooking at the attention
    module gets us the full sequence.

    Must be called before the model is instantiated.
    """
    if os.environ.get("LVSA_WAN_HOOK", "").lower() not in ("1", "true", "yes"):
        return

    try:
        from lvsa_vllm_omni.wan_hook import install_wan_lvsa_hook
        T_lat = os.environ.get("LVSA_TOTAL_LATENT_FRAMES")
        if T_lat:
            install_wan_lvsa_hook(total_latent_frames=int(T_lat))
        else:
            print("[LVSA] Warning: LVSA_WAN_HOOK=1 but LVSA_TOTAL_LATENT_FRAMES not set")
    except Exception as e:
        print(f"[LVSA] Wan hook installation failed: {e}")
