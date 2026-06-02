"""Launch vllm serve --omni with LVSA backend pre-registered.

Usage::

    python -m lvsa_vllm_omni.serve MODEL --port 8091 --dtype bfloat16

Equivalent to::

    vllm serve MODEL --omni --port 8091 --dtype bfloat16

but with LVSA registered in the attention backend enum first.
"""
import sys


def main():
    # Register LVSA backend before any vllm-omni import
    from lvsa_vllm_omni.register import register_lvsa_backend
    register_lvsa_backend()

    # Build argv: vllm serve MODEL --omni [rest of args]
    # sys.argv = [serve.py, MODEL, --port, 8091, ...]
    # We need: [vllm, serve, MODEL, --omni, --port, 8091, ...]
    args = sys.argv[1:]

    # Inject "serve" and "--omni"
    sys.argv = [sys.argv[0], "serve"] + args
    if "--omni" not in sys.argv:
        # Insert --omni after "serve"
        sys.argv.insert(2, "--omni")

    # Select LVSA for the self-attention role (vllm-omni 0.22 replaced the
    # DIFFUSION_ATTENTION_BACKEND env var with the per-role AttentionConfig).
    # Skip if the user already passed their own attention-backend flag.
    _has_attn_flag = any(
        a == "--diffusion-attention-config"
        or a == "-dac"
        or a == "--diffusion-attention-backend"
        or a.startswith("--diffusion-attention-config.")
        or a.startswith("--diffusion-attention-backend")
        for a in sys.argv
    )
    if not _has_attn_flag:
        sys.argv += [
            "--diffusion-attention-config",
            '{"per_role": {"self": {"backend": "LVSA"}}}',
        ]

    # Use vllm's main (vllm-omni's main.py checks for --omni in sys.argv)
    from vllm_omni.entrypoints.cli.main import main as vllm_omni_main
    vllm_omni_main()


if __name__ == "__main__":
    main()
