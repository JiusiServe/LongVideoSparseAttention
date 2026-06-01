"""Add parent directories to sys.path so lvsa and lvsa_vllm_omni are importable."""
import sys
from pathlib import Path

# lvsa-vllm-omni package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# lvsa package root (sibling directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
