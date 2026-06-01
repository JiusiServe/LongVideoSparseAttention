"""Shared fixtures and path setup for LVSA tests."""

import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `from lvsa.xxx import ...` works without installation.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
