"""Shared test utilities: optional-dependency guards and script loading."""

from __future__ import annotations

import importlib
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


HAS_TORCH = _importable("torch")
HAS_TRANSFORMERS = HAS_TORCH and _importable("transformers")
HAS_WANDB = _importable("wandb")

requires_torch = unittest.skipUnless(HAS_TORCH, "requires torch")
requires_transformers = unittest.skipUnless(HAS_TRANSFORMERS, "requires torch and transformers")
requires_wandb = unittest.skipUnless(HAS_WANDB, "requires wandb")


def load_script(relative_path: str) -> ModuleType:
    """Import a repository script (e.g. ``scripts/core7/x.py``) as a module."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
