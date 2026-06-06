"""Utility functions for reproducible WQRI experiments."""

from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix random seeds for Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed_everything] seed: {seed}")


def clear_memory() -> None:
    """Release Python and CUDA cached memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory if it does not already exist and return it as a Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
