"""Hybrid tests exercise the official implementation or explicitly skip."""
import pytest
import torch


def require_mamba(cuda=True):
    if cuda and not torch.cuda.is_available():
        pytest.skip('Official Mamba-2 CUDA tests require a GPU')
    from models import require_official_mamba
    try:return require_official_mamba()
    except ImportError as error:pytest.skip(str(error))
