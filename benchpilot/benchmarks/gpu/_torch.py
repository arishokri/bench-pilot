from __future__ import annotations


def torch_or_skip():
    """Return torch module or None. Caller emits a 'skipped' result if None."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    return torch


def cuda_or_skip():
    t = torch_or_skip()
    if t is None or not t.cuda.is_available():
        return None
    return t
