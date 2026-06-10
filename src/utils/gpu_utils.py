"""
GPU memory management and mixed-precision helpers for H200 GPUs.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        gpu_id: GPU index (0 or 1). If None, uses CUDA if available, else CPU.

    Returns:
        torch.device
    """
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu_id}")
        else:
            logger.warning(
                f"GPU {gpu_id} not available ({torch.cuda.device_count()} GPUs found). "
                f"Falling back to CPU."
            )
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_amp_dtype(config_dtype: str = "bfloat16") -> torch.dtype:
    """
    Map dtype string to torch.dtype for AMP.

    Args:
        config_dtype: "bfloat16" or "float16"

    Returns:
        torch.dtype
    """
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(config_dtype, torch.bfloat16)


@contextmanager
def autocast_ctx(device: torch.device, use_amp: bool, dtype: torch.dtype = torch.bfloat16):
    """
    Context manager for automatic mixed precision.

    On H200, uses bfloat16 for optimal throughput.

    Usage:
        with autocast_ctx(device, use_amp=True):
            output = model(input)
    """
    if use_amp and device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=dtype):
            yield
    else:
        yield


def free_memory(model: Optional[nn.Module] = None):
    """
    Free GPU memory by clearing cache and running GC.

    Args:
        model: Optional model to move to CPU before clearing.
    """
    if model is not None:
        model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_gpu_memory(device: torch.device, prefix: str = ""):
    """
    Log current GPU memory usage.

    Args:
        device: CUDA device to report
        prefix: String prefix for log message
    """
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    logger.info(
        f"{prefix}GPU {device.index}: "
        f"{allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total"
    )


def set_memory_fraction(fraction: float = 0.95, device: Optional[torch.device] = None):
    """
    Set the fraction of GPU memory PyTorch may use.

    Useful for running multiple processes on the same GPU.
    """
    if torch.cuda.is_available():
        gpu_id = device.index if device is not None else 0
        torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Move all tensors in a batch dict to the given device.

    Args:
        batch: Dict of {key: tensor}
        device: Target device

    Returns:
        Dict with tensors on target device
    """
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class GradScalerWrapper:
    """
    Wraps torch.cuda.amp.GradScaler for mixed-precision training.
    Falls back gracefully when AMP is disabled or on CPU.
    """

    def __init__(self, use_amp: bool = True, device: torch.device = None):
        self.use_amp = use_amp and (device is not None) and (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.use_amp:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer):
        if self.use_amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        if self.use_amp:
            self.scaler.update()

    def unscale_(self, optimizer):
        if self.use_amp:
            self.scaler.unscale_(optimizer)
