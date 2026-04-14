"""
Training time and resource tracker.

Records wall-clock time and peak GPU/MPS memory for each model run.
Supports CUDA (NVIDIA) and MPS (Apple Silicon) backends.
"""

import time

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    # MPS: Apple Silicon GPU backend (PyTorch 2.0+)
    HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    HAS_CUDA = False
    HAS_MPS = False


class Timer:
    """Context manager that records elapsed time and peak GPU/MPS memory.

    On CUDA devices, reports true peak memory via reset_peak_memory_stats()
    + max_memory_allocated(). On MPS (Apple Silicon), reports the current
    allocated memory at exit as a proxy (MPS has no peak-memory API).
    Reports 0 on CPU-only runs.

    Usage
    -----
    >>> with Timer() as t:
    ...     model.fit(df)
    >>> print(t.elapsed, t.peak_gpu_mb)
    """

    def __enter__(self):
        if HAS_CUDA:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        elif HAS_MPS:
            torch.mps.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if HAS_CUDA:
            torch.cuda.synchronize()
            self.peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        elif HAS_MPS:
            torch.mps.synchronize()
            # MPS exposes no peak-memory counter; current_allocated_memory()
            # is the closest available proxy (PyTorch 2.0+).
            self.peak_gpu_mb = torch.mps.current_allocated_memory() / 1e6
        else:
            self.peak_gpu_mb = 0.0

    def __repr__(self):
        return f"Timer(elapsed={self.elapsed:.1f}s, gpu={self.peak_gpu_mb:.0f}MB)"
