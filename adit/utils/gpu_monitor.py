import os
import threading
from typing import Dict, List, Optional

import torch
import lightning as L
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger

from adit.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

try:
    import pynvml
except ImportError:
    pynvml = None


class JobGPUMonitor(Callback):
    """Logs utilization/memory for every GPU assigned to this SLURM job
    (as many as CUDA_VISIBLE_DEVICES lists), from a single process (rank 0),
    to W&B specifically.

    Only W&B is targeted (not CSVLogger/TensorBoardLogger/etc.): this callback
    logs from a background thread on an interval, concurrently with Lightning's
    own main-thread logging calls. Lightning's built-in loggers (CSVLogger in
    particular) are not thread-safe and can corrupt their internal state under
    concurrent log_metrics() calls, causing spurious
    "dict contains fields not in fieldnames" crashes unrelated to the actual
    metric keys being logged. WandbLogger's underlying run object is safe to
    call from multiple threads, so we restrict ourselves to it.

    Bypasses W&B's default `system.gpu.*` metrics, which report every GPU
    on the node regardless of job allocation.

    NVML enumerates GPUs by their physical node index (e.g. 2, 5 if
    CUDA_VISIBLE_DEVICES="2,5"), while PyTorch always renumbers visible
    devices starting from 0 in CUDA_VISIBLE_DEVICES order (torch device 0 ==
    physical GPU 2, torch device 1 == physical GPU 5 in that example). All
    `my_gpu/gpu{i}/*` keys are keyed by the physical (NVML) index.

    IMPORTANT limitation on torch_* metrics: this callback only runs on rank
    0, and torch.cuda.memory_allocated/reserved are *per-process* counters,
    not GPU-wide like NVML's mem.used. In standard multi-GPU DDP (one
    process per GPU), rank 0 only ever allocates on its own local device —
    so querying torch.cuda.memory_allocated() for a *different* rank's GPU
    from rank 0 would silently return 0, not the real usage on that card
    (NVML's memory_used_MB/pct remain accurate for every GPU regardless).
    To avoid logging a misleading "0 MB used" for GPUs owned by other
    ranks, torch_* metrics are only populated for the single physical GPU
    that rank 0 itself uses (torch.cuda.current_device()); all other GPUs
    keep torch_* as NaN, and NVML metrics remain the source of truth for
    them.
    """

    def __init__(self, interval: float = 15.0) -> None:
        super().__init__()
        self.interval = interval
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._stop()

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._stop()

    def _start(self, trainer: L.Trainer) -> None:
        if pynvml is None:
            log.warning("pynvml not installed, skipping GPU monitor. `pip install nvidia-ml-py`")
            return
        if not trainer.is_global_zero:
            return  # rank 0 polls every GPU of the job; other ranks do nothing

        # Avoid starting a second thread if one is already running (e.g. if
        # on_fit_start/on_test_start ever fire back-to-back without an
        # intervening on_fit_end/on_test_end).
        if self._thread is not None and self._thread.is_alive():
            log.warning("JobGPUMonitor thread already running, skipping restart.")
            return

        all_loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
        # Only target W&B: it's the only logger here guaranteed safe to call
        # concurrently from this background thread. CSVLogger/TensorBoardLogger
        # are not thread-safe and get corrupted by concurrent log_metrics()
        # calls from the main training thread + this monitor thread at once.
        loggers = [lg for lg in all_loggers if isinstance(lg, WandbLogger)]
        if not loggers:
            log.warning("No W&B logger configured, skipping GPU monitor.")
            return

        gpu_indices = self._resolve_gpu_indices()
        if not gpu_indices:
            log.warning("Could not resolve assigned GPU indices, skipping GPU monitor.")
            return

        local_nvml_idx = self._resolve_local_nvml_index(gpu_indices)
        if local_nvml_idx is None:
            log.warning(
                "Could not determine which physical GPU is local to rank 0 "
                "(mismatch between CUDA_VISIBLE_DEVICES and torch.cuda.device_count()); "
                "torch_* memory metrics will be skipped for all GPUs."
            )

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._log_loop,
            args=(trainer, loggers, gpu_indices, local_nvml_idx, self._stop_event),
            daemon=True,
        )
        self._thread.start()
        log.info(
            f"Started JobGPUMonitor on physical GPU indices {gpu_indices} "
            f"(torch_* metrics limited to physical GPU {local_nvml_idx}, rank 0's own device)"
        )

    def _stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._stop_event = None

    @staticmethod
    def _resolve_gpu_indices() -> List[int]:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            try:
                return [int(i) for i in cvd.split(",") if i != ""]
            except ValueError:
                pass
        for var in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
            val = os.environ.get(var)
            if val:
                try:
                    return [int(i) for i in val.split(",") if i != ""]
                except ValueError:
                    pass
        return []

    @staticmethod
    def _resolve_local_nvml_index(gpu_indices: List[int]) -> Optional[int]:
        """Returns the physical/NVML index of the single GPU rank 0 actually
        uses (torch.cuda.current_device()), or None if it can't be resolved.

        Relies on CUDA_VISIBLE_DEVICES order: torch always renumbers visible
        devices starting at 0, in the order they appear in that variable.
        `gpu_indices` is assumed to already be in that same order (see
        `_resolve_gpu_indices`), so gpu_indices[torch_idx] gives the physical
        index for a given torch device index.
        """
        try:
            torch_idx = torch.cuda.current_device()
        except Exception as e:
            log.warning(f"torch.cuda.current_device() failed: {e}")
            return None

        if torch_idx >= len(gpu_indices):
            log.warning(
                f"torch.cuda.current_device()={torch_idx} is out of range for resolved "
                f"physical GPU indices {gpu_indices}; CUDA_VISIBLE_DEVICES and torch's "
                f"view of devices appear inconsistent."
            )
            return None

        return gpu_indices[torch_idx]

    def _log_loop(
        self,
        trainer: L.Trainer,
        loggers: list,
        gpu_indices: List[int],
        local_nvml_idx: Optional[int],
        stop_event: threading.Event,
    ) -> None:
        pynvml.nvmlInit()
        handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices}  # index physiques du noeud

        # Fixed key set, computed once: guarantees every logged dict has the
        # exact same schema regardless of which per-GPU NVML/torch calls
        # succeed or fail on a given iteration. Must include every suffix
        # that _log_loop can possibly set below, torch_* included.
        metric_keys = [
            f"my_gpu/gpu{i}/{suffix}"
            for i in gpu_indices
            for suffix in (
                "utilization_pct",
                "memory_used_MB",
                "memory_total_MB",
                "memory_used_pct",
                "torch_allocated_MB",
                "torch_reserved_MB",
                "torch_max_allocated_MB",
            )
        ] + ["my_gpu/avg_utilization_pct", "my_gpu/avg_memory_used_pct"]

        while not stop_event.is_set():
            metrics: Dict[str, float] = {k: float("nan") for k in metric_keys}
            util_vals, mem_pct_vals = [], []

            for i, handle in handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_pct = 100 * mem.used / mem.total

                    metrics[f"my_gpu/gpu{i}/utilization_pct"] = util.gpu
                    metrics[f"my_gpu/gpu{i}/memory_used_MB"] = mem.used / 1024**2
                    metrics[f"my_gpu/gpu{i}/memory_total_MB"] = mem.total / 1024**2
                    metrics[f"my_gpu/gpu{i}/memory_used_pct"] = mem_used_pct

                    util_vals.append(util.gpu)
                    mem_pct_vals.append(mem_used_pct)
                except Exception as e:
                    log.warning(f"GPU monitor (NVML) failed on physical index {i}: {e}")
                    # Keep the default NaN for this GPU; the keys stay present.

                # torch_* metrics are only meaningful for rank 0's own GPU
                # (see class docstring): torch.cuda.memory_allocated() is a
                # per-process counter, so querying it for a GPU owned by a
                # different rank would silently report 0, not real usage.
                # Every other GPU keeps torch_* as NaN; NVML above already
                # gives an accurate memory_used_MB/pct for them.
                if i == local_nvml_idx:
                    try:
                        metrics[f"my_gpu/gpu{i}/torch_allocated_MB"] = torch.cuda.memory_allocated() / 1024**2
                        metrics[f"my_gpu/gpu{i}/torch_reserved_MB"] = torch.cuda.memory_reserved() / 1024**2
                        metrics[f"my_gpu/gpu{i}/torch_max_allocated_MB"] = (
                            torch.cuda.max_memory_allocated() / 1024**2
                        )
                    except Exception as e:
                        log.warning(f"GPU monitor (torch) failed on rank 0's own GPU (physical index {i}): {e}")

            if util_vals:
                metrics["my_gpu/avg_utilization_pct"] = sum(util_vals) / len(util_vals)
            if mem_pct_vals:
                metrics["my_gpu/avg_memory_used_pct"] = sum(mem_pct_vals) / len(mem_pct_vals)

            step = trainer.global_step
            for lg in loggers:
                try:
                    lg.log_metrics(metrics, step=step)
                except Exception as e:
                    log.warning(f"Failed to log GPU metrics to {type(lg).__name__}: {e}")

            stop_event.wait(self.interval)
        pynvml.nvmlShutdown()
