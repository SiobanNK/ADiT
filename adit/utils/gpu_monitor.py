import os
import threading
from typing import List, Optional

import lightning as L
from lightning import Callback

from adit.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

try:
    import pynvml
except ImportError:
    pynvml = None


class JobGPUMonitor(Callback):
    """Logs utilization/memory for every GPU assigned to this SLURM job
    (as many as CUDA_VISIBLE_DEVICES lists), from a single process (rank 0),
    to whatever logger(s) are configured (W&B, TensorBoard, CSV, ...).
    Bypasses W&B's default `system.gpu.*` metrics, which report every GPU
    on the node regardless of job allocation, and works identically when
    W&B isn't used at all.
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

        loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
        loggers = [lg for lg in loggers if lg is not None]
        if not loggers:
            log.warning("No logger configured, skipping GPU monitor.")
            return

        gpu_indices = self._resolve_gpu_indices()
        if not gpu_indices:
            log.warning("Could not resolve assigned GPU indices, skipping GPU monitor.")
            return

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._log_loop, args=(trainer, loggers, gpu_indices, self._stop_event), daemon=True
        )
        self._thread.start()
        log.info(f"Started JobGPUMonitor on physical GPU indices {gpu_indices}")

    def _stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

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

    def _log_loop(self, trainer: L.Trainer, loggers: list, gpu_indices: List[int], stop_event: threading.Event) -> None:
        pynvml.nvmlInit()
        handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices}
        while not stop_event.is_set():
            metrics = {}
            for i, handle in handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics[f"my_gpu/gpu{i}/utilization_pct"] = util.gpu
                    metrics[f"my_gpu/gpu{i}/memory_used_MB"] = mem.used / 1024**2
                    metrics[f"my_gpu/gpu{i}/memory_total_MB"] = mem.total / 1024**2
                    metrics[f"my_gpu/gpu{i}/memory_used_pct"] = 100 * mem.used / mem.total   # <-- new
                except Exception as e:
                    log.warning(f"GPU monitor failed on index {i}: {e}")
            if metrics:
                metrics["my_gpu/avg_utilization_pct"] = sum(
                    v for k, v in metrics.items() if k.endswith("utilization_pct") and "avg" not in k
                ) / len(gpu_indices)
                metrics["my_gpu/avg_memory_used_pct"] = sum(
                    v for k, v in metrics.items() if k.endswith("memory_used_pct")
                ) / len(gpu_indices)
                step = trainer.global_step
                for lg in loggers:
                    try:
                        lg.log_metrics(metrics, step=step)
                    except Exception as e:
                        log.warning(f"Failed to log GPU metrics to {type(lg).__name__}: {e}")
            stop_event.wait(self.interval)
        pynvml.nvmlShutdown()
