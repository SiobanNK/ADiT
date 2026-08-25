"""
throughput_profiler.py

PyTorch Lightning callback to determine whether a training run is
CPU/IO-bound (dataloader) or GPU-bound, without instrumenting the
forward/backward code at all.

Method
------
On every training batch we split wall-clock time into:

  data_time : gap between the previous `on_train_batch_end` and the current
              `on_train_batch_start`. This is dataloader fetch + collate +
              any blocking host->device transfer that happens before the
              step starts.
  step_time : time spent inside `on_train_batch_start` -> `on_train_batch_end`,
              i.e. forward + backward + optimizer.step + Lightning-side logging.

Every `log_every_n_steps`, we log the averages and print a verdict:
  - data_time_pct high  -> CPU/IO-bound: more num_workers / prefetch_factor /
                            faster storage / lighter collate will help.
  - data_time_pct low    -> GPU-bound: dataloading isn't the bottleneck,
                            look at batch size, AMP/bf16, model, comm (DDP).

We also sample GPU utilization (via nvidia-smi, through torch.cuda.utilization)
once per window as a cheap sanity check -- it should be consistently high
(~90-100%) when you are truly GPU-bound.

Notes
-----
- Uses wall-clock (CPU-side) timing on purpose: no torch.cuda.synchronize()
  per step, so it adds negligible overhead and is safe to leave on for the
  whole training run.
- `warmup_steps` skips the very first steps, where dataloader worker
  spin-up / cudnn autotuning / first cudaMalloc would pollute the stats.
- Safe under DDP: printing is rank_zero_only, logging uses rank_zero_only=True
  so you get per-rank curves in your logger (useful to spot an unbalanced
  shard or a slow rank on Jean-Zay) without spamming stdout from every rank.
"""

import time
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class ThroughputProfilerCallback(Callback):
    def __init__(self, log_every_n_steps: int = 50, warmup_steps: int = 10):
        """
        Args:
            log_every_n_steps: window size (in steps) over which timings are averaged.
            warmup_steps: steps skipped at the very start of training.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self._reset()
        self._prev_end = None
        self._global_step_seen = 0

    def _reset(self):
        self._data_time_sum = 0.0
        self._step_time_sum = 0.0
        self._n = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._prev_end is not None:
            self._data_time_sum += now - self._prev_end
        self._step_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        self._step_time_sum += now - self._step_start
        self._prev_end = now
        self._global_step_seen += 1

        if self._global_step_seen <= self.warmup_steps:
            self._reset()
            return

        self._n += 1
        if self._n >= self.log_every_n_steps:
            self._log_and_reset(trainer, pl_module)

    def _log_and_reset(self, trainer, pl_module):
        data_time = self._data_time_sum / self._n
        step_time = self._step_time_sum / self._n
        total = data_time + step_time
        data_pct = 100 * data_time / total if total > 0 else 0.0

        gpu_util = None
        if torch.cuda.is_available():
            try:
                idx = pl_module.device.index
                gpu_util = torch.cuda.utilization(idx if idx is not None else 0)
            except Exception:
                gpu_util = None

        metrics = {
            "profile/data_time_ms": data_time * 1000,
            "profile/step_time_ms": step_time * 1000,
            "profile/data_time_pct": data_pct,
        }
        if gpu_util is not None:
            metrics["profile/gpu_util_pct"] = float(gpu_util)

        pl_module.log_dict(metrics, prog_bar=False, sync_dist=False, rank_zero_only=True)
        self._print_verdict(trainer, data_time, step_time, data_pct, gpu_util)
        self._reset()

    @rank_zero_only
    def _print_verdict(self, trainer, data_time, step_time, data_pct, gpu_util):
        verdict = "CPU/IO-bound (dataloader)" if data_pct > 15 else "GPU-bound"
        gpu_str = f", gpu_util={gpu_util}%" if gpu_util is not None else ""
        print(
            f"[ThroughputProfiler] step {trainer.global_step}: "
            f"data={data_time * 1000:.1f}ms step={step_time * 1000:.1f}ms "
            f"({data_pct:.1f}% waiting for data{gpu_str}) -> {verdict}"
        )
