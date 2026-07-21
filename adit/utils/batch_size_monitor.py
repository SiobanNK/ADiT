from typing import Any, Optional

import torch
import lightning as L
from lightning import Callback

from adit.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BatchSizeMonitor(Callback):
    """Logs per-batch size/memory-cost metrics for ADiT training batches, on
    every rank, to whatever logger(s) are configured (W&B, CSV, TensorBoard, ...).

    Logs synchronously on `on_train_batch_end` (not polled on an interval like
    JobGPUMonitor) since what explains a given step's peak memory is *that
    step's* batch, not a rolling average.

    Reads the batch format produced by `BatchTensorConverter` /
    `DynamicBatchSampler` (see protein_datamodule.py / dynamic_batch_sampler.py):
    a dict of tensors dense-padded to the batch's max length, with one or more
    `*_seq_mask` keys (shape (B, L), 1 = real token / 0 = pad) and an
    `atom_mask` key (real atom occupancy within the fixed 37 per-token slots).

    Handles the SKEMPI/HER2 case where a sample bundles `wt_seq_mask` +
    `mt_seq_mask`: each seq_mask-like key is summed in independently, matching
    `_default_length_fn`'s definition of L in dynamic_batch_sampler.py.

    Logged per step (all under the `batch_size/` prefix):
      - batch_size/batch_size          number of samples B
      - batch_size/max_seq_len         L_max this batch got padded to
      - batch_size/num_tokens_real     sum of real (unpadded) token counts
      - batch_size/num_tokens_padded   B * L_max, i.e. what's actually allocated
      - batch_size/padding_waste_pct   100 * (1 - real/padded); dense-pad tax
      - batch_size/num_atoms           real atom count, from atom_mask.sum()
      - batch_size/pair_cost_sum_L_sq  sum(L_i**2) over samples -- the actual
                                        quadratic driver of the token-level
                                        pair-representation memory (block-
                                        diagonal edges, per DynamicBatchSampler's
                                        docstring), not B * L_max**2.
    """

    def __init__(self, log_every_n_steps: int = 1, prefix: str = "batch_size") -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.prefix = prefix

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return

        loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
        loggers = [lg for lg in loggers if lg is not None]
        if not loggers:
            return

        sizes = self._extract_sizes(batch)
        if sizes is None:
            log.warning("BatchSizeMonitor: could not extract batch sizes, skipping.")
            return

        metrics = {f"{self.prefix}/{k}": v for k, v in sizes.items()}
        step = trainer.global_step
        for lg in loggers:
            try:
                lg.log_metrics(metrics, step=step)
            except Exception as e:
                log.warning(f"BatchSizeMonitor: failed to log to {type(lg).__name__}: {e}")

    @staticmethod
    def _extract_sizes(batch: Any) -> Optional[dict]:
        if not isinstance(batch, dict):
            log.warning(f"BatchSizeMonitor: expected a dict batch, got {type(batch)}.")
            return None

        seq_mask_keys = [k for k in batch if k == "seq_mask" or k.endswith("_seq_mask")]
        if not seq_mask_keys:
            return None

        batch_size = None
        max_seq_len = 0
        num_tokens_real = 0
        num_tokens_padded = 0
        # per-sample real length, accumulated across seq_mask keys (SKEMPI/HER2:
        # wt_ + mt_ both contribute to the same sample's total L, matching
        # _default_length_fn in dynamic_batch_sampler.py)
        per_sample_len = None

        for k in seq_mask_keys:
            mask = batch[k]
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            if mask.dim() == 1:  # unbatched edge case, treat as B=1
                mask = mask.unsqueeze(0)

            b, seq_len = mask.shape[0], mask.shape[1]
            batch_size = b if batch_size is None else batch_size
            max_seq_len = max(max_seq_len, seq_len)
            num_tokens_real += int(mask.sum().item())
            num_tokens_padded += b * seq_len

            this_len = mask.sum(dim=-1)  # (B,) real length contributed by this key
            per_sample_len = this_len if per_sample_len is None else per_sample_len + this_len

        pair_cost_sum_L_sq = int((per_sample_len.to(torch.float64) ** 2).sum().item())

        num_atoms = None
        if "atom_mask" in batch:
            atom_mask = batch["atom_mask"]
            if not torch.is_tensor(atom_mask):
                atom_mask = torch.as_tensor(atom_mask)
            num_atoms = int(atom_mask.sum().item())

        padding_waste_pct = (
            100.0 * (1 - num_tokens_real / num_tokens_padded) if num_tokens_padded else 0.0
        )

        return {
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "num_tokens_real": num_tokens_real,
            "num_tokens_padded": num_tokens_padded,
            "padding_waste_pct": padding_waste_pct,
            "num_atoms": num_atoms if num_atoms is not None else num_tokens_real,
            "pair_cost_sum_L_sq": pair_cost_sum_L_sq,
        }
